# Filename: agents/orchestrator.py
# Description: Central coordinator for the AI Agency, managing agents, workflows, and resources.
# Version: 1.2 (Implemented Agentic Core Logic)

import os
import asyncio
from datetime import datetime, timedelta, timezone
import logging
import time
import json
import uuid
import random
import re
from collections import deque
from typing import Dict, Optional, Tuple, Any, List, AsyncGenerator, Callable # Added Callable

# --- Core Framework Imports ---
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, delete, func, update, text, case # Added case
from quart import Quart, request, jsonify, websocket, send_file, url_for
import psutil
from openai import AsyncOpenAI as AsyncLLMClient # Standardized name
import pybreaker
import websockets
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from prometheus_client import Counter, Gauge, start_http_server

# --- Project Imports ---
from config.settings import settings # Assuming settings object is validated on import
from utils.secure_storage import SecureStorage, VaultError
from utils.database import encrypt_data, decrypt_data_fixed_salt_migration, get_session # Assuming get_session is defined
from utils.notifications import send_notification # Assuming this handles SMTP password via Vault now
# Import all agent classes
from agents.base_agent import GeniusAgentBase, KBInterface # Import base and KB interface placeholder
from agents.think_tool import ThinkTool
from agents.browsing_agent import BrowsingAgent
from agents.email_agent import EmailAgent
from agents.legal_compliance_agent import LegalComplianceAgent # Assuming this is the primary compliance agent
from agents.osint_agent import OSINTAgent
from agents.scoring_agent import ScoringAgent
from agents.voice_sales_agent import VoiceSalesAgent
from agents.optimization_agent import OptimizationAgent
from agents.programmer_agent import ProgrammerAgent
from agents.social_media_manager import SocialMediaManager
from agents.legal_agent import LegalAgent # Strategic legal agent
from agents.budget_agent import BudgetAgent # Added BudgetAgent

# Import database models
from models import Base, Client, Metric, ExpenseLog, MigrationStatus, KnowledgeFragment, Account, CallLog, Invoice # Added missing models

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("agency.log", mode='a'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Metrics & Circuit Breakers ---
agent_status_gauge = Gauge('agent_status', 'Status of agents (1=running, 0=stopped/error)', ['agent_name'])
error_counter = Counter('agent_errors_total', 'Total number of errors per agent', ['agent_name'])
tasks_processed_counter = Counter('tasks_processed_total', 'Total number of tasks processed by agent', ['agent_name'])
llm_client_breaker = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=60 * 5, name="LLMClientBreaker") # 5 min timeout

# --- Helper Functions ---
def create_session_maker(db_url: str, schema: str = 'public') -> async_sessionmaker[AsyncSession]:
    """Creates an async session maker configured for a specific schema."""
    try:
        engine = create_async_engine(db_url, echo=False, pool_pre_ping=True) # Disable echo, enable pre-ping
        # Factory to create sessions
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        logger.info(f"Async session maker created for schema '{schema}'.")
        return session_factory
    except Exception as e:
        logger.critical(f"Failed to create database engine or session maker: {e}", exc_info=True)
        raise

# --- KB Interface Wrapper (Optional - Can be expanded) ---
class KBInterfaceWrapper(KBInterface):
    """Provides a consistent interface for agents to interact with the KB (via ThinkTool)."""
    def __init__(self, think_tool_agent: ThinkTool):
        self._think_tool = think_tool_agent
        self.logger = logging.getLogger(f"{__name__}.KBInterfaceWrapper")

    async def add_knowledge(self, *args, **kwargs):
        if not self._think_tool: return None
        try:
            # Delegate to ThinkTool's method
            return await self._think_tool.log_knowledge_fragment(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error delegating add_knowledge to ThinkTool: {e}", exc_info=True)
            return None

    async def get_knowledge(self, *args, **kwargs):
        if not self._think_tool: return []
        try:
            # Delegate to ThinkTool's method
            return await self._think_tool.query_knowledge_base(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error delegating get_knowledge to ThinkTool: {e}", exc_info=True)
            return []

    async def add_email_composition(self, *args, **kwargs):
        if not self._think_tool: return None
        try:
            # Delegate to ThinkTool's method
            return await self._think_tool.add_email_composition(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error delegating add_email_composition to ThinkTool: {e}", exc_info=True)
            return None

    async def log_learned_pattern(self, *args, **kwargs):
        if not self._think_tool: return None
        try:
            # Delegate to ThinkTool's method
            return await self._think_tool.log_learned_pattern(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error delegating log_learned_pattern to ThinkTool: {e}", exc_info=True)
            return None
    # Add other KB methods as needed, delegating to ThinkTool

# --- Orchestrator Class ---
class Orchestrator:
    """Central coordinator of the AI agency, managing agents, workflows, and operations autonomously."""

    def __init__(self, schema='public'):
        self.config = settings # Use imported settings object
        self.session_maker = create_session_maker(self.config.DATABASE_URL, schema)
        self.agents: Dict[str, GeniusAgentBase] = {} # Type hint agent values
        self.app = Quart(__name__, template_folder='../ui/templates', static_folder='../ui/static') # Adjust paths
        self.setup_routes()
        try:
            start_http_server(8001) # Use a different port for Prometheus metrics
            logger.info("Prometheus metrics server started on port 8001.")
        except OSError as e:
            logger.warning(f"Could not start Prometheus server on port 8001 (maybe already running?): {e}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}", exc_info=True)

        self.meta_prompt = self.config.META_PROMPT
        self.approved = False # Requires explicit approval via API
        self.secure_storage = SecureStorage() # Initialize Vault interface

        # LLM Client Management State
        self._llm_client_cache: Dict[str, AsyncLLMClient] = {} # api_key_identifier -> client instance
        self._llm_client_status: Dict[str, Dict[str, Any]] = {} # api_key_identifier -> {'status': 'available'/'unavailable', 'reason': str, 'unavailable_until': timestamp}
        self._llm_client_keys: List[str] = [] # List of available API key identifiers (e.g., last 6 chars)
        self._llm_client_round_robin_index = 0

        # In-memory Cache
        self._cache: Dict[str, Tuple[Any, float]] = {} # Key -> (Value, Expiry Timestamp)

        # Deepgram WebSocket Registry
        self.deepgram_connections: Dict[str, websockets.client.WebSocketClientProtocol] = {}
        self.temp_audio_dir = "/app/temp_audio" # Use path within container
        os.makedirs(self.temp_audio_dir, exist_ok=True)

        # Feedback loop timing attributes
        self.feedback_interval_seconds: int = int(self.config.get("THINKTOOL_FEEDBACK_INTERVAL_SECONDS", 300)) # Default 5 mins
        self.last_feedback_time: float = 0.0

        # Data Purge timing attributes (Orchestrator triggers ThinkTool's purge)
        self.purge_interval_seconds: int = int(self.config.get("DATA_PURGE_INTERVAL_SECONDS", 86400)) # Default 24 hours
        self.last_purge_time: float = 0.0

        self.running: bool = False # Controls the main loop in run()
        self.background_tasks = set() # Keep track of background tasks

        logger.info("Orchestrator initialized.")
        # Initialization sequence (DB, clients, agents) is called from run()

    async def initialize_database(self):
        """Initialize or update the database schema."""
        logger.info("Initializing database schema...")
        try:
            async with create_async_engine(self.config.DATABASE_URL).begin() as conn:
                # Create tables if they don't exist
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database schema initialization complete.")
            return True
        except Exception as e:
            logger.critical(f"Failed to initialize database: {e}", exc_info=True)
            # Attempt to notify user if possible, but this is critical
            # await self.send_notification("CRITICAL ERROR", f"Database initialization failed: {e}")
            return False # Indicate failure

    async def _run_encryption_migration_v2(self):
        """Checks and performs data migration for encryption v2 (per-value salt)."""
        # This logic remains largely the same as provided previously, ensuring it uses
        # the correct models and decryption/encryption functions.
        migration_name = 'encryption_v2'
        logger.info(f"Checking status for migration: {migration_name}")
        async with self.session_maker() as session:
            try:
                stmt_check = select(MigrationStatus).where(MigrationStatus.migration_name == migration_name)
                result_check = await session.execute(stmt_check)
                status_record = result_check.scalar_one_or_none()

                if status_record and status_record.completed_at:
                    logger.info(f"Migration '{migration_name}' already completed. Skipping.")
                    return True

                logger.info(f"Starting migration '{migration_name}' for ExpenseLog.description...")
                total_processed, total_updated, total_failed = 0, 0, 0
                batch_size = 100
                offset = 0

                while True:
                    logger.info(f"Processing migration batch offset {offset}...")
                    stmt_select = select(ExpenseLog.id, ExpenseLog.description).order_by(ExpenseLog.id).offset(offset).limit(batch_size)
                    result_select = await session.execute(stmt_select)
                    records = result_select.fetchall()

                    if not records: break

                    updates_to_commit = []
                    for record_id, old_encrypted_desc in records:
                        total_processed += 1
                        if not old_encrypted_desc: continue

                        decrypted_desc = decrypt_data_fixed_salt_migration(old_encrypted_desc)
                        if decrypted_desc is None:
                            logger.error(f"[Migration] Failed decryption for ExpenseLog ID {record_id}. Skipping.")
                            total_failed += 1
                            continue

                        try:
                            new_encrypted_desc = encrypt_data(decrypted_desc)
                            if new_encrypted_desc is None: raise ValueError("New encryption returned None")
                        except Exception as enc_err:
                            logger.error(f"[Migration] Failed NEW encryption for ExpenseLog ID {record_id}: {enc_err}. Skipping.")
                            continue

                        if new_encrypted_desc != old_encrypted_desc:
                            updates_to_commit.append({'id': record_id, 'description': new_encrypted_desc})
                            total_updated += 1

                    if updates_to_commit:
                        try:
                            # Use SQLAlchemy Core update for potential efficiency with many updates
                            update_stmt = update(ExpenseLog).where(ExpenseLog.id == text(':id')).values(description=text(':description'))
                            await session.execute(update_stmt, updates_to_commit)
                            logger.info(f"Migration batch update successful for {len(updates_to_commit)} records (offset {offset}).")
                        except Exception as batch_err:
                            logger.error(f"[Migration] Failed batch update (offset {offset}): {batch_err}")
                            await session.rollback()
                            raise Exception(f"Migration '{migration_name}' failed during batch update.") from batch_err

                    offset += len(records)

                completion_time = datetime.now(timezone.utc)
                if status_record:
                    status_record.completed_at = completion_time
                else:
                    session.add(MigrationStatus(migration_name=migration_name, completed_at=completion_time))

                await session.commit()
                logger.info(f"--- Migration '{migration_name}' Summary ---")
                logger.info(f"Processed: {total_processed}, Updated: {total_updated}, Failed Decryption: {total_failed}")
                logger.info(f"Migration marked complete at {completion_time}.")
                if total_failed > 0: logger.warning("Some records failed decryption and were not migrated.")
                return True

            except Exception as e:
                logger.error(f"CRITICAL: Error during migration '{migration_name}': {e}", exc_info=True)
                await session.rollback()
                return False # Indicate failure

    async def initialize_clients(self):
        """Initialize primary and potentially secondary LLM API clients from Vault."""
        logger.info("Initializing LLM clients from Vault...")
        self._llm_client_keys = [] # Reset available keys
        self._llm_client_cache = {} # Clear cache
        self._llm_client_status = {} # Clear status

        # --- Fetch Primary OpenRouter Key ---
        try:
            primary_openrouter_key = await self.secure_storage.get_secret('openrouter-api-key') # Standard key name
            if not primary_openrouter_key:
                raise VaultError("Primary 'openrouter-api-key' not found in Vault.")

            key_id = primary_openrouter_key[-6:] # Use last 6 chars as identifier
            self._llm_client_cache[key_id] = AsyncLLMClient(api_key=primary_openrouter_key, base_url="https://openrouter.ai/api/v1")
            self._llm_client_status[key_id] = {'status': 'available', 'reason': None, 'unavailable_until': 0}
            self._llm_client_keys.append(key_id)
            logger.info(f"Primary OpenRouter client initialized (Key ID: ...{key_id}).")

        except VaultError as ve:
            logger.critical(f"Failed to initialize primary OpenRouter client: {ve}")
            # Decide if startup should halt - for now, allow continuing if other clients might exist
        except Exception as e:
            logger.critical(f"Unexpected error initializing primary OpenRouter client: {e}", exc_info=True)

        # --- Fetch Primary DeepSeek Key ---
        # (Similar logic as OpenRouter, adjust key name and base_url)
        try:
            primary_deepseek_key = await self.secure_storage.get_secret('deepseek-api-key')
            if not primary_deepseek_key:
                 raise VaultError("Primary 'deepseek-api-key' not found in Vault.")

            key_id = primary_deepseek_key[-6:]
            # Note: DeepSeek might use a different client library or base class
            # Assuming AsyncLLMClient is compatible for this example
            self._llm_client_cache[key_id] = AsyncLLMClient(api_key=primary_deepseek_key, base_url="https://api.deepseek.com")
            self._llm_client_status[key_id] = {'status': 'available', 'reason': None, 'unavailable_until': 0}
            self._llm_client_keys.append(key_id)
            logger.info(f"Primary DeepSeek client initialized (Key ID: ...{key_id}).")
        except VaultError as ve:
            logger.warning(f"Failed to initialize primary DeepSeek client: {ve}") # Warning, maybe not critical
        except Exception as e:
            logger.error(f"Unexpected error initializing primary DeepSeek client: {e}", exc_info=True)


        # --- Fetch Additional Keys (Example: From Account Table) ---
        # This part depends on how BrowsingAgent stores account info.
        # Assuming it creates Account records with service='openrouter.ai' and a vault_path.
        logger.info("Attempting to load additional OpenRouter keys from Account table...")
        additional_keys_loaded = 0
        try:
            async with self.session_maker() as session:
                stmt = select(Account.vault_path).where(Account.service == 'openrouter.ai', Account.is_available == True)
                result = await session.execute(stmt)
                account_vault_paths = result.scalars().all()

                for vault_path in account_vault_paths:
                    try:
                        api_key = await self.secure_storage.get_secret(vault_path) # Fetch using path from DB
                        if api_key:
                            key_id = api_key[-6:]
                            if key_id not in self._llm_client_cache: # Avoid duplicates
                                self._llm_client_cache[key_id] = AsyncLLMClient(api_key=api_key, base_url="https://openrouter.ai/api/v1")
                                self._llm_client_status[key_id] = {'status': 'available', 'reason': None, 'unavailable_until': 0}
                                self._llm_client_keys.append(key_id)
                                additional_keys_loaded += 1
                                logger.debug(f"Loaded additional OpenRouter client (Key ID: ...{key_id}) from Vault path: {vault_path}")
                    except VaultError as ve:
                         logger.warning(f"Failed to load key from Vault path {vault_path}: {ve}")
                    except Exception as e:
                         logger.error(f"Unexpected error loading key from Vault path {vault_path}: {e}")

            logger.info(f"Loaded {additional_keys_loaded} additional OpenRouter keys.")
        except Exception as db_err:
             logger.error(f"Failed to query Account table for additional keys: {db_err}", exc_info=True)

        if not self._llm_client_keys:
             logger.critical("CRITICAL: No LLM API clients could be initialized. ThinkTool and other agents will fail.")
             return False # Indicate failure

        logger.info(f"LLM Client Initialization complete. Total usable keys: {len(self._llm_client_keys)}")
        return True # Indicate success

    async def initialize_agents(self):
        """Initialize all agents, fetching necessary secrets from Vault."""
        logger.info("Fetching secrets required for agent initialization...")
        # Define secrets needed by specific agents beyond LLM keys
        required_secrets = {
            'hostinger-smtp-pass': None, 'hostinger-imap-pass': None,
            'twilio-auth-token': None, 'deepgram-api-key': None,
            'smartproxy-password': None, 'shodan-api-key': None, # Added Shodan
            'spiderfoot-api-key': None # Added Spiderfoot
            # Add other non-LLM secrets here
        }
        initialization_failed = False
        try:
            # Fetch secrets concurrently
            fetch_tasks = {name: asyncio.create_task(self.secure_storage.get_secret(name)) for name in required_secrets}
            await asyncio.gather(*fetch_tasks.values(), return_exceptions=True)

            for name, task in fetch_tasks.items():
                result = task.result()
                if isinstance(result, Exception):
                    # Log critical only if essential, warning otherwise
                    log_level = logging.CRITICAL if name in ['hostinger-smtp-pass', 'twilio-auth-token', 'deepgram-api-key', 'smartproxy-password'] else logging.WARNING
                    logger.log(log_level, f"Failed to fetch required secret '{name}' from Vault: {result}. Dependent agents may fail.")
                    if log_level == logging.CRITICAL: initialization_failed = True
                else:
                    required_secrets[name] = result

            if initialization_failed:
                 raise VaultError("Failed to fetch one or more critical secrets for agent initialization.")

            logger.info("Required secrets fetched successfully. Initializing agents...")

            # --- Agent Initialization ---
            # Create KB Interface wrapper around ThinkTool (ThinkTool must be initialized first)
            self.agents['think'] = ThinkTool(self.session_maker, self.config, self)
            kb_interface = KBInterfaceWrapper(self.agents['think']) # Create wrapper

            # Initialize agents, passing dependencies including the KB interface
            self.agents['email'] = EmailAgent(
                self.session_maker, self, kb_interface, # Pass orchestrator and KB interface
                smtp_password=required_secrets['hostinger-smtp-pass'],
                imap_password=required_secrets['hostinger-imap-pass']
            )
            self.agents['legal_compliance'] = LegalComplianceAgent( # Use distinct key
                self.session_maker, self, kb_interface
            )
            self.agents['osint'] = OSINTAgent(
                self.session_maker, self, kb_interface,
                shodan_api_key=required_secrets['shodan-api-key'],
                spiderfoot_api_key=required_secrets['spiderfoot-api-key']
            )
            self.agents['scoring'] = ScoringAgent(
                 self.session_maker, self, kb_interface # Pass orchestrator and KB interface
            )
            self.agents['voice_sales'] = VoiceSalesAgent(
                self.session_maker, self, kb_interface,
                twilio_auth_token=required_secrets['twilio-auth-token'],
                deepgram_api_key=required_secrets['deepgram-api-key']
            )
            self.agents['optimization'] = OptimizationAgent(
                 self.session_maker, self, kb_interface # Pass orchestrator and KB interface
            )
            self.agents['browsing'] = BrowsingAgent(
                self.session_maker, self, kb_interface, # Pass orchestrator and KB interface
                smartproxy_password=required_secrets['smartproxy-password']
            )
            self.agents['programmer'] = ProgrammerAgent(self) # Pass orchestrator
            self.agents['social_media'] = SocialMediaManager(self) # Pass orchestrator
            self.agents['strategic_legal'] = LegalAgent(self) # Pass orchestrator
            self.agents['budget'] = BudgetAgent(self.session_maker, self) # Pass orchestrator

            # Set Prometheus gauges for initial status (idle)
            for name in self.agents.keys():
                agent_status_gauge.labels(agent_name=name).set(0) # 0 for idle/stopped

            logger.info("All agents initialized successfully.")
            return True

        except VaultError as ve:
            logger.critical(f"CRITICAL: Failed to fetch secrets during agent initialization: {ve}. Cannot proceed.")
            return False # Indicate failure
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}", exc_info=True)
            return False # Indicate failure

    # --- LLM Client Management ---
    @llm_client_breaker
    async def get_available_openrouter_clients(self) -> List[AsyncLLMClient]:
        """Gets a list of currently available OpenRouter client instances based on status and rotation."""
        now = time.time()
        available_keys = [
            key_id for key_id in self._llm_client_keys
            if self._llm_client_status.get(key_id, {}).get('status') == 'available' or
               now >= self._llm_client_status.get(key_id, {}).get('unavailable_until', 0)
        ]

        # Mark previously unavailable keys as available if cooldown expired
        for key_id in list(self._llm_client_status.keys()): # Iterate over copy
             status_info = self._llm_client_status[key_id]
             if status_info['status'] == 'unavailable' and now >= status_info.get('unavailable_until', 0):
                  status_info['status'] = 'available'
                  status_info['reason'] = None
                  status_info['unavailable_until'] = 0
                  if key_id not in available_keys: # Add back if not already present
                       available_keys.append(key_id)
                  logger.info(f"LLM client key ...{key_id} marked as available after cooldown.")

        if not available_keys:
            logger.error("No available LLM clients found!")
            # Optionally: Trigger alert or attempt to provision new keys?
            return [] # Return empty list, caller must handle

        # Simple round-robin selection
        selected_key_id = available_keys[self._llm_client_round_robin_index % len(available_keys)]
        self._llm_client_round_robin_index += 1

        selected_client = self._llm_client_cache.get(selected_key_id)
        if not selected_client:
             # This should not happen if cache is managed correctly
             logger.error(f"Client instance not found in cache for available key ID ...{selected_key_id}. Re-initializing.")
             # Attempt re-initialization (requires fetching key again) - potentially complex
             # For now, remove from available and return empty
             self._llm_client_keys.remove(selected_key_id)
             self._llm_client_status.pop(selected_key_id, None)
             self._llm_client_cache.pop(selected_key_id, None)
             return await self.get_available_openrouter_clients() # Retry selection

        logger.debug(f"Selected LLM client with key ID: ...{selected_key_id}")
        return [selected_client] # Return list with one client

    async def report_client_issue(self, api_key_identifier: str, issue_type: str):
        """Marks an LLM client as unavailable due to an issue."""
        if api_key_identifier not in self._llm_client_status:
            logger.warning(f"Attempted to report issue for unknown LLM client identifier: ...{api_key_identifier}")
            return

        now = time.time()
        cooldown_seconds = 60 * 5 # Default 5 minutes cooldown
        status = 'unavailable'
        reason = issue_type

        if issue_type == 'auth_error':
            cooldown_seconds = 60 * 60 * 24 * 365 # Effectively permanent disable for auth errors
            reason = "Authentication Error"
            logger.critical(f"LLM client key ...{api_key_identifier} marked permanently unavailable due to auth error.")
        elif issue_type == 'rate_limit':
            cooldown_seconds = 60 * 2 # Shorter cooldown for rate limits
            reason = "Rate Limited"
        elif issue_type == 'timeout_error':
             cooldown_seconds = 60 * 3
             reason = "Timeout"
        else: # General llm_error
             reason = "General Error"

        self._llm_client_status[api_key_identifier] = {
            'status': status,
            'reason': reason,
            'unavailable_until': now + cooldown_seconds
        }
        logger.warning(f"LLM client key ...{api_key_identifier} marked unavailable until {datetime.fromtimestamp(now + cooldown_seconds)}. Reason: {reason}")

    @llm_client_breaker # Apply circuit breaker to the core LLM call logic
    async def call_llm(self, agent_name: str, prompt: str, temperature: float = 0.5, max_tokens: int = 1024, is_json_output: bool = False, model_preference: Optional[List[str]] = None) -> Optional[str]:
        """Handles selecting an available client and making the LLM call."""
        selected_clients = await self.get_available_openrouter_clients()
        if not selected_clients:
            logger.error(f"Agent '{agent_name}' failed to get an available LLM client.")
            return None # No clients available

        llm_client = selected_clients[0] # Get the single selected client
        api_key_identifier = getattr(llm_client, 'api_key', 'unknown_key')[-6:]

        # Determine model (simplified - use default based on agent or task type)
        # TODO: Implement more sophisticated model selection based on model_preference
        model_key = 'default_llm' # Fallback
        if agent_name == 'ThinkTool': model_key = 'think'
        elif agent_name == 'EmailAgent': model_key = 'email_draft_llm'
        elif agent_name == 'VoiceSalesAgent': model_key = 'voice_response_llm'
        # Add other agent mappings...
        model_name = self.config.OPENROUTER_MODELS.get(model_key, self.config.OPENROUTER_MODELS['default_llm'])

        # --- Caching Logic ---
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        cache_key_parts = ["llm_call", prompt_hash, model_name, str(temperature), str(max_tokens), str(is_json_output)]
        cache_key = ":".join(cache_key_parts)
        cache_ttl = 3600

        cached_result = self.get_from_cache(cache_key)
        if cached_result is not None:
            logger.debug(f"LLM call cache hit (Orchestrator) for key: {cache_key[:20]}...")
            return cached_result
        # --- End Cache Check ---

        try:
            response_format = {"type": "json_object"} if is_json_output else None
            messages = [{"role": "user", "content": prompt}]
            logger.debug(f"Orchestrator making LLM Call: Agent={agent_name}, Model={model_name}, Key=...{api_key_identifier}")

            response = await llm_client.chat.completions.create(
                model=model_name, messages=messages, temperature=temperature,
                max_tokens=max_tokens, response_format=response_format, timeout=120 # Increased timeout
            )
            content = response.choices[0].message.content.strip()

            # Track usage & cost (simplified)
            input_tokens = response.usage.prompt_tokens if response.usage else len(prompt) // 4
            output_tokens = response.usage.completion_tokens if response.usage else len(content) // 4
            total_tokens = input_tokens + output_tokens
            cost = total_tokens * 0.000001 # Placeholder cost
            logger.debug(f"LLM Call Usage: Model={model_name}, Tokens={total_tokens}, Est. Cost=${cost:.6f}")
            await self.report_expense(agent_name, cost, "LLM", f"LLM call ({model_name}). Tokens: {total_tokens}.")

            # Add to cache
            self.add_to_cache(cache_key, content, ttl_seconds=cache_ttl)

            return content

        except Exception as e:
            error_str = str(e).lower()
            issue_type = "llm_error"
            if "rate limit" in error_str or "quota" in error_str: issue_type = "rate_limit"
            elif "authentication" in error_str: issue_type = "auth_error"
            elif "timeout" in error_str: issue_type = "timeout_error"
            logger.warning(f"Orchestrator LLM call failed: Agent={agent_name}, Model={model_name}, Key=...{api_key_identifier}, Type={issue_type}, Error={e}")
            await self.report_client_issue(api_key_identifier, issue_type)
            raise # Re-raise for tenacity/caller handling

    # --- Proxy Management ---
    async def get_proxy(self, purpose: str = "general", target_url: Optional[str] = None) -> Optional[str]:
        """Gets an available proxy from the BrowsingAgent's manager."""
        browsing_agent = self.agents.get('browsing')
        if browsing_agent and hasattr(browsing_agent, '_proxy_manager'):
            try:
                # Pass service name derived from target_url if available
                service_name = None
                if target_url:
                    try:
                        # Basic extraction, might need refinement
                        domain = target_url.split('//')[-1].split('/')[0].replace('www.', '')
                        service_name = domain
                    except Exception: pass # Ignore errors deriving service name

                proxy_dict = await browsing_agent._proxy_manager.get_next_proxy(service=service_name)
                if proxy_dict:
                    # Format for aiohttp
                    proxy_url = f"http://{proxy_dict['username']}:{proxy_dict['password']}@{proxy_dict['server']}"
                    return proxy_url
                else:
                    logger.warning(f"No available proxy from BrowsingAgent for purpose '{purpose}'.")
                    return None
            except Exception as e:
                logger.error(f"Error getting proxy from BrowsingAgent: {e}", exc_info=True)
                return None
        else:
            logger.error("BrowsingAgent or its _proxy_manager not available for proxy retrieval.")
            return None

    # --- Tool/Task Delegation ---
    async def use_tool(self, tool_name: str, tool_params: Dict[str, Any]) -> Dict[str, Any]:
        """Delegates execution of standard tools (read, write, exec) to ProgrammerAgent."""
        programmer = self.agents.get('programmer')
        if not programmer:
            return {"status": "error", "message": "ProgrammerAgent not available."}

        # Map tool names to ProgrammerAgent methods (or task types)
        tool_map = {
            'read_file': 'read_file', # Assumes ProgrammerAgent has read_file method/handler
            'apply_diff': 'apply_diff',
            'write_to_file': 'write_to_file',
            'insert_content': 'insert_content',
            'search_and_replace': 'search_and_replace',
            'execute_command': 'execute_command',
            'search_files': 'search_files',
            'list_files': 'list_files',
            # Add other mappings as needed
        }

        if tool_name not in tool_map:
            return {"status": "error", "message": f"Tool '{tool_name}' not recognized or handled by ProgrammerAgent."}

        # Prepare task details for ProgrammerAgent
        task_details = {
            "action": tool_map[tool_name], # Map to action type ProgrammerAgent understands
            "params": tool_params,
            "description": f"Execute tool '{tool_name}' with params: {tool_params}"
        }

        try:
            result = await programmer.execute_task(task_details)
            return result
        except Exception as e:
            logger.error(f"Error delegating tool '{tool_name}' to ProgrammerAgent: {e}", exc_info=True)
            return {"status": "error", "message": f"Exception calling ProgrammerAgent for tool '{tool_name}': {e}"}

    async def delegate_task(self, target_agent_name: str, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Delegates a task to a specific agent."""
        target_agent = self.agents.get(target_agent_name)
        if not target_agent:
            logger.error(f"Cannot delegate task: Agent '{target_agent_name}' not found.")
            return {"status": "error", "message": f"Agent '{target_agent_name}' not found."}

        logger.info(f"Delegating task to {target_agent_name}: {task_details.get('action', task_details.get('description', 'No description'))}")
        try:
            # Check if agent uses internal queue first
            if hasattr(target_agent, 'internal_state') and 'task_queue' in target_agent.internal_state:
                # Assign a default priority or get from task_details
                priority = task_details.get('priority', 5.0) # Default priority
                await target_agent.internal_state['task_queue'].put((priority, task_details))
                return {"status": "queued", "message": f"Task queued for {target_agent_name}."}
            elif hasattr(target_agent, 'execute_task') and callable(target_agent.execute_task):
                # If no queue, execute directly
                result = await target_agent.execute_task(task_details)
                return result
            else:
                logger.error(f"Agent '{target_agent_name}' has no task queue and no callable execute_task method.")
                return {"status": "error", "message": f"Agent '{target_agent_name}' cannot execute tasks."}
        except Exception as e:
            logger.error(f"Error delegating task to {target_agent_name}: {e}", exc_info=True)
            return {"status": "error", "message": f"Exception during task delegation to {target_agent_name}: {e}"}

    # --- Workflow Management ---
    async def start_ugc_workflow(self, client_industry: str, num_videos: int = 1, script: str = None, target_services: list = None):
        """Initiates the UGC content generation workflow."""
        # (Implementation remains largely the same as provided previously)
        await self.send_notification("UGC Workflow Started", f"Starting UGC workflow for {client_industry} ({num_videos} videos).")
        logger.info(f"Initiating UGC workflow for {client_industry} ({num_videos} videos).")

        if not target_services:
            target_services = ['heygen.com', 'descript.com'] # Default services

        workflow_id = str(uuid.uuid4())
        logger.info(f"UGC Workflow ID: {workflow_id}")

        workflow_state = {
            'workflow_id': workflow_id, 'client_industry': client_industry,
            'num_videos': num_videos, 'target_services': target_services,
            'current_step': 'account_acquisition',
            'required_accounts': {service: False for service in target_services},
            'script': script or f"Default UGC script for {client_industry}.",
            'generated_video_path': None, 'edited_video_path': None,
            'last_error': None, 'retry_count': 0
        }

        # Queue the first step
        first_service = target_services[0]
        task = {
            'action': 'acquire_or_verify_account',
            'service_name': first_service,
            'workflow_id': workflow_id,
            'workflow_state': workflow_state
        }
        await self.delegate_task('browsing', task) # Use delegate_task

    async def report_ugc_step_complete(self, workflow_id: str, completed_step: str, result: dict, current_state: dict):
        """Handles completion of a UGC step and queues the next one, with error handling."""
        logger.info(f"Received completion report for UGC workflow {workflow_id}, step: {completed_step}. Success: {result.get('success')}")
        # TODO: Load/Update persistent workflow_state if used

        updated_state = current_state.copy()
        updated_state['last_error'] = None # Clear last error on step report
        next_task = None
        max_retries = 3 # Max retries per step

        try:
            if not result.get('success', False):
                error_reason = result.get('reason', 'Unknown error')
                updated_state['last_error'] = f"Step '{completed_step}' failed: {error_reason}"
                updated_state['retry_count'] = current_state.get('retry_count', 0) + 1
                logger.error(f"Step '{completed_step}' failed for workflow {workflow_id}. Reason: {error_reason}. Retry attempt {updated_state['retry_count']}.")

                if updated_state['retry_count'] <= max_retries:
                    # Retry the same step
                    logger.info(f"Retrying step '{completed_step}' for workflow {workflow_id}.")
                    # Re-queue the task for the failed step
                    failed_service = current_state.get('failed_service', result.get('service_name')) # Try to get service context
                    if completed_step == 'acquire_or_verify_account' and failed_service:
                         next_task = {'action': 'acquire_or_verify_account', 'service_name': failed_service, 'workflow_id': workflow_id, 'workflow_state': updated_state}
                    elif completed_step == 'generate_ugc_video' and failed_service:
                         next_task = {'action': 'generate_ugc_video', 'target_service': failed_service, 'script': updated_state['script'], 'workflow_id': workflow_id, 'workflow_state': updated_state}
                    elif completed_step == 'edit_ugc_video' and failed_service:
                         next_task = {'action': 'edit_ugc_video', 'target_service': failed_service, 'source_video_path': updated_state['generated_video_path'], 'workflow_id': workflow_id, 'workflow_state': updated_state}
                    else:
                         logger.error(f"Cannot determine task details for retry of step '{completed_step}'. Halting workflow {workflow_id}.")
                         await self.send_notification("UGC Workflow Error", f"Retry failed for step {completed_step} in workflow {workflow_id}. Unknown task details.")
                         return # Halt workflow
                else:
                    # Max retries exceeded
                    logger.error(f"Max retries ({max_retries}) exceeded for step '{completed_step}' in workflow {workflow_id}. Halting workflow.")
                    await self.send_notification("UGC Workflow Failed", f"Step {completed_step} failed after {max_retries} retries in workflow {workflow_id}. Reason: {error_reason}")
                    # TODO: Mark workflow as failed in DB if persisting state
                    return # Halt workflow
            else:
                # Step succeeded, reset retry count and proceed
                updated_state['retry_count'] = 0

                if completed_step == 'acquire_or_verify_account':
                    service_name = result.get('service_name') # Should be in result on success
                    if service_name: updated_state['required_accounts'][service_name] = True
                    all_accounts_ready = all(updated_state['required_accounts'].values())

                    if all_accounts_ready:
                        updated_state['current_step'] = 'video_generation'
                        video_gen_service = next((s for s in updated_state['target_services'] if s in ['heygen.com', 'argil.ai']), None)
                        if video_gen_service:
                            next_task = {'action': 'generate_ugc_video', 'target_service': video_gen_service, 'script': updated_state['script'], 'workflow_id': workflow_id, 'workflow_state': updated_state}
                        else: logger.error(f"No suitable video generation service found for workflow {workflow_id}.") # Error handled below if next_task is None
                    else:
                        next_service = next((s for s, ready in updated_state['required_accounts'].items() if not ready), None)
                        if next_service:
                            next_task = {'action': 'acquire_or_verify_account', 'service_name': next_service, 'workflow_id': workflow_id, 'workflow_state': updated_state}
                        else: logger.error(f"Inconsistent state: Not all accounts ready, but no next service found for workflow {workflow_id}.")

                elif completed_step == 'generate_ugc_video':
                    generated_video = result.get('video_id') # Assuming BrowsingAgent returns 'video_id' or similar
                    if generated_video:
                        updated_state['generated_video_path'] = generated_video # Store identifier/path
                        updated_state['current_step'] = 'video_editing'
                        editing_service = next((s for s in updated_state['target_services'] if s == 'descript.com'), None)
                        if editing_service:
                            next_task = {'action': 'edit_ugc_video', 'target_service': editing_service, 'source_video_id': generated_video, 'workflow_id': workflow_id, 'workflow_state': updated_state}
                        else:
                            logger.warning(f"No editing service found for workflow {workflow_id}. Marking complete.")
                            updated_state['current_step'] = 'completed'
                            # Workflow complete logic here
                    else: logger.error(f"Video generation step succeeded but no video identifier returned for workflow {workflow_id}.")

                elif completed_step == 'edit_ugc_video':
                    edited_video = result.get('edited_video_id') # Assuming BrowsingAgent returns 'edited_video_id'
                    if edited_video:
                        updated_state['edited_video_path'] = edited_video
                        updated_state['current_step'] = 'completed'
                        # Workflow complete logic here
                    else: logger.error(f"Video editing step succeeded but no edited video identifier returned for workflow {workflow_id}.")

            # --- Queue Next Task or Complete Workflow ---
            if updated_state['current_step'] == 'completed':
                final_video = updated_state.get('edited_video_path') or updated_state.get('generated_video_path')
                logger.info(f"UGC Workflow {workflow_id} completed. Final video identifier: {final_video}")
                await self.send_notification("UGC Workflow Complete", f"Workflow {workflow_id} for {updated_state['client_industry']} finished. Video ID: {final_video}")
                # TODO: Update workflow status in DB if persisting state
            elif next_task:
                # Update the state in the task being queued
                next_task['workflow_state'] = updated_state
                await self.delegate_task('browsing', next_task)
                logger.info(f"Queued next UGC task for workflow {workflow_id}: {next_task.get('action')} for {next_task.get('target_service') or next_task.get('service_name')}")
            else:
                # If no next task determined and not completed, log an error
                logger.error(f"UGC Workflow {workflow_id} stalled at step '{completed_step}'. No next task determined.")
                await self.send_notification("UGC Workflow Error", f"Workflow {workflow_id} stalled at step {completed_step}.")

        except Exception as e:
            logger.error(f"Error processing UGC step completion for workflow {workflow_id}: {e}", exc_info=True)
            await self.send_notification("UGC Workflow Error", f"Internal error processing step {completed_step} for workflow {workflow_id}: {e}")

    # --- Feedback Loop ---
    async def handle_feedback_trigger(self, all_insights: Dict[str, Dict[str, Any]]):
        """Triggers ThinkTool to process collected agent insights."""
        think_agent = self.agents.get("think")
        if think_agent and hasattr(think_agent, 'process_feedback'):
            logger.info("Forwarding collected insights to ThinkTool for processing.")
            try:
                await think_agent.process_feedback(all_insights)
            except Exception as e:
                logger.error(f"Error calling ThinkTool.process_feedback: {e}", exc_info=True)
        else:
            logger.warning("ThinkTool agent not found or missing 'process_feedback' method.")

    # --- Tool Installation ---
    async def handle_install_tool_request(self, requesting_agent_name: str, tool_details: Dict[str, Any]):
        """Dispatches a tool installation task to the ProgrammerAgent."""
        logger.info(f"Received tool install request from {requesting_agent_name} for: {tool_details.get('tool_name')}")
        programmer_agent = self.agents.get('programmer')
        if not programmer_agent:
            logger.error("ProgrammerAgent not available to handle tool installation request.")
            # Notify requesting agent?
            return

        install_task = {
            "action": "install_tool", # Define this action type for ProgrammerAgent
            "tool_name": tool_details.get('tool_name'),
            "package_manager": tool_details.get('package_manager', 'apt'), # Default or specified
            "package_name": tool_details.get('package_name'), # Name for apt/pip etc.
            "git_repo": tool_details.get('git_repo'), # If installation is via git clone
            "description": f"Install tool '{tool_details.get('tool_name')}' requested by {requesting_agent_name}."
        }
        await self.delegate_task('programmer', install_task)

    # --- Invoice Generation ---
    async def request_invoice_generation(self, client_id: int, amount: float, source_ref: str):
        """Handles request to generate an invoice (logs for now)."""
        logger.info(f"Invoice generation requested: ClientID={client_id}, Amount=${amount:.2f}, SourceRef={source_ref}")
        try:
            async with self.session_maker() as session:
                client = await session.get(Client, client_id)
                if not client:
                    logger.error(f"Cannot generate invoice: Client ID {client_id} not found.")
                    return

                # Create Invoice record in DB (status='pending')
                invoice = Invoice(
                    client_id=client_id,
                    amount=amount,
                    status='pending',
                    timestamp=datetime.now(timezone.utc),
                    # due_date=datetime.now(timezone.utc) + timedelta(days=14) # Example due date
                )
                session.add(invoice)
                await session.commit()
                await session.refresh(invoice)
                logger.info(f"Created pending Invoice record ID {invoice.id} for Client {client_id}.")

                # TODO: Integrate with actual PDF generation (ReportLab?) and payment processor (Lemon Squeezy?)
                # For now, notify user
                await self.send_notification(
                    "Invoice Ready for Generation",
                    f"Please generate invoice for Client: {client.name} (ID: {client_id}), Amount: ${amount:.2f}. Ref: {source_ref}. DB Record ID: {invoice.id}"
                )

        except Exception as e:
            logger.error(f"Error during invoice request processing for Client {client_id}: {e}", exc_info=True)
            await self.send_notification("Invoice Generation Error", f"Failed to process invoice request for Client {client_id}: {e}")


    # --- Periodic Tasks ---
    async def _run_periodic_data_purge(self):
        """Periodically triggers ThinkTool's data purge."""
        while self.running:
            try:
                await asyncio.sleep(self.purge_interval_seconds)
                if not self.running: break # Check flag again after sleep

                logger.info("Orchestrator triggering periodic data purge via ThinkTool...")
                think_agent = self.agents.get("think")
                if think_agent and hasattr(think_agent, 'purge_old_knowledge'):
                    await think_agent.purge_old_knowledge() # Default 30 days handled in ThinkTool
                else:
                    logger.warning("ThinkTool agent not found or missing 'purge_old_knowledge' for periodic purge.")

            except asyncio.CancelledError:
                logger.info("Periodic data purge task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in periodic data purge task: {e}", exc_info=True)
                await asyncio.sleep(60 * 5) # Wait longer after error

    async def _run_periodic_feedback_collection(self):
        """Periodically collects insights from agents and triggers processing."""
        while self.running:
            try:
                await asyncio.sleep(self.feedback_interval_seconds)
                if not self.running: break

                logger.info("Orchestrator initiating periodic feedback collection...")
                all_insights = {}
                agent_items = list(self.agents.items()) # Copy for safe iteration

                insight_tasks = []
                for agent_name, agent_instance in agent_items:
                    if agent_instance and hasattr(agent_instance, 'collect_insights') and callable(agent_instance.collect_insights):
                        insight_tasks.append(asyncio.create_task(agent_instance.collect_insights(), name=f"CollectInsights_{agent_name}"))
                    else:
                         logger.debug(f"Agent {agent_name} does not have collect_insights method.")

                # Gather insights concurrently
                results = await asyncio.gather(*insight_tasks, return_exceptions=True)

                for i, agent_tuple in enumerate(agent_items):
                     agent_name = agent_tuple[0]
                     result = results[i]
                     if isinstance(result, Exception):
                          logger.error(f"Error collecting insights from {agent_name}: {result}", exc_info=result)
                          all_insights[agent_name] = {"error": f"Failed to collect insights: {result}"}
                     elif result: # Check if insights were returned
                          all_insights[agent_name] = result
                          logger.debug(f"Received insights from {agent_name}.")
                     else:
                          logger.debug(f"No insights returned from {agent_name}.")


                if all_insights:
                    logger.info(f"Collected insights from {len(all_insights)} agents. Triggering feedback handler.")
                    await self.handle_feedback_trigger(all_insights)
                else:
                    logger.info("No insights collected in this feedback cycle.")

            except asyncio.CancelledError:
                logger.info("Periodic feedback collection task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in periodic feedback collection task: {e}", exc_info=True)
                await asyncio.sleep(60 * 5) # Wait longer after error

    # --- Cache Management ---
    def get_from_cache(self, key: str) -> Optional[Any]:
        """Retrieves an item from the in-memory cache if valid."""
        if key in self._cache:
            value, expiry_ts = self._cache[key]
            if time.time() < expiry_ts:
                logger.debug(f"Cache hit for key: {key}")
                return value
            else:
                logger.debug(f"Cache expired for key: {key}")
                del self._cache[key] # Clean up expired
        logger.debug(f"Cache miss for key: {key}")
        return None

    def add_to_cache(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Adds an item to the in-memory cache with a TTL."""
        if ttl_seconds <= 0: return
        expiry_ts = time.time() + ttl_seconds
        self._cache[key] = (value, expiry_ts)
        logger.debug(f"Added key '{key}' to cache with TTL {ttl_seconds}s.")
        # Optional: Prune cache if needed
        # self._prune_cache_if_needed()

    # --- Notification & Error Reporting ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=5, max=60))
    async def send_notification(self, subject, body):
        """Sends email notification via utility, fetching password from Vault."""
        try:
            smtp_pass = await self.secure_storage.get_secret('hostinger-smtp-pass')
            if not smtp_pass: raise VaultError("SMTP password not found.")
            # Assuming send_notification utility is updated to accept password
            await send_notification(subject, body, self.config, smtp_password=smtp_pass)
        except VaultError as ve: logger.error(f"Failed notification '{subject}' due to Vault error: {ve}")
        except Exception as e: logger.error(f"Error sending notification '{subject}': {e}", exc_info=True); raise # Re-raise for retry

    async def report_error(self, agent_name: str, error_message: str):
        """Report errors via notification and log metric."""
        logger.error(f"ERROR reported by {agent_name}: {error_message}")
        error_counter.labels(agent_name=agent_name).inc()
        subject = f"ERROR in {agent_name}"
        body = f"Timestamp: {datetime.now(timezone.utc).isoformat()}\nAgent: {agent_name}\nError details:\n{error_message}"
        await self.send_notification(subject, body)

    async def report_expense(self, agent_name: str, amount: float, category: str, description: str):
        """Forwards expense report to BudgetAgent."""
        logger.debug(f"Received expense report from {agent_name}: ${amount:.4f} [{category}] - {description}")
        budget_agent = self.agents.get('budget')
        if budget_agent and hasattr(budget_agent, 'record_expense'):
            try:
                # Use await since BudgetAgent methods are async
                success = await budget_agent.record_expense(agent_name=agent_name, amount=amount, category=category, description=description)
                if success:
                     logger.info(f"Successfully forwarded expense from {agent_name} to BudgetAgent.")
                else:
                     # record_expense should log its own errors if it returns False
                     logger.warning(f"BudgetAgent indicated failure recording expense from {agent_name}.")
            except Exception as e:
                logger.error(f"Failed to track expense via BudgetAgent for {agent_name}: {e}", exc_info=True)
        else:
            logger.error("BudgetAgent not found or missing 'record_expense' method. Cannot track expense.")

    # --- User Education Trigger ---
    async def handle_user_education_trigger(self, topic: str, context: Optional[str] = None):
        """Handles triggers for user education via ThinkTool."""
        logger.info(f"User education trigger received for topic: {topic}")
        think_agent = self.agents.get("think")
        if think_agent and hasattr(think_agent, 'generate_educational_content'):
            try:
                explanation = await think_agent.generate_educational_content(topic, context)
                if explanation:
                    logger.info(f"Educational content generated for '{topic}'. Notifying user.")
                    # Use the notification system for user education messages
                    await self.send_notification(f"Educational Note: {topic}", explanation)
                else:
                    logger.warning(f"ThinkTool failed to generate educational content for topic: {topic}")
            except Exception as e:
                logger.error(f"Error during educational content generation for '{topic}': {e}", exc_info=True)
        else:
            logger.error("ThinkTool agent not found or missing 'generate_educational_content' method.")

    # --- Deepgram WebSocket Management ---
    async def register_deepgram_connection(self, call_sid: str, ws_connection: websockets.client.WebSocketClientProtocol):
        logger.info(f"Registering Deepgram WS connection for call_sid: {call_sid}")
        self.deepgram_connections[call_sid] = ws_connection

    async def unregister_deepgram_connection(self, call_sid: str):
        logger.info(f"Unregistering Deepgram WS connection for call_sid: {call_sid}")
        self.deepgram_connections.pop(call_sid, None)

    async def get_deepgram_connection(self, call_sid: str) -> Optional[websockets.client.WebSocketClientProtocol]:
        return self.deepgram_connections.get(call_sid)

    # --- Temporary Audio Hosting ---
    async def host_temporary_audio(self, audio_data: bytes, filename: str) -> Optional[str]:
        """Saves audio data locally and returns a URL accessible by Twilio."""
        try:
            safe_filename = re.sub(r'[^\w\.-]', '_', filename)
            filepath = os.path.join(self.temp_audio_dir, safe_filename)
            with open(filepath, 'wb') as f: f.write(audio_data)
            # Assumes Quart app is running and accessible externally at AGENCY_BASE_URL
            # and the '/hosted_audio/' route is correctly configured.
            base_url = self.config.AGENCY_BASE_URL.rstrip('/')
            audio_url = f"{base_url}/hosted_audio/{safe_filename}"
            # Use url_for if running within Quart request context, otherwise construct manually
            # audio_url = url_for('serve_hosted_audio', filename=safe_filename, _external=True) # Use if in request context
            logger.info(f"Hosted temporary audio at: {audio_url}")
            return audio_url
        except Exception as e:
            logger.error(f"Failed to host temporary audio {filename}: {e}", exc_info=True)
            return None

    # --- Route Setup ---
    def setup_routes(self):
        """Configure Quart routes for UI interaction and webhooks."""
        # Existing routes...
        @self.app.route('/')
        async def index():
            # Serve the main UI page
            try:
                 # Adjust path relative to orchestrator.py location
                 template_path = os.path.join(os.path.dirname(__file__), '..', 'ui', 'templates', 'index.html')
                 # This requires Quart-Render or similar setup, or manually reading file
                 # For simplicity, let's assume render_template works if configured
                 # return await render_template('index.html')
                 # Manual read as fallback:
                 if os.path.exists(template_path):
                      with open(template_path, 'r') as f:
                           return f.read(), 200, {'Content-Type': 'text/html'}
                 else:
                      return "UI Template not found.", 404
            except Exception as e:
                 logger.error(f"Error serving index.html: {e}", exc_info=True)
                 return "Internal Server Error", 500

        @self.app.route('/api/approve', methods=['POST'])
        async def approve():
            if self.approved:
                 return jsonify({"status": "already_approved"}), 200
            self.approved = True
            logger.info("!!! AGENCY APPROVED FOR FULL OPERATION via API !!!")
            await self.send_notification("Agency Approved", "The agency has been approved for full operation via API.")
            # Potentially trigger agents that were waiting for approval
            # (e.g., set an asyncio.Event)
            return jsonify({"status": "approved"})

        @self.app.route('/api/status', methods=['GET'])
        async def api_status():
            agent_statuses = {name: agent.get_status() for name, agent in self.agents.items()}
            # Add LLM client status
            llm_status = {key: info['status'] for key, info in self._llm_client_status.items()}
            return jsonify({
                "orchestrator_status": "running" if self.running else "stopped",
                "approved_for_operation": self.approved,
                "agent_statuses": agent_statuses,
                "llm_client_status": llm_status,
                "global_email_limit_today": f"{self.internal_state.get('global_sent_today', 0)}/{self.internal_state.get('global_daily_limit', 'N/A')}"
            })

        @self.app.route('/api/start_ugc', methods=['POST'])
        async def handle_start_ugc():
            try:
                data = await request.get_json()
                if not data: return jsonify({"error": "Invalid JSON payload"}), 400
                client_industry = data.get('client_industry')
                if not client_industry: return jsonify({"error": "Missing 'client_industry'"}), 400

                num_videos = int(data.get('num_videos', 1))
                script = data.get('script')
                target_services = data.get('target_services') # Optional list

                # Run workflow in background
                asyncio.create_task(self.start_ugc_workflow(
                    client_industry=client_industry, num_videos=num_videos,
                    script=script, target_services=target_services
                ))
                return jsonify({"status": "UGC workflow initiated", "client_industry": client_industry}), 202
            except Exception as e:
                logger.error(f"Failed to initiate UGC workflow via API: {e}", exc_info=True)
                return jsonify({"status": "error", "message": str(e)}), 500

        # Route for serving temporary hosted audio
        @self.app.route('/hosted_audio/<path:filename>')
        async def serve_hosted_audio(filename):
            # Security: Basic path traversal prevention
            if '..' in filename or filename.startswith('/'):
                return "Forbidden", 403
            try:
                safe_path = os.path.join(self.temp_audio_dir, filename)
                if os.path.exists(safe_path) and os.path.isfile(safe_path):
                     # Determine mimetype (basic)
                     mimetype = 'audio/wav' if filename.lower().endswith('.wav') else 'audio/mpeg'
                     return await send_file(safe_path, mimetype=mimetype)
                else:
                    return jsonify({"error": "File not found"}), 404
            except Exception as e:
                logger.error(f"Error serving hosted audio {filename}: {e}")
                return jsonify({"error": "Internal server error"}), 500

        # TODO: Add Twilio Webhook endpoint for incoming calls/status updates
        # @self.app.route('/twilio/voice', methods=['POST'])
        # async def handle_twilio_voice(): ...

        # TODO: Add Tracking Pixel endpoint
        # @self.app.route('/track/<tracking_id>.png')
        # async def handle_tracking_pixel(tracking_id): ...

        logger.info("Quart routes configured.")


    # --- Main Execution Loop ---
    async def run(self):
        """Initializes and runs the AI Agency Orchestrator."""
        logger.info("Orchestrator starting full initialization sequence...")
        self.running = False # Ensure not running until setup is complete

        try:
            # --- Initialization Sequence ---
            if not await self.initialize_database(): raise RuntimeError("Database initialization failed.")
            if not await self._run_encryption_migration_v2(): raise RuntimeError("Encryption migration failed.")
            if not await self.initialize_clients(): raise RuntimeError("LLM Client initialization failed.")
            if not await self.initialize_agents(): raise RuntimeError("Agent initialization failed.")

            logger.info("Orchestrator initialization complete.")
            # Note: Approval is now handled via API, not blocking startup.

            # --- Start Background Tasks ---
            logger.info("Starting background agent run loops and periodic tasks...")
            self.background_tasks = set()
            # Start individual agent run loops
            for agent_name, agent in self.agents.items():
                 if hasattr(agent, 'run') and callable(agent.run):
                      task = asyncio.create_task(agent.run(), name=f"AgentLoop_{agent_name}")
                      self.background_tasks.add(task)
                 else:
                      logger.warning(f"Agent {agent_name} does not have a callable run method.")

            # Start orchestrator periodic tasks
            self.background_tasks.add(asyncio.create_task(self._run_periodic_data_purge(), name="PeriodicDataPurge"))
            self.background_tasks.add(asyncio.create_task(self._run_periodic_feedback_collection(), name="PeriodicFeedback"))
            # Add other orchestrator background tasks if needed (e.g., monitoring)

            logger.info(f"Started {len(self.background_tasks)} background tasks.")

            # --- Main Orchestration Loop ---
            logger.info("Orchestrator entering main operational state (API driven).")
            self.running = True # Set running flag
            self.last_feedback_time = time.time()
            self.last_purge_time = time.time()

            # The Quart app runs the web server loop. Orchestrator logic is now primarily
            # event-driven (API calls, agent reports) or handled by periodic background tasks.
            # We keep this loop alive to manage background tasks and potential future polling needs.
            while self.running:
                try:
                    # Check health of background tasks periodically
                    tasks_to_remove = set()
                    for task in self.background_tasks:
                        if task.done():
                            tasks_to_remove.add(task)
                            try:
                                # Log result or exception of completed/failed tasks
                                result = task.result()
                                logger.info(f"Background task {task.get_name()} completed.")
                            except asyncio.CancelledError:
                                 logger.info(f"Background task {task.get_name()} was cancelled.")
                            except Exception as task_exc:
                                logger.error(f"Background task {task.get_name()} failed: {task_exc}", exc_info=True)
                                # Optionally restart failed tasks? Requires careful state management.
                    self.background_tasks -= tasks_to_remove

                    await asyncio.sleep(60) # Main loop check interval

                except asyncio.CancelledError:
                     logger.info("Orchestrator main loop cancelled.")
                     break

            logger.info("Orchestrator main loop finished.")

        except asyncio.CancelledError:
            logger.info("Orchestrator run task cancelled during setup.")
        except Exception as e:
            logger.critical(f"CRITICAL ERROR during Orchestrator setup or main loop: {e}", exc_info=True)
            self.running = False
            try: await self.send_notification("CRITICAL Orchestrator Failure", f"Orchestrator failed: {e}")
            except Exception as report_err: logger.error(f"Failed to send critical failure report: {report_err}")
        finally:
            logger.info("Orchestrator shutdown sequence initiated.")
            self.running = False
            cancelled_tasks = []
            for task in self.background_tasks:
                if task and not task.done():
                    task.cancel()
                    cancelled_tasks.append(task)
            if cancelled_tasks:
                 logger.info(f"Waiting for {len(cancelled_tasks)} background tasks to cancel...")
                 await asyncio.gather(*cancelled_tasks, return_exceptions=True)
                 logger.info("Background tasks cancellation complete.")
            else:
                 logger.info("No active background tasks needed cancellation.")
            logger.info("Orchestrator shutdown complete.")

# --- Main Execution Guard ---
# Note: Running the Quart app is typically done separately, not within this class structure.
# The main.py file should handle creating the Orchestrator instance and running the Quart app.
# This run() method focuses on the internal async logic and background tasks.

# Example of how main.py might use this:
# if __name__ == "__main__":
#     orchestrator = Orchestrator()
#     # Start the orchestrator's internal loops/tasks in the background
#     asyncio.create_task(orchestrator.run())
#     # Run the Quart web server
#     orchestrator.app.run(host='0.0.0.0', port=5000, debug=False) # Set debug=False for production