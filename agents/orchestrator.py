# Filename: agents/orchestrator.py
# Description: Central coordinator for the AI Agency, managing agents, workflows, and resources.
# Version: 2.0 (Genius Agentic - Streamlined, Postgres Focus, No Vault)

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
from typing import Dict, Optional, Tuple, Any, List, AsyncGenerator, Callable, Type # Added Type

# --- Core Framework Imports ---
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, delete, func, update, text, case # Added case
from quart import Quart, request, jsonify, websocket, send_file, url_for # Keep Quart for UI
import psutil
from openai import AsyncOpenAI as AsyncLLMClient # Standardized name
import pybreaker
import websockets
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from prometheus_client import Counter, Gauge, start_http_server

# --- Project Imports ---
from config.settings import settings # Use validated settings
# Removed VaultError import
from utils.database import encrypt_data, decrypt_data, get_session, AsyncSessionMaker # Use updated DB utils
from utils.notifications import send_notification # Keep notifications for now

# Import *only* the core agent classes needed in the streamlined architecture
# Use production-ready base class if available
try:
    from agents.base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
except ImportError:
    logger.warning("Production base agent not found, using GeniusAgentBase. Ensure base_agent_prod.py is used.")
    from agents.base_agent import GeniusAgentBase

from agents.think_tool import ThinkTool
from agents.browsing_agent import BrowsingAgent # Keep for execution
from agents.email_agent import EmailAgent
from agents.voice_sales_agent import VoiceSalesAgent
from agents.legal_agent import LegalAgent
# Removed imports for consolidated agents: OSINTAgent, ScoringAgent, OptimizationAgent, ProgrammerAgent, SocialMediaManager, BudgetAgent

# Import database models
from models import Base, Client, Metric, ExpenseLog, MigrationStatus, KnowledgeFragment, AccountCredentials, CallLog, Invoice, StrategicDirective # Use AccountCredentials

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.FileHandler("agency.log", mode='a'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Metrics & Circuit Breakers ---
agent_status_gauge = Gauge('agent_status', 'Status of agents (1=running, 0=stopped/error)', ['agent_name'])
error_counter = Counter('agent_errors_total', 'Total number of errors per agent', ['agent_name'])
tasks_processed_counter = Counter('tasks_processed_total', 'Total number of tasks processed by agent', ['agent_name'])
llm_client_breaker = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=60 * 5, name="LLMClientBreaker")

# --- Helper Functions ---
def create_session_maker(db_url: str, schema: str = 'public') -> async_sessionmaker[AsyncSession]:
    """Creates an async session maker configured for a specific schema."""
    try:
        engine = create_async_engine(db_url, echo=False, pool_pre_ping=True, pool_recycle=3600)
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        logger.info(f"Async session maker created for schema '{schema}'.")
        return session_factory
    except Exception as e:
        logger.critical(f"Failed to create database engine or session maker: {e}", exc_info=True)
        raise

# --- Orchestrator Class ---
class Orchestrator:
    """Central coordinator, streamlined for core sales workflow and Postgres."""

    def __init__(self, schema='public'):
        self.config = settings # Use imported, validated settings
        self.session_maker = create_session_maker(self.config.DATABASE_URL, schema)
        self.agents: Dict[str, GeniusAgentBase] = {} # Core agents only
        self.app = Quart(__name__, template_folder='../ui/templates', static_folder='../ui/static')
        self.setup_routes()
        try:
            start_http_server(8001)
            logger.info("Prometheus metrics server started on port 8001.")
        except OSError as e: logger.warning(f"Could not start Prometheus server on port 8001: {e}")
        except Exception as e: logger.error(f"Failed to start Prometheus server: {e}", exc_info=True)

        self.meta_prompt = self.config.META_PROMPT
        self.approved = False # Requires API approval

        # LLM Client Management State (Reads keys from settings/env)
        # ### Phase 2 Plan Ref: 5.1 (Load keys from settings)
        self._llm_client_cache: Dict[str, AsyncLLMClient] = {}
        self._llm_client_status: Dict[str, Dict[str, Any]] = {}
        self._llm_client_keys: List[str] = []
        self._llm_client_round_robin_index = 0

        # In-memory Cache (Simple)
        self._cache: Dict[str, Tuple[Any, float]] = {}

        # Deepgram WebSocket Registry
        self.deepgram_connections: Dict[str, websockets.client.WebSocketClientProtocol] = {}
        self.temp_audio_dir = "/app/temp_audio"
        os.makedirs(self.temp_audio_dir, exist_ok=True)

        # Periodic Task Timing
        self.feedback_interval_seconds: int = int(self.config.get("THINKTOOL_FEEDBACK_INTERVAL_SECONDS", 300))
        self.purge_interval_seconds: int = int(self.config.get("DATA_PURGE_INTERVAL_SECONDS", 86400))
        self.last_feedback_time: float = 0.0
        self.last_purge_time: float = 0.0

        self.running: bool = False
        self.background_tasks = set()

        logger.info("Orchestrator v2.0 (Streamlined) initialized.")
        # Initialization sequence called from run()

    async def initialize_database(self):
        """Initialize or update the database schema."""
        logger.info("Initializing database schema...")
        try:
            # Use the session_maker's engine
            async with self.session_maker() as session:
                 async with session.begin():
                      # This assumes models.py defines Base.metadata correctly
                      # We need the engine associated with the session_maker
                      engine = session.get_bind()
                      await engine.run_sync(Base.metadata.create_all)
            logger.info("Database schema initialization complete.")
            return True
        except Exception as e:
            logger.critical(f"Failed to initialize database: {e}", exc_info=True)
            return False

    async def initialize_clients(self):
        """Initialize LLM API clients from settings (env vars)."""
        # ### Phase 2 Plan Ref: 5.1 (Load keys from settings)
        logger.info("Initializing LLM clients from settings/environment variables...")
        self._llm_client_keys = []
        self._llm_client_cache = {}
        self._llm_client_status = {}

        # --- Load Primary OpenRouter Key ---
        primary_openrouter_key = os.getenv("OPENROUTER_API_KEY") # Read directly
        if primary_openrouter_key:
            try:
                key_id = primary_openrouter_key[-6:]
                self._llm_client_cache[key_id] = AsyncLLMClient(api_key=primary_openrouter_key, base_url="https://openrouter.ai/api/v1")
                self._llm_client_status[key_id] = {'status': 'available', 'reason': None, 'unavailable_until': 0}
                self._llm_client_keys.append(key_id)
                logger.info(f"Primary OpenRouter client initialized (Key ID: ...{key_id}).")
            except Exception as e:
                logger.error(f"Failed to initialize primary OpenRouter client: {e}", exc_info=True)
        else:
            logger.warning("Primary OPENROUTER_API_KEY not found in environment variables.")

        # --- Load other keys similarly if needed (e.g., DeepSeek) ---
        # primary_deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        # if primary_deepseek_key: ... etc.

        # --- Load Additional Keys (Example: If you have OR_KEY_1, OR_KEY_2 etc. in env) ---
        i = 1
        while True:
            additional_key = os.getenv(f"OPENROUTER_API_KEY_{i}")
            if not additional_key: break
            try:
                key_id = additional_key[-6:]
                if key_id not in self._llm_client_cache:
                    self._llm_client_cache[key_id] = AsyncLLMClient(api_key=additional_key, base_url="https://openrouter.ai/api/v1")
                    self._llm_client_status[key_id] = {'status': 'available', 'reason': None, 'unavailable_until': 0}
                    self._llm_client_keys.append(key_id)
                    logger.info(f"Loaded additional OpenRouter client {i} (Key ID: ...{key_id}).")
            except Exception as e:
                logger.error(f"Failed to initialize additional OpenRouter client {i}: {e}")
            i += 1

        if not self._llm_client_keys:
             logger.critical("CRITICAL: No LLM API clients could be initialized. Core functions will fail.")
             return False # Indicate failure

        logger.info(f"LLM Client Initialization complete. Total usable keys: {len(self._llm_client_keys)}")
        return True

    async def initialize_agents(self):
        """Initialize the CORE agents, passing necessary secrets from settings."""
        # ### Phase 2 Plan Ref: 5.3 (Initialize CORE agents)
        logger.info("Initializing CORE agents...")
        initialization_failed = False
        try:
            # ThinkTool is essential and needs DB access
            self.agents['think'] = ThinkTool(self.session_maker, self.config, self)

            # BrowsingAgent needs proxy password
            smartproxy_pass = os.getenv("SMARTPROXY_PASSWORD")
            if not smartproxy_pass: logger.warning("SMARTPROXY_PASSWORD not set, BrowsingAgent proxy auth will fail.")
            self.agents['browsing'] = BrowsingAgent(self.session_maker, self, smartproxy_password=smartproxy_pass) # Pass orchestrator

            # EmailAgent needs SMTP/IMAP passwords
            smtp_pass = os.getenv("HOSTINGER_SMTP_PASS")
            imap_pass = os.getenv("HOSTINGER_IMAP_PASS")
            if not smtp_pass or not imap_pass: logger.critical("Hostinger SMTP/IMAP password missing. EmailAgent will fail.") ; initialization_failed = True
            self.agents['email'] = EmailAgent(self.session_maker, self, smtp_password=smtp_pass, imap_password=imap_pass)

            # VoiceSalesAgent needs Twilio/Deepgram keys
            twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
            deepgram_key = os.getenv("DEEPGRAM_API_KEY")
            if not twilio_token or not deepgram_key: logger.critical("Twilio Auth Token or Deepgram API Key missing. VoiceSalesAgent will fail.") ; initialization_failed = True
            self.agents['voice_sales'] = VoiceSalesAgent(self.session_maker, self, twilio_auth_token=twilio_token, deepgram_api_key=deepgram_key)

            # LegalAgent
            self.agents['legal'] = LegalAgent(self.session_maker, self) # Pass orchestrator

            # Set Prometheus gauges
            for name in self.agents.keys():
                agent_status_gauge.labels(agent_name=name).set(0) # idle

            if initialization_failed:
                 raise RuntimeError("Failed to initialize one or more critical agents due to missing secrets.")

            logger.info(f"Core agents initialized: {list(self.agents.keys())}")
            return True

        except Exception as e:
            logger.error(f"Core agent initialization failed: {e}", exc_info=True)
            return False

    # --- LLM Client Management (Adapted for direct env var access) ---
    @llm_client_breaker
    async def get_available_llm_clients(self) -> List[AsyncLLMClient]: # Renamed for clarity
        """Gets available LLM client instances based on status and rotation."""
        now = time.time()
        available_keys = []
        # Check status and cooldowns
        for key_id in self._llm_client_keys:
             status_info = self._llm_client_status.get(key_id, {'status': 'available', 'unavailable_until': 0})
             if status_info['status'] == 'available' or now >= status_info.get('unavailable_until', 0):
                  if status_info['status'] == 'unavailable': # Cooldown expired
                       status_info['status'] = 'available'
                       status_info['reason'] = None
                       status_info['unavailable_until'] = 0
                       logger.info(f"LLM client key ...{key_id} marked as available after cooldown.")
                  available_keys.append(key_id)

        if not available_keys:
            logger.error("No available LLM clients found!")
            return []

        # Simple round-robin
        selected_key_id = available_keys[self._llm_client_round_robin_index % len(available_keys)]
        self._llm_client_round_robin_index += 1

        selected_client = self._llm_client_cache.get(selected_key_id)
        if not selected_client:
             logger.error(f"Client instance not found in cache for available key ID ...{selected_key_id}. This should not happen.")
             # Attempt to remove the bad key entry
             if key_id in self._llm_client_keys: self._llm_client_keys.remove(key_id)
             self._llm_client_status.pop(key_id, None)
             self._llm_client_cache.pop(key_id, None)
             return await self.get_available_llm_clients() # Retry selection

        logger.debug(f"Selected LLM client with key ID: ...{selected_key_id}")
        return [selected_client] # Return list with one client for now

    async def report_client_issue(self, api_key_identifier: str, issue_type: str):
        """Marks an LLM client as unavailable due to an issue."""
        if api_key_identifier not in self._llm_client_status:
            logger.warning(f"Attempted to report issue for unknown LLM client identifier: ...{api_key_identifier}")
            return

        now = time.time()
        cooldown_seconds = 60 * 5 # Default 5 minutes
        status = 'unavailable'
        reason = issue_type

        if issue_type == 'auth_error':
            cooldown_seconds = 60 * 60 * 24 * 365 # Permanent disable
            reason = "Authentication Error"
            logger.critical(f"LLM client key ...{api_key_identifier} marked permanently unavailable due to auth error.")
        elif issue_type == 'rate_limit': cooldown_seconds = 60 * 2; reason = "Rate Limited"
        elif issue_type == 'timeout_error': cooldown_seconds = 60 * 3; reason = "Timeout"
        else: reason = "General Error"

        self._llm_client_status[api_key_identifier] = {
            'status': status, 'reason': reason, 'unavailable_until': now + cooldown_seconds
        }
        logger.warning(f"LLM client key ...{api_key_identifier} marked unavailable until {datetime.fromtimestamp(now + cooldown_seconds)}. Reason: {reason}")

    @llm_client_breaker
    async def call_llm(self, agent_name: str, prompt: str, temperature: float = 0.5, max_tokens: int = 1024, is_json_output: bool = False, model_preference: Optional[List[str]] = None) -> Optional[str]:
        """Handles selecting an available client and making the LLM call."""
        # ### Phase 2 Plan Ref: 5.4 (Implement call_llm)
        selected_clients = await self.get_available_llm_clients()
        if not selected_clients:
            logger.error(f"Agent '{agent_name}' failed to get an available LLM client.")
            return None

        llm_client = selected_clients
        # Find the key ID associated with this client instance
        api_key_identifier = "unknown"
        for key_id, client_instance in self._llm_client_cache.items():
            if client_instance == llm_client:
                api_key_identifier = key_id
                break

        # Determine model using the mapping in settings
        model_key = 'default_llm' # Fallback
        # Simple mapping based on agent name (can be made more sophisticated)
        if agent_name == 'ThinkTool': model_key = 'think_general' # Use the general think model
        elif agent_name == 'EmailAgent': model_key = 'email_draft'
        elif agent_name == 'VoiceSalesAgent': model_key = 'voice_response'
        elif agent_name == 'LegalAgent': model_key = 'legal_validation'
        # Add more specific mappings if needed based on task_context passed to call_llm in future
        model_name = self.config.OPENROUTER_MODELS.get(model_key, self.config.OPENROUTER_MODELS['default_llm'])

        # --- Caching Logic ---
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        cache_key_parts = ["llm_call", prompt_hash, model_name, str(temperature), str(max_tokens), str(is_json_output)]
        cache_key = ":".join(cache_key_parts)
        cache_ttl = 3600 # 1 hour cache

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
                max_tokens=max_tokens, response_format=response_format, timeout=120
            )
            content = response.choices.message.content.strip()

            # Track usage & cost
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = input_tokens + output_tokens
            # Rough cost estimate (adjust based on actual model pricing)
            cost_per_million_input = 0.50 # Example for a cheap model
            cost_per_million_output = 1.50 # Example
            cost = (input_tokens / 1_000_000 * cost_per_million_input) + (output_tokens / 1_000_000 * cost_per_million_output)
            logger.debug(f"LLM Call Usage: Model={model_name}, Tokens={total_tokens}, Est. Cost=${cost:.6f}")
            # Report expense via internal method
            await self.report_expense(agent_name, cost, "LLM", f"LLM call ({model_name}). Tokens: {total_tokens}.")

            # Add to cache
            self.add_to_cache(cache_key, content, ttl_seconds=cache_ttl)

            return content

        except Exception as e:
            error_str = str(e).lower()
            issue_type = "llm_error" # Default
            if "rate limit" in error_str or "quota" in error_str: issue_type = "rate_limit"
            elif "authentication" in error_str: issue_type = "auth_error"
            elif "timeout" in error_str: issue_type = "timeout_error"
            logger.warning(f"Orchestrator LLM call failed: Agent={agent_name}, Model={model_name}, Key=...{api_key_identifier}, Type={issue_type}, Error={e}")
            await self.report_client_issue(api_key_identifier, issue_type)
            raise # Re-raise for tenacity/caller handling

    # --- Proxy Management ---
    async def get_proxy(self, purpose: str = "general", target_url: Optional[str] = None, account_identifier: Optional[str] = None) -> Optional[str]:
        """Gets an available proxy, potentially specific to an account via BrowsingAgent."""
        # ### Phase 2 Plan Ref: 5.5 (Implement get_proxy calling BrowsingAgent)
        browsing_agent = self.agents.get('browsing')
        if browsing_agent and hasattr(browsing_agent, 'get_proxy_for_account'):
            try:
                proxy_url = await browsing_agent.get_proxy_for_account(account_identifier=account_identifier, purpose=purpose, target_url=target_url)
                if proxy_url:
                    logger.debug(f"Obtained proxy via BrowsingAgent for purpose '{purpose}' (Account: {account_identifier or 'any'})")
                    return proxy_url
                else:
                    logger.warning(f"No available proxy from BrowsingAgent for purpose '{purpose}'.")
                    return None
            except Exception as e:
                logger.error(f"Error getting proxy from BrowsingAgent: {e}", exc_info=True)
                return None
        else:
            logger.error("BrowsingAgent or its get_proxy_for_account method not available.")
            return None

    # --- Tool/Task Delegation ---
    async def use_tool(self, tool_name: str, tool_params: Dict[str, Any]) -> Dict[str, Any]:
        """Delegates execution of standard tools (read, write, exec) - Now directly via asyncio."""
        # ### Phase 2 Plan Ref: 5.6 (Refactor use_tool - Direct execution)
        # This bypasses ProgrammerAgent for basic filesystem/shell ops for efficiency
        self.logger.debug(f"Orchestrator executing tool: {tool_name} with params: {tool_params}")
        result = {"status": "failure", "message": f"Tool '{tool_name}' execution failed."}
        command_list = None

        try:
            if tool_name == 'read_file':
                file_path = tool_params.get('path')
                if not file_path: raise ValueError("Missing 'path' parameter for read_file")
                # Security: Add path validation here if needed (e.g., ensure it's within workspace)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # TODO: Handle start/end lines if needed by reading line by line
                        content = f.read()
                    result = {"status": "success", "content": content}
                except FileNotFoundError:
                    result = {"status": "failure", "message": f"File not found: {file_path}"}
                except Exception as e:
                    result = {"status": "failure", "message": f"Error reading file {file_path}: {e}"}

            elif tool_name == 'write_to_file':
                file_path = tool_params.get('path')
                content = tool_params.get('content')
                append = tool_params.get('append', False)
                if not file_path or content is None: raise ValueError("Missing 'path' or 'content' for write_to_file")
                # Security: Add path validation
                os.makedirs(os.path.dirname(file_path), exist_ok=True) # Ensure directory exists
                mode = 'a' if append else 'w'
                try:
                    with open(file_path, mode, encoding='utf-8') as f:
                        if tool_params.get('leading_newline') and append: f.write("\n")
                        f.write(content)
                        if tool_params.get('trailing_newline'): f.write("\n")
                    result = {"status": "success", "message": f"File '{file_path}' written successfully."}
                except Exception as e:
                    result = {"status": "failure", "message": f"Error writing file {file_path}: {e}"}

            elif tool_name == 'execute_command':
                command_str = tool_params.get('command')
                if not command_str: raise ValueError("Missing 'command' for execute_command")
                # Security WARNING: Executing arbitrary commands is dangerous.
                # Implement strict validation or sandboxing in a real system.
                # Using asyncio.create_subprocess_shell for simplicity here, but consider security implications.
                self.logger.warning(f"Executing command via shell: {command_str}") # Log potentially dangerous operation
                process = await asyncio.create_subprocess_shell(
                    command_str,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300) # 5 min timeout
                stdout_str = stdout.decode(errors='ignore').strip()
                stderr_str = stderr.decode(errors='ignore').strip()
                if process.returncode == 0:
                    result = {"status": "success", "returncode": 0, "stdout": stdout_str, "stderr": stderr_str}
                else:
                    result = {"status": "failure", "returncode": process.returncode, "stdout": stdout_str, "stderr": stderr_str, "message": f"Command failed with code {process.returncode}"}

            # Add other direct tool implementations if needed (e.g., list_files, search_files)
            # elif tool_name == 'list_files': ...
            # elif tool_name == 'search_files': ...

            else:
                result["message"] = f"Tool '{tool_name}' not directly implemented by Orchestrator."
                self.logger.warning(result["message"])

        except ValueError as ve:
             result["message"] = f"Tool '{tool_name}' parameter error: {ve}"
             self.logger.error(result["message"])
        except asyncio.TimeoutError:
             result["message"] = f"Tool '{tool_name}' (command) timed out."
             self.logger.error(result["message"])
        except Exception as e:
            result["message"] = f"Exception executing tool '{tool_name}': {e}"
            self.logger.error(result["message"], exc_info=True)

        return result

    async def delegate_task(self, target_agent_name: str, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Delegates a task to a specific CORE agent."""
        # ### Phase 2 Plan Ref: 5.7 (Implement delegate_task for core agents)
        target_agent = self.agents.get(target_agent_name)
        if not target_agent:
            logger.error(f"Cannot delegate task: Agent '{target_agent_name}' not found or not initialized.")
            return {"status": "error", "message": f"Agent '{target_agent_name}' not found."}

        action_desc = task_details.get('action', task_details.get('description', 'No description'))
        logger.info(f"Delegating task to {target_agent_name}: {action_desc}")
        try:
            # Core agents use execute_task directly now (no internal queues in base)
            if hasattr(target_agent, 'execute_task') and callable(target_agent.execute_task):
                # Execute directly and await result
                result = await target_agent.execute_task(task_details)
                # Log task processing for metrics
                tasks_processed_counter.labels(agent_name=target_agent_name).inc()
                return result
            else:
                logger.error(f"Agent '{target_agent_name}' has no callable execute_task method.")
                return {"status": "error", "message": f"Agent '{target_agent_name}' cannot execute tasks."}
        except Exception as e:
            logger.error(f"Error delegating task to {target_agent_name}: {e}", exc_info=True)
            # Report error via internal method
            await self.report_error(target_agent_name, f"Exception during task delegation: {e}")
            return {"status": "error", "message": f"Exception during task delegation to {target_agent_name}: {e}"}

    # --- Workflow Management (Example: UGC) ---
    async def start_ugc_workflow(self, client_industry: str, num_videos: int = 1, script: str = None, target_services: list = None):
        """Initiates the UGC content generation workflow (delegated steps)."""
        # This workflow now relies on ThinkTool for planning and BrowsingAgent for execution
        self.logger.info(f"Received request to start UGC workflow for {client_industry}.")
        # Delegate planning to ThinkTool
        plan_task = {
            "action": "plan_ugc_workflow", # Define this action for ThinkTool
            "client_industry": client_industry,
            "num_videos": num_videos,
            "initial_script": script,
            "target_services": target_services or ["heygen.com", "descript.com"], # Default services
            "description": f"Plan UGC video workflow for {client_industry}"
        }
        await self.delegate_task("ThinkTool", plan_task)
        # ThinkTool will generate directives for BrowsingAgent to execute steps

    async def report_ugc_step_complete(self, workflow_id: str, completed_step: str, result: dict, current_state: dict):
        """Handles completion reports and triggers ThinkTool for next step planning."""
        # This is now likely handled by ThinkTool analyzing results and issuing next directive
        self.logger.info(f"Received UGC step completion for {workflow_id}, step: {completed_step}. Forwarding to ThinkTool.")
        feedback_task = {
            "action": "process_workflow_step_feedback", # Define for ThinkTool
            "workflow_id": workflow_id,
            "completed_step": completed_step,
            "step_result": result,
            "current_workflow_state": current_state, # Pass state for context
            "description": f"Process feedback for UGC workflow {workflow_id} step {completed_step}"
        }
        await self.delegate_task("ThinkTool", feedback_task)

    # --- Feedback Loop ---
    async def handle_feedback_trigger(self, all_insights: Dict[str, Dict[str, Any]]):
        """Triggers ThinkTool to process collected agent insights."""
        # ### Phase 2 Plan Ref: 5.8 (Implement handle_feedback_trigger)
        think_agent = self.agents.get("think")
        if think_agent and hasattr(think_agent, 'handle_feedback'):
            logger.info("Forwarding collected insights to ThinkTool for processing.")
            try:
                # Delegate the processing to ThinkTool
                feedback_task = {
                    "action": "process_feedback",
                    "feedback_data": all_insights,
                    "description": "Process periodic agent feedback insights"
                }
                await self.delegate_task("ThinkTool", feedback_task)
            except Exception as e:
                logger.error(f"Error delegating feedback processing to ThinkTool: {e}", exc_info=True)
        else:
            logger.warning("ThinkTool agent not found or missing 'handle_feedback' capability.")

    # --- Tool Installation (Simplified) ---
    async def handle_install_tool_request(self, requesting_agent_name: str, tool_details: Dict[str, Any]):
        """Handles tool installation request - attempts direct execution."""
        # ### Phase 2 Plan Ref: 5.9 (Simplify handle_install_tool_request)
        tool_name = tool_details.get('tool_name')
        logger.info(f"Received tool install request from {requesting_agent_name} for: {tool_name}")

        # Basic command determination (could be moved to ThinkTool/Programmer if kept)
        command = self._determine_basic_install_command(tool_details)
        if not command:
            logger.error(f"Could not determine install command for {tool_name}.")
            await self.report_error(requesting_agent_name, f"Failed to determine install command for {tool_name}")
            return

        logger.warning(f"Attempting direct execution of install command (requires sudo privileges in container potentially): {command}")
        install_result = await self.use_tool('execute_command', {'command': command})

        if install_result.get('status') == 'success':
            logger.info(f"Installation command for {tool_name} executed successfully (check logs for details).")
            # Optionally notify requesting agent?
        else:
            error_msg = install_result.get('message', install_result.get('stderr', 'Unknown install error'))
            logger.error(f"Installation command failed for {tool_name}: {error_msg}")
            await self.report_error(requesting_agent_name, f"Installation failed for {tool_name}: {error_msg}")

    def _determine_basic_install_command(self, tool_details: Dict[str, Any]) -> Optional[str]:
        """Basic helper to guess install command (less robust than ProgrammerAgent)."""
        tool_name = tool_details.get('tool_name')
        pkg_manager = tool_details.get('package_manager', 'apt').lower()
        pkg_name = tool_details.get('package_name') or tool_name.lower()
        git_repo = tool_details.get('git_repo')

        if git_repo: return f"git clone --depth 1 {shlex.quote(git_repo)} /tmp/{tool_name}_install && cd /tmp/{tool_name}_install && pip3 install . && cd / && rm -rf /tmp/{tool_name}_install" # Simplified git install
        if pkg_manager == 'apt': return f"sudo apt-get update && sudo apt-get install -y {shlex.quote(pkg_name)}"
        if pkg_manager == 'pip': return f"pip3 install --user {shlex.quote(pkg_name)}"
        # Add other managers if needed
        return None

    # --- Invoice Generation ---
    async def request_invoice_generation(self, client_id: int, amount: float, source_ref: str):
        """Handles request to generate an invoice (logs to DB)."""
        # ### Phase 2 Plan Ref: 5.10 (Implement request_invoice_generation)
        logger.info(f"Invoice generation requested: ClientID={client_id}, Amount=${amount:.2f}, SourceRef={source_ref}")
        try:
            async with self.session_maker() as session:
                async with session.begin(): # Transaction
                    client = await session.get(Client, client_id)
                    if not client:
                        logger.error(f"Cannot generate invoice: Client ID {client_id} not found.")
                        return

                    # Create Invoice record
                    invoice = Invoice(
                        client_id=client_id, amount=amount, status='pending',
                        timestamp=datetime.now(timezone.utc)
                        # Add due_date, payment_link etc. if generated by ThinkTool/LegalAgent later
                    )
                    session.add(invoice)
                    # Commit happens automatically
                await session.refresh(invoice) # Get the ID after commit
                logger.info(f"Created pending Invoice record ID {invoice.id} for Client {client_id}.")

                # Notify user/trigger external process
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
        # ### Phase 2 Plan Ref: 5.11 (Implement _run_periodic_data_purge)
        while self.running:
            try:
                await asyncio.sleep(self.purge_interval_seconds)
                if not self.running or self._stop_event.is_set(): break
                logger.info("Orchestrator triggering periodic data purge via ThinkTool...")
                await self.delegate_task("ThinkTool", {"action": "purge_old_knowledge"})
            except asyncio.CancelledError: logger.info("Periodic data purge task cancelled."); break
            except Exception as e: logger.error(f"Error in periodic data purge task: {e}", exc_info=True); await asyncio.sleep(60 * 5)

    async def _run_periodic_feedback_collection(self):
        """Periodically collects insights from agents and triggers processing."""
        # ### Phase 2 Plan Ref: 5.12 (Implement _run_periodic_feedback_collection)
        while self.running:
            try:
                await asyncio.sleep(self.feedback_interval_seconds)
                if not self.running or self._stop_event.is_set(): break
                logger.info("Orchestrator initiating periodic feedback collection...")
                all_insights = {}
                agent_items = list(self.agents.items())
                insight_tasks = []

                for agent_name, agent_instance in agent_items:
                    if agent_instance and hasattr(agent_instance, 'collect_insights') and callable(agent_instance.collect_insights):
                        insight_tasks.append(asyncio.create_task(agent_instance.collect_insights(), name=f"CollectInsights_{agent_name}"))
                    else: logger.debug(f"Agent {agent_name} does not have collect_insights method.")

                results = await asyncio.gather(*insight_tasks, return_exceptions=True)

                for i, agent_tuple in enumerate(agent_items):
                     agent_name = agent_tuple
                     # Check if index exists before accessing results
                     if i < len(results):
                          result = results[i]
                          if isinstance(result, Exception):
                               logger.error(f"Error collecting insights from {agent_name}: {result}", exc_info=result)
                               all_insights[agent_name] = {"error": f"Failed to collect insights: {result}"}
                          elif result: all_insights[agent_name] = result
                          else: logger.debug(f"No insights returned from {agent_name}.")
                     else: logger.error(f"Result index out of bounds for agent {agent_name}")


                if all_insights:
                    logger.info(f"Collected insights from {len(all_insights)} agents. Triggering feedback handler.")
                    await self.handle_feedback_trigger(all_insights)
                else: logger.info("No insights collected in this feedback cycle.")

            except asyncio.CancelledError: logger.info("Periodic feedback collection task cancelled."); break
            except Exception as e: logger.error(f"Error in periodic feedback collection task: {e}", exc_info=True); await asyncio.sleep(60 * 5)

    # --- Cache Management ---
    def get_from_cache(self, key: str) -> Optional[Any]:
        """Retrieves an item from the in-memory cache if valid."""
        if key in self._cache:
            value, expiry_ts = self._cache[key]
            if time.time() < expiry_ts: logger.debug(f"Cache hit for key: {key}"); return value
            else: logger.debug(f"Cache expired for key: {key}"); del self._cache[key]
        logger.debug(f"Cache miss for key: {key}")
        return None

    def add_to_cache(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Adds an item to the in-memory cache with a TTL."""
        if ttl_seconds <= 0: return
        expiry_ts = time.time() + ttl_seconds
        self._cache[key] = (value, expiry_ts)
        logger.debug(f"Added key '{key}' to cache with TTL {ttl_seconds}s.")
        # Optional: Prune cache if size exceeds a limit

    # --- Notification & Error Reporting ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=5, max=60))
    async def send_notification(self, subject, body):
        """Sends email notification via utility, fetching password from env vars."""
        # ### Phase 2 Plan Ref: 5.13 (Update send_notification - No Vault)
        try:
            smtp_pass = os.getenv('HOSTINGER_SMTP_PASS') # Read directly
            if not smtp_pass: raise ValueError("SMTP password not found in environment variables.")
            # Assuming send_notification utility is updated to accept password
            await send_notification(subject, body, self.config, smtp_password=smtp_pass)
        except ValueError as ve: logger.error(f"Failed notification '{subject}' due to config error: {ve}")
        except Exception as e: logger.error(f"Error sending notification '{subject}': {e}", exc_info=True); raise

    async def report_error(self, agent_name: str, error_message: str):
        """Report errors via notification and log metric."""
        logger.error(f"ERROR reported by {agent_name}: {error_message}")
        error_counter.labels(agent_name=agent_name).inc()
        subject = f"ERROR in {agent_name}"
        body = f"Timestamp: {datetime.now(timezone.utc).isoformat()}\nAgent: {agent_name}\nError details:\n{error_message}"
        await self.send_notification(subject, body)

    async def report_expense(self, agent_name: str, amount: float, category: str, description: str):
        """Logs expense directly to the database."""
        # ### Phase 2 Plan Ref: 5.2 (Implement report_expense)
        logger.debug(f"Received expense report from {agent_name}: ${amount:.6f} [{category}] - {description}")
        if not self.session_maker:
            logger.error("Cannot log expense: Database session maker unavailable.")
            return
        try:
            # Encrypt description before logging
            encrypted_desc = encrypt_data(description)
            if encrypted_desc is None:
                logger.error(f"Failed to encrypt expense description for {agent_name}. Logging without encryption.")
                encrypted_desc = description # Log unencrypted as fallback? Risky. Better to fail?

            async with self.session_maker() as session:
                async with session.begin(): # Transaction
                    log_entry = ExpenseLog(
                        amount=amount, category=category, description=encrypted_desc,
                        timestamp=datetime.now(timezone.utc), agent_source=agent_name
                    )
                    session.add(log_entry)
                    # Commit happens automatically
            logger.info(f"Logged expense from {agent_name}: ${amount:.6f} [{category}]")
        except Exception as e:
            logger.error(f"Failed to log expense for {agent_name}: {e}", exc_info=True)
            # Optionally report this failure as another error
            # await self.report_error("Orchestrator", f"Failed to log expense from {agent_name}: {e}")

    # --- User Education Trigger ---
    async def handle_user_education_trigger(self, topic: str, context: Optional[str] = None):
        """Handles triggers for user education via ThinkTool."""
        # ### Phase 2 Plan Ref: 5.14 (Implement handle_user_education_trigger)
        logger.info(f"User education trigger received for topic: {topic}")
        await self.delegate_task("ThinkTool", {
            "action": "generate_educational_content",
            "topic": topic,
            "context": context,
            "description": f"Generate educational content on '{topic}'"
        })
        # Notification is handled by ThinkTool after generation

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
            base_url = self.config.AGENCY_BASE_URL.rstrip('/')
            audio_url = f"{base_url}/hosted_audio/{safe_filename}"
            logger.info(f"Hosted temporary audio at: {audio_url}")
            # Schedule cleanup task
            async def cleanup_audio(path):
                await asyncio.sleep(300) # Keep for 5 minutes
                try: os.remove(path); logger.debug(f"Cleaned up temp audio: {path}")
                except Exception as e: logger.warning(f"Failed to cleanup temp audio {path}: {e}")
            asyncio.create_task(cleanup_audio(filepath))
            return audio_url
        except Exception as e:
            logger.error(f"Failed to host temporary audio {filename}: {e}", exc_info=True)
            return None

    # --- Route Setup ---
    def setup_routes(self):
        """Configure Quart routes for UI interaction and webhooks."""
        # ### Phase 2 Plan Ref: 5.15 (Implement setup_routes)
        @self.app.route('/')
        async def index():
            try:
                template_path = os.path.join(os.path.dirname(__file__), '..', 'ui', 'templates', 'index.html')
                if os.path.exists(template_path):
                     with open(template_path, 'r') as f: return f.read(), 200, {'Content-Type': 'text/html'}
                else: return "UI Template not found.", 404
            except Exception as e: logger.error(f"Error serving index.html: {e}", exc_info=True); return "Internal Server Error", 500

        @self.app.route('/api/approve', methods=['POST'])
        async def approve():
            if self.approved: return jsonify({"status": "already_approved"}), 200
            self.approved = True
            logger.info("!!! AGENCY APPROVED FOR FULL OPERATION via API !!!")
            await self.send_notification("Agency Approved", "The agency has been approved for full operation via API.")
            return jsonify({"status": "approved"})

        @self.app.route('/api/status', methods=['GET'])
        async def api_status():
            agent_statuses = {name: agent.get_status_summary() for name, agent in self.agents.items() if hasattr(agent, 'get_status_summary')}
            llm_status = {key: info['status'] for key, info in self._llm_client_status.items()}
            # Fetch budget summary if BudgetAgent logic is integrated
            budget_summary = {}
            # if hasattr(self, 'get_budget_summary'): # Check if method exists after potential consolidation
            #     budget_summary = await self.get_budget_summary() # Needs implementation if BudgetAgent removed

            return jsonify({
                "orchestrator_status": "running" if self.running else "stopped",
                "approved_for_operation": self.approved,
                "agent_statuses": agent_statuses,
                "llm_client_status": llm_status,
                # "budget_summary": budget_summary # Add if available
            })

        @self.app.route('/api/start_ugc', methods=['POST'])
        async def handle_start_ugc():
            # Delegate UGC planning to ThinkTool
            try:
                data = await request.get_json()
                if not data: return jsonify({"error": "Invalid JSON payload"}), 400
                client_industry = data.get('client_industry')
                if not client_industry: return jsonify({"error": "Missing 'client_industry'"}), 400
                task = {
                    "action": "plan_ugc_workflow",
                    "client_industry": client_industry,
                    "num_videos": int(data.get('num_videos', 1)),
                    "initial_script": data.get('script'),
                    "target_services": data.get('target_services'),
                    "description": f"Plan UGC workflow for {client_industry}"
                }
                await self.delegate_task("ThinkTool", task)
                return jsonify({"status": "UGC workflow planning initiated via ThinkTool"}), 202
            except Exception as e: logger.error(f"Failed to initiate UGC workflow planning: {e}", exc_info=True); return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route('/hosted_audio/<path:filename>')
        async def serve_hosted_audio(filename):
            if '..' in filename or filename.startswith('/'): return "Forbidden", 403
            try:
                safe_path = os.path.join(self.temp_audio_dir, filename)
                if os.path.exists(safe_path) and os.path.isfile(safe_path):
                     mimetype = 'audio/wav' if filename.lower().endswith('.wav') else 'audio/mpeg'
                     return await send_file(safe_path, mimetype=mimetype)
                else: return jsonify({"error": "File not found"}), 404
            except Exception as e: logger.error(f"Error serving hosted audio {filename}: {e}"); return jsonify({"error": "Internal server error"}), 500

        # Add Twilio WebSocket endpoint for audio streaming
        @self.app.websocket('/twilio_call')
        async def handle_twilio_websocket():
            call_sid = None
            deepgram_ws = None
            try:
                while True:
                    message = await websocket.receive()
                    data = json.loads(message)
                    event = data.get('event')

                    if event == 'connected': logger.info("Twilio WS: Connected event received.")
                    elif event == 'start':
                        call_sid = data.get('start', {}).get('callSid')
                        logger.info(f"Twilio WS: Start event received for call SID: {call_sid}")
                        # Retrieve the corresponding Deepgram connection
                        deepgram_ws = await self.get_deepgram_connection(call_sid)
                        if not deepgram_ws: logger.error(f"Twilio WS: No Deepgram connection found for call SID {call_sid}. Cannot forward audio.") ; break
                        else: logger.info(f"Twilio WS: Found Deepgram connection for {call_sid}. Ready to stream.")
                    elif event == 'media':
                        if not deepgram_ws or deepgram_ws.closed: logger.warning(f"Twilio WS: Received media for {call_sid}, but Deepgram WS is missing or closed."); continue
                        payload = data.get('media', {}).get('payload')
                        if payload:
                            # Decode base64 audio and send to Deepgram
                            audio_bytes = base64.b64decode(payload)
                            await deepgram_ws.send(audio_bytes)
                    elif event == 'stop':
                        logger.info(f"Twilio WS: Stop event received for call SID: {call_sid}. Closing connections.")
                        break # Exit loop on stop event
                    elif event == 'error': # Added error handling
                         error_msg = data.get('error', {}).get('message', 'Unknown Twilio WS error')
                         logger.error(f"Twilio WS: Error event received for call SID {call_sid}: {error_msg}")
                         break # Exit on error
            except websockets.exceptions.ConnectionClosedOK: logger.info(f"Twilio WS: Connection closed normally for {call_sid}.")
            except websockets.exceptions.ConnectionClosedError as e: logger.error(f"Twilio WS: Connection closed with error for {call_sid}: {e}")
            except asyncio.CancelledError: logger.info(f"Twilio WS: Task cancelled for {call_sid}.")
            except Exception as e: logger.error(f"Twilio WS: Unexpected error for call SID {call_sid}: {e}", exc_info=True)
            finally:
                logger.info(f"Twilio WS: Cleaning up for call SID: {call_sid}")
                if deepgram_ws and not deepgram_ws.closed:
                    try: await deepgram_ws.close(); logger.debug(f"Twilio WS: Closed associated Deepgram WS for {call_sid}.")
                    except Exception: pass
                if call_sid: await self.unregister_deepgram_connection(call_sid) # Ensure unregister

        # Add Tracking Pixel endpoint
        @self.app.route('/track/<tracking_id>.png')
        async def handle_tracking_pixel(tracking_id):
            # ### Phase 2 Plan Ref: 5.16 (Implement tracking pixel endpoint)
            logger.info(f"Tracking pixel hit: {tracking_id}")
            # Trigger EmailAgent processing in the background
            email_agent = self.agents.get('email')
            if email_agent and hasattr(email_agent, 'process_email_open'):
                asyncio.create_task(email_agent.process_email_open(tracking_id))
            else: logger.warning(f"EmailAgent not found or cannot process opens for tracking ID: {tracking_id}")
            # Return a 1x1 transparent pixel
            pixel_data = base64.b64decode('R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==')
            return pixel_data, 200, {'Content-Type': 'image/gif', 'Cache-Control': 'no-cache, no-store, must-revalidate', 'Pragma': 'no-cache', 'Expires': '0'}

        logger.info("Quart routes configured.")


    # --- Main Execution Loop ---
    async def run(self):
        """Initializes and runs the AI Agency Orchestrator."""
        # ### Phase 2 Plan Ref: 5.17 (Implement run loop)
        logger.info("Orchestrator starting full initialization sequence...")
        self.running = False

        try:
            # --- Initialization Sequence ---
            if not await self.initialize_database(): raise RuntimeError("Database initialization failed.")
            # Removed migration call - assume done or handled separately
            if not await self.initialize_clients(): raise RuntimeError("LLM Client initialization failed.")
            if not await self.initialize_agents(): raise RuntimeError("Agent initialization failed.")

            logger.info("Orchestrator initialization complete.")

            # --- Start Background Tasks ---
            logger.info("Starting background agent run loops and periodic tasks...")
            self.background_tasks = set()
            # Start CORE agent run loops
            for agent_name, agent in self.agents.items():
                 if hasattr(agent, 'run') and callable(agent.run):
                      task = asyncio.create_task(agent.run(), name=f"AgentLoop_{agent_name}")
                      self.background_tasks.add(task)
                 else: logger.warning(f"Agent {agent_name} does not have a callable run method.")

            # Start orchestrator periodic tasks
            self.background_tasks.add(asyncio.create_task(self._run_periodic_data_purge(), name="PeriodicDataPurge"))
            self.background_tasks.add(asyncio.create_task(self._run_periodic_feedback_collection(), name="PeriodicFeedback"))

            logger.info(f"Started {len(self.background_tasks)} background tasks.")

            # --- Main Operational Loop ---
            logger.info("Orchestrator entering main operational state (API/Event driven).")
            self.running = True
            self.last_feedback_time = time.time()
            self.last_purge_time = time.time()

            # Keep loop alive to manage background tasks
            while self.running:
                try:
                    # Check health of background tasks
                    tasks_to_remove = set()
                    for task in self.background_tasks:
                        if task.done():
                            tasks_to_remove.add(task)
                            try: task.result(); logger.info(f"Background task {task.get_name()} completed.")
                            except asyncio.CancelledError: logger.info(f"Background task {task.get_name()} was cancelled.")
                            except Exception as task_exc: logger.error(f"Background task {task.get_name()} failed: {task_exc}", exc_info=True)
                    self.background_tasks -= tasks_to_remove

                    await asyncio.sleep(60) # Check interval

                except asyncio.CancelledError: logger.info("Orchestrator main loop cancelled."); break

            logger.info("Orchestrator main loop finished.")

        except asyncio.CancelledError: logger.info("Orchestrator run task cancelled during setup.")
        except Exception as e:
            logger.critical(f"CRITICAL ERROR during Orchestrator setup or main loop: {e}", exc_info=True)
            self.running = False
            try: await self.send_notification("CRITICAL Orchestrator Failure", f"Orchestrator failed: {e}")
            except Exception as report_err: logger.error(f"Failed to send critical failure report: {report_err}")
        finally:
            logger.info("Orchestrator shutdown sequence initiated.")
            self.running = False
            # Gracefully cancel background tasks
            cancelled_tasks = []
            for task in self.background_tasks:
                if task and not task.done(): task.cancel(); cancelled_tasks.append(task)
            if cancelled_tasks:
                 logger.info(f"Waiting for {len(cancelled_tasks)} background tasks to cancel...")
                 await asyncio.gather(*cancelled_tasks, return_exceptions=True)
                 logger.info("Background tasks cancellation complete.")
            else: logger.info("No active background tasks needed cancellation.")
            logger.info("Orchestrator shutdown complete.")

# --- End of agents/orchestrator.py ---