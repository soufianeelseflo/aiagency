import os
import asyncio
from datetime import datetime, timedelta
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from quart import Quart, request, jsonify, websocket, send_file, url_for # Architect-Zero: Added websocket, send_file, url_for
import psutil
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from openai import AsyncOpenAI as AsyncDeepSeekClient
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from prometheus_client import Counter, Gauge, start_http_server
from stable_baselines3 import PPO
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import deque
import spacy
import json
import pybreaker
import websockets # Architect-Zero: Added import
import base64    # Architect-Zero: Added import
import uuid      # Architect-Zero: Added import
from typing import Dict, Optional # Architect-Zero: Added import
from quart.wrappers.request import Websocket # Architect-Zero: Added import
from utils.think import think_step, think_controller
# --- Database & Migration Imports ---
from sqlalchemy import select, update, text
from datetime import datetime, timezone
from utils.database import encrypt_data, decrypt_data_fixed_salt_migration # Import both encryption methods
from models import Base, Client, Metric, ExpenseLog, MigrationStatus # Import necessary models

# Register basic checklists
think_controller.register("initialize_clients_pre", lambda args, kwargs: True)
think_controller.register("initialize_clients_post", lambda result: True)
think_controller.register("initialize_database_pre", lambda args, kwargs: True)
think_controller.register("initialize_database_post", lambda result: True)
think_controller.register("run_encryption_migration_pre", lambda args, kwargs: True) # New checklist for migration
think_controller.register("run_encryption_migration_post", lambda result: True)
think_controller.register("initialize_agents_pre", lambda args, kwargs: True)
think_controller.register("initialize_agents_post", lambda result: True)
think_controller.register("run_pre", lambda args, kwargs: True)
think_controller.register("run_post", lambda result: True)

from config.settings import settings
from utils.secure_storage import SecureStorage, VaultError # Import VaultError
# encrypt_data, decrypt_data_fixed_salt_migration imported above
from twilio.rest import Client as TwilioClient
# Base, Client, Metric, ExpenseLog, MigrationStatus imported above
from agents.browsing_agent import BrowsingAgent
from agents.email_agent import EmailAgent
from agents.legal_compliance_agent import LegalComplianceAgent
from agents.osint_agent import OSINTAgent
from agents.scoring_agent import ScoringAgent
from agents.think_tool import ThinkTool
from agents.voice_sales_agent import VoiceSalesAgent
from agents.optimization_agent import OptimizationAgent
from utils.notifications import send_notification


            
email_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=300)
agent_status = Gauge('agent_status', 'Status of agents', ['agent_name'])
error_count = Counter('agent_errors', 'Number of errors per agent', ['agent_name'])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("agency.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def create_session_maker(engine, schema='public'):
    async def session_factory():
        session = AsyncSession(engine)
        await session.execute(f"SET search_path TO {schema}")
        return session
    return session_factory

class Orchestrator:
    """Central coordinator of the AI agency, managing agents and operations autonomously."""

    # --- Methods related to direct API key storage in Account table (Marked for Removal/Rework) ---
    # async def initialize_primary_api_key(self):
    #     # This method stores OPENROUTER_API_KEY directly in DB, conflicts with Vault strategy.
    #     # Needs removal or rework based on how multiple keys/accounts are managed via Vault.
    #     logger.warning("DEPRECATED: initialize_primary_api_key relies on direct DB key storage.")
    #     pass # Commented out implementation

    # async def get_available_openrouter_clients(self):
    #     # This method reads api_key directly from DB, conflicts with Vault strategy.
    #     # Needs rework to fetch keys from Vault based on account info (e.g., vault_path).
    #     logger.warning("DEPRECATED: get_available_openrouter_clients relies on direct DB key storage.")
    #     # Placeholder: Return only the primary client initialized via Vault for now
    #     if self.openrouter_client:
    #         return [self.openrouter_client]
    #     else:
    #         logger.error("Primary OpenRouter client not initialized. Cannot provide clients.")
    #         return []

    # async def reset_api_key_availability(self):
    #     # This method operates on the Account table's is_available flag, related to direct key management.
    #     # May need rework depending on how account rotation/availability is handled with Vault.
    #     logger.warning("DEPRECATED: reset_api_key_availability related to direct DB key management.")
    #     pass # Commented out implementation

    async def create_openrouter_accounts(self, num_accounts):
        # This queues tasks for BrowsingAgent, which should handle Vault storage upon creation. OK.
        if 'browsing' not in self.agents:
             logger.error("Browsing agent not initialized. Cannot queue OpenRouter account creation.")
             return
        for _ in range(num_accounts):
            # Ensure the task specifies Vault storage is expected
            await self.agents['browsing'].task_queue.put({
                'action': 'create_account',
                'service_url': 'https://openrouter.ai',
                'store_in_vault': True # Add flag indicating Vault usage
            })
        logger.info(f"Queued {num_accounts} OpenRouter account creation tasks (for Vault storage).")
    # --- End Deprecated/Reworked Methods ---
        
    def __init__(self, schema='public'):
        self.config = settings
        self.engine = create_async_engine(self.config.DATABASE_URL, echo=True)
        self.session_maker = create_session_maker(self.engine, schema)
        self.agents = {}
        self.app = Quart(__name__)
        self.setup_routes()
        start_http_server(8000)
        self.meta_prompt = settings.META_PROMPT
        self.approved = False
        self.performance_history = deque(maxlen=100)
        self.concurrency_model = LinearRegression()
        self.model_trained = False
        self.base_concurrency = 5
        self.rl_model = PPO("MlpPolicy", "MultiInputPolicy", verbose=1)
        self.concurrency_limit = self.base_concurrency
        self.sandbox_initialized = False
        self.openrouter_client = None  # Initialize as None; set in initialize_clients
        self.deepseek_client = None    # Initialize as None; set in initialize_clients
        self.max_retries = 3
        self.secure_storage = SecureStorage()
        self.deepgram_connections: Dict[str, websockets.client.WebSocketClientProtocol] = {} # Architect-Zero: Added registry for Deepgram WS
        self.temp_audio_dir = "/tmp/hosted_audio" # Architect-Zero: Added temp dir for audio hosting
        os.makedirs(self.temp_audio_dir, exist_ok=True) # Architect-Zero: Ensure temp dir exists
        # Validate essential non-secret env vars (DB URL, Vault config are checked in settings.py)
        essential_config = [
            "DATABASE_URL", "DATABASE_ENCRYPTION_KEY", # Checked in settings.py
            "HCP_ORGANIZATION_ID", "HCP_PROJECT_ID", "HCP_APP_NAME", "HCP_API_TOKEN", # Checked in settings.py
            "TWILIO_ACCOUNT_SID", # Needed early?
            "USER_EMAIL", # Needed for notifications
            "HOSTINGER_EMAIL", # Needed for notifications/IMAP
        ]
        missing_config = [var for var in essential_config if not self.config.get(var)]
        if missing_config:
            # Log critical error but allow startup to potentially fetch from Vault later if possible?
            # For now, maintain critical failure if essential bootstrap config is missing.
            error_msg = f"CRITICAL: Missing essential configuration settings: {', '.join(missing_config)}. Cannot start."
            logger.critical(error_msg)
            # Cannot send notification yet as components might not be ready
            raise ValueError(error_msg)

    async def check_system_health(self):
        """Check system health for auto-approval."""
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent      
        async with self.session_maker() as session:
            error_count = await session.execute(
                "SELECT COUNT(*) FROM metrics WHERE metric_name = 'error' AND timestamp > :threshold",
                {"threshold": datetime.utcnow() - timedelta(hours=1)}
            )
            recent_errors = error_count.scalar() or 0
        health_ok = cpu_usage < 70 and memory_usage < 70 and recent_errors < 5
        logger.info(f"System health: CPU={cpu_usage}%, Memory={memory_usage}%, Errors={recent_errors}, Healthy={health_ok}")
        return health_ok
        
    @think_step("initialize_agents")
    async def initialize_agents(self):
        """Initialize all agents, fetching necessary secrets from Vault."""
        logger.info("Fetching secrets required for agent initialization...")
        try:
            # Fetch secrets concurrently
            results = await asyncio.gather(
                self.secure_storage.get_secret('hostinger-smtp-pass'),
                self.secure_storage.get_secret('hostinger-imap-pass'),
                self.secure_storage.get_secret('twilio-auth-token'),
                self.secure_storage.get_secret('deepgram-api-key'),
                self.secure_storage.get_secret('smartproxy-password'),
                # Add other secrets needed by agents here...
                return_exceptions=True # Return exceptions instead of raising immediately
            )
            # Assign fetched secrets, checking for errors
            secret_names = ['hostinger-smtp-pass', 'hostinger-imap-pass', 'twilio-auth-token', 'deepgram-api-key', 'smartproxy-password']
            fetched_secrets = {}
            initialization_failed = False
            for name, result in zip(secret_names, results):
                if isinstance(result, Exception):
                    logger.critical(f"Failed to fetch required secret '{name}' from Vault: {result}. Agent initialization cannot proceed.")
                    initialization_failed = True
                else:
                    fetched_secrets[name] = result
            
            if initialization_failed:
                 raise VaultError("Failed to fetch one or more required secrets for agent initialization.")

            logger.info("Required secrets fetched successfully. Initializing agents...")

            # --- Agent Initialization with Fetched Secrets ---
            # Ensure LLM clients are ready (assuming initialize_clients was called first)
            # TODO: Rework self.openrouter_clients / self.deepseek_clients based on Vault account management
            # For now, assume primary clients exist if initialize_clients succeeded
            llm_clients = []
            if self.openrouter_client: llm_clients.append(self.openrouter_client)
            if self.deepseek_client: llm_clients.append(self.deepseek_client)
            if not llm_clients: logger.warning("No LLM clients available during agent initialization.")

            # ThinkTool (Doesn't seem to need specific secrets directly)
            self.agents['think'] = ThinkTool(
                self.session_maker, self.config, self,
                # Pass appropriate client/model mapping
                clients_models=[(c, self.config.OPENROUTER_MODELS.get('think', self.config.DEEPSEEK_MODEL)) for c in llm_clients]
            )
            # EmailAgent (Needs SMTP/IMAP pass)
            self.agents['email'] = EmailAgent(
                self.session_maker, self.config, self,
                smtp_password=fetched_secrets['hostinger-smtp-pass'],
                imap_password=fetched_secrets['hostinger-imap-pass'],
                clients_models=[(c, self.config.OPENROUTER_MODELS.get('email_draft_llm', self.config.DEEPSEEK_MODEL)) for c in llm_clients]
            )
            # LegalComplianceAgent
            self.agents['legal'] = LegalComplianceAgent(
                self.session_maker, self.config, self,
                clients_models=[(c, self.config.OPENROUTER_MODELS.get('legal_validate', self.config.DEEPSEEK_MODEL)) for c in llm_clients]
            )
            # OSINTAgent
            self.agents['osint'] = OSINTAgent(
                self.session_maker, self.config, self,
                clients_models=[(c, self.config.OPENROUTER_MODELS.get('osint_analyze', self.config.DEEPSEEK_MODEL)) for c in llm_clients]
            )
            # ScoringAgent
            self.agents['scoring'] = ScoringAgent(
                self.session_maker, self.config, self,
                clients_models=[(c, self.config.OPENROUTER_MODELS.get('scoring', self.config.DEEPSEEK_MODEL)) for c in llm_clients] # Assuming a 'scoring' model key exists
            )
            # VoiceSalesAgent (Needs Twilio Auth, Deepgram Key)
            self.agents['voice_sales'] = VoiceSalesAgent(
                self.session_maker, self.config, self,
                twilio_auth_token=fetched_secrets['twilio-auth-token'],
                deepgram_api_key=fetched_secrets['deepgram-api-key'],
                clients_models=[(c, self.config.OPENROUTER_MODELS.get('voice_response_llm', self.config.DEEPSEEK_MODEL)) for c in llm_clients]
            )
            # OptimizationAgent
            self.agents['optimization'] = OptimizationAgent(
                self.session_maker, self.config, self,
                clients_models=[(c, self.config.OPENROUTER_MODELS.get('optimization', self.config.DEEPSEEK_MODEL)) for c in llm_clients] # Assuming 'optimization' model key
            )
            # BrowsingAgent (Needs Smartproxy Pass)
            self.agents['browsing'] = BrowsingAgent(
                self.session_maker, self.config, self,
                smartproxy_password=fetched_secrets['smartproxy-password'],
                # Pass appropriate client/model mapping, maybe specific browsing models
                clients_models=[(c, self.config.OPENROUTER_MODELS.get('browsing_infer_steps', self.config.DEEPSEEK_MODEL)) for c in llm_clients]
            )
            logger.info("All agents initialized successfully.")
        # Removed duplicate except block below
        # except Exception as e:
        #     logger.error(f"Agent initialization failed: {e}")
        #     await self.send_notification("Agent Initialization Failed", str(e))
        #     raise
        # Removed duplicate logger.info below
        except VaultError as ve:
            logger.critical(f"CRITICAL: Failed to fetch secrets during agent initialization: {ve}. Cannot proceed.")
            # Optionally send notification if possible at this stage
            raise # Halt startup
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}", exc_info=True)
            # await self.send_notification("Agent Initialization Failed", str(e)) # May fail if notification depends on failed secrets
            raise

    @think_step("initialize_clients")
    async def initialize_clients(self):
        """Initialize primary LLM API clients by fetching keys from Vault."""
        logger.info("Initializing primary LLM clients...")
        try:
            # Fetch keys from Vault
            openrouter_api_key = await self.secure_storage.get_secret('openrouter-api-key')
            deepseek_api_key = await self.secure_storage.get_secret('deepseek-api-key')
            # gemini_api_key = await self.secure_storage.get_secret('gemini-api-key') # Fetch if needed directly

            if not openrouter_api_key:
                raise VaultError("Required secret 'openrouter-api-key' not found in Vault.")
            self.openrouter_client = AsyncOpenAI(
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            logger.info("OpenRouter client initialized successfully.")

            if not deepseek_api_key:
                raise VaultError("Required secret 'deepseek-api-key' not found in Vault.")
            self.deepseek_client = AsyncDeepSeekClient(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com"
            )
            logger.info("DeepSeek client initialized successfully.")

            # Initialize other clients like Gemini if needed directly

        except VaultError as ve:
            error_msg = f"Failed to initialize API clients due to Vault error: {ve}"
            logger.critical(error_msg)
            # Cannot send notification reliably if basic clients fail
            raise # Halt startup
        except Exception as e:
            error_msg = f"Unexpected error initializing API clients: {e}"
            logger.critical(error_msg, exc_info=True)
            raise # Halt startup

    @think_step("initialize_database")
    async def initialize_database(self):
        """Initialize or update the PostgreSQL database schema."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database schema initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    @think_step("run_encryption_migration")
    async def _run_encryption_migration_v2(self):
        """Checks and performs data migration for encryption v2 (per-value salt)."""
        migration_name = 'encryption_v2'
        logger.info(f"Checking status for migration: {migration_name}")
        async with self.session_maker() as session:
            try:
                # Check if migration is already marked as complete
                stmt_check = select(MigrationStatus).where(MigrationStatus.migration_name == migration_name)
                result_check = await session.execute(stmt_check)
                status_record = result_check.scalar_one_or_none()

                if status_record and status_record.completed_at:
                    logger.info(f"Migration '{migration_name}' already completed at {status_record.completed_at}. Skipping.")
                    return True # Indicate success/completion

                logger.info(f"Migration '{migration_name}' not found or not completed. Starting migration process...")

                # --- Migration Logic ---
                total_processed = 0
                total_updated = 0
                total_failed_decryption = 0
                batch_size = 100
                offset = 0

                while True:
                    logger.info(f"Processing migration batch starting from offset {offset}...")
                    stmt_select = select(ExpenseLog.id, ExpenseLog.description).order_by(ExpenseLog.id).offset(offset).limit(batch_size)
                    result_select = await session.execute(stmt_select)
                    records = result_select.fetchall()

                    if not records:
                        logger.info("No more records found for migration.")
                        break

                    updates_to_commit = []
                    for record_id, old_encrypted_desc in records:
                        total_processed += 1
                        if not old_encrypted_desc: continue # Skip null descriptions

                        # Decrypt using OLD fixed-salt logic
                        decrypted_desc = decrypt_data_fixed_salt_migration(old_encrypted_desc)

                        if decrypted_desc is None:
                            logger.error(f"[Migration] Failed to decrypt description for ExpenseLog ID {record_id}. Skipping update.")
                            total_failed_decryption += 1
                            continue

                        # Re-encrypt using NEW per-value salt logic
                        try:
                            new_encrypted_desc = encrypt_data(decrypted_desc)
                            if new_encrypted_desc is None: raise ValueError("New encryption returned None")
                        except Exception as enc_err:
                            logger.error(f"[Migration] Failed during NEW encryption for ExpenseLog ID {record_id}: {enc_err}. Skipping update.")
                            continue

                        if new_encrypted_desc != old_encrypted_desc:
                            updates_to_commit.append({'id': record_id, 'description': new_encrypted_desc})
                            total_updated += 1

                    # Perform batch update
                    if updates_to_commit:
                        try:
                            update_stmt = update(ExpenseLog).where(ExpenseLog.id == text(':id')).values(description=text(':description'))
                            await session.execute(update_stmt, updates_to_commit)
                            logger.info(f"Migration batch update successful for {len(updates_to_commit)} records (offset {offset}).")
                        except Exception as batch_err:
                            logger.error(f"[Migration] Failed to commit batch update (offset {offset}): {batch_err}")
                            await session.rollback()
                            logger.critical(f"CRITICAL: Migration '{migration_name}' failed during batch update. Halting application startup.")
                            raise Exception(f"Migration '{migration_name}' failed: {batch_err}") from batch_err

                    offset += len(records)

                # --- Mark Migration as Complete ---
                completion_time = datetime.now(timezone.utc)
                if status_record: # Update existing record
                    status_record.completed_at = completion_time
                else: # Insert new record
                    new_status = MigrationStatus(migration_name=migration_name, completed_at=completion_time)
                    session.add(new_status)

                await session.commit() # Commit updates and the status flag
                logger.info(f"--- Migration '{migration_name}' Summary ---")
                logger.info(f"Total records processed: {total_processed}")
                logger.info(f"Total records successfully updated: {total_updated}")
                logger.info(f"Total records failed decryption (skipped): {total_failed_decryption}")
                logger.info(f"Migration marked as complete at {completion_time}.")
                logger.info("-------------------------------------")
                if total_failed_decryption > 0:
                     logger.warning("Some records failed decryption and were not migrated. Review logs.")
                return True # Indicate success

            except Exception as e:
                logger.error(f"CRITICAL: Error during migration '{migration_name}' check or execution: {e}", exc_info=True)
                await session.rollback() # Ensure rollback on any error
                raise Exception(f"Migration '{migration_name}' failed: {e}") from e # Halt startup

    # --- Removed Duplicate Method Definitions ---
    # The following methods were duplicated due to a previous partial diff application
    # and have been removed:
    # - initialize_primary_api_key (also deprecated)
    # - get_available_openrouter_clients (also deprecated)
    # - reset_api_key_availability (also deprecated)
    # - create_openrouter_accounts
    # - initialize_agents
    # --- End Removed Duplicates ---

    async def collect_agent_feedback(self):
        nlp = spacy.load("en_core_web_sm")
        while True:
            try:
                async with self.session_maker() as session:
                    feedback_data = {}
                    for name, agent in self.agents.items():
                        if hasattr(agent, 'get_insights'):
                            insights = await agent.get_insights()
                            if insights:
                                feedback_data[name] = insights
                                metric = Metric(
                                    agent_name=name,
                                    timestamp=datetime.utcnow(),
                                    metric_name="insights",
                                    value=json.dumps(insights)
                                )
                                session.add(metric)
                    await session.commit()
                    
                    actionable_insights = {}
                    for agent_name, insights in feedback_data.items():
                        doc = nlp(json.dumps(insights))
                        actions = [sent.text for sent in doc.sents if "improve" in sent.text.lower() or "optimize" in sent.text.lower()]
                        actionable_insights[agent_name] = actions
                    
                    if actionable_insights:
                        for agent_name, actions in actionable_insights.items():
                            if actions and agent_name in self.agents and hasattr(self.agents[agent_name], 'apply_insights'):
                                await self.agents[agent_name].apply_insights({"actions": actions})
                        await self.send_notification(
                            "Feedback Processed",
                            f"Actionable insights distributed to {len(actionable_insights)} agents: {json.dumps(actionable_insights, indent=2)}"
                        )
                await asyncio.sleep(3600)  # Hourly
            except Exception as e:
                logger.error(f"Feedback collection failed: {e}")
                await asyncio.sleep(60)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=5, max=60))
    @email_breaker
    async def send_notification(self, subject, body):
        """Sends email notification, fetching password via Vault."""
        try:
            # Fetch SMTP password from Vault
            smtp_pass = await self.secure_storage.get_secret('hostinger-smtp-pass')
            if not smtp_pass:
                raise VaultError("SMTP password ('hostinger-smtp-pass') not found in Vault.")
            
            # Call the utility function, passing the fetched password
            # Assuming send_notification utility function signature is updated to accept smtp_password
            await send_notification(subject, body, self.config, smtp_password=smtp_pass)
        except VaultError as ve:
             logger.error(f"Failed to send notification '{subject}' due to Vault error: {ve}")
             # Decide if this should re-raise or just log
        except Exception as e:
             # Catch errors from send_notification itself (already logged there)
             logger.error(f"Error occurred during send_notification call: {e}")

    async def send_whatsapp_notification(self, message):
        try:
            twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER")
            user_whatsapp_number = os.getenv("USER_WHATSAPP_NUMBER")
            if not all([twilio_account_sid, twilio_auth_token, twilio_whatsapp_number, user_whatsapp_number]):
                raise ValueError("Twilio or WhatsApp configuration missing")
            client = TwilioClient(twilio_account_sid, twilio_auth_token)
            client.messages.create(
                body=message,
                from_=f"whatsapp:{twilio_whatsapp_number}",
                to=f"whatsapp:{user_whatsapp_number}"
            )
            logger.info(f"WhatsApp notification sent: {message}")
        except Exception as e:
            logger.error(f"Failed to send WhatsApp notification: {e}")

    async def report_error(self, agent_name, error):
        """Report errors to the user via email."""
        subject = f"Error in {agent_name}"
        body = f"Error details: {error}"
        await self.send_notification(subject, body)

    async def start_testing_phase(self):
        try:
            async with self.session_maker() as session:
                test_campaign = Campaign(name="Test UGC Campaign", status="testing")
                session.add(test_campaign)
                await session.commit()
                test_campaign_id = test_campaign.id
                logger.info(f"Test campaign created: ID {test_campaign_id}")

            video_task = asyncio.create_task(self.agents['browsing'].generate_test_videos(["tech", "finance", "healthcare"]))
            call_task = asyncio.create_task(self.agents['voice_sales'].simulate_call("test_client"))
            results = await asyncio.gather(video_task, call_task, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Testing task failed: {result}")
                    await self.report_error("TestingTask", str(result))
            logger.info("Testing phase completed: sample outputs generated.")
            await self.send_whatsapp_notification("Testing Phase Completed", "Sample outputs have been generated and are ready for review.")
        except Exception as e:
            logger.error(f"Testing phase failed: {e}")
            await self.report_error("TestingPhase", str(e))
            raise


    # --- UGC Workflow Management ---

    async def start_ugc_workflow(self, client_industry: str, num_videos: int = 1, script: str = None, target_services: list = None):
        """Initiates the UGC content generation workflow."""
        await self.send_notification("UGC Workflow Started", f"Starting UGC workflow for {client_industry} ({num_videos} videos).")
        logger.info(f"Initiating UGC workflow for {client_industry} ({num_videos} videos).")

        if not target_services:
            target_services = ['heygen.com', 'descript.com'] # Default services

        # Generate a unique ID for this workflow instance
        workflow_id = str(uuid.uuid4())
        logger.info(f"UGC Workflow ID: {workflow_id}")

        # Initial state for the workflow
        workflow_state = {
            'workflow_id': workflow_id,
            'client_industry': client_industry,
            'num_videos': num_videos,
            'target_services': target_services,
            'current_step': 'account_acquisition',
            'required_accounts': {service: False for service in target_services}, # Track if we have accounts
            'script': script or f"Default UGC script for {client_industry}.", # Generate/fetch script later
            'generated_video_path': None,
            'edited_video_path': None,
        }

        # TODO: Persist workflow_state somewhere (DB?) if needed for long-running workflows or recovery

        # Queue the first step: Check/Acquire accounts for all target services
        # We can queue multiple account creation tasks if needed.
        # For simplicity, let's assume BrowsingAgent checks internally or we queue one by one via report_ugc_step_complete.
        # Let's start by trying to get credentials for the first service. If unavailable, BrowsingAgent should ideally queue creation.
        # A better approach might be a dedicated 'acquire_resources' task.
        # For now, let's queue a task that implies checking/getting accounts.

        first_service = target_services[0]
        task = {
            'action': 'acquire_or_verify_account', # New action type for BrowsingAgent
            'service_name': first_service,
            'workflow_id': workflow_id,
            'workflow_state': workflow_state # Pass state for context
        }
        if 'browsing' in self.agents:
            await self.agents['browsing'].task_queue.put(task)
            logger.info(f"Queued initial UGC task for workflow {workflow_id}: {task['action']} for {first_service}")
        else:
            logger.error(f"Browsing agent not found. Cannot start UGC workflow {workflow_id}.")
            await self.send_notification("UGC Workflow Error", f"Browsing agent not found for workflow {workflow_id}.")


    async def report_ugc_step_complete(self, workflow_id: str, completed_step: str, result: dict, current_state: dict):
        """Handles completion of a UGC step and queues the next one."""
        logger.info(f"Received completion report for UGC workflow {workflow_id}, step: {completed_step}. Result: {result}")

        # TODO: Load/Update persistent workflow_state if used

        next_task = None
        updated_state = current_state.copy() # Work with a copy

        try:
            if completed_step == 'acquire_or_verify_account':
                service_name = result.get('service_name')
                success = result.get('success', False)
                if success:
                    updated_state['required_accounts'][service_name] = True
                    logger.info(f"Account verified/acquired for {service_name} in workflow {workflow_id}.")
                else:
                    logger.error(f"Failed to acquire/verify account for {service_name} in workflow {workflow_id}. Workflow stalled.")
                    await self.send_notification("UGC Workflow Error", f"Account acquisition failed for {service_name} in workflow {workflow_id}.")
                    # TODO: Implement retry logic or failure handling
                    return # Stop processing this workflow for now

                # Check if all required accounts are ready
                all_accounts_ready = all(updated_state['required_accounts'].values())
                if all_accounts_ready:
                    logger.info(f"All required accounts ready for workflow {workflow_id}. Proceeding to video generation.")
                    updated_state['current_step'] = 'video_generation'
                    # Queue video generation task (e.g., using Heygen)
                    video_gen_service = next((s for s in updated_state['target_services'] if s in ['heygen.com', 'argil.ai']), None)
                    if video_gen_service:
                        next_task = {
                            'action': 'generate_ugc_video', # Use the action BrowsingAgent expects
                            'target_service': video_gen_service,
                            'script': updated_state['script'],
                            'avatar_prefs': {}, # Add preferences later
                            'workflow_id': workflow_id,
                            'workflow_state': updated_state
                        }
                    else:
                         logger.error(f"No suitable video generation service found in target_services for workflow {workflow_id}.")
                         # Handle error - maybe try editing first?
                else:
                    # Find the next service that needs an account
                    next_service_to_acquire = next((s for s, ready in updated_state['required_accounts'].items() if not ready), None)
                    if next_service_to_acquire:
                        logger.info(f"Queueing next account acquisition for {next_service_to_acquire} in workflow {workflow_id}.")
                        next_task = {
                            'action': 'acquire_or_verify_account',
                            'service_name': next_service_to_acquire,
                            'workflow_id': workflow_id,
                            'workflow_state': updated_state
                        }
                    else:
                        # Should not happen if all_accounts_ready was false, but log just in case
                        logger.error(f"Inconsistent state: Not all accounts ready, but no next service found for workflow {workflow_id}.")


            elif completed_step == 'generate_ugc_video':
                # Result should contain path/URL to the generated video
                generated_video = result.get('generated_video_path_or_url')
                if generated_video:
                    updated_state['generated_video_path'] = generated_video
                    logger.info(f"Video generated for workflow {workflow_id} at {generated_video}. Proceeding to editing.")
                    updated_state['current_step'] = 'video_editing'
                    # Queue video editing task (e.g., using Descript)
                    editing_service = next((s for s in updated_state['target_services'] if s == 'descript.com'), None)
                    if editing_service:
                        next_task = {
                            'action': 'edit_ugc_video', # New action type for BrowsingAgent
                            'target_service': editing_service,
                            'source_video_path': generated_video, # Pass the generated video path
                            'assets': {}, # Add B-roll etc. later
                            'workflow_id': workflow_id,
                            'workflow_state': updated_state
                        }
                    else:
                        logger.warning(f"No editing service (Descript) found in target_services for workflow {workflow_id}. Skipping editing.")
                        # If no editing, consider the workflow complete or move to final step
                        updated_state['current_step'] = 'completed'
                        logger.info(f"UGC Workflow {workflow_id} completed (no editing step). Final video: {generated_video}")
                        await self.send_notification("UGC Workflow Complete", f"Workflow {workflow_id} for {updated_state['client_industry']} finished. Video: {generated_video}")

                else:
                    logger.error(f"Video generation step failed for workflow {workflow_id}. Result: {result}")
                    await self.send_notification("UGC Workflow Error", f"Video generation failed for workflow {workflow_id}.")
                    # Handle error

            elif completed_step == 'edit_ugc_video':
                 # Result should contain path/URL to the edited video
                edited_video = result.get('edited_video_path_or_url')
                if edited_video:
                    updated_state['edited_video_path'] = edited_video
                    logger.info(f"Video editing completed for workflow {workflow_id} at {edited_video}.")
                    updated_state['current_step'] = 'completed'
                    # TODO: Add final storage/organization step if needed
                    logger.info(f"UGC Workflow {workflow_id} completed. Final video: {edited_video}")
                    await self.send_notification("UGC Workflow Complete", f"Workflow {workflow_id} for {updated_state['client_industry']} finished. Video: {edited_video}")
                else:
                    logger.error(f"Video editing step failed for workflow {workflow_id}. Result: {result}")
                    await self.send_notification("UGC Workflow Error", f"Video editing failed for workflow {workflow_id}.")
                    # Handle error

            else:
                logger.warning(f"Received completion report for unknown UGC step '{completed_step}' in workflow {workflow_id}.")

            # Queue the next task if one was determined
            if next_task:
                if 'browsing' in self.agents:
                    # Update the state in the task being queued
                    next_task['workflow_state'] = updated_state
                    await self.agents['browsing'].task_queue.put(next_task)
                    logger.info(f"Queued next UGC task for workflow {workflow_id}: {next_task.get('action')} for {next_task.get('target_service') or next_task.get('service_name')}")
                else:
                    logger.error(f"Browsing agent not found. Cannot queue next UGC task for workflow {workflow_id}.")
                    await self.send_notification("UGC Workflow Error", f"Browsing agent not found for workflow {workflow_id} step {updated_state['current_step']}.")

        except Exception as e:
            logger.error(f"Error processing UGC step completion for workflow {workflow_id}: {e}", exc_info=True)
            await self.send_notification("UGC Workflow Error", f"Internal error processing step {completed_step} for workflow {workflow_id}: {e}")


    # --- End UGC Workflow Management ---


    # --- End Voice Agent Support Methods ---

    # --- Route Setup ---
    def setup_routes(self):
        """Stores the active Deepgram WebSocket connection for a given call SID."""
        logger.info(f"Registering Deepgram WS connection for call_sid: {call_sid}")
        self.deepgram_connections[call_sid] = ws_connection

    async def unregister_deepgram_connection(self, call_sid: str):
        """Removes the Deepgram WebSocket connection for a given call SID."""
        logger.info(f"Unregistering Deepgram WS connection for call_sid: {call_sid}")
        self.deepgram_connections.pop(call_sid, None) # Remove safely

    async def get_deepgram_connection(self, call_sid: str) -> Optional[websockets.client.WebSocketClientProtocol]:
        """Retrieves the active Deepgram WebSocket connection for a given call SID."""
        return self.deepgram_connections.get(call_sid)

    async def host_temporary_audio(self, audio_data: bytes, filename: str) -> Optional[str]:
        """Saves audio data locally and returns a URL accessible by Twilio (requires proper web server setup)."""
        try:
            # Ensure filename is safe
            safe_filename = re.sub(r'[^\w\.-]', '_', filename)
            filepath = os.path.join(self.temp_audio_dir, safe_filename)
            with open(filepath, 'wb') as f:
                f.write(audio_data)
            # Generate a URL that the Quart app can serve
            # This assumes the '/hosted_audio/<filename>' route is set up below
            audio_url = url_for('serve_hosted_audio', filename=safe_filename, _external=True)
            logger.info(f"Hosted temporary audio at: {audio_url}")
            return audio_url
        except Exception as e:
            logger.error(f"Failed to host temporary audio {filename}: {e}", exc_info=True)
            return None

        """Configure Quart routes for UI interaction."""
        @self.app.route('/start_testing', methods=['POST'])
        async def start_testing():
            try:
                await self.start_testing_phase()
                return jsonify({"status": "testing started"})
            except Exception as e:
                logger.error(f"Start testing failed: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route('/approve', methods=['POST'])
        async def approve():
            try:
                self.approved = True
                logger.info("Agency approved for full operation.")
                await self.send_notification("Agency Approved", "The agency has been approved for full operation.")
                return jsonify({"status": "approved"})
            except Exception as e:
                logger.error(f"Approval failed: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route('/status', methods=['GET'])
        async def status():
            try:
                return jsonify({"approved": self.approved, "concurrency_limit": self.concurrency_limit})
            except Exception as e:
                logger.error(f"Status check failed: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route('/suggest', methods=['POST'])
        async def suggest():
            try:
                data = await request.get_json()
                suggestion = data.get('suggestion')
                if not suggestion:
                    return jsonify({"error": "No suggestion provided"}), 400
                validation = await self.agents['think'].evaluate_suggestion(suggestion)
                if validation['approved']:
                    await self.agents['think'].implement_suggestion(suggestion)
                    logger.info(f"User suggestion applied: {suggestion}")
                    return jsonify({"status": "suggestion accepted"})
                logger.warning(f"Suggestion rejected: {suggestion}")
                return jsonify({"status": "suggestion rejected", "reason": validation['reason']})
            except Exception as e:
                logger.error(f"Suggestion processing failed: {e}")
                return jsonify({"error": str(e)}), 500

        # New route for starting UGC workflow
        @self.app.route('/start_ugc', methods=['POST'])
        async def handle_start_ugc():
            try:
                data = await request.get_json()
                client_industry = data.get('client_industry')
                num_videos = data.get('num_videos', 1)
                script = data.get('script') # Optional script override
                target_services = data.get('target_services') # Optional list like ['heygen.com', 'descript.com']

                if not client_industry:
                    return jsonify({"error": "Missing 'client_industry' parameter"}), 400

                # Use asyncio.create_task to run the workflow in the background
                # so the API request returns quickly.
                asyncio.create_task(self.start_ugc_workflow(
                    client_industry=client_industry,
                    num_videos=num_videos,
                    script=script,
                    target_services=target_services
                ))

                return jsonify({"status": "UGC workflow initiated", "client_industry": client_industry}), 202 # Accepted

            except Exception as e:
                logger.error(f"Failed to initiate UGC workflow via API: {e}", exc_info=True)
                return jsonify({"status": "error", "message": str(e)}), 500

        # Route for serving temporary hosted audio (if needed by Voice Agent)
        @self.app.route('/hosted_audio/<filename>')
        async def serve_hosted_audio(filename):
            try:
                # Security: Ensure filename is safe and doesn't allow directory traversal
                safe_filename = re.sub(r'[^\w\.-]', '_', filename)
                filepath = os.path.join(self.temp_audio_dir, safe_filename)
                if os.path.exists(filepath):
                    return await send_file(filepath, mimetype='audio/mpeg') # Adjust mimetype if needed
                else:
                    return jsonify({"error": "File not found"}), 404
            except Exception as e:
                logger.error(f"Error serving hosted audio {filename}: {e}")
                return jsonify({"error": "Internal server error"}), 500
    # --- End Route Setup ---


    @think_step("run")
    async def run(self):
        try:
            await self.initialize_clients()
            await self.initialize_database() # Ensures tables exist, including MigrationStatus
            await self._run_encryption_migration_v2() # Run migration after DB init, before agents
            # await self.initialize_primary_api_key() # Removed - Relied on direct key storage
            await self.initialize_agents() # Agents initialized after migration & client init
            
            # Create initial OpenRouter accounts immediately (via BrowsingAgent -> Vault)
            await self.create_openrouter_accounts(5)
            
            await self.start_testing_phase()
            self.approved = True # Assuming testing phase implies approval for now
            logger.info("Testing phase completed successfully. Agency approved for full operation.")
            
            boost_task = asyncio.create_task(self.adjust_concurrency())
            monitor_task = asyncio.create_task(self.monitor_agents())
            sandbox_task = asyncio.create_task(self.manage_sandbox())
            cleanup_task = asyncio.create_task(self.cleanup_old_logs())
            reset_task = asyncio.create_task(self.reset_api_key_availability())
            feedback_task = asyncio.create_task(self.run_feedback_loop()) # Create feedback loop task
            agent_tasks = [asyncio.create_task(agent.run()) for agent in self.agents.values()]
            # Add feedback_task to gather
            await asyncio.gather(*agent_tasks, boost_task, monitor_task, sandbox_task, cleanup_task, reset_task, feedback_task)
        except Exception as e:
            logger.error(f"Agency run failed: {e}")
            await self.report_error("Orchestrator", str(e))
            raise

    async def run_feedback_loop(self):
        """Runs the periodic feedback collection and application loop."""
        logger.info("Starting periodic feedback loop (daily).")
        while True:
            try:
                # Wait initially before the first check
                await asyncio.sleep(86400)  # Daily cycle

                feedback_data = {}
                # Use self.agents directly if accessible
                for agent_name, agent in self.agents.items():
                    # Check if agent exists and has the method
                    if agent and hasattr(agent, 'collect_insights') and callable(agent.collect_insights):
                        try:
                            feedback_data[agent_name] = await agent.collect_insights()
                        except Exception as insight_err:
                            logger.error(f"Error collecting insights from {agent_name}: {insight_err}")
                    # else: logger.debug(f"Agent {agent_name} missing or lacks collect_insights method.") # Optional debug

                if feedback_data:
                    # Ensure think agent exists and has the method
                    think_agent = self.agents.get('think')
                    if think_agent and hasattr(think_agent, 'process_feedback') and callable(think_agent.process_feedback):
                        try:
                            validated_feedback = await think_agent.process_feedback(feedback_data)
                            applied_count = 0
                            for agent_name, feedback in validated_feedback.items():
                                target_agent = self.agents.get(agent_name)
                                # Check if target agent exists and has the method
                                if target_agent and feedback and hasattr(target_agent, 'apply_insights') and callable(target_agent.apply_insights):
                                    try:
                                        await target_agent.apply_insights(feedback)
                                        applied_count += 1
                                    except Exception as apply_err:
                                        logger.error(f"Error applying insights to {agent_name}: {apply_err}")
                            logger.info(f"Feedback distributed to {applied_count} agents.")
                            if hasattr(self, 'send_notification'): # Check if send_notification exists
                                await self.send_notification(
                                    "Feedback Cycle Complete",
                                    f"Distributed feedback to {applied_count} agents."
                                )
                        except Exception as process_err:
                            logger.error(f"Error processing feedback with ThinkTool: {process_err}")
                    else:
                        logger.warning("ThinkTool agent or process_feedback method not available for feedback loop.")
                else:
                    logger.info("No feedback data collected in this cycle.")

            except asyncio.CancelledError:
                 logger.info("Feedback loop cancelled.")
                 break # Exit loop if cancelled
            except Exception as e:
                logger.error(f"Feedback loop failed: {e}")
                if hasattr(self, 'report_error'): # Check if report_error exists
                    await self.report_error("FeedbackLoop", str(e))
                # Avoid rapid retries on persistent errors
                await asyncio.sleep(3600) # Wait an hour before retrying after a failure

if __name__ == "__main__":
    orchestrator = Orchestrator()
    asyncio.run(orchestrator.run())
