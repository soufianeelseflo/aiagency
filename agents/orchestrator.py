# Filename: agents/orchestrator.py
# Description: Central coordinator for the AI Agency, managing agents, workflows, resources, and UI backend integration.
# Version: 4.2 (Docker Path Handling & Route Setup Refinement)

import os
import asyncio
from datetime import datetime, timedelta, timezone
import logging
import time
import json
import uuid
import random
import re
import base64
import hashlib # Added for caching
import glob # Added for list_files tool
import sys # Added for path manipulation if needed
from collections import deque
from typing import Dict, Optional, Tuple, Any, List, AsyncGenerator, Callable, Type, Union

# --- Core Framework Imports ---
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker,
)
from sqlalchemy import select, delete, func, update, text, case, or_
from sqlalchemy.exc import SQLAlchemyError
from quart import Quart, request, jsonify, websocket, send_file, url_for, Response, current_app, render_template # Added render_template
import psutil
from openai import AsyncOpenAI as AsyncLLMClient
import pybreaker
import websockets
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
)
import aiohttp # Added for proxy health check

# --- Project Imports ---
from config.settings import settings
from utils.database import (
    encrypt_data, decrypt_data, get_session, get_session_maker,
)
from utils.notifications import send_notification

# Import ALL agent classes
try:
    from .base_agent import GeniusAgentBase
except ImportError:
    logging.warning("Production base agent not found, using GeniusAgentBase.")
    from agents.base_agent import GeniusAgentBase

from agents.think_tool import ThinkTool
from agents.browsing_agent import BrowsingAgent
from agents.email_agent import EmailAgent
from agents.voice_sales_agent import VoiceSalesAgent
from agents.legal_agent import LegalAgent
from agents.social_media_manager import SocialMediaManager
from agents.gmail_creator_agent import GmailCreatorAgent

# Import UI route registration function - will attach routes defined THERE
try:
    # Import the specific handlers needed by setup_routes
    from ui.app import register_ui_routes, index_route_handler, get_status_and_kpi_handler, \
                       approve_agency_handler, export_data_handler, submit_feedback_handler, \
                       test_voice_call_handler, generate_videos_handler
except ImportError:
    logging.critical("Failed to import UI route handlers from ui.app. Check structure.")
    # Define dummy handlers if import fails, though this indicates a problem
    def register_ui_routes(app: Quart): pass
    async def index_route_handler(): return "UI Error: Handler not loaded", 500
    async def get_status_and_kpi_handler(): return jsonify({"error": "Handler not loaded"}), 500
    async def approve_agency_handler(): return jsonify({"error": "Handler not loaded"}), 500
    async def export_data_handler(): return jsonify({"error": "Handler not loaded"}), 500
    async def submit_feedback_handler(): return jsonify({"error": "Handler not loaded"}), 500
    async def test_voice_call_handler(): return jsonify({"error": "Handler not loaded"}), 500
    async def generate_videos_handler(): return jsonify({"error": "Handler not loaded"}), 500


# Import database models
from models import (
    Base, Client, ExpenseLog, MigrationStatus, KnowledgeFragment,
    AccountCredentials, CallLog, Invoice, StrategicDirective, PromptTemplate
)

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
op_logger = logging.getLogger("OperationalLog")

# --- Circuit Breakers ---
llm_client_breaker = pybreaker.CircuitBreaker(
    fail_max=5, reset_timeout=60 * 3, name="LLMClientBreaker_L40"
)
proxy_breaker = pybreaker.CircuitBreaker(
    fail_max=7, reset_timeout=60 * 10, name="ProxyProviderBreaker_L40"
)

# --- Global Shutdown Event ---
shutdown_event = asyncio.Event()


# --- Orchestrator Class ---
class Orchestrator:
    """Central coordinator (L40+), managing agents, workflows, resources, and UI backend."""

    # Inside the Orchestrator class:
    def __init__(self, schema="public"):
        self.config = settings
        self.session_maker = get_session_maker()
        self.agents: Dict[str, GeniusAgentBase] = {}

        # Determine project root for template/static paths relative to THIS file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ui_dir = os.path.join(project_root, "ui")

        # Create the Quart app instance HERE, configuring paths correctly
        self.app = Quart(
            __name__,
            template_folder=os.path.join(ui_dir, "templates"),
            static_folder=os.path.join(ui_dir, "static")
        )
        # Ensure routes are set up AFTER the app is created
        self.setup_routes()

        self.meta_prompt = self.config.META_PROMPT
        self.approved = False

        # LLM Client Management State (No changes needed here for deployment)
        self._llm_client: Optional[AsyncLLMClient] = None
        self._llm_client_status: Dict[str, Any] = {"status": "unavailable", "reason": "Not initialized", "unavailable_until": 0}

        # In-memory Cache (No changes needed here for deployment)
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_ttl_default = 3600

        # Deepgram WebSocket Registry (No changes needed here for deployment)
        self.deepgram_connections: Dict[str, Any] = {}

        # Use absolute path from settings for TEMP_AUDIO_DIR, ensure it exists
        self.temp_audio_dir = self.config.TEMP_AUDIO_DIR
        if not os.path.isabs(self.temp_audio_dir):
            logger.warning(f"TEMP_AUDIO_DIR '{self.temp_audio_dir}' in settings is not absolute. Defaulting to '/app/temp_audio'. Ensure this is correct in settings.")
            self.temp_audio_dir = "/app/temp_audio" # Default safe path inside container
        try:
            # Create the directory if it doesn't exist. Dockerfile also does this, but good practice.
            os.makedirs(self.temp_audio_dir, exist_ok=True)
            logger.info(f"Temporary audio directory set to: {self.temp_audio_dir}")
        except OSError as e:
            logger.error(f"Failed to create temporary audio directory {self.temp_audio_dir}: {e}. Audio hosting may fail.")
            # Decide if this is critical enough to halt initialization? For now, log error.
            # raise RuntimeError(f"Failed to create essential directory: {self.temp_audio_dir}") from e

        # Periodic Task Timing (No changes needed here for deployment)
        self.feedback_interval_seconds: int = int(self.config.THINKTOOL_FEEDBACK_INTERVAL_SECONDS)
        self.purge_interval_seconds: int = int(self.config.DATA_PURGE_INTERVAL_SECONDS)
        self.proxy_health_check_interval_seconds: int = int(self.config.get("PROXY_HEALTH_CHECK_INTERVAL_S", 3600 * 2))
        self.last_feedback_time: float = 0.0
        self.last_purge_time: float = 0.0
        self.last_proxy_health_check_time: float = 0.0

        self.running: bool = False
        self.background_tasks = set()
        self.status = "initializing"

        # Secure Storage Interface (No changes needed here for deployment)
        self.secure_storage = self._DatabaseSecureStorage(self.session_maker)

        # Proxy Management State (No changes needed here for deployment, uses config)
        self._proxy_list: List[Dict[str, Any]] = []
        self._proxy_lock = asyncio.Lock()
        self._load_initial_proxies() # Loads from config

        logger.info("Orchestrator v4.2 (Docker Path Handling) initialized.")

    # --- Initialization Methods ---
    # Inside the Orchestrator class:
    async def initialize_database(self):
        """Initialize or update the database schema."""
        logger.info("Initializing database schema...")
        try:
            db_url_str = str(self.config.DATABASE_URL) if self.config.DATABASE_URL else None
            if not db_url_str: raise ValueError("DATABASE_URL is not configured.")
            # Retry connection briefly on startup
            for attempt in range(3): # Try up to 3 times
                try:
                    engine = create_async_engine(db_url_str, echo=False, pool_pre_ping=True, pool_recycle=3600)
                    # Test connection before creating tables
                    async with engine.connect() as conn_test:
                        # Execute a simple query common to PostgreSQL to verify connection
                        await conn_test.execute(text("SELECT 1"))
                        logger.info(f"Database connection successful on attempt {attempt+1}.")
                    # If connection test succeeds, proceed to create tables
                    async with engine.begin() as conn:
                        await conn.run_sync(Base.metadata.create_all)
                    await engine.dispose() # Dispose engine after use
                    logger.info("Database schema initialization complete.")
                    return True # Success
                except Exception as db_conn_err:
                    if attempt < 2: # Check if more retries left
                        logger.warning(f"Database connection attempt {attempt+1} failed: {db_conn_err}. Retrying in 5s...")
                        await asyncio.sleep(5)
                    else:
                        # Log critical error after last attempt fails and re-raise
                        logger.critical(f"Database connection failed after multiple attempts: {db_conn_err}", exc_info=True)
                        raise # Reraise the final error to signal critical failure
            return False # Should not be reached if raise occurs
        except Exception as e:
            # Catch errors from the overall process (e.g., getting DB URL)
            logger.critical(f"Failed to initialize database: {e}", exc_info=True)
            return False

    async def initialize_clients(self):
        """Initialize the primary LLM API client."""
        logger.info("Initializing primary LLM client...")
        self._llm_client = None
        self._llm_client_status = {"status": "unavailable", "reason": "Initialization failed", "unavailable_until": 0}

        primary_key = self.config.get_secret("OPENROUTER_API_KEY")
        if primary_key:
            try:
                # Use OpenAI SDK compatibility
                self._llm_client = AsyncLLMClient(api_key=primary_key, base_url="https://openrouter.ai/api/v1")
                # Simple test call to verify connectivity and key during init
                await self._llm_client.models.list(timeout=15) # Short timeout for init check
                self._llm_client_status = {"status": "available", "reason": None, "unavailable_until": 0}
                logger.info(f"Successfully initialized and tested OpenRouter client (Primary Key: ...{primary_key[-4:]}).")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize or test primary OpenRouter client: {e}", exc_info=True)
                self._llm_client_status["reason"] = f"Initialization/Test Error: {e}"
                return False
        else:
            logger.critical("CRITICAL: OPENROUTER_API_KEY not set. LLM functionality disabled.")
            self._llm_client_status["reason"] = "API Key Missing"
            return False

    # Inside the Orchestrator class:
    async def initialize_agents(self):
        """Initialize the CORE agents, passing necessary secrets."""
        logger.info("Initializing CORE agents...")
        initialization_failed = False
        # Define Agent Classes to load dynamically
        agent_classes = {
            "ThinkTool": ThinkTool, "BrowseAgent": BrowseAgent, "EmailAgent": EmailAgent,
            "VoiceSalesAgent": VoiceSalesAgent, "LegalAgent": LegalAgent,
            "SocialMediaManager": SocialMediaManager, "GmailCreatorAgent": GmailCreatorAgent
        }
        # Define constructor parameters for agents requiring specific secrets/config
        agent_params = {
            "EmailAgent": {"imap_password": self.config.get_secret("HOSTINGER_IMAP_PASS")},
            "VoiceSalesAgent": {"twilio_auth_token": self.config.get_secret("TWILIO_AUTH_TOKEN"), "deepgram_api_key": self.config.get_secret("DEEPGRAM_API_KEY")},
            "GmailCreatorAgent": {"config": self.config}, # Pass config if agent needs specific settings
            # BrowseAgent gets proxy via Orchestrator.get_proxy if needed
        }
        # Define which secrets are CRITICAL for each agent's basic function
        required_secrets = {
            "EmailAgent": ["HOSTINGER_IMAP_PASS"],
            "VoiceSalesAgent": ["TWILIO_AUTH_TOKEN", "DEEPGRAM_API_KEY"]
            # Add other agents here if they have absolutely critical secrets for basic init
        }

        for name, AgentClass in agent_classes.items():
            try:
                params = agent_params.get(name, {})
                # Check required secrets for this agent BEFORE initialization
                missing_secrets = []
                for secret_key in required_secrets.get(name, []):
                    # Check if the key exists in params and has a truthy value
                    if not params.get(secret_key.lower()): # Secrets are passed as lowercase kwargs
                        missing_secrets.append(secret_key)
                if missing_secrets:
                    # Log critical error but allow orchestrator to continue starting other agents
                    logger.critical(f"Cannot initialize {name}: Missing required secrets {missing_secrets}. Agent will be unavailable.")
                    initialization_failed = True # Mark overall failure but continue loop
                    continue # Skip initializing this specific agent

                # Instantiate the agent if secrets are present (or none required)
                # Use .lower() for the key in self.agents for consistent access
                self.agents[name.lower()] = AgentClass(session_maker=self.session_maker, orchestrator=self, **params)
                logger.info(f"Initialized {name}.")
            except ImportError as e:
                # Handle cases where an agent file might be missing
                logger.error(f"Failed to import or initialize {name}: {e}. Skipping.")
                initialization_failed = True # Treat import error as failure
            except Exception as e:
                logger.error(f"Unexpected error initializing {name}: {e}", exc_info=True)
                initialization_failed = True

        # ProgrammerAgent (Optional, check file existence)
        programmer_agent_path = os.path.join(os.path.dirname(__file__), 'programmer_agent.py')
        if os.path.exists(programmer_agent_path):
            try:
                from agents.programmer_agent import ProgrammerAgent
                self.agents["programmer"] = ProgrammerAgent(orchestrator=self, session_maker=self.session_maker)
                logger.info("Initialized ProgrammerAgent.")
            except ImportError as e: logger.warning(f"Found programmer_agent.py but failed to import/init: {e}")
            except Exception as e: logger.error(f"Unexpected error initializing ProgrammerAgent: {e}", exc_info=True)
        else: logger.info("ProgrammerAgent file not found, skipping initialization.")

        # Check if any critical initialization failed
        if initialization_failed:
            # Log critical error but don't raise exception here to allow app to start
            # User can check logs in Coolify to see which agents failed
            logger.critical("One or more agents failed to initialize properly due to missing secrets or errors. Check logs above. Functionality may be impaired.")

        logger.info(f"Core agents initialized: {list(self.agents.keys())}")
        # Return True even if some agents failed, as Orchestrator itself might still run
        # The critical log indicates the problem to the user.
        return True


    @llm_client_breaker
    async def get_available_llm_client(self) -> Optional[AsyncLLMClient]:
        """Returns the primary LLM client if available and not in cooldown."""
        now = time.time()
        status_info = self._llm_client_status
        if status_info["status"] == "available" or now >= status_info.get("unavailable_until", 0):
            if status_info["status"] == "unavailable":
                # Attempt re-initialization if client object is None (e.g., first check after failure)
                if self._llm_client is None:
                    logger.info("Attempting to re-initialize LLM client after cooldown/failure...")
                    if not await self.initialize_clients():
                        # Re-initialization failed, keep status as unavailable but update timestamp
                        status_info["unavailable_until"] = time.time() + 60 * 15 # Wait longer after failed re-init
                        logger.error("Failed to re-initialize LLM client. Extending cooldown.")
                        return None
                # If client is available (or re-initialized successfully), reset status
                status_info["status"] = "available"
                status_info["reason"] = None
                status_info["unavailable_until"] = 0
                logger.info("Primary LLM client available.")
            # Ensure client is not None before returning
            return self._llm_client if self._llm_client else None
        else:
            logger.warning(f"Primary LLM client unavailable. Reason: {status_info.get('reason', 'Unknown')}. Cooldown until {datetime.fromtimestamp(status_info.get('unavailable_until', 0)).isoformat()}")
            return None

    async def report_client_issue(self, issue_type: str):
        """Reports an issue with the primary LLM client, triggering cooldown."""
        now = time.time(); cooldown_seconds = 60 * 5; status = "unavailable"; reason = issue_type
        # Differentiate cooldown based on severity
        if issue_type == "auth_error": cooldown_seconds = 60 * 60 * 24 * 365; reason = "Authentication Error"; logger.critical("Primary LLM client marked permanently unavailable (auth error).")
        elif issue_type == "rate_limit": cooldown_seconds = 60 * 2; reason = "Rate Limited"
        elif issue_type == "timeout_error": cooldown_seconds = 60 * 3; reason = "Timeout"
        else: reason = "General Error" # Shorter cooldown for generic errors

        # If it's an auth error, also clear the client object
        if issue_type == "auth_error":
            self._llm_client = None

        self._llm_client_status = {"status": status, "reason": reason, "unavailable_until": now + cooldown_seconds}
        logger.warning(f"Primary LLM client marked unavailable until {datetime.fromtimestamp(now + cooldown_seconds).isoformat()}. Reason: {reason}")

    @llm_client_breaker
    async def call_llm(
        self, agent_name: str, prompt: str, temperature: float = 0.5, max_tokens: int = 1024,
        is_json_output: bool = False, model_preference: Optional[List[str]] = None,
        image_data: Optional[bytes] = None, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Handles making the LLM call using the primary client via OpenAI SDK compatibility."""
        llm_client = await self.get_available_llm_client()
        if not llm_client:
            logger.error(f"Agent '{agent_name}' failed: Primary LLM client unavailable.")
            return None

        # Model Selection Logic
        # Define a default mapping or get from settings
        default_model_map = settings.OPENROUTER_MODELS
        fallback_model = default_model_map["default_llm"]

        # Determine base model key by agent
        agent_key_lower = agent_name.lower()
        if agent_key_lower == "thinktool": model_key = "think_synthesize"
        elif agent_key_lower == "emailagent": model_key = "email_draft"
        elif agent_key_lower == "voicesalesagent": model_key = "voice_response"
        elif agent_key_lower == "legalagent": model_key = "legal_analysis"
        elif agent_key_lower == "Browseagent": model_key = "Browse_visual_analysis" if image_data else "Browse_summarize"
        elif agent_key_lower == "socialmediamanager": model_key = "email_draft" # Reuse creative model
        else: model_key = "default_llm" # Default for other agents

        # Task-specific overrides based on prompt keywords
        task_specific_model_key = None
        prompt_lower = prompt.lower()
        if "synthesize" in prompt_lower or "strategize" in prompt_lower: task_specific_model_key = "think_strategize"
        elif "critique" in prompt_lower: task_specific_model_key = "think_critique"
        elif "radar" in prompt_lower or "scouting" in prompt_lower: task_specific_model_key = "think_radar"
        elif "validate" in prompt_lower: task_specific_model_key = "think_validate"
        elif "educational content" in prompt_lower: task_specific_model_key = "think_user_education"
        elif "humanize" in prompt_lower: task_specific_model_key = "email_humanize"
        elif "intent" in prompt_lower: task_specific_model_key = "voice_intent"
        elif "visual analysis" in prompt_lower and image_data: task_specific_model_key = "Browse_visual_analysis"

        final_model_key = task_specific_model_key or model_key
        # Get model name from settings map, use fallback if key doesn't exist
        model_name = default_model_map.get(final_model_key, fallback_model)
        logger.debug(f"Selected model '{model_name}' for agent '{agent_name}' task (key: {final_model_key}).")

        # Caching Logic
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        cache_key_parts = ["llm_call", prompt_hash, model_name, str(temperature), str(max_tokens), str(is_json_output), str(image_data is not None)]
        cache_key = ":".join(cache_key_parts); cache_ttl = self.cache_ttl_default
        cached_result = self.get_from_cache(cache_key)
        if cached_result is not None:
            logger.debug(f"LLM call cache hit (Orchestrator) for key: {cache_key[:20]}..."); return cached_result

        try:
            # Prepare messages for OpenAI SDK format
            messages = []
            content_parts = [{"type": "text", "text": prompt}]
            if image_data:
                base64_image = base64.b64encode(image_data).decode('utf-8')
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
                logger.debug("Image data included in LLM request.")
            messages.append({"role": "user", "content": content_parts})

            # Prepare request parameters
            request_params = {
                "model": model_name, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "timeout": timeout or settings.OPENROUTER_API_TIMEOUT_S or 120.0,
            }
            if is_json_output:
                request_params["response_format"] = {"type": "json_object"}

            # Add optional OpenRouter headers for tracking
            extra_headers = {
                "HTTP-Referer": str(settings.AGENCY_BASE_URL),
                "X-Title": settings.APP_NAME,
            }
            request_params["extra_headers"] = extra_headers

            logger.debug(f"Orchestrator making LLM Call: Agent={agent_name}, Model={model_name}, Multimodal={image_data is not None}")

            # Make the API call
            response = await llm_client.chat.completions.create(**request_params)

            content = response.choices[0].message.content.strip() if response.choices and response.choices[0].message and response.choices[0].message.content else None

            # Track usage & cost
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = input_tokens + output_tokens
            # Placeholder costs - Ideally fetch from a config or service
            cost_per_million_input = 0.50
            cost_per_million_output = 1.50
            cost = (input_tokens / 1_000_000 * cost_per_million_input + output_tokens / 1_000_000 * cost_per_million_output)
            logger.debug(f"LLM Call Usage: Model={model_name}, Tokens={total_tokens}, Est. Cost=${cost:.6f}")
            await self.report_expense(agent_name, cost, "LLM", f"LLM call ({model_name}). Tokens: {total_tokens}.")

            result_dict = {"content": content, "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": total_tokens, "estimated_cost": cost}}
            self.add_to_cache(cache_key, result_dict, ttl_seconds=cache_ttl)
            return result_dict

        except Exception as e:
            error_str = str(e).lower(); issue_type = "llm_error"
            if "rate limit" in error_str or "quota" in error_str: issue_type = "rate_limit"
            elif "authentication" in error_str: issue_type = "auth_error"
            elif "timeout" in error_str: issue_type = "timeout_error"
            logger.warning(f"Orchestrator LLM call failed: Agent={agent_name}, Model={model_name}, Type={issue_type}, Error={e}")
            await self.report_client_issue(issue_type)
            raise # Re-raise the exception

    # Inside the Orchestrator class:
    def setup_routes(self):
        """Configure Quart routes for UI interaction and webhooks."""

        # --- UI Routes (Using imported handlers) ---
        @self.app.route('/')
        async def index():
            # Use render_template for safer HTML serving
            try:
                # Ensure the template path set during app creation is correct
                return await render_template('index.html')
            except Exception as e:
                logger.error(f"Error rendering index.html: {e}", exc_info=True)
                # Provide a simple error response if template fails
                return "<html><body><h1>Internal Server Error</h1><p>Could not render dashboard template.</p></body></html>", 500

        # Attach imported handlers from ui.app to the app instance
        self.app.route('/api/status_kpi', methods=['GET'])(get_status_and_kpi_handler)
        self.app.route('/api/approve_agency', methods=['POST'])(approve_agency_handler)
        self.app.route('/api/export_data', methods=['POST'])(export_data_handler)
        self.app.route('/api/submit_feedback', methods=['POST'])(submit_feedback_handler)
        self.app.route('/api/test_voice_call', methods=['POST'])(test_voice_call_handler)
        self.app.route('/api/generate_videos', methods=['POST'])(generate_videos_handler)

        # --- Webhook Routes ---
        @self.app.websocket("/twilio_call")
        async def handle_twilio_websocket():
            # ...(No changes needed for this specific websocket handler for deployment)
            # Ensure VoiceSalesAgent correctly registers/unregisters connections via Orchestrator methods
            call_sid = None; deepgram_live_client = None
            try:
                while True:
                    message = await websocket.receive(); data = json.loads(message); event = data.get("event")
                    if event == "start":
                        call_sid = data.get("start", {}).get("callSid"); logger.info(f"Twilio WS: Start event for {call_sid}")
                        deepgram_live_client = await self.get_deepgram_connection_sdk(call_sid)
                        if not deepgram_live_client: logger.error(f"Twilio WS: No Deepgram SDK client found for {call_sid}."); break
                        else: logger.info(f"Twilio WS: Found Deepgram SDK client for {call_sid}.")
                    elif event == "media":
                        if not deepgram_live_client: continue
                        payload = data.get("media", {}).get("payload")
                        if payload:
                            audio_bytes = base64.b64decode(payload)
                            if hasattr(deepgram_live_client, 'send') and callable(deepgram_live_client.send): await deepgram_live_client.send(audio_bytes)
                            else: logger.warning(f"Deepgram client for {call_sid} lacks send method.")
                    elif event == "stop": logger.info(f"Twilio WS: Stop event for {call_sid}."); break
                    elif event == "error": logger.error(f"Twilio WS: Error event for {call_sid}: {data.get('error')}"); break
            except websockets.exceptions.ConnectionClosedOK: logger.info(f"Twilio WS: Connection closed normally for {call_sid}.")
            except Exception as e: logger.error(f"Twilio WS: Unexpected error for {call_sid}: {e}", exc_info=True)
            finally:
                logger.info(f"Twilio WS: Cleaning up for {call_sid}");
                if call_sid: await self.unregister_deepgram_connection_sdk(call_sid)


        @self.app.route("/track/<tracking_id>.png")
        async def handle_tracking_pixel(tracking_id):
            # ...(No changes needed for this specific handler for deployment)
            logger.info(f"Tracking pixel hit: {tracking_id}")
            email_agent = self.agents.get("email");
            if email_agent and hasattr(email_agent, "process_email_open"): asyncio.create_task(email_agent.process_email_open(tracking_id))
            else: logger.warning(f"EmailAgent not found or cannot process opens for tracking ID: {tracking_id}")
            pixel_data = base64.b64decode("R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==")
            return Response(pixel_data, mimetype="image/gif", headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"})


        @self.app.route("/webhooks/mailersend", methods=["POST"])
        async def handle_mailersend_webhook():
            # ...(No changes needed for this specific handler for deployment)
            try:
                payload = await request.get_json()
                if not payload or 'type' not in payload or 'data' not in payload: logger.warning("Received invalid/incomplete MailerSend webhook payload."); return jsonify({"status": "error", "message": "Invalid payload"}), 400
                event_type = payload.get('type', '').lower(); event_data = payload.get('data', {}); email_data = event_data.get('email', {})
                recipient_email = email_data.get('recipient', {}).get('email'); message_id = email_data.get('message', {}).get('id')
                logger.info(f"Received MailerSend Webhook: Type='{event_type}', Recipient='{recipient_email}', MsgID='{message_id}'")
                if not recipient_email: return jsonify({"status": "ok", "message": "Missing recipient"}), 200
                update_values = {}; log_op_level = 'info'; log_op_message = f"Webhook '{event_type}' received for {recipient_email}."
                thinktool_tags = ["webhook", "mailersend", event_type]
                if event_type in ['activity.hard_bounced', 'activity.spam_complaint']: update_values['is_deliverable'] = False; update_values['opt_in'] = False; log_op_level = 'warning'; log_op_message = f"Marking {recipient_email} undeliverable/opt-out due to MailerSend webhook: {event_type}."; thinktool_tags.append("deliverability_issue")
                elif event_type == 'activity.unsubscribed': update_values['opt_in'] = False; log_op_level = 'warning'; log_op_message = f"Marking {recipient_email} opted-out due to MailerSend webhook: {event_type}."; thinktool_tags.append("opt_out")
                if self.agents.get("think"): asyncio.create_task(self.agents["think"].log_knowledge_fragment(agent_source="MailerSendWebhook", data_type="mailersend_event", content=payload, tags=thinktool_tags, relevance_score=0.6, source_reference=f"Recipient:{recipient_email}_MsgID:{message_id}"))
                if update_values:
                    async def update_client_status():
                        session_maker = get_session_maker();
                        if not session_maker: return
                        try:
                            async with session_maker() as session:
                                async with session.begin(): await session.execute(update(Client).where(Client.email == recipient_email).values(**update_values))
                                logger.info(f"Updated client status for {recipient_email} based on webhook '{event_type}'.")
                        except Exception as db_err: logger.error(f"DB Error updating client status from webhook for {recipient_email}: {db_err}", exc_info=True)
                    asyncio.create_task(update_client_status())
                    await self.log_operation(log_op_level, log_op_message)
                return jsonify({"status": "received"}), 200
            except Exception as e: logger.error(f"Error processing MailerSend webhook: {e}", exc_info=True); return jsonify({"status": "error", "message": "Internal server error"}), 500


        @self.app.route("/webhooks/clay/enrichment_results", methods=["POST"]) # More specific endpoint
        async def handle_clay_webhook():
            """Handles incoming webhooks from Clay.com enrichment results."""
            # --- Optional Security Check ---
            expected_secret = self.config.get_secret("CLAY_CALLBACK_SECRET")
            if expected_secret:
                provided_secret = request.headers.get("X-Auth-Token")
                if not provided_secret or provided_secret != expected_secret:
                    logger.warning(f"Unauthorized Clay webhook attempt. Missing or incorrect X-Auth-Token. IP: {request.remote_addr}")
                    return jsonify({"status": "error", "message": "Unauthorized"}), 401
                else:
                    logger.debug("Clay webhook authentication successful.")

            # --- Process Payload ---
            try:
                payload = await request.get_json()
                if not payload:
                    logger.warning("Received empty Clay webhook payload.")
                    return jsonify({"status": "error", "message": "Empty payload"}), 400

                # Clay usually sends one enriched item per webhook call
                enriched_data = payload # Assume payload IS the enriched data
                correlation_id = enriched_data.get("_correlation_id") # Get our tracking ID

                logger.info(f"Received Clay Enrichment Webhook. Correlation ID: {correlation_id}")

                # Delegate processing to ThinkTool in the background
                think_tool = self.agents.get("think") # Use lowercase key
                if think_tool:
                    # Fire-and-forget task delegation
                    asyncio.create_task(think_tool.execute_task({
                        "action": "process_clay_webhook_data", # Action remains same internally for ThinkTool
                        "content": {
                            "enriched_data": enriched_data,
                            "original_input_parameters": {}, # Not typically in webhook
                            "source_reference": f"Clay_Webhook_Corr_{correlation_id}", # Use correlation ID
                            "clay_run_id": None # Not typically in webhook
                        }
                    }))
                    # Acknowledge receipt immediately
                    return jsonify({"status": "received"}), 200
                else:
                    logger.error("ThinkTool not available to process Clay webhook.")
                    return jsonify({"status": "error", "message": "Internal processing error"}), 500
            except Exception as e:
                logger.error(f"Error processing Clay webhook: {e}", exc_info=True)
                return jsonify({"status": "error", "message": "Internal server error"}), 500

        @self.app.route("/hosted_audio/<path:filename>")
        async def serve_hosted_audio(filename):
            # ...(Code from v4.2 - Corrected path handling is now inside __init__ and this method)
            if ".." in filename or filename.startswith("/"): logger.warning(f"Attempted path traversal: {filename}"); return jsonify({"error": "Forbidden"}), 403
            try:
                safe_path = os.path.abspath(os.path.join(self.temp_audio_dir, filename))
                if not safe_path.startswith(os.path.abspath(self.temp_audio_dir)): logger.warning(f"Attempted access outside temp_audio_dir: {filename}"); return jsonify({"error": "Forbidden"}), 403
                if os.path.exists(safe_path) and os.path.isfile(safe_path):
                    mimetype = ("audio/wav" if filename.lower().endswith(".wav") else "audio/mpeg"); logger.debug(f"Serving hosted audio: {filename} from {safe_path} (Mime: {mimetype})")
                    return await send_file(safe_path, mimetype=mimetype)
                else: logger.warning(f"Hosted audio file not found: {filename} (Path: {safe_path})"); return jsonify({"error": "File not found"}), 404
            except Exception as e: logger.error(f"Error serving hosted audio {filename}: {e}", exc_info=True); return jsonify({"error": "Internal server error"}), 500

        logger.info("Orchestrator routes configured.")

    # Inside the Orchestrator class:
    async def run(self):
        """Initializes and runs the AI Agency Orchestrator."""
        logger.info("Orchestrator starting full initialization sequence...")
        self.running = False; self.status = "initializing"
        try:
            # Initialization Steps
            if not await self.initialize_database(): raise RuntimeError("Database initialization failed.")
            if not await self.initialize_clients(): raise RuntimeError("LLM Client initialization failed.")
            if not await self.initialize_agents():
                logger.critical("Agent initialization reported failures. Orchestrator will continue but functionality may be impaired.")
                # Do not raise RuntimeError here, allow limited startup for debugging
                # raise RuntimeError("Agent initialization failed.")

            logger.info("Orchestrator initialization complete (or attempted with failures).")

            # Start Background Tasks (Agent Loops & Periodic Tasks)
            logger.info("Starting background agent run/start loops and periodic tasks...")
            self.background_tasks = set()
            for agent_name, agent in self.agents.items(): # Iterate over successfully initialized agents
                start_method = getattr(agent, 'start', None)
                run_method = getattr(agent, 'run', None)
                # Prefer start() if available (newer base agent pattern), otherwise use run()
                if start_method and callable(start_method):
                    task = asyncio.create_task(start_method(), name=f"AgentStart_{agent_name}")
                    self.background_tasks.add(task)
                    logger.info(f"Called start() for agent {agent_name}")
                elif run_method and callable(run_method):
                    task = asyncio.create_task(run_method(), name=f"AgentRun_{agent_name}")
                    self.background_tasks.add(task)
                    logger.info(f"Called run() for agent {agent_name}")
                else:
                    # This might be normal for agents managed entirely by directives (like LegalAgent?)
                    logger.info(f"Agent {agent_name} does not have a callable start or run method. Assuming event-driven.")

            # Add periodic tasks managed by Orchestrator
            self.background_tasks.add(asyncio.create_task(self._run_periodic_data_purge(), name="PeriodicDataPurge"))
            self.background_tasks.add(asyncio.create_task(self._run_periodic_feedback_collection(), name="PeriodicFeedback"))
            self.background_tasks.add(asyncio.create_task(self._run_periodic_proxy_health_check(), name="PeriodicProxyHealth"))
            logger.info(f"Started {len(self.background_tasks)} background tasks.")

            # Set operational state
            logger.info("Orchestrator entering main operational state (API/Event driven).")
            self.running = True; self.status = "running"
            # Initialize last run times for periodic tasks
            now_ts = time.time()
            self.last_feedback_time = now_ts; self.last_purge_time = now_ts; self.last_proxy_health_check_time = now_ts

            # Main monitoring loop
            while self.running and not shutdown_event.is_set():
                try:
                    # Check status of background tasks and log completion/errors
                    tasks_to_remove = set()
                    for task in list(self.background_tasks): # Iterate over a copy
                        if task.done():
                            tasks_to_remove.add(task)
                            try:
                                task.result() # Calling result() raises exception if task failed
                                logger.info(f"Background task {task.get_name()} completed normally.")
                            except asyncio.CancelledError:
                                logger.info(f"Background task {task.get_name()} was cancelled.")
                            except Exception as task_exc:
                                logger.error(f"Background task {task.get_name()} failed: {task_exc}", exc_info=True)
                                # TODO: Implement policy for restarting critical failed tasks?
                    self.background_tasks -= tasks_to_remove

                    # Main loop sleep
                    await asyncio.sleep(30) # Check task status periodically

                except asyncio.CancelledError:
                    logger.info("Orchestrator main loop cancelled.")
                    break # Exit loop if cancelled
                except Exception as e:
                    logger.critical(f"CRITICAL ERROR in Orchestrator main loop: {e}", exc_info=True)
                    self.running = False; self.status = "error"
                    try: await self.send_notification("CRITICAL Orchestrator Failure", f"Orchestrator failed: {e}")
                    except Exception as report_err: logger.error(f"Failed send critical failure report: {report_err}")
                    # Break loop on critical error
                    break

        except (ValueError, RuntimeError) as init_err:
            logger.critical(f"Fatal Error: Orchestrator initialization failed: {init_err}", exc_info=True)
            self.status = "failed_initialization"
        except Exception as e:
            logger.critical(f"Fatal Error: Unhandled exception during Orchestrator run setup: {e}", exc_info=True)
            self.status = "error"
        finally:
            logger.info("Orchestrator run loop is exiting. Stop sequence should be handled by shutdown event/stop method.")
            # Final status is set either by error or by the stop() method

    # Inside the Orchestrator class:
    async def stop(self, timeout: float = 30.0):
        """Handles graceful shutdown of agents and tasks."""
        if self.status in [self.STATUS_STOPPING, self.STATUS_STOPPED]:
            logger.info(f"Orchestrator stop requested but already {self.status}.")
            return
        self.logger.info(f"Orchestrator stop method called. Timeout: {timeout}s")
        self.status = self.STATUS_STOPPING
        self.running = False # Prevent new operations
        shutdown_event.set() # Signal all loops depending on this event

        # 1. Cancel background tasks managed by orchestrator (periodic tasks)
        tasks_to_await = list(self.background_tasks)
        agent_tasks = set() # Keep track of tasks started BY agents if possible (tricky)

        # Separate orchestrator periodic tasks from potential agent tasks if needed
        # For now, assume background_tasks contains mostly orchestrator's periodic tasks
        if tasks_to_await:
            self.logger.info(f"Cancelling {len(tasks_to_await)} Orchestrator background tasks...")
            for task in tasks_to_await:
                if task and not task.done():
                    task.cancel()
            # Wait for tasks to acknowledge cancellation
            # Use a portion of the timeout for these tasks
            done, pending = await asyncio.wait(tasks_to_await, timeout=timeout*0.3, return_when=asyncio.ALL_COMPLETED)
            if pending:
                self.logger.warning(f"{len(pending)} Orchestrator tasks did not cancel gracefully within timeout.")
            else:
                self.logger.info("Orchestrator background tasks cancellation complete.")

        # 2. Stop individual agents (gives them remaining time)
        agent_stop_tasks = []
        remaining_timeout = timeout * 0.6 # Give agents a good portion of time
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'stop') and callable(agent.stop):
                agent_status = getattr(agent, 'status', 'unknown')
                # Check if agent is running or idle before stopping
                if agent_status not in [getattr(agent, 'STATUS_STOPPING', 'stopping'), getattr(agent, 'STATUS_STOPPED', 'stopped')]:
                    self.logger.info(f"Calling stop() for agent {agent_name}...")
                    agent_stop_tasks.append(asyncio.create_task(agent.stop(timeout=max(5.0, remaining_timeout)), name=f"AgentStop_{agent_name}"))
                else:
                    logger.info(f"Agent {agent_name} already stopping/stopped ({agent_status}). Skipping stop call.")
            else:
                logger.warning(f"Agent {agent_name} has no stop method.")

        if agent_stop_tasks:
            self.logger.info(f"Waiting for {len(agent_stop_tasks)} agents to stop...")
            # Wait for all agent stop tasks to complete
            results = await asyncio.gather(*agent_stop_tasks, return_exceptions=True)
            # Log any errors during agent stop
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    task_name = agent_stop_tasks[i].get_name()
                    logger.error(f"Error stopping agent task '{task_name}': {result}")
            self.logger.info("Agent stop sequence complete.")

        self.status = self.STATUS_STOPPED
        self.logger.info("Orchestrator stop sequence finished.")

    async def delegate_task(self, agent_name: str, task_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Delegates a task to a specific agent, ensuring agent name is lowercased for lookup."""
        agent_name_lower = agent_name.lower() # Normalize name for lookup
        agent = self.agents.get(agent_name_lower)
        if not agent:
            logger.error(f"Delegation failed: Agent '{agent_name}' (normalized: {agent_name_lower}) not found in {list(self.agents.keys())}.")
            return {"status": "error", "message": f"Agent '{agent_name}' not found."}
        if not hasattr(agent, 'execute_task') or not callable(agent.execute_task):
            logger.error(f"Delegation failed: Agent '{agent_name}' has no callable execute_task method.")
            return {"status": "error", "message": f"Agent '{agent_name}' cannot execute tasks."}

        task_id = task_details.get('id', str(uuid.uuid4()))
        task_details['id'] = task_id
        action = task_details.get('action', 'N/A')
        directive_id = task_details.get('directive_id') # Get directive ID if present

        logger.info(f"Delegating task {task_id} (Action: {action}, Directive: {directive_id}) to {agent_name}.")
        try:
            # Add directive ID to content for downstream logging/linking if needed
            if directive_id:
                task_details.setdefault('content', {})['directive_id'] = directive_id

            # Execute the task
            result = await agent.execute_task(task_details)

            # Robustly update directive status based on result
            if directive_id and result and isinstance(result, dict) and 'status' in result:
                final_status = result['status']
                # Map common success/failure statuses
                if final_status == 'success': final_status = 'completed'
                elif final_status == 'error': final_status = 'failed'
                # Preserve specific statuses like 'skipped', 'cancelled', etc.
                elif final_status not in ['completed', 'failed', 'skipped', 'cancelled', 'completed_no_data', 'halted_by_reflection']:
                     final_status = 'completed' # Default other non-error/skip statuses to completed

                result_msg = result.get('message', 'Task finished.')
                await self.update_directive_status(directive_id, final_status, result_msg)

            # Ensure a dictionary is returned
            return result if isinstance(result, dict) else {"status": "error", "message": "Agent returned non-dict result"}

        except Exception as e:
            logger.error(f"Error during task delegation to {agent_name} (Task ID: {task_id}): {e}", exc_info=True)
            await self.report_error(agent_name, f"Task delegation failed: {e}", task_id)
            # Update directive on delegation exception
            if directive_id:
                await self.update_directive_status(directive_id, 'failed', f"Exception during delegation: {e}")
            return {"status": "error", "message": f"Exception during task execution: {e}"}

    async def update_directive_status(self, directive_id: int, status: str, result_summary: Optional[str] = None):
        """Updates the status and result of a StrategicDirective."""
        if not self.session_maker or not directive_id:
             logger.warning(f"Cannot update directive status: session_maker={self.session_maker}, directive_id={directive_id}")
             return
        try:
            async with self.session_maker() as session:
                async with session.begin():
                    # Fetch the current status first to avoid redundant updates (optional)
                    # current_status = await session.scalar(select(StrategicDirective.status).where(StrategicDirective.id == directive_id))
                    # if current_status == status:
                    #     logger.debug(f"Directive {directive_id} already has status '{status}'. Skipping update.")
                    #     return

                    # Update the directive
                    stmt = update(StrategicDirective).where(StrategicDirective.id == directive_id).\
                        values(status=status, result_summary=result_summary)
                    # Optionally update timestamp on status change: values(status=status, result_summary=result_summary, timestamp=datetime.now(timezone.utc))
                    result = await session.execute(stmt)
                    if result.rowcount > 0:
                        self.logger.info(f"Updated directive {directive_id} status to '{status}'.")
                    else:
                        self.logger.warning(f"Attempted to update directive {directive_id} status to '{status}', but directive not found or status unchanged.")
        except Exception as e:
            self.logger.error(f"Failed to update directive {directive_id} status to '{status}': {e}", exc_info=True)

    # Inside the Orchestrator class:
    async def use_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Allows agents to request tool usage via the orchestrator."""
        logger.info(f"Orchestrator received tool use request: {tool_name} with params: {str(params)[:200]}") # Log param preview
        try:
            if tool_name == "read_file":
                file_path = params.get('path')
                # Validate path parameter type
                if not isinstance(file_path, str):
                    return {"status": "failure", "message": f"Invalid path parameter type: {type(file_path)}"}

                # SECURITY: Prevent path traversal and enforce absolute paths
                if not file_path or ".." in file_path or not os.path.isabs(file_path):
                    logger.error(f"Tool 'read_file' denied access to invalid/relative path: {file_path}")
                    return {"status": "failure", "message": "Invalid or non-absolute file path provided."}

                # SECURITY: Check against allowed directories (ensure these are absolute paths)
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                # Resolve paths from config safely, handling None
                learning_dir_abs = os.path.abspath(self.config.LEARNING_MATERIALS_DIR) if self.config.LEARNING_MATERIALS_DIR and os.path.isabs(self.config.LEARNING_MATERIALS_DIR) else None
                temp_audio_abs = os.path.abspath(self.temp_audio_dir) # Already validated in init
                temp_download_abs = os.path.abspath(self.config.TEMP_DOWNLOAD_DIR) if self.config.TEMP_DOWNLOAD_DIR and os.path.isabs(self.config.TEMP_DOWNLOAD_DIR) else None

                allowed_dirs = [d for d in [learning_dir_abs, temp_audio_abs, temp_download_abs, os.path.abspath(project_root)] if d] # Filter out None paths

                abs_requested_path = os.path.abspath(file_path)
                # Check if the requested path starts with any of the allowed directory paths
                if not any(abs_requested_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
                    logger.error(f"Tool 'read_file' access denied outside allowed directories: Request='{file_path}', Absolute='{abs_requested_path}'. Allowed: {allowed_dirs}")
                    return {"status": "failure", "message": "Access denied to specified file path."}

                # Proceed only if path is valid and allowed
                if os.path.exists(abs_requested_path) and os.path.isfile(abs_requested_path):
                    # Read the file content
                    with open(abs_requested_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    return {"status": "success", "content": content}
                else:
                    return {"status": "failure", "message": f"File not found or is not a file: {file_path}"}

            elif tool_name == "list_files":
                dir_path = params.get('path')
                if not isinstance(dir_path, str):
                    return {"status": "failure", "message": f"Invalid path parameter type: {type(dir_path)}"}

                # SECURITY: Prevent path traversal and enforce absolute paths
                if not dir_path or ".." in dir_path or not os.path.isabs(dir_path):
                    logger.error(f"Tool 'list_files' denied access to invalid/relative path: {dir_path}")
                    return {"status": "failure", "message": "Invalid or non-absolute directory path provided."}

                # SECURITY: Optional: Check against allowed directories (similar to read_file)
                # ... add check here if needed ...

                recursive = params.get('recursive', False)
                pattern = params.get('pattern', '*') # Default to all files
                abs_dir_path = os.path.abspath(dir_path)
                if os.path.isdir(abs_dir_path):
                    # Use safe join and glob
                    glob_pattern = os.path.join(abs_dir_path, pattern)
                    if recursive:
                        # For recursive, ensure pattern correctly handles subdirs if needed, e.g., '**/' + pattern
                        glob_pattern = os.path.join(abs_dir_path, '**', pattern)

                    files = glob.glob(glob_pattern, recursive=recursive)
                    # Optionally filter results further based on security policy
                    return {"status": "success", "files": files}
                else:
                    return {"status": "failure", "message": f"Directory not found or invalid path: {dir_path}"}
            # --- Add other tools here ---
            else:
                logger.warning(f"Attempted to use unknown tool: {tool_name}")
                return {"status": "failure", "message": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            return {"status": "failure", "message": f"Exception during tool execution: {e}"}

    async def report_error(self, agent_name: str, error_message: str, task_id: Optional[str] = None):
        """Handles error reporting from agents."""
        log_msg = f"ERROR reported by {agent_name}: {error_message}"
        if task_id: log_msg += f" (Task: {task_id})"
        logger.error(log_msg)
        # Optionally send notification for specific errors or agents
        # if "critical" in error_message.lower() or agent_name == "ThinkTool":
        #     await self.send_notification(f"Agent Error: {agent_name}", log_msg, level="error")

    async def send_notification(self, title: str, message: str, level: str = "info"):
        """Sends notifications (e.g., email) using the utility function."""
        # Run in background to avoid blocking orchestrator logic
        asyncio.create_task(send_notification(title, message, level, self.config))

    def _load_initial_proxies(self):
        """Loads initial proxy list from settings."""
        user = self.config.get_secret("SMARTPROXY_USER")
        password = self.config.get_secret("SMARTPROXY_PASSWORD")
        host = self.config.get("SMARTPROXY_HOST")
        port = self.config.get("SMARTPROXY_PORT")

        if user and password and host and port:
            # Construct the full proxy URL
            proxy_server_url = f"http://{user}:{password}@{host}:{port}"
            self._proxy_list.append({
                "server": proxy_server_url,
                "username": user, # Store separately if needed by Playwright context
                "password": password, # Store separately if needed by Playwright context
                "status": "unknown", # Initial status
                "last_used": None,
                "success_count": 0,
                "fail_count": 0,
                # Define potential use cases for this proxy pool
                "purpose_affinity": ["general", "Browse", "social_media", "account_creation", "legal_scan"],
                "quality_level": "high" # Assuming residential proxies are high quality
            })
            logger.info(f"Loaded initial Decodo proxy endpoint: {host}:{port}")
        else:
            logger.warning("Decodo (Smartproxy) credentials/host/port not fully configured. Proxy functionality limited.")
        # TODO: Add logic to load proxies from other providers or a database table

    @proxy_breaker
    async def get_proxy(self, purpose: str = "general", quality_level: str = "standard", target_url: Optional[str] = None, specific_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Gets the best available proxy based on purpose, quality, health, and least recent use."""
        async with self._proxy_lock:
            if not self._proxy_list:
                logger.warning(f"No proxies configured/available for purpose '{purpose}'.")
                return None

            now = time.time()
            eligible_proxies = []
            # Filter proxies based on status, purpose, and quality
            for proxy in self._proxy_list:
                # Skip banned or errored proxies
                if proxy.get("status") in ["banned", "error"]:
                    # Check if error state has expired (optional cooldown for errored proxies)
                    # error_cooldown = 60 * 30 # 30 minutes cooldown for errored proxies
                    # if proxy.get("status") == "error" and proxy.get("last_used", 0) + error_cooldown > now:
                    #     continue
                    # else: # Reset error status after cooldown to allow re-check
                    #     proxy["status"] = "unknown"
                    continue # Skip permanently banned or recently errored

                # Check purpose affinity
                if purpose != "general" and purpose not in proxy.get("purpose_affinity", ["general"]):
                    continue

                # Check quality level (allow equal or higher quality if lower requested, strict if higher requested)
                proxy_quality = proxy.get("quality_level", "standard")
                if quality_level == "high" and proxy_quality != "high": continue
                if quality_level == "premium_residential" and proxy_quality not in ["high", "premium_residential"]: continue # Be specific
                # If standard is requested, any quality is fine

                eligible_proxies.append(proxy)

            if not eligible_proxies:
                logger.warning(f"No eligible proxies found for purpose '{purpose}' and quality '{quality_level}'.")
                return None

            # Sort eligible proxies: prefer active, then unknown, then by least recently used
            eligible_proxies.sort(key=lambda p: (
                0 if p.get("status") == "active" else 1 if p.get("status") == "unknown" else 2,
                p.get("last_used") or 0 # Treat None last_used as very old
            ))

            selected_proxy = eligible_proxies[0]
            # Update last_used timestamp immediately upon selection
            selected_proxy["last_used"] = now
            proxy_display = selected_proxy['server'].split('@')[-1] if '@' in selected_proxy['server'] else selected_proxy['server']
            logger.info(f"Selected proxy for purpose '{purpose}': {proxy_display} (Status: {selected_proxy['status']})")
            # Return a copy to prevent modification of the internal list object outside the lock
            return selected_proxy.copy()

    async def report_proxy_status(self, proxy_server_url: str, success: bool):
        """Updates the status and stats of a used proxy."""
        async with self._proxy_lock:
            proxy_found = False
            for proxy in self._proxy_list:
                if proxy.get("server") == proxy_server_url:
                    proxy_found = True
                    proxy["last_used"] = time.time() # Update last used time regardless of outcome
                    if success:
                        proxy["success_count"] = proxy.get("success_count", 0) + 1
                        # Reset fail count on success to give proxy another chance after recovery
                        proxy["fail_count"] = 0
                        proxy["status"] = "active"
                    else:
                        proxy["fail_count"] = proxy.get("fail_count", 0) + 1
                        # Mark as error after consecutive failures
                        failure_threshold = 5 # Configurable threshold
                        if proxy["fail_count"] >= failure_threshold:
                            proxy["status"] = "error"
                            proxy_display = proxy_server_url.split('@')[-1] if '@' in proxy_server_url else proxy_server_url
                            logger.warning(f"Marking proxy {proxy_display} as 'error' after {proxy['fail_count']} consecutive failures.")
                        else:
                            # Mark as unknown after a failure, requires health check to reactivate
                            proxy["status"] = "unknown"
                    proxy_display = proxy_server_url.split('@')[-1] if '@' in proxy_server_url else proxy_server_url
                    logger.debug(f"Updated proxy stats for {proxy_display}: Success={success}, Status={proxy['status']}, Fails={proxy['fail_count']}")
                    break
            if not proxy_found:
                logger.warning(f"Attempted to report status for unknown proxy: {proxy_server_url}")

    async def _run_periodic_proxy_health_check(self):
        """Periodically checks the health of proxies marked as 'unknown' or 'error'."""
        while not shutdown_event.is_set():
            await asyncio.sleep(self.proxy_health_check_interval_seconds)
            if self.approved and not shutdown_event.is_set():
                logger.info("Orchestrator triggering periodic proxy health check...")
                proxies_to_check = []
                async with self._proxy_lock:
                    # Check proxies marked as unknown or those in error state (to see if they recovered)
                    proxies_to_check = [p.copy() for p in self._proxy_list if p.get("status") in ["unknown", "error"]]

                if not proxies_to_check:
                    logger.info("No proxies require health check in this cycle.")
                    continue

                logger.info(f"Checking health of {len(proxies_to_check)} proxies...")
                # Use a reliable, common target for checking proxy connectivity
                check_url = "https://ip.decodo.com/json" # Decodo's IP checker is good for this
                tasks = [self._check_single_proxy(proxy, check_url) for proxy in proxies_to_check]
                await asyncio.gather(*tasks)
                logger.info("Proxy health check cycle complete.")

    async def _check_single_proxy(self, proxy: Dict[str, Any], check_url: str):
        """Performs a health check on a single proxy."""
        proxy_url = proxy.get("server")
        if not proxy_url: return
        proxy_display = proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url
        success = False
        error_detail = None
        try:
            timeout = aiohttp.ClientTimeout(total=20) # Reasonable timeout for health check
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Use the proxy URL directly with aiohttp's proxy parameter
                async with session.get(check_url, proxy=proxy_url, ssl=False) as response: # Allow non-HTTPS check if needed
                    if response.status == 200:
                        # Optionally check response content if needed
                        # content = await response.json()
                        success = True
                        logger.debug(f"Proxy health check SUCCESS for {proxy_display}")
                    else:
                        error_detail = f"Status: {response.status}"
                        logger.warning(f"Proxy health check FAILED for {proxy_display}. {error_detail}")
        except asyncio.TimeoutError:
            error_detail = "Timeout"
            logger.warning(f"Proxy health check TIMEOUT for {proxy_display}.")
        except aiohttp.ClientProxyConnectionError as proxy_conn_err:
             error_detail = f"ProxyConnectionError: {proxy_conn_err}"
             logger.warning(f"Proxy health check FAILED for {proxy_display}. {error_detail}")
        except Exception as e:
            error_detail = f"General Error: {e}"
            logger.warning(f"Proxy health check FAILED for {proxy_display}. {error_detail}", exc_info=True)

        # Report status back (success=True/False)
        await self.report_proxy_status(proxy_url, success)

    # Inside the Orchestrator class:
    class _DatabaseSecureStorage:
        """Provides an interface similar to Vault/KMS using DB encryption."""
        def __init__(self, session_maker):
            self.session_maker = session_maker
            self.logger = logging.getLogger("SecureStorageShim") # Use specific logger

        async def store_new_account(self, service: str, identifier: str, password: str, status: str = 'active', metadata: Optional[Dict] = None) -> Optional[Dict]:
            """Stores new account credentials securely in the database."""
            if not self.session_maker: self.logger.error("DB session_maker unavailable."); return None
            if not service or not identifier or not password:
                self.logger.error("Missing service, identifier, or password for storing account.")
                return None

            encrypted_password = encrypt_data(password)
            if not encrypted_password: self.logger.error("Failed to encrypt password."); return None
            notes_str = json.dumps(metadata) if metadata else None

            try:
                async with self.session_maker() as session:
                    async with session.begin():
                        # Check if account already exists
                        stmt_check = select(AccountCredentials.id).where(AccountCredentials.service == service, AccountCredentials.account_identifier == identifier).limit(1)
                        existing_id = (await session.execute(stmt_check)).scalar_one_or_none()

                        if existing_id:
                            # Account exists - update it
                            self.logger.warning(f"Account {identifier} for {service} already exists (ID: {existing_id}). Updating status/notes/password.")
                            stmt_update = update(AccountCredentials).\
                                where(AccountCredentials.id == existing_id).\
                                values(
                                    status=status,
                                    notes=notes_str,
                                    last_used=None, # Reset last used on update? Or keep old? Decide policy.
                                    password=encrypted_password, # Update password
                                    last_status_update_ts=datetime.now(timezone.utc) # Track update time
                                )
                            await session.execute(stmt_update)
                            # Fetch updated details to return (excluding password)
                            stmt_get = select(AccountCredentials.id, AccountCredentials.service, AccountCredentials.account_identifier, AccountCredentials.status).where(AccountCredentials.id == existing_id)
                            updated_details = (await session.execute(stmt_get)).mappings().first()
                            return dict(updated_details) if updated_details else None
                        else:
                            # Account doesn't exist - create it
                            new_account = AccountCredentials(
                                service=service, account_identifier=identifier,
                                password=encrypted_password, status=status, notes=notes_str,
                                created_at=datetime.now(timezone.utc), # Set creation time
                                last_status_update_ts=datetime.now(timezone.utc)
                            )
                            session.add(new_account)
                            await session.flush() # Flush to get the ID
                            account_id = new_account.id
                            self.logger.info(f"Stored new account credentials for {service}/{identifier} (ID: {account_id}).")
                            # Return details of the newly created account
                            return {"id": account_id, "service": service, "account_identifier": identifier, "status": status}
            except Exception as e:
                self.logger.error(f"Error storing account credentials for {service}/{identifier}: {e}", exc_info=True)
                return None

        async def get_account_details_by_id(self, account_id: int) -> Optional[Dict]:
            """Fetches account details (excluding password) by ID."""
            if not self.session_maker: self.logger.error("DB session_maker unavailable."); return None
            if not isinstance(account_id, int): self.logger.error("Invalid account_id type provided."); return None
            try:
                async with self.session_maker() as session:
                    # Use session.get for efficient primary key lookup
                    account = await session.get(AccountCredentials, account_id)
                    if account:
                        # Return relevant details, explicitly excluding password
                        return {
                            "id": account.id, "service": account.service, "account_identifier": account.account_identifier,
                            "status": account.status, "proxy_used": account.proxy_used,
                            "created_at": account.created_at, "last_used": account.last_used,
                            "notes": account.notes, "last_status_update_ts": account.last_status_update_ts
                        }
                    else:
                        self.logger.warning(f"Account details not found for ID {account_id}.")
                        return None
            except Exception as e:
                self.logger.error(f"Error fetching account details ID {account_id}: {e}", exc_info=True)
                return None

        async def find_active_account_for_service(self, service: str) -> Optional[Dict]:
            """Finds an active account for a service (excluding password), marks as used."""
            if not self.session_maker: self.logger.error("DB session_maker unavailable."); return None
            if not service: self.logger.error("Service name required to find account."); return None
            try:
                async with self.session_maker() as session:
                    stmt = select(AccountCredentials).where(
                        AccountCredentials.service == service,
                        AccountCredentials.status == 'active'
                    ).order_by(func.random()).limit(1) # Random selection among active ones
                    account = await session.scalar(stmt)

                    if account:
                        account_details_no_pw = {
                            "id": account.id, "service": account.service, "account_identifier": account.account_identifier,
                            "status": account.status, "proxy_used": account.proxy_used,
                            "created_at": account.created_at, "last_used": account.last_used,
                            "notes": account.notes, "last_status_update_ts": account.last_status_update_ts
                        }
                        # Update last_used timestamp in a separate transaction/commit after returning details
                        async def update_last_used():
                            async with self.session_maker() as update_session:
                                async with update_session.begin():
                                    await update_session.execute(update(AccountCredentials).where(AccountCredentials.id == account.id).values(last_used=datetime.now(timezone.utc)))
                        asyncio.create_task(update_last_used()) # Fire and forget update
                        return account_details_no_pw
                    else:
                        self.logger.info(f"No active account found for service: {service}")
                        return None
            except Exception as e:
                self.logger.error(f"Error finding active account for {service}: {e}", exc_info=True)
                return None

        async def get_secret(self, identifier: Union[int, str]) -> Optional[str]:
            """Retrieves and decrypts a secret (e.g., password) using account ID."""
            if not self.session_maker: self.logger.error("DB session_maker unavailable."); return None
            try:
                # Ensure identifier is an integer (account ID)
                account_id = int(identifier)
            except (ValueError, TypeError):
                self.logger.error(f"Invalid identifier for get_secret (expected integer account ID): {identifier}")
                return None

            try:
                async with self.session_maker() as session:
                    # Fetch only the encrypted password field
                    stmt = select(AccountCredentials.password).where(AccountCredentials.id == account_id)
                    encrypted_pw = await session.scalar(stmt)

                    if encrypted_pw:
                        # Attempt decryption
                        decrypted_pw = decrypt_data(encrypted_pw)
                        if decrypted_pw:
                            return decrypted_pw
                        else:
                            # Log decryption failure, might indicate key mismatch or data corruption
                            self.logger.error(f"Failed to decrypt password for account ID {account_id}. Check encryption key or data integrity.")
                            return None
                    else:
                        # Password field might be null or account doesn't exist
                        self.logger.warning(f"No encrypted password found for account ID {account_id}.")
                        return None
            except Exception as e:
                self.logger.error(f"Error retrieving secret for ID {identifier}: {e}", exc_info=True)
                return None    

    def add_to_cache(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Adds an item to the in-memory cache with a TTL."""
        if not isinstance(key, str) or not key:
            logger.warning("Invalid cache key provided.")
            return
        ttl = ttl_seconds if ttl_seconds is not None else self.cache_ttl_default
        if ttl <= 0: # Don't cache if TTL is zero or negative
             return
        expires_at = time.time() + ttl
        # Consider size limits for the cache if necessary
        # MAX_CACHE_SIZE = 1000
        # if len(self._cache) > MAX_CACHE_SIZE:
        #     # Implement eviction strategy (e.g., remove oldest)
        #     pass
        self._cache[key] = (value, expires_at)
        logger.debug(f"Added item to cache. Key: {key[:20]}..., TTL: {ttl}s")

    def get_from_cache(self, key: str) -> Optional[Any]:
        """Retrieves an item from the cache if it exists and hasn't expired."""
        cached_item = self._cache.get(key)
        if cached_item:
            value, expires_at = cached_item
            if time.time() < expires_at:
                logger.debug(f"Cache hit for key: {key[:20]}...")
                return value
            else:
                # Cache expired, remove item
                logger.debug(f"Cache expired for key: {key[:20]}...")
                del self._cache[key]
        return None


    async def register_deepgram_connection_sdk(self, call_sid: str, dg_client: Any):
        """Stores the active Deepgram SDK client instance for a call SID."""
        if not call_sid or not dg_client:
            logger.warning("Attempted to register Deepgram connection with invalid SID or client.")
            return
        self.deepgram_connections[call_sid] = dg_client
        logger.info(f"Registered Deepgram SDK client for Call SID: {call_sid}")

    async def get_deepgram_connection_sdk(self, call_sid: str) -> Optional[Any]:
        """Retrieves the Deepgram SDK client instance for a call SID."""
        client = self.deepgram_connections.get(call_sid)
        if not client:
            logger.warning(f"No active Deepgram SDK client found for Call SID: {call_sid}")
        return client

    async def unregister_deepgram_connection_sdk(self, call_sid: str):
        """Removes the Deepgram SDK client instance for a call SID."""
        if call_sid in self.deepgram_connections:
            dg_client = self.deepgram_connections.pop(call_sid)
            logger.info(f"Unregistered Deepgram SDK client for Call SID: {call_sid}. Attempting to close connection.")
            # Attempt to gracefully close the Deepgram connection if possible
            if hasattr(dg_client, 'finish') and callable(dg_client.finish):
                try:
                    await dg_client.finish()
                    logger.debug(f"Gracefully finished Deepgram connection for {call_sid} during unregister.")
                except Exception as e:
                    # Log errors during closure but don't prevent unregistering
                    logger.warning(f"Error finishing Deepgram connection for {call_sid} during unregister: {e}")
        else:
            logger.debug(f"Attempted to unregister Deepgram client for non-existent Call SID: {call_sid}")


    async def host_temporary_audio(self, audio_data: bytes, filename: str) -> Optional[str]:
        """Saves audio data temporarily and returns a publicly accessible URL."""
        try:
            if not audio_data or not filename:
                 logger.error("Cannot host audio: Missing audio data or filename.")
                 return None

            # Sanitize filename
            safe_filename = re.sub(r'[^\w\.\-]', '_', filename)
            # Ensure filename has a common audio extension for mimetype detection
            if not any(safe_filename.lower().endswith(ext) for ext in ['.wav', '.mp3', '.ogg']):
                 safe_filename += ".wav" # Default to wav if extension missing/unknown

            # Construct absolute path using configured directory
            filepath = os.path.abspath(os.path.join(self.temp_audio_dir, safe_filename))

            # Security check: ensure the final path is still within the intended directory
            if not filepath.startswith(os.path.abspath(self.temp_audio_dir)):
                logger.error(f"Security Alert: Attempted to write hosted audio outside designated directory: {filepath}")
                return None

            # Write the audio data
            with open(filepath, 'wb') as f:
                f.write(audio_data)

            # Generate the URL using the configured AGENCY_BASE_URL
            base_url = str(self.config.AGENCY_BASE_URL).rstrip('/')
            # Ensure filename is URL-safe
            safe_url_filename = quote_plus(safe_filename)
            audio_url = f"{base_url}/hosted_audio/{safe_url_filename}"

            # Cannot use url_for reliably outside request context in Quart background tasks
            # try:
            #     # This assumes the route '/hosted_audio/<path:filename>' is registered
            #     # This requires being within an app context, might fail in background tasks
            #     with self.app.app_context():
            #         audio_url = url_for('serve_hosted_audio', filename=safe_filename, _external=True)
            # except RuntimeError: # Not in app context
            #      base_url = str(self.config.AGENCY_BASE_URL).rstrip('/')
            #      audio_url = f"{base_url}/hosted_audio/{safe_filename}"

            logger.info(f"Hosted temporary audio at: {audio_url}")
            return audio_url
        except Exception as e:
            logger.error(f"Failed to host temporary audio {filename}: {e}", exc_info=True)
            return None


    async def request_invoice_generation(self, client_id: int, amount: float, source_call_sid: str):
        """Sends a directive to ThinkTool/LegalAgent to generate an invoice."""
        logger.info(f"Requesting invoice generation for Client {client_id}, Amount ${amount:.2f}, Source Call {source_call_sid}")
        directive_content = {
            "client_id": client_id,
            "amount": amount,
            "source_reference": f"CallLog:{source_call_sid}",
            "due_date_days": 14 # Example: due in 14 days
        }
        # Delegate to ThinkTool to plan and manage the invoice workflow
        think_tool = self.agents.get("think")
        if think_tool:
            await self.delegate_task("ThinkTool", {
                "action": "generate_invoice", # ThinkTool should have this action
                "content": directive_content,
                "priority": 3 # High priority for potential sales
            })
        else:
            logger.error("Cannot request invoice generation: ThinkTool agent is unavailable.")


    async def _run_periodic_data_purge(self):
        """Periodically triggers ThinkTool to purge old data."""
        while not shutdown_event.is_set():
            await asyncio.sleep(self.purge_interval_seconds) # Use configured interval
            if self.approved and not shutdown_event.is_set():
                logger.info("Orchestrator triggering periodic data purge via ThinkTool...")
                think_tool = self.agents.get("think")
                if think_tool:
                    await self.delegate_task("ThinkTool", {"action": "purge_old_knowledge"})
                else:
                    logger.error("Cannot trigger data purge: ThinkTool agent is unavailable.")


    async def _run_periodic_feedback_collection(self):
        """Periodically collects insights from agents and sends to ThinkTool."""
        while not shutdown_event.is_set():
            await asyncio.sleep(self.feedback_interval_seconds) # Use configured interval
            if self.approved and not shutdown_event.is_set():
                logger.info("Orchestrator collecting feedback from agents...")
                all_insights = {}
                # Collect insights safely, handling potential agent errors
                for agent_name, agent in self.agents.items():
                    if hasattr(agent, 'collect_insights') and callable(agent.collect_insights):
                        try:
                            insights = await agent.collect_insights()
                            if insights:
                                all_insights[agent_name] = insights
                        except Exception as e:
                            logger.error(f"Error collecting insights from {agent_name}: {e}")
                            # Optionally report error to ThinkTool or log persistently

                # Delegate feedback processing to ThinkTool if insights were collected
                think_tool = self.agents.get("think")
                if all_insights and think_tool:
                    logger.info(f"Sending collected insights from {len(all_insights)} agents to ThinkTool.")
                    await self.delegate_task("ThinkTool", {"action": "process_feedback", "feedback_data": all_insights})
                # Continuing from the previous response, inside the _run_periodic_feedback_collection method:
                elif not all_insights:
                    logger.info("No agent insights collected in this cycle.")
                else: # Handle case where think_tool is not available but insights were collected
                    logger.warning("ThinkTool agent unavailable to process collected insights.")