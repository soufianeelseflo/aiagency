# Filename: agents/orchestrator.py
# Description: Central coordinator for the AI Agency, managing agents, workflows, resources, and UI backend integration.
# Version: 4.0 (Level 40+ Transmutation - Enhanced Proxy Mgmt, Webhooks, UI Integration)

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
from collections import deque
from typing import Dict, Optional, Tuple, Any, List, AsyncGenerator, Callable, Type, Union

# --- Core Framework Imports ---
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker,
)
from sqlalchemy import select, delete, func, update, text, case, or_
from sqlalchemy.exc import SQLAlchemyError
from quart import Quart, request, jsonify, websocket, send_file, url_for, Response, current_app
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
    from agents.base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
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

# Import UI route registration function
try:
    from ui.app import register_ui_routes
except ImportError:
    logging.critical("Failed to import register_ui_routes from ui.app. Check structure.")
    def register_ui_routes(app: Quart): pass # Dummy function

# Import database models
from models import (
    Base, Client, Metric, ExpenseLog, MigrationStatus, KnowledgeFragment,
    AccountCredentials, CallLog, Invoice, StrategicDirective, PromptTemplate
)

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
op_logger = logging.getLogger("OperationalLog")

# --- Circuit Breakers ---
llm_client_breaker = pybreaker.CircuitBreaker(
    fail_max=5, reset_timeout=60 * 3, name="LLMClientBreaker_L40" # Slightly faster reset
)
proxy_breaker = pybreaker.CircuitBreaker(
    fail_max=7, reset_timeout=60 * 10, name="ProxyProviderBreaker_L40" # Breaker for proxy provider itself
)

# --- Global Shutdown Event ---
shutdown_event = asyncio.Event()


# --- Orchestrator Class ---
class Orchestrator:
    """Central coordinator (L40+), managing agents, workflows, resources, and UI backend."""

    def __init__(self, schema="public"):
        self.config = settings
        self.session_maker = get_session_maker()
        self.agents: Dict[str, GeniusAgentBase] = {}
        # Create the Quart app instance HERE
        self.app = Quart(
            __name__,
            template_folder=os.path.join(os.path.dirname(__file__), "..", "ui", "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "..", "ui", "static")
        )
        self.setup_routes() # Call route setup immediately after app creation

        self.meta_prompt = self.config.META_PROMPT
        self.approved = False

        # LLM Client Management State
        self._llm_client: Optional[AsyncLLMClient] = None
        self._llm_client_status: Dict[str, Any] = {"status": "unavailable", "reason": "Not initialized", "unavailable_until": 0}

        # In-memory Cache
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_ttl_default = 3600

        # Deepgram WebSocket Registry
        self.deepgram_connections: Dict[str, Any] = {}
        self.temp_audio_dir = self.config.TEMP_AUDIO_DIR
        os.makedirs(self.temp_audio_dir, exist_ok=True)

        # Periodic Task Timing
        self.feedback_interval_seconds: int = int(self.config.THINKTOOL_FEEDBACK_INTERVAL_SECONDS)
        self.purge_interval_seconds: int = int(self.config.DATA_PURGE_INTERVAL_SECONDS)
        self.proxy_health_check_interval_seconds: int = int(self.config.get("PROXY_HEALTH_CHECK_INTERVAL_S", 3600 * 2)) # Check every 2 hours
        self.last_feedback_time: float = 0.0
        self.last_purge_time: float = 0.0
        self.last_proxy_health_check_time: float = 0.0

        self.running: bool = False
        self.background_tasks = set()
        self.status = "initializing"

        # Secure Storage Interface
        self.secure_storage = self._DatabaseSecureStorage(self.session_maker)

        # Proxy Management State (Enhanced)
        self._proxy_list: List[Dict[str, Any]] = [] # List of available proxies {server, username, password, status, last_used, success_count, fail_count}
        self._proxy_lock = asyncio.Lock()
        self._load_initial_proxies() # Load proxies from config on init

        logger.info("Orchestrator v4.0 (L40+ Transmutation) initialized.")

    # --- Initialization Methods ---
    async def initialize_database(self):
        """Initialize or update the database schema."""
        logger.info("Initializing database schema...")
        try:
            db_url_str = str(self.config.DATABASE_URL) if self.config.DATABASE_URL else None
            if not db_url_str: raise ValueError("DATABASE_URL is not configured.")

            engine = create_async_engine(db_url_str, echo=False, pool_pre_ping=True, pool_recycle=3600)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            await engine.dispose()
            logger.info("Database schema initialization complete.")
            return True
        except Exception as e:
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
                # Use OpenAI SDK compatibility [1, 2, 3, 7]
                self._llm_client = AsyncLLMClient(api_key=primary_key, base_url="https://openrouter.ai/api/v1")
                # Perform a simple test call (optional but recommended)
                # await self._llm_client.models.list() # Example test
                self._llm_client_status = {"status": "available", "reason": None, "unavailable_until": 0}
                logger.info(f"Initialized OpenRouter client (Primary Key: ...{primary_key[-4:]}).")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize primary OpenRouter client: {e}", exc_info=True)
                self._llm_client_status["reason"] = f"Initialization Error: {e}"
                return False
        else:
            logger.critical("CRITICAL: OPENROUTER_API_KEY not set. LLM functionality disabled.")
            self._llm_client_status["reason"] = "API Key Missing"
            return False

    async def initialize_agents(self):
        """Initialize the CORE agents, passing necessary secrets."""
        logger.info("Initializing CORE agents...")
        initialization_failed = False
        try:
            # ThinkTool
            self.agents["think"] = ThinkTool(self.session_maker, self.config, self)

            # BrowsingAgent
            # Note: Uses get_proxy internally, no direct proxy pass needed here
            self.agents["browsing"] = BrowsingAgent(self.session_maker, self)

            # EmailAgent
            imap_pass = self.config.get_secret("HOSTINGER_IMAP_PASS")
            if not imap_pass: logger.critical("Hostinger IMAP password missing. EmailAgent reply checking will fail."); initialization_failed = True
            self.agents["email"] = EmailAgent(self.session_maker, self, imap_password=imap_pass)

            # VoiceSalesAgent
            twilio_token = self.config.get_secret("TWILIO_AUTH_TOKEN")
            deepgram_key = self.config.get_secret("DEEPGRAM_API_KEY")
            if not twilio_token or not deepgram_key: logger.critical("Twilio/Deepgram secrets missing. VoiceSalesAgent will fail."); initialization_failed = True
            self.agents["voice_sales"] = VoiceSalesAgent(self.session_maker, self, twilio_auth_token=twilio_token, deepgram_api_key=deepgram_key)

            # GmailCreatorAgent
            self.agents["gmail_creator"] = GmailCreatorAgent(self.session_maker, self, self.config)
            logger.info("Initialized GmailCreatorAgent.")

            # LegalAgent
            self.agents["legal"] = LegalAgent(self.session_maker, self)

            # SocialMediaManager
            self.agents["social_media"] = SocialMediaManager(self, self.session_maker)

            # ProgrammerAgent (Check if file exists before importing/initializing)
            programmer_agent_path = os.path.join(os.path.dirname(__file__), 'programmer_agent.py')
            if os.path.exists(programmer_agent_path):
                try:
                    from agents.programmer_agent import ProgrammerAgent
                    self.agents["programmer"] = ProgrammerAgent(self, self.session_maker)
                    logger.info("Initialized ProgrammerAgent.")
                except ImportError: logger.warning("Found programmer_agent.py but failed to import ProgrammerAgent class.")
            else: logger.info("ProgrammerAgent file not found, skipping initialization.")

            if initialization_failed: raise RuntimeError("Failed to initialize critical agents due to missing secrets.")

            logger.info(f"Core agents initialized: {list(self.agents.keys())}")
            return True

        except Exception as e: logger.error(f"Core agent initialization failed: {e}", exc_info=True); return False

    # --- LLM Client Management ---
    @llm_client_breaker
    async def get_available_llm_client(self) -> Optional[AsyncLLMClient]:
        """Returns the primary LLM client if available and not in cooldown."""
        now = time.time()
        status_info = self._llm_client_status
        if status_info["status"] == "available" or now >= status_info.get("unavailable_until", 0):
            if status_info["status"] == "unavailable":
                status_info["status"] = "available"; status_info["reason"] = None; status_info["unavailable_until"] = 0
                logger.info("Primary LLM client available after cooldown.")
            return self._llm_client
        else:
            logger.warning(f"Primary LLM client unavailable. Reason: {status_info.get('reason', 'Unknown')}. Cooldown until {datetime.fromtimestamp(status_info.get('unavailable_until', 0)).isoformat()}")
            return None

    async def report_client_issue(self, issue_type: str):
        """Reports an issue with the primary LLM client, triggering cooldown."""
        now = time.time(); cooldown_seconds = 60 * 5; status = "unavailable"; reason = issue_type
        if issue_type == "auth_error": cooldown_seconds = 60 * 60 * 24 * 365; reason = "Authentication Error"; logger.critical("Primary LLM client marked permanently unavailable (auth error).")
        elif issue_type == "rate_limit": cooldown_seconds = 60 * 2; reason = "Rate Limited"
        elif issue_type == "timeout_error": cooldown_seconds = 60 * 3; reason = "Timeout"
        else: reason = "General Error"
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
        if not llm_client: logger.error(f"Agent '{agent_name}' failed: Primary LLM client unavailable."); return None

        # Model Selection Logic (remains the same as v3.2)
        model_key = "default_llm" # Fallback
        if agent_name == "ThinkTool": model_key = "think_general"
        elif agent_name == "EmailAgent": model_key = "email_draft"
        elif agent_name == "VoiceSalesAgent": model_key = "voice_response"
        elif agent_name == "LegalAgent": model_key = "legal_analysis" # Changed default for Legal
        elif agent_name == "BrowsingAgent": model_key = "browsing_visual_analysis" if image_data else "browsing_summarize"
        # Task-specific overrides
        task_specific_model_key = None
        if isinstance(prompt, str):
             if "synthesize" in prompt.lower() or "strategize" in prompt.lower(): task_specific_model_key = "think_strategize"
             elif "critique" in prompt.lower(): task_specific_model_key = "think_critique"
             elif "radar" in prompt.lower() or "scouting" in prompt.lower(): task_specific_model_key = "think_radar"
             elif "validate" in prompt.lower(): task_specific_model_key = "think_validate"
             elif "educational content" in prompt.lower(): task_specific_model_key = "think_user_education"
             elif "humanize" in prompt.lower(): task_specific_model_key = "email_humanize"
             elif "intent" in prompt.lower(): task_specific_model_key = "voice_intent"
        final_model_key = task_specific_model_key or model_key
        model_name = self.config.OPENROUTER_MODELS.get(final_model_key, self.config.OPENROUTER_MODELS["default_llm"])
        logger.debug(f"Selected model '{model_name}' for agent '{agent_name}' task (key: {final_model_key}).")

        # Caching Logic (remains the same)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        cache_key_parts = ["llm_call", prompt_hash, model_name, str(temperature), str(max_tokens), str(is_json_output), str(image_data is not None)]
        cache_key = ":".join(cache_key_parts); cache_ttl = self.cache_ttl_default
        cached_result = self.get_from_cache(cache_key)
        if cached_result is not None: logger.debug(f"LLM call cache hit (Orchestrator) for key: {cache_key[:20]}..."); return cached_result

        try:
            # Prepare messages for OpenAI SDK format [1, 7]
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
            if is_json_output: request_params["response_format"] = {"type": "json_object"}

            # Add optional OpenRouter headers for tracking [1, 2]
            extra_headers = {
                "HTTP-Referer": str(settings.AGENCY_BASE_URL), # Use validated URL
                "X-Title": settings.APP_NAME,
            }
            request_params["extra_headers"] = extra_headers

            logger.debug(f"Orchestrator making LLM Call: Agent={agent_name}, Model={model_name}, Multimodal={image_data is not None}")

            response = await llm_client.chat.completions.create(**request_params)

            content = response.choices[0].message.content.strip() if response.choices and response.choices[0].message and response.choices[0].message.content else None

            # Track usage & cost (remains the same)
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = input_tokens + output_tokens
            cost_per_million_input = 0.50; cost_per_million_output = 1.50 # Example costs
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
            raise # Re-raise the exception for retry logic or task failure handling

    # --- Route Setup ---
    def setup_routes(self):
        """Configure Quart routes for UI interaction and webhooks."""
        # Register routes defined in ui/app.py
        register_ui_routes(self.app)

        # Add routes specific to Orchestrator functionality (e.g., webhooks, audio hosting)
        @self.app.websocket("/twilio_call")
        async def handle_twilio_websocket():
            """Handles the bidirectional audio stream from Twilio."""
            call_sid = None; deepgram_live_client = None
            try:
                while True:
                    message = await websocket.receive(); data = json.loads(message); event = data.get("event")
                    if event == "start":
                        call_sid = data.get("start", {}).get("callSid"); logger.info(f"Twilio WS: Start event for {call_sid}")
                        # Retrieve the Deepgram client registered by VoiceSalesAgent
                        deepgram_live_client = await self.get_deepgram_connection_sdk(call_sid)
                        if not deepgram_live_client: logger.error(f"Twilio WS: No Deepgram SDK client found for {call_sid}."); break
                        else: logger.info(f"Twilio WS: Found Deepgram SDK client for {call_sid}.")
                    elif event == "media":
                        if not deepgram_live_client: continue
                        payload = data.get("media", {}).get("payload")
                        if payload:
                            audio_bytes = base64.b64decode(payload)
                            # Send audio to Deepgram SDK client [9, 11, 14, 18]
                            if hasattr(deepgram_live_client, 'send') and callable(deepgram_live_client.send):
                                await deepgram_live_client.send(audio_bytes)
                            else: logger.warning(f"Deepgram client for {call_sid} lacks send method.")
                    elif event == "stop": logger.info(f"Twilio WS: Stop event for {call_sid}."); break
                    elif event == "error": logger.error(f"Twilio WS: Error event for {call_sid}: {data.get('error')}"); break
            except websockets.exceptions.ConnectionClosedOK: logger.info(f"Twilio WS: Connection closed normally for {call_sid}.")
            except Exception as e: logger.error(f"Twilio WS: Unexpected error for {call_sid}: {e}", exc_info=True,)
            finally:
                logger.info(f"Twilio WS: Cleaning up for {call_sid}");
                if call_sid: await self.unregister_deepgram_connection_sdk(call_sid)

        @self.app.route("/track/<tracking_id>.png")
        async def handle_tracking_pixel(tracking_id):
            """Handles email open tracking pixel requests."""
            logger.info(f"Tracking pixel hit: {tracking_id}")
            email_agent = self.agents.get("email");
            if email_agent and hasattr(email_agent, "process_email_open"): asyncio.create_task(email_agent.process_email_open(tracking_id))
            else: logger.warning(f"EmailAgent not found or cannot process opens for tracking ID: {tracking_id}")
            pixel_data = base64.b64decode("R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==")
            return (pixel_data, 200, {"Content-Type": "image/gif", "Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0",})

        @self.app.route("/webhooks/mailersend", methods=["POST"])
        async def handle_mailersend_webhook():
            """Handles incoming webhooks from MailerSend for bounces, complaints, etc. [5, 20, 26]"""
            try:
                payload = await request.get_json()
                if not payload or 'type' not in payload or 'data' not in payload:
                    logger.warning("Received invalid/incomplete MailerSend webhook payload.")
                    return jsonify({"status": "error", "message": "Invalid payload"}), 400

                event_type = payload.get('type', '').lower()
                event_data = payload.get('data', {})
                email_data = event_data.get('email', {})
                recipient_email = email_data.get('recipient', {}).get('email')
                message_id = email_data.get('message', {}).get('id') # MailerSend Message ID

                logger.info(f"Received MailerSend Webhook: Type='{event_type}', Recipient='{recipient_email}', MsgID='{message_id}'")

                if not recipient_email:
                    logger.warning("MailerSend webhook missing recipient email.")
                    return jsonify({"status": "ok", "message": "Missing recipient"}), 200 # Acknowledge receipt

                update_values = {}
                log_op_level = 'info'
                log_op_message = f"Webhook '{event_type}' received for {recipient_email}."
                thinktool_tags = ["webhook", "mailersend", event_type]

                if event_type in ['activity.hard_bounced', 'activity.spam_complaint']:
                    update_values['is_deliverable'] = False
                    update_values['opt_in'] = False # Treat hard bounce/spam complaint as opt-out
                    log_op_level = 'warning'
                    log_op_message = f"Marking {recipient_email} undeliverable/opt-out due to MailerSend webhook: {event_type}."
                    thinktool_tags.append("deliverability_issue")
                elif event_type == 'activity.unsubscribed':
                    update_values['opt_in'] = False
                    log_op_level = 'warning'
                    log_op_message = f"Marking {recipient_email} opted-out due to MailerSend webhook: {event_type}."
                    thinktool_tags.append("opt_out")

                # Log the raw event to ThinkTool KB for potential pattern analysis
                if self.agents.get("think"):
                    asyncio.create_task(self.agents["think"].log_knowledge_fragment(
                        agent_source="MailerSendWebhook", data_type="mailersend_event",
                        content=payload, tags=thinktool_tags, relevance_score=0.6,
                        source_reference=f"Recipient:{recipient_email}_MsgID:{message_id}"
                    ))

                if update_values:
                    # Run DB update in background to respond quickly
                    async def update_client_status():
                        session_maker = get_session_maker() # Get session maker
                        if not session_maker: return
                        try:
                            async with session_maker() as session:
                                async with session.begin():
                                    stmt = update(Client).where(Client.email == recipient_email).values(**update_values)
                                    await session.execute(stmt)
                                logger.info(f"Updated client status for {recipient_email} based on webhook '{event_type}'.")
                        except Exception as db_err:
                            logger.error(f"DB Error updating client status from webhook for {recipient_email}: {db_err}", exc_info=True)

                    asyncio.create_task(update_client_status())
                    # Log operation immediately
                    await self.log_operation(log_op_level, log_op_message)

                # Acknowledge receipt to MailerSend quickly
                return jsonify({"status": "received"}), 200

            except Exception as e:
                logger.error(f"Error processing MailerSend webhook: {e}", exc_info=True)
                return jsonify({"status": "error", "message": "Internal server error"}), 500

        @self.app.route("/webhooks/clay", methods=["POST"])
        async def handle_clay_webhook():
            """Handles incoming webhooks from Clay.com enrichment runs."""
            try:
                payload = await request.get_json()
                if not payload:
                    logger.warning("Received empty Clay webhook payload.")
                    return jsonify({"status": "error", "message": "Empty payload"}), 400

                # Extract relevant data - structure might vary, adjust based on Clay's actual webhook format
                # Assume payload contains the enriched data directly or nested under a key like 'results'
                # Also look for identifying info passed back (e.g., original input, metadata)
                clay_run_id = payload.get("run_id") # Example field
                source_reference = payload.get("metadata", {}).get("source_reference") # If you pass metadata
                original_input = payload.get("input_parameters", {}) # If Clay returns original input
                enriched_data = payload # Assume top-level is the data, adjust if nested

                logger.info(f"Received Clay Webhook. Run ID: {clay_run_id}, Source Ref: {source_reference}")

                # Delegate processing to ThinkTool
                if self.agents.get("think"):
                    asyncio.create_task(self.agents["think"].execute_task({
                        "action": "process_clay_webhook_data",
                        "content": {
                            "enriched_data": enriched_data,
                            "original_input_parameters": original_input,
                            "source_reference": source_reference,
                            "clay_run_id": clay_run_id
                        }
                    }))
                    return jsonify({"status": "received"}), 200
                else:
                    logger.error("ThinkTool not available to process Clay webhook.")
                    return jsonify({"status": "error", "message": "Internal processing error"}), 500

            except Exception as e:
                logger.error(f"Error processing Clay webhook: {e}", exc_info=True)
                return jsonify({"status": "error", "message": "Internal server error"}), 500

        @self.app.route("/hosted_audio/<path:filename>")
        async def serve_hosted_audio(filename):
            """Serves temporary audio files generated by VoiceSalesAgent."""
            # Basic security check
            if ".." in filename or filename.startswith("/"):
                logger.warning(f"Attempted path traversal in hosted_audio: {filename}")
                return jsonify({"error": "Forbidden"}), 403
            try:
                # Construct safe path within the designated directory
                safe_path = os.path.abspath(os.path.join(self.temp_audio_dir, filename))
                if not safe_path.startswith(os.path.abspath(self.temp_audio_dir)):
                    logger.warning(f"Attempted access outside temp_audio_dir: {filename}")
                    return jsonify({"error": "Forbidden"}), 403

                if os.path.exists(safe_path) and os.path.isfile(safe_path):
                    mimetype = ("audio/wav" if filename.lower().endswith(".wav") else "audio/mpeg")
                    logger.debug(f"Serving hosted audio: {filename} with mimetype {mimetype}")
                    return await send_file(safe_path, mimetype=mimetype)
                else:
                    logger.warning(f"Hosted audio file not found: {filename} (Path: {safe_path})")
                    return jsonify({"error": "File not found"}), 404
            except Exception as e:
                logger.error(f"Error serving hosted audio {filename}: {e}", exc_info=True)
                return jsonify({"error": "Internal server error"}), 500

        logger.info("Orchestrator routes configured.")

    # --- Main Execution Loop ---
    async def run(self):
        """Initializes and runs the AI Agency Orchestrator."""
        logger.info("Orchestrator starting full initialization sequence...")
        self.running = False; self.status = "initializing"
        try:
            if not await self.initialize_database(): raise RuntimeError("Database initialization failed.")
            if not await self.initialize_clients(): raise RuntimeError("LLM Client initialization failed.")
            if not await self.initialize_agents(): raise RuntimeError("Agent initialization failed.")
            logger.info("Orchestrator initialization complete.")

            logger.info("Starting background agent run loops and periodic tasks...")
            self.background_tasks = set()
            for agent_name, agent in self.agents.items():
                if hasattr(agent, "run") and callable(agent.run): task = asyncio.create_task(agent.run(), name=f"AgentLoop_{agent_name}"); self.background_tasks.add(task)
                elif hasattr(agent, "start") and callable(agent.start): task = asyncio.create_task(agent.start(), name=f"AgentStart_{agent_name}"); self.background_tasks.add(task); logger.info(f"Called start() for agent {agent_name}")
                else: logger.warning(f"Agent {agent_name} does not have a callable run or start method.")

            self.background_tasks.add(asyncio.create_task(self._run_periodic_data_purge(), name="PeriodicDataPurge"))
            self.background_tasks.add(asyncio.create_task(self._run_periodic_feedback_collection(), name="PeriodicFeedback"))
            self.background_tasks.add(asyncio.create_task(self._run_periodic_proxy_health_check(), name="PeriodicProxyHealth")) # Add proxy health check
            logger.info(f"Started {len(self.background_tasks)} background tasks.")

            logger.info("Orchestrator entering main operational state (API/Event driven).")
            self.running = True; self.status = "running"
            self.last_feedback_time = time.time(); self.last_purge_time = time.time(); self.last_proxy_health_check_time = time.time()

            # Main loop to keep orchestrator alive and monitor tasks
            while self.running:
                try:
                    tasks_to_remove = set()
                    for task in self.background_tasks:
                        if task.done():
                            tasks_to_remove.add(task)
                            try: task.result(); logger.info(f"Background task {task.get_name()} completed.")
                            except asyncio.CancelledError: logger.info(f"Background task {task.get_name()} was cancelled.")
                            except Exception as task_exc: logger.error(f"Background task {task.get_name()} failed: {task_exc}", exc_info=True)
                    self.background_tasks -= tasks_to_remove
                    await asyncio.sleep(60) # Check task status periodically
                except asyncio.CancelledError: logger.info("Orchestrator main loop cancelled."); break
                except Exception as e:
                    logger.critical(f"CRITICAL ERROR in Orchestrator main loop: {e}", exc_info=True)
                    self.running = False; self.status = "error"
                    try: await self.send_notification("CRITICAL Orchestrator Failure", f"Orchestrator failed: {e}")
                    except Exception as report_err: logger.error(f"Failed send critical failure report: {report_err}")

        except (ValueError, RuntimeError) as init_err: logger.critical(f"Fatal Error: Orchestrator initialization failed: {init_err}", exc_info=True); self.status = "failed_initialization"
        except Exception as e: logger.critical(f"Fatal Error: Unhandled exception during Orchestrator run setup: {e}", exc_info=True); self.status = "error"
        finally:
            logger.info("Initiating graceful shutdown sequence...")
            self.running = False; self.status = "stopping"; shutdown_event.set()
            all_tasks = list(self.background_tasks)
            if all_tasks:
                logger.info(f"Cancelling {len(all_tasks)} running background tasks...")
                for task in all_tasks:
                    if task and not task.done(): task.cancel()
                await asyncio.gather(*all_tasks, return_exceptions=True)
                logger.info("Background tasks cancellation complete.")
            agent_stop_tasks = []
            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'stop') and callable(agent.stop):
                    agent_status = getattr(agent, 'status', 'unknown')
                    if agent_status not in [getattr(agent, 'STATUS_STOPPING', 'stopping'), getattr(agent, 'STATUS_STOPPED', 'stopped')]:
                        logger.info(f"Calling stop() for agent {agent_name}...")
                        agent_stop_tasks.append(asyncio.create_task(agent.stop(timeout=25.0)))
                    else: logger.info(f"Agent {agent_name} already stopping/stopped ({agent_status}). Skipping stop call.")
                else: logger.warning(f"Agent {agent_name} has no stop method.")
            if agent_stop_tasks:
                logger.info(f"Waiting for {len(agent_stop_tasks)} agents to stop...")
                await asyncio.gather(*agent_stop_tasks, return_exceptions=True)
                logger.info("Agent stop sequence complete.")
            self.status = "stopped"
            logger.info("-------------------- Application Stopping --------------------")
            logging.shutdown()
            print("[INFO] Orchestrator: Process stopped.")

    # --- Agent Interaction & Tooling ---
    async def delegate_task(self, agent_name: str, task_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Delegates a task to a specific agent."""
        agent = self.agents.get(agent_name)
        if not agent: logger.error(f"Delegation failed: Agent '{agent_name}' not found."); return {"status": "error", "message": f"Agent '{agent_name}' not found."}
        if not hasattr(agent, 'execute_task') or not callable(agent.execute_task): logger.error(f"Delegation failed: Agent '{agent_name}' has no callable execute_task method."); return {"status": "error", "message": f"Agent '{agent_name}' cannot execute tasks."}

        task_id = task_details.get('id', str(uuid.uuid4()))
        task_details['id'] = task_id
        logger.info(f"Delegating task {task_id} ({task_details.get('action', 'N/A')}) to {agent_name}.")
        try:
            # Add directive ID to task details if available (for linking KFs)
            if task_details.get('directive_id'):
                task_details.setdefault('content', {})['directive_id'] = task_details['directive_id']

            result = await agent.execute_task(task_details)
            # Update directive status if applicable
            if task_details.get('directive_id') and result and 'status' in result:
                final_status = result['status']
                if final_status == 'success': final_status = 'completed'
                elif final_status == 'error': final_status = 'failed'
                await self.update_directive_status(task_details['directive_id'], final_status, result.get('message', 'Task finished.'))
            return result
        except Exception as e:
            logger.error(f"Error during task delegation to {agent_name} (Task ID: {task_id}): {e}", exc_info=True)
            await self.report_error(agent_name, f"Task delegation failed: {e}", task_id)
            if task_details.get('directive_id'): await self.update_directive_status(task_details['directive_id'], 'failed', f"Exception: {e}")
            return {"status": "error", "message": f"Exception during task execution: {e}"}

    async def update_directive_status(self, directive_id: int, status: str, result_summary: Optional[str] = None):
        """Updates the status and result of a StrategicDirective."""
        if not self.session_maker: return
        try:
            async with self.session_maker() as session:
                async with session.begin():
                    stmt = update(StrategicDirective).where(StrategicDirective.id == directive_id).values(status=status, result_summary=result_summary)
                    await session.execute(stmt)
                self.logger.info(f"Updated directive {directive_id} status to '{status}'.")
        except Exception as e: self.logger.error(f"Failed to update directive {directive_id} status: {e}")

    async def use_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Allows agents to request tool usage via the orchestrator."""
        logger.info(f"Orchestrator received tool use request: {tool_name} with params: {params}")
        if tool_name == "read_file":
            file_path = params.get('path')
            if file_path and os.path.exists(file_path) and os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
                    return {"status": "success", "content": content}
                except Exception as e: return {"status": "failure", "message": f"Error reading file: {e}"}
            else: return {"status": "failure", "message": f"File not found or invalid path: {file_path}"}
        elif tool_name == "list_files":
            dir_path = params.get('path')
            recursive = params.get('recursive', False)
            pattern = params.get('pattern', '*')
            if dir_path and os.path.isdir(dir_path):
                try:
                    glob_pattern = os.path.join(dir_path, '**', pattern) if recursive else os.path.join(dir_path, pattern)
                    files = glob.glob(glob_pattern, recursive=recursive)
                    return {"status": "success", "files": files}
                except Exception as e: return {"status": "failure", "message": f"Error listing files: {e}"}
            else: return {"status": "failure", "message": f"Directory not found or invalid path: {dir_path}"}
        # Add other tools: search_engine_api, store_artifact, etc.
        else: logger.warning(f"Attempted to use unknown tool: {tool_name}"); return {"status": "failure", "message": f"Unknown tool: {tool_name}"}

    async def report_error(self, agent_name: str, error_message: str, task_id: Optional[str] = None):
        """Handles error reporting from agents."""
        log_msg = f"ERROR reported by {agent_name}: {error_message}"
        if task_id: log_msg += f" (Task: {task_id})"
        logger.error(log_msg)
        # Consider sending notification only for critical/repeated errors
        # await self.send_notification(f"Agent Error: {agent_name}", log_msg, level="error")

    async def send_notification(self, title: str, message: str, level: str = "info"):
        """Sends notifications (e.g., email, Slack) using the utility function."""
        await send_notification(title, message, level, self.config)

    # --- Proxy Management (Enhanced) ---
    def _load_initial_proxies(self):
        """Loads initial proxy list from settings."""
        # Decodo (formerly Smartproxy) uses endpoint:port format [6, 10, 15, 32]
        user = self.config.get_secret("SMARTPROXY_USER")
        password = self.config.get_secret("SMARTPROXY_PASSWORD")
        host = self.config.get("SMARTPROXY_HOST")
        port = self.config.get("SMARTPROXY_PORT")

        if user and password and host and port:
            # Assuming a rotating residential setup for this example
            # Specific endpoint might vary based on location/session type chosen in Decodo dashboard
            proxy_server_url = f"http://{user}:{password}@{host}:{port}"
            # Add multiple entries if using different session types or locations
            self._proxy_list.append({
                "server": proxy_server_url, "username": user, "password": password, # Store components if needed later
                "status": "unknown", "last_used": None, "success_count": 0, "fail_count": 0,
                "purpose_affinity": ["general", "browsing", "social_media", "account_creation"], # Example affinities
                "quality_level": "high" # Assume high quality for paid residential
            })
            logger.info(f"Loaded initial Decodo proxy endpoint: {host}:{port}")
        else:
            logger.warning("Decodo (Smartproxy) credentials/host/port not fully configured. Proxy functionality limited.")
        # TODO: Add logic to load proxies from other providers or a file/DB if needed

    @proxy_breaker
    async def get_proxy(self, purpose: str = "general", quality_level: str = "standard", target_url: Optional[str] = None, specific_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Gets the best available proxy based on purpose, quality, health, and least recent use."""
        async with self._proxy_lock:
            if not self._proxy_list:
                logger.warning(f"No proxies available in the list for purpose '{purpose}'.")
                return None

            now = time.time()
            eligible_proxies = []
            for proxy in self._proxy_list:
                # Filter by status (allow unknown initially)
                if proxy.get("status") in ["banned", "error"]: continue
                # Filter by purpose affinity (optional)
                if purpose != "general" and purpose not in proxy.get("purpose_affinity", ["general"]): continue
                # Filter by quality (optional)
                if quality_level == "high" and proxy.get("quality_level") != "high": continue
                # Add more filters (e.g., region based on specific_hint or target_url)

                eligible_proxies.append(proxy)

            if not eligible_proxies:
                logger.warning(f"No eligible proxies found for purpose '{purpose}' and quality '{quality_level}'.")
                return None

            # Sort eligible proxies: Prioritize healthy, then least recently used
            eligible_proxies.sort(key=lambda p: (
                0 if p.get("status") == "active" else 1 if p.get("status") == "unknown" else 2, # Healthy first
                p.get("last_used") or 0 # Least recently used first
            ))

            selected_proxy = eligible_proxies[0]
            selected_proxy["last_used"] = now # Optimistically mark as used
            logger.info(f"Selected proxy for purpose '{purpose}': {selected_proxy['server'].split('@')[-1] if '@' in selected_proxy['server'] else 'Direct?'} (Status: {selected_proxy['status']})")

            # Return a copy to avoid modifying the internal state directly outside the lock
            return selected_proxy.copy()

    async def report_proxy_status(self, proxy_server_url: str, success: bool):
        """Updates the status and stats of a used proxy."""
        async with self._proxy_lock:
            for proxy in self._proxy_list:
                if proxy.get("server") == proxy_server_url:
                    proxy["last_used"] = time.time()
                    if success:
                        proxy["success_count"] = proxy.get("success_count", 0) + 1
                        proxy["fail_count"] = 0 # Reset fail count on success
                        proxy["status"] = "active"
                    else:
                        proxy["fail_count"] = proxy.get("fail_count", 0) + 1
                        # Implement logic to mark as 'error' or 'banned' after consecutive failures
                        if proxy["fail_count"] >= 5: # Example threshold
                            proxy["status"] = "error"
                            logger.warning(f"Marking proxy {proxy_server_url.split('@')[-1]} as 'error' after {proxy['fail_count']} consecutive failures.")
                        else:
                            proxy["status"] = "unknown" # Revert to unknown after a failure
                    logger.debug(f"Updated proxy stats for {proxy_server_url.split('@')[-1]}: Success={success}, Status={proxy['status']}")
                    break

    async def _run_periodic_proxy_health_check(self):
        """Periodically checks the health of proxies marked as 'unknown' or 'error'."""
        while not shutdown_event.is_set():
            await asyncio.sleep(self.proxy_health_check_interval_seconds)
            if self.approved and not shutdown_event.is_set():
                logger.info("Orchestrator triggering periodic proxy health check...")
                proxies_to_check = []
                async with self._proxy_lock:
                    proxies_to_check = [p for p in self._proxy_list if p.get("status") in ["unknown", "error"]]

                if not proxies_to_check: logger.info("No proxies require health check."); continue

                logger.info(f"Checking health of {len(proxies_to_check)} proxies...")
                check_url = "https://ip.decodo.com/json" # Use Decodo's IP check [6]
                tasks = [self._check_single_proxy(proxy, check_url) for proxy in proxies_to_check]
                await asyncio.gather(*tasks)
                logger.info("Proxy health check cycle complete.")

    async def _check_single_proxy(self, proxy: Dict[str, Any], check_url: str):
        """Performs a health check on a single proxy."""
        proxy_url = proxy.get("server")
        if not proxy_url: return
        proxy_display = proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url

        success = False
        try:
            timeout = aiohttp.ClientTimeout(total=15) # Shorter timeout for health check
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(check_url, proxy=proxy_url, ssl=False) as response: # Use proxy directly
                    if response.status == 200:
                        # Optionally check response content for validity
                        # content = await response.json()
                        success = True
                        logger.debug(f"Proxy health check SUCCESS for {proxy_display}")
                    else:
                        logger.warning(f"Proxy health check FAILED for {proxy_display}. Status: {response.status}")
        except Exception as e:
            logger.warning(f"Proxy health check FAILED for {proxy_display}. Error: {e}")

        # Update status based on check result
        await self.report_proxy_status(proxy_url, success)


    # --- Cache Methods ---
    def add_to_cache(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Adds an item to the in-memory cache with a TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self.cache_ttl_default
        expires_at = time.time() + ttl
        self._cache[key] = (value, expires_at)
        logger.debug(f"Added item to cache. Key: {key[:20]}..., TTL: {ttl}s")

    def get_from_cache(self, key: str) -> Optional[Any]:
        """Retrieves an item from the cache if it exists and hasn't expired."""
        cached_item = self._cache.get(key)
        if cached_item:
            value, expires_at = cached_item
            if time.time() < expires_at: logger.debug(f"Cache hit for key: {key[:20]}..."); return value
            else: logger.debug(f"Cache expired for key: {key[:20]}..."); del self._cache[key]
        return None

    # --- Deepgram Connection Registry ---
    async def register_deepgram_connection_sdk(self, call_sid: str, dg_client: Any):
        """Stores the active Deepgram SDK client instance for a call SID."""
        self.deepgram_connections[call_sid] = dg_client
        logger.info(f"Registered Deepgram SDK client for Call SID: {call_sid}")

    async def get_deepgram_connection_sdk(self, call_sid: str) -> Optional[Any]:
        """Retrieves the Deepgram SDK client instance for a call SID."""
        client = self.deepgram_connections.get(call_sid)
        if not client: logger.warning(f"No active Deepgram SDK client found for Call SID: {call_sid}")
        return client

    async def unregister_deepgram_connection_sdk(self, call_sid: str):
        """Removes the Deepgram SDK client instance for a call SID."""
        if call_sid in self.deepgram_connections:
            # Attempt to gracefully close the Deepgram connection if possible
            dg_client = self.deepgram_connections.pop(call_sid)
            if hasattr(dg_client, 'finish') and callable(dg_client.finish):
                try:
                    await dg_client.finish()
                    logger.debug(f"Gracefully finished Deepgram connection for {call_sid} during unregister.")
                except Exception as e:
                    logger.warning(f"Error finishing Deepgram connection for {call_sid} during unregister: {e}")
            logger.info(f"Unregistered Deepgram SDK client for Call SID: {call_sid}")

    # --- Temporary Audio Hosting ---
    async def host_temporary_audio(self, audio_data: bytes, filename: str) -> Optional[str]:
        """Saves audio data temporarily and returns a publicly accessible URL."""
        try:
            # Sanitize filename
            safe_filename = re.sub(r'[^\w\.\-]', '_', filename)
            # Ensure filename has a common audio extension for mimetype detection
            if not safe_filename.lower().endswith(('.wav', '.mp3')): safe_filename += ".wav"

            filepath = os.path.join(self.temp_audio_dir, safe_filename)
            with open(filepath, 'wb') as f: f.write(audio_data)

            # Use url_for if running within Quart request context, otherwise construct manually
            try:
                # This assumes the route '/hosted_audio/<path:filename>' is registered
                audio_url = url_for('serve_hosted_audio', filename=safe_filename, _external=True)
            except RuntimeError: # Not in request context
                 base_url = str(self.config.AGENCY_BASE_URL).rstrip('/')
                 audio_url = f"{base_url}/hosted_audio/{safe_filename}"

            logger.info(f"Hosted temporary audio at: {audio_url}")
            return audio_url
        except Exception as e:
            logger.error(f"Failed to host temporary audio {filename}: {e}", exc_info=True)
            return None

    # --- Invoice Generation Request ---
    async def request_invoice_generation(self, client_id: int, amount: float, source_call_sid: str):
        """Sends a directive to ThinkTool/LegalAgent to generate an invoice."""
        logger.info(f"Requesting invoice generation for Client {client_id}, Amount ${amount:.2f}, Source Call {source_call_sid}")
        directive_content = {
            "client_id": client_id, "amount": amount,
            "source_reference": f"CallLog:{source_call_sid}", "due_date_days": 14
        }
        # ThinkTool is likely better suited to plan the invoice generation workflow
        await self.delegate_task("ThinkTool", {
            "action": "generate_invoice", "content": directive_content, "priority": 3
        })

    # --- Periodic Tasks ---
    async def _run_periodic_data_purge(self):
        """Periodically triggers ThinkTool to purge old data."""
        while not shutdown_event.is_set():
            await asyncio.sleep(self.purge_interval_seconds)
            if self.approved and not shutdown_event.is_set():
                logger.info("Orchestrator triggering periodic data purge via ThinkTool...")
                await self.delegate_task("ThinkTool", {"action": "purge_old_knowledge"})

    async def _run_periodic_feedback_collection(self):
        """Periodically collects insights from agents and sends to ThinkTool."""
        while not shutdown_event.is_set():
            await asyncio.sleep(self.feedback_interval_seconds)
            if self.approved and not shutdown_event.is_set():
                logger.info("Orchestrator collecting feedback from agents...")
                all_insights = {}
                for agent_name, agent in self.agents.items():
                    if hasattr(agent, 'collect_insights') and callable(agent.collect_insights):
                        try:
                            insights = await agent.collect_insights()
                            if insights: all_insights[agent_name] = insights
                        except Exception as e: logger.error(f"Error collecting insights from {agent_name}: {e}")

                if all_insights:
                    logger.info(f"Sending collected insights from {len(all_insights)} agents to ThinkTool.")
                    await self.delegate_task("ThinkTool", {"action": "process_feedback", "feedback_data": all_insights})
                else: logger.info("No agent insights collected in this cycle.")

    async def _run_periodic_proxy_health_check(self):
         """Wrapper for the periodic proxy health check task."""
         while not shutdown_event.is_set():
             await asyncio.sleep(self.proxy_health_check_interval_seconds)
             if self.approved and not shutdown_event.is_set():
                 await self._check_proxy_health() # Call the actual check logic

    # --- Secure Storage Shim (Using DB Encryption) ---
    class _DatabaseSecureStorage:
        """Provides an interface similar to Vault/KMS using DB encryption."""
        def __init__(self, session_maker):
            self.session_maker = session_maker
            self.logger = logging.getLogger("SecureStorageShim")

        async def store_new_account(self, service: str, identifier: str, password: str, status: str = 'active', metadata: Optional[Dict] = None) -> Optional[Dict]:
            """Stores new account credentials securely in the database."""
            if not self.session_maker: self.logger.error("DB session_maker unavailable."); return None
            encrypted_password = encrypt_data(password)
            if not encrypted_password: self.logger.error("Failed to encrypt password."); return None
            notes_str = json.dumps(metadata) if metadata else None

            try:
                async with self.session_maker() as session:
                    async with session.begin():
                        stmt_check = select(AccountCredentials.id).where(AccountCredentials.service == service, AccountCredentials.account_identifier == identifier).limit(1)
                        existing = await session.scalar(stmt_check)
                        if existing:
                            self.logger.warning(f"Account {identifier} for {service} already exists (ID: {existing}). Updating status/notes.")
                            stmt_update = update(AccountCredentials).where(AccountCredentials.id == existing).values(status=status, notes=notes_str, last_used=None, password=encrypted_password) # Update password too if needed
                            await session.execute(stmt_update)
                            stmt_get = select(AccountCredentials.id, AccountCredentials.service, AccountCredentials.account_identifier, AccountCredentials.status).where(AccountCredentials.id == existing)
                            updated_details = (await session.execute(stmt_get)).mappings().first()
                            return dict(updated_details) if updated_details else None
                        else:
                            new_account = AccountCredentials(
                                service=service, account_identifier=identifier,
                                password=encrypted_password, status=status, notes=notes_str
                            )
                            session.add(new_account)
                            await session.flush()
                            account_id = new_account.id
                            self.logger.info(f"Stored new account credentials for {service}/{identifier} (ID: {account_id}).")
                            return {"id": account_id, "service": service, "account_identifier": identifier, "status": status}
            except Exception as e:
                self.logger.error(f"Error storing account credentials for {service}/{identifier}: {e}", exc_info=True)
                return None

        async def get_account_details_by_id(self, account_id: int) -> Optional[Dict]:
            """Fetches account details (excluding password) by ID."""
            if not self.session_maker: self.logger.error("DB session_maker unavailable."); return None
            try:
                async with self.session_maker() as session:
                    stmt = select(AccountCredentials).where(AccountCredentials.id == account_id)
                    account = await session.scalar(stmt)
                    if account:
                        return {c.name: getattr(account, c.name) for c in AccountCredentials.__table__.columns if c.name != 'password'}
                    else: return None
            except Exception as e: self.logger.error(f"Error fetching account details ID {account_id}: {e}"); return None

        async def find_active_account_for_service(self, service: str) -> Optional[Dict]:
            """Finds an active account for a service (excluding password)."""
            if not self.session_maker: self.logger.error("DB session_maker unavailable."); return None
            try:
                async with self.session_maker() as session:
                    stmt = select(AccountCredentials).where(
                        AccountCredentials.service == service,
                        AccountCredentials.status == 'active'
                    ).order_by(func.random()).limit(1) # Random selection for basic load balancing
                    account = await session.scalar(stmt)
                    if account:
                        async with session.begin(): # Use transaction for update
                                stmt_update = update(AccountCredentials).where(AccountCredentials.id == account.id).values(last_used=datetime.now(timezone.utc))
                                await session.execute(stmt_update)
                        return {c.name: getattr(account, c.name) for c in AccountCredentials.__table__.columns if c.name != 'password'}
                    else: return None
            except Exception as e: self.logger.error(f"Error finding active account for {service}: {e}"); return None

        async def get_secret(self, identifier: Union[int, str]) -> Optional[str]:
            """Retrieves and decrypts a secret (e.g., password) using account ID."""
            if not self.session_maker: self.logger.error("DB session_maker unavailable."); return None
            try:
                account_id = int(identifier) # Expect integer ID
                async with self.session_maker() as session:
                    stmt = select(AccountCredentials.password).where(AccountCredentials.id == account_id)
                    encrypted_pw = await session.scalar(stmt)
                    if encrypted_pw:
                        decrypted_pw = decrypt_data(encrypted_pw)
                        if decrypted_pw: return decrypted_pw
                        else: self.logger.error(f"Failed to decrypt password for account ID {account_id}."); return None
                    else: self.logger.warning(f"No password found for account ID {account_id}."); return None
            except ValueError: self.logger.error(f"Invalid identifier for get_secret (expected integer account ID): {identifier}"); return None
            except Exception as e: self.logger.error(f"Error retrieving secret for ID {identifier}: {e}"); return None

# --- End of agents/orchestrator.py ---