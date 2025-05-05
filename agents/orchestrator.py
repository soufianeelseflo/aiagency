# Filename: agents/orchestrator.py
# Description: Central coordinator for the AI Agency, managing core agents, workflows, and resources.
# Version: 3.1 (Added MailerSend Webhook Handler)

import os
import asyncio
from datetime import datetime, timedelta, timezone
import logging
import time
import json
import uuid
import random
import re
import base64  # For tracking pixel response
from collections import deque
from typing import Dict, Optional, Tuple, Any, List, AsyncGenerator, Callable, Type

# --- Core Framework Imports ---
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
# Make sure all needed SQLAlchemy components are imported
from sqlalchemy import select, delete, func, update, text, case, or_
from quart import Quart, request, jsonify, websocket, send_file, url_for, Response # Added Response
import psutil
from openai import AsyncOpenAI as AsyncLLMClient
import pybreaker
import websockets
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from prometheus_client import Counter, Gauge, start_http_server

# --- Project Imports ---
from config.settings import settings
from utils.database import (
    encrypt_data,
    decrypt_data,
    get_session,
    get_session_maker,
)
from utils.notifications import send_notification # Assuming this utility exists

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
from agents.gmail_creator_agent import GmailCreatorAgent # Import the new agent

# Import database models
from models import (
    Base, Client, Metric, ExpenseLog, MigrationStatus, KnowledgeFragment,
    AccountCredentials, CallLog, Invoice, StrategicDirective, PromptTemplate # Added PromptTemplate
)

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
op_logger = logging.getLogger("OperationalLog")

# --- Metrics & Circuit Breakers ---
agent_status_gauge = Gauge(
    "agent_status", "Status of agents (1=running, 0=stopped/error)", ["agent_name"]
)
error_counter = Counter(
    "agent_errors_total", "Total number of errors per agent", ["agent_name"]
)
tasks_processed_counter = Counter(
    "tasks_processed_total", "Total number of tasks processed by agent", ["agent_name"]
)
llm_client_breaker = pybreaker.CircuitBreaker(
    fail_max=3, reset_timeout=60 * 5, name="LLMClientBreaker"
)
# Add breakers for other external services if needed
# clay_breaker = pybreaker.CircuitBreaker(...)
# mailer_breaker = pybreaker.CircuitBreaker(...)


# --- Orchestrator Class ---
class Orchestrator:
    """Central coordinator, streamlined for core sales workflow and Postgres."""

    def __init__(self, schema="public"):
        self.config = settings
        self.session_maker = get_session_maker()
        self.agents: Dict[str, GeniusAgentBase] = {}
        self.app = Quart(
            __name__, template_folder="../ui/templates", static_folder="../ui/static"
        )
        self.setup_routes() # Call route setup here

        # Prometheus setup (keep existing)
        try:
            start_http_server(8001)
            logger.info("Prometheus metrics server started on port 8001.")
        except OSError as e: logger.warning(f"Could not start Prometheus server on port 8001: {e}")
        except Exception as e: logger.error(f"Failed to start Prometheus server: {e}", exc_info=True)

        self.meta_prompt = self.config.META_PROMPT
        self.approved = False

        # LLM Client Management State
        self._llm_client_cache: Dict[str, AsyncLLMClient] = {}
        self._llm_client_status: Dict[str, Dict[str, Any]] = {}
        self._llm_client_keys: List[str] = []
        self._llm_client_round_robin_index = 0

        # In-memory Cache (Simple)
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_ttl_default = 3600 # 1 hour default cache

        # Deepgram WebSocket Registry
        self.deepgram_connections: Dict[str, Any] = {}
        self.temp_audio_dir = self.config.TEMP_AUDIO_DIR # Use path from settings
        os.makedirs(self.temp_audio_dir, exist_ok=True)

        # Periodic Task Timing
        self.feedback_interval_seconds: int = int(self.config.THINKTOOL_FEEDBACK_INTERVAL_SECONDS)
        self.purge_interval_seconds: int = int(self.config.DATA_PURGE_INTERVAL_SECONDS)
        self.last_feedback_time: float = 0.0
        self.last_purge_time: float = 0.0

        self.running: bool = False
        self.background_tasks = set()
        self.status = "initializing" # Add status attribute

        # Secure Storage Interface (Conceptual - replace with actual Vault/KMS if needed)
        # For now, it uses the DB with encryption
        self.secure_storage = self._DatabaseSecureStorage(self.session_maker)

        logger.info("Orchestrator v3.1 (Webhook Handler) initialized.")
        # Initialization sequence called from run()

    # --- Initialization Methods ---
    async def initialize_database(self):
        """Initialize or update the database schema."""
        logger.info("Initializing database schema...")
        try:
            engine = create_async_engine(self.config.DATABASE_URL.unicode_string(), echo=False) # Use unicode_string() for Pydantic v2 DSN
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            await engine.dispose()
            logger.info("Database schema initialization complete.")
            return True
        except Exception as e:
            logger.critical(f"Failed to initialize database: {e}", exc_info=True)
            return False

    async def initialize_clients(self):
        """Initialize LLM API clients from settings (env vars)."""
        logger.info("Initializing LLM clients...")
        self._llm_client_keys = []
        self._llm_client_cache = {}
        self._llm_client_status = {}
        keys_found = []

        # Load primary OpenRouter key
        primary_key = self.config.get_secret("OPENROUTER_API_KEY")
        if primary_key: keys_found.append(("OR_Primary", primary_key))

        # Load additional keys
        i = 1
        while True:
            add_key = self.config.get_secret(f"OPENROUTER_API_KEY_{i}")
            if not add_key: break
            keys_found.append((f"OR_Extra{i}", add_key))
            i += 1

        for key_name_prefix, api_key in keys_found:
            try:
                key_id = f"{key_name_prefix}_{api_key[-4:]}"
                if key_id not in self._llm_client_cache:
                    self._llm_client_cache[key_id] = AsyncLLMClient(api_key=api_key, base_url="https://openrouter.ai/api/v1")
                    self._llm_client_status[key_id] = {"status": "available", "reason": None, "unavailable_until": 0}
                    self._llm_client_keys.append(key_id)
                    logger.info(f"Initialized OpenRouter client (Key ID: {key_id}).")
            except Exception as e: logger.error(f"Failed to initialize OpenRouter client {key_name_prefix}: {e}", exc_info=True)

        if not self._llm_client_keys: logger.critical("CRITICAL: No LLM API clients initialized."); return False
        logger.info(f"LLM Client Initialization complete. Total usable keys: {len(self._llm_client_keys)}")
        return True

    async def initialize_agents(self):
        """Initialize the CORE agents + kept agents, passing necessary secrets."""
        logger.info("Initializing CORE agents...")
        initialization_failed = False
        try:
            # ThinkTool
            self.agents["think"] = ThinkTool(self.session_maker, self.config, self)

            # BrowsingAgent
            smartproxy_pass = self.config.get_secret("SMARTPROXY_PASSWORD")
            self.agents["browsing"] = BrowsingAgent(self.session_maker, self, smartproxy_password=smartproxy_pass)

            # EmailAgent
            imap_pass = self.config.get_secret("HOSTINGER_IMAP_PASS")
            if not imap_pass: logger.critical("Hostinger IMAP password missing. EmailAgent reply checking will fail."); initialization_failed = True
            self.agents["email"] = EmailAgent(self.session_maker, self, imap_password=imap_pass) # Only pass IMAP pass

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


            # Set Prometheus gauges
            for name in self.agents.keys(): agent_status_gauge.labels(agent_name=name).set(0) # idle

            if initialization_failed: raise RuntimeError("Failed to initialize critical agents due to missing secrets.")

            logger.info(f"Core agents initialized: {list(self.agents.keys())}")
            return True

        except Exception as e: logger.error(f"Core agent initialization failed: {e}", exc_info=True); return False

    # --- LLM Client Management ---
    @llm_client_breaker
    async def get_available_llm_clients(self) -> List[AsyncLLMClient]:
        # ... (Implementation remains the same as v3.0) ...
        now = time.time(); available_keys = []
        for key_id in self._llm_client_keys:
            status_info = self._llm_client_status.get(key_id, {"status": "available", "unavailable_until": 0})
            if status_info["status"] == "available" or now >= status_info.get("unavailable_until", 0):
                if status_info["status"] == "unavailable": status_info["status"] = "available"; status_info["reason"] = None; status_info["unavailable_until"] = 0; logger.info(f"LLM client key {key_id} available after cooldown.")
                available_keys.append(key_id)
        if not available_keys: logger.error("No available LLM clients!"); return []
        selected_key_id = available_keys[self._llm_client_round_robin_index % len(available_keys)]; self._llm_client_round_robin_index += 1
        selected_client = self._llm_client_cache.get(selected_key_id)
        if not selected_client:
            logger.error(f"Client instance not found for key ID {selected_key_id}. Init issue.");
            if selected_key_id in self._llm_client_keys: self._llm_client_keys.remove(selected_key_id)
            self._llm_client_status.pop(selected_key_id, None); self._llm_client_cache.pop(selected_key_id, None)
            return await self.get_available_llm_clients() # Retry
        logger.debug(f"Selected LLM client key ID: {selected_key_id}")
        return [selected_client]


    async def report_client_issue(self, api_key_identifier: str, issue_type: str):
        # ... (Implementation remains the same as v3.0) ...
        if api_key_identifier not in self._llm_client_status: logger.warning(f"Attempted report issue for unknown LLM client: {api_key_identifier}"); return
        now = time.time(); cooldown_seconds = 60 * 5; status = "unavailable"; reason = issue_type
        if issue_type == "auth_error": cooldown_seconds = 60 * 60 * 24 * 365; reason = "Authentication Error"; logger.critical(f"LLM client key {api_key_identifier} marked permanently unavailable (auth error).")
        elif issue_type == "rate_limit": cooldown_seconds = 60 * 2; reason = "Rate Limited"
        elif issue_type == "timeout_error": cooldown_seconds = 60 * 3; reason = "Timeout"
        else: reason = "General Error"
        self._llm_client_status[api_key_identifier] = {"status": status, "reason": reason, "unavailable_until": now + cooldown_seconds}
        logger.warning(f"LLM client key {api_key_identifier} marked unavailable until {datetime.fromtimestamp(now + cooldown_seconds)}. Reason: {reason}")

    # MODIFIED: Added image_data parameter
    @llm_client_breaker
    async def call_llm(
        self,
        agent_name: str,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        is_json_output: bool = False,
        model_preference: Optional[List[str]] = None,
        image_data: Optional[bytes] = None, # Added for multimodal
        timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]: # Return dict including content and usage
        """Handles selecting an available client and making the LLM call, now with multimodal support."""
        selected_clients = await self.get_available_llm_clients()
        if not selected_clients: logger.error(f"Agent '{agent_name}' failed to get an available LLM client."); return None
        llm_client = selected_clients[0]; api_key_identifier = "unknown"
        for key_id, client_instance in self._llm_client_cache.items():
            if client_instance == llm_client: api_key_identifier = key_id; break

        # Determine model
        # Prioritize model_preference if provided and valid
        model_name = None
        if model_preference:
            for pref in model_preference:
                if pref in self.config.OPENROUTER_MODELS.values(): # Check if it's a known valid model value
                     model_name = pref
                     break
                 elif pref in self.config.OPENROUTER_MODELS: # Check if it's a known key
                      model_name = self.config.OPENROUTER_MODELS[pref]
                      break
            if model_name: logger.debug(f"Using preferred model: {model_name}")
            else: logger.warning(f"Model preference {model_preference} not found in config, using default.")

        # Fallback to agent-specific or default model
        if not model_name:
            model_key = "default_llm"
            # Agent specific model selection logic...
            if agent_name == "ThinkTool": model_key = "think_general"
            elif agent_name == "EmailAgent": model_key = "email_draft"
            elif agent_name == "VoiceSalesAgent": model_key = "voice_response"
            elif agent_name == "LegalAgent": model_key = "legal_validation"
            elif agent_name == "BrowsingAgent": model_key = "browsing_visual_analysis" if image_data else "browsing_summarize" # Choose visual model if image provided
            # ... add other agents
            model_name = self.config.OPENROUTER_MODELS.get(model_key, self.config.OPENROUTER_MODELS["default_llm"])

        # Caching Logic (remains the same)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        cache_key_parts = ["llm_call", prompt_hash, model_name, str(temperature), str(max_tokens), str(is_json_output), str(image_data is not None)] # Added image flag to key
        cache_key = ":".join(cache_key_parts); cache_ttl = self.cache_ttl_default
        cached_result = self.get_from_cache(cache_key)
        if cached_result is not None: logger.debug(f"LLM call cache hit (Orchestrator) for key: {cache_key[:20]}..."); return cached_result # Return cached dict

        try:
            response_format = {"type": "json_object"} if is_json_output else None
            # --- MODIFIED: Construct messages for multimodal ---
            messages = []
            content_parts = [{"type": "text", "text": prompt}]
            if image_data:
                # Encode image to base64
                base64_image = base64.b64encode(image_data).decode('utf-8')
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}" # Assuming PNG, adjust if needed
                    }
                })
                logger.debug("Image data included in LLM request.")
            messages.append({"role": "user", "content": content_parts})
            # --- END MODIFICATION ---

            logger.debug(f"Orchestrator making LLM Call: Agent={agent_name}, Model={model_name}, Key=...{api_key_identifier[-4:]}, Multimodal={image_data is not None}")

            api_timeout = timeout or settings.OPENROUTER_API_TIMEOUT_S or 120.0
            response = await llm_client.chat.completions.create(
                model=model_name, messages=messages, temperature=temperature,
                max_tokens=max_tokens, response_format=response_format, timeout=api_timeout,
            )
            content = response.choices[0].message.content.strip() if response.choices[0].message.content else None

            # Track usage & cost
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = input_tokens + output_tokens
            # TODO: Get actual costs per model from OpenRouter or estimate
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
            logger.warning(f"Orchestrator LLM call failed: Agent={agent_name}, Model={model_name}, Key=...{api_key_identifier[-4:]}, Type={issue_type}, Error={e}")
            await self.report_client_issue(api_key_identifier, issue_type)
            raise # Re-raise for tenacity/caller handling

    # --- Route Setup ---
    def setup_routes(self):
        """Configure Quart routes for UI interaction and webhooks."""

        @self.app.route("/")
        async def index():
            # ... (index route handler remains the same) ...
            try:
                template_path = os.path.join(os.path.dirname(__file__), "..", "ui", "templates", "index.html")
                if os.path.exists(template_path):
                    with open(template_path, "r") as f: return f.read(), 200, {"Content-Type": "text/html"}
                else: logger.error(f"UI Template not found at {template_path}"); return "UI Template not found.", 404
            except Exception as e: logger.error(f"Error serving index.html: {e}", exc_info=True); return "Internal Server Error", 500

        @self.app.route("/api/approve", methods=["POST"])
        async def approve():
            # ... (approve route handler remains the same) ...
            if self.approved: return jsonify({"status": "already_approved"}), 200
            self.approved = True; logger.info("!!! AGENCY APPROVED FOR FULL OPERATION via API !!!")
            await self.send_notification("Agency Approved", "Agency approved via API.")
            return jsonify({"status": "approved"})

        @self.app.route("/api/status", methods=["GET"])
        async def api_status():
             # ... (api_status route handler remains the same) ...
            agent_statuses = { name: agent.get_status_summary() for name, agent in self.agents.items() if hasattr(agent, "get_status_summary") }
            llm_status = { key: info["status"] for key, info in self._llm_client_status.items() }
            return jsonify({ "orchestrator_status": self.status if hasattr(self, "status") else "unknown", "approved_for_operation": self.approved, "agent_statuses": agent_statuses, "llm_client_status": llm_status, })

        @self.app.route("/api/start_ugc", methods=["POST"])
        async def handle_start_ugc():
             # ... (handle_start_ugc route handler remains the same) ...
            try:
                data = await request.get_json(); client_industry = data.get("client_industry")
                if not client_industry: return jsonify({"error": "Missing 'client_industry'"}), 400
                task = { "action": "plan_ugc_workflow", "client_industry": client_industry, "num_videos": int(data.get("num_videos", 1)), "initial_script": data.get("script"), "target_services": data.get("target_services"), }
                await self.delegate_task("ThinkTool", task)
                return jsonify({ "status": "UGC workflow planning initiated via ThinkTool" }), 202,
            except Exception as e: logger.error(f"Failed initiate UGC workflow planning: {e}", exc_info=True); return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route("/hosted_audio/<path:filename>")
        async def serve_hosted_audio(filename):
             # ... (serve_hosted_audio route handler remains the same) ...
            if ".." in filename or filename.startswith("/"): return "Forbidden", 403
            try:
                safe_path = os.path.join(self.temp_audio_dir, filename)
                if os.path.exists(safe_path) and os.path.isfile(safe_path): mimetype = ("audio/wav" if filename.lower().endswith(".wav") else "audio/mpeg"); return await send_file(safe_path, mimetype=mimetype)
                else: return jsonify({"error": "File not found"}), 404
            except Exception as e: logger.error(f"Error serving hosted audio {filename}: {e}"); return jsonify({"error": "Internal server error"}), 500

        @self.app.websocket("/twilio_call")
        async def handle_twilio_websocket():
             # ... (handle_twilio_websocket remains the same) ...
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
                        if payload: audio_bytes = base64.b64decode(payload); await deepgram_live_client.send(audio_bytes)
                    elif event == "stop": logger.info(f"Twilio WS: Stop event for {call_sid}."); break
                    elif event == "error": logger.error(f"Twilio WS: Error event for {call_sid}: {data.get('error')}"); break
            except websockets.exceptions.ConnectionClosedOK: logger.info(f"Twilio WS: Connection closed normally for {call_sid}.")
            except Exception as e: logger.error(f"Twilio WS: Unexpected error for {call_sid}: {e}", exc_info=True,)
            finally: logger.info(f"Twilio WS: Cleaning up for {call_sid}");
            if call_sid: await self.unregister_deepgram_connection_sdk(call_sid)

        @self.app.route("/track/<tracking_id>.png")
        async def handle_tracking_pixel(tracking_id):
             # ... (handle_tracking_pixel remains the same) ...
            logger.info(f"Tracking pixel hit: {tracking_id}")
            email_agent = self.agents.get("email");
            if email_agent and hasattr(email_agent, "process_email_open"): asyncio.create_task(email_agent.process_email_open(tracking_id))
            else: logger.warning(f"EmailAgent not found or cannot process opens for tracking ID: {tracking_id}")
            pixel_data = base64.b64decode("R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==")
            return (pixel_data, 200, {"Content-Type": "image/gif", "Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0",})

        # --- ADDED: MailerSend Webhook Handler ---
        @self.app.route("/webhooks/mailersend", methods=["POST"])
        async def handle_mailersend_webhook():
            """Handles incoming webhooks from MailerSend for bounces, complaints, etc."""
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

                if event_type in ['activity.hard_bounced', 'activity.spam_complaint']:
                    update_values['is_deliverable'] = False
                    update_values['opt_in'] = False # Treat hard bounce/spam complaint as opt-out
                    log_op_level = 'warning'
                    log_op_message = f"Marking {recipient_email} undeliverable/opt-out due to MailerSend webhook: {event_type}."
                elif event_type == 'activity.unsubscribed':
                    update_values['opt_in'] = False
                    log_op_level = 'warning'
                    log_op_message = f"Marking {recipient_email} opted-out due to MailerSend webhook: {event_type}."

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
                    if hasattr(self, 'log_operation'): # Check if log_operation exists
                         await self.log_operation(log_op_level, log_op_message)
                    else: # Fallback logging
                         log_func = getattr(logger, log_op_level, logger.info)
                         log_func(f"[Orchestrator Webhook] {log_op_message}")


                # Acknowledge receipt to MailerSend quickly
                return jsonify({"status": "received"}), 200

            except Exception as e:
                logger.error(f"Error processing MailerSend webhook: {e}", exc_info=True)
                # Still return 200 to MailerSend if possible, to avoid retries,
                # unless it's a fundamental error processing the request itself.
                return jsonify({"status": "error", "message": "Internal server error"}), 500 # Or 200 if you want MailerSend to stop retrying
        # --- END ADDED ---

        logger.info("Quart routes configured.")

    # --- Main Execution Loop ---
    async def run(self):
        """Initializes and runs the AI Agency Orchestrator."""
        logger.info("Orchestrator starting full initialization sequence...")
        self.running = False
        self.status = "initializing" # Set initial status
        try:
            if not await self.initialize_database(): raise RuntimeError("Database initialization failed.")
            if not await self.initialize_clients(): raise RuntimeError("LLM Client initialization failed.")
            if not await self.initialize_agents(): raise RuntimeError("Agent initialization failed.")
            logger.info("Orchestrator initialization complete.")

            logger.info("Starting background agent run loops and periodic tasks...")
            self.background_tasks = set()
            for agent_name, agent in self.agents.items():
                if hasattr(agent, "run") and callable(agent.run):
                    task = asyncio.create_task(agent.run(), name=f"AgentLoop_{agent_name}")
                    self.background_tasks.add(task)
                # --- MODIFIED: Start agent if it has start() method ---
                elif hasattr(agent, "start") and callable(agent.start):
                     task = asyncio.create_task(agent.start(), name=f"AgentStart_{agent_name}")
                     self.background_tasks.add(task) # Track start task if needed, or just launch
                     logger.info(f"Called start() for agent {agent_name}")
                # --- END MODIFICATION ---
                else: logger.warning(f"Agent {agent_name} does not have a callable run or start method.")

            self.background_tasks.add(asyncio.create_task(self._run_periodic_data_purge(), name="PeriodicDataPurge"))
            self.background_tasks.add(asyncio.create_task(self._run_periodic_feedback_collection(), name="PeriodicFeedback"))
            logger.info(f"Started {len(self.background_tasks)} background tasks.")

            logger.info("Orchestrator entering main operational state (API/Event driven).")
            self.running = True
            self.status = "running" # Update status
            self.last_feedback_time = time.time()
            self.last_purge_time = time.time()

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

        except (ValueError, RuntimeError) as init_err:
             logger.critical(f"Fatal Error: Orchestrator initialization failed: {init_err}", exc_info=True)
             self.status = "failed_initialization"
        except Exception as e:
             logger.critical(f"Fatal Error: Unhandled exception during Orchestrator run setup: {e}", exc_info=True)
             self.status = "error"
        finally:
            # Graceful Shutdown
            logger.info("Initiating graceful shutdown sequence...")
            self.running = False
            self.status = "stopping"

            # Cancel any remaining background tasks
            all_tasks = list(self.background_tasks) # Copy set before iterating
            if all_tasks:
                logger.info(f"Cancelling {len(all_tasks)} running background tasks...")
                for task in all_tasks:
                    if task and not task.done(): task.cancel()
                await asyncio.gather(*all_tasks, return_exceptions=True)
                logger.info("Background tasks cancellation complete.")

            # Explicitly stop agents
            agent_stop_tasks = []
            for agent_name, agent in self.agents.items():
                 if hasattr(agent, 'stop') and callable(agent.stop):
                      # Check agent status before stopping
                      agent_status = getattr(agent, 'status', 'unknown')
                      if agent_status not in [agent.STATUS_STOPPING, agent.STATUS_STOPPED]:
                           logger.info(f"Calling stop() for agent {agent_name}...")
                           agent_stop_tasks.append(asyncio.create_task(agent.stop(timeout=25.0)))
                      else:
                           logger.info(f"Agent {agent_name} already stopping/stopped ({agent_status}). Skipping stop call.")
                 else: logger.warning(f"Agent {agent_name} has no stop method.")

            if agent_stop_tasks:
                 logger.info(f"Waiting for {len(agent_stop_tasks)} agents to stop...")
                 await asyncio.gather(*agent_stop_tasks, return_exceptions=True)
                 logger.info("Agent stop sequence complete.")

            self.status = "stopped"
            logger.info("-------------------- Application Stopping --------------------")
            logging.shutdown() # Ensure all logs are flushed before exiting
            print("[INFO] Orchestrator: Process stopped.") # Final print statement

    # --- Agent Interaction & Tooling ---
    async def delegate_task(self, agent_name: str, task_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Delegates a task to a specific agent."""
        agent = self.agents.get(agent_name)
        if not agent:
            logger.error(f"Delegation failed: Agent '{agent_name}' not found.")
            return {"status": "error", "message": f"Agent '{agent_name}' not found."}

        if not hasattr(agent, 'execute_task') or not callable(agent.execute_task):
             logger.error(f"Delegation failed: Agent '{agent_name}' has no callable execute_task method.")
             return {"status": "error", "message": f"Agent '{agent_name}' cannot execute tasks."}

        task_id = task_details.get('id', str(uuid.uuid4()))
        task_details['id'] = task_id # Ensure task has an ID
        logger.info(f"Delegating task {task_id} ({task_details.get('action', 'N/A')}) to {agent_name}.")
        try:
            # Use asyncio.create_task for true background execution if execute_task is long-running
            # result = await asyncio.create_task(agent.execute_task(task_details))
            # For now, await directly, assuming execute_task handles backgrounding if needed
            result = await agent.execute_task(task_details)
            tasks_processed_counter.labels(agent_name=agent_name).inc()
            return result
        except Exception as e:
            logger.error(f"Error during task delegation to {agent_name} (Task ID: {task_id}): {e}", exc_info=True)
            error_counter.labels(agent_name=agent_name).inc()
            await self.report_error(agent_name, f"Task delegation failed: {e}", task_id)
            return {"status": "error", "message": f"Exception during task execution: {e}"}

    async def use_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Allows agents to request tool usage via the orchestrator."""
        # This is a placeholder - expand with actual tool implementations
        logger.info(f"Orchestrator received tool use request: {tool_name} with params: {params}")
        if tool_name == "read_file":
            file_path = params.get('path')
            if file_path and os.path.exists(file_path) and os.path.isfile(file_path):
                 try:
                      with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                           content = f.read()
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
        else:
            logger.warning(f"Attempted to use unknown tool: {tool_name}")
            return {"status": "failure", "message": f"Unknown tool: {tool_name}"}

    async def report_error(self, agent_name: str, error_message: str, task_id: Optional[str] = None):
        """Handles error reporting from agents."""
        log_msg = f"ERROR reported by {agent_name}: {error_message}"
        if task_id: log_msg += f" (Task: {task_id})"
        logger.error(log_msg)
        error_counter.labels(agent_name=agent_name).inc()
        # Optionally send notification on critical errors
        # await self.send_notification(f"Agent Error: {agent_name}", log_msg)

    async def send_notification(self, title: str, message: str, level: str = "info"):
        """Sends notifications (e.g., email, Slack) using the utility function."""
        # Use the imported send_notification function
        await send_notification(title, message, level, self.config)

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
            if time.time() < expires_at:
                logger.debug(f"Cache hit for key: {key[:20]}...")
                return value
            else:
                logger.debug(f"Cache expired for key: {key[:20]}...")
                del self._cache[key] # Remove expired item
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
            del self.deepgram_connections[call_sid]
            logger.info(f"Unregistered Deepgram SDK client for Call SID: {call_sid}")

    # --- Temporary Audio Hosting ---
    async def host_temporary_audio(self, audio_data: bytes, filename: str) -> Optional[str]:
         """Saves audio data temporarily and returns a publicly accessible URL."""
         try:
             # Ensure filename is safe
             safe_filename = re.sub(r'[^\w\.\-]', '_', filename)
             filepath = os.path.join(self.temp_audio_dir, safe_filename)
             with open(filepath, 'wb') as f:
                 f.write(audio_data)
             # Construct URL using AGENCY_BASE_URL and the Quart route
             # Use url_for if Quart context is available, otherwise construct manually
             base_url = self.config.AGENCY_BASE_URL.rstrip('/')
             audio_url = f"{base_url}/hosted_audio/{safe_filename}"
             logger.info(f"Hosted temporary audio at: {audio_url}")
             # TODO: Add cleanup mechanism for old audio files
             return audio_url
         except Exception as e:
             logger.error(f"Failed to host temporary audio {filename}: {e}", exc_info=True)
             return None

    # --- Invoice Generation Request ---
    async def request_invoice_generation(self, client_id: int, amount: float, source_call_sid: str):
         """Sends a directive to ThinkTool/LegalAgent to generate an invoice."""
         logger.info(f"Requesting invoice generation for Client {client_id}, Amount ${amount:.2f}, Source Call {source_call_sid}")
         # Decide which agent handles invoice creation (e.g., ThinkTool plans, Legal adds notes?)
         # For now, send directive to ThinkTool to coordinate
         directive_content = {
             "client_id": client_id,
             "amount": amount,
             "source_reference": f"CallLog:{source_call_sid}",
             "due_date_days": 14 # Example: Due in 14 days
         }
         await self.delegate_task("ThinkTool", { # Or maybe LegalAgent?
             "action": "generate_invoice", # Define this action
             "content": directive_content,
             "priority": 3 # High priority after successful call
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
                        # Check if account already exists
                        stmt_check = select(AccountCredentials.id).where(AccountCredentials.service == service, AccountCredentials.account_identifier == identifier).limit(1)
                        existing = await session.scalar(stmt_check)
                        if existing:
                             self.logger.warning(f"Account {identifier} for {service} already exists (ID: {existing}). Updating status/notes.")
                             stmt_update = update(AccountCredentials).where(AccountCredentials.id == existing).values(status=status, notes=notes_str, last_used=None) # Reset last_used?
                             await session.execute(stmt_update)
                             # Fetch updated details (excluding password)
                             stmt_get = select(AccountCredentials.id, AccountCredentials.service, AccountCredentials.account_identifier, AccountCredentials.status).where(AccountCredentials.id == existing)
                             updated_details = (await session.execute(stmt_get)).mappings().first()
                             return dict(updated_details) if updated_details else None
                        else:
                             new_account = AccountCredentials(
                                 service=service,
                                 account_identifier=identifier,
                                 password=encrypted_password, # Store encrypted password
                                 status=status,
                                 notes=notes_str
                             )
                             session.add(new_account)
                             await session.flush() # Get the ID before commit finishes
                             account_id = new_account.id
                             self.logger.info(f"Stored new account credentials for {service}/{identifier} (ID: {account_id}).")
                             # Return details (excluding password)
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
                           # Return as dict, excluding password
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
                      ).order_by(func.random()).limit(1) # Random selection for basic rotation
                      account = await session.scalar(stmt)
                      if account:
                           # Update last used time
                           async with session.begin(): # Start new transaction for update
                                account.last_used = datetime.now(timezone.utc)
                                await session.merge(account)
                           # Return details excluding password
                           return {c.name: getattr(account, c.name) for c in AccountCredentials.__table__.columns if c.name != 'password'}
                      else: return None
             except Exception as e: self.logger.error(f"Error finding active account for {service}: {e}"); return None

        async def get_secret(self, identifier: str) -> Optional[str]:
             """Retrieves and decrypts a secret (e.g., password) using identifier."""
             # This assumes identifier uniquely maps to one secret (e.g., account ID or vault path)
             # For account passwords, use identifier = account_id
             if not self.session_maker: self.logger.error("DB session_maker unavailable."); return None
             try:
                 account_id = int(identifier) # Assume identifier is account ID for now
                 async with self.session_maker() as session:
                      stmt = select(AccountCredentials.password).where(AccountCredentials.id == account_id)
                      encrypted_pw = await session.scalar(stmt)
                      if encrypted_pw:
                           decrypted_pw = decrypt_data(encrypted_pw)
                           if decrypted_pw: return decrypted_pw
                           else: self.logger.error(f"Failed to decrypt password for account ID {account_id}."); return None
                      else: self.logger.warning(f"No password found for account ID {account_id}."); return None
             except ValueError: self.logger.error(f"Invalid identifier for get_secret (expected account ID): {identifier}"); return None
             except Exception as e: self.logger.error(f"Error retrieving secret for ID {identifier}: {e}"); return None

# --- End of agents/orchestrator.py ---
</o>```python
# Filename: agents/orchestrator.py
# Description: Central coordinator for the AI Agency, managing core agents, workflows, and resources.
# Version: 3.1 (Added MailerSend Webhook Handler)

import os
import asyncio
from datetime import datetime, timedelta, timezone
import logging
import time
import json
import uuid
import random
import re
import base64  # For tracking pixel response
from collections import deque
from typing import Dict, Optional, Tuple, Any, List, AsyncGenerator, Callable, Type

# --- Core Framework Imports ---
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
# Make sure all needed SQLAlchemy components are imported
from sqlalchemy import select, delete, func, update, text, case, or_
from quart import Quart, request, jsonify, websocket, send_file, url_for, Response # Added Response
import psutil
from openai import AsyncOpenAI as AsyncLLMClient
import pybreaker
import websockets
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from prometheus_client import Counter, Gauge, start_http_server

# --- Project Imports ---
from config.settings import settings
from utils.database import (
    encrypt_data,
    decrypt_data,
    get_session,
    get_session_maker,
)
from utils.notifications import send_notification # Assuming this utility exists

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
from agents.gmail_creator_agent import GmailCreatorAgent # Import the new agent

# Import database models
from models import (
    Base, Client, Metric, ExpenseLog, MigrationStatus, KnowledgeFragment,
    AccountCredentials, CallLog, Invoice, StrategicDirective, PromptTemplate # Added PromptTemplate
)

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
op_logger = logging.getLogger("OperationalLog")

# --- Metrics & Circuit Breakers ---
agent_status_gauge = Gauge(
    "agent_status", "Status of agents (1=running, 0=stopped/error)", ["agent_name"]
)
error_counter = Counter(
    "agent_errors_total", "Total number of errors per agent", ["agent_name"]
)
tasks_processed_counter = Counter(
    "tasks_processed_total", "Total number of tasks processed by agent", ["agent_name"]
)
llm_client_breaker = pybreaker.CircuitBreaker(
    fail_max=3, reset_timeout=60 * 5, name="LLMClientBreaker"
)
# Add breakers for other external services if needed
# clay_breaker = pybreaker.CircuitBreaker(...)
# mailer_breaker = pybreaker.CircuitBreaker(...)


# --- Orchestrator Class ---
class Orchestrator:
    """Central coordinator, streamlined for core sales workflow and Postgres."""

    def __init__(self, schema="public"):
        self.config = settings
        self.session_maker = get_session_maker()
        self.agents: Dict[str, GeniusAgentBase] = {}
        self.app = Quart(
            __name__, template_folder="../ui/templates", static_folder="../ui/static"
        )
        self.setup_routes() # Call route setup here

        # Prometheus setup (keep existing)
        try:
            start_http_server(8001)
            logger.info("Prometheus metrics server started on port 8001.")
        except OSError as e: logger.warning(f"Could not start Prometheus server on port 8001: {e}")
        except Exception as e: logger.error(f"Failed to start Prometheus server: {e}", exc_info=True)

        self.meta_prompt = self.config.META_PROMPT
        self.approved = False

        # LLM Client Management State
        self._llm_client_cache: Dict[str, AsyncLLMClient] = {}
        self._llm_client_status: Dict[str, Dict[str, Any]] = {}
        self._llm_client_keys: List[str] = []
        self._llm_client_round_robin_index = 0

        # In-memory Cache (Simple)
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_ttl_default = 3600 # 1 hour default cache

        # Deepgram WebSocket Registry
        self.deepgram_connections: Dict[str, Any] = {}
        self.temp_audio_dir = self.config.TEMP_AUDIO_DIR # Use path from settings
        os.makedirs(self.temp_audio_dir, exist_ok=True)

        # Periodic Task Timing
        self.feedback_interval_seconds: int = int(self.config.THINKTOOL_FEEDBACK_INTERVAL_SECONDS)
        self.purge_interval_seconds: int = int(self.config.DATA_PURGE_INTERVAL_SECONDS)
        self.last_feedback_time: float = 0.0
        self.last_purge_time: float = 0.0

        self.running: bool = False
        self.background_tasks = set()
        self.status = "initializing" # Add status attribute

        # Secure Storage Interface (Conceptual - replace with actual Vault/KMS if needed)
        # For now, it uses the DB with encryption
        self.secure_storage = self._DatabaseSecureStorage(self.session_maker)

        logger.info("Orchestrator v3.1 (Webhook Handler) initialized.")
        # Initialization sequence called from run()

    # --- Initialization Methods ---
    async def initialize_database(self):
        """Initialize or update the database schema."""
        logger.info("Initializing database schema...")
        try:
            engine = create_async_engine(self.config.DATABASE_URL.unicode_string(), echo=False) # Use unicode_string() for Pydantic v2 DSN
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            await engine.dispose()
            logger.info("Database schema initialization complete.")
            return True
        except Exception as e:
            logger.critical(f"Failed to initialize database: {e}", exc_info=True)
            return False

    async def initialize_clients(self):
        """Initialize LLM API clients from settings (env vars)."""
        logger.info("Initializing LLM clients...")
        self._llm_client_keys = []
        self._llm_client_cache = {}
        self._llm_client_status = {}
        keys_found = []

        # Load primary OpenRouter key
        primary_key = self.config.get_secret("OPENROUTER_API_KEY")
        if primary_key: keys_found.append(("OR_Primary", primary_key))

        # Load additional keys
        i = 1
        while True:
            add_key = self.config.get_secret(f"OPENROUTER_API_KEY_{i}")
            if not add_key: break
            keys_found.append((f"OR_Extra{i}", add_key))
            i += 1

        for key_name_prefix, api_key in keys_found:
            try:
                key_id = f"{key_name_prefix}_{api_key[-4:]}"
                if key_id not in self._llm_client_cache:
                    self._llm_client_cache[key_id] = AsyncLLMClient(api_key=api_key, base_url="https://openrouter.ai/api/v1")
                    self._llm_client_status[key_id] = {"status": "available", "reason": None, "unavailable_until": 0}
                    self._llm_client_keys.append(key_id)
                    logger.info(f"Initialized OpenRouter client (Key ID: {key_id}).")
            except Exception as e: logger.error(f"Failed to initialize OpenRouter client {key_name_prefix}: {e}", exc_info=True)

        if not self._llm_client_keys: logger.critical("CRITICAL: No LLM API clients initialized."); return False
        logger.info(f"LLM Client Initialization complete. Total usable keys: {len(self._llm_client_keys)}")
        return True

    async def initialize_agents(self):
        """Initialize the CORE agents + kept agents, passing necessary secrets."""
        logger.info("Initializing CORE agents...")
        initialization_failed = False
        try:
            # ThinkTool
            self.agents["think"] = ThinkTool(self.session_maker, self.config, self)

            # BrowsingAgent
            smartproxy_pass = self.config.get_secret("SMARTPROXY_PASSWORD")
            self.agents["browsing"] = BrowsingAgent(self.session_maker, self, smartproxy_password=smartproxy_pass)

            # EmailAgent
            imap_pass = self.config.get_secret("HOSTINGER_IMAP_PASS")
            if not imap_pass: logger.critical("Hostinger IMAP password missing. EmailAgent reply checking will fail."); initialization_failed = True
            self.agents["email"] = EmailAgent(self.session_maker, self, imap_password=imap_pass) # Only pass IMAP pass

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


            # Set Prometheus gauges
            for name in self.agents.keys(): agent_status_gauge.labels(agent_name=name).set(0) # idle

            if initialization_failed: raise RuntimeError("Failed to initialize critical agents due to missing secrets.")

            logger.info(f"Core agents initialized: {list(self.agents.keys())}")
            return True

        except Exception as e: logger.error(f"Core agent initialization failed: {e}", exc_info=True); return False

    # --- LLM Client Management ---
    @llm_client_breaker
    async def get_available_llm_clients(self) -> List[AsyncLLMClient]:
        # ... (Implementation remains the same as v3.0) ...
        now = time.time(); available_keys = []
        for key_id in self._llm_client_keys:
            status_info = self._llm_client_status.get(key_id, {"status": "available", "unavailable_until": 0})
            if status_info["status"] == "available" or now >= status_info.get("unavailable_until", 0):
                if status_info["status"] == "unavailable": status_info["status"] = "available"; status_info["reason"] = None; status_info["unavailable_until"] = 0; logger.info(f"LLM client key {key_id} available after cooldown.")
                available_keys.append(key_id)
        if not available_keys: logger.error("No available LLM clients!"); return []
        selected_key_id = available_keys[self._llm_client_round_robin_index % len(available_keys)]; self._llm_client_round_robin_index += 1
        selected_client = self._llm_client_cache.get(selected_key_id)
        if not selected_client:
            logger.error(f"Client instance not found for key ID {selected_key_id}. Init issue.");
            if selected_key_id in self._llm_client_keys: self._llm_client_keys.remove(selected_key_id)
            self._llm_client_status.pop(selected_key_id, None); self._llm_client_cache.pop(selected_key_id, None)
            return await self.get_available_llm_clients() # Retry
        logger.debug(f"Selected LLM client key ID: {selected_key_id}")
        return [selected_client]


    async def report_client_issue(self, api_key_identifier: str, issue_type: str):
        # ... (Implementation remains the same as v3.0) ...
        if api_key_identifier not in self._llm_client_status: logger.warning(f"Attempted report issue for unknown LLM client: {api_key_identifier}"); return
        now = time.time(); cooldown_seconds = 60 * 5; status = "unavailable"; reason = issue_type
        if issue_type == "auth_error": cooldown_seconds = 60 * 60 * 24 * 365; reason = "Authentication Error"; logger.critical(f"LLM client key {api_key_identifier} marked permanently unavailable (auth error).")
        elif issue_type == "rate_limit": cooldown_seconds = 60 * 2; reason = "Rate Limited"
        elif issue_type == "timeout_error": cooldown_seconds = 60 * 3; reason = "Timeout"
        else: reason = "General Error"
        self._llm_client_status[api_key_identifier] = {"status": status, "reason": reason, "unavailable_until": now + cooldown_seconds}
        logger.warning(f"LLM client key {api_key_identifier} marked unavailable until {datetime.fromtimestamp(now + cooldown_seconds)}. Reason: {reason}")

    # MODIFIED: Added image_data parameter
    @llm_client_breaker
    async def call_llm(
        self,
        agent_name: str,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        is_json_output: bool = False,
        model_preference: Optional[List[str]] = None,
        image_data: Optional[bytes] = None, # Added for multimodal
        timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]: # Return dict including content and usage
        """Handles selecting an available client and making the LLM call, now with multimodal support."""
        selected_clients = await self.get_available_llm_clients()
        if not selected_clients: logger.error(f"Agent '{agent_name}' failed to get an available LLM client."); return None
        llm_client = selected_clients[0]; api_key_identifier = "unknown"
        for key_id, client_instance in self._llm_client_cache.items():
            if client_instance == llm_client: api_key_identifier = key_id; break

        # Determine model
        # Prioritize model_preference if provided and valid
        model_name = None
        if model_preference:
            for pref in model_preference:
                if pref in self.config.OPENROUTER_MODELS.values(): # Check if it's a known valid model value
                     model_name = pref
                     break
                 elif pref in self.config.OPENROUTER_MODELS: # Check if it's a known key
                      model_name = self.config.OPENROUTER_MODELS[pref]
                      break
            if model_name: logger.debug(f"Using preferred model: {model_name}")
            else: logger.warning(f"Model preference {model_preference} not found in config, using default.")

        # Fallback to agent-specific or default model
        if not model_name:
            model_key = "default_llm"
            # Agent specific model selection logic...
            if agent_name == "ThinkTool": model_key = "think_general"
            elif agent_name == "EmailAgent": model_key = "email_draft"
            elif agent_name == "VoiceSalesAgent": model_key = "voice_response"
            elif agent_name == "LegalAgent": model_key = "legal_validation"
            elif agent_name == "BrowsingAgent": model_key = "browsing_visual_analysis" if image_data else "browsing_summarize" # Choose visual model if image provided
            # ... add other agents
            model_name = self.config.OPENROUTER_MODELS.get(model_key, self.config.OPENROUTER_MODELS["default_llm"])

        # Caching Logic (remains the same)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        cache_key_parts = ["llm_call", prompt_hash, model_name, str(temperature), str(max_tokens), str(is_json_output), str(image_data is not None)] # Added image flag to key
        cache_key = ":".join(cache_key_parts); cache_ttl = self.cache_ttl_default
        cached_result = self.get_from_cache(cache_key)
        if cached_result is not None: logger.debug(f"LLM call cache hit (Orchestrator) for key: {cache_key[:20]}..."); return cached_result # Return cached dict

        try:
            response_format = {"type": "json_object"} if is_json_output else None
            # --- MODIFIED: Construct messages for multimodal ---
            messages = []
            content_parts = [{"type": "text", "text": prompt}]
            if image_data:
                # Encode image to base64
                base64_image = base64.b64encode(image_data).decode('utf-8')
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}" # Assuming PNG, adjust if needed
                    }
                })
                logger.debug("Image data included in LLM request.")
            messages.append({"role": "user", "content": content_parts})
            # --- END MODIFICATION ---

            logger.debug(f"Orchestrator making LLM Call: Agent={agent_name}, Model={model_name}, Key=...{api_key_identifier[-4:]}, Multimodal={image_data is not None}")

            api_timeout = timeout or settings.OPENROUTER_API_TIMEOUT_S or 120.0
            response = await llm_client.chat.completions.create(
                model=model_name, messages=messages, temperature=temperature,
                max_tokens=max_tokens, response_format=response_format, timeout=api_timeout,
            )
            content = response.choices[0].message.content.strip() if response.choices[0].message.content else None

            # Track usage & cost
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = input_tokens + output_tokens
            # TODO: Get actual costs per model from OpenRouter or estimate
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
            logger.warning(f"Orchestrator LLM call failed: Agent={agent_name}, Model={model_name}, Key=...{api_key_identifier[-4:]}, Type={issue_type}, Error={e}")
            await self.report_client_issue(api_key_identifier, issue_type)
            raise # Re-raise for tenacity/caller handling

    # --- Route Setup ---
    def setup_routes(self):
        """Configure Quart routes for UI interaction and webhooks."""

        @self.app.route("/")
        async def index():
            # ... (index route handler remains the same) ...
            try:
                template_path = os.path.join(os.path.dirname(__file__), "..", "ui", "templates", "index.html")
                if os.path.exists(template_path):
                    with open(template_path, "r") as f: return f.read(), 200, {"Content-Type": "text/html"}
                else: logger.error(f"UI Template not found at {template_path}"); return "UI Template not found.", 404
            except Exception as e: logger.error(f"Error serving index.html: {e}", exc_info=True); return "Internal Server Error", 500

        @self.app.route("/api/approve", methods=["POST"])
        async def approve():
            # ... (approve route handler remains the same) ...
            if self.approved: return jsonify({"status": "already_approved"}), 200
            self.approved = True; logger.info("!!! AGENCY APPROVED FOR FULL OPERATION via API !!!")
            await self.send_notification("Agency Approved", "Agency approved via API.")
            return jsonify({"status": "approved"})

        @self.app.route("/api/status", methods=["GET"])
        async def api_status():
             # ... (api_status route handler remains the same) ...
            agent_statuses = { name: agent.get_status_summary() for name, agent in self.agents.items() if hasattr(agent, "get_status_summary") }
            llm_status = { key: info["status"] for key, info in self._llm_client_status.items() }
            return jsonify({ "orchestrator_status": self.status if hasattr(self, "status") else "unknown", "approved_for_operation": self.approved, "agent_statuses": agent_statuses, "llm_client_status": llm_status, })

        @self.app.route("/api/start_ugc", methods=["POST"])
        async def handle_start_ugc():
             # ... (handle_start_ugc route handler remains the same) ...
            try:
                data = await request.get_json(); client_industry = data.get("client_industry")
                if not client_industry: return jsonify({"error": "Missing 'client_industry'"}), 400
                task = { "action": "plan_ugc_workflow", "client_industry": client_industry, "num_videos": int(data.get("num_videos", 1)), "initial_script": data.get("script"), "target_services": data.get("target_services"), }
                await self.delegate_task("ThinkTool", task)
                return jsonify({ "status": "UGC workflow planning initiated via ThinkTool" }), 202,
            except Exception as e: logger.error(f"Failed initiate UGC workflow planning: {e}", exc_info=True); return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route("/hosted_audio/<path:filename>")
        async def serve_hosted_audio(filename):
             # ... (serve_hosted_audio route handler remains the same) ...
            if ".." in filename or filename.startswith("/"): return "Forbidden", 403
            try:
                safe_path = os.path.join(self.temp_audio_dir, filename)
                if os.path.exists(safe_path) and os.path.isfile(safe_path): mimetype = ("audio/wav" if filename.lower().endswith(".wav") else "audio/mpeg"); return await send_file(safe_path, mimetype=mimetype)
                else: return jsonify({"error": "File not found"}), 404
            except Exception as e: logger.error(f"Error serving hosted audio {filename}: {e}"); return jsonify({"error": "Internal server error"}), 500

        @self.app.websocket("/twilio_call")
        async def handle_twilio_websocket():
             # ... (handle_twilio_websocket remains the same) ...
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
                        if payload: audio_bytes = base64.b64decode(payload); await deepgram_live_client.send(audio_bytes)
                    elif event == "stop": logger.info(f"Twilio WS: Stop event for {call_sid}."); break
                    elif event == "error": logger.error(f"Twilio WS: Error event for {call_sid}: {data.get('error')}"); break
            except websockets.exceptions.ConnectionClosedOK: logger.info(f"Twilio WS: Connection closed normally for {call_sid}.")
            except Exception as e: logger.error(f"Twilio WS: Unexpected error for {call_sid}: {e}", exc_info=True,)
            finally: logger.info(f"Twilio WS: Cleaning up for {call_sid}");
            if call_sid: await self.unregister_deepgram_connection_sdk(call_sid)

        @self.app.route("/track/<tracking_id>.png")
        async def handle_tracking_pixel(tracking_id):
             # ... (handle_tracking_pixel remains the same) ...
            logger.info(f"Tracking pixel hit: {tracking_id}")
            email_agent = self.agents.get("email");
            if email_agent and hasattr(email_agent, "process_email_open"): asyncio.create_task(email_agent.process_email_open(tracking_id))
            else: logger.warning(f"EmailAgent not found or cannot process opens for tracking ID: {tracking_id}")
            pixel_data = base64.b64decode("R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==")
            return (pixel_data, 200, {"Content-Type": "image/gif", "Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0",})

        # --- ADDED: MailerSend Webhook Handler ---
        @self.app.route("/webhooks/mailersend", methods=["POST"])
        async def handle_mailersend_webhook():
            """Handles incoming webhooks from MailerSend for bounces, complaints, etc."""
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

                if event_type in ['activity.hard_bounced', 'activity.spam_complaint']:
                    update_values['is_deliverable'] = False
                    update_values['opt_in'] = False # Treat hard bounce/spam complaint as opt-out
                    log_op_level = 'warning'
                    log_op_message = f"Marking {recipient_email} undeliverable/opt-out due to MailerSend webhook: {event_type}."
                elif event_type == 'activity.unsubscribed':
                    update_values['opt_in'] = False
                    log_op_level = 'warning'
                    log_op_message = f"Marking {recipient_email} opted-out due to MailerSend webhook: {event_type}."

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
                    if hasattr(self, 'log_operation'): # Check if log_operation exists
                         await self.log_operation(log_op_level, log_op_message)
                    else: # Fallback logging
                         log_func = getattr(logger, log_op_level, logger.info)
                         log_func(f"[Orchestrator Webhook] {log_op_message}")


                # Acknowledge receipt to MailerSend quickly
                return jsonify({"status": "received"}), 200

            except Exception as e:
                logger.error(f"Error processing MailerSend webhook: {e}", exc_info=True)
                # Still return 200 to MailerSend if possible, to avoid retries,
                # unless it's a fundamental error processing the request itself.
                return jsonify({"status": "error", "message": "Internal server error"}), 500 # Or 200 if you want MailerSend to stop retrying
        # --- END ADDED ---

        logger.info("Quart routes configured.")

    # --- Main Execution Loop ---
    async def run(self):
        """Initializes and runs the AI Agency Orchestrator."""
        logger.info("Orchestrator starting full initialization sequence...")
        self.running = False
        self.status = "initializing" # Set initial status
        try:
            if not await self.initialize_database(): raise RuntimeError("Database initialization failed.")
            if not await self.initialize_clients(): raise RuntimeError("LLM Client initialization failed.")
            if not await self.initialize_agents(): raise RuntimeError("Agent initialization failed.")
            logger.info("Orchestrator initialization complete.")

            logger.info("Starting background agent run loops and periodic tasks...")
            self.background_tasks = set()
            for agent_name, agent in self.agents.items():
                if hasattr(agent, "run") and callable(agent.run):
                    task = asyncio.create_task(agent.run(), name=f"AgentLoop_{agent_name}")
                    self.background_tasks.add(task)
                # --- MODIFIED: Start agent if it has start() method ---
                elif hasattr(agent, "start") and callable(agent.start):
                     task = asyncio.create_task(agent.start(), name=f"AgentStart_{agent_name}")
                     self.background_tasks.add(task) # Track start task if needed, or just launch
                     logger.info(f"Called start() for agent {agent_name}")
                # --- END MODIFICATION ---
                else: logger.warning(f"Agent {agent_name} does not have a callable run or start method.")

            self.background_tasks.add(asyncio.create_task(self._run_periodic_data_purge(), name="PeriodicDataPurge"))
            self.background_tasks.add(asyncio.create_task(self._run_periodic_feedback_collection(), name="PeriodicFeedback"))
            logger.info(f"Started {len(self.background_tasks)} background tasks.")

            logger.info("Orchestrator entering main operational state (API/Event driven).")
            self.running = True
            self.status = "running" # Update status
            self.last_feedback_time = time.time()
            self.last_purge_time = time.time()

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

        except (ValueError, RuntimeError) as init_err:
             logger.critical(f"Fatal Error: Orchestrator initialization failed: {init_err}", exc_info=True)
             self.status = "failed_initialization"
        except Exception as e:
             logger.critical(f"Fatal Error: Unhandled exception during Orchestrator run setup: {e}", exc_info=True)
             self.status = "error"
        finally:
            # Graceful Shutdown
            logger.info("Initiating graceful shutdown sequence...")
            self.running = False
            self.status = "stopping"

            # Cancel any remaining background tasks
            all_tasks = list(self.background_tasks) # Copy set before iterating
            if all_tasks:
                logger.info(f"Cancelling {len(all_tasks)} running background tasks...")
                for task in all_tasks:
                    if task and not task.done(): task.cancel()
                await asyncio.gather(*all_tasks, return_exceptions=True)
                logger.info("Background tasks cancellation complete.")

            # Explicitly stop agents
            agent_stop_tasks = []
            for agent_name, agent in self.agents.items():
                 if hasattr(agent, 'stop') and callable(agent.stop):
                      # Check agent status before stopping
                      agent_status = getattr(agent, 'status', 'unknown')
                      if agent_status not in [agent.STATUS_STOPPING, agent.STATUS_STOPPED]:
                           logger.info(f"Calling stop() for agent {agent_name}...")
                           agent_stop_tasks.append(asyncio.create_task(agent.stop(timeout=25.0)))
                      else:
                           logger.info(f"Agent {agent_name} already stopping/stopped ({agent_status}). Skipping stop call.")
                 else: logger.warning(f"Agent {agent_name} has no stop method.")

            if agent_stop_tasks:
                 logger.info(f"Waiting for {len(agent_stop_tasks)} agents to stop...")
                 await asyncio.gather(*agent_stop_tasks, return_exceptions=True)
                 logger.info("Agent stop sequence complete.")

            self.status = "stopped"
            logger.info("-------------------- Application Stopping --------------------")
            logging.shutdown() # Ensure all logs are flushed before exiting
            print("[INFO] Orchestrator: Process stopped.") # Final print statement

    # --- Agent Interaction & Tooling ---
    async def delegate_task(self, agent_name: str, task_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Delegates a task to a specific agent."""
        agent = self.agents.get(agent_name)
        if not agent:
            logger.error(f"Delegation failed: Agent '{agent_name}' not found.")
            return {"status": "error", "message": f"Agent '{agent_name}' not found."}

        if not hasattr(agent, 'execute_task') or not callable(agent.execute_task):
             logger.error(f"Delegation failed: Agent '{agent_name}' has no callable execute_task method.")
             return {"status": "error", "message": f"Agent '{agent_name}' cannot execute tasks."}

        task_id = task_details.get('id', str(uuid.uuid4()))
        task_details['id'] = task_id # Ensure task has an ID
        logger.info(f"Delegating task {task_id} ({task_details.get('action', 'N/A')}) to {agent_name}.")
        try:
            # Use asyncio.create_task for true background execution if execute_task is long-running
            # result = await asyncio.create_task(agent.execute_task(task_details))
            # For now, await directly, assuming execute_task handles backgrounding if needed
            result = await agent.execute_task(task_details)
            tasks_processed_counter.labels(agent_name=agent_name).inc()
            return result
        except Exception as e:
            logger.error(f"Error during task delegation to {agent_name} (Task ID: {task_id}): {e}", exc_info=True)
            error_counter.labels(agent_name=agent_name).inc()
            await self.report_error(agent_name, f"Task delegation failed: {e}", task_id)
            return {"status": "error", "message": f"Exception during task execution: {e}"}

    async def use_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Allows agents to request tool usage via the orchestrator."""
        # This is a placeholder - expand with actual tool implementations
        logger.info(f"Orchestrator received tool use request: {tool_name} with params: {params}")
        if tool_name == "read_file":
            file_path = params.get('path')
            if file_path and os.path.exists(file_path) and os.path.isfile(file_path):
                 try:
                      with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                           content = f.read()
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
        else:
            logger.warning(f"Attempted to use unknown tool: {tool_name}")
            return {"status": "failure", "message": f"Unknown tool: {tool_name}"}

    async def report_error(self, agent_name: str, error_message: str, task_id: Optional[str] = None):
        """Handles error reporting from agents."""
        log_msg = f"ERROR reported by {agent_name}: {error_message}"
        if task_id: log_msg += f" (Task: {task_id})"
        logger.error(log_msg)
        error_counter.labels(agent_name=agent_name).inc()
        # Optionally send notification on critical errors
        # await self.send_notification(f"Agent Error: {agent_name}", log_msg)

    async def send_notification(self, title: str, message: str, level: str = "info"):
        """Sends notifications (e.g., email, Slack) using the utility function."""
        # Use the imported send_notification function
        await send_notification(title, message, level, self.config)

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
            if time.time() < expires_at:
                logger.debug(f"Cache hit for key: {key[:20]}...")
                return value
            else:
                logger.debug(f"Cache expired for key: {key[:20]}...")
                del self._cache[key] # Remove expired item
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
            del self.deepgram_connections[call_sid]
            logger.info(f"Unregistered Deepgram SDK client for Call SID: {call_sid}")

    # --- Temporary Audio Hosting ---
    async def host_temporary_audio(self, audio_data: bytes, filename: str) -> Optional[str]:
         """Saves audio data temporarily and returns a publicly accessible URL."""
         try:
             # Ensure filename is safe
             safe_filename = re.sub(r'[^\w\.\-]', '_', filename)
             filepath = os.path.join(self.temp_audio_dir, safe_filename)
             with open(filepath, 'wb') as f:
                 f.write(audio_data)
             # Construct URL using AGENCY_BASE_URL and the Quart route
             # Use url_for if Quart context is available, otherwise construct manually
             base_url = self.config.AGENCY_BASE_URL.rstrip('/')
             audio_url = f"{base_url}/hosted_audio/{safe_filename}"
             logger.info(f"Hosted temporary audio at: {audio_url}")
             # TODO: Add cleanup mechanism for old audio files
             return audio_url
         except Exception as e:
             logger.error(f"Failed to host temporary audio {filename}: {e}", exc_info=True)
             return None

    # --- Invoice Generation Request ---
    async def request_invoice_generation(self, client_id: int, amount: float, source_call_sid: str):
         """Sends a directive to ThinkTool/LegalAgent to generate an invoice."""
         logger.info(f"Requesting invoice generation for Client {client_id}, Amount ${amount:.2f}, Source Call {source_call_sid}")
         # Decide which agent handles invoice creation (e.g., ThinkTool plans, Legal adds notes?)
         # For now, send directive to ThinkTool to coordinate
         directive_content = {
             "client_id": client_id,
             "amount": amount,
             "source_reference": f"CallLog:{source_call_sid}",
             "due_date_days": 14 # Example: Due in 14 days
         }
         await self.delegate_task("ThinkTool", { # Or maybe LegalAgent?
             "action": "generate_invoice", # Define this action
             "content": directive_content,
             "priority": 3 # High priority after successful call
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
                        # Check if account already exists
                        stmt_check = select(AccountCredentials.id).where(AccountCredentials.service == service, AccountCredentials.account_identifier == identifier).limit(1)
                        existing = await session.scalar(stmt_check)
                        if existing:
                             self.logger.warning(f"Account {identifier} for {service} already exists (ID: {existing}). Updating status/notes.")
                             stmt_update = update(AccountCredentials).where(AccountCredentials.id == existing).values(status=status, notes=notes_str, last_used=None) # Reset last_used?
                             await session.execute(stmt_update)
                             # Fetch updated details (excluding password)
                             stmt_get = select(AccountCredentials.id, AccountCredentials.service, AccountCredentials.account_identifier, AccountCredentials.status).where(AccountCredentials.id == existing)
                             updated_details = (await session.execute(stmt_get)).mappings().first()
                             return dict(updated_details) if updated_details else None
                        else:
                             new_account = AccountCredentials(
                                 service=service,
                                 account_identifier=identifier,
                                 password=encrypted_password, # Store encrypted password
                                 status=status,
                                 notes=notes_str
                             )
                             session.add(new_account)
                             await session.flush() # Get the ID before commit finishes
                             account_id = new_account.id
                             self.logger.info(f"Stored new account credentials for {service}/{identifier} (ID: {account_id}).")
                             # Return details (excluding password)
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
                           # Return as dict, excluding password
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
                      ).order_by(func.random()).limit(1) # Random selection for basic rotation
                      account = await session.scalar(stmt)
                      if account:
                           # Update last used time
                           async with session.begin(): # Start new transaction for update
                                account.last_used = datetime.now(timezone.utc)
                                await session.merge(account)
                           # Return details excluding password
                           return {c.name: getattr(account, c.name) for c in AccountCredentials.__table__.columns if c.name != 'password'}
                      else: return None
             except Exception as e: self.logger.error(f"Error finding active account for {service}: {e}"); return None

        async def get_secret(self, identifier: str) -> Optional[str]:
             """Retrieves and decrypts a secret (e.g., password) using identifier."""
             # This assumes identifier uniquely maps to one secret (e.g., account ID or vault path)
             # For account passwords, use identifier = account_id
             if not self.session_maker: self.logger.error("DB session_maker unavailable."); return None
             try:
                 account_id = int(identifier) # Assume identifier is account ID for now
                 async with self.session_maker() as session:
                      stmt = select(AccountCredentials.password).where(AccountCredentials.id == account_id)
                      encrypted_pw = await session.scalar(stmt)
                      if encrypted_pw:
                           decrypted_pw = decrypt_data(encrypted_pw)
                           if decrypted_pw: return decrypted_pw
                           else: self.logger.error(f"Failed to decrypt password for account ID {account_id}."); return None
                      else: self.logger.warning(f"No password found for account ID {account_id}."); return None
             except ValueError: self.logger.error(f"Invalid identifier for get_secret (expected account ID): {identifier}"); return None
             except Exception as e: self.logger.error(f"Error retrieving secret for ID {identifier}: {e}"); return None

# --- End of agents/orchestrator.py ---