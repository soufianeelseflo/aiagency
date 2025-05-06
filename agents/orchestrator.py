# Filename: agents/orchestrator.py
# Description: Central coordinator for the AI Agency, managing core agents, workflows, and resources.
# Version: 3.2 (Removed Optional API Key Logic & Prometheus)

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
from typing import Dict, Optional, Tuple, Any, List, AsyncGenerator, Callable, Type

# --- Core Framework Imports ---
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy import select, delete, func, update, text, case, or_
from quart import Quart, request, jsonify, websocket, send_file, url_for, Response
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
# REMOVED Prometheus imports

# --- Project Imports ---
from config.settings import settings
from utils.database import (
    encrypt_data,
    decrypt_data,
    get_session,
    get_session_maker,
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
    fail_max=3, reset_timeout=60 * 5, name="LLMClientBreaker"
)

# --- Global Shutdown Event ---
shutdown_event = asyncio.Event()


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

        # REMOVED Prometheus setup

        self.meta_prompt = self.config.META_PROMPT
        self.approved = False

        # LLM Client Management State (Simplified)
        self._llm_client: Optional[AsyncLLMClient] = None
        self._llm_client_status: Dict[str, Any] = {"status": "unavailable", "reason": "Not initialized"}

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
        self.last_feedback_time: float = 0.0
        self.last_purge_time: float = 0.0

        self.running: bool = False
        self.background_tasks = set()
        self.status = "initializing"

        # Secure Storage Interface
        self.secure_storage = self._DatabaseSecureStorage(self.session_maker)

        logger.info("Orchestrator v3.2 (Removed Optional API Key Logic & Prometheus) initialized.")

    # --- Initialization Methods ---
    async def initialize_database(self):
        """Initialize or update the database schema."""
        logger.info("Initializing database schema...")
        try:
            # Ensure DATABASE_URL is accessed correctly
            db_url_str = str(self.config.DATABASE_URL) if self.config.DATABASE_URL else None
            if not db_url_str: raise ValueError("DATABASE_URL is not configured.")

            engine = create_async_engine(db_url_str, echo=False)
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
        self._llm_client_status = {"status": "unavailable", "reason": "Initialization failed"}

        primary_key = self.config.get_secret("OPENROUTER_API_KEY")
        if primary_key:
            try:
                self._llm_client = AsyncLLMClient(api_key=primary_key, base_url="https://openrouter.ai/api/v1")
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
            smartproxy_pass = self.config.get_secret("SMARTPROXY_PASSWORD")
            self.agents["browsing"] = BrowsingAgent(self.session_maker, self, smartproxy_password=smartproxy_pass)

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

            # REMOVED Prometheus gauge setting

            if initialization_failed: raise RuntimeError("Failed to initialize critical agents due to missing secrets.")

            logger.info(f"Core agents initialized: {list(self.agents.keys())}")
            return True

        except Exception as e: logger.error(f"Core agent initialization failed: {e}", exc_info=True); return False

    # --- LLM Client Management (Simplified) ---
    @llm_client_breaker
    async def get_available_llm_client(self) -> Optional[AsyncLLMClient]:
        """Returns the primary LLM client if available."""
        now = time.time()
        status_info = self._llm_client_status
        if status_info["status"] == "available" or now >= status_info.get("unavailable_until", 0):
            if status_info["status"] == "unavailable":
                status_info["status"] = "available"
                status_info["reason"] = None
                status_info["unavailable_until"] = 0
                logger.info("Primary LLM client available after cooldown.")
            return self._llm_client
        else:
            logger.error(f"Primary LLM client unavailable. Reason: {status_info.get('reason', 'Unknown')}")
            return None

    async def report_client_issue(self, issue_type: str):
        """Reports an issue with the primary LLM client."""
        now = time.time(); cooldown_seconds = 60 * 5; status = "unavailable"; reason = issue_type
        if issue_type == "auth_error":
            cooldown_seconds = 60 * 60 * 24 * 365 # Effectively permanent
            reason = "Authentication Error"
            logger.critical("Primary LLM client marked permanently unavailable (auth error).")
        elif issue_type == "rate_limit": cooldown_seconds = 60 * 2; reason = "Rate Limited"
        elif issue_type == "timeout_error": cooldown_seconds = 60 * 3; reason = "Timeout"
        else: reason = "General Error"

        self._llm_client_status = {"status": status, "reason": reason, "unavailable_until": now + cooldown_seconds}
        logger.warning(f"Primary LLM client marked unavailable until {datetime.fromtimestamp(now + cooldown_seconds)}. Reason: {reason}")

    @llm_client_breaker
    async def call_llm(
        self,
        agent_name: str,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        is_json_output: bool = False,
        model_preference: Optional[List[str]] = None, # Still accept preference, but use settings value
        image_data: Optional[bytes] = None,
        timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Handles making the LLM call using the primary client."""
        llm_client = await self.get_available_llm_client()
        if not llm_client:
            logger.error(f"Agent '{agent_name}' failed: Primary LLM client unavailable.")
            return None

        # Determine model name from settings based on agent/task type
        model_name = None
        if agent_name == "ThinkTool": model_key = "think_general" # Default for ThinkTool
        elif agent_name == "EmailAgent": model_key = "email_draft"
        elif agent_name == "VoiceSalesAgent": model_key = "voice_response"
        elif agent_name == "LegalAgent": model_key = "legal_validation"
        elif agent_name == "BrowsingAgent": model_key = "browsing_visual_analysis" if image_data else "browsing_summarize"
        else: model_key = "default_llm" # Fallback

        # Allow task context to override the default model key for the agent
        task_specific_model_key = None
        if isinstance(prompt, str): # Basic check if prompt might contain task info
             if "synthesize" in prompt.lower() or "strategize" in prompt.lower(): task_specific_model_key = "think_strategize"
             elif "critique" in prompt.lower(): task_specific_model_key = "think_critique"
             elif "radar" in prompt.lower() or "scouting" in prompt.lower(): task_specific_model_key = "think_radar"
             elif "validate" in prompt.lower(): task_specific_model_key = "think_validate"
             elif "educational content" in prompt.lower(): task_specific_model_key = "think_user_education"
             elif "humanize" in prompt.lower(): task_specific_model_key = "email_humanize"
             elif "intent" in prompt.lower(): task_specific_model_key = "voice_intent"
             # Add more specific task checks if needed

        final_model_key = task_specific_model_key or model_key
        model_name = self.config.OPENROUTER_MODELS.get(final_model_key, self.config.OPENROUTER_MODELS["default_llm"])
        logger.debug(f"Selected model '{model_name}' for agent '{agent_name}' task (key: {final_model_key}).")


        # Caching Logic
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        cache_key_parts = ["llm_call", prompt_hash, model_name, str(temperature), str(max_tokens), str(is_json_output), str(image_data is not None)]
        cache_key = ":".join(cache_key_parts); cache_ttl = self.cache_ttl_default
        cached_result = self.get_from_cache(cache_key)
        if cached_result is not None: logger.debug(f"LLM call cache hit (Orchestrator) for key: {cache_key[:20]}..."); return cached_result

        try:
            response_format = {"type": "json_object"} if is_json_output else None
            messages = []
            content_parts = [{"type": "text", "text": prompt}]
            if image_data:
                base64_image = base64.b64encode(image_data).decode('utf-8')
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
                logger.debug("Image data included in LLM request.")
            messages.append({"role": "user", "content": content_parts})

            logger.debug(f"Orchestrator making LLM Call: Agent={agent_name}, Model={model_name}, Multimodal={image_data is not None}")

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
            await self.report_client_issue(issue_type) # Report issue for the primary client
            raise

    # --- Route Setup ---
    def setup_routes(self):
        """Configure Quart routes for UI interaction and webhooks."""

        @self.app.route("/")
        async def index():
            try:
                template_path = os.path.join(os.path.dirname(__file__), "..", "ui", "templates", "index.html")
                if os.path.exists(template_path):
                    with open(template_path, "r") as f: return f.read(), 200, {"Content-Type": "text/html"}
                else: logger.error(f"UI Template not found at {template_path}"); return "UI Template not found.", 404
            except Exception as e: logger.error(f"Error serving index.html: {e}", exc_info=True); return "Internal Server Error", 500

        @self.app.route("/api/approve", methods=["POST"])
        async def approve():
            if self.approved: return jsonify({"status": "already_approved"}), 200
            self.approved = True; logger.info("!!! AGENCY APPROVED FOR FULL OPERATION via API !!!")
            await self.send_notification("Agency Approved", "Agency approved via API.")
            return jsonify({"status": "approved"})

        @self.app.route("/api/status", methods=["GET"])
        async def api_status():
            agent_statuses = { name: agent.get_status_summary() for name, agent in self.agents.items() if hasattr(agent, "get_status_summary") }
            llm_status = self._llm_client_status.get("status", "unavailable") # Simplified status
            return jsonify({
                "orchestrator_status": self.status if hasattr(self, "status") else "unknown",
                "approved_for_operation": self.approved,
                "agent_statuses": agent_statuses,
                "llm_client_status": {"primary": llm_status} # Show only primary status
            })

        @self.app.route("/api/start_ugc", methods=["POST"])
        async def handle_start_ugc():
            try:
                data = await request.get_json(); client_industry = data.get("client_industry")
                if not client_industry: return jsonify({"error": "Missing 'client_industry'"}), 400
                task = { "action": "plan_ugc_workflow", "client_industry": client_industry, "num_videos": int(data.get("num_videos", 1)), "initial_script": data.get("script"), "target_services": data.get("target_services"), }
                await self.delegate_task("ThinkTool", task)
                return jsonify({ "status": "UGC workflow planning initiated via ThinkTool" }), 202
            except Exception as e: logger.error(f"Failed initiate UGC workflow planning: {e}", exc_info=True); return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route("/hosted_audio/<path:filename>")
        async def serve_hosted_audio(filename):
            if ".." in filename or filename.startswith("/"): return "Forbidden", 403
            try:
                safe_path = os.path.join(self.temp_audio_dir, filename)
                if os.path.exists(safe_path) and os.path.isfile(safe_path): mimetype = ("audio/wav" if filename.lower().endswith(".wav") else "audio/mpeg"); return await send_file(safe_path, mimetype=mimetype)
                else: return jsonify({"error": "File not found"}), 404
            except Exception as e: logger.error(f"Error serving hosted audio {filename}: {e}"); return jsonify({"error": "Internal server error"}), 500

        @self.app.websocket("/twilio_call")
        async def handle_twilio_websocket():
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
            logger.info(f"Tracking pixel hit: {tracking_id}")
            email_agent = self.agents.get("email");
            if email_agent and hasattr(email_agent, "process_email_open"): asyncio.create_task(email_agent.process_email_open(tracking_id))
            else: logger.warning(f"EmailAgent not found or cannot process opens for tracking ID: {tracking_id}")
            pixel_data = base64.b64decode("R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==")
            return (pixel_data, 200, {"Content-Type": "image/gif", "Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0",})

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
                return jsonify({"status": "error", "message": "Internal server error"}), 500
        # --- END ADDED ---

        logger.info("Quart routes configured.")

    # --- Main Execution Loop ---
    async def run(self):
        """Initializes and runs the AI Agency Orchestrator."""
        logger.info("Orchestrator starting full initialization sequence...")
        self.running = False
        self.status = "initializing"
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
                elif hasattr(agent, "start") and callable(agent.start):
                    task = asyncio.create_task(agent.start(), name=f"AgentStart_{agent_name}")
                    self.background_tasks.add(task)
                    logger.info(f"Called start() for agent {agent_name}")
                else: logger.warning(f"Agent {agent_name} does not have a callable run or start method.")

            self.background_tasks.add(asyncio.create_task(self._run_periodic_data_purge(), name="PeriodicDataPurge"))
            self.background_tasks.add(asyncio.create_task(self._run_periodic_feedback_collection(), name="PeriodicFeedback"))
            logger.info(f"Started {len(self.background_tasks)} background tasks.")

            logger.info("Orchestrator entering main operational state (API/Event driven).")
            self.running = True
            self.status = "running"
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
            shutdown_event.set() # Signal shutdown to periodic tasks

            # Cancel any remaining background tasks
            all_tasks = list(self.background_tasks)
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
            logging.shutdown()
            print("[INFO] Orchestrator: Process stopped.")

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
        task_details['id'] = task_id
        logger.info(f"Delegating task {task_id} ({task_details.get('action', 'N/A')}) to {agent_name}.")
        try:
            result = await agent.execute_task(task_details)
            # REMOVED Prometheus counter
            return result
        except Exception as e:
            logger.error(f"Error during task delegation to {agent_name} (Task ID: {task_id}): {e}", exc_info=True)
            # REMOVED Prometheus counter
            await self.report_error(agent_name, f"Task delegation failed: {e}", task_id)
            return {"status": "error", "message": f"Exception during task execution: {e}"}

    async def use_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Allows agents to request tool usage via the orchestrator."""
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
        # REMOVED Prometheus counter
        # await self.send_notification(f"Agent Error: {agent_name}", log_msg)

    async def send_notification(self, title: str, message: str, level: str = "info"):
        """Sends notifications (e.g., email, Slack) using the utility function."""
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
            safe_filename = re.sub(r'[^\w\.\-]', '_', filename)
            filepath = os.path.join(self.temp_audio_dir, safe_filename)
            with open(filepath, 'wb') as f: f.write(audio_data)
            base_url = str(self.config.AGENCY_BASE_URL).rstrip('/') # Ensure string conversion
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
                            stmt_update = update(AccountCredentials).where(AccountCredentials.id == existing).values(status=status, notes=notes_str, last_used=None)
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
                    ).order_by(func.random()).limit(1)
                    account = await session.scalar(stmt)
                    if account:
                        async with session.begin():
                                account.last_used = datetime.now(timezone.utc)
                                await session.merge(account)
                        return {c.name: getattr(account, c.name) for c in AccountCredentials.__table__.columns if c.name != 'password'}
                    else: return None
            except Exception as e: self.logger.error(f"Error finding active account for {service}: {e}"); return None

        async def get_secret(self, identifier: str) -> Optional[str]:
            """Retrieves and decrypts a secret (e.g., password) using identifier."""
            if not self.session_maker: self.logger.error("DB session_maker unavailable."); return None
            try:
                account_id = int(identifier)
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