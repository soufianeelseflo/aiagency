    # Filename: agents/orchestrator.py
    # Description: Central coordinator for the AI Agency, managing core agents, workflows, and resources.
    # Version: 3.0 (Genius Agentic - Finalized for Core Agents & Postgres)

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
    from sqlalchemy import select, delete, func, update, text, case
    from quart import Quart, request, jsonify, websocket, send_file, url_for
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
    from utils.notifications import send_notification

    # Import *only* the core agent classes needed + kept agents
    try:
        from agents.base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
    except ImportError:
        logging.warning(
            "Production base agent not found, using GeniusAgentBase. Ensure"
            " base_agent_prod.py is used."
        )
        from agents.base_agent import GeniusAgentBase

    from agents.think_tool import ThinkTool
    from agents.browsing_agent import BrowsingAgent
    from agents.email_agent import EmailAgent
    from agents.voice_sales_agent import VoiceSalesAgent
    from agents.legal_agent import LegalAgent
    from agents.social_media_manager import SocialMediaManager

    # Import database models
    from models import (
        Base,
        Client,
        Metric,
        ExpenseLog,
        MigrationStatus,
        KnowledgeFragment,
        AccountCredentials,
        CallLog,
        Invoice,
        StrategicDirective,
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
            self.setup_routes()

            try:
                start_http_server(8001)
                logger.info("Prometheus metrics server started on port 8001.")
            except OSError as e:
                logger.warning(
                    "Could not start Prometheus server on port 8001: {e}"
                )
            except Exception as e:
                logger.error(
                    "Failed to start Prometheus server: {e}", exc_info=True
                )

            self.meta_prompt = self.config.META_PROMPT
            self.approved = False

            # LLM Client Management State
            self._llm_client_cache: Dict[str, AsyncLLMClient] = {}
            self._llm_client_status: Dict[str, Dict[str, Any]] = {}
            self._llm_client_keys: List[str] = []
            self._llm_client_round_robin_index = 0

            # In-memory Cache (Simple)
            self._cache: Dict[str, Tuple[Any, float]] = {}

            # Deepgram WebSocket Registry
            self.deepgram_connections: Dict[str, Any] = {}
            self.temp_audio_dir = "/app/temp_audio"
            os.makedirs(self.temp_audio_dir, exist_ok=True)

            # Periodic Task Timing
            self.feedback_interval_seconds: int = int(
                self.config.get("THINKTOOL_FEEDBACK_INTERVAL_SECONDS", 300)
            )
            self.purge_interval_seconds: int = int(
                self.config.get("DATA_PURGE_INTERVAL_SECONDS", 86400)
            )
            self.last_feedback_time: float = 0.0
            self.last_purge_time: float = 0.0

            self.running: bool = False
            self.background_tasks = set()

            logger.info("Orchestrator v3.0 (Finalized Core) initialized.")
            # Initialization sequence called from run()

        async def initialize_database(self):
            """Initialize or update the database schema."""
            logger.info("Initializing database schema...")
            try:
                engine = create_async_engine(self.config.DATABASE_URL, echo=False)
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
            logger.info(
                "Initializing LLM clients from settings/environment variables..."
            )
            self._llm_client_keys = []
            self._llm_client_cache = {}
            self._llm_client_status = {}

            # Load primary OpenRouter key
            primary_openrouter_key = self.config.get_secret(
                "OPENROUTER_API_KEY"
            )
            if primary_openrouter_key:
                try:
                    key_id = "OR_Primary_" + primary_openrouter_key[-4:]
                    self._llm_client_cache[
                        key_id
                    ] = AsyncLLMClient(
                        api_key=primary_openrouter_key,
                        base_url="https://openrouter.ai/api/v1",
                    )
                    self._llm_client_status[key_id] = {
                        "status": "available",
                        "reason": None,
                        "unavailable_until": 0,
                    }
                    self._llm_client_keys.append(key_id)
                    logger.info(
                        "Primary OpenRouter client initialized (Key ID:"
                        f" {key_id})."
                    )
                except Exception as e:
                    logger.error(
                        "Failed to initialize primary OpenRouter client: {e}",
                        exc_info=True,
                    )
            else:
                logger.warning("Primary OPENROUTER_API_KEY not found.")

            # Load additional keys if needed (example)
            i = 1
            while True:
                additional_key = self.config.get_secret(f"OPENROUTER_API_KEY_{i}")
                if not additional_key:
                    break
                try:
                    key_id = f"OR_Extra{i}_" + additional_key[-4:]
                    if key_id not in self._llm_client_cache:
                        self._llm_client_cache[
                            key_id
                        ] = AsyncLLMClient(
                            api_key=additional_key,
                            base_url="https://openrouter.ai/api/v1",
                        )
                        self._llm_client_status[key_id] = {
                            "status": "available",
                            "reason": None,
                            "unavailable_until": 0,
                        }
                        self._llm_client_keys.append(key_id)
                        logger.info(
                            "Loaded additional OpenRouter client {i} (Key ID:"
                            f" {key_id})."
                        )
                except Exception as e:
                    logger.error(
                        "Failed to initialize additional OpenRouter client {i}:"
                        f" {e}"
                    )
                i += 1

            if not self._llm_client_keys:
                logger.critical("CRITICAL: No LLM API clients could be initialized.")
                return False
            logger.info(
                "LLM Client Initialization complete. Total usable keys:"
                f" {len(self._llm_client_keys)}"
            )
            return True

        async def initialize_agents(self):
            """Initialize the CORE agents + kept agents, passing necessary secrets."""
            logger.info("Initializing CORE agents...")
            initialization_failed = False
            try:
                # ThinkTool
                self.agents["think"] = ThinkTool(
                    self.session_maker, self.config, self
                )

                # BrowsingAgent
                smartproxy_pass = self.config.get_secret(
                    "SMARTPROXY_PASSWORD"
                )
                if not smartproxy_pass:
                    logger.warning(
                        "SMARTPROXY_PASSWORD not set, BrowsingAgent proxy auth will"
                        " fail."
                    )
                self.agents["browsing"] = BrowsingAgent(
                    self.session_maker, self, smartproxy_password=smartproxy_pass
                )

                # EmailAgent
                smtp_pass = self.config.get_secret("HOSTINGER_SMTP_PASS")
                imap_pass = self.config.get_secret("HOSTINGER_IMAP_PASS")
                if not smtp_pass or not imap_pass:
                    logger.critical(
                        "Hostinger SMTP/IMAP password missing. EmailAgent will fail."
                    )
                    initialization_failed = True
                self.agents["email"] = EmailAgent(
                    self.session_maker, self, smtp_password=smtp_pass, imap_password=imap_pass
                )

                # VoiceSalesAgent
                twilio_token = self.config.get_secret("TWILIO_AUTH_TOKEN")
                deepgram_key = self.config.get_secret("DEEPGRAM_API_KEY")
                if not twilio_token or not deepgram_key:
                    logger.critical(
                        "Twilio Auth Token or Deepgram API Key missing."
                        " VoiceSalesAgent will fail."
                    )
                    initialization_failed = True
                self.agents["voice_sales"] = VoiceSalesAgent(
                    self.session_maker,
                    self,
                    twilio_auth_token=twilio_token,
                    deepgram_api_key=deepgram_key,
                )

                # LegalAgent
                self.agents["legal"] = LegalAgent(self.session_maker, self)

                # Kept Agents
                self.agents["social_media"] = SocialMediaManager(
                    self, self.session_maker
                )
                self.agents["programmer"] = ProgrammerAgent(
                    self, self.session_maker
                )

                # Set Prometheus gauges
                for name in self.agents.keys():
                    agent_status_gauge.labels(agent_name=name).set(0)  # idle

                if initialization_failed:
                    raise RuntimeError(
                        "Failed to initialize one or more critical agents due to"
                        " missing secrets."
                    )

                logger.info(f"Core agents initialized: {list(self.agents.keys())}")
                return True

            except Exception as e:
                logger.error(
                    f"Core agent initialization failed: {e}", exc_info=True
                )
                return False

        # --- LLM Client Management ---
        @llm_client_breaker
        async def get_available_llm_clients(self) -> List[AsyncLLMClient]:
            """Gets available LLM client instances based on status and rotation."""

            now = time.time()
            available_keys = []
            for key_id in self._llm_client_keys:
                status_info = self._llm_client_status.get(
                    key_id, {"status": "available", "unavailable_until": 0}
                )
                if (
                    status_info["status"] == "available"
                    or now >= status_info.get("unavailable_until", 0)
                ):
                    if status_info["status"] == "unavailable":
                        status_info["status"] = "available"
                        status_info["reason"] = None
                        status_info["unavailable_until"] = 0
                        logger.info(
                            "LLM client key {key_id} marked as available after"
                            " cooldown."
                        )
                    available_keys.append(key_id)

            if not available_keys:
                logger.error("No available LLM clients found!")
                return []

            selected_key_id = available_keys[
                self._llm_client_round_robin_index % len(available_keys)
            ]
            self._llm_client_round_robin_index += 1
            selected_client = self._llm_client_cache.get(selected_key_id)

            if not selected_client:
                logger.error(
                    "Client instance not found in cache for available key ID"
                    " {selected_key_id}. This indicates an initialization issue."
                )
                # Attempt to remove the bad key entry
                if selected_key_id in self._llm_client_keys:
                    self._llm_client_keys.remove(selected_key_id)
                self._llm_client_status.pop(selected_key_id, None)
                self._llm_client_cache.pop(selected_key_id, None)
                return await self.get_available_llm_clients()  # Retry selection

            logger.debug("Selected LLM client with key ID: {selected_key_id}")
            return [selected_client]

        async def report_client_issue(self, api_key_identifier: str, issue_type: str):
            """Marks an LLM client as unavailable due to an issue."""

            if api_key_identifier not in self._llm_client_status:
                logger.warning(
                    "Attempted to report issue for unknown LLM client identifier:"
                    f" {api_key_identifier}"
                )
                return
            now = time.time()
            cooldown_seconds = 60 * 5
            status = "unavailable"
            reason = issue_type
            if issue_type == "auth_error":
                cooldown_seconds = 60 * 60 * 24 * 365
                reason = "Authentication Error"
                logger.critical(
                    "LLM client key {api_key_identifier} marked permanently"
                    " unavailable (auth error)."
                )
            elif issue_type == "rate_limit":
                cooldown_seconds = 60 * 2
                reason = "Rate Limited"
            elif issue_type == "timeout_error":
                cooldown_seconds = 60 * 3
                reason = "Timeout"
            else:
                reason = "General Error"
            self._llm_client_status[api_key_identifier] = {
                "status": status,
                "reason": reason,
                "unavailable_until": now + cooldown_seconds,
            }
            logger.warning(
                "LLM client key {api_key_identifier} marked unavailable until"
                " {datetime.fromtimestamp(now + cooldown_seconds)}. Reason: {reason}"
            )

        @llm_client_breaker
        async def call_llm(
            self,
            agent_name: str,
            prompt: str,
            temperature: float = 0.5,
            max_tokens: int = 1024,
            is_json_output: bool = False,
            model_preference: Optional[List[str]] = None,
        ) -> Optional[str]:
            """Handles selecting an available client and making the LLM call."""

            selected_clients = await self.get_available_llm_clients()
            if not selected_clients:
                logger.error(
                    "Agent '{agent_name}' failed to get an available LLM client."
                )
                return None

            llm_client = selected_clients[0]  # Use the first available client from rotation
            api_key_identifier = "unknown"
            for key_id, client_instance in self._llm_client_cache.items():
                if client_instance == llm_client:
                    api_key_identifier = key_id
                    break

            # Determine model using the mapping in settings
            model_key = "default_llm"  # Fallback
            if agent_name == "ThinkTool":
                model_key = "think_general"
            elif agent_name == "EmailAgent":
                model_key = "email_draft"
            elif agent_name == "VoiceSalesAgent":
                model_key = "voice_response"
            elif agent_name == "LegalAgent":
                model_key = "legal_validation"
            # Add more specific mappings if needed
            model_name = self.config.OPENROUTER_MODELS.get(
                model_key, self.config.OPENROUTER_MODELS["default_llm"]
            )

            # --- Caching Logic ---
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            cache_key_parts = [
                "llm_call",
                prompt_hash,
                model_name,
                str(temperature),
                str(max_tokens),
                str(is_json_output),
            ]
            cache_key = ":".join(cache_key_parts)
            cache_ttl = 3600  # 1 hour cache
            cached_result = self.get_from_cache(cache_key)
            if cached_result is not None:
                logger.debug(
                    "LLM call cache hit (Orchestrator) for key: {cache_key[:20]}..."
                )
                return cached_result

            try:
                response_format = {"type": "json_object"} if is_json_output else None
                messages = [{"role": "user", "content": prompt}]
                logger.debug(
                    "Orchestrator making LLM Call: Agent={agent_name}, Model={model_name},"
                    " Key=...{api_key_identifier[-4:]}"
                )

                api_timeout = settings.OPENROUTER_API_TIMEOUT_S or 120.0
                response = await llm_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    timeout=api_timeout,
                )
                content = response.choices[0].message.content.strip()

                # Track usage & cost
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
                total_tokens = input_tokens + output_tokens
                cost_per_million_input = 0.50
                cost_per_million_output = 1.50  # Example costs
                cost = (
                    input_tokens / 1_000_000 * cost_per_million_input
                    + output_tokens / 1_000_000 * cost_per_million_output
                )
                logger.debug(
                    "LLM Call Usage: Model={model_name}, Tokens={total_tokens},"
                    " Est. Cost=${cost:.6f}"
                )
                await self.report_expense(
                    agent_name,
                    cost,
                    "LLM",
                    "LLM call ({model_name}). Tokens: {total_tokens}.",
                )

                self.add_to_cache(cache_key, content, ttl_seconds=cache_ttl)
                return content

            except Exception as e:
                error_str = str(e).lower()
                issue_type = "llm_error"
                if "rate limit" in error_str or "quota" in error_str:
                    issue_type = "rate_limit"
                elif "authentication" in error_str:
                    issue_type = "auth_error"
                elif "timeout" in error_str:
                    issue_type = "timeout_error"
                logger.warning(
                    "Orchestrator LLM call failed: Agent={agent_name}, Model={model_name},"
                    " Key=...{api_key_identifier[-4:]}, Type={issue_type}, Error={e}"
                )
                await self.report_client_issue(api_key_identifier, issue_type)
                raise  # Re-raise for tenacity/caller handling

        # --- Route Setup ---
        def setup_routes(self):
            """Configure Quart routes for UI interaction and webhooks."""

            @self.app.route("/")
            async def index():
                try:
                    template_path = os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "ui",
                        "templates",
                        "index.html",
                    )
                    if os.path.exists(template_path):
                        with open(template_path, "r") as f:
                            return f.read(), 200, {"Content-Type": "text/html"}
                    else:
                        logger.error("UI Template not found at {template_path}")
                        return "UI Template not found.", 404
                except Exception as e:
                    logger.error("Error serving index.html: {e}", exc_info=True)
                    return "Internal Server Error", 500

            @self.app.route("/api/approve", methods=["POST"])
            async def approve():
                if self.approved:
                    return jsonify({"status": "already_approved"}), 200
                self.approved = True
                logger.info("!!! AGENCY APPROVED FOR FULL OPERATION via API !!!")
                await self.send_notification("Agency Approved", "Agency approved via API.")
                return jsonify({"status": "approved"})

            @self.app.route("/api/status", methods=["GET"])  # Renamed from /api/status_kpi for simplicity
            async def api_status():
                # Fetch KPIs and Status (Simplified version of ui/app.py logic)
                agent_statuses = {
                    name: agent.get_status_summary()
                    for name, agent in self.agents.items()
                    if hasattr(agent, "get_status_summary")
                }
                llm_status = {
                    key: info["status"] for key, info in self._llm_client_status.items()
                }
                # Add basic KPI fetching if needed, or rely on Grafana
                return jsonify(
                    {
                        "orchestrator_status": self.status
                        if hasattr(self, "status")
                        else "unknown",  # Use internal status if available
                        "approved_for_operation": self.approved,
                        "agent_statuses": agent_statuses,
                        "llm_client_status": llm_status,
                    }
                )

            @self.app.route("/api/start_ugc", methods=["POST"])
            async def handle_start_ugc():
                try:
                    data = await request.get_json()
                    client_industry = data.get("client_industry")
                    if not client_industry:
                        return jsonify({"error": "Missing 'client_industry'"}), 400
                    task = {
                        "action": "plan_ugc_workflow",
                        "client_industry": client_industry,
                        "num_videos": int(data.get("num_videos", 1)),
                        "initial_script": data.get("script"),
                        "target_services": data.get("target_services"),
                    }
                    await self.delegate_task("ThinkTool", task)
                    return (
                        jsonify(
                            {
                                "status": "UGC workflow planning initiated via"
                                " ThinkTool"
                            }
                        ),
                        202,
                    )
                except Exception as e:
                    logger.error(
                        "Failed initiate UGC workflow planning: {e}", exc_info=True
                    )
                    return jsonify({"status": "error", "message": str(e)}), 500

            @self.app.route("/hosted_audio/<path:filename>")
            async def serve_hosted_audio(filename):
                if ".." in filename or filename.startswith("/"):
                    return "Forbidden", 403
                try:
                    safe_path = os.path.join(self.temp_audio_dir, filename)
                    if os.path.exists(safe_path) and os.path.isfile(safe_path):
                        mimetype = (
                            "audio/wav"
                            if filename.lower().endswith(".wav")
                            else "audio/mpeg"
                        )
                        return await send_file(safe_path, mimetype=mimetype)
                    else:
                        return jsonify({"error": "File not found"}), 404
                except Exception as e:
                    logger.error("Error serving hosted audio {filename}: {e}")
                    return jsonify({"error": "Internal server error"}), 500

            @self.app.websocket("/twilio_call")
            async def handle_twilio_websocket():
                # Handles Twilio media stream and forwards to Deepgram SDK client
                call_sid = None
                deepgram_live_client = None
                try:
                    while True:
                        message = await websocket.receive()
                        data = json.loads(message)
                        event = data.get("event")
                        if event == "start":
                            call_sid = data.get("start", {}).get("callSid")
                            logger.info("Twilio WS: Start event for {call_sid}")
                            deepgram_live_client = (
                                await self.get_deepgram_connection_sdk(call_sid)
                            )
                            if not deepgram_live_client:
                                logger.error(
                                    "Twilio WS: No Deepgram SDK client found for"
                                    " {call_sid}."
                                )
                                break
                            else:
                                logger.info(
                                    "Twilio WS: Found Deepgram SDK client for"
                                    f" {call_sid}."
                                )
                        elif event == "media":
                            if not deepgram_live_client:
                                continue  # Should not happen if start worked
                            payload = data.get("media", {}).get("payload")
                            if payload:
                                audio_bytes = base64.b64decode(payload)
                                await deepgram_live_client.send(
                                    audio_bytes
                                )  # Send audio to Deepgram SDK
                        elif event == "stop":
                            logger.info("Twilio WS: Stop event for {call_sid}.")
                            break
                        elif event == "error":
                            logger.error(
                                "Twilio WS: Error event for {call_sid}:"
                                f" {data.get('error')}"
                            )
                            break
                except websockets.exceptions.ConnectionClosedOK:
                    logger.info(
                        "Twilio WS: Connection closed normally for {call_sid}."
                    )
                except Exception as e:
                    logger.error(
                        "Twilio WS: Unexpected error for {call_sid}: {e}",
                        exc_info=True,
                    )
                finally:
                    logger.info("Twilio WS: Cleaning up for {call_sid}")
                    # SDK client finish/unregister is handled in VoiceSalesAgent.handle_call
                    # finally block
                    if call_sid:
                        await self.unregister_deepgram_connection_sdk(call_sid)  # Ensure unregister if loop breaks unexpectedly

            @self.app.route("/track/<tracking_id>.png")
            async def handle_tracking_pixel(tracking_id):
                logger.info("Tracking pixel hit: {tracking_id}")
                email_agent = self.agents.get("email")
                if email_agent and hasattr(email_agent, "process_email_open"):
                    asyncio.create_task(email_agent.process_email_open(tracking_id))
                else:
                    logger.warning(
                        "EmailAgent not found or cannot process opens for tracking ID:"
                        f" {tracking_id}"
                    )
                pixel_data = base64.b64decode(
                    "R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
                )
                return (
                    pixel_data,
                    200,
                    {
                        "Content-Type": "image/gif",
                        "Cache-Control": "no-cache, no-store, must-revalidate",
                        "Pragma": "no-cache",
                        "Expires": "0",
                    },
                )

            logger.info("Quart routes configured.")

        # --- Main Execution Loop ---
        async def run(self):
            """Initializes and runs the AI Agency Orchestrator."""

            logger.info("Orchestrator starting full initialization sequence...")
            self.running = False
            try:
                if not await self.initialize_database():
                    raise RuntimeError("Database initialization failed.")
                if not await self.initialize_clients():
                    raise RuntimeError("LLM Client initialization failed.")
                if not await self.initialize_agents():
                    raise RuntimeError("Agent initialization failed.")
                logger.info("Orchestrator initialization complete.")

                logger.info(
                    "Starting background agent run loops and periodic tasks..."
                )
                self.background_tasks = set()
                for agent_name, agent in self.agents.items():
                    if hasattr(agent, "run") and callable(agent.run):
                        task = asyncio.create_task(
                            agent.run(), name=f"AgentLoop_{agent_name}"
                        )
                        self.background_tasks.add(task)
                    else:
                        logger.warning(
                            "Agent {agent_name} does not have a callable run method."
                        )

                self.background_tasks.add(
                    asyncio.create_task(
                        self._run_periodic_data_purge(), name="PeriodicDataPurge"
                    )
                )
                self.background_tasks.add(
                    asyncio.create_task(
                        self._run_periodic_feedback_collection(),
                        name="PeriodicFeedback",
                    )
                )
                logger.info("Started {len(self.background_tasks)} background tasks.")

                logger.info(
                    "Orchestrator entering main operational state (API/Event driven)."
                )
                self.running = True
                self.last_feedback_time = time.time()
                self.last_purge_time = time.time()

                while self.running:
                    try:
                        tasks_to_remove = set()
                        for task in self.background_tasks:
                            if task.done():
                                tasks_to_remove.add(task)
                                try:
                                    task.result()
                                    logger.info(
                                        "Background task {task.get_name()} completed."
                                    )
                                except asyncio.CancelledError:
                                    logger.info(
                                        "Background task {task.get_name()} was"
                                        " cancelled."
                                    )
                                except Exception as task_exc:
                                    logger.error(
                                        "Background task {task.get_name()} failed:"
                                        f" {task_exc}",
                                        exc_info=True,
                                    )
                        self.background_tasks -= tasks_to_remove
                        await asyncio.sleep(60)
                    except asyncio.CancelledError:
                        logger.info("Orchestrator main loop cancelled.")
                        break

                    except Exception as e:
                        logger.critical(
                            "CRITICAL ERROR during Orchestrator setup or main loop: {e}",
                            exc_info=True,
                        )
                        self.running = False
                        try:
                            await self.send_notification(
                                "CRITICAL Orchestrator Failure",
                                "Orchestrator failed: {e}",
                            )
                        except Exception as report_err:
                            logger.error(
                                "Failed send critical failure report: {report_err}"
                            )
                    finally:
                        logger.info("Orchestrator shutdown sequence initiated.")
                        self.running = False
                        cancelled_tasks = []
                        for task in self.background_tasks:
                            if task and not task.done():
                                task.cancel()
                                cancelled_tasks.append(task)
                        if cancelled_tasks:
                            logger.info(
                                "Waiting for {len(cancelled_tasks)} background tasks to"
                                " cancel..."
                            )
                            await asyncio.gather(
                                *cancelled_tasks, return_exceptions=True
                            )
                            logger.info("Background tasks cancellation complete.")
                        else:
                            logger.info(
                                "No active background tasks needed cancellation."
                            )
                        logger.info("Orchestrator shutdown complete.")