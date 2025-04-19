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

from config.settings import settings
from utils.secure_storage import SecureStorage
from utils.database import encrypt_data, decrypt_data
from twilio.rest import Client as TwilioClient
from models import Base, Client, Metric
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

    async def initialize_primary_api_key(self):
        async with self.session_maker() as session:
            result = await session.execute(
                "SELECT COUNT(*) FROM accounts WHERE service = 'openrouter.ai'"
            )
            count = result.scalar()
            if count == 0:
                primary_api_key = os.getenv("OPENROUTER_API_KEY")
                if primary_api_key:
                    account = Account(
                        service="openrouter.ai",
                        email="primary@example.com",  # Dummy email
                        password="",
                        api_key=primary_api_key,
                        phone="",
                        cookies="",
                        is_available=True
                    )
                    session.add(account)
                    await session.commit()
                    logger.info("Added primary OpenRouter API key to database")

        async def get_available_openrouter_clients(self):
            async with self.session_maker() as session:
                result = await session.execute(
                    "SELECT api_key FROM accounts WHERE service = 'openrouter.ai' AND is_available = TRUE"
                )
                api_keys = [row[0] for row in result.fetchall()]
                if not api_keys:
                    logger.warning("No available OpenRouter API keys; creating new accounts.")
                    await self.create_openrouter_accounts(5)  # Create 5 more if none available
                    result = await session.execute(
                        "SELECT api_key FROM accounts WHERE service = 'openrouter.ai' AND is_available = TRUE"
                    )
                    api_keys = [row[0] for row in result.fetchall()]
                clients = [AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1") for key in api_keys]
                return clients

        async def reset_api_key_availability(self):
            while True:
                await asyncio.sleep(86400)  # 24 hours
                async with self.session_maker() as session:
                    await session.execute(
                        "UPDATE accounts SET is_available = TRUE WHERE service = 'openrouter.ai'"
                    )
                    await session.commit()
                logger.info("Reset API key availability for OpenRouter")

        async def create_openrouter_accounts(self, num_accounts):
            for _ in range(num_accounts):
                await self.agents['browsing'].task_queue.put({'service_url': 'https://openrouter.ai'})
            logger.info(f"Queued {num_accounts} OpenRouter account creation tasks.")
        
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
        required_env_vars = [
            "HOSTINGER_EMAIL", "HOSTINGER_SMTP_PASS", "USER_EMAIL",
            "MOROCCAN_BANK_ACCOUNT", "MOROCCAN_SWIFT_CODE",
            "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
            "TWILIO_WHATSAPP_NUMBER", "USER_WHATSAPP_NUMBER",
            "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"
        ]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            asyncio.create_task(self.send_notification("Missing Settings", error_msg))
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
        
    async def initialize_agents(self):
        """Initialize all agents with their respective API clients and models."""
        try:
            # ThinkTool
            self.agents['think'] = ThinkTool(
                self.session_maker, self.config, self,
                clients_models=[(client, self.config.OPENROUTER_MODELS['think']) for client in self.openrouter_clients] + 
                            [(client, self.config.DEEPSEEK_MODEL) for client in self.deepseek_clients]
            )
            # EmailAgent
            self.agents['email'] = EmailAgent(
                self.session_maker, self.config, self,
                clients_models=[(client, self.config.OPENROUTER_MODELS['email']) for client in self.openrouter_clients] + 
                            [(client, self.config.DEEPSEEK_MODEL) for client in self.deepseek_clients]
            )
            # LegalComplianceAgent
            self.agents['legal'] = LegalComplianceAgent(
                self.session_maker, self.config, self,
                clients_models=[(client, self.config.OPENROUTER_MODELS['legal']) for client in self.openrouter_clients] + 
                            [(client, self.config.DEEPSEEK_MODEL) for client in self.deepseek_clients]
            )
            # OSINTAgent
            self.agents['osint'] = OSINTAgent(
                self.session_maker, self.config, self,
                clients_models=[(client, self.config.OPENROUTER_MODELS['osint']) for client in self.openrouter_clients] + 
                            [(client, self.config.DEEPSEEK_MODEL) for client in self.deepseek_clients]
            )
            # ScoringAgent
            self.agents['scoring'] = ScoringAgent(
                self.session_maker, self.config, self,
                clients_models=[(client, self.config.OPENROUTER_MODELS['scoring']) for client in self.openrouter_clients] + 
                            [(client, self.config.DEEPSEEK_MODEL) for client in self.deepseek_clients]
            )
            # VoiceSalesAgent
            self.agents['voice_sales'] = VoiceSalesAgent(
                self.session_maker, self.config, self,
                clients_models=[(client, self.config.OPENROUTER_MODELS['voice_sales']) for client in self.openrouter_clients] + 
                            [(client, self.config.DEEPSEEK_MODEL) for client in self.deepseek_clients]
            )
            # OptimizationAgent
            self.agents['optimization'] = OptimizationAgent(
                self.session_maker, self.config, self,
                clients_models=[(client, self.config.OPENROUTER_MODELS['optimization']) for client in self.openrouter_clients] + 
                            [(client, self.config.DEEPSEEK_MODEL) for client in self.deepseek_clients]
            )
            # BrowsingAgent (uses DeepSeek primarily)
            self.agents['browsing'] = BrowsingAgent(
                self.session_maker, self.config, self,
                clients_models=[(client, self.config.DEEPSEEK_MODEL) for client in self.deepseek_clients]
            )
            logger.info("All agents initialized successfully.")
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            await self.send_notification("Agent Initialization Failed", str(e))
            raise

    async def initialize_clients(self):
        """Initialize OpenRouter as the primary client and DeepSeek for website tasks."""
        try:
            # Initialize OpenRouter client (primary)
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                error_msg = "OPENROUTER_API_KEY not set in environment variables"
                logger.error(error_msg)
                await self.send_notification("API Key Error", error_msg)
                raise ValueError(error_msg)
            self.openrouter_client = AsyncOpenAI(
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            logger.info("OpenRouter client initialized successfully.")

            # Initialize DeepSeek client (for websites)
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if not deepseek_api_key:
                error_msg = "DEEPSEEK_API_KEY not set in environment variables"
                logger.error(error_msg)
                await self.send_notification("API Key Error", error_msg)
                raise ValueError(error_msg)
            self.deepseek_client = AsyncDeepSeekClient(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com"
            )
            logger.info("DeepSeek client initialized successfully.")
        except Exception as e:
            error_msg = f"Failed to initialize API clients: {e}"
            logger.error(error_msg)
            await self.send_notification("API Client Initialization Failed", error_msg)
            raise

    async def initialize_database(self):
        """Initialize or update the PostgreSQL database schema."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database schema initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

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
        await send_notification(subject, body, self.config)

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

    def setup_routes(self):
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

    # --- Architect-Zero: Added Voice Agent Support Methods ---

    async def register_deepgram_connection(self, call_sid: str, ws_connection: websockets.client.WebSocketClientProtocol):
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

    # --- End Voice Agent Support Methods ---

    async def adjust_concurrency(self):
        while True:
            try:
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_usage = psutil.virtual_memory().percent
                current_profit = (await self.check_funding_goal())[1]
                self.performance_history.append([cpu_usage, memory_usage, current_profit, self.concurrency_limit])
                
                if len(self.performance_history) >= 10 and not self.model_trained:
                    X = np.array([[h[0], h[1], h[2]] for h in self.performance_history])
                    y = np.array([h[3] for h in self.performance_history])
                    self.concurrency_model.fit(X, y)
                    self.model_trained = True
                    logger.info("Concurrency prediction model trained.")
                
                previous_limit = self.concurrency_limit
                if self.model_trained:
                    predicted_limit = self.concurrency_model.predict([[cpu_usage, memory_usage, current_profit]])[0]
                    self.concurrency_limit = max(5, min(15 * self.base_concurrency, int(predicted_limit)))
                else:
                    if cpu_usage < 50 and memory_usage < 50:
                        self.concurrency_limit = min(15 * self.base_concurrency, self.concurrency_limit + 5)
                    elif cpu_usage > 70 or memory_usage > 70:
                        self.concurrency_limit = max(5, self.concurrency_limit - 5)
                
                logger.info(f"Concurrency limit adjusted to {self.concurrency_limit}")
                if previous_limit != self.concurrency_limit:
                    await self.send_notification(
                        "Concurrency Limit Adjusted",
                        f"Concurrency limit changed from {previous_limit} to {self.concurrency_limit} (CPU: {cpu_usage}%, Memory: {memory_usage}%, Profit: ${current_profit:.2f})"
                    )
                await asyncio.sleep(300)  # Adjust every 5 minutes
            except Exception as e:
                logger.error(f"Concurrency adjustment failed: {e}")
                await asyncio.sleep(60)

    async def monitor_agents(self):
        while True:
            try:
                for name, agent in self.agents.items():
                    async with self.session_maker() as session:
                        metric = Metric(
                            agent_name=name,
                            timestamp=datetime.utcnow(),
                            metric_name="status",
                            value="running"
                        )
                        session.add(metric)
                        await session.commit()
                        agent_status.labels(agent_name=name).set(1)  # Prometheus metric
                logger.info("Agent status updated in PostgreSQL and Prometheus.")
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Agent monitoring failed: {e}")
                error_count.labels(agent_name=name).inc()
                await asyncio.sleep(60)

    async def manage_sandbox(self):
        sandbox_count = 0
        while True:
            try:
                is_goal_met, total_profit = await self.check_funding_goal()
                osint_insights = await self.agents['osint'].get_insights()
                recommended_scale = osint_insights.get('recommended_scale', 1)
                max_sandboxes = min(3, int(total_profit / 6000) * recommended_scale)

                if is_goal_met and sandbox_count < max_sandboxes:
                    # Clone an agent (e.g., OSINT) and tweak it
                    sandbox_schema = f"sandbox_osint_{int(datetime.utcnow().timestamp())}_{sandbox_count}"
                    sandbox_session_maker = create_session_maker(self.engine, sandbox_schema)
                    sandbox_orchestrator = Orchestrator(schema=sandbox_schema)
                    sandbox_orchestrator.session_maker = sandbox_session_maker
                    await sandbox_orchestrator.initialize_database()
                    await sandbox_orchestrator.initialize_clients()
                    await sandbox_orchestrator.initialize_agents()

                    # Modify OSINT Agent with a new tool
                    sandbox_orchestrator.agents['osint'].tool_success_rates['NewTool'] = 1.0
                    sandbox_tasks = [agent.run() for agent in sandbox_orchestrator.agents.values()]
                    asyncio.create_task(asyncio.gather(*sandbox_tasks))

                    # Evaluate performance after 1 hour
                    await asyncio.sleep(3600)
                    sandbox_profit = await sandbox_orchestrator.agents['scoring'].calculate_total_profit(sandbox_session_maker())
                    main_profit = await self.agents['scoring'].calculate_total_profit(self.session_maker())
                    if sandbox_profit > main_profit:
                        self.agents['osint'].tool_success_rates['NewTool'] = 1.0
                        logger.info(f"Promoted sandbox change: Added NewTool to OSINT Agent")
                    sandbox_count += 1
                    logger.info(f"Sandbox {sandbox_count} initialized and evaluated")
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"Sandbox management failed: {e}")
                await self.report_error("SandboxManagement", str(e))
                await asyncio.sleep(3600)

    async def check_funding_goal(self):
        try:
            async with self.session_maker() as session:
                total_profit = await self.agents['scoring'].calculate_total_profit(session)
                return total_profit >= 6000, total_profit
        except Exception as e:
            logger.error(f"Funding goal check failed: {e}")
            await self.report_error("FundingGoalCheck", str(e))
            return False, 0


    async def generate_invoice(self, client_id, amount, user_role):
        try:
            # Restrict invoice generation to admins
            if user_role != 'admin':
                raise PermissionError("Only admins can generate invoices.")

            # Initialize secure storage for retrieving secrets
            secure_storage = SecureStorage()

            # Retrieve and parse bank details stored as a JSON string
            bank_details_str = await secure_storage.get_secret("bank_details")
            if not bank_details_str:
                raise ValueError("Bank details missing in Vault.")
            bank_details = json.loads(bank_details_str)
            required_bank_fields = ["account", "swift", "name", "address"]
            if not all(key in bank_details for key in required_bank_fields):
                raise ValueError("Bank details incomplete in Vault.")

            # Access client data and legal compliance
            async with self.session_maker() as session:
                client = await session.get(Client, client_id)
                if not client:
                    raise ValueError(f"Client with ID {client_id} not found")
                if 'legal' not in self.agents:
                    raise ValueError("LegalComplianceAgent not initialized")

                legal_data = await self.agents['legal'].get_invoice_details(client.country)
                legal_note = self.config.LEGAL_NOTE
                contract_details = legal_data['contract'] if legal_data['contract'] else "Standard agreement applies."

                # Validate invoice content
                is_valid = await self.agents['legal'].validate_invoice_content({
                    'amount': amount,
                    'note': legal_note,
                    'contract': contract_details
                })
                if not is_valid:
                    raise ValueError("Invoice content failed legal validation")

                # Handle W-8BEN data for U.S. clients
                w8ben_data = None
                if client.country == "USA":
                    w8ben_data_str = await secure_storage.get_secret("w8ben_data")
                    if not w8ben_data_str:
                        raise ValueError("W-8BEN data missing for USA client")
                    w8ben_data = json.loads(w8ben_data_str)
                    required_w8_fields = ["name", "country", "tin"]
                    if not all(field in w8ben_data for field in required_w8_fields):
                        raise ValueError("W-8BEN data incomplete; requires name, country, and TIN.")

                # Generate the invoice PDF
                filename = f"invoice_{client_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pdf"
                filepath = os.path.join("/app/invoices", filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                c = canvas.Canvas(filepath, pagesize=letter)

                # Dynamic layout starting from the top
                y = 750
                c.drawString(100, y, f"Invoice for {client.name}")
                y -= 20
                c.drawString(100, y, f"Amount: ${amount:.2f}")
                y -= 20
                c.drawString(100, y, "Payment to Agency Designated Account")  # NEW LINE HERE
                y -= 20
                c.drawString(100, y, "Payment Details (Morocco):")
                y -= 20
                c.drawString(100, y, f"IBAN: {bank_details['account']}")
                y -= 20
                c.drawString(100, y, f"SWIFT/BIC: {bank_details['swift']}")
                y -= 20
                c.drawString(100, y, f"Bank: {bank_details['name']}")
                y -= 20
                c.drawString(100, y, f"Address: {bank_details['address']}")
                y -= 20

                # Add W-8BEN data for U.S. clients
                if w8ben_data:
                    c.drawString(100, y, "Tax Information (W-8BEN):")
                    y -= 20
                    c.drawString(100, y, f"Name: {w8ben_data['name']}")
                    y -= 20
                    c.drawString(100, y, f"Country: {w8ben_data['country']}")
                    y -= 20
                    c.drawString(100, y, f"TIN: {w8ben_data['tin']}")
                    y -= 20

                # Add legal note
                c.drawString(100, y, "Note:")
                text = c.beginText(100, y - 20)
                text.textLines(legal_note)
                c.drawText(text)

                # Adjust position for contract details based on legal note length
                note_lines = len(legal_note.split('\n'))
                y -= 20 + (note_lines * 20)

                # Add contract details
                c.drawString(100, y, "Contract Details:")
                text = c.beginText(100, y - 20)
                text.textLines(contract_details)
                c.drawText(text)

                # Finalize and save the PDF
                c.save()
                logger.info(f"Invoice generated: {filepath}")
                await self.send_notification(
                    "InvoiceGenerated",
                    f"Invoice for {client.name} (ID: {client_id}) for ${amount:.2f}. File: {filepath}"
                )
                return filepath

        except Exception as e:
            logger.error(f"Invoice generation failed for client {client_id}: {e}")
            await self.report_error("InvoiceGeneration", str(e))
            raise

    async def cleanup_old_logs(self):
        while True:
            try:
                async with self.session_maker() as session:
                    threshold = datetime.utcnow() - timedelta(days=30)
                    await session.execute("DELETE FROM metrics WHERE timestamp < :threshold", {"threshold": threshold})
                    await session.commit()
                    logger.info("Old logs cleaned up.")
                    await self.send_notification(
                        "Log Cleanup Completed",
                        "Old logs older than 30 days have been successfully cleaned up."
                    )
                await asyncio.sleep(86400)  # Daily cleanup
            except Exception as e:
                logger.error(f"Log cleanup failed: {e}")
                await asyncio.sleep(3600)

    async def run(self):
        try:
            await self.initialize_clients()
            await self.initialize_database()
            await self.initialize_primary_api_key()
            await self.initialize_agents()
            
            # Create initial OpenRouter accounts immediately
            await self.create_openrouter_accounts(5)  # Start with 5 accounts
            
            await self.start_testing_phase()
            self.approved = True
            logger.info("Testing phase completed successfully. Agency approved for full operation.")
            
            boost_task = asyncio.create_task(self.adjust_concurrency())
            monitor_task = asyncio.create_task(self.monitor_agents())
            sandbox_task = asyncio.create_task(self.manage_sandbox())
            cleanup_task = asyncio.create_task(self.cleanup_old_logs())
            reset_task = asyncio.create_task(self.reset_api_key_availability())
            agent_tasks = [asyncio.create_task(agent.run()) for agent in self.agents.values()]
            await asyncio.gather(*agent_tasks, boost_task, monitor_task, sandbox_task, cleanup_task, reset_task)
        except Exception as e:
            logger.error(f"Agency run failed: {e}")
            await self.report_error("Orchestrator", str(e))
            raise

        while True:
            try:
                feedback_data = {}
                for agent_name, agent in self.agents.items():
                    if hasattr(agent, 'collect_insights'):
                        feedback_data[agent_name] = await agent.collect_insights()
                if feedback_data:
                    validated_feedback = await self.agents['think'].process_feedback(feedback_data)
                    for agent_name, feedback in validated_feedback.items():
                        if agent_name in self.agents and feedback:
                            await self.agents[agent_name].apply_insights(feedback)
                    logger.info("Feedback distributed to agents")
                    await self.send_notification(
                        "Feedback Cycle Complete",
                        f"Distributed feedback to {len(validated_feedback)} agents"
                    )
                await asyncio.sleep(86400)  # Daily cycle
            except Exception as e:
                logger.error(f"Feedback loop failed: {e}")
                await self.report_error("FeedbackLoop", str(e))
                await asyncio.sleep(3600)

if __name__ == "__main__":
    orchestrator = Orchestrator()
    asyncio.run(orchestrator.run())
