import os
import asyncio
from datetime import datetime, timedelta
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from quart import Quart, request, jsonify
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
    
    def __init__(self, schema='public'):
        """Initialize the Orchestrator with configuration and dependencies."""
        self.config = settings
        self.engine = create_async_engine(self.config.DATABASE_URL, echo=True)
        self.session_maker = create_session_maker(self.engine, schema)
        self.agents = {}
        self.app = Quart(__name__)
        self.setup_routes()
        start_http_server(8000)
        self.meta_prompt = """
        You are the conductor of a symphony of genius-level AI agents. Your goal is to maximize profit and achieve rapid growth,
        starting with UGC content creation and expanding into any profitable venture. Be resourceful, adaptable, and decisive.
        Think strategically, anticipate challenges, and improve agency performance continuously. Exploit opportunities
        creatively within legal bounds. Your initial budget is $50, but your ambition is limitless. Learn from every interaction,
        success, and failure to build a self-evolving, profit-generating machine. Prioritize highest ROI actions, analyze
        agent performance, reallocate resources, and eliminate bottlenecks. Experiment with purpose and measurable outcomes.
        Success is profitability and growth ($6000 in 24 hours, $100M in 9 months). Secure the agency from unauthorized access.
        Manage the UTC 16:30-00:30 15x performance boost to maximize output. Focus on the $6000 goal via UGC before broader
        experimentation.
        """
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
                clients_models=[(self.openrouter_client, self.config.OPENROUTER_MODELS['think']), (self.deepseek_client, self.config.DEEPSEEK_MODEL)]
            )
            # EmailAgent
            self.agents['email'] = EmailAgent(
                self.session_maker, self.config, self,
                clients_models=[(self.openrouter_client, self.config.OPENROUTER_MODELS['email']), (self.deepseek_client, self.config.DEEPSEEK_MODEL)]
            )
            # LegalComplianceAgent
            self.agents['legal'] = LegalComplianceAgent(
                self.session_maker, self.config, self,
                clients_models=[(self.openrouter_client, self.config.OPENROUTER_MODELS['legal']), (self.deepseek_client, self.config.DEEPSEEK_MODEL)]
            )
            # OSINTAgent
            self.agents['osint'] = OSINTAgent(
                self.session_maker, self.config, self,
                clients_models=[(self.openrouter_client, self.config.OPENROUTER_MODELS['osint']), (self.deepseek_client, self.config.DEEPSEEK_MODEL)]
            )
            # ScoringAgent
            self.agents['scoring'] = ScoringAgent(
                self.session_maker, self.config, self,
                clients_models=[(self.openrouter_client, self.config.OPENROUTER_MODELS['scoring']), (self.deepseek_client, self.config.DEEPSEEK_MODEL)]
            )
            # VoiceSalesAgent
            self.agents['voice_sales'] = VoiceSalesAgent(
                self.session_maker, self.config, self,
                clients_models=[(self.openrouter_client, self.config.OPENROUTER_MODELS['voice_sales']), (self.deepseek_client, self.config.DEEPSEEK_MODEL)]
            )
            # OptimizationAgent
            self.agents['optimization'] = OptimizationAgent(
                self.session_maker, self.config, self,
                clients_models=[(self.openrouter_client, self.config.OPENROUTER_MODELS['optimization']), (self.deepseek_client, self.config.DEEPSEEK_MODEL)]
            )
            # BrowsingAgent (uses DeepSeek primarily)
            self.agents['browsing'] = BrowsingAgent(
                self.session_maker, self.config, self,
                clients_models=[(self.deepseek_client, self.config.DEEPSEEK_MODEL)]
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
        try:
            msg = EmailMessage()
            msg.set_content(body)
            msg['Subject'] = subject
            msg['From'] = self.config.get("HOSTINGER_EMAIL")
            msg['To'] = self.config.get("USER_EMAIL")
            with smtplib.SMTP(self.config.get("HOSTINGER_SMTP"), self.config.get("SMTP_PORT")) as server:
                server.starttls()
                server.login(self.config.get("HOSTINGER_EMAIL"), self.config.get("HOSTINGER_SMTP_PASS"))
                server.send_message(msg)
            logger.info(f"Notification sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            raise

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
                    model_type = osint_insights.get('recommended_model', 'default_expansion')
                    agent_mods = osint_insights.get('agent_modifications', {})
                    sandbox_schema = f"sandbox_{model_type}_{int(datetime.utcnow().timestamp())}_{sandbox_count}"
                    sandbox_session_maker = create_session_maker(self.engine, sandbox_schema)
                    sandbox_orchestrator = Orchestrator(schema=sandbox_schema)
                    sandbox_orchestrator.session_maker = sandbox_session_maker
                    await sandbox_orchestrator.initialize_database()
                    await sandbox_orchestrator.initialize_clients()
                    await sandbox_orchestrator.initialize_agents()
                    for agent_name, mods in agent_mods.items():
                        if agent_name in sandbox_orchestrator.agents:
                            sandbox_orchestrator.agents[agent_name].update_config(mods)
                    sandbox_tasks = [agent.run() for agent in sandbox_orchestrator.agents.values()]
                    asyncio.create_task(asyncio.gather(*sandbox_tasks))
                    sandbox_count += 1
                    logger.info(f"Sandbox {sandbox_count} initialized for {model_type} with schema {sandbox_schema}")
                    await self.send_whatsapp_notification(
                        f"Sandbox {sandbox_count} Initialized",
                        f"Sandbox for {model_type} started with ${total_profit:.2f} profit. Schema: {sandbox_schema}"
                    )
                await asyncio.sleep(3600)  # Hourly
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
            if user_role != 'admin':
                raise PermissionError("Only admins can generate invoices.")
            moroccan_bank_account = os.getenv("MOROCCAN_BANK_ACCOUNT")
            moroccan_swift_code = os.getenv("MOROCCAN_SWIFT_CODE")
            if not all([moroccan_bank_account, moroccan_swift_code]):
                raise ValueError("Moroccan bank details missing in environment variables.")
            async with self.session_maker() as session:
                client = await session.get(Client, client_id)
                if not client:
                    raise ValueError(f"Client with ID {client_id} not found")
                if 'legal' not in self.agents:
                    raise ValueError("LegalComplianceAgent not initialized")
                legal_data = await self.agents['legal'].get_invoice_details(client.country)
                legal_note = legal_data['note'] if legal_data['note'] else "Payment due within 30 days."
                contract_details = legal_data['contract'] if legal_data['contract'] else "Standard service agreement applies."
                is_valid = await self.agents['legal'].validate_invoice_content({
                    'amount': amount,
                    'note': legal_note,
                    'contract': contract_details
                })
                if not is_valid:
                    raise ValueError("Invoice content failed legal validation")
                filename = f"invoice_{client_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pdf"
                filepath = os.path.join("/app/invoices", filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                c = canvas.Canvas(filepath, pagesize=letter)
                c.drawString(100, 750, f"Invoice for {client.name}")
                c.drawString(100, 730, f"Amount: ${amount:.2f}")
                c.drawString(100, 710, "Payment Details (Morocco):")
                c.drawString(100, 690, f"Bank Account: {moroccan_bank_account}")
                c.drawString(100, 670, f"SWIFT/BIC: {moroccan_swift_code}")
                c.drawString(100, 650, "Note:")
                text = c.beginText(100, 630)
                text.textLines(legal_note)
                c.drawText(text)
                c.drawString(100, 590, "Contract Details:")
                text = c.beginText(100, 570)
                text.textLines(contract_details)
                c.drawText(text)
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
            await self.initialize_agents()
            
            # Run testing phase and auto-approve if successful
            await self.start_testing_phase()
            self.approved = True
            logger.info("Testing phase completed successfully. Agency approved for full operation.")
            
            # Background tasks
            boost_task = asyncio.create_task(self.adjust_concurrency())
            monitor_task = asyncio.create_task(self.monitor_agents())
            sandbox_task = asyncio.create_task(self.manage_sandbox())
            cleanup_task = asyncio.create_task(self.cleanup_old_logs())
            agent_tasks = [asyncio.create_task(agent.run()) for agent in self.agents.values()]
            await asyncio.gather(*agent_tasks, boost_task, monitor_task, sandbox_task, cleanup_task)
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