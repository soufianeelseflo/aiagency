# soufianeelseflo-aiagency/orchestrator.py
import os
import json
import time
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from agents.email_manager import EmailManager
from agents.executor import AcquisitionEngine
from agents.voice_agent import VoiceSalesAgent
from agents.legal_compliance_agent import LegalComplianceAgent
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator
from agents.argil_automation_agent import ArgilAutomationAgent
import psycopg2
from psycopg2 import pool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

class Orchestrator:
    def __init__(self):
        self.budget_manager = BudgetManager(
            total_budget=float(os.getenv("TOTAL_BUDGET", "20.0")),
            input_cost_per_million=float(os.getenv("INPUT_COST_PER_M", "0.80")),
            output_cost_per_million=float(os.getenv("OUTPUT_COST_PER_M", "2.40"))
        )
        self.proxy_rotator = ProxyRotator()
        self.services = {
            "argil_ugc": {"endpoint": None, "cost_per_unit": 0.0, "fallback": "open_router_ugc", "rate_limit": (int(os.getenv("ARGIL_RATE_LIMIT", "50")), 60)},
            "open_router_ugc": {"endpoint": os.getenv("OPEN_ROUTER_ENDPOINT", "https://openrouter.ai/api/v1/ugc"), "cost_per_unit": float(os.getenv("OPEN_ROUTER_COST", "0.10")), "rate_limit": (int(os.getenv("OPEN_ROUTER_RATE_LIMIT", "100")), 60)}
        }
        self.db_pool = self._init_db_pool()
        self.email_manager = EmailManager()
        self.acquisition_engine = AcquisitionEngine()
        self.legal_agent = LegalComplianceAgent()
        self.deepseek_r1 = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)
        self.argil_agent = ArgilAutomationAgent()
        self.twilio_client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))
        self.my_whatsapp_number = os.getenv("WHATSAPP_NUMBER")
        if not self.my_whatsapp_number:
            raise ValueError("WHATSAPP_NUMBER must be set.")
        self.twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        self.twiml_bin_url = os.getenv("TWIML_BIN_URL")
        if not self.twiml_bin_url:
            raise ValueError("TWIML_BIN_URL must be set.")
        self.twilio_voice_number = os.getenv("TWILIO_VOICE_NUMBER")
        if not self.twilio_voice_number:
            raise ValueError("TWILIO_VOICE_NUMBER must be set.")
        self.service_usage = {service: {'count': 0, 'errors': 0} for service in self.services}
        self.last_optimization = datetime.utcnow()
        self.first_video_approved = False
        self.initialize_database()

    def _init_db_pool(self):
        """Initialize PostgreSQL per https://www.psycopg.org/docs/pool.html"""
        try:
            pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=int(os.getenv("DB_MIN_CONN", "5")),
                maxconn=int(os.getenv("DB_MAX_CONN", "20")),
                dbname=os.getenv('POSTGRES_DB', 'smma_db'),
                user=os.getenv('POSTGRES_USER', 'postgres'),
                password=os.getenv('POSTGRES_PASSWORD'),
                host=os.getenv('POSTGRES_HOST', 'postgres')
            )
            logging.info("Database pool initialized.")
            return pool
        except psycopg2.Error as e:
            logging.error(f"Failed to initialize DB pool: {str(e)}")
            raise

    def initialize_database(self):
        """Ensure tables exist."""
        create_tables_query = """
        CREATE TABLE IF NOT EXISTS api_usage (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255),
            tokens_used INT,
            cost FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS critical_alerts (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message TEXT,
            resolved BOOLEAN DEFAULT FALSE
        );
        CREATE TABLE IF NOT EXISTS optimizations (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_path VARCHAR(255),
            new_code TEXT,
            reason TEXT
        );
        CREATE TABLE IF NOT EXISTS client_interactions (
            id SERIAL PRIMARY KEY,
            client_id VARCHAR(255),
            interaction_type VARCHAR(50),
            details JSONB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_tables_query)
                conn.commit()
                self.db_pool.putconn(conn)
                logging.info("Database tables initialized.")
        except psycopg2.Error as e:
            logging.error(f"Database initialization failed: {str(e)}")
            raise

    def get_status(self):
        """Expose status."""
        return {
            "service_usage": self.service_usage,
            "last_optimization": self.last_optimization.isoformat(),
            "total_costs": self._calculate_costs(),
            "remaining_budget": self.budget_manager.get_remaining_budget(),
            "first_video_approved": self.first_video_approved,
            "remaining_emails": self.email_manager.get_remaining_emails()
        }

    def update_parameters(self, updates):
        """Update parameters per https://flask.palletsprojects.com/en/3.0.x/"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logging.info(f"Updated parameter {key} to {value}")

    def _critical_alert(self, message: str) -> None:
        """Send alerts via Twilio per https://www.twilio.com/docs/sms/api"""
        try:
            self.twilio_client.messages.create(
                body=f"@Orchestrator CRITICAL: {message}",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
            remaining_budget = self.budget_manager.get_remaining_budget()
            if remaining_budget < self.budget_manager.total_budget * 0.2:
                self.twilio_client.messages.create(
                    body=f"@Orchestrator Budget Alert: ${remaining_budget:.2f} left!",
                    from_=self.twilio_whatsapp_number,
                    to=self.my_whatsapp_number
                )
        except TwilioRestException as e:
            logging.error(f"Failed to send alert: {str(e)}")
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO critical_alerts (timestamp, message, resolved) VALUES (%s, %s, %s)",
                        (datetime.utcnow(), message, False)
                    )
                conn.commit()
                self.db_pool.putconn(conn)
        except psycopg2.Error as e:
            logging.error(f"Failed to log alert: {str(e)}")

    def _calculate_costs(self) -> float:
        """Calculate service costs."""
        total = 0.0
        for service, data in self.service_usage.items():
            total += data['count'] * self.services[service]['cost_per_unit']
        return total

    def _send_status_update(self, message: str) -> None:
        """Send real-time updates."""
        try:
            self.twilio_client.messages.create(
                body=f"@Orchestrator Update: {message}",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
            logging.info(f"Sent update: {message}")
        except TwilioRestException as e:
            logging.error(f"Failed to send update: {str(e)}")

    def _count_clients_24h(self) -> int:
        """Count clients acquired."""
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT COUNT(*) FROM api_usage 
                        WHERE timestamp > NOW() - INTERVAL '24 hours'
                        AND model_name LIKE '%client_acquisition%'
                    """)
                    count = cursor.fetchone()[0]
                self.db_pool.putconn(conn)
            return count
        except psycopg2.Error as e:
            logging.error(f"Failed to count clients: {str(e)}")
            return 0

    def _extract_phone_number(self, whatsapp_number: str) -> str:
        """Extract E.164 number."""
        if whatsapp_number.startswith("whatsapp:"):
            return whatsapp_number[len("whatsapp:"):]
        return whatsapp_number

    def _initiate_voice_call(self):
        """Initiate voice call."""
        try:
            to_number = self._extract_phone_number(self.my_whatsapp_number)
            call = self.twilio_client.calls.create(
                url=self.twiml_bin_url,
                to=to_number,
                from_=self.twilio_voice_number
            )
            logging.info(f"Initiated call to {to_number}: SID {call.sid}")
        except TwilioRestException as e:
            logging.error(f"Failed to initiate call: {str(e)}")
            self._critical_alert(f"Voice call failed: {e}")

    def _poll_whatsapp_messages(self):
        """Poll WhatsApp per https://www.twilio.com/docs/sms/api"""
        try:
            messages = self.twilio_client.messages.list(
                to=self.twilio_whatsapp_number,
                date_sent=datetime.utcnow().date(),
                limit=10
            )
            return [(msg.body, msg.from_) for msg in messages if msg.body.startswith("@")]
        except TwilioRestException as e:
            logging.error(f"Failed to poll messages: {str(e)}")
            return []

    def _handle_whatsapp_command(self, message: str, from_number: str) -> None:
        """Handle commands."""
        if from_number != self.my_whatsapp_number:
            logging.warning(f"Unauthorized attempt from {from_number}")
            self.twilio_client.messages.create(
                body="Unauthorized access.",
                from_=self.twilio_whatsapp_number,
                to=from_number
            )
            return
        if message.startswith("@videoagent"):
            command = message.split(" ", 1)[1].strip().lower() if len(message.split(" ")) > 1 else ""
            if command == "yes" and not self.first_video_approved:
                self.first_video_approved = True
                self._record_video_feedback("First video approved.")
                self.twilio_client.messages.create(
                    body="@Orchestrator First video approved! Starting operations.",
                    from_=self.twilio_whatsapp_number,
                    to=self.my_whatsapp_number
                )
            elif command == "yes this is exactly what i want":
                self._record_video_feedback("This is the ad type to create.")
                self.twilio_client.messages.create(
                    body="@Orchestrator Feedback recorded.",
                    from_=self.twilio_whatsapp_number,
                    to=self.my_whatsapp_number
                )
            else:
                self.twilio_client.messages.create(
                    body="@Orchestrator Unknown command.",
                    from_=self.twilio_whatsapp_number,
                    to=self.my_whatsapp_number
                )
        elif message == "@orchestrator call_me":
            self._initiate_voice_call()
            self.twilio_client.messages.create(
                body="@Orchestrator Initiating call.",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
        elif message == "@orchestrator begin":
            self.argil_agent.run()
            self._send_status_update("Agency initialized, video generation started!")
        else:
            self.twilio_client.messages.create(
                body="@Orchestrator Unknown command.",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
        logging.info(f"Processed command: '{message}' from {from_number}")

    def _record_video_feedback(self, feedback: str) -> None:
        """Record feedback."""
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO optimizations (timestamp, file_path, new_code, reason)
                        VALUES (%s, %s, %s, %s)
                    """, (datetime.utcnow(), "agents/argil_automation_agent.py", "", feedback))
                conn.commit()
                self.db_pool.putconn(conn)
            logging.info(f"Recorded feedback: {feedback}")
        except psycopg2.Error as e:
            logging.error(f"Failed to record feedback: {str(e)}")

    def run(self) -> None:
        """Main loop."""
        while True:
            try:
                if self.first_video_approved:
                    lead = {"company": "TechCo", "phone": "+12025550123", "industry": "Tech", "pains": "Low engagement", "email": "lead@techco.com", "decision_maker_email": "dm@techco.com", "news": "Recent funding"}
                    if not self.acquisition_engine.is_european(self.acquisition_engine.get_country_from_phone(lead["phone"])):
                        outreach_result = self.acquisition_engine.cold_outreach(lead)
                        if outreach_result["status"] == "prepared_for_email":
                            remaining = self.email_manager.get_remaining_emails()
                            if remaining > 0:
                                self.email_manager.send_campaign([lead["decision_maker_email"]], outreach_result["email_payload"]["body"])
                                self._send_status_update(f"New client acquired: {lead['company']}!")
                            else:
                                self._critical_alert("Daily email limit reached.")
                        voice_agent = VoiceSalesAgent()
                        voice_result = voice_agent.handle_lead(lead, {"response": "Interested"})
                        logging.info(f"Voice call result: {json.dumps(voice_result, indent=2)}")
                        if voice_result["outcome"] == "completed":
                            self._send_status_update(f"Client {lead['company']} confirmed via call!")
                time.sleep(300)
                logging.info("Cycle completed.")
                messages = self._poll_whatsapp_messages()
                for message_body, from_number in messages:
                    self._handle_whatsapp_command(message_body, from_number)
                if self.budget_manager.get_remaining_budget() < 10:
                    self._send_status_update(f"Budget low: ${self.budget_manager.get_remaining_budget():.2f} left!")
            except Exception as e:
                self._critical_alert(f"Main loop failure: {str(e)}")
                time.sleep(60)

@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    return jsonify(orchestrator.get_status())

@app.route('/api/update-agent', methods=['POST'])
def update_agent():
    data = request.get_json()
    updates = data.get("updates", {})
    orchestrator.update_parameters(updates)
    return jsonify({"message": "Orchestrator updated."})

@app.route('/api/initialize', methods=['POST'])
def initialize_agency():
    try:
        orchestrator.argil_agent.run()
        orchestrator._send_status_update("Agency initialized, video generation started!")
        return jsonify({"message": "Agency initialized, check WhatsApp!"})
    except Exception as e:
        logging.error(f"Initialization failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    orchestrator = Orchestrator()
    app.run(host=os.getenv("WEB_UI_HOST", "0.0.0.0"), port=int(os.getenv("WEB_UI_PORT", "5000")))