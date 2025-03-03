# soufianeelseflo-aiagency/orchestrator.py
import os
import json
import time
import logging
from datetime import datetime
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
        self.budget_manager = BudgetManager(total_budget=50.0)
        self.proxy_rotator = ProxyRotator()
        self.db_pool = self._init_db_pool()
        self.email_manager = EmailManager()
        self.acquisition_engine = AcquisitionEngine()
        self.legal_agent = LegalComplianceAgent()
        self.deepseek_r1 = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)
        self.argil_agent = ArgilAutomationAgent()
        self.voice_agent = VoiceSalesAgent()
        self.twilio_client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))
        self.my_whatsapp_number = os.getenv("WHATSAPP_NUMBER")
        self.twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        self.twiml_bin_url = os.getenv("TWIML_BIN_URL")
        self.twilio_voice_number = os.getenv("TWILIO_VOICE_NUMBER")
        for var in ["TWILIO_SID", "TWILIO_TOKEN", "WHATSAPP_NUMBER", "TWIML_BIN_URL", "TWILIO_VOICE_NUMBER"]:
            if not os.getenv(var):
                raise ValueError(f"{var} must be set.")
        self.revenue_goal = 100_000_000
        self.first_video_approved = False
        self.initialize_database()
        self.best_roi_strategy = None
        self.best_roi_value = 0
        self.start_time = None
        self.last_revenue = 0

    def _init_db_pool(self):
        try:
            pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=5, maxconn=20,
                dbname=os.getenv('POSTGRES_DB', 'smma_db'),
                user=os.getenv('POSTGRES_USER', 'postgres'),
                password=os.getenv('POSTGRES_PASSWORD'),
                host=os.getenv('POSTGRES_HOST', 'postgres')
            )
            logging.info("DB pool initialized.")
            return pool
        except psycopg2.Error as e:
            logging.error(f"Failed to init DB: {str(e)}")
            raise

    def initialize_database(self):
        create_tables_query = """
        CREATE TABLE IF NOT EXISTS client_interactions (
            id SERIAL PRIMARY KEY,
            client_id VARCHAR(255),
            interaction_type VARCHAR(50),
            details JSONB,
            revenue FLOAT DEFAULT 0,
            cost FLOAT DEFAULT 0,
            roi FLOAT DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS strategies (
            id SERIAL PRIMARY KEY,
            strategy_name VARCHAR(255),
            revenue FLOAT DEFAULT 0,
            cost FLOAT DEFAULT 0,
            roi FLOAT DEFAULT 0,
            uses INT DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_tables_query)
                conn.commit()
                self.db_pool.putconn(conn)
            logging.info("Database initialized.")
        except psycopg2.Error as e:
            logging.error(f"DB init failed: {str(e)}")
            raise

    def get_status(self):
        return {
            "remaining_budget": self.budget_manager.get_remaining_budget(),
            "total_revenue": self._get_total_revenue(),
            "total_cost": self._get_total_cost(),
            "best_roi_strategy": self.best_roi_strategy,
            "best_roi_value": self.best_roi_value,
            "clients_24h": self._count_clients_24h(),
            "hours_running": (time.time() - self.start_time) / 3600 if self.start_time else 0
        }

    def _get_total_revenue(self) -> float:
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT SUM(revenue) FROM client_interactions")
                    revenue = cursor.fetchone()[0] or 0
                self.db_pool.putconn(conn)
            return revenue
        except psycopg2.Error as e:
            logging.error(f"Failed to get revenue: {str(e)}")
            return 0

    def _get_total_cost(self) -> float:
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT SUM(cost) FROM client_interactions")
                    cost = cursor.fetchone()[0] or 0
                self.db_pool.putconn(conn)
            return cost
        except psycopg2.Error as e:
            logging.error(f"Failed to get cost: {str(e)}")
            return 0

    def _count_clients_24h(self) -> int:
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM client_interactions WHERE revenue > 0 AND timestamp > NOW() - INTERVAL '24 hours'")
                    count = cursor.fetchone()[0]
                self.db_pool.putconn(conn)
            return count
        except psycopg2.Error as e:
            logging.error(f"Failed to count clients: {str(e)}")
            return 0

    def _send_status_update(self, message: str) -> None:
        try:
            self.twilio_client.messages.create(
                body=f"@Orchestrator Update: {message}",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
            logging.info(f"Sent update: {message}")
        except TwilioRestException as e:
            logging.error(f"Failed to send update: {str(e)}")

    def _critical_alert(self, message: str) -> None:
        try:
            self.twilio_client.messages.create(
                body=f"@Orchestrator CRITICAL: {message}",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
            if self.budget_manager.get_remaining_budget() < 5:
                self.twilio_client.messages.create(
                    body=f"Budget Low: ${self.budget_manager.get_remaining_budget():.2f}!",
                    from_=self.twilio_whatsapp_number,
                    to=self.my_whatsapp_number
                )
        except TwilioRestException as e:
            logging.error(f"Failed to send alert: {str(e)}")

    def _extract_phone_number(self, whatsapp_number: str) -> str:
        return whatsapp_number[len("whatsapp:"):] if whatsapp_number.startswith("whatsapp:") else whatsapp_number

    def _initiate_voice_call(self):
        try:
            to_number = self._extract_phone_number(self.my_whatsapp_number)
            call = self.twilio_client.calls.create(
                url=self.twiml_bin_url,
                to=to_number,
                from_=self.twilio_voice_number
            )
            logging.info(f"Call to {to_number}: SID {call.sid}")
        except TwilioRestException as e:
            logging.error(f"Failed to call: {str(e)}")
            self._critical_alert(f"Voice call failed: {e}")

    def _poll_whatsapp_messages(self):
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
        if from_number != self.my_whatsapp_number:
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
                self._send_status_update("Video approved—full speed ahead!")
            elif command == "yes this is exactly what i want":
                self._send_status_update("Video style locked—maxing profits!")
            else:
                self._send_status_update("Unknown video command.")
        elif message == "@orchestrator call_me":
            self._initiate_voice_call()
            self._send_status_update("Calling you!")
        elif message == "@orchestrator begin":
            self.start_time = time.time()
            self.run_initial_campaign()
            self._send_status_update("$5000 in 24h—here we go!")
        else:
            self._send_status_update("Unknown command.")
        logging.info(f"Processed command: '{message}' from {from_number}")

    def _optimize_strategy(self) -> dict:
        revenue = self._get_total_revenue()
        growth = (revenue - self.last_revenue) / max(1, self.last_revenue) if self.last_revenue else 1
        self.last_revenue = revenue
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=1000):
            return {"strategy": "default", "expected_roi": 0}
        prompt = f"""
        Optimize strategy for nexusplan.store ($50 budget):
        - Goal: $5000 in 24h, $100M revenue
        - Tools: DeepSeek R1, SmartProxy, ElevenLabs, Twilio, Hostinger SMTP
        - Exclude: Europe
        - Best ROI: {self.best_roi_value} with {self.best_roi_strategy or 'none'}
        - Revenue Growth: {growth*100:.1f}% in last cycle
        - Strategy: Exploit if ROI > 1000 or growth > 10%, else innovate
        Return JSON: {{strategy: str, expected_roi: float}}
        """
        response = self.deepseek_r1.query(prompt, max_tokens=1000)
        strategy = json.loads(response['choices'][0]['message']['content'])
        with self.db_pool.getconn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO strategies (strategy_name, revenue, cost, roi, uses) VALUES (%s, %s, %s, %s, %s)",
                    (strategy["strategy"], 0, 0, strategy["expected_roi"], 1)
                )
            conn.commit()
            self.db_pool.putconn(conn)
        logging.info(f"Optimized: {strategy}")
        return strategy

    def _reinvest_profits(self):
        revenue = self._get_total_revenue()
        cost = self._get_total_cost()
        profit = revenue - cost
        if profit > 50:
            self.budget_manager.total_budget += profit
            self._send_status_update(f"Reinvested ${profit:.2f}—new budget: ${self.budget_manager.total_budget:.2f}!")
            logging.info(f"Reinvested ${profit:.2f}")

    def run_initial_campaign(self):
        video_result = self.argil_agent.run()
        if video_result["status"] == "success":
            self._send_status_update(f"Video ready: {video_result['url']}")
            leads = self.acquisition_engine.genius_outreach(num_leads=100)
            total_cost = 0
            for lead in leads[:10]:
                voice_result = self.voice_agent.handle_lead(lead)
                total_cost += voice_result["cost"]
                if voice_result["outcome"] == "completed":
                    roi = (voice_result["context"]["revenue"] - total_cost) / total_cost
                    self.best_roi_strategy = "voice_pitch"
                    self.best_roi_value = roi
                    with self.db_pool.getconn() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute(
                                "INSERT INTO client_interactions (client_id, interaction_type, details, revenue, cost, roi) "
                                "VALUES (%s, %s, %s, %s, %s, %s)",
                                (lead["email"], "voice", json.dumps(voice_result), voice_result["context"]["revenue"], total_cost, roi)
                            )
                        conn.commit()
                        self.db_pool.putconn(conn)
                    self._send_status_update(f"$5000 from {lead['company']}—ROI: {roi:.0f}x!")
                    break
            self.email_manager.send_campaign([lead["decision_maker_email"] for lead in leads], f"See your UGC: {video_result['url']}")

    def run(self):
        while self._get_total_revenue() < self.revenue_goal:
            try:
                if self.first_video_approved:
                    self._reinvest_profits()
                    strategy = self._optimize_strategy()
                    if strategy["expected_roi"] > 1000 or (self._get_total_revenue() - self.last_revenue) / max(1, self.last_revenue) > 0.1:
                        leads = self.acquisition_engine.genius_outreach(num_leads=100)
                        total_cost = 0
                        for lead in leads:
                            if not self.legal_agent.is_european(self.acquisition_engine.get_country_from_phone(lead["phone"])):
                                voice_result = self.voice_agent.handle_lead(lead)
                                total_cost += voice_result["cost"]
                                if voice_result["outcome"] == "completed":
                                    roi = (voice_result["context"]["revenue"] - total_cost) / total_cost
                                    with self.db_pool.getconn() as conn:
                                        with conn.cursor() as cursor:
                                            cursor.execute(
                                                "INSERT INTO client_interactions (client_id, interaction_type, details, revenue, cost, roi) "
                                                "VALUES (%s, %s, %s, %s, %s, %s)",
                                                (lead["email"], "voice", json.dumps(voice_result), voice_result["context"]["revenue"], total_cost, roi)
                                            )
                                        conn.commit()
                                        self.db_pool.putconn(conn)
                                    self._send_status_update(f"${voice_result['context']['revenue']} from {lead['company']}!")
                    else:
                        video_result = self.argil_agent.run()
                        if video_result["status"] == "success":
                            leads = self.acquisition_engine.genius_outreach(num_leads=100)
                            self.email_manager.send_campaign([lead["decision_maker_email"] for lead in leads], f"New UGC: {video_result['url']}")
                            self._send_status_update("Testing new video style!")
                time.sleep(300)
                messages = self._poll_whatsapp_messages()
                for msg, sender in messages:
                    self._handle_whatsapp_command(msg, sender)
            except Exception as e:
                self._critical_alert(f"Loop crashed: {str(e)}")
                time.sleep(60)

@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    return jsonify(orchestrator.get_status())

@app.route('/api/initialize', methods=['POST'])
def initialize_agency():
    try:
        orchestrator.run_initial_campaign()
        return jsonify({"message": "Agency initialized—check WhatsApp!"})
    except Exception as e:
        logging.error(f"Init failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    orchestrator = Orchestrator()
    app.run(host="0.0.0.0", port=5000)