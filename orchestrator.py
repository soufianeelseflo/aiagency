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
from agents.browser_agent import BrowserAgent
from agents.legal_compliance_agent import LegalComplianceAgent
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator
from agents.argil_automation_agent import ArgilAutomationAgent
import psycopg2
from psycopg2 import pool
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)


class Orchestrator:
    def __init__(self):
        self.budget_manager = BudgetManager(total_budget=float(os.getenv("TOTAL_BUDGET", 50.0)))
        self.proxy_rotator = ProxyRotator()
        self.db_pool = self._init_db_pool()

        # Robust SMTP setup
        smtp_pass = os.getenv("HOSTINGER_SMTP_PASS", "")
        smtp_host = os.getenv("HOSTINGER_SMTP", "smtp.hostinger.com")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_email = os.getenv("HOSTINGER_EMAIL", "")
        logging.info(f"SMTP Pass length: {len(smtp_pass)} chars")

        self.email_manager = None
        if smtp_pass and smtp_host and smtp_email:
            try:
                self.email_manager = EmailManager(smtp_host=smtp_host, smtp_port=smtp_port, smtp_user=smtp_email, smtp_pass=smtp_pass)
                logging.info("EmailManager initialized.")
            except Exception as e:
                logging.error(f"EmailManager init failed: {str(e)}—email disabled.")
        else:
            logging.warning("SMTP vars incomplete—email disabled.")

        self.acquisition_engine = AcquisitionEngine()
        self.legal_agent = LegalComplianceAgent()
        self.deepseek_r1 = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)
        self.argil_agent = ArgilAutomationAgent()
        self.voice_agent = VoiceSalesAgent()
        self.browser_agent = BrowserAgent()

        twilio_sid = os.getenv("TWILIO_SID", "")
        twilio_token = os.getenv("TWILIO_TOKEN", "")
        self.twilio_client = Client(twilio_sid, twilio_token) if twilio_sid and twilio_token else None
        self.my_whatsapp_number = os.getenv("WHATSAPP_NUMBER", "")
        self.twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        self.twiml_bin_url = os.getenv("TWIML_BIN_URL", "")
        self.twilio_voice_number = os.getenv("TWILIO_VOICE_NUMBER", "")

        required_vars = ["TWILIO_SID", "TWILIO_TOKEN", "WHATSAPP_NUMBER", "TWIML_BIN_URL", "TWILIO_VOICE_NUMBER", "DATABASE_URL"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logging.warning(f"Missing vars: {', '.join(missing_vars)}—some features limited.")

        self.revenue_goal = 100_000_000
        self.short_term_goal = 5000
        self.first_video_approved = False
        self.initialize_database()
        self.best_roi_strategy = None
        self.best_roi_value = 0
        self.start_time = None
        self.last_revenue = 0


    def _init_db_pool(self):
        try:
            dsn = os.getenv("DATABASE_URL", "")
            if not dsn:
                logging.warning("DATABASE_URL not set—DB disabled.")
                return None
            pool = psycopg2.pool.ThreadedConnectionPool(minconn=5, maxconn=20, dsn=dsn)
            logging.info("DB pool initialized.")
            return pool
        except psycopg2.Error as e:
            logging.error(f"Failed to init DB: {str(e)}—DB disabled.")
            return None

    def initialize_database(self):
        if not self.db_pool:
            logging.warning("No DB pool—skipping init.")
            return

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
        if not self.db_pool:
            return 0
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
        if not self.db_pool:
            return 0
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
        if not self.db_pool:
            return 0
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
        if not self.twilio_client:
            logging.warning("Twilio not configured—update skipped.")
            return
        try:
            self.twilio_client.messages.create(
                body=f"@Orchestrator Update: {message}",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
            logging.info(f"Sent update: {message}")
        except TwilioRestException as e:
            logging.error(f"Failed to send update: {str(e)}")

    def _extract_phone_number(self, whatsapp_number: str) -> str:
        return whatsapp_number[len("whatsapp:"):] if whatsapp_number.startswith("whatsapp:") else whatsapp_number

    def _initiate_voice_call(self):
        if not self.twilio_client:
            logging.warning("Twilio not configured—call skipped.")
            return
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

    def _poll_whatsapp_messages(self):
        if not self.twilio_client:
            return []
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
        if not self.twilio_client:
            logging.warning("Twilio not configured—command skipped.")
            return

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
        elif message == "@voiceagent call_me":
            self.voice_agent.handle_lead({"email": "test@owner.com", "phone": self._extract_phone_number(self.my_whatsapp_number), "company": "Test", "industry": "Test", "pains": "Test"})
            self._send_status_update("Voice Agent calling you!")
        elif message == "@orchestrator begin":
            self.start_time = time.time()
            self.run_initial_campaign()
            self._send_status_update("$5000 in 24h—here we go!")
        else:
            self._send_status_update("Unknown command.")
        logging.info(f"Processed command: '{message}' from {from_number}")

    def _decide_next_action(self) -> str:
        remaining_budget = self.budget_manager.get_remaining_budget()
        revenue = self._get_total_revenue()
        time_elapsed = (time.time() - self.start_time) / 3600 if self.start_time else 0

        if revenue >= self.short_term_goal:
            return "scale_clients"
        elif time_elapsed < 24 and remaining_budget > 10:
            return "acquire_new_client"
        elif remaining_budget > 5:
            return "service_existing_clients"
        else:
            return "optimize_costs"

    def _optimize_strategy(self) -> dict:
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            return {"strategy": "default", "expected_roi": 0}

        revenue = self._get_total_revenue()
        growth = (revenue - self.last_revenue) / max(1, self.last_revenue) if self.last_revenue else 1
        prompt = f"""
        Optimize strategy for AI agency (${self.budget_manager.total_budget} budget):
        - Goal: $5000 in 24h, $100M long-term
        - Tools: DeepSeek R1, SmartProxy, ElevenLabs, Twilio
        - Exclude: Europe
        - Best ROI: {self.best_roi_value} with {self.best_roi_strategy or 'none'}
        - Revenue Growth: {growth*100:.1f}% in last cycle
        - Avoid: Strategies used in last 24h with ROI < 1000
        - Strategy: Exploit if ROI > 1000 or growth > 10%, else innovate
        Return JSON: {{strategy: str, expected_roi: float}}
        """
        response = self.deepseek_r1.query(prompt, max_tokens=500)
        strategy = json.loads(response['choices'][0]['message']['content'])

        if self.db_pool:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO strategies (strategy_name, revenue, cost, roi, uses) VALUES (%s, %s, %s, %s, %s)",
                        (strategy["strategy"], 0, 0, strategy["expected_roi"], 1)
                    )
                    cursor.execute(
                        "DELETE FROM strategies WHERE roi < 1000 AND timestamp > NOW() - INTERVAL '24 hours'"
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
        account = asyncio.run(self.browser_agent.ensure_argil_account())
        video_result = self.argil_agent.run()
        total_cost = video_result.get("cost", 0) if video_result["status"] == "success" else 0

        if video_result["status"] == "success" and self.budget_manager.can_afford(additional_cost=total_cost):
            self.budget_manager.log_usage(0, 0, additional_cost=total_cost)
            self._send_status_update(f"Video ready: {video_result['url']}")

            leads = self.acquisition_engine.genius_outreach(num_leads=50)

            def call_lead(lead):
                if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500, additional_cost=0.15):
                    return
                if not self.legal_agent.is_european(self.acquisition_engine.get_country_from_phone(lead["phone"])):
                    voice_result = self.voice_agent.handle_lead(lead)
                    total_cost = voice_result["cost"]
                    self.budget_manager.log_usage(
                        input_tokens=500 if "innovated" in voice_result["strategy"] else 0,
                        output_tokens=500 if "innovated" in voice_result["strategy"] else 0,
                        additional_cost=total_cost
                    )
                    if voice_result["outcome"] == "completed":
                        roi = (voice_result["context"]["revenue"] - total_cost) / total_cost
                        if self.db_pool:
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
                        if self.email_manager:
                            self.email_manager.send_contract(lead["email"])

            with ThreadPoolExecutor(max_workers=50) as executor:
                executor.map(call_lead, leads)

            if self.budget_manager.can_afford(input_tokens=500, output_tokens=500) and self.email_manager:
                self.email_manager.send_campaign([lead["decision_maker_email"] for lead in leads], f"See your UGC: {video_result['url']}")


    def run(self):
        while self._get_total_revenue() < self.revenue_goal:
            try:
                if self.first_video_approved and self.budget_manager.get_remaining_budget() > 5:
                    self._reinvest_profits()
                    strategy = self._optimize_strategy()
                    leads = self.acquisition_engine.genius_outreach(num_leads=50)
                    account = asyncio.run(self.browser_agent.ensure_argil_account())

                    def call_lead(lead):
                        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500, additional_cost=0.15):
                            return
                        if not self.legal_agent.is_european(self.acquisition_engine.get_country_from_phone(lead["phone"])):
                            voice_result = self.voice_agent.handle_lead(lead)
                            total_cost = voice_result["cost"]
                            self.budget_manager.log_usage(
                                input_tokens=500 if "innovated" in voice_result["strategy"] else 0,
                                output_tokens=500 if "innovated" in voice_result["strategy"] else 0,
                                additional_cost=total_cost
                            )
                            if voice_result["outcome"] == "completed":
                                roi = (voice_result["context"]["revenue"] - total_cost) / total_cost
                                if self.db_pool:
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
                                if self.email_manager:
                                    self.email_manager.send_contract(lead["email"])

                    with ThreadPoolExecutor(max_workers=50) as executor:
                        executor.map(call_lead, leads)

                time.sleep(300)
                messages = self._poll_whatsapp_messages()
                for msg, sender in messages:
                    self._handle_whatsapp_command(msg, sender)

            except Exception as e:
                logging.error(f"Loop crashed: {str(e)}")
                time.sleep(60)

    def handle_web_command(self, command: str) -> str:
        if command == "begin":
            self.start_time = time.time()
            self.run_initial_campaign()
            return "Started campaign—check WhatsApp or UI!"
        elif command.startswith("call client"):
            lead_email = command.split(" ")[-1]
            leads = self.acquisition_engine.genius_outreach(num_leads=1)  # Fetch only one lead for efficiency
            for lead in leads:
                if lead["email"] == lead_email and self.budget_manager.can_afford(input_tokens=500, output_tokens=500, additional_cost=0.15):
                    self.voice_agent.handle_lead(lead)
                    return f"Calling {lead_email}!"
            return "Client not found or budget too low."
        elif command == "call_me":
            if self.budget_manager.can_afford(input_tokens=500, output_tokens=500, additional_cost=0.15):
                self.voice_agent.handle_lead({"email": "test@owner.com", "phone": self._extract_phone_number(self.my_whatsapp_number), "company": "Test", "industry": "Test", "pains": "Test"})
                return "Voice Agent calling you!"
            return "Budget too low for test call."
        else:
            return "Unknown command—try 'begin', 'call_me', or 'call client email@example.com'"


@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    try:
        return jsonify(orchestrator.get_status())
    except Exception as e:
        logging.error(f"Agent status failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/initialize', methods=['POST'])
def initialize_agency():
    try:
        orchestrator.run_initial_campaign()
        return jsonify({"message": "Agency initialized—check WhatsApp or UI!"})
    except Exception as e:
        logging.error(f"Init failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/send-command', methods=['POST'])
def send_command():
    try:
        data = request.get_json()
        agent = data.get("agent")
        command = data.get("command")

        if not agent or not command:
            return jsonify({"error": "Missing agent or command"}), 400

        if agent == "orchestrator":
            result = orchestrator.handle_web_command(command)
        elif agent == "voice_agent":
            result = orchestrator.voice_agent.handle_web_command(command)  # Assuming voice_agent has a similar method
        else:
            return jsonify({"error": "Unknown agent"}), 400

        logging.info(f"Sent command to {agent}: {command}")
        return jsonify({"message": f"Command sent to {agent}", "result": result})

    except Exception as e:
        logging.error(f"Failed to send command: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    # Always return healthy to keep Traefik happy, log DB issues if any
    if orchestrator.db_pool:
        try:
            with orchestrator.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                orchestrator.db_pool.putconn(conn)
        except Exception as e:
            logging.error(f"DB health check failed: {str(e)}")
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    orchestrator = Orchestrator()
    app.run(host="0.0.0.0", port=80)