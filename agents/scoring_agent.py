# soufianeelseflo-aiagency/agents/scoring_agent.py
import os
import logging
import json
from datetime import datetime
import psycopg2
from psycopg2 import pool
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScoringAgent:
    def __init__(self):
        self.budget_manager = BudgetManager(total_budget=50.0)
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)
        self.twilio_client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))
        self.twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        self.my_whatsapp_number = os.getenv("WHATSAPP_NUMBER")
        if not all([self.twilio_client, self.my_whatsapp_number]):
            raise ValueError("TWILIO_SID, TWILIO_TOKEN, WHATSAPP_NUMBER must be set.")
        self.db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=5, maxconn=20,
            dbname=os.getenv('POSTGRES_DB', 'smma_db'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host=os.getenv('POSTGRES_HOST', 'postgres')
        )
        self._initialize_database()
        self.last_adapted = None

    def _initialize_database(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS email_campaigns (
            id SERIAL PRIMARY KEY,
            client_id VARCHAR(255),
            email TEXT,
            subject TEXT,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            opened BOOLEAN DEFAULT FALSE,
            clicked BOOLEAN DEFAULT FALSE,
            replied BOOLEAN DEFAULT FALSE
        );
        """
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_query)
                conn.commit()
                self.db_pool.putconn(conn)
            logging.info("ScoringAgent DB initialized.")
        except psycopg2.Error as e:
            logging.error(f"DB init failed: {str(e)}")
            raise

    def fetch_agency_data(self) -> dict:
        """Fetch 24-hour performance data from PostgreSQL."""
        try:
            conn = self.db_pool.getconn()
            data = {}
            with conn.cursor() as cursor:
                cursor.execute("SELECT AVG(score), COUNT(*) FROM video_feedback WHERE received_at > NOW() - INTERVAL '24 hours'")
                video_data = cursor.fetchone()
                data["video"] = {"avg_score": video_data[0] or 0, "count": video_data[1]}

                cursor.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE opened) * 100.0 / COUNT(*) AS open_rate,
                        COUNT(*) FILTER (WHERE clicked) * 100.0 / COUNT(*) AS click_rate,
                        COUNT(*) FILTER (WHERE replied) * 100.0 / COUNT(*) AS reply_rate,
                        COUNT(*)
                    FROM email_campaigns WHERE sent_at > NOW() - INTERVAL '24 hours'
                """)
                email_data = cursor.fetchone()
                data["email"] = {
                    "open_rate": email_data[0] or 0,
                    "click_rate": email_data[1] or 0,
                    "reply_rate": email_data[2] or 0,
                    "count": email_data[3]
                }

                cursor.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE outcome = 'completed') * 100.0 / COUNT(*) AS success_rate,
                        COUNT(*)
                    FROM client_interactions WHERE timestamp > NOW() - INTERVAL '24 hours'
                """)
                call_data = cursor.fetchone()
                data["calls"] = {"success_rate": call_data[0] or 0, "count": call_data[1]}

            self.db_pool.putconn(conn)
            return data
        except psycopg2.Error as e:
            logging.error(f"Failed to fetch agency data: {str(e)}")
            return {}

    def score_agency(self) -> dict:
        """Score agency performance and update strategies daily."""
        if not self.budget_manager.can_afford(input_tokens=1500, output_tokens=1000):
            logging.warning("Budget exceeded for scoring.")
            return {"error": "Budget exceeded."}
        
        now = datetime.now()
        if self.last_adapted and (now - self.last_adapted).days < 1:
            logging.info("Using cached strategies—no update today.")
            try:
                with self.db_pool.getconn() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT strategy_name, roi FROM strategies WHERE timestamp > NOW() - INTERVAL '24 hours'")
                        strategies = {row[0]: row[1] for row in cursor.fetchall()}
                    self.db_pool.putconn(conn)
                return strategies
            except psycopg2.Error as e:
                logging.error(f"Failed to fetch strategies: {str(e)}")
                return {"error": "Strategy fetch failed."}

        agency_data = self.fetch_agency_data()
        if not agency_data:
            logging.warning("No agency data available.")
            return {"error": "No agency data available."}
        
        prompt = f"""
        Analyze AI agency performance (24h data):
        {json.dumps(agency_data, indent=2)}
        Provide:
        - Scores (0-100): videos (avg_score-based), emails (rate avg), calls (success_rate-based)
        - Total agency score
        - Genius-level recommendations to maximize revenue (short, actionable)
        Return JSON: {{videos: int, emails: int, calls: int, total: int, recommendations: str}}
        """
        response = self.ds.query(prompt, max_tokens=1000)
        analysis = json.loads(response['choices'][0]['message']['content'])
        logging.info(f"Agency analysis: {analysis}")

        # Update strategies table
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO strategies (strategy_name, revenue, cost, roi) VALUES (%s, %s, %s, %s) "
                        "ON CONFLICT (strategy_name) DO UPDATE SET roi = EXCLUDED.roi, timestamp = CURRENT_TIMESTAMP",
                        ("video_strategy", 0, 0, analysis["videos"])
                    )
                    cursor.execute(
                        "INSERT INTO strategies (strategy_name, revenue, cost, roi) VALUES (%s, %s, %s, %s) "
                        "ON CONFLICT (strategy_name) DO UPDATE SET roi = EXCLUDED.roi, timestamp = CURRENT_TIMESTAMP",
                        ("email_strategy", 0, 0, analysis["emails"])
                    )
                    cursor.execute(
                        "INSERT INTO strategies (strategy_name, revenue, cost, roi, uses) VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT (strategy_name) DO UPDATE SET roi = EXCLUDED.roi, timestamp = CURRENT_TIMESTAMP",
                        ("call_strategy", 0, 0, analysis["calls"], 1)
                    )
                    cursor.execute(
                        "INSERT INTO strategies (strategy_name, revenue, cost, roi) VALUES (%s, %s, %s, %s) "
                        "ON CONFLICT (strategy_name) DO UPDATE SET roi = EXCLUDED.roi, timestamp = CURRENT_TIMESTAMP",
                        ("recommendations", 0, 0, analysis["total"])
                    )
                conn.commit()
                self.db_pool.putconn(conn)
        except psycopg2.Error as e:
            logging.error(f"Failed to update strategies: {str(e)}")

        # Send WhatsApp report
        try:
            self.twilio_client.messages.create(
                body=f"Agency Report: Videos: {analysis['videos']}, Emails: {analysis['emails']}, Calls: {analysis['calls']}, Total: {analysis['total']}. Tips: {analysis['recommendations']}",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
            logging.info("Sent agency report via WhatsApp.")
        except TwilioRestException as e:
            logging.error(f"Failed to send report: {str(e)}")

        self.last_adapted = now
        return analysis

    def run(self) -> dict:
        """Run scoring and return results."""
        return self.score_agency()

if __name__ == "__main__":
    agent = ScoringAgent()
    print(json.dumps(agent.run(), indent=2))