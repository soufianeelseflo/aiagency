# scoring_agent.py
import os
import logging
import psycopg2
from psycopg2 import pool
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScoringAgent:
    def __init__(self):
        self.budget_manager = BudgetManager()
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)
        self.twilio_client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))
        self.twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        self.my_whatsapp_number = os.getenv("WHATSAPP_NUMBER")
        
        # PostgreSQL setup
        self.db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=5, maxconn=20,
            dbname=os.getenv('POSTGRES_DB', 'smma_db'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host=os.getenv('POSTGRES_HOST', 'postgres')
        )

    def fetch_agency_data(self):
        """Fetch performance data from all agents."""
        try:
            conn = self.db_pool.getconn()
            data = {}
            with conn.cursor() as cursor:
                # Video feedback
                cursor.execute("SELECT AVG(score), COUNT(*) FROM video_feedback")
                video_data = cursor.fetchone()
                data["video"] = {"avg_score": video_data[0] or 0, "count": video_data[1]}

                # Email campaigns
                cursor.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE opened) * 100.0 / COUNT(*) AS open_rate,
                        COUNT(*) FILTER (WHERE clicked) * 100.0 / COUNT(*) AS click_rate,
                        COUNT(*) FILTER (WHERE replied) * 100.0 / COUNT(*) AS reply_rate,
                        COUNT(*)
                    FROM email_campaigns
                """)
                email_data = cursor.fetchone()
                data["email"] = {
                    "open_rate": email_data[0] or 0,
                    "click_rate": email_data[1] or 0,
                    "reply_rate": email_data[2] or 0,
                    "count": email_data[3]
                }

                # Client interactions (from VoiceSalesAgent)
                cursor.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE outcome = 'completed') * 100.0 / COUNT(*) AS success_rate,
                        COUNT(*)
                    FROM client_interactions
                """)
                call_data = cursor.fetchone()
                data["calls"] = {"success_rate": call_data[0] or 0, "count": call_data[1]}

            self.db_pool.putconn(conn)
            return data
        except psycopg2.Error as e:
            logging.error(f"Failed to fetch agency data: {str(e)}")
            return {}

    def score_agency(self):
        """Score agency performance and provide optimization insights."""
        if not self.budget_manager.can_afford(input_tokens=1500, output_tokens=1000):
            logging.warning("Budget exceeded for scoring.")
            return "Budget exceeded."
        
        agency_data = self.fetch_agency_data()
        if not agency_data:
            return "No agency data available."

        prompt = f"""
        Analyze the following AI agency performance data:
        {json.dumps(agency_data, indent=2)}
        
        Provide:
        - Overall performance score (0-100) for each agent (videos, emails, calls)
        - Total agency score
        - Actionable recommendations to maximize revenue per client
        """
        response = self.ds.query(prompt, max_tokens=1000)
        analysis = response['choices'][0]['message']['content']
        logging.info(f"Agency analysis: {analysis}")
        
        # Send report via WhatsApp
        try:
            self.twilio_client.messages.create(
                body=f"Agency Performance Report: {analysis}",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
            logging.info("Sent agency report via WhatsApp.")
        except TwilioRestException as e:
            logging.error(f"Failed to send report: {str(e)}")
        
        return analysis

    def run(self):
        """Run the scoring agent periodically (e.g., daily)."""
        return self.score_agency()

if __name__ == "__main__":
    agent = ScoringAgent()
    print(agent.run())