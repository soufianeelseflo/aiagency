# email_manager_grader.py
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

class EmailManagerGrader:
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
        self._initialize_database()

    def _initialize_database(self):
        """Set up email campaigns table."""
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
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
            conn.commit()
            self.db_pool.putconn(conn)
            logging.info("Email campaigns table initialized.")
        except psycopg2.Error as e:
            logging.error(f"Database initialization failed: {str(e)}")
            raise

    def fetch_campaign_data(self):
        """Fetch recent campaign data (last 30 days for simplicity)."""
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT client_id, email, subject, sent_at, opened, clicked, replied
                    FROM email_campaigns
                    WHERE sent_at > NOW() - INTERVAL '30 days'
                    """
                )
                rows = cursor.fetchall()
            self.db_pool.putconn(conn)
            return [
                {
                    "client_id": row[0],
                    "email": row[1],
                    "subject": row[2],
                    "sent_at": row[3],
                    "opened": row[4],
                    "clicked": row[5],
                    "replied": row[6]
                } for row in rows
            ]
        except psycopg2.Error as e:
            logging.error(f"Failed to fetch campaign data: {str(e)}")
            return []

    def grade_campaigns(self):
        """Grade email campaigns and provide recommendations."""
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=500):
            logging.warning("Budget exceeded for grading.")
            return "Budget exceeded."
        
        campaign_data = self.fetch_campaign_data()
        if not campaign_data:
            return "No campaign data available."

        prompt = f"""
        Analyze the following email campaign data:
        {json.dumps(campaign_data, default=str, indent=2)}
        
        Calculate:
        - Open rate
        - Click-through rate
        - Reply rate
        
        Provide:
        - Overall performance score (0-100)
        - Actionable recommendations to improve future campaigns
        """
        response = self.ds.query(prompt, max_tokens=500)
        analysis = response['choices'][0]['message']['content']
        logging.info(f"Campaign analysis: {analysis}")
        
        # Send report via WhatsApp
        try:
            self.twilio_client.messages.create(
                body=f"Email Campaign Report: {analysis}",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
            logging.info("Sent campaign report via WhatsApp.")
        except TwilioRestException as e:
            logging.error(f"Failed to send report: {str(e)}")
        
        return analysis

    def run(self):
        """Run the grader periodically (e.g., daily)."""
        return self.grade_campaigns()

if __name__ == "__main__":
    grader = EmailManagerGrader()
    print(grader.run())