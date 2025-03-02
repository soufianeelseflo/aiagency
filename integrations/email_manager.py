# email_manager.py
import os
import smtplib
import re
import random
import logging
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator
import dns.resolver
import psycopg2
from psycopg2 import pool
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

# Configure logging per Python's official logging docs: https://docs.python.org/3/library/logging.html
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmailManager:
    def __init__(self):
        # SMTP setup for Hostinger, using smtplib: https://docs.python.org/3/library/smtplib.html
        self.smtp_server = os.getenv('HOSTINGER_SMTP', 'smtp.hostinger.com')
        self.port = int(os.getenv('SMTP_PORT', 587))
        self.base_email = os.getenv('HOSTINGER_EMAIL', 'user@yourdomain.com').split('@')[0]
        self.domain = os.getenv('HOSTINGER_EMAIL', 'user@yourdomain.com').split('@')[1]
        self.password = os.getenv('HOSTINGER_SMTP_PASS')
        if not all([self.base_email, self.domain, self.password]):
            raise ValueError("HOSTINGER_EMAIL and HOSTINGER_SMTP_PASS must be set in environment variables.")
        self.sent_count = 0  # Total emails sent since initialization
        self.alias_index = 0  # For alias generation
        self.sent_timestamps = []  # Tracks timestamps of emails sent in last 24 hours
        self.max_daily_emails = 1000  # Your plan's limit
        self.max_recipients_per_message = 100  # To, Cc, Bcc combined limit

        # Twilio setup per official docs: https://www.twilio.com/docs/sms/quickstart/python
        self.twilio_client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))
        self.my_whatsapp_number = os.getenv("WHATSAPP_NUMBER")
        self.twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        if not all([self.twilio_client, self.my_whatsapp_number]):
            raise ValueError("TWILIO_SID, TWILIO_TOKEN, and WHATSAPP_NUMBER must be set.")

        # DeepSeek R1 with budget tracking (assumed functional as per your setup)
        self.budget_manager = BudgetManager(
            total_budget=float(os.getenv("TOTAL_BUDGET", 20.0)),
            input_cost_per_million=float(os.getenv("INPUT_COST_PER_M", 0.80)),
            output_cost_per_million=float(os.getenv("OUTPUT_COST_PER_M", 2.40))
        )
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

        # PostgreSQL setup with psycopg2: https://www.psycopg.org/docs/
        self.db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=5,
            maxconn=20,
            dbname=os.getenv('POSTGRES_DB', 'smma_db'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host=os.getenv('POSTGRES_HOST', 'postgres')
        )
        self._initialize_database()

        # RAG setup with transformers and FAISS: https://huggingface.co/docs/transformers, https://github.com/facebookresearch/faiss/wiki
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.faiss_index = faiss.IndexFlatL2(384)  # 384-dim embeddings
        self.interaction_ids = []
        self._load_rag_data()

    def _initialize_database(self):
        """Initialize PostgreSQL table for email interactions."""
        create_tables_query = """
        CREATE TABLE IF NOT EXISTS email_interactions (
            id SERIAL PRIMARY KEY,
            client_id VARCHAR(255),
            email TEXT,
            response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(create_tables_query)
            conn.commit()
            self.db_pool.putconn(conn)
            logging.info("Database tables initialized.")
        except psycopg2.Error as e:
            logging.error(f"Database initialization failed: {str(e)}")
            raise

    def _load_rag_data(self):
        """Load past email interactions into FAISS for RAG."""
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, email, response FROM email_interactions")
                rows = cursor.fetchall()
                self.interaction_ids = [row[0] for row in rows]
                embeddings = [self._get_embedding(f"{row[1]} {row[2] or ''}") for row in rows]
                if embeddings:
                    self.faiss_index.add(np.array(embeddings))
            self.db_pool.putconn(conn)
            logging.info(f"Loaded {self.faiss_index.ntotal} RAG entries for email interactions.")
        except psycopg2.Error as e:
            logging.error(f"Failed to load RAG data: {str(e)}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for RAG retrieval."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

    def _fetch_rag_interactions(self, prospect: Dict) -> str:
        """Retrieve relevant past email interactions using RAG."""
        query = f"{prospect['name']} {prospect['industry']} {prospect['pain_points']}"
        embedding = self._get_embedding(query)
        D, I = self.faiss_index.search(np.array([embedding]), k=5)  # Top 5 matches
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                valid_ids = [self.interaction_ids[i] for i in I[0] if 0 <= i < len(self.interaction_ids)]
                if valid_ids:
                    cursor.execute(
                        "SELECT email, response FROM email_interactions WHERE id IN %s",
                        (tuple(valid_ids),)
                    )
                    matches = cursor.fetchall()
                    return "\n".join([f"Past Email: {m[0]} - Response: {m[1] or 'No reply'}" for m in matches]) or ""
                return ""
        except psycopg2.Error as e:
            logging.error(f"Failed to fetch RAG interactions: {str(e)}")
            return ""
        finally:
            self.db_pool.putconn(conn)

    def create_professional_alias(self) -> str:
        """Generate a professional email alias, respecting the 50 aliases limit conceptually."""
        first_names = ["sarah", "mike", "emma", "john", "lisa", "alex", "kate"]
        last_names = ["lee", "taylor", "smith", "brown", "jones", "davis", "clark"]
        name = f"{random.choice(first_names)}.{random.choice(last_names)}{self.alias_index % 50}"
        alias = f"{name}+{self.alias_index}@{self.domain}"
        self.alias_index = (self.alias_index + 1) % 50  # Cycle within 50 for safety
        logging.debug(f"Created alias: {alias}")
        return alias

    def _validate_email(self, email: str) -> bool:
        """Validate email format and MX records per dns.resolver: https://dnspython.readthedocs.io/"""
        if not re.match(r"^[a-z0-9]+[\._-]?[a-z0-9]+@\w+\.\w{2,3}$", email):
            logging.warning(f"Invalid email format: {email}")
            return False
        try:
            domain = email.split('@')[1]
            dns.resolver.resolve(domain, 'MX')
            logging.debug(f"Email {email} is valid.")
            return True
        except dns.resolver.NXDOMAIN:
            logging.warning(f"No MX record for {email}")
            return False

    def _manage_replies(self, email: str, subject: str) -> None:
        """Log sent email for future reply tracking."""
        logging.info(f"Sent email to {email} with subject '{subject}', expecting reply to {self.base_email}@{self.domain}")

    def _clean_old_timestamps(self):
        """Remove timestamps older than 24 hours (86400 seconds)."""
        now = time.time()
        self.sent_timestamps = [ts for ts in self.sent_timestamps if now - ts < 86400]

    def can_send_email(self) -> bool:
        """Check if sending another email is within the daily limit."""
        self._clean_old_timestamps()
        return len(self.sent_timestamps) < self.max_daily_emails

    def get_remaining_emails(self) -> int:
        """Return the number of emails that can still be sent today."""
        self._clean_old_timestamps()
        return self.max_daily_emails - len(self.sent_timestamps)

    def send_campaign(self, emails: List[str], template: str) -> Dict:
        """
        Send a bulk email campaign, respecting daily limits and recipient caps.
        Uses ThreadPoolExecutor: https://docs.python.org/3/library/concurrent.futures.html
        """
        results = {'success': 0, 'blocked': 0}
        remaining = self.get_remaining_emails()
        if remaining <= 0:
            logging.error("Daily email limit of 1,000 reached. Cannot send campaign.")
            results['blocked'] = len(emails)
            return results
        emails_to_send = emails[:remaining]  # Cap at remaining daily limit

        def send_single_email(email: str):
            nonlocal results
            if not self.can_send_email():
                logging.error(f"Daily limit reached while processing {email}.")
                results['blocked'] += 1
                return
            self.alias_index += 1
            current_alias = self.create_professional_alias()

            if not self._validate_email(email):
                logging.warning(f"Invalid email: {email}")
                results['blocked'] += 1
                return

            msg = MIMEMultipart()
            msg['From'] = current_alias
            msg['To'] = email  # Single recipient per message to comply with 100 limit
            msg['Subject'] = "AI Marketing Solutions"
            msg.attach(MIMEText(template, 'plain'))

            retries = 3
            for attempt in range(retries):
                try:
                    with smtplib.SMTP(self.smtp_server, self.port) as server:
                        server.starttls()
                        server.login(f"{self.base_email}@{self.domain}", self.password)
                        server.send_message(msg)
                        results['success'] += 1
                        self.sent_count += 1
                        self.sent_timestamps.append(time.time())
                        self._manage_replies(email, msg['Subject'])
                        logging.info(f"Sent email to {email} from {current_alias}")
                        break
                except smtplib.SMTPException as e:
                    if attempt == retries - 1:
                        results['blocked'] += 1
                        logging.error(f"Failed to send to {email} after {retries} tries: {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff

        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(send_single_email, emails_to_send)
        logging.info(f"Campaign completed! Results: {results}")
        return results

    def cold_outreach(self, prospects: List[Dict]) -> Dict:
        """Send personalized cold emails with RAG-enhanced templates, respecting limits."""
        results = {'success': 0, 'failures': 0}
        remaining = self.get_remaining_emails()
        if remaining <= 0:
            logging.error("Daily email limit of 1,000 reached. Cannot send cold outreach.")
            results['failures'] = len(prospects)
            return results
        prospects_to_contact = prospects[:remaining]

        def template_generator(prospect: Dict) -> str:
            if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
                logging.error("Budget exceeded for email generation.")
                return "Sorry, we’re out of budget—reach out later!"
            rag_interactions = self._fetch_rag_interactions(prospect)
            prompt = f"""
            Write a personalized cold email for {prospect['name']}:
            - Industry: {prospect['industry']}
            - Pain Points: {prospect['pain_points']}
            - Offer: AI-powered marketing to skyrocket their business
            - Tone: Friendly, clever, professional
            - Past Interactions: {rag_interactions if rag_interactions else 'No prior interactions; make it engaging.'}
            """
            response = self.ds.query(prompt, max_tokens=500)
            return response['choices'][0]['message']['content']

        def send_single_email(prospect: Dict):
            nonlocal results
            email = prospect.get('email')
            if not email:
                results['failures'] += 1
                return
            if not self.can_send_email():
                logging.error(f"Daily limit reached while processing {email}.")
                results['failures'] += 1
                return
            self.alias_index += 1
            current_alias = self.create_professional_alias()

            if not self._validate_email(email):
                results['failures'] += 1
                return

            template = template_generator(prospect)
            msg = MIMEMultipart()
            msg['From'] = current_alias
            msg['To'] = email  # Single recipient per message
            msg['Subject'] = f"AI Boost for {prospect['name']}"
            msg['Reply-To'] = f"{self.base_email}@{self.domain}"
            msg.attach(MIMEText(template, 'plain'))

            retries = 3
            for attempt in range(retries):
                try:
                    with smtplib.SMTP(self.smtp_server, self.port) as server:
                        server.starttls()
                        server.login(f"{self.base_email}@{self.domain}", self.password)
                        server.send_message(msg)
                        results['success'] += 1
                        self.sent_count += 1
                        self.sent_timestamps.append(time.time())
                        self._manage_replies(email, msg['Subject'])
                        conn = self.db_pool.getconn()
                        with conn.cursor() as cursor:
                            cursor.execute(
                                "INSERT INTO email_interactions (client_id, email, response) VALUES (%s, %s, %s)",
                                (prospect.get('name', 'anonymous'), email, None)
                            )
                        conn.commit()
                        self.db_pool.putconn(conn)
                        self._load_rag_data()
                        logging.info(f"Sent email to {email} from {current_alias}")
                        break
                except smtplib.SMTPException as e:
                    if attempt == retries - 1:
                        results['failures'] += 1
                        logging.error(f"Failed to send to {email}: {str(e)}")
                    time.sleep(2 ** attempt)

        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(send_single_email, prospects_to_contact)
        logging.info(f"Cold outreach completed! Results: {results}")
        return results

    def send_critical_alert(self, message: str) -> None:
        """Send an alert via WhatsApp."""
        try:
            self.twilio_client.messages.create(
                body=f"@EmailManager ALERT: {message}",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
            logging.info(f"Sent alert: {message}")
        except TwilioRestException as e:
            logging.error(f"Failed to send alert: {str(e)}")

    def get_status(self) -> Dict:
        """Return current status with remaining daily emails."""
        status = {
            "emails_sent_total": self.sent_count,
            "emails_sent_today": len(self.sent_timestamps),
            "remaining_emails_today": self.get_remaining_emails(),
            "current_alias": self.create_professional_alias(),
            "budget_remaining": self.budget_manager.get_remaining_budget()
        }
        logging.info(f"Status: {status}")
        return status

    def update_parameters(self, updates: Dict) -> None:
        """Update configuration parameters dynamically."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logging.info(f"Updated {key} to {value}")

if __name__ == "__main__":
    email_manager = EmailManager()
    prospects = [{"name": "TechCo", "email": "lead@techco.com", "industry": "Tech", "pain_points": "Low ROI"}]
    import json
    print(json.dumps(email_manager.cold_outreach(prospects), indent=2))