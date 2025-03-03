# soufianeelseflo-aiagency/agents/email_manager.py
import os
import smtplib
import re
import random
import logging
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
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
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmailManager:
    def __init__(self):
        self.smtp_server = os.getenv('HOSTINGER_SMTP', 'smtp.hostinger.com')
        self.port = int(os.getenv('SMTP_PORT', 587))
        self.base_email = os.getenv('HOSTINGER_EMAIL', 'user@yourdomain.com').split('@')[0]
        self.domain = os.getenv('HOSTINGER_EMAIL', 'user@yourdomain.com').split('@')[1]
        self.password = os.getenv('HOSTINGER_SMTP_PASS')
        if not all([self.base_email, self.domain, self.password]):
            raise ValueError("HOSTINGER_EMAIL and HOSTINGER_SMTP_PASS must be set.")
        self.sent_count = 0
        self.alias_index = 0
        self.sent_timestamps = []
        self.max_daily_emails = 1000
        self.max_recipients_per_message = 100

        self.twilio_client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))
        self.my_whatsapp_number = os.getenv("WHATSAPP_NUMBER")
        self.twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        if not all([self.twilio_client, self.my_whatsapp_number]):
            raise ValueError("TWILIO_SID, TWILIO_TOKEN, and WHATSAPP_NUMBER must be set.")

        self.budget_manager = BudgetManager(total_budget=50.0)
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

        self.db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=5, maxconn=20,
            dbname=os.getenv('POSTGRES_DB', 'smma_db'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host=os.getenv('POSTGRES_HOST', 'postgres')
        )
        self._initialize_database()

        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.faiss_index = faiss.IndexFlatL2(384)
        self.interaction_ids = []
        self._load_rag_data()

    def _initialize_database(self):
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
            logging.info(f"Loaded {self.faiss_index.ntotal} RAG entries.")
        except psycopg2.Error as e:
            logging.error(f"Failed to load RAG: {str(e)}")

    def _get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

    def _fetch_rag_interactions(self, prospect: Dict) -> str:
        query = f"{prospect['name']} {prospect['industry']} {prospect['pain_points']}"
        embedding = self._get_embedding(query)
        D, I = self.faiss_index.search(np.array([embedding]), k=5)
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                valid_ids = [self.interaction_ids[i] for i in I[0] if 0 <= i < len(self.interaction_ids)]
                if valid_ids:
                    cursor.execute("SELECT email, response FROM email_interactions WHERE id IN %s", (tuple(valid_ids),))
                    matches = cursor.fetchall()
                    return "\n".join([f"Past: {m[0]} - Response: {m[1] or 'None'}" for m in matches]) or ""
                return ""
        except psycopg2.Error as e:
            logging.error(f"Failed to fetch RAG: {str(e)}")
            return ""
        finally:
            self.db_pool.putconn(conn)

    def create_professional_alias(self) -> str:
        first_names = ["sarah", "mike", "emma", "john", "lisa", "alex", "kate"]
        last_names = ["lee", "taylor", "smith", "brown", "jones", "davis", "clark"]
        name = f"{random.choice(first_names)}.{random.choice(last_names)}{self.alias_index % 50}"
        alias = f"{name}+{self.alias_index}@{self.domain}"
        self.alias_index = (self.alias_index + 1) % 50
        logging.debug(f"Created alias: {alias}")
        return alias

    def _validate_email(self, email: str) -> bool:
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
        logging.info(f"Sent email to {email} with subject '{subject}', expecting reply to {self.base_email}@{self.domain}")

    def _clean_old_timestamps(self):
        now = time.time()
        self.sent_timestamps = [ts for ts in self.sent_timestamps if now - ts < 86400]

    def can_send_email(self) -> bool:
        self._clean_old_timestamps()
        return len(self.sent_timestamps) < self.max_daily_emails

    def get_remaining_emails(self) -> int:
        self._clean_old_timestamps()
        return self.max_daily_emails - len(self.sent_timestamps)

    def generate_contract_pdf(self, client_email: str) -> io.BytesIO:
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            raise ValueError("Budget exceeded for contract generation.")
        prompt = f"""
        Generate a professional contract for {client_email}:
        - Service: AI-powered UGC marketing
        - Amount: $5000
        - Terms: Payment via bank wire within 7 days, services delivered post-payment
        - Tone: Formal, trustworthy
        Return plain text content.
        """
        response = self.ds.query(prompt, max_tokens=500)
        contract_text = response['choices'][0]['message']['content']
        self.budget_manager.log_usage(input_tokens=500, output_tokens=500)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph(contract_text, styles["Normal"])]
        doc.build(story)
        buffer.seek(0)
        return buffer

    def send_contract(self, client_email: str):
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.error("Budget exceeded for contract email.")
            return
        msg = MIMEMultipart()
        msg['From'] = self.create_professional_alias()
        msg['To'] = client_email
        msg['Subject'] = "Your AI Agency Service Contract"
        body = "Hello! Attached is your contract for our $5,000 AI-powered UGC marketing service. Please wire payment as outlined and reply with confirmation—videos delivered post-payment!"
        msg.attach(MIMEText(body, 'plain'))

        pdf_buffer = self.generate_contract_pdf(client_email)
        pdf_part = MIMEApplication(pdf_buffer.getvalue(), _subtype="pdf")
        pdf_part.add_header('Content-Disposition', 'attachment', filename="contract.pdf")
        msg.attach(pdf_part)

        with smtplib.SMTP(self.smtp_server, self.port) as server:
            server.starttls()
            server.login(f"{self.base_email}@{self.domain}", self.password)
            server.send_message(msg)
        logging.info(f"Sent contract to {client_email}")

    def send_campaign(self, emails: List[str], template: str) -> Dict:
        results = {'success': 0, 'blocked': 0}
        remaining = self.get_remaining_emails()
        if remaining <= 0 or not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.error("Daily email limit or budget exceeded.")
            results['blocked'] = len(emails)
            return results
        emails_to_send = emails[:remaining]
        def send_single_email(email: str):
            nonlocal results
            if not self.can_send_email():
                results['blocked'] += 1
                return
            self.alias_index += 1
            current_alias = self.create_professional_alias()
            if not self._validate_email(email):
                results['blocked'] += 1
                return
            msg = MIMEMultipart()
            msg['From'] = current_alias
            msg['To'] = email
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
                        logging.error(f"Failed to send to {email}: {str(e)}")
                    time.sleep(2 ** attempt)
        with ThreadPoolExecutor(max_workers=50) as executor:
            executor.map(send_single_email, emails_to_send)
        logging.info(f"Campaign completed! Results: {results}")
        return results

    def cold_outreach(self, prospects: List[Dict]) -> Dict:
        results = {'success': 0, 'failures': 0}
        remaining = self.get_remaining_emails()
        if remaining <= 0 or not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.error("Daily email limit or budget exceeded.")
            results['failures'] = len(prospects)
            return results
        prospects_to_contact = prospects[:remaining]
        def template_generator(prospect: Dict) -> str:
            if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
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
            self.budget_manager.log_usage(input_tokens=500, output_tokens=500)
            return response['choices'][0]['message']['content']
        def send_single_email(prospect: Dict):
            nonlocal results
            email = prospect.get('email')
            if not email:
                results['failures'] += 1
                return
            if not self.can_send_email():
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
            msg['To'] = email
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
        with ThreadPoolExecutor(max_workers=50) as executor:
            executor.map(send_single_email, prospects_to_contact)
        logging.info(f"Cold outreach completed! Results: {results}")
        return results

    def send_critical_alert(self, message: str) -> None:
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
        status = {
            "emails_sent_total": self.sent_count,
            "emails_sent_today": len(self.sent_timestamps),
            "remaining_emails_today": self.get_remaining_emails(),
            "current_alias": self.create_professional_alias(),
            "budget_remaining": self.budget_manager.get_remaining_budget()
        }
        logging.info(f"Status: {status}")
        return status

if __name__ == "__main__":
    email_manager = EmailManager()
    prospects = [{"name": "TechCo", "email": "lead@techco.com", "industry": "Tech", "pain_points": "Low ROI"}]
    import json
    print(json.dumps(email_manager.cold_outreach(prospects), indent=2))