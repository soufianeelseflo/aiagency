import os
import smtplib
import re
import time
import dns.resolver
import logging
import random
import string
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, jsonify, request
from twilio.rest import Client  # Ref: https://www.twilio.com/docs/libraries/python
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

class EmailManager:
    def __init__(self):
        # SMTP setup for Hostinger (ref: https://www.hostinger.com/tutorials/how-to-use-free-hostinger-smtp)
        self.smtp_server = os.getenv('HOSTINGER_SMTP', 'smtp.hostinger.com')
        self.port = int(os.getenv('SMTP_PORT', 587))
        self.base_email = os.getenv('HOSTINGER_EMAIL', '').split('@')[0]
        self.domain = os.getenv('HOSTINGER_EMAIL', '').split('@')[1] if '@' in os.getenv('HOSTINGER_EMAIL', '') else 'yourdomain.com'
        self.password = os.getenv('HOSTINGER_SMTP_PASS')
        if not all([self.base_email, self.domain, self.password]):
            raise ValueError("HOSTINGER_EMAIL and HOSTINGER_SMTP_PASS must be set in environment variables.")
        self.sent_count = 0  # No daily limit, managed internally
        self.alias_index = 0

        # Twilio WhatsApp setup
        self.twilio_client = Client(
            os.getenv("TWILIO_SID"),
            os.getenv("TWILIO_TOKEN")
        )
        self.my_whatsapp_number = os.getenv("WHATSAPP_NUMBER")
        self.twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER")

        # DeepSeek R1 with budget tracking
        self.budget_manager = BudgetManager(total_budget=20.0, input_cost_per_million=0.80, output_cost_per_million=2.40)
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

    def create_professional_alias(self) -> str:
        """Generate a professional-looking email alias from the base email."""
        first_names = ["sarah", "mike", "emma", "john", "lisa"]
        last_names = ["lee", "taylor", "smith", "brown", "jones"]
        name = f"{random.choice(first_names)}.{random.choice(last_names)}{self.alias_index}"
        return f"{name}+{self.alias_index}@{self.domain}"  # Use subaddressing (plus addressing)

    def _validate_email(self, email: str) -> bool:
        """Ultra-strict email validation with domain existence check (ref: https://dnspython.readthedocs.io/)."""
        if not re.match(r"^[a-z0-9]+[\._-]?[a-z0-9]+@\w+\.\w{2,3}$", email):
            return False
        try:
            domain = email.split('@')[1]
            dns.resolver.resolve(domain, 'MX')
            return True
        except Exception:
            return False

    def _manage_replies(self, email: str, subject: str) -> None:
        """Route replies to a single inbox by filtering and logging."""
        # Simulate reply handling by logging and redirecting to your inbox
        logging.info(f"Reply expected from {email} for subject: {subject}. Redirecting to {self.base_email}@{self.domain}")
        # No additional service needed; replies naturally route to your Hostinger inbox

    def send_campaign(self, emails: List[str], template: str) -> dict:
        """Send a bulk email campaign with dynamic aliases, no daily limit."""
        results = {'success': 0, 'blocked': 0}

        def send_single_email(email: str):
            nonlocal results
            self.alias_index += 1  # Increment for each email, no daily cap
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
                        server.starttls()  # Ref: https://docs.python.org/3/library/smtplib.html#smtplib.SMTP.starttls
                        server.login(f"{self.base_email}@{self.domain}", self.password)
                        server.send_message(msg)
                        results['success'] += 1
                        self.sent_count += 1
                        self._manage_replies(email, msg['Subject'])
                        break
                except Exception as e:
                    if attempt == retries - 1:
                        results['blocked'] += 1
                        logging.error(f"Failed to send email to {email}: {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff

        with ThreadPoolExecutor(max_workers=10) as executor:  # Ref: https://docs.python.org/3/library/concurrent.futures.html
            executor.map(send_single_email, emails)

        return results

    def cold_outreach(self, prospects: List[dict]) -> dict:
        """AI-powered personalized cold outreach with unlimited aliases."""
        results = {'success': 0, 'failures': 0}

        def template_generator(prospect: dict) -> str:
            if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
                logging.error("Insufficient budget for email generation.")
                return "Budget depleted, please acquire a client."
            return self.ds.query(
                f"""
                Generate personalized email for {prospect['name']}:
                - Industry: {prospect['industry']}
                - Pain Points: {prospect['pain_points']}
                - Offer: AI-driven marketing solutions to boost ROI.
                """,
                max_tokens=500
            )['choices'][0]['message']['content']

        def send_single_email(prospect: dict):
            nonlocal results
            email = prospect.get('email')
            self.alias_index += 1  # Increment for each email, no daily cap
            current_alias = self.create_professional_alias()

            if not self._validate_email(email):
                results['failures'] += 1
                return

            template = template_generator(prospect)
            msg = MIMEMultipart()
            msg['From'] = current_alias
            msg['To'] = email
            msg['Subject'] = f"AI UGC Solutions for {prospect['name']}"
            msg['Reply-To'] = f"{self.base_email}@{self.domain}"  # Route replies to your inbox
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
                        self._manage_replies(email, msg['Subject'])
                        logging.info(f"Email sent to {email}.")
                        break
                except Exception as e:
                    if attempt == retries - 1:
                        results['failures'] += 1
                        logging.error(f"Failed to send email to {email}: {str(e)}")
                    time.sleep(2 ** attempt)

        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(send_single_email, prospects)

        return results

    def send_critical_alert(self, message: str) -> None:
        """Send critical alerts via WhatsApp."""
        try:
            self.twilio_client.messages.create(
                body=f"@EmailManager {message}",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
            logging.info(f"Critical alert sent: {message}")
        except Exception as e:
            logging.error(f"Failed to send WhatsApp alert: {str(e)}")

    def get_remaining_capacity(self) -> int:
        """Track remaining email capacity (no limit, managed internally)."""
        return float('inf')  # No practical limit, managed by code

    def get_status(self):
        """Expose current status of the email manager."""
        return {
            "emails_sent": self.sent_count,
            "remaining_capacity": self.get_remaining_capacity(),
            "current_alias": self.create_professional_alias(),
            "budget_remaining": self.budget_manager.get_remaining_budget()
        }

    def update_parameters(self, updates):
        """Update parameters safely."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logging.info(f"Updated parameter {key} to {value}")

# Backend API endpoints
@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    """Fetch real-time status of the email manager (ref: https://flask.palletsprojects.com/en/2.3.x/api/)."""
    email_manager = EmailManager()
    return jsonify(email_manager.get_status())

@app.route('/api/update-agent', methods=['POST'])
def update_agent():
    """Update email manager parameters via web interface."""
    data = request.get_json()
    updates = data.get("updates", {})
    email_manager = EmailManager()
    email_manager.update_parameters(updates)
    return jsonify({"message": "EmailManager updated successfully."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)