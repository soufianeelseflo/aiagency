import os
import smtplib
import re
import time
import dns.resolver
import logging
from twilio.rest import Client  # Import Twilio client for WhatsApp integration <button class="citation-flag" data-index="1">
from typing import List
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, jsonify, request
from integrations.deepseek_r1 import DeepSeekOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)

class EmailManager:
    def __init__(self):
        # SMTP email setup for client communication
        self.smtp_server = os.getenv('HOSTINGER_SMTP')
        self.port = 587
        self.base_email = os.getenv('HOSTINGER_EMAIL').split('@')[0]
        self.domain = os.getenv('HOSTINGER_EMAIL').split('@')[1]
        self.password = os.getenv('HOSTINGER_SMTP_PASS')
        self.daily_limit = 500
        self.sent_count = 0
        self.current_alias = self.create_alias("campaign-0")
        self.alias_index = 0
        
        # Twilio WhatsApp setup for internal alerts
        self.twilio_client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )
        self.my_whatsapp_number = os.getenv("MY_WHATSAPP_NUMBER")  # Your WhatsApp number
        self.twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER")  # Twilio's WhatsApp number
        
        # DeepSeek-R1 integration
        self.ds = DeepSeekOrchestrator()

    def create_alias(self, alias_name: str) -> str:
        """Create a masked email alias dynamically."""
        return f"{alias_name}@{self.domain}"

    def _validate_email(self, email: str) -> bool:
        """Ultra-strict email validation with domain existence check."""
        if not re.match(r"^[a-z0-9]+[\._-]?[a-z0-9]+@\w+\.\w{2,3}$", email):
            return False
        try:
            domain = email.split('@')[1]
            dns.resolver.resolve(domain, 'MX')  # Validate domain's MX records
            return True
        except Exception:
            return False

    def send_campaign(self, emails: List[str], template: str) -> dict:
        """
        Fully autonomous email campaign system.
        Dynamically rotates aliases to bypass daily limits.
        Uses parallel processing for scalability.
        """
        results = {'success': 0, 'blocked': 0}

        def send_email(email: str):
            nonlocal results
            if self.sent_count >= self.daily_limit:
                # Rotate to a new alias
                self.alias_index += 1
                self.current_alias = self.create_alias(f"campaign-{self.alias_index}")
                self.sent_count = 0  # Reset counter for the new alias

            if not self._validate_email(email):
                results['blocked'] += 1
                return

            msg = MIMEMultipart()
            msg['From'] = self.current_alias
            msg['To'] = email
            msg['Subject'] = "AI Marketing Solutions"
            msg.attach(MIMEText(template, 'plain'))

            retries = 3
            for attempt in range(retries):
                try:
                    with smtplib.SMTP(self.smtp_server, self.port) as server:
                        server.starttls()  # Upgrade connection to secure TLS
                        server.login(self.base_email + "@" + self.domain, self.password)
                        server.send_message(msg)
                        results['success'] += 1
                        self.sent_count += 1
                        break
                except Exception as e:
                    if attempt == retries - 1:
                        results['blocked'] += 1
                        logging.error(f"Failed to send email to {email}: {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff for retries

        # Use parallel processing for scalability
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(send_email, emails)

        return results

    def cold_outreach(self, prospects: list) -> dict:
        """
        AI-powered personalized cold outreach via email.
        Generates personalized emails using DeepSeek-R1.
        """
        results = {'success': 0, 'failures': 0}

        def template_generator(prospect: dict) -> str:
            """Generate personalized email content using DeepSeek-R1."""
            return self.ds.query(f"""
            Generate personalized email for {prospect['name']}:
            - Industry: {prospect['industry']}
            - Pain Points: {prospect['pain_points']}
            - Offer: AI-driven marketing solutions to boost ROI.
            """)

        def send_email(prospect: dict):
            nonlocal results
            email = prospect.get('email')

            if self.sent_count >= self.daily_limit:
                # Rotate to a new alias
                self.alias_index += 1
                self.current_alias = self.create_alias(f"campaign-{self.alias_index}")
                self.sent_count = 0  # Reset counter for the new alias

            if not self._validate_email(email):
                results['failures'] += 1
                return

            template = template_generator(prospect)
            msg = MIMEMultipart()
            msg['From'] = self.current_alias
            msg['To'] = email
            msg['Subject'] = "AI UGC Solutions"
            msg.attach(MIMEText(template, 'plain'))

            retries = 3
            for attempt in range(retries):
                try:
                    with smtplib.SMTP(self.smtp_server, self.port) as server:
                        server.starttls()  # Upgrade connection to secure TLS
                        server.login(self.base_email + "@" + self.domain, self.password)
                        server.send_message(msg)
                        results['success'] += 1
                        self.sent_count += 1
                        break
                except Exception as e:
                    if attempt == retries - 1:
                        results['failures'] += 1
                        logging.error(f"Failed to send email to {email}: {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff for retries

        # Use parallel processing for scalability
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(send_email, prospects)

        return results

    def send_critical_alert(self, message: str) -> None:
        """
        Send critical alerts via WhatsApp for internal monitoring.
        """
        try:
            self.twilio_client.messages.create(
                body=f"@EmailManager {message}",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
        except Exception as e:
            logging.error(f"Failed to send WhatsApp alert: {str(e)}")

    def get_remaining_capacity(self) -> int:
        """Track remaining email capacity for the current alias."""
        return self.daily_limit - self.sent_count

    def get_status(self):
        """Expose current status of the email manager."""
        return {
            "emails_sent": self.sent_count,
            "remaining_capacity": self.get_remaining_capacity(),
            "current_alias": self.current_alias
        }

    def update_parameters(self, updates):
        """Update parameters safely."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)

# Backend API endpoints
@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    """Fetch real-time status of the email manager."""
    email_manager = EmailManager()
    return jsonify(email_manager.get_status())

@app.route('/api/update-agent', methods=['POST'])
def update_agent():
    """Update email manager parameters via web interface."""
    data = request.json
    updates = data.get("updates", {})
    email_manager = EmailManager()
    email_manager.update_parameters(updates)
    return jsonify({"message": "EmailManager updated successfully."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)