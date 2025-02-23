import os
import smtplib
import re
import time
import dns.resolver
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List
from concurrent.futures import ThreadPoolExecutor
from integrations.deepseek_r1 import DeepSeekOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)

class EliteEmailSystem:
    def __init__(self):
        self.smtp_server = os.getenv('HOSTINGER_SMTP')
        self.port = 587
        self.base_email = os.getenv('HOSTINGER_EMAIL').split('@')[0]
        self.domain = os.getenv('HOSTINGER_EMAIL').split('@')[1]
        self.password = os.getenv('HOSTINGER_SMTP_PASS')
        self.daily_limit = 500
        self.sent_count = 0
        self.current_alias = self.create_alias("campaign-0")
        self.alias_index = 0
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

    def send_elite_campaign(self, emails: List[str], template: str) -> dict:
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
        AI-powered personalized cold outreach.
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

    def get_remaining_capacity(self) -> int:
        """Track remaining email capacity for the current alias."""
        return self.daily_limit - self.sent_count