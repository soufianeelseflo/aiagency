# executor.py
import os
import logging
from typing import Dict, List
import phonenumbers
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EUROPEAN_COUNTRIES = {'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'}

class AcquisitionEngine:
    def __init__(self):
        self.budget_manager = BudgetManager(
            total_budget=float(os.getenv("TOTAL_BUDGET", 20.0)),
            input_cost_per_million=float(os.getenv("INPUT_COST_PER_M", 0.80)),
            output_cost_per_million=float(os.getenv("OUTPUT_COST_PER_M", 2.40))
        )
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)
        self.pricing = {
            'starter': int(os.getenv("PRICING_STARTER", 3000)),
            'pro': int(os.getenv("PRICING_PRO", 5000)),
            'enterprise': int(os.getenv("PRICING_ENTERPRISE", 10000))
        }

    def get_country_from_phone(self, phone_number: str) -> str:
        """Parse country code from phone number."""
        try:
            parsed = phonenumbers.parse(phone_number)
            return phonenumbers.region_code_for_number(parsed) or "Unknown"
        except phonenumbers.NumberParseException:
            logging.warning(f"Could not parse phone number: {phone_number}")
            return "Unknown"

    def is_european(self, country_code: str) -> bool:
        """Check if the country is in Europe."""
        return country_code.upper() in EUROPEAN_COUNTRIES

    def handle_inbound_lead(self, lead_data: Dict) -> Dict:
        """Process an inbound lead with personalized offers."""
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.error("Budget exceeded for lead offer generation.")
            return {"status": "failure", "error": "Budget of $20 exceeded"}

        country_code = self.get_country_from_phone(lead_data['phone'])
        if self.is_european(country_code):
            logging.info(f"Skipping inbound lead from {country_code} (European)")
            return {"status": "skipped", "reason": "European contact"}

        offer = self.ds.query(
            f"""
            Create personalized offer for:
            - Company: {lead_data['company']}
            - Industry: {lead_data['industry']}
            - Budget: {lead_data['budget']}
            - Pain Points: {lead_data['pains']}
            """,
            max_tokens=500
        )['choices'][0]['message']['content']

        tier = 'starter' if lead_data['budget'] < 4000 else 'pro' if lead_data['budget'] < 8000 else 'enterprise'
        email_payload = {
            'to': lead_data['email'],
            'template_name': 'onboarding',
            'variables': {
                'offer_price': self.pricing[tier],
                'custom_solutions': offer
            }
        }
        logging.info(f"Prepared offer for {lead_data['company']}: {offer}")
        return {
            'status': 'prepared_for_email',
            'offer': offer,
            'email_payload': email_payload,
            'next_step': 'schedule_ai_call'
        }

    def cold_outreach(self, company_data: Dict) -> Dict:
        """AI-powered personalized cold outreach."""
        country_code = self.get_country_from_phone(company_data['phone'])
        if self.is_european(country_code):
            logging.info(f"Skipping cold outreach to {company_data['name']} ({country_code})")
            return {"status": "skipped", "reason": "European contact"}

        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.error("Budget exceeded for cold outreach.")
            return {"status": "failure", "error": "Budget exceeded"}

        message = self.ds.query(
            f"""
            Write cold email for {company_data['name']}:
            - Industry: {company_data['industry']}
            - Recent news: {company_data['news']}
            - Pain points: {company_data['pains']}
            - Offer: ${self.pricing['starter']}/month
            """,
            max_tokens=500
        )['choices'][0]['message']['content']

        proxy = self.proxy_rotator.get_proxy()
        email_payload = {
            'to': company_data['decision_maker_email'],
            'subject': f"AI UGC Solutions for {company_data['name']}",
            'body': message,
            'proxy': proxy
        }
        logging.info(f"Prepared cold outreach for {company_data['name']} via proxy: {proxy}")
        return {
            'status': 'prepared_for_email',
            'email_payload': email_payload
        }

if __name__ == "__main__":
    engine = AcquisitionEngine()
    lead = {
        'company': 'TechCorp',
        'industry': 'Tech',
        'budget': 6000,
        'pains': 'High ad costs',
        'email': 'lead@techcorp.com',
        'phone': '+12025550123',
        'tier': 'pro'
    }
    company = {
        'name': 'RetailInc',
        'industry': 'Retail',
        'news': 'Expanding online',
        'pains': 'Low conversion rates',
        'decision_maker_email': 'dm@retailinc.com',
        'phone': '+12025550124'
    }
    print(json.dumps(engine.handle_inbound_lead(lead), indent=2))
    print(json.dumps(engine.cold_outreach(company), indent=2))