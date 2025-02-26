import os
import logging
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AcquisitionEngine:
    def __init__(self):
        self.budget_manager = BudgetManager()  # Hardcoded $20 budget
        self.proxy_rotator = ProxyRotator()  # Your SmartProxy setup
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)
        self.pricing = {
            'starter': 3000,
            'pro': 5000,
            'enterprise': 10000
        }

    def handle_inbound_lead(self, lead_data: dict) -> dict:
        """Process inbound leads with personalized offers."""
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.error("Budget exceeded for lead offer generation.")
            return {"status": "failure", "error": "Budget of $20 exceeded"}

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

        email_payload = {
            'to': lead_data['email'],
            'template_name': 'onboarding',
            'variables': {
                'offer_price': self.pricing[lead_data['tier']],
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

    def cold_outreach(self, company_data: dict) -> dict:
        """AI-powered personalized cold outreach."""
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.error("Budget exceeded for cold outreach.")
            return {"status": "failure", "error": "Budget of $20 exceeded"}

        message = self.ds.query(
            f"""
            Write cold email for {company_data['name']}:
            - Industry: {company_data['industry']}
            - Recent news: {company_data['news']}
            - Pain points: {company_data['pains']}
            - Offer: {self.pricing['starter']}/month
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
        'tier': 'pro'
    }
    company = {
        'name': 'RetailInc',
        'industry': 'Retail',
        'news': 'Expanding online',
        'pains': 'Low conversion rates',
        'decision_maker_email': 'dm@retailinc.com'
    }
    print(engine.handle_inbound_lead(lead))
    print(engine.cold_outreach(company))