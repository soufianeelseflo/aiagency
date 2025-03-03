# soufianeelseflo-aiagency/agents/executor.py
import os
import logging
from typing import Dict, List
import phonenumbers
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EUROPEAN_COUNTRIES = {'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'}

class AcquisitionEngine:
    def __init__(self):
        self.budget_manager = BudgetManager(total_budget=50.0)
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)
        self.pricing = {'starter': 3000, 'pro': 5000, 'enterprise': 10000}

    def get_country_from_phone(self, phone_number: str) -> str:
        try:
            parsed = phonenumbers.parse(phone_number)
            return phonenumbers.region_code_for_number(parsed) or "Unknown"
        except phonenumbers.NumberParseException:
            logging.warning(f"Could not parse phone: {phone_number}")
            return "Unknown"

    def is_european(self, country_code: str) -> bool:
        return country_code.upper() in EUROPEAN_COUNTRIES

    def genius_outreach(self, num_leads: int = 100) -> List[Dict]:
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=1000):
            logging.error("Budget exceeded for outreach.")
            return []
        prompt = f"""
        Generate {num_leads} high-ROI US leads for a UGC AI agency:
        - Exclude Europe
        - Industries: E-commerce, Tech, Health ($5000+ potential)
        - Fields: company, phone (+1...), industry, pains, decision_maker_email
        - Tone: Profit-driven
        Return JSON list of leads.
        """
        proxy_cost = 0.001 * num_leads  # Proxy per lead
        token_cost = (1000 / 1_000_000 * 0.80) + (1000 / 1_000_000 * 2.40)
        total_cost = proxy_cost + token_cost
        self.budget_manager.log_usage(input_tokens=1000, output_tokens=1000, additional_cost=proxy_cost)
        response = self.ds.query(prompt, max_tokens=1000)
        leads = json.loads(response['choices'][0]['message']['content'])
        logging.info(f"Generated {len(leads)} leads, cost: ${total_cost:.4f}")
        return leads

    def cold_outreach(self, company_data: Dict) -> Dict:
        if self.is_european(self.get_country_from_phone(company_data['phone'])):
            return {"status": "skipped", "reason": "European contact"}
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            return {"status": "failure", "error": "Budget exceeded"}
        message = self.ds.query(
            f"""
            Write a cold email for {company_data['company']}:
            - Industry: {company_data['industry']}
            - Pain Points: {company_data['pains']}
            - Offer: $5000/month AI UGC
            - Tone: Friendly, profit-focused
            """,
            max_tokens=500
        )['choices'][0]['message']['content']
        proxy = self.proxy_rotator.get_proxy()
        token_cost = (500 / 1_000_000 * 0.80) + (500 / 1_000_000 * 2.40)
        proxy_cost = 0.001  # One proxy request
        total_cost = token_cost + proxy_cost
        self.budget_manager.log_usage(input_tokens=500, output_tokens=500, additional_cost=proxy_cost)
        email_payload = {
            'to': company_data['decision_maker_email'],
            'subject': f"Boost {company_data['company']} with AI UGC",
            'body': message,
            'proxy': proxy
        }
        logging.info(f"Prepared outreach for {company_data['company']}, cost: ${total_cost:.4f}")
        return {"status": "prepared_for_email", "email_payload": email_payload, "cost": total_cost}

if __name__ == "__main__":
    engine = AcquisitionEngine()
    leads = engine.genius_outreach(5)
    print(json.dumps(leads, indent=2))