# financials_agent.py
import os
import json
import logging
import requests
from typing import Dict
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EliteFinancials:
    def __init__(self):
        # Lemon Squeezy API setup (ref: https://docs.lemonsqueezy.com/api/payments)
        self.lemon_squeezy_api_key = os.getenv('LEMON_SQUEEZY_API_KEY')
        if not self.lemon_squeezy_api_key:
            raise ValueError("LEMON_SQUEEZY_API_KEY must be set in environment variables.")
        
        # Currency rates (configurable via env for flexibility)
        self.currency_rates = {
            "USD": 1.0,
            "EUR": float(os.getenv("EUR_RATE", 0.92)),
            "GBP": float(os.getenv("GBP_RATE", 0.78)),
            "AUD": float(os.getenv("AUD_RATE", 1.48))
        }
        self.fee_rate = float(os.getenv("PAYMENT_FEE_RATE", 0.029))  # 2.9% Lemon Squeezy fee
        
        # Budget manager for DeepSeek R1 usage
        self.budget_manager = BudgetManager(
            total_budget=float(os.getenv("TOTAL_BUDGET", 20.0)),
            input_cost_per_million=float(os.getenv("INPUT_COST_PER_M", 0.80)),
            output_cost_per_million=float(os.getenv("OUTPUT_COST_PER_M", 2.40))
        )
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

    def process_payment(self, amount: float, currency: str = "USD") -> Dict:
        """Process payments with Lemon Squeezy and fraud checking."""
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.error("Budget exceeded for payment processing.")
            return {"status": "failure", "error": "Insufficient budget for fraud check."}
        
        amount_in_usd = amount * self.currency_rates.get(currency, 1.0)
        net_amount = amount_in_usd * (1 - self.fee_rate)

        fraud_prompt = f"Fraud check for ${amount_in_usd} payment in {currency}."
        decision = json.loads(self.ds.query(fraud_prompt, max_tokens=500)['choices'][0]['message']['content'])
        if not decision.get('approve', False):
            logging.warning(f"Payment of ${amount} {currency} declined due to fraud risk.")
            return {"status": "declined", "reason": "Fraud risk detected"}

        headers = {"Authorization": f"Bearer {self.lemon_squeezy_api_key}", "Content-Type": "application/json"}
        payload = {
            "data": {
                "type": "payments",
                "attributes": {
                    "amount": int(net_amount * 100),  # Convert to cents
                    "currency": currency
                }
            }
        }
        proxy = self.proxy_rotator.get_proxy()
        try:
            response = requests.post(
                "https://api.lemonsqueezy.com/v1/payments",
                headers=headers,
                json=payload,
                proxies={"http": proxy, "https": proxy},
                timeout=10  # Ref: https://requests.readthedocs.io/en/latest/user/advanced/#timeouts
            )
            response.raise_for_status()
            payment_data = response.json().get("data", {})
            payment_id = payment_data.get("id", "unknown")
            logging.info(f"Payment processed: ${net_amount} {currency}, Payment ID: {payment_id}")
            return {"status": "success", "net_amount": net_amount, "payment_id": payment_id}
        except requests.RequestException as e:
            logging.error(f"Payment processing failed: {str(e)}")
            return {"status": "failure", "error": str(e)}

    def generate_invoice(self, client_id: str, currency: str = "USD") -> Dict:
        """Generate multi-currency invoices with genius-level detail."""
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.error("Budget exceeded for invoice generation.")
            return {"status": "failure", "error": "Insufficient budget for invoice generation"}
        
        prompt = f"""
        Generate a multi-currency invoice for {client_id} in {currency}:
        - Amount: $5000
        - Include payment terms (Net 30), due date ({(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')}),
          and a detailed breakdown of services (e.g., AI UGC campaign, support).
        - Tone: Professional, concise, genius-level clarity.
        """
        invoice = self.ds.query(prompt, max_tokens=500)['choices'][0]['message']['content']
        logging.info(f"Generated invoice for {client_id} in {currency}")
        return {"status": "success", "invoice": invoice}

if __name__ == "__main__":
    financials = EliteFinancials()
    payment = financials.process_payment(5000.0, "USD")
    print(f"Processed payment: {json.dumps(payment, indent=2)}")
    invoice = financials.generate_invoice("client123", "USD")
    print(f"Generated invoice: {json.dumps(invoice, indent=2)}")