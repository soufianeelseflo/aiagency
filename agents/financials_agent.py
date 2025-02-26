import os
import json
import logging
import requests
from typing import Dict
from graders.financials_grader import Grader
from reflectors.financials_reflector import Reflector
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator
from integrations.deepseek_r1 import DeepSeekOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EliteFinancials:
    def __init__(self):
        self.lemon_squeezy_api_key = os.getenv('LEMON_SQUEEZY_API_KEY')
        if not self.lemon_squeezy_api_key:
            raise ValueError("LEMON_SQUEEZY_API_KEY must be set in environment variables.")
        self.currency_rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.78}  # Hardcoded for simplicity
        self.fee_rate = 0.029  # 2.9% Lemon Squeezy fee
        self.budget_manager = BudgetManager()  # Hardcoded $20 budget
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)
        self.grader = Grader()
        self.reflector = Reflector()

    def process_payment(self, amount: float, currency: str = "USD") -> float:
        """Process payments with Lemon Squeezy (ref: https://docs.lemonsqueezy.com/api/payments)."""
        amount_in_usd = amount * self.currency_rates.get(currency, 1.0)
        net_amount = amount_in_usd * (1 - self.fee_rate)

        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.error("Budget exceeded for fraud check.")
            raise Exception("Budget of $20 exceeded. Acquire a client to replenish.")

        decision = self._fraud_check(amount_in_usd)
        graded_decision = self.grader.grade_output(decision, {"amount": amount_in_usd, "currency": currency})
        reflected_decision = self.reflector.reflect_output(graded_decision, {"amount": amount_in_usd, "currency": currency})

        if json.loads(reflected_decision)['approve']:
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
                    timeout=10
                )
                response.raise_for_status()
                logging.info(f"Payment processed: ${net_amount} {currency}")
                return net_amount
            except requests.RequestException as e:
                logging.error(f"Payment processing failed: {str(e)}")
                raise Exception("Payment processing failed.")
        else:
            logging.warning("Payment declined due to fraud risk.")
            raise Exception("Payment declined due to fraud risk.")

    def generate_invoice(self, client_id: str, currency: str = "USD") -> str:
        """Generate multi-currency invoices."""
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.error("Budget exceeded for invoice generation.")
            return "Invoice generation failed: Budget exceeded."

        raw_invoice = self._create_invoice(client_id, currency)
        graded_invoice = self.grader.grade_output(raw_invoice, {"client_id": client_id, "currency": currency})
        reflected_invoice = self.reflector.reflect_output(graded_invoice, {"client_id": client_id, "currency": currency})
        return reflected_invoice

    def _fraud_check(self, amount_in_usd: float) -> str:
        """Perform fraud check with DeepSeek R1."""
        prompt = f"Fraud check for ${amount_in_usd} payment."
        response = self.ds.query(prompt, max_tokens=500)
        return response['choices'][0]['message']['content']

    def _create_invoice(self, client_id: str, currency: str) -> str:
        """Create an invoice with DeepSeek R1."""
        prompt = f"Create invoice for {client_id} in {currency}."
        response = self.ds.query(prompt, max_tokens=500)
        return response['choices'][0]['message']['content']

if __name__ == "__main__":
    financials = EliteFinancials()
    payment = financials.process_payment(5000.0, "USD")
    print(f"Processed payment: ${payment}")
    invoice = financials.generate_invoice("client123", "USD")
    print(f"Generated invoice: {invoice}")