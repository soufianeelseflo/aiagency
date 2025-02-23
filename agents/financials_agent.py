import os
import json
import requests
from agents.grader import Grader
from agents.reflector import Reflector
from utils.order_manager import PaymentTracker

class EliteFinancials:
    def __init__(self):
        self.tracker = PaymentTracker()
        self.lemon_squeezy_api_key = os.getenv('LEMON_SQUEEZY_API_KEY')
        self.currency_rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.78}  # Example rates
        self.fee_rate = 0.029  # 2.9% fee
        self.grader = Grader()
        self.reflector = Reflector()
    
    def process_payment(self, amount: float, currency: str = "USD") -> float:
        """
        Process payments with multi-currency support and advanced fraud detection.
        """
        # Convert to USD if necessary
        amount_in_usd = amount * self.currency_rates.get(currency, 1.0)
        net_amount = amount_in_usd * (1 - self.fee_rate)
        
        # Advanced fraud check
        decision = self._fraud_check(amount_in_usd)
        
        # Grade the fraud check output
        graded_decision = self.grader.grade_output(decision, {"amount": amount_in_usd, "currency": currency})
        
        # Reflect on the graded decision
        reflected_decision = self.reflector.reflect_output(graded_decision, {"amount": amount_in_usd, "currency": currency})
        
        if json.loads(reflected_decision)['approve']:
            # Process payment via Lemon Squeezy
            response = requests.post(
                "https://api.lemonsqueezy.com/v1/payments",
                headers={"Authorization": f"Bearer {self.lemon_squeezy_api_key}"},
                json={
                    "amount": net_amount,
                    "currency": currency
                }
            )
            response.raise_for_status()
            return net_amount
        else:
            raise Exception("Payment declined due to fraud risk.")
    
    def generate_invoice(self, client_id: str, currency: str = "USD"):
        """
        Generate multi-currency invoices.
        """
        raw_invoice = self._create_invoice(client_id, currency)
        
        # Grade the invoice
        graded_invoice = self.grader.grade_output(raw_invoice, {"client_id": client_id, "currency": currency})
        
        # Reflect on the graded invoice
        reflected_invoice = self.reflector.reflect_output(graded_invoice, {"client_id": client_id, "currency": currency})
        
        return reflected_invoice
    
    def _fraud_check(self, amount_in_usd: float) -> str:
        """
        Perform fraud check using DeepSeek R1.
        """
        return requests.post(
            os.getenv("DEEPSEEK_R1_ENDPOINT"),
            headers={"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-r1",
                "messages": [
                    {"role": "system", "content": "You are an expert fraud detection system."},
                    {"role": "user", "content": f"Fraud check for ${amount_in_usd} payment."}
                ]
            }
        ).json()['choices'][0]['message']['content']
    
    def _create_invoice(self, client_id: str, currency: str) -> str:
        """
        Create an invoice using DeepSeek R1.
        """
        return requests.post(
            os.getenv("DEEPSEEK_R1_ENDPOINT"),
            headers={"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-r1",
                "messages": [
                    {"role": "system", "content": "You are an expert invoice generator."},
                    {"role": "user", "content": f"Create invoice for {client_id} in {currency}."}
                ]
            }
        ).json()['choices'][0]['message']['content']