import os
import logging
import requests
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeepSeekOrchestrator:
    def __init__(self, budget_manager: BudgetManager = None, proxy_rotator: ProxyRotator = None):
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = os.getenv('OPENROUTER_API_KEY', '')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be set in environment variables.")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.budget_manager = budget_manager or BudgetManager()  # Hardcoded $20 budget
        self.proxy_rotator = proxy_rotator or ProxyRotator()    # Your SmartProxy setup

    def query(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> dict:
        """Generate content using DeepSeek R1 (ref: https://openrouter.ai/docs)."""
        input_tokens = len(prompt.split()) * 2  # Rough estimate: 2 tokens per word
        if not self.budget_manager.can_afford(input_tokens=input_tokens, output_tokens=max_tokens):
            logging.error("Budget exceeded for DeepSeek R1 query.")
            raise ValueError("Budget of $20 exceeded. Acquire a client to replenish.")

        proxy = self.proxy_rotator.get_proxy()
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",  # Updated endpoint for chat
                headers=self.headers,
                json={
                    "model": "deepseek/deepseek-r1",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                proxies={"http": proxy, "https": proxy},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            self.budget_manager.log_usage(input_tokens=input_tokens, output_tokens=max_tokens)
            logging.info("DeepSeek R1 query completed.")
            return result
        except requests.RequestException as e:
            logging.error(f"Query failed: {str(e)}")
            raise ConnectionError(f"DeepSeek R1 query failed: {str(e)}")

    def generate_email_template(self, prospect_info: dict) -> str:
        """Generate personalized email template."""
        prompt = f"""
        Generate a personalized cold email for:
        - Name: {prospect_info.get('name', 'Unknown')}
        - Company: {prospect_info.get('company', 'Unknown')}
        - Industry: {prospect_info.get('industry', 'Unknown')}
        - Pain Points: {prospect_info.get('pain_points', 'Unknown')}
        - Offer: AI-driven marketing solutions to boost ROI.
        """
        response = self.query(prompt, max_tokens=500)
        return response['choices'][0]['message']['content']

if __name__ == "__main__":
    orchestrator = DeepSeekOrchestrator()
    result = orchestrator.query("Test query")
    print(result)