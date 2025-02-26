import logging
from typing import Dict
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Reflector:
    def __init__(self):
        self.budget_manager = BudgetManager()  # Hardcoded $20 budget
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

    def reflect_output(self, graded_output: str, context: Dict) -> str:
        """Reflect on graded financial output."""
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.error("Budget exceeded for reflection.")
            return '{"status": "failure", "error": "Budget of $20 exceeded"}'

        prompt = f"""
        Reflect on the following graded financial output:
        - Context: {json.dumps(context)}
        - Graded Output: {graded_output}
        
        Provide actionable recommendations for optimization and an improved version.
        """
        response = self.ds.query(prompt, max_tokens=500, temperature=0.3)
        logging.info("Reflected on financial output.")
        return response['choices'][0]['message']['content']

if __name__ == "__main__":
    reflector = Reflector()
    graded = '{"accuracy": 85, "relevance": 90}'
    context = {"amount": 5000, "currency": "USD"}
    result = reflector.reflect_output(graded, context)
    print(result)