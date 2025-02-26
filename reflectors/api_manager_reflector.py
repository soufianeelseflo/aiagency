import logging
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EliteEmailSystemReflector:
    def __init__(self):
        self.budget_manager = BudgetManager()  # Hardcoded $20 budget
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

    def reflect_output(self, graded_output: str, context: dict) -> str:
        """Reflect on graded email campaign output."""
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=500):
            logging.error("Budget exceeded for reflection.")
            return '{"status": "failure", "error": "Budget of $20 exceeded"}'

        prompt = f"""
        Reflect on the following graded email campaign output:
        - Context: {json.dumps(context)}
        - Graded Output: {graded_output}
        
        Provide actionable recommendations for optimization and an improved version.
        """
        response = self.ds.query(prompt, max_tokens=500, temperature=0.3)
        logging.info("Reflected on email campaign output.")
        return response['choices'][0]['message']['content']

if __name__ == "__main__":
    reflector = EliteEmailSystemReflector()
    graded = '{"accuracy": 80, "relevance": 90}'
    context = {"emails": ["test@example.com"], "template": "Boost ROI"}
    result = reflector.reflect_output(graded, context)
    print(result)