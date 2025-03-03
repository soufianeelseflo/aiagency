# soufianeelseflo-aiagency/reflectors/orchestrator_reflector.py
import logging
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OrchestratorReflector:
    """Reflects on orchestrator output for profit optimization."""
    def __init__(self):
        self.budget_manager = BudgetManager()
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

    def reflect_output(self, graded_output: str, context: dict) -> str:
        """Reflect with profit focus per https://openrouter.ai/docs."""
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=500):
            logging.error("Budget exceeded for reflection.")
            return '{"status": "failure", "error": "Budget exceeded"}'

        prompt = f"""
        Reflect on this graded orchestrator output for profit:
        - Context: {json.dumps(context)}
        - Graded Output: {graded_output}
        
        Provide actionable, profit-driven recommendations:
        - Maximize client revenue (e.g., target high-value industries).
        - Boost efficiency for cost savings.
        - Adapt to client trends using feedback.
        Return JSON with improved strategies.
        """
        response = self.ds.query(prompt, max_tokens=500, temperature=0.3)
        result = response['choices'][0]['message']['content']
        logging.info("Reflected on orchestrator output for profit.")
        return result

if __name__ == "__main__":
    reflector = OrchestratorReflector()
    graded = '{"profit_potential": 85, "avg_score": 80}'
    context = {"client_id": "123", "budget": 5000}
    print(reflector.reflect_output(graded, context))