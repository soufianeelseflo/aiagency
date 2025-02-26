import logging
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OrchestratorReflector:
    def __init__(self):
        self.budget_manager = BudgetManager()  # Hardcoded $20 budget
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

    def reflect_output(self, graded_output: str, context: dict) -> str:
        """Reflect on graded orchestrator output for optimization."""
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=500):
            logging.error("Budget exceeded for reflection.")
            return '{"status": "failure", "error": "Budget of $20 exceeded"}'

        prompt = f"""
        Reflect on this graded orchestrator output:
        - Context: {json.dumps(context)}
        - Graded Output: {graded_output}
        
        Provide actionable recommendations to optimize UGC ad delivery, beat competitors in speed and cost,
        and scale client acquisition to dominate the market.
        """
        response = self.ds.query(prompt, max_tokens=500, temperature=0.3)
        logging.info("Reflected on orchestrator output.")
        return response['choices'][0]['message']['content']

if __name__ == "__main__":
    reflector = OrchestratorReflector()
    graded = '{"accuracy": 90, "cost_efficiency": 85}'
    context = {"client_id": "123", "budget": 5000}
    result = reflector.reflect_output(graded, context)
    print(result)