import logging
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OrchestratorGrader:
    def __init__(self):
        self.budget_manager = BudgetManager()  # Hardcoded $20 budget
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

    def grade_output(self, output: dict, context: dict) -> str:
        """Grade orchestrator output."""
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=500):
            logging.error("Budget exceeded for grading.")
            return '{"status": "failure", "error": "Budget of $20 exceeded"}'

        prompt = f"""
        Grade the following orchestrator output:
        - Context: {json.dumps(context)}
        - Output: {json.dumps(output)}
        
        Criteria:
        1. Accuracy (0–100): Does it address the purpose?
        2. Relevance (0–100): Aligned with client needs?
        3. Cost Efficiency (0–100): Minimizes costs?
        
        Provide a final score and recommendations.
        """
        response = self.ds.query(prompt, max_tokens=500)
        logging.info("Graded orchestrator output.")
        return response['choices'][0]['message']['content']

if __name__ == "__main__":
    grader = OrchestratorGrader()
    output = {"status": "success", "content": "Generated UGC"}
    context = {"client_id": "123", "budget": 5000}
    result = grader.grade_output(output, context)
    print(result)