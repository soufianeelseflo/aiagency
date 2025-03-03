# soufianeelseflo-aiagency/graders/orchestrator_grader.py
import logging
import json
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OrchestratorGrader:
    """Grades orchestrator output for profit and performance."""
    def __init__(self):
        self.budget_manager = BudgetManager()
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

    def grade_output(self, output: dict, context: dict) -> str:
        """Grade with profit focus per https://openrouter.ai/docs."""
        if not self.budget_manager.can_afford(input_tokens=1500, output_tokens=800):
            logging.error("Budget exceeded for grading.")
            return '{"status": "failure", "error": "Budget exceeded"}'
        
        prompt = f"""
        Grade this orchestrator output for profit potential:
        - Context: {json.dumps(context)}
        - Output: {json.dumps(output)}
        
        Criteria (0-100 each):
        1. Profit Potential: Will this drive client revenue?
        2. Client Appeal: Does it attract high-value clients?
        3. Efficiency: Minimizes costs while maximizing impact?
        4. Adaptability: Can it adjust to market trends?
        
        Return JSON with scores, avg_score, and profit-driven feedback.
        """
        response = self.ds.query(prompt, max_tokens=800)
        result = response['choices'][0]['message']['content']
        logging.info("Graded orchestrator output for profit.")
        return result

if __name__ == "__main__":
    grader = OrchestratorGrader()
    output = {"status": "success", "content": "Generated UGC"}
    context = {"client_id": "123", "budget": 5000}
    print(grader.grade_output(output, context))