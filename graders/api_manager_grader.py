import logging
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EliteEmailSystemGrader:
    def __init__(self):
        self.budget_manager = BudgetManager()  # Hardcoded $20 budget
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

    def grade_output(self, output: dict, context: dict) -> str:
        """Grade email campaign output for competitive outreach."""
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=500):
            logging.error("Budget exceeded for grading.")
            return '{"status": "failure", "error": "Budget of $20 exceeded"}'

        prompt = f"""
        Grade this email campaign output:
        - Context: {json.dumps(context)}
        - Output: {json.dumps(output)}
        
        Criteria:
        1. Accuracy (0–100): Valid recipients?
        2. Relevance (0–100): Beats competitor messaging?
        3. Cost Efficiency (0–100): Outperforms rival costs?
        
        Provide a score and tips to dominate competitors in client Outreach.
        """
        response = self.ds.query(prompt, max_tokens=500)
        logging.info("Graded email campaign output.")
        return response['choices'][0]['message']['content']

if __name__ == "__main__":
    grader = EliteEmailSystemGrader()
    output = {"success": 50, "blocked": 10}
    context = {"emails": ["test@example.com"], "template": "Boost ROI with AI"}
    result = grader.grade_output(output, context)
    print(result)