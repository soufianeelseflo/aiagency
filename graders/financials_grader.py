import logging
from typing import Dict
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Grader:
    def __init__(self):
        self.budget_manager = BudgetManager()  # Hardcoded $20 budget
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

    def grade_output(self, output: str, context: Dict) -> str:
        """Grade financial output."""
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.error("Budget exceeded for grading.")
            return '{"status": "failure", "error": "Budget of $20 exceeded"}'

        prompt = f"""
        Grade the following output:
        - Context: {json.dumps(context)}
        - Output: {output}
        
        Criteria:
        1. Accuracy (0–100): Does the output address the intended purpose?
        2. Relevance (0–100): Is it aligned with the target audience's needs?
        
        Provide a final score and recommendations.
        """
        response = self.ds.query(prompt, max_tokens=500)
        logging.info("Graded financial output.")
        return response['choices'][0]['message']['content']

if __name__ == "__main__":
    grader = Grader()
    output = "Payment of $5000 processed"
    context = {"amount": 5000, "currency": "USD"}
    result = grader.grade_output(output, context)
    print(result)