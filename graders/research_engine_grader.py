import logging
from typing import Dict
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResearchEngineGrader:
    def __init__(self):
        self.budget_manager = BudgetManager()  # Hardcoded $20 budget
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

    def grade_output(self, output: str, context: Dict) -> str:
        """Grade research engine output."""
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.error("Budget exceeded for grading.")
            return '{"status": "failure", "error": "Budget of $20 exceeded"}'

        prompt = f"""
        Grade the following research output:
        - Context: {json.dumps(context)}
        - Output: {output}
        
        Criteria:
        1. Accuracy (0–100): Does it address the purpose?
        2. Relevance (0–100): Aligned with audience needs?
        
        Provide a final score and recommendations.
        """
        response = self.ds.query(prompt, max_tokens=500)
        logging.info("Graded research engine output.")
        return response['choices'][0]['message']['content']

if __name__ == "__main__":
    grader = ResearchEngineGrader()
    output = "Research on AI trends completed."
    context = {"query": "AI trends 2025"}
    result = grader.grade_output(output, context)
    print(result)