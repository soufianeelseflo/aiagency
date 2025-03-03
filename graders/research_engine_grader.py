# soufianeelseflo-aiagency/graders/research_engine_grader.py
import logging
from typing import Dict
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResearchEngineGrader:
    """Grades research output for revenue impact."""
    def __init__(self):
        self.budget_manager = BudgetManager()
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

    def grade_output(self, output: str, context: Dict) -> str:
        """Grade with profit focus per https://openrouter.ai/docs."""
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=600):
            logging.error("Budget exceeded for grading.")
            return '{"status": "failure", "error": "Budget exceeded"}'
        
        prompt = f"""
        Grade this research output for revenue potential:
        - Context: {json.dumps(context)}
        - Output: {output}
        
        Criteria (0-100 each):
        1. Revenue Relevance: Does it support high-profit tasks?
        2. Client Value: Useful for client acquisition?
        3. Depth: Rich enough for strategic decisions?
        4. Speed: Delivered fast for quick wins?
        
        Return JSON with scores, avg_score, and revenue-focused feedback.
        """
        response = self.ds.query(prompt, max_tokens=600)
        result = response['choices'][0]['message']['content']
        logging.info("Graded research engine output for profit.")
        return result

if __name__ == "__main__":
    grader = ResearchEngineGrader()
    output = "Research on AI trends completed."
    context = {"query": "AI trends 2025"}
    print(grader.grade_output(output, context))