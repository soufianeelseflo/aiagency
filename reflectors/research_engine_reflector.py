# soufianeelseflo-aiagency/reflectors/research_engine_reflector.py
import logging
from typing import Dict
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResearchEngineReflector:
    """Reflects on research output for revenue gains."""
    def __init__(self):
        self.budget_manager = BudgetManager()
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

    def reflect_output(self, graded_output: str, context: Dict) -> str:
        """Reflect with revenue focus per https://openrouter.ai/docs."""
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=500):
            logging.error("Budget exceeded for reflection.")
            return '{"status": "failure", "error": "Budget exceeded"}'

        prompt = f"""
        Reflect on this graded research output for revenue:
        - Context: {json.dumps(context)}
        - Graded Output: {graded_output}
        
        Provide actionable, revenue-driven recommendations:
        - Enhance client value (e.g., target profitable niches).
        - Optimize research speed for quick sales.
        - Use feedback to refine profitable queries.
        Return JSON with improved strategies.
        """
        response = self.ds.query(prompt, max_tokens=500, temperature=0.3)
        result = response['choices'][0]['message']['content']
        logging.info("Reflected on research engine output for revenue.")
        return result

if __name__ == "__main__":
    reflector = ResearchEngineReflector()
    graded = '{"revenue_relevance": 90, "avg_score": 85}'
    context = {"query": "AI trends 2025"}
    print(reflector.reflect_output(graded, context))