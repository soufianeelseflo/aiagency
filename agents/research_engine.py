import os
import time
import logging
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_fixed
from flask import Flask, jsonify, request
from utils.proxy_rotator import ProxyRotator
from utils.cache_manager import CacheManager
from graders.research_engine_grader import ResearchEngineGrader
from reflectors.research_engine_reflector import ResearchEngineReflector
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

class ResearchEngine:
    def __init__(self):
        # Hardcoded budget and pricing for simplicity
        self.budget_manager = BudgetManager(total_budget=20.0, input_cost_per_million=0.80, output_cost_per_million=2.40)
        self.proxy_rotator = ProxyRotator()
        self.cache_manager = CacheManager()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)
        self.grader = ResearchEngineGrader()
        self.reflector = ResearchEngineReflector()

    def get_status(self):
        """Expose current status of the research engine."""
        return {
            "queries_processed": self.cache_manager.size(),
            "budget_remaining": self.budget_manager.get_remaining_budget()
        }

    def update_parameters(self, updates):
        """Update parameters safely."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logging.info(f"Updated parameter {key} to {value}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))  # Ref: https://tenacity.readthedocs.io/en/latest/
    def document_search(self, query: str, max_tokens: int = 500, temperature: float = 0.7) -> Optional[dict]:
        """Perform iterative research with DeepSeek R1, caching, and proxies."""
        logging.info(f"Executing query: {query}")

        # Check budget before proceeding
        if not self.budget_manager.can_afford(input_tokens=len(query) + 100, output_tokens=max_tokens):
            logging.error(f"Budget exceeded for query: {query}")
            raise Exception("Budget of $20 exceeded. Acquire a client to replenish.")

        # Check cache
        cached_result = self.cache_manager.get(query)
        if cached_result:
            logging.info("Cache hit. Returning cached result.")
            return cached_result

        # Generate refined queries
        refined_queries = self._generate_refined_queries(query)
        results = []
        for refined_query in refined_queries:
            try:
                response = self.ds.query(refined_query, max_tokens=max_tokens, temperature=temperature)
                results.append(response['choices'][0]['message']['content'])
                self.budget_manager.log_usage(input_tokens=len(refined_query) + 100, output_tokens=max_tokens)
            except Exception as e:
                logging.error(f"Query failed: {str(e)}")
                raise  # Retry handled by tenacity

        # Summarize results
        raw_result = self._summarize_results(results)

        # Grade and reflect
        graded_result = self.grader.grade_output(raw_result, {"query": query})
        reflected_result = self.reflector.reflect_output(graded_result, {"query": query})

        # Cache the result
        self.cache_manager.set(query, reflected_result)
        return reflected_result

    def _generate_refined_queries(self, query: str) -> list:
        """Generate up to four refined queries."""
        base_queries = [
            f"{query} overview",
            f"{query} detailed analysis",
            f"{query} latest trends",
            f"{query} case studies"
        ]
        return base_queries[:4]

    def _summarize_results(self, results: list) -> str:
        """Summarize multiple research results."""
        combined_text = "\n".join(results)
        if not self.budget_manager.can_afford(input_tokens=len(combined_text) + 100, output_tokens=300):
            logging.error("Budget exceeded for summarization.")
            return "Summary unavailable due to budget constraints."
        summary = self.ds.query(
            f"Summarize the following research findings: {combined_text}",
            max_tokens=300,
            temperature=0.7
        )['choices'][0]['message']['content']
        self.budget_manager.log_usage(input_tokens=len(combined_text) + 100, output_tokens=300)
        return summary

# Backend API endpoints
@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    """Fetch real-time status (ref: https://flask.palletsprojects.com/en/2.3.x/api/)."""
    research_engine = ResearchEngine()
    return jsonify(research_engine.get_status())

@app.route('/api/update-agent', methods=['POST'])
def update_agent():
    """Update parameters via web interface."""
    data = request.get_json()
    updates = data.get("updates", {})
    research_engine = ResearchEngine()
    research_engine.update_parameters(updates)
    return jsonify({"message": "ResearchEngine updated successfully."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)