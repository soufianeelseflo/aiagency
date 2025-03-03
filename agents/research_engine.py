# soufianeelseflo-aiagency/agents/research_engine.py
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
from browser_use import Agent  # Web UI integration

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

class ResearchEngine:
    def __init__(self):
        self.budget_manager = BudgetManager(total_budget=20.0, input_cost_per_million=0.80, output_cost_per_million=2.40)
        self.proxy_rotator = ProxyRotator()
        self.cache_manager = CacheManager()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)
        self.grader = ResearchEngineGrader()
        self.reflector = ResearchEngineReflector()
        self.agent = Agent(
            llm="deepseek/deepseek-r1",
            chrome_path=os.getenv("CHROME_PATH", "/usr/bin/google-chrome"),
            chrome_user_data=os.getenv("CHROME_USER_DATA", ""),
            persistent_session=True
        )

    def get_status(self):
        """Expose current status."""
        return {
            "queries_processed": self.cache_manager.size(),
            "budget_remaining": self.budget_manager.get_remaining_budget()
        }

    def update_parameters(self, updates):
        """Update parameters via web UI."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logging.info(f"Updated parameter {key} to {value}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def document_search(self, query: str, max_tokens: int = 500, temperature: float = 0.7) -> Optional[dict]:
        """Perform genius-level research with web UI and DeepSeek R1."""
        logging.info(f"Executing query: {query}")
        if not self.budget_manager.can_afford(input_tokens=len(query) + 100, output_tokens=max_tokens):
            logging.error(f"Budget exceeded for query: {query}")
            raise Exception("Budget of $20 exceeded.")
        
        cached_result = self.cache_manager.get(query)
        if cached_result:
            logging.info("Cache hit.")
            return cached_result

        # Web search with browser-use Agent
        task = f"""
        Navigate to 'https://www.google.com'.
        Fill 'input[name="q"]' with '{query}'.
        Click 'input[type="submit"]' or 'Search'.
        Wait for '.g' (search results, timeout 30000ms).
        Extract text of first '.g' result.
        Return the text or 'No results' if not found.
        """
        result = self.agent.run(task)
        web_result = result.split("returned: ")[-1].strip() if "Success" in result else "No results found"

        # Refine queries and search with DeepSeek R1
        refined_queries = self._generate_refined_queries(query)
        results = [web_result]
        for refined_query in refined_queries:
            try:
                response = self.ds.query(refined_query, max_tokens=max_tokens, temperature=temperature)
                results.append(response['choices'][0]['message']['content'])
                self.budget_manager.log_usage(input_tokens=len(refined_query) + 100, output_tokens=max_tokens)
            except Exception as e:
                logging.error(f"Query failed: {str(e)}")
                raise

        raw_result = self._summarize_results(results)
        graded_result = self.grader.grade_output(raw_result, {"query": query})
        reflected_result = self.reflector.reflect_output(graded_result, {"query": query})
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

@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    research_engine = ResearchEngine()
    return jsonify(research_engine.get_status())

@app.route('/api/update-agent', methods=['POST'])
def update_agent():
    data = request.get_json()
    updates = data.get("updates", {})
    research_engine = ResearchEngine()
    research_engine.update_parameters(updates)
    return jsonify({"message": "ResearchEngine updated successfully."})

@app.route('/api/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get("query")
    research_engine = ResearchEngine()
    result = research_engine.document_search(query)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)