import os
import time
import logging
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_fixed
from proxy_rotator import ProxyRotator
from cache_manager import CacheManager
from flask import Flask, jsonify, request
from graders.research_engine_grader import ResearchEngineGrader
from reflectors.research_engine_reflector import ResearchEngineReflector

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)

class ResearchEngine:
    def __init__(self):
        # Initialize proxy rotator
        self.proxy_rotator = ProxyRotator()
        
        # Initialize cache manager
        self.cache_manager = CacheManager()
        
        # Budget tracking
        self.budget = float(os.getenv("RESEARCH_BUDGET", 10.0))  # Monthly budget in USD
        self.token_cost = float(os.getenv("TOKEN_COST_PER_1K", 0.001))  # Cost per 1K tokens
        self.used_budget = 0.0
        
        # Graders and Reflectors
        self.grader = ResearchEngineGrader()
        self.reflector = ResearchEngineReflector()

    def get_status(self):
        """Expose current status of the research engine."""
        return {
            "queries_processed": len(self.cache_manager.cache),
            "budget_used": self.used_budget,
            "remaining_budget": self.budget - self.used_budget
        }

    def update_parameters(self, updates):
        """Update parameters safely."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def document_search(self, query: str, max_tokens: int = 500, temperature: float = 0.7) -> Optional[dict]:
        """
        Perform iterative research using proxies, caching, and DeepSeek R1.
        """
        logging.info(f"Executing query: {query}")
        
        # Check budget before proceeding
        if not self._check_budget(max_tokens):
            raise Exception("Monthly budget exceeded. Please wait until next month.")
        
        # Step 1: Check cache for pre-existing results
        cached_result = self.cache_manager.get(query)
        if cached_result:
            logging.info("Cache hit. Returning cached result.")
            return cached_result
        
        # Step 2: Rotate proxy for the request
        proxy = self.proxy_rotator.get_proxy()
        if not proxy:
            raise Exception("No available proxies. Unable to proceed.")
        
        # Step 3: Generate refined queries
        refined_queries = self._generate_refined_queries(query)
        results = []
        for refined_query in refined_queries:
            try:
                response = self._deepseek_query(refined_query, max_tokens, temperature, proxy)
                results.append(response)
                self.used_budget += (max_tokens / 1000) * self.token_cost  # Update budget usage
            except Exception as e:
                logging.error(f"Query failed: {str(e)}")
        
        # Combine and summarize results
        raw_result = self._summarize_results(results)
        
        # Grade the result
        graded_result = self.grader.grade_output(raw_result, {"query": query})
        
        # Reflect on the graded result
        reflected_result = self.reflector.reflect_output(graded_result, {"query": query})
        
        # Cache the result
        self.cache_manager.set(query, reflected_result)
        return reflected_result

    def _generate_refined_queries(self, query: str) -> list:
        """Generate up to four refined queries based on initial query."""
        base_queries = [
            f"{query} overview",
            f"{query} detailed analysis",
            f"{query} latest trends",
            f"{query} case studies"
        ]
        return base_queries[:4]  # Limit to four refined queries

    def _deepseek_query(self, prompt: str, max_tokens: int, temperature: float, proxy: str) -> str:
        """Query DeepSeek R1 for refined results using a proxy."""
        from integrations.deepseek_r1 import DeepSeekOrchestrator
        ds = DeepSeekOrchestrator(proxy=proxy)
        response = ds.query(prompt, max_tokens=max_tokens, temperature=temperature)
        return response['choices'][0]['message']['content']

    def _summarize_results(self, results: list) -> str:
        """Summarize multiple research results into a concise output."""
        combined_text = "\n".join(results)
        summary = self._deepseek_query(
            f"Summarize the following research findings: {combined_text}",
            max_tokens=300,
            temperature=0.7,
            proxy=self.proxy_rotator.get_proxy()
        )
        return summary

    def _check_budget(self, tokens: int) -> bool:
        """Check if the remaining budget allows for the requested token usage."""
        cost = (tokens / 1000) * self.token_cost
        if self.used_budget + cost > self.budget:
            return False
        return True

# Backend API endpoints
@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    """Fetch real-time status of the research engine."""
    research_engine = ResearchEngine()
    return jsonify(research_engine.get_status())

@app.route('/api/update-agent', methods=['POST'])
def update_agent():
    """Update research engine parameters via web interface."""
    data = request.json
    updates = data.get("updates", {})
    research_engine = ResearchEngine()
    research_engine.update_parameters(updates)
    return jsonify({"message": "ResearchEngine updated successfully."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)