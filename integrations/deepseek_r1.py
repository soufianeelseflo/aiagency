import os
import requests
import json

class DeepSeekOrchestrator:
    def __init__(self):
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }

    def query(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> dict:
        """
        Generate content using DeepSeek-R1.
        """
        try:
            response = requests.post(
                f"{self.base_url}/completions",
                headers=self.headers,
                json={
                    "model": "deepseek-r1",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Query failed: {str(e)}")

    def generate_email_template(self, prospect_info: dict) -> str:
        """
        Generate personalized email template using DeepSeek-R1.
        """
        prompt = f"""
        Generate a personalized cold email for the following prospect:
        Name: {prospect_info.get('name', 'Unknown')}
        Company: {prospect_info.get('company', 'Unknown')}
        Industry: {prospect_info.get('industry', 'Unknown')}
        Pain Points: {prospect_info.get('pain_points', 'Unknown')}
        Offer: AI-driven marketing solutions to boost ROI.
        """
        response = self.query(prompt)
        return response['choices'][0]['message']['content']

    def optimize_workflow(self, context: dict) -> dict:
        """
        Optimize workflows using DeepSeek-R1.
        """
        prompt = f"""
        Analyze the following workflow data:
        Context: {json.dumps(context)}
        
        Recommend optimizations considering:
        1. Cost reduction opportunities
        2. Service reliability improvements
        3. Load balancing
        4. New features/services to implement
        """
        response = self.query(prompt)
        return response['choices'][0]['message']['content']