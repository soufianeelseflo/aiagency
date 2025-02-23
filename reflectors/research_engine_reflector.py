import os
import json
import requests
from typing import Dict

class ResearchEngineReflector:
    def __init__(self):
        self.deepseek_endpoint = os.getenv("DEEPSEEK_R1_ENDPOINT")
        self.api_key = os.getenv("DEEPSEEK_API_KEY")

    def reflect_output(self, graded_output: str, context: Dict) -> Dict:
        """
        Reflect on the graded research output and provide actionable insights for optimization.
        """
        prompt = f"""
        Reflect on the following graded research output:
        - Context: {json.dumps(context)}
        - Graded Output: {graded_output}
        
        Provide actionable recommendations for optimization and an improved version.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-r1",
            "messages": [
                {"role": "system", "content": "You are an expert AI reflector."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }

        response = requests.post(self.deepseek_endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']