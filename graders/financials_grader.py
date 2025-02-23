import os
import json
import requests
from typing import Dict

class Grader:
    def __init__(self):
        self.deepseek_endpoint = os.getenv("DEEPSEEK_R1_ENDPOINT")
        self.api_key = os.getenv("DEEPSEEK_API_KEY")

    def grade_output(self, output: str, context: Dict) -> Dict:
        """
        Grade the given output for accuracy, relevance, and compliance.
        """
        prompt = f"""
        Grade the following output:
        - Context: {json.dumps(context)}
        - Output: {output}
        
        Criteria:
        1. Accuracy (0–100): Does the output address the intended purpose?
        2. Relevance (0–100): Is the output aligned with the target audience's needs?
        3. Compliance (0–100): Does the output adhere to legal and ethical standards?
        
        Provide a final score and recommendations for improvement.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-r1",
            "messages": [
                {"role": "system", "content": "You are an expert AI grader."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }

        response = requests.post(self.deepseek_endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']