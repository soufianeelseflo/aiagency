import os
import json
import requests

class OrchestratorGrader:
    def __init__(self):
        self.deepseek_endpoint = os.getenv("DEEPSEEK_R1_ENDPOINT")
        self.api_key = os.getenv("DEEPSEEK_API_KEY")

    def grade_output(self, output: dict, context: dict) -> dict:
        """
        Grade the given output for accuracy, relevance, and compliance.
        """
        prompt = f"""
        Grade the following orchestrator output:
        - Context: {json.dumps(context)}
        - Output: {json.dumps(output)}
        
        Criteria:
        1. Accuracy (0–100): Does the output address the intended purpose?
        2. Relevance (0–100): Is the output aligned with the client's needs?
        3. Compliance (0–100): Does the output adhere to legal and ethical standards?
        4. Cost Efficiency (0–100): Are costs minimized without compromising quality?
        
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