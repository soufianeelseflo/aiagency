# legal_compliance_agent.py
import os
import requests
import json
import logging
from typing import Dict
from integrations.deepseek_r1 import DeepSeekOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Non-European legal sites for compliance
LEGAL_SITES = {
    'US': 'https://www.law.cornell.edu',
    'CA': 'https://laws-lois.justice.gc.ca/eng',
    'MX': 'https://www.diputados.gob.mx/LeyesBiblio',
    'AU': 'https://www.legislation.gov.au',
    'JP': 'https://www.japaneselawtranslation.go.jp',
    'IN': 'https://www.indiacode.nic.in',
    'ZA': 'https://www.gov.za/documents/acts'
}

class LegalComplianceAgent:
    def __init__(self):
        # DeepSeek R1 for legal analysis
        self.budget_manager = BudgetManager(
            total_budget=float(os.getenv("TOTAL_BUDGET", 20.0)),
            input_cost_per_million=float(os.getenv("INPUT_COST_PER_M", 0.80)),
            output_cost_per_million=float(os.getenv("OUTPUT_COST_PER_M", 2.40))
        )
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)

    def scrape_laws(self, country_code: str) -> str:
        """Scrape legal data from official government sites (ref: https://requests.readthedocs.io/en/latest/)."""
        if country_code not in LEGAL_SITES:
            logging.warning(f"No legal site available for {country_code}")
            return ""
        url = LEGAL_SITES[country_code]
        try:
            response = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            response.raise_for_status()
            legal_text = response.text[:3000]  # Cap for processing efficiency
            logging.info(f"Scraped legal data from {url}")
            return legal_text
        except requests.RequestException as e:
            logging.error(f"Failed to scrape {url}: {str(e)}")
            return ""

    def analyze_laws(self, country_code: str, business_model: str = "AI agency") -> Dict:
        """Analyze laws and suggest creative, compliant strategies."""
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=1000):
            logging.error("Budget exceeded for legal analysis.")
            return {"status": "failure", "error": "Insufficient budget for analysis"}
        
        raw_laws = self.scrape_laws(country_code)
        if not raw_laws:
            return {"status": "error", "message": f"No legal data available for {country_code}"}
        
        prompt = f"""
        Analyze business laws for an {business_model} in {country_code}:
        - Raw data: {raw_laws}
        - Suggest compliant, aggressive strategies to maximize profit.
        - Bend rules creatively like a top 0.001% entrepreneur, without breaking them.
        - Include specific compliance checkpoints (e.g., data privacy, advertising, tax).
        - Tone: Genius-level, strategic, actionable.
        """
        analysis = self.ds.query(prompt, max_tokens=1200)['choices'][0]['message']['content']
        logging.info(f"Legal analysis completed for {country_code}")
        return {"status": "success", "strategies": analysis, "country": country_code}

    def check_compliance(self, operation: Dict, country_code: str) -> Dict:
        """Check if an operation complies with local laws."""
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.error("Budget exceeded for compliance check.")
            return {"status": "failure", "error": "Insufficient budget"}
        
        analysis = self.analyze_laws(country_code)
        if analysis["status"] != "success":
            return {"status": "error", "message": analysis["message"]}
        
        prompt = f"""
        Check if this operation complies with {country_code} laws:
        - Operation: {json.dumps(operation)}
        - Legal Analysis: {analysis['strategies']}
        Return a JSON object with 'compliant' (bool) and 'details' (str).
        """
        result = json.loads(self.ds.query(prompt, max_tokens=500)['choices'][0]['message']['content'])
        logging.info(f"Compliance check for {operation['type']} in {country_code}: {result['compliant']}")
        return {"status": "success", "compliant": result['compliant'], "details": result['details']}

if __name__ == "__main__":
    agent = LegalComplianceAgent()
    strategies = agent.analyze_laws("US")
    print(json.dumps(strategies, indent=2))
    op = {"type": "voice_sales", "data_collection": "phone_numbers"}
    compliance = agent.check_compliance(op, "US")
    print(json.dumps(compliance, indent=2))