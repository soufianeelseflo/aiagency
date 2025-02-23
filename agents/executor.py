import os
import time
import requests
from integrations.deepseek_r1 import DeepSeekOrchestrator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class AcquisitionEngine:
    def __init__(self):
        self.ds = DeepSeekOrchestrator()
        self.smartproxy_api_key = os.getenv('SMARTPROXY_API_KEY')
        self.smartproxy_endpoint = "http://gate.smartproxy.com:7000"  # Example endpoint
        self.pricing = {
            'starter': 3000,
            'pro': 5000,
            'enterprise': 10000
        }
    
    def handle_inbound_lead(self, lead_data: dict) -> dict:
        """
        Processes new leads with AI-driven personalization.
        Delegates email-sending tasks to the centralized EmailManager.
        """
        offer = self.ds.query(f"""
        Create personalized offer for:
        Company: {lead_data['company']}
        Industry: {lead_data['industry']}
        Budget: {lead_data['budget']}
        Pain Points: {lead_data['pains']}
        """)
        
        # Delegate email sending to the centralized EmailManager
        email_payload = {
            'to': lead_data['email'],
            'template_name': 'onboarding',
            'variables': {
                'offer_price': self.pricing[lead_data['tier']],
                'custom_solutions': offer
            }
        }
        return {
            'status': 'prepared_for_email',
            'offer': offer,
            'email_payload': email_payload,
            'next_step': 'schedule_ai_call'
        }
    
    def cold_outreach(self, company_data: dict) -> dict:
        """
        AI-powered personalized cold outreach.
        Uses Smartproxy for rotating proxies to avoid IP bans.
        Delegates email-sending tasks to the centralized EmailManager.
        """
        message = self.ds.query(f"""
        Write cold email for {company_data['name']}:
        - Industry: {company_data['industry']}
        - Recent news: {company_data['news']}
        - Pain points: {company_data['pains']}
        - Offer: {self.pricing['starter']}/month
        """)
        
        # Use Smartproxy for rotating proxies
        proxy = self._get_rotating_proxy()
        
        # Delegate email sending to the centralized EmailManager
        email_payload = {
            'to': company_data['decision_maker_email'],
            'subject': f"AI UGC Solutions for {company_data['name']}",
            'body': message,
            'proxy': proxy
        }
        return {
            'status': 'prepared_for_email',
            'email_payload': email_payload
        }
    
    def _get_rotating_proxy(self) -> str:
        """
        Fetch a rotating proxy from Smartproxy.
        """
        try:
            response = requests.get(
                f"{self.smartproxy_endpoint}/get_proxy",
                headers={"Authorization": f"Bearer {self.smartproxy_api_key}"}
            )
            response.raise_for_status()
            proxy_data = response.json()
            return f"{proxy_data['ip']}:{proxy_data['port']}"
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch proxy from Smartproxy: {str(e)}")
            raise Exception("Proxy rotation failed. Please check Smartproxy configuration.")