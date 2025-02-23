import os
import requests
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class ProxyRotator:
    def __init__(self):
        # Initialize SmartProxy configuration
        self.api_key = os.getenv("SMARTPROXY_API_KEY")
        self.base_url = "https://api.smartproxy.com/v1"
        self.proxies = self._fetch_proxies()
        self.current_proxy_index = 0

    def _fetch_proxies(self) -> list:
        """
        Fetch available proxies from SmartProxy API.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.get(f"{self.base_url}/proxies", headers=headers)
            response.raise_for_status()
            proxy_data = response.json()
            proxies = [f"http://{proxy['ip']}:{proxy['port']}" for proxy in proxy_data['proxies']]
            logging.info(f"Fetched {len(proxies)} proxies from SmartProxy.")
            return proxies
        except Exception as e:
            logging.error(f"Failed to fetch proxies: {str(e)}")
            return []

    def get_proxy(self) -> str:
        """
        Rotate and return a proxy.
        """
        if not self.proxies:
            raise Exception("No available proxies. Unable to proceed.")
        
        # Rotate to the next proxy
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
        proxy = self.proxies[self.current_proxy_index]
        logging.info(f"Using proxy: {proxy}")
        return proxy

    def refresh_proxies(self) -> None:
        """
        Refresh the list of proxies from SmartProxy API.
        """
        logging.info("Refreshing proxy list...")
        self.proxies = self._fetch_proxies()