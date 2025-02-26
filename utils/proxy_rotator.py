import os
import requests
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProxyRotator:
    def __init__(self):
        """Initialize SmartProxy configuration (ref: https://smartproxy.com/docs/api-endpoints)."""
        self.api_key = os.getenv("SMARTPROXY_API_KEY")
        if not self.api_key:
            raise ValueError("SMARTPROXY_API_KEY environment variable not set.")
        self.base_url = "https://api.smartproxy.com/v1"
        self.proxies = self._fetch_proxies()
        self.current_proxy_index = 0

    def _fetch_proxies(self) -> list:
        """Fetch available proxies from SmartProxy API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.get(f"{self.base_url}/proxies", headers=headers, timeout=10)
            response.raise_for_status()
            proxy_data = response.json()
            proxies = [f"http://{proxy['ip']}:{proxy['port']}" for proxy in proxy_data.get('proxies', [])]
            if not proxies:
                raise ValueError("No proxies returned from SmartProxy.")
            logging.info(f"Fetched {len(proxies)} proxies from SmartProxy.")
            return proxies
        except requests.RequestException as e:
            logging.error(f"Failed to fetch proxies: {str(e)}")
            return []

    def get_proxy(self) -> str:
        """Rotate and return a proxy."""
        if not self.proxies:
            self.proxies = self._fetch_proxies()
            if not self.proxies:
                raise Exception("No available proxies. Check SmartProxy API key and network.")
        
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
        proxy = self.proxies[self.current_proxy_index]
        logging.info(f"Using proxy: {proxy}")
        return proxy

    def refresh_proxies(self) -> None:
        """Refresh the list of proxies from SmartProxy API."""
        logging.info("Refreshing proxy list...")
        self.proxies = self._fetch_proxies()