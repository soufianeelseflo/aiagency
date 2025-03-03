# soufianeelseflo-aiagency/utils/proxy_rotator.py
import os
import requests
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProxyRotator:
    """Rotates proxies smartly for optimal performance and stealth."""
    def __init__(self):
        self.api_key = os.getenv("SMARTPROXY_API_KEY")
        if not self.api_key:
            raise ValueError("SMARTPROXY_API_KEY not set.")
        self.base_url = "https://api.smartproxy.com/v1"
        self.proxies = self._fetch_proxies()
        self.performance = {}  # Track proxy success/failure
        logging.info(f"ProxyRotator initialized with {len(self.proxies)} proxies.")

    def _fetch_proxies(self) -> list:
        """Fetch proxies from SmartProxy per https://smartproxy.com/docs/api-endpoints."""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            response = requests.get(f"{self.base_url}/proxies", headers=headers, timeout=10)
            response.raise_for_status()
            proxy_data = response.json()
            proxies = [f"http://{proxy['ip']}:{proxy['port']}" for proxy in proxy_data.get('proxies', [])]
            if not proxies:
                raise ValueError("No proxies returned.")
            for proxy in proxies:
                self.performance[proxy] = {"success": 0, "fail": 0, "latency": 0}
            return proxies
        except requests.RequestException as e:
            logging.error(f"Failed to fetch proxies: {str(e)}")
            return []

    def get_proxy(self) -> str:
        """Pick best proxy based on performance."""
        if not self.proxies:
            self.proxies = self._fetch_proxies()
            if not self.proxies:
                raise Exception("No proxies available.")
        # Sort by success rate and latency
        viable = [(proxy, stats["success"] / max(1, stats["success"] + stats["fail"]) - stats["latency"]) 
                  for proxy, stats in self.performance.items()]
        if viable:
            best_proxy = max(viable, key=lambda x: x[1])[0]
            logging.info(f"Using best proxy: {best_proxy}")
            return best_proxy
        proxy = random.choice(self.proxies)
        logging.info(f"Using random proxy: {proxy}")
        return proxy

    def refresh_proxies(self) -> None:
        """Refresh proxy list."""
        logging.info("Refreshing proxy list...")
        self.proxies = self._fetch_proxies()

    def report_performance(self, proxy: str, success: bool, latency: float) -> None:
        """Update proxy stats for profit optimization."""
        if proxy in self.performance:
            self.performance[proxy]["success"] += 1 if success else 0
            self.performance[proxy]["fail"] += 1 if not success else 0
            self.performance[proxy]["latency"] = (self.performance[proxy]["latency"] + latency) / 2
            logging.info(f"Proxy {proxy} updated: {self.performance[proxy]}")
            if self.performance[proxy]["fail"] > 5 and len(self.proxies) < 10:
                logging.warning("Proxy pool low—refreshing!")
                self.refresh_proxies()