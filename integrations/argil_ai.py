import os
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential  # Ref: https://tenacity.readthedocs.io/
from prometheus_client import Histogram  # Ref: https://github.com/prometheus/client_python
from urllib.parse import urlparse
from utils.proxy_rotator import ProxyRotator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Metric for video generation cost
VIDEO_COST = Histogram('video_generation_cost', 'Cost per video generation', ['resolution'])

class ArgilVideoProducer:
    def __init__(self):
        self.session = requests.Session()
        self.api_key = os.getenv('ARGIL_API_KEY', '')
        if not self.api_key:
            raise ValueError("ARGIL_API_KEY must be set in environment variables.")
        self.session.headers.update({
            "x-api-key": self.api_key,  # Ref: https://docs.argil.ai/api/introduction
            "Content-Type": "application/json"
        })
        self.webhook_url = os.getenv('WEBHOOK_CALLBACK_URL', 'https://yourdomain.com/webhook')
        self.proxy_rotator = ProxyRotator()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def create_avatar(self, name: str, dataset_video_url: str, consent_video_url: str):
        """Create a new avatar (ref: https://docs.argil.ai/api/avatars)."""
        payload = {
            "name": name,
            "datasetVideo": {"url": dataset_video_url},
            "consentVideo": {"url": consent_video_url}
        }
        proxy = self.proxy_rotator.get_proxy()
        response = self.session.post(
            "https://api.argil.ai/v1/avatars",
            json=payload,
            proxies={"http": proxy, "https": proxy},
            timeout=10
        )
        response.raise_for_status()
        logging.info(f"Created avatar: {name}")
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_avatar(self, avatar_id: str):
        """Retrieve avatar details."""
        proxy = self.proxy_rotator.get_proxy()
        response = self.session.get(
            f"https://api.argil.ai/v1/avatars/{avatar_id}",
            proxies={"http": proxy, "https": proxy},
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def list_avatars(self):
        """List all avatars."""
        proxy = self.proxy_rotator.get_proxy()
        response = self.session.get(
            "https://api.argil.ai/v1/avatars",
            proxies={"http": proxy, "https": proxy},
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def list_voices(self):
        """List available voices."""
        proxy = self.proxy_rotator.get_proxy()
        response = self.session.get(
            "https://api.argil.ai/v1/voices",
            proxies={"http": proxy, "https": proxy},
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_voice(self, voice_id: str):
        """Retrieve voice details."""
        proxy = self.proxy_rotator.get_proxy()
        response = self.session.get(
            f"https://api.argil.ai/v1/voices/{voice_id}",
            proxies={"http": proxy, "https": proxy},
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_video(self, video_id: str, resolution: str = "1080p"):
        """Trigger video rendering."""
        with VIDEO_COST.labels(resolution).time():
            proxy = self.proxy_rotator.get_proxy()
            response = self.session.post(
                f"https://api.argil.ai/v1/videos/{video_id}/render",
                proxies={"http": proxy, "https": proxy},
                timeout=(3.05, 30)
            )
            response.raise_for_status()
            logging.info(f"Generated video: {video_id}")
            return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_video(self, video_id: str):
        """Retrieve video details."""
        proxy = self.proxy_rotator.get_proxy()
        response = self.session.get(
            f"https://api.argil.ai/v1/videos/{video_id}",
            proxies={"http": proxy, "https": proxy},
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def delete_video(self, video_id: str):
        """Delete a video."""
        proxy = self.proxy_rotator.get_proxy()
        response = self.session.delete(
            f"https://api.argil.ai/v1/videos/{video_id}",
            proxies={"http": proxy, "https": proxy},
            timeout=10
        )
        response.raise_for_status()
        return response.status_code == 204

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def setup_webhooks(self):
        """Set up webhooks."""
        payload = {
            "callbackUrl": self.webhook_url,
            "events": [
                "VIDEO_GENERATION_SUCCESS",
                "VIDEO_GENERATION_FAILED",
                "AVATAR_TRAINING_SUCCESS",
                "AVATAR_TRAINING_FAILED"
            ]
        }
        proxy = self.proxy_rotator.get_proxy()
        response = self.session.post(
            "https://api.argil.ai/v1/webhooks",
            json=payload,
            proxies={"http": proxy, "https": proxy},
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def update_webhook(self, webhook_id: str, callback_url: str, events: list):
        """Update a webhook."""
        payload = {
            "callbackUrl": callback_url,
            "events": events
        }
        proxy = self.proxy_rotator.get_proxy()
        response = self.session.put(
            f"https://api.argil.ai/v1/webhooks/{webhook_id}",
            json=payload,
            proxies={"http": proxy, "https": proxy},
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def delete_webhook(self, webhook_id: str):
        """Delete a webhook."""
        proxy = self.proxy_rotator.get_proxy()
        response = self.session.delete(
            f"https://api.argil.ai/v1/webhooks/{webhook_id}",
            proxies={"http": proxy, "https": proxy},
            timeout=10
        )
        response.raise_for_status()
        return response.status_code == 204

    def _preprocess_text(self, text):
        """Preprocess input text for video generation."""
        return text[:5000]  # Truncate to API limit

    def _cache_result(self, video_id, url):
        """Autonomous caching system."""
        domain = urlparse(url).netloc
        logging.info(f"Cached video {video_id} from {domain}")
        # Placeholder—no budget impact here