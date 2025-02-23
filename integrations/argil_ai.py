import os
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from prometheus_client import Histogram
from urllib.parse import urlparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Metric for video generation cost
VIDEO_COST = Histogram('video_generation_cost', 'Cost per video generation', ['resolution'])

class ArgilVideoProducer:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": os.getenv('ARGIL_API_KEY'),  # Corrected header based on documentation
            "Content-Type": "application/json"
        })
        self.webhook_url = os.getenv('WEBHOOK_CALLBACK_URL')  # URL for receiving webhooks

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def create_avatar(self, name: str, dataset_video_url: str, consent_video_url: str):
        """
        Create a new avatar by uploading source videos and launching training.
        """
        payload = {
            "name": name,
            "datasetVideo": {"url": dataset_video_url},
            "consentVideo": {"url": consent_video_url}
        }
        response = self.session.post(
            "https://api.argil.ai/v1/avatars",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_avatar(self, avatar_id: str):
        """
        Retrieve details about a specific avatar.
        """
        response = self.session.get(f"https://api.argil.ai/v1/avatars/{avatar_id}")
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def list_avatars(self):
        """
        List all avatars associated with your account.
        """
        response = self.session.get("https://api.argil.ai/v1/avatars")
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def list_voices(self):
        """
        List all available voices for text-to-speech generation.
        """
        response = self.session.get("https://api.argil.ai/v1/voices")
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_voice(self, voice_id: str):
        """
        Retrieve details about a specific voice.
        """
        response = self.session.get(f"https://api.argil.ai/v1/voices/{voice_id}")
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_video(self, video_id: str, resolution: str = "1080p"):
        """
        Trigger video rendering for a given video ID.
        """
        with VIDEO_COST.labels(resolution).time():
            response = self.session.post(
                f"https://api.argil.ai/v1/videos/{video_id}/render",
                timeout=(3.05, 30)
            )
            response.raise_for_status()
            return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_video(self, video_id: str):
        """
        Retrieve details about a specific video.
        """
        response = self.session.get(f"https://api.argil.ai/v1/videos/{video_id}")
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def delete_video(self, video_id: str):
        """
        Delete a specific video.
        """
        response = self.session.delete(f"https://api.argil.ai/v1/videos/{video_id}")
        response.raise_for_status()
        return response.status_code == 204

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def setup_webhooks(self):
        """
        Set up webhooks to receive notifications for video generation and avatar training events.
        """
        payload = {
            "callbackUrl": self.webhook_url,
            "events": [
                "VIDEO_GENERATION_SUCCESS",
                "VIDEO_GENERATION_FAILED",
                "AVATAR_TRAINING_SUCCESS",
                "AVATAR_TRAINING_FAILED"
            ]
        }
        response = self.session.post(
            "https://api.argil.ai/v1/webhooks",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def update_webhook(self, webhook_id: str, callback_url: str, events: list):
        """
        Update an existing webhook configuration.
        """
        payload = {
            "callbackUrl": callback_url,
            "events": events
        }
        response = self.session.put(f"https://api.argil.ai/v1/webhooks/{webhook_id}", json=payload)
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def delete_webhook(self, webhook_id: str):
        """
        Delete a specific webhook.
        """
        response = self.session.delete(f"https://api.argil.ai/v1/webhooks/{webhook_id}")
        response.raise_for_status()
        return response.status_code == 204

    def _preprocess_text(self, text):
        """Preprocess input text for video generation."""
        return text[:5000]  # Truncate to API limit

    def _cache_result(self, video_id, url):
        """Autonomous caching system."""
        domain = urlparse(url).netloc
        logging.info(f"Cached video {video_id} from {domain}")
        # Implement actual caching logic here (e.g., Redis)