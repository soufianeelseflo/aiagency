import os
import json
import logging
import time
import random
import string
from typing import Dict, Optional
from twilio.rest import Client  # Ref: https://www.twilio.com/docs/libraries/python
from elevenlabs import generate, save, VoiceSettings  # Ref: https://elevenlabs.io/docs/api-reference/text-to-speech
from pathlib import Path

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VoiceSalesAgent:
    def __init__(self):
        # Initialize Twilio client (ref: https://www.twilio.com/docs/libraries/python)
        self.twilio_client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))
        self.twilio_number = os.getenv("TWILIO_PHONE_NUMBER", "+14155238886")  # Default US-based Twilio number

        # Initialize ElevenLabs API key (ref: https://elevenlabs.io/docs/api-reference/authentication)
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY must be set in environment variables.")

        # File paths for scripts, context, and strategy cache
        self.scripts_file = Path("scripts.json")  # Pre-written scripts you provide
        self.context_file = Path("voice_agent_context.json")
        self.strategy_cache_file = Path("strategy_cache.json")

        # Load or initialize data
        self._load_scripts()
        self._load_context()
        self._load_strategy_cache()

    def _load_scripts(self) -> None:
        """Load pre-written sales scripts from scripts.json."""
        if not self.scripts_file.exists():
            raise FileNotFoundError("scripts.json not found. Please create it with script templates.")
        with open(self.scripts_file, "r") as f:
            self.scripts = json.load(f)  # Format: {"strategy_name": "script text", ...}
        logging.info(f"Loaded {len(self.scripts)} sales scripts.")

    def _load_context(self) -> None:
        """Load or initialize client interaction context from JSON."""
        if self.context_file.exists():
            with open(self.context_file, "r") as f:
                self.context_memory = json.load(f)
        else:
            self.context_memory = {"clients": {}}
            self._save_context()
        logging.info("Context memory loaded.")

    def _save_context(self) -> None:
        """Save context memory to JSON."""
        with open(self.context_file, "w") as f:
            json.dump(self.context_memory, f, indent=2)

    def _load_strategy_cache(self) -> None:
        """Load or initialize strategy cache with scores from JSON."""
        if self.strategy_cache_file.exists():
            with open(self.strategy_cache_file, "r") as f:
                self.strategy_cache = json.load(f)
        else:
            self.strategy_cache = {}  # Format: {"strategy_name": {"score": int, "uses": int, "successes": int}}
            self._save_strategy_cache()
        logging.info("Strategy cache loaded.")

    def _save_strategy_cache(self) -> None:
        """Save strategy cache to JSON."""
        with open(self.strategy_cache_file, "w") as f:
            json.dump(self.strategy_cache, f, indent=2)

    def _select_strategy(self, lead_data: Dict) -> str:
        """Select the highest-scoring strategy for the lead, or random if new."""
        client_id = lead_data.get('email', 'anonymous')
        past_interactions = self.context_memory["clients"].get(client_id, {}).get("interactions", [])

        # Filter out low-scoring strategies (<50)
        viable_strategies = {
            name: data for name, data in self.strategy_cache.items()
            if data.get("score", 50) >= 50 or name not in self.strategy_cache
        }

        if not viable_strategies:
            # Fallback to random if no viable strategies
            strategy = random.choice(list(self.scripts.keys()))
        else:
            # Choose highest-scoring strategy
            strategy = max(viable_strategies.items(), key=lambda x: x[1].get("score", 50))[0]

        # Initialize strategy if new
        if strategy not in self.strategy_cache:
            self.strategy_cache[strategy] = {"score": 50, "uses": 0, "successes": 0}
        return strategy

    def _update_strategy_score(self, strategy: str, outcome: str) -> None:
        """Update strategy score based on call outcome (success/failure)."""
        cache = self.strategy_cache[strategy]
        cache["uses"] += 1
        if outcome == "success":
            cache["successes"] += 1
        # Score = (successes / uses) * 100, default to 50 if no uses yet
        cache["score"] = (cache["successes"] / cache["uses"]) * 100 if cache["uses"] > 0 else 50
        self._save_strategy_cache()
        logging.info(f"Updated strategy '{strategy}' score to {cache['score']:.1f}")

    def synthesize_voice(self, script: str) -> str:
        """Synthesize casual, human-like audio using ElevenLabs (ref: https://elevenlabs.io/docs/api-reference/text-to-speech)."""
        script = script[:9000]  # Stay within ElevenLabs free tier limit (10k chars/month)
        voice_settings = VoiceSettings(
            stability=0.4,  # Lower for natural variation (ref: https://elevenlabs.io/docs/api-reference/voice-settings)
            similarity_boost=0.7,  # Balance consistency and realism
            style=0.1,  # Minimal style for casual tone
            use_speaker_boost=True  # Enhance clarity
        )
        audio = generate(
            text=script,
            voice="Rachel",  # Casual, natural female voice (ref: https://elevenlabs.io/docs/api-reference/voices)
            model="eleven_multilingual_v2",  # High-quality, multilingual (ref: https://elevenlabs.io/docs/api-reference/models)
            api_key=self.elevenlabs_api_key,
            voice_settings=voice_settings
        )
        audio_file = f"call_{int(time.time())}_{''.join(random.choices(string.ascii_lowercase, k=5))}.mp3"
        save(audio, audio_file)  # Save audio file locally (ref: https://elevenlabs.io/docs/api-reference/text-to-speech#save-audio)
        logging.info(f"Synthesized audio saved to {audio_file}")
        return audio_file

    def initiate_call(self, lead_number: str, audio_file: str) -> str:
        """Initiate a Twilio voice call with synthesized audio (ref: https://www.twilio.com/docs/voice/api/call-resource)."""
        try:
            call = self.twilio_client.calls.create(
                twiml=f'<Response><Play>{os.path.abspath(audio_file)}</Play></Response>',
                from_=self.twilio_number,  # US-based number
                to=lead_number
            )
            logging.info(f"Call initiated to {lead_number}: {call.sid}")
            return call.sid
        except Exception as e:
            logging.error(f"Failed to initiate call: {str(e)}")
            raise

    def handle_lead(self, lead_data: Dict) -> Dict:
        """Handle a lead with a voice call, learning from the outcome."""
        client_id = lead_data.get('email', 'anonymous')
        
        # Select best strategy and corresponding script
        strategy = self._select_strategy(lead_data)
        script = self.scripts.get(strategy, "Hey, just wanted to chat about boosting your ad game!")
        
        # Synthesize and make the call
        audio_file = self.synthesize_voice(script)
        call_sid = self.initiate_call(lead_data['phone'], audio_file)

        # Simulate or get outcome (replace with real callback logic if available)
        outcome = input(f"Enter outcome for {lead_data['company']} (success/failure): ").lower()
        
        # Update context and strategy cache
        if client_id not in self.context_memory["clients"]:
            self.context_memory["clients"][client_id] = {"interactions": []}
        self.context_memory["clients"][client_id]["interactions"].append({
            "strategy": strategy,
            "outcome": outcome,
            "timestamp": time.time()
        })
        self._save_context()
        self._update_strategy_score(strategy, outcome)

        return {"call_sid": call_sid, "strategy": strategy, "outcome": outcome}

if __name__ == "__main__":
    # Example usage
    agent = VoiceSalesAgent()
    lead = {
        "company": "EcomElite",
        "email": "sales@ecomelite.com",
        "phone": "+1234567890"  # Replace with a real test number
    }
    try:
        result = agent.handle_lead(lead)
        print(f"Call result: {result}")
    except Exception as e:
        logging.error(f"Failed to handle lead: {str(e)}")
