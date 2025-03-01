import os
import json
import logging
import time
import random
import string
from typing import Dict, Optional, List
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
        self.scripts_file = Path("scripts.json")  # Pre-written scripts
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
            self.context_memory = {"clients": {}, "global": {"patterns": {}}}
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

    def _select_strategy(self, lead_data: Dict, context: Dict) -> str:
        """Select the highest-scoring strategy, adapting to client context and avoiding robotic replies."""
        client_id = lead_data.get('email', 'anonymous')
        past_interactions = self.context_memory["clients"].get(client_id, {}).get("interactions", [])
        global_patterns = self.context_memory["global"]["patterns"]

        # Analyze context for client patterns (e.g., objections, confusion, pivots)
        patterns = self._detect_patterns(context, past_interactions)
        if patterns:
            logging.info(f"Detected patterns for {client_id}: {patterns}")
            self.context_memory["global"]["patterns"][client_id] = patterns

        # Filter out low-scoring strategies (<50) and robotic responses
        viable_strategies = {
            name: data for name, data in self.strategy_cache.items()
            if data.get("score", 50) >= 50 and not self._is_robotic(name, patterns)
        }

        if not viable_strategies:
            strategy = random.choice(list(self.scripts.keys()))  # Fallback to random
        else:
            strategy = max(viable_strategies.items(), key=lambda x: x[1].get("score", 50))[0]

        # Initialize strategy if new
        if strategy not in self.strategy_cache:
            self.strategy_cache[strategy] = {"score": 50, "uses": 0, "successes": 0}
        return strategy

    def _detect_patterns(self, context: Dict, past_interactions: List[Dict]) -> List[str]:
        """Detect client interaction patterns to avoid robotic replies (e.g., objections, confusion)."""
        patterns = []
        if "objection" in context.get("response", "").lower() or any("objection" in i.get("response", "").lower() for i in past_interactions):
            patterns.append("objection")
        if "confused" in context.get("response", "").lower() or any("confused" in i.get("response", "").lower() for i in past_interactions):
            patterns.append("confusion")
        if "pivot" in context.get("response", "").lower() or any("pivot" in i.get("response", "").lower() for i in past_interactions):
            patterns.append("pivot")
        return patterns

    def _is_robotic(self, strategy: str, patterns: List[str]) -> bool:
        """Check if a strategy would produce a robotic reply based on detected patterns."""
        robotic_triggers = {
            "objection": ["only 3 spots left", "act now"],
            "confusion": ["boost ROI", "save time"],
            "pivot": ["generic pitch", "repeat offer"]
        }
        for pattern in patterns:
            if any(trigger in self.scripts.get(strategy, "").lower() for trigger in robotic_triggers.get(pattern, [])):
                return True
        return False

    def _update_strategy_score(self, strategy: str, outcome: str, context: Dict) -> None:
        """Update strategy score based on call outcome and context, avoiding robotic replies."""
        cache = self.strategy_cache[strategy]
        cache["uses"] += 1
        if outcome == "success":
            cache["successes"] += 1
        # Score = (successes / uses) * 100, default to 50 if no uses yet
        cache["score"] = (cache["successes"] / cache["uses"]) * 100 if cache["uses"] > 0 else 50
        # Update global patterns to avoid robotic responses
        patterns = self._detect_patterns(context, [])
        if patterns and strategy in self.scripts:
            self.context_memory["global"]["patterns"][strategy] = patterns
        self._save_strategy_cache()
        logging.info(f"Updated strategy '{strategy}' score to {cache['score']:.1f}")

    def generate_sales_script(self, lead_data: Dict, context: Dict) -> str:
        """Generate a dynamic, non-salesy, non-robotic sales script using Hormozi’s techniques."""
        client_id = lead_data.get('email', 'anonymous')
        strategy = self._select_strategy(lead_data, context)
        patterns = self.context_memory["global"]["patterns"].get(client_id, [])

        prompt = f"""
        Generate a professional, non-salesy sales script for an AI UGC ad agency targeting {lead_data['company']}:
        - Industry: {lead_data['industry']}
        - Pain Points: {lead_data['pains']}
        - Offer: $5,000/month for fully automated AI UGC ads delivering 8x ROI.
        - Previous Context: {json.dumps(context)}
        - Detected Patterns: {json.dumps(patterns)}
        - Avoid Robotic Replies: Do not use generic phrases, repetitive offers, or urgency like 'only 3 spots left.'
        
        Use Alex Hormozi’s techniques:
        1. Value Stack: Highlight $100,000+ revenue potential, 8x ROI, and 100+ hours saved monthly.
        2. Pain Point Focus: Address high ad costs, slow content creation, operational headaches.
        3. Conversational Tone: Sound human, friendly, sophisticated—like a Twitter expert.
        4. Dynamic Adaptation: Adjust based on client patterns (objections, confusion, pivots) to avoid robotic responses.

        Example structure:
        - "Hey [name], I know [pain point] is tough for [company]..."
        - "Imagine [value stack]... We’ve crushed this for [industry] leaders."
        - "What’s the biggest challenge you’re facing with ads? Let’s solve it together."
        """
        from integrations.deepseek_r1 import DeepSeekOrchestrator  # Imported here to avoid circular imports
        ds = DeepSeekOrchestrator()
        response = ds.query(prompt, max_tokens=1000, temperature=0.7)
        script = response['choices'][0]['message']['content']
        self.context_memory["clients"][client_id] = self.context_memory["clients"].get(client_id, {}) | {
            "last_script": script,
            "patterns": patterns,
            "timestamp": time.time()
        }
        self._save_context()
        return script

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

    def handle_lead(self, lead_data: Dict, context: Dict = {}) -> Dict:
        """Handle a lead with a voice call, learning from the outcome and avoiding robotic replies."""
        client_id = lead_data.get('email', 'anonymous')
        
        # Generate script, avoiding robotic replies
        script = self.generate_sales_script(lead_data, context)
        audio_file = self.synthesize_voice(script)
        call_sid = self.initiate_call(lead_data['phone'], audio_file)

        # Simulate or get outcome (replace with real callback logic if available)
        outcome = input(f"Enter outcome for {lead_data['company']} (success/failure): ").lower()
        response_context = {"response": input(f"Client response or notes for {lead_data['company']}: ")}
        
        # Update context and strategy cache
        if client_id not in self.context_memory["clients"]:
            self.context_memory["clients"][client_id] = {"interactions": []}
        self.context_memory["clients"][client_id]["interactions"].append({
            "script": script,
            "outcome": outcome,
            "context": response_context,
            "timestamp": time.time()
        })
        self._save_context()
        self._update_strategy_score(script, outcome, response_context)

        return {"call_sid": call_sid, "strategy": script, "outcome": outcome, "context": response_context}

if __name__ == "__main__":
    # Example usage
    agent = VoiceSalesAgent()
    lead = {
        "company": "EcomElite",
        "email": "sales@ecomelite.com",
        "phone": "+1234567890"  # Replace with a real test number
    }
    try:
        result = agent.handle_lead(lead, {"response": "I need cheaper ads"})
        print(f"Call result: {result}")
    except Exception as e:
        logging.error(f"Failed to handle lead: {str(e)}")