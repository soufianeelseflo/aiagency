# soufianeelseflo-aiagency/agents/voice_agent.py
import os
import json
import logging
import time
import random
import string
from typing import Dict, Optional, List
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from elevenlabs import generate, save, VoiceSettings
from pathlib import Path
from langdetect import detect, DetectorFactory
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator
from utils.cache_manager import CacheManager
import psycopg2
from psycopg2 import pool
import asyncio
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DetectorFactory.seed = 0

class VoiceSalesAgent:
    """Genius voice agent for sales and boss updates, maximizing profit."""
    def __init__(self, country_code: str = "US"):
        self.twilio_client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))
        self.twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
        if not all([self.twilio_client, self.twilio_number]):
            raise ValueError("TWILIO_SID, TWILIO_TOKEN, and TWILIO_PHONE_NUMBER must be set.")
        self.callback_url = os.getenv("CALLBACK_URL")
        if not self.callback_url:
            raise ValueError("CALLBACK_URL must be set.")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY must be set.")
        self.db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=5, maxconn=20,
            dbname=os.getenv('POSTGRES_DB', 'smma_db'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host=os.getenv('POSTGRES_HOST', 'postgres')
        )
        self._initialize_database()
        self.budget_manager = BudgetManager()
        self.proxy_rotator = ProxyRotator()
        self.ds = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)
        self.cache = CacheManager()
        self.scripts_file = Path("scripts.json")
        self.strategy_cache = {}
        self.context_memory = {"clients": {}, "global": {"patterns": {}}}
        self.update_queue = []
        self.last_save_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.faiss_index = faiss.IndexFlatL2(384)
        self.interaction_ids = []
        self._load_scripts()
        self._load_strategy_cache()
        self._load_rag_data()
        self.current_language = self._determine_language(country_code)
        self.language_switches = 0
        self.pricing = {"base": 7000, "min": 3000, "discount_step": 500}
        self.boss_number = os.getenv("WHATSAPP_NUMBER")[9:]  # For updates

    def _initialize_database(self):
        """Set up tables per https://www.psycopg.org/docs/pool.html."""
        create_tables = """
        CREATE TABLE IF NOT EXISTS scripts (
            strategy_name VARCHAR(255) PRIMARY KEY,
            script_text TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS client_interactions (
            id SERIAL PRIMARY KEY,
            client_id VARCHAR(255),
            script TEXT,
            outcome VARCHAR(50),
            context JSONB,
            analysis TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS strategy_scores (
            strategy_name VARCHAR(255) PRIMARY KEY,
            uses INTEGER DEFAULT 0,
            successes INTEGER DEFAULT 0,
            score FLOAT DEFAULT 50,
            revenue FLOAT DEFAULT 0
        );
        """
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(create_tables)
                cursor.execute(
                    "SELECT COUNT(*) FROM scripts WHERE strategy_name IN ('pain_point_focus', 'value_stack')"
                )
                if cursor.fetchone()[0] < 2:
                    cursor.execute(
                        "INSERT INTO scripts (strategy_name, script_text) VALUES (%s, %s), (%s, %s) ON CONFLICT DO NOTHING",
                        ("pain_point_focus", "Hey [name], I know high ad costs hurt [company]. Slash costs, skyrocket sales—your biggest headache?",
                         "value_stack", "Hi [name], imagine [company] saving 100 hours and doubling revenue with cheap ads. Wanna see how?")
                    )
            conn.commit()
            self.db_pool.putconn(conn)
            logging.info("Database initialized with scripts.")
        except psycopg2.Error as e:
            logging.error(f"Database init failed: {str(e)}")
            raise

    def _load_scripts(self) -> None:
        """Load scripts."""
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT strategy_name, script_text FROM scripts")
                self.scripts = {row[0]: row[1] for row in cursor.fetchall()}
            self.db_pool.putconn(conn)
            logging.info(f"Loaded {len(self.scripts)} sales scripts.")
        except psycopg2.Error as e:
            logging.error(f"Failed to load scripts: {str(e)}")
            raise

    def _load_context(self, client_id: str) -> Dict:
        """Fetch context per https://www.psycopg.org/docs/."""
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT context FROM client_interactions WHERE client_id = %s ORDER BY timestamp DESC LIMIT 1",
                    (client_id,)
                )
                row = cursor.fetchone()
                context = row[0] if row else {"response": "", "objections": 0, "past_responses": [], "revenue": 0}
            self.db_pool.putconn(conn)
            return context
        except psycopg2.Error as e:
            logging.error(f"Failed to load context: {str(e)}")
            return {"response": "", "objections": 0, "past_responses": [], "revenue": 0}

    def _save_context(self, client_id: str, script: str, outcome: str, context: Dict, analysis: str) -> None:
        """Queue context updates."""
        self.update_queue.append((client_id, script, outcome, context, analysis))
        if len(self.update_queue) >= 10 or time.time() - self.last_save_time > 300:
            asyncio.run(self._batch_save_updates())

    def _load_strategy_cache(self) -> None:
        """Load strategy scores."""
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT strategy_name, uses, successes, score, revenue FROM strategy_scores")
                self.strategy_cache = {row[0]: {"uses": row[1], "successes": row[2], "score": row[3], "revenue": row[4]} for row in cursor.fetchall()}
            self.db_pool.putconn(conn)
            logging.info(f"Loaded strategy cache with {len(self.strategy_cache)} entries.")
        except psycopg2.Error as e:
            logging.error(f"Failed to load strategy cache: {str(e)}")
            self.strategy_cache = {}

    async def _batch_save_updates(self):
        """Batch save updates per https://www.psycopg.org/docs/."""
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                for client_id, script, outcome, context, analysis in self.update_queue:
                    revenue = context.get("revenue", 0)
                    cursor.execute(
                        "INSERT INTO client_interactions (client_id, script, outcome, context, analysis) VALUES (%s, %s, %s, %s, %s)",
                        (client_id, script, outcome, json.dumps(context), analysis)
                    )
                    strategy = script[:50]
                    cursor.execute(
                        "INSERT INTO strategy_scores (strategy_name, uses, successes, score, revenue) "
                        "VALUES (%s, 1, %s, %s, %s) ON CONFLICT (strategy_name) "
                        "DO UPDATE SET uses = strategy_scores.uses + 1, "
                        "successes = strategy_scores.successes + CASE WHEN %s = 'completed' THEN 1 ELSE 0 END, "
                        "score = (strategy_scores.successes::float + CASE WHEN %s = 'completed' THEN 1 ELSE 0 END) / (strategy_scores.uses + 1) * 100, "
                        "revenue = strategy_scores.revenue + %s",
                        (strategy, 1 if outcome == 'completed' else 0, 50 if not self.strategy_cache.get(strategy) else self.strategy_cache[strategy]['score'],
                         revenue, outcome, outcome, revenue)
                    )
                conn.commit()
            self.db_pool.putconn(conn)
            self.update_queue.clear()
            self._load_strategy_cache()
            logging.info("Batch updates saved.")
        except psycopg2.Error as e:
            logging.error(f"Failed to save batch updates: {str(e)}")

    def _load_rag_data(self):
        """Load interactions into FAISS per https://github.com/facebookresearch/faiss."""
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, script, context FROM client_interactions")
                rows = cursor.fetchall()
                self.interaction_ids = [row[0] for row in rows]
                embeddings = [self._get_embedding(f"{row[1]} {json.dumps(row[2])}") for row in rows]
                if embeddings:
                    self.faiss_index.add(np.array(embeddings))
            self.db_pool.putconn(conn)
            logging.info(f"Loaded {self.faiss_index.ntotal} RAG entries.")
        except psycopg2.Error as e:
            logging.error(f"Failed to load RAG data: {str(e)}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding per https://huggingface.co/docs/transformers."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

    def _fetch_rag_context(self, lead_data: Dict, context: Dict) -> str:
        """Fetch profit-driven context."""
        query = f"{lead_data.get('company', 'Boss')} {lead_data.get('pains', 'profit')} {json.dumps(context)}"
        embedding = self._get_embedding(query)
        D, I = self.faiss_index.search(np.array([embedding]), 5)
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                valid_ids = [self.interaction_ids[i] for i in I[0] if i < len(self.interaction_ids)]
                if valid_ids:
                    cursor.execute(
                        "SELECT script, context FROM client_interactions WHERE id IN %s",
                        (tuple(valid_ids),)
                    )
                    matches = cursor.fetchall()
                    return "\n".join([f"Past: {m[0]} - Revenue: {json.loads(m[1]).get('revenue', 0)}" for m in matches])
                return ""
        except psycopg2.Error as e:
            logging.error(f"Failed to fetch RAG context: {str(e)}")
            return ""
        finally:
            self.db_pool.putconn(conn)

    def _determine_language(self, country_code: str) -> str:
        """Determine language per https://pypi.org/project/langdetect/."""
        language_map = {'US': 'en', 'CA': 'en', 'MX': 'es', 'BR': 'pt', 'AU': 'en', 'JP': 'ja', 'IN': 'hi', 'ZA': 'en'}
        lang = language_map.get(country_code.upper(), 'en')
        logging.info(f"Determined language for {country_code}: {lang}")
        return lang

    def detect_language(self, text: str) -> str:
        """Detect language."""
        try:
            return detect(text) if text.strip() else self.current_language
        except Exception as e:
            logging.error(f"Language detection failed: {str(e)}")
            return self.current_language

    def _is_troll(self, context: Dict) -> bool:
        """Detect trolls."""
        responses = context.get("past_responses", [])
        if len(responses) > 3 and len(set(responses[-3:])) == 1:
            logging.info("Detected repetitive troll behavior.")
            return True
        troll_phrases = ["are you a bot", "this sucks", "spam", "lmao"]
        last_response = context.get("response", "").lower()
        if any(phrase in last_response for phrase in troll_phrases):
            logging.info("Detected troll phrase.")
            return True
        return False

    def _select_strategy(self, lead_data: Dict, context: Dict) -> str:
        """Pick profit-driven strategy."""
        client_id = lead_data.get('email', 'anonymous')
        patterns = self._detect_patterns(context, [])
        viable_strategies = {
            name: data for name, data in self.strategy_cache.items()
            if data.get("score", 50) >= 50 and data.get("revenue", 0) > 0 and not self._is_robotic(name, patterns)
        }
        if not viable_strategies:
            strategy = random.choice(list(self.scripts.keys()))
            logging.info(f"No profitable strategies for {client_id}, using random: {strategy}")
        else:
            strategy = max(viable_strategies.items(), key=lambda x: x[1].get("revenue", 0))[0]
            logging.info(f"Selected profit strategy for {client_id}: {strategy} (Revenue: {viable_strategies[strategy]['revenue']})")
        if strategy not in self.strategy_cache:
            self.strategy_cache[strategy] = {"score": 50, "uses": 0, "successes": 0, "revenue": 0}
        return strategy

    def _detect_patterns(self, context: Dict, past_interactions: List[Dict]) -> List[str]:
        """Detect patterns for profit."""
        patterns = []
        current_response = context.get("response", "").lower()
        for keyword, pattern in [("objection", "objection"), ("confused", "confusion"), ("pivot", "pivot"), ("buy", "purchase")]:
            if keyword in current_response:
                patterns.append(pattern)
        return patterns

    def _is_robotic(self, strategy: str, patterns: List[str]) -> bool:
        """Avoid robotic replies."""
        robotic_triggers = {
            "objection": ["only 3 spots left", "act now"],
            "confusion": ["boost ROI", "save time"],
            "pivot": ["generic pitch", "repeat offer"],
            "purchase": ["buy now", "limited time"]
        }
        script_text = self.scripts.get(strategy, "").lower()
        for pattern in patterns:
            if any(trigger in script_text for trigger in robotic_triggers.get(pattern, [])):
                logging.info(f"Strategy '{strategy}' flagged as robotic for pattern: {pattern}")
                return True
        return False

    def _negotiate_pricing(self, context: Dict) -> str:
        """Negotiate for max profit per https://www.elevenlabs.io/docs."""
        objections = context.get("objections", 0)
        revenue = context.get("revenue", 0)
        discount = min(self.pricing["discount_step"] * objections, self.pricing["base"] - self.pricing["min"])
        current_price = self.pricing["base"] - discount + (revenue // 1000 * 500)  # Boost price with past revenue
        pricing_options = [
            f"${current_price:,}/month—premium AI ads, 8x ROI guaranteed!",
            f"${current_price:,}/month—huge value, tailored for you!",
            f"${current_price:,}/month—custom fit, massive gains!",
            f"${current_price:,}/month—best deal for top results!",
            f"${self.pricing['min']:,}/month—rock-bottom, still elite!"
        ]
        return pricing_options[min(objections, len(pricing_options) - 1)]

    def generate_sales_script(self, lead_data: Dict, context: Dict) -> str:
        """Generate profit-driven script per https://openrouter.ai/docs."""
        client_id = lead_data.get('email', 'anonymous')
        cached_script = self.cache.get(f"script_{client_id}")
        if cached_script:
            return cached_script

        is_boss = client_id == "anonymous" and "video_type" in lead_data
        strategy = self._select_strategy(lead_data, context)
        patterns = self.context_memory["global"]["patterns"].get(client_id, [])
        country_code = lead_data.get("country_code", "US")
        if not context.get("response"):
            self.current_language = self._determine_language(country_code)
            self.language_switches = 0
        else:
            client_language = self.detect_language(context.get("response", ""))
            if client_language != self.current_language and self.language_switches == 0:
                self.current_language = client_language
                self.language_switches += 1
                intro = f"Noticed you’re comfy with {client_language}—let’s roll with that!"
            else:
                intro = f"Sticking with {self.current_language}—here’s the deal!"

        pricing = self._negotiate_pricing(context) if not is_boss else ""
        rag_context = self._fetch_rag_context(lead_data, context)

        if is_boss:
            prompt = f"""
            Generate a concise update script in {self.current_language} for the boss:
            - Video Type: {lead_data.get('video_type', 'short')}
            - URL: {lead_data.get('video_url', 'https://nexusplan.store/videos/sample.mp4')}
            - Tone: Excited, professional, profit-focused
            - Say: "Boss, here’s a new type of video: [type] at [url]!"
            """
        else:
            prompt = f"""
            Generate a professional, profit-driven sales script in {self.current_language} for {lead_data['company']}:
            - Industry: {lead_data['industry']}
            - Pain Points: {lead_data['pains']}
            - Offer: {pricing}
            - Previous Context: {json.dumps(context)}
            - RAG Context: {rag_context}
            - Avoid Robotic Replies: No generic urgency or cheesy lines.
            
            Use Alex Hormozi’s techniques:
            1. Value Stack: $100,000+ revenue, 8x ROI, 100+ hours saved.
            2. Pain Point Focus: Address costs, slow content, headaches.
            3. Conversational: Human, friendly, Twitter-expert vibe.
            4. Dynamic: Adapt to patterns (objections, purchases).
            
            Structure:
            - "{intro}"
            - "Hey [name], I know [pain point] kills {lead_data['company']}..."
            - "Imagine [value stack]... We’ve crushed it for [industry] leaders."
            - "What’s your biggest challenge? Let’s double your revenue!"
            """
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=1000):
            logging.error("Budget exceeded for script generation.")
            return "Sorry, budget’s tapped—chat later!"
        
        response = self.ds.query(prompt, max_tokens=1000, temperature=0.7)
        script = response['choices'][0]['message']['content']

        if not is_boss and self._is_troll(context):
            troll_deterrent = random.choice([
                "\n\nHey, I love a chat, but let’s keep it real—how can I double your sales?",
                "\n\nYou’re testing me? I’m here for profits, not games—whatcha got?",
                "\n\nI solve problems, not dodge trolls—how can I help you cash in?"
            ])
            script += troll_deterrent
            logging.info(f"Added troll deterrent for {client_id}")

        self.cache.set(f"script_{client_id}", script)
        return script

    def synthesize_voice(self, script: str) -> str:
        """Synthesize voice per https://www.elevenlabs.io/docs."""
        script = script[:9000]  # ElevenLabs free tier limit
        voice_settings = VoiceSettings(stability=0.4, similarity_boost=0.7, style=0.1, use_speaker_boost=True)
        try:
            audio = generate(
                text=script,
                voice="Rachel",
                model="eleven_multilingual_v2",
                api_key=self.elevenlabs_api_key,
                voice_settings=voice_settings
            )
            audio_file = f"call_{int(time.time())}_{''.join(random.choices(string.ascii_lowercase, k=5))}.mp3"
            save(audio, audio_file)
            logging.info(f"Synthesized audio saved to {audio_file}")
            return audio_file
        except Exception as e:
            logging.error(f"Failed to synthesize audio: {str(e)}")
            raise

    def initiate_call(self, lead_number: str, audio_file: str) -> str:
        """Initiate call per https://www.twilio.com/docs/voice/api."""
        try:
            call = self.twilio_client.calls.create(
                twiml=f'<Response><Play>{os.path.abspath(audio_file)}</Play></Response>',
                from_=self.twilio_number,
                to=lead_number,
                status_callback=self.callback_url,
                status_callback_event=['completed']
            )
            logging.info(f"Call initiated to {lead_number}: {call.sid}")
            return call.sid
        except TwilioRestException as e:
            logging.error(f"Failed to initiate call: {str(e)}")
            raise

    def get_call_outcome(self, call_sid: str, timeout: int = 30) -> str:
        """Check call outcome."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                call = self.twilio_client.calls(call_sid).fetch()
                if call.status in ['completed', 'failed', 'busy', 'no-answer']:
                    logging.info(f"Call {call_sid} status: {call.status}")
                    return call.status
                time.sleep(1)
            except TwilioRestException as e:
                logging.error(f"Failed to fetch call status: {str(e)}")
                return "failed"
        logging.warning(f"Call {call_sid} timed out.")
        return "no-answer"

    def handle_lead(self, lead_data: Dict, context: Dict = {}) -> Dict:
        """Handle lead or boss update with profit focus."""
        client_id = lead_data.get('email', 'anonymous')
        is_boss = client_id == "anonymous" and "video_type" in lead_data
        context = self._load_context(client_id) if not context else context
        script = self.generate_sales_script(lead_data, context)
        audio_file = self.synthesize_voice(script)
        call_sid = self.initiate_call(self.boss_number if is_boss else lead_data['phone'], audio_file)

        outcome = self.get_call_outcome(call_sid)
        revenue = 5000 if outcome == "completed" and not is_boss else 0  # Assume $5000 per sale
        response_context = {
            "response": context.get("response", "Interested"),
            "objections": context.get("objections", 0) + (1 if outcome in ['failed', 'busy', 'no-answer'] and not is_boss else 0),
            "past_responses": context.get("past_responses", []) + [context.get("response", "")],
            "revenue": revenue
        }

        if self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            analysis_prompt = f"""
            Analyze this call for profit optimization:
            - Script: {script}
            - Outcome: {outcome}
            - Context: {json.dumps(response_context)}
            Suggest profit-driven improvements (e.g., higher revenue pitches).
            """
            analysis = self.ds.query(analysis_prompt, max_tokens=500)['choices'][0]['message']['content']
            logging.info(f"Call analysis: {analysis}")
        else:
            analysis = "Budget exceeded, no analysis."
            logging.warning(analysis)

        if client_id not in self.context_memory["clients"]:
            self.context_memory["clients"][client_id] = {"interactions": []}
        self.context_memory["clients"][client_id]["interactions"].append({
            "script": script,
            "outcome": outcome,
            "context": response_context,
            "analysis": analysis,
            "timestamp": time.time()
        })
        self._save_context(client_id, script, outcome, response_context, analysis)

        return {"call_sid": call_sid, "strategy": script, "outcome": outcome, "context": response_context, "analysis": analysis}

if __name__ == "__main__":
    agent = VoiceSalesAgent("US")
    lead = {"company": "EcomElite", "email": "sales@ecomelite.com", "phone": "+12025550123", "industry": "E-commerce", "pains": "High ad costs"}
    print(json.dumps(agent.handle_lead(lead, {"response": "Interested"}), indent=2))