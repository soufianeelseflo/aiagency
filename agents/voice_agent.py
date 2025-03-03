# soufianeelseflo-aiagency/agents/voice_agent.py
import os
import json
import logging
import time
import random
import string
from typing import Dict, List
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
    def __init__(self, country_code: str = "US"):
        self.twilio_client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))
        self.twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
        if not all([self.twilio_client, self.twilio_number]):
            raise ValueError("TWILIO_SID, TWILIO_TOKEN, TWILIO_PHONE_NUMBER must be set.")
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
        self.budget_manager = BudgetManager(total_budget=50.0)
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
        self.pricing = {"base": 7000, "min": 3000, "discount_step": 500}  # Updated pricing
        self.boss_number = os.getenv("WHATSAPP_NUMBER")[9:]

    def _initialize_database(self):
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
            revenue FLOAT DEFAULT 0,
            cost FLOAT DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS strategy_scores (
            strategy_name VARCHAR(255) PRIMARY KEY,
            uses INTEGER DEFAULT 0,
            successes INTEGER DEFAULT 0,
            revenue FLOAT DEFAULT 0,
            cost FLOAT DEFAULT 0,
            roi FLOAT DEFAULT 0
        );
        """
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(create_tables)
                cursor.execute("SELECT COUNT(*) FROM scripts WHERE strategy_name IN ('pain_point_focus', 'value_stack')")
                if cursor.fetchone()[0] < 2:
                    cursor.execute(
                        "INSERT INTO scripts (strategy_name, script_text) VALUES (%s, %s), (%s, %s) ON CONFLICT DO NOTHING",
                        ("pain_point_focus", "Hey [name], I know high ad costs hurt [company]. Slash costs, skyrocket sales—your biggest headache?",
                         "value_stack", "Hi [name], imagine [company] saving 100 hours and doubling revenue with cheap ads. Wanna see how?")
                    )
            conn.commit()
            self.db_pool.putconn(conn)
            logging.info("DB initialized.")
        except psycopg2.Error as e:
            logging.error(f"DB init failed: {str(e)}")
            raise

    def _load_scripts(self):
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT strategy_name, script_text FROM scripts")
                self.scripts = {row[0]: row[1] for row in cursor.fetchall()}
            self.db_pool.putconn(conn)
            logging.info(f"Loaded {len(self.scripts)} scripts.")
        except psycopg2.Error as e:
            logging.error(f"Failed to load scripts: {str(e)}")
            raise

    def _load_context(self, client_id: str) -> Dict:
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT context FROM client_interactions WHERE client_id = %s ORDER BY timestamp DESC LIMIT 1", (client_id,))
                row = cursor.fetchone()
                context = row[0] if row else {"response": "", "objections": 0, "past_responses": [], "revenue": 0}
            self.db_pool.putconn(conn)
            return context
        except psycopg2.Error as e:
            logging.error(f"Failed to load context: {str(e)}")
            return {"response": "", "objections": 0, "past_responses": [], "revenue": 0}

    def _save_context(self, client_id: str, script: str, outcome: str, context: Dict, revenue: float, cost: float):
        self.update_queue.append((client_id, script, outcome, context, revenue, cost))
        if len(self.update_queue) >= 10 or time.time() - self.last_save_time > 300:
            asyncio.run(self._batch_save_updates())

    async def _batch_save_updates(self):
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                for client_id, script, outcome, context, revenue, cost in self.update_queue:
                    roi = (revenue - cost) / cost if cost > 0 and revenue > 0 else 0
                    cursor.execute(
                        "INSERT INTO client_interactions (client_id, script, outcome, context, revenue, cost) VALUES (%s, %s, %s, %s, %s, %s)",
                        (client_id, script, outcome, json.dumps(context), revenue, cost)
                    )
                    strategy = script[:50]
                    cursor.execute(
                        "INSERT INTO strategy_scores (strategy_name, uses, successes, revenue, cost, roi) "
                        "VALUES (%s, 1, %s, %s, %s, %s) ON CONFLICT (strategy_name) "
                        "DO UPDATE SET uses = strategy_scores.uses + 1, "
                        "successes = strategy_scores.successes + CASE WHEN %s = 'completed' THEN 1 ELSE 0 END, "
                        "revenue = strategy_scores.revenue + %s, "
                        "cost = strategy_scores.cost + %s, "
                        "roi = (strategy_scores.revenue + %s - strategy_scores.cost - %s) / (strategy_scores.cost + %s)",
                        (strategy, 1 if outcome == 'completed' else 0, revenue, cost, roi, outcome, revenue, cost, revenue, cost, cost)
                    )
                conn.commit()
            self.db_pool.putconn(conn)
            self.update_queue.clear()
            self._load_strategy_cache()
            logging.info("Batch updates saved.")
        except psycopg2.Error as e:
            logging.error(f"Failed to save updates: {str(e)}")

    def _load_strategy_cache(self):
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT strategy_name, uses, successes, revenue, cost, roi FROM strategy_scores")
                self.strategy_cache = {row[0]: {"uses": row[1], "successes": row[2], "revenue": row[3], "cost": row[4], "roi": row[5]} for row in cursor.fetchall()}
            self.db_pool.putconn(conn)
            logging.info(f"Loaded {len(self.strategy_cache)} strategies.")
        except psycopg2.Error as e:
            logging.error(f"Failed to load strategy cache: {str(e)}")
            self.strategy_cache = {}

    def _load_rag_data(self):
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, script, context, revenue FROM client_interactions")
                rows = cursor.fetchall()
                self.interaction_ids = [row[0] for row in rows]
                embeddings = [self._get_embedding(f"{row[1]} {json.dumps(row[2])} {row[3]}") for row in rows]
                if embeddings:
                    self.faiss_index.add(np.array(embeddings))
            self.db_pool.putconn(conn)
            logging.info(f"Loaded {self.faiss_index.ntotal} RAG entries.")
        except psycopg2.Error as e:
            logging.error(f"Failed to load RAG: {str(e)}")

    def _get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

    def _fetch_rag_context(self, lead_data: Dict, context: Dict) -> str:
        query = f"{lead_data.get('company', 'Boss')} {lead_data.get('pains', 'profit')} {json.dumps(context)}"
        embedding = self._get_embedding(query)
        D, I = self.faiss_index.search(np.array([embedding]), 5)
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                valid_ids = [self.interaction_ids[i] for i in I[0] if 0 <= i < len(self.interaction_ids)]
                if valid_ids:
                    cursor.execute("SELECT script, revenue FROM client_interactions WHERE id IN %s", (tuple(valid_ids),))
                    matches = cursor.fetchall()
                    return "\n".join([f"Past: {m[0]} - ${m[1]}" for m in matches])
                return ""
        except psycopg2.Error as e:
            logging.error(f"Failed to fetch RAG: {str(e)}")
            return ""
        finally:
            self.db_pool.putconn(conn)

    def _determine_language(self, country_code: str) -> str:
        language_map = {'US': 'en', 'CA': 'en', 'MX': 'es', 'BR': 'pt', 'AU': 'en', 'JP': 'ja', 'IN': 'hi', 'ZA': 'en'}
        return language_map.get(country_code.upper(), 'en')

    def detect_language(self, text: str) -> str:
        try:
            return detect(text) if text.strip() else self.current_language
        except Exception as e:
            logging.error(f"Language detection failed: {str(e)}")
            return self.current_language

    def _is_troll(self, context: Dict) -> bool:
        responses = context.get("past_responses", [])
        if len(responses) > 3 and len(set(responses[-3:])) == 1:
            return True
        troll_phrases = ["are you a bot", "this sucks", "spam", "lmao"]
        last_response = context.get("response", "").lower()
        return any(phrase in last_response for phrase in troll_phrases)

    def _select_strategy(self, lead_data: Dict, context: Dict) -> str:
        client_id = lead_data.get('email', 'anonymous')
        patterns = self._detect_patterns(context, [])
        viable = {name: data for name, data in self.strategy_cache.items() if data.get("roi", 0) > 1000 and not self._is_robotic(name, patterns)}
        if viable:
            strategy = max(viable.items(), key=lambda x: x[1]["roi"])[0]
            logging.info(f"Exploiting strategy for {client_id}: {strategy} (ROI: {viable[strategy]['roi']})")
        else:
            strategy = self._innovate_strategy(lead_data, context)
        return strategy

    def _innovate_strategy(self, lead_data: Dict, context: Dict) -> str:
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            return random.choice(list(self.scripts.keys()))
        rag_context = self._fetch_rag_context(lead_data, context)
        prompt = f"""
        Innovate a concise sales script for {lead_data['company']}:
        - Industry: {lead_data['industry']}
        - Pain Points: {lead_data['pains']}
        - Goal: $7000 sale
        - Past Successes: {rag_context or 'None—go big!'}
        - Tone: Friendly, profit-driven, professional
        - Keep it short (under 200 chars)
        Return script text.
        """
        response = self.ds.query(prompt, max_tokens=500)
        script = response['choices'][0]['message']['content']
        strategy_name = f"innovated_{random.randint(1000, 9999)}"
        self.scripts[strategy_name] = script
        self.strategy_cache[strategy_name] = {"uses": 0, "successes": 0, "revenue": 0, "cost": 0, "roi": 0}
        logging.info(f"Innovated strategy: {strategy_name}")
        return strategy_name

    def _detect_patterns(self, context: Dict, past_interactions: List[Dict]) -> List[str]:
        patterns = []
        response = context.get("response", "").lower()
        for keyword, pattern in [("objection", "objection"), ("confused", "confusion"), ("buy", "purchase")]:
            if keyword in response:
                patterns.append(pattern)
        return patterns

    def _is_robotic(self, strategy: str, patterns: List[str]) -> bool:
        robotic_triggers = {
            "objection": ["only 3 spots left", "act now"],
            "confusion": ["boost ROI", "save time"],
            "purchase": ["buy now", "limited time"]
        }
        script_text = self.scripts.get(strategy, "").lower()
        for pattern in patterns:
            if any(trigger in script_text for trigger in robotic_triggers.get(pattern, [])):
                return True
        return False

    def _negotiate_pricing(self, context: Dict) -> str:
        objections = context.get("objections", 0)
        current_price = max(self.pricing["min"], self.pricing["base"] - (self.pricing["discount_step"] * objections))
        return f"${current_price}/mo—AI UGC rocks!"  # Professional, not salesy

    def generate_sales_script(self, lead_data: Dict, context: Dict) -> str:
        client_id = lead_data.get('email', 'anonymous')
        cached_script = self.cache.get(f"script_{client_id}")
        if cached_script:
            return cached_script
        is_boss = client_id == "anonymous" and "video_type" in lead_data
        strategy = self._select_strategy(lead_data, context)
        country_code = lead_data.get("country_code", "US")
        if not context.get("response"):
            self.current_language = self._determine_language(country_code)
            self.language_switches = 0
        else:
            client_language = self.detect_language(context.get("response", ""))
            if client_language != self.current_language and self.language_switches == 0:
                self.current_language = client_language
                self.language_switches += 1
                intro = f"Switching to {client_language}—deal time!"
            else:
                intro = f"Hey {lead_data.get('name', 'there')}! "
        pricing = self._negotiate_pricing(context) if not is_boss else ""
        script_template = self.scripts.get(strategy, "Hey [name], big wins with us!")
        script = f"{intro}{script_template.replace('[name]', lead_data.get('name', 'there')).replace('[company]', lead_data['company'])}"
        if not is_boss:
            script += f" {pricing} Let’s double your cash!"
        if is_boss:
            script = f"{intro}Boss, video {lead_data['video_type']} at {lead_data['video_url']}!"
        token_cost = (500 / 1_000_000 * 0.80) + (500 / 1_000_000 * 2.40) if "innovated" in strategy else 0
        self.budget_manager.log_usage(input_tokens=500 if "innovated" in strategy else 0, output_tokens=500 if "innovated" in strategy else 0)
        self.cache.set(f"script_{client_id}", script)
        return script

    def synthesize_voice(self, script: str) -> str:
        script = script[:500]
        voice_settings = VoiceSettings(stability=0.4, similarity_boost=0.7, style=0.1, use_speaker_boost=True)
        try:
            audio = generate(
                text=script,
                voice="Rachel",
                model="eleven_multilingual_v2",
                api_key=self.elevenlabs_api_key,
                voice_settings=voice_settings
            )
            audio_file = f"call_{int(time.time())}_{''.join(random.choices(string.ascii_lowercase, k=5))}.mp4"
            save(audio, audio_file)
            char_cost = len(script) * 0.0002
            self.budget_manager.log_usage(0, 0, additional_cost=char_cost)
            logging.info(f"Synthesized audio: {audio_file}, cost: ${char_cost:.4f}")
            return audio_file
        except Exception as e:
            logging.error(f"Failed to synthesize: {str(e)}")
            raise

    def initiate_call(self, lead_number: str, audio_file: str) -> str:
        try:
            call = self.twilio_client.calls.create(
                twiml=f'<Response><Play>{os.path.abspath(audio_file)}</Play></Response>',
                from_=self.twilio_number,
                to=lead_number,
                status_callback=self.callback_url,
                status_callback_event=['completed']
            )
            twilio_cost = 0.013
            self.budget_manager.log_usage(0, 0, additional_cost=twilio_cost)
            logging.info(f"Call to {lead_number}: {call.sid}, cost: ${twilio_cost:.4f}")
            return call.sid
        except TwilioRestException as e:
            logging.error(f"Failed to initiate call: {str(e)}")
            raise

    def get_call_outcome(self, call_sid: str, timeout: int = 30) -> str:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                call = self.twilio_client.calls(call_sid).fetch()
                if call.status in ['completed', 'failed', 'busy', 'no-answer']:
                    logging.info(f"Call {call_sid} status: {call.status}")
                    return call.status
                time.sleep(1)
            except TwilioRestException as e:
                logging.error(f"Failed to fetch status: {str(e)}")
                return "failed"
        logging.warning(f"Call {call_sid} timed out.")
        return "no-answer"

    def handle_lead(self, lead_data: Dict, context: Dict = {}) -> Dict:
        client_id = lead_data.get('email', 'anonymous')
        is_boss = client_id == "anonymous" and "video_type" in lead_data
        context = self._load_context(client_id) if not context else context
        script = self.generate_sales_script(lead_data, context)
        audio_file = self.synthesize_voice(script)
        call_sid = self.initiate_call(self.boss_number if is_boss else lead_data['phone'], audio_file)
        total_cost = (500 / 1_000_000 * 0.80 + 500 / 1_000_000 * 2.40 if "innovated" in script else 0) + len(script) * 0.0002 + 0.013
        outcome = self.get_call_outcome(call_sid)
        revenue = context.get("revenue", 7000) if outcome == "completed" and not is_boss else 0
        response_context = {
            "response": context.get("response", "Interested"),
            "objections": context.get("objections", 0) + (1 if outcome in ['failed', 'busy', 'no-answer'] and not is_boss else 0),
            "past_responses": context.get("past_responses", []) + [context.get("response", "")],
            "revenue": revenue
        }
        self._save_context(client_id, script, outcome, response_context, revenue, total_cost)
        return {"call_sid": call_sid, "strategy": script, "outcome": outcome, "context": response_context, "cost": total_cost}

    def get_status(self) -> dict:
        return {
            "calls_made": len(self.update_queue),
            "total_revenue": sum([item[4] for item in self.update_queue]),
            "remaining_budget": self.budget_manager.get_remaining_budget()
        }

    def handle_web_command(self, command: str) -> str:
        if command.startswith("call client"):
            lead_email = command.split(" ")[-1]
            lead = {"email": lead_email, "phone": "+12025550123", "company": "Unknown", "industry": "Unknown", "pains": "Unknown"}
            self.handle_lead(lead)
            return f"Calling {lead_email}!"
        return "Unknown command—try 'call client email@example.com'"

if __name__ == "__main__":
    agent = VoiceSalesAgent("US")
    lead = {"company": "EcomElite", "email": "sales@ecomelite.com", "phone": "+12025550123", "industry": "E-commerce", "pains": "High ad costs"}
    print(json.dumps(agent.handle_lead(lead), indent=2))