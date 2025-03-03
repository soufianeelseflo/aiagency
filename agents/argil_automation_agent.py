# soufianeelseflo-aiagency/agents/argil_automation_agent.py
import os
import logging
import random
import string
import asyncio
import json
import subprocess
import requests
from typing import Dict, List
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from utils.proxy_rotator import ProxyRotator
from agents.research_engine import ResearchEngine
from utils.budget_manager import BudgetManager
from utils.cache_manager import CacheManager
import psycopg2
from psycopg2 import pool
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import re
from tenacity import retry, stop_after_attempt, wait_exponential
from browser_use import Agent  # https://github.com/browser-use/web-ui

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArgilAutomationAgent:
    def __init__(self):
        self.proxy_rotator = ProxyRotator()
        self.argil_register_url = "https://app.argil.ai/register"
        self.disposable_email_url = "https://tempmail.plus/en/"
        self.research_engine = ResearchEngine()
        self.budget_manager = BudgetManager(
            total_budget=float(os.getenv("TOTAL_BUDGET", "20.0")),
            input_cost_per_million=float(os.getenv("INPUT_COST_PER_M", "0.80")),
            output_cost_per_million=float(os.getenv("OUTPUT_COST_PER_M", "2.40"))
        )
        self.cache = CacheManager()
        self.twilio_client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))
        self.twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        self.my_whatsapp_number = os.getenv("WHATSAPP_NUMBER")
        if not all([self.twilio_client, self.my_whatsapp_number]):
            raise ValueError("TWILIO_SID, TWILIO_TOKEN, and WHATSAPP_NUMBER must be set.")
        self.db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=5,
            maxconn=20,
            dbname=os.getenv('POSTGRES_DB', 'smma_db'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host=os.getenv('POSTGRES_HOST', 'postgres')
        )
        self._initialize_database()
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.faiss_index = faiss.IndexFlatL2(384)
        self.feedback_ids = []
        self._load_rag_data()
        self.video_types_seen = set()
        self.browser_agent = Agent(
            llm="deepseek/deepseek-r1",
            chrome_path=os.getenv("CHROME_PATH", "/usr/bin/google-chrome"),
            chrome_user_data=os.getenv("CHROME_USER_DATA", ""),
            persistent_session=os.getenv("CHROME_PERSISTENT_SESSION", "true") == "true"
        )
        self.video_dir = "/var/www/nexusplan.store/videos"
        os.makedirs(self.video_dir, exist_ok=True)  # Auto-create folder per https://docs.python.org/3/library/os.html

    def _initialize_database(self):
        """Set up tables per https://www.psycopg.org/docs/pool.html"""
        create_tables_query = """
        CREATE TABLE IF NOT EXISTS video_deliveries (
            id SERIAL PRIMARY KEY,
            client_id VARCHAR(255),
            video_id VARCHAR(255),
            url TEXT,
            delivered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS video_feedback (
            id SERIAL PRIMARY KEY,
            client_id VARCHAR(255),
            video_id VARCHAR(255),
            feedback TEXT,
            score INTEGER,
            received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(create_tables_query)
            conn.commit()
            self.db_pool.putconn(conn)
            logging.info("Database tables initialized.")
        except psycopg2.Error as e:
            logging.error(f"Database initialization failed: {str(e)}")
            raise

    def _load_rag_data(self):
        """Load feedback into FAISS per https://github.com/facebookresearch/faiss"""
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, feedback, score FROM video_feedback")
                rows = cursor.fetchall()
                self.feedback_ids = [row[0] for row in rows]
                embeddings = [self._get_embedding(f"{row[1]} {row[2]}") for row in rows]
                if embeddings:
                    self.faiss_index.add(np.array(embeddings))
            self.db_pool.putconn(conn)
            logging.info(f"Loaded {self.faiss_index.ntotal} RAG entries.")
        except psycopg2.Error as e:
            logging.error(f"Failed to load RAG data: {str(e)}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding per https://huggingface.co/docs/transformers"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

    def _fetch_rag_feedback(self, client_data: Dict) -> str:
        """Retrieve past feedback."""
        query = f"{client_data['company']} {client_data.get('feedback', '')} {client_data.get('score', '')}"
        embedding = self._get_embedding(query)
        D, I = self.faiss_index.search(np.array([embedding]), k=5)
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                valid_ids = [self.feedback_ids[i] for i in I[0] if 0 <= i < len(self.feedback_ids)]
                if valid_ids:
                    cursor.execute(
                        "SELECT feedback, score FROM video_feedback WHERE id IN %s",
                        (tuple(valid_ids),)
                    )
                    matches = cursor.fetchall()
                    return "\n".join([f"Past Feedback: {m[0]} (Score: {m[1]})" for m in matches]) or ""
                return ""
        except psycopg2.Error as e:
            logging.error(f"Failed to fetch RAG feedback: {str(e)}")
            return ""
        finally:
            self.db_pool.putconn(conn)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_disposable_email(self, page):
        """Generate temp email per https://playwright.dev/python/docs/api/class-page"""
        try:
            await page.goto(self.disposable_email_url)
            await page.wait_for_selector('input#email', timeout=10000)
            email = await page.input_value('input#email')
            logging.info(f"Generated email: {email}")
            self.cache.set(f"email_{email}", {"email": email})
            return email
        except PlaywrightTimeoutError:
            logging.error("Failed to load TempMail.")
            raise

    async def create_argil_account(self) -> Dict:
        """Create Argil account with browser-use/web-ui per https://github.com/browser-use/web-ui"""
        async with async_playwright() as p:
            proxy = self.proxy_rotator.get_proxy()
            browser = await p.chromium.launch(headless=True, proxy={"server": proxy})
            page = await browser.new_page()
            email = await self.generate_disposable_email(page)
            password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
            task = f"""
            Navigate to {self.argil_register_url}.
            Analyze the page for an email field and fill it with '{email}'.
            Find the password field and fill it with '{password}'.
            Locate the submit button (might say 'Sign Up' or 'Register') and click it.
            Wait for a verification page (URL may include 'verify').
            Return 'Verification needed' if 'verify' in URL, else 'Failed'.
            """
            result = await self.browser_agent.run(task)
            if "Verification needed" in result:
                code = await self.retrieve_verification_code(page)
                if code:
                    verify_task = f"""
                    Analyze the page for a code input field and fill it with '{code}'.
                    Find the submit button (might say 'Verify' or 'Confirm') and click it.
                    Wait for the dashboard (URL may include 'dashboard').
                    Return 'Success' if 'dashboard' in URL, else 'Failed'.
                    """
                    verify_result = await self.browser_agent.run(verify_task)
                    if "Success" in verify_result:
                        logging.info(f"Created Argil account: {email}")
                        return {"email": email, "password": password}
            logging.error("Account creation failed.")
            await browser.close()
            return {"email": None, "password": None}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def retrieve_verification_code(self, page):
        """Retrieve code from TempMail."""
        try:
            await page.goto(self.disposable_email_url)
            await page.wait_for_selector('.message-item', timeout=30000)
            await page.click('.message-item')
            await page.wait_for_selector('.message-body', timeout=10000)
            code_element = await page.query_selector('.message-body')
            code_text = await code_element.inner_text()
            code = re.search(r'\b\d{6}\b', code_text).group(0) if re.search(r'\b\d{6}\b', code_text) else None
            if code:
                logging.info(f"Extracted code: {code}")
                return code
            logging.error("No code found.")
            return None
        except PlaywrightTimeoutError:
            logging.error("Timeout retrieving code.")
            raise

    async def find_client_images(self, client_data: Dict) -> List[str]:
        """Search Google for images per research_engine.py"""
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=500):
            logging.warning("Budget exceeded for image search.")
            return []
        query = f"images related to {client_data['company']} {client_data['industry']}"
        cached = self.cache.get(query)
        if cached:
            logging.info(f"Cache hit: {query}")
            return cached
        images = self.research_engine.document_search(query, max_tokens=500)
        image_urls = [img['url'] for img in json.loads(images) if 'url' in img] if images else []
        self.cache.set(query, image_urls)
        return image_urls[:1]  # Take first image

    def edit_video_locally(self, video_path: str, video_type: str) -> str:
        """Edit video with FFmpeg per https://ffmpeg.org/documentation.html"""
        output_path = f"{self.video_dir}/edited_{os.path.basename(video_path)}"
        try:
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-ss", "5" if video_type == "short" else "0",
                "-t", "30" if video_type == "short" else "60",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-y",
                output_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info(f"Video edited at {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg failed: {str(e)}")
            return video_path

    async def deliver_video(self, client_data: Dict, video_path: str):
        """Deliver to client via email_manager.py"""
        video_url = f"{os.getenv('VIDEO_STORAGE_URL', 'https://nexusplan.store/videos')}/{os.path.basename(video_path)}"
        email_body = f"Dear {client_data['company']},\n\nHere’s your custom UGC video: {video_url}\n\nBest,\nNexusPlan Elite Agency"
        from agents.email_manager import EmailManager
        email_manager = EmailManager()
        email_manager.send_campaign([client_data.get('email', 'default@client.com')], email_body)
        logging.info(f"Video emailed to {client_data.get('email', 'anonymous')}")
        conn = self.db_pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO video_deliveries (client_id, video_id, url) VALUES (%s, %s, %s)",
                (client_data.get('email', 'anonymous'), os.path.basename(video_path), video_url)
            )
        conn.commit()
        self.db_pool.putconn(conn)

    async def notify_owner_voice(self, video_type: str, video_url: str):
        """Voice notify via voice_agent.py per https://www.twilio.com/docs/voice"""
        from agents.voice_agent import VoiceSalesAgent
        voice_agent = VoiceSalesAgent()
        context = {"response": f"Boss, here’s a new type of video: {video_type} at {video_url}!"}
        voice_agent.handle_lead({"phone": self.my_whatsapp_number[9:], "country_code": "US"}, context)
        logging.info(f"Voice notified owner of new video type: {video_type}")

    def learn_from_video(self, client_data: Dict, video_id: str, feedback: str, score: int):
        """Log feedback."""
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            logging.warning("Budget exceeded for learning.")
            return
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO video_feedback (client_id, video_id, feedback, score) VALUES (%s, %s, %s, %s)",
                    (client_data.get('email', 'anonymous'), video_id, feedback, score)
                )
            conn.commit()
            self.db_pool.putconn(conn)
            self._load_rag_data()
            logging.info(f"Learned from feedback for {video_id}: {feedback} (Score: {score})")
        except psycopg2.Error as e:
            logging.error(f"Failed to save feedback: {str(e)}")

    async def create_ugc_video(self, client_data: Dict) -> Dict:
        """Create UGC video autonomously."""
        account = await self.create_argil_account()
        if not account["email"]:
            return {"status": "failure", "error": "Account creation failed"}
        
        background_images = await self.find_client_images(client_data)
        background_url = background_images[0] if background_images else "https://via.placeholder.com/1920x1080"  # Fallback
        video_type = client_data.get('video_type', random.choice(['short', 'long']))
        rag_feedback = self._fetch_rag_feedback(client_data)
        
        task = f"""
        Log in to https://app.argil.ai with email '{account["email"]}' and password '{account["password"]}'.
        Analyze the page for a video creation option (might say 'Create Video' or 'Generate').
        Click the video creation button.
        If a script field exists, enter: "Hey {client_data['company']}, boost your {client_data['industry']} with our ads—save time and money!"
        If a background field exists, set it to '{background_url}'.
        If a style option exists, choose 'Casual, human-like, {'30-second' if video_type == 'short' else '60-second'} testimonial'.
        Submit the form (look for 'Generate', 'Create', or 'Submit').
        Wait for the video to process (check for a download link or completion message).
        Download the video and return the URL.
        """
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=2000):
            return {"status": "failure", "error": "Budget exceeded"}
        
        video_url = await self.browser_agent.run(task)
        if not video_url or "http" not in video_url:
            return {"status": "failure", "error": "Video generation failed"}
        
        video_id = f"video_{random.randint(1000, 9999)}"
        video_path = f"{self.video_dir}/{video_id}.mp4"
        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            with open(video_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        edited_video_path = self.edit_video_locally(video_path, video_type)
        final_url = f"{os.getenv('VIDEO_STORAGE_URL', 'https://nexusplan.store/videos')}/{os.path.basename(edited_video_path)}"
        
        if video_type not in self.video_types_seen:
            self.video_types_seen.add(video_type)
            await self.notify_owner_voice(video_type, final_url)
        else:
            await self.deliver_video(client_data, edited_video_path)
        
        feedback = client_data.get('feedback', "Awaiting owner approval")
        score = client_data.get('score', 80)
        self.learn_from_video(client_data, video_id, feedback, score)
        
        return {"video_id": video_id, "url": final_url, "status": "success"}

    def run(self) -> Dict:
        """Run with sample data."""
        return asyncio.run(self.create_ugc_video({"company": "EcomElite", "industry": "E-commerce"}))