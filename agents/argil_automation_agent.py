# soufianeelseflo-aiagency/agents/argil_automation_agent.py
import os
import logging
import random
import string
import asyncio
import json
import subprocess
import requests
from typing import Dict
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
from browser_use import Agent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArgilAutomationAgent:
    def __init__(self):
        self.proxy_rotator = ProxyRotator()
        self.argil_register_url = "https://app.argil.ai/register"
        self.disposable_email_url = "https://tempmail.plus/en/"
        self.research_engine = ResearchEngine()
        self.budget_manager = BudgetManager(total_budget=50.0)
        self.cache = CacheManager()
        self.twilio_client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))
        self.twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        self.my_whatsapp_number = os.getenv("WHATSAPP_NUMBER")
        if not all([self.twilio_client, self.my_whatsapp_number]):
            raise ValueError("TWILIO_SID, TWILIO_TOKEN, WHATSAPP_NUMBER must be set.")
        self.db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=5, maxconn=20,
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
        os.makedirs(self.video_dir, exist_ok=True)

    def _initialize_database(self):
        create_tables_query = """
        CREATE TABLE IF NOT EXISTS video_deliveries (
            id SERIAL PRIMARY KEY,
            client_id VARCHAR(255),
            video_id VARCHAR(255),
            url TEXT,
            revenue FLOAT DEFAULT 0,
            cost FLOAT DEFAULT 0,
            delivered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS video_feedback (
            id SERIAL PRIMARY KEY,
            client_id VARCHAR(255),
            video_id VARCHAR(255),
            feedback TEXT,
            score INTEGER,
            revenue FLOAT DEFAULT 0,
            cost FLOAT DEFAULT 0,
            received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(create_tables_query)
            conn.commit()
            self.db_pool.putconn(conn)
            logging.info("DB initialized.")
        except psycopg2.Error as e:
            logging.error(f"DB init failed: {str(e)}")
            raise

    def _load_rag_data(self):
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, feedback, score, revenue FROM video_feedback")
                rows = cursor.fetchall()
                self.feedback_ids = [row[0] for row in rows]
                embeddings = [self._get_embedding(f"{row[1]} {row[2]} {row[3]}") for row in rows]
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

    def _fetch_rag_feedback(self, client_data: Dict) -> str:
        query = f"{client_data.get('company', 'generic')} {client_data.get('feedback', 'profit')} {client_data.get('score', 0)}"
        embedding = self._get_embedding(query)
        D, I = self.faiss_index.search(np.array([embedding]), k=5)
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                valid_ids = [self.feedback_ids[i] for i in I[0] if 0 <= i < len(self.feedback_ids)]
                if valid_ids:
                    cursor.execute("SELECT feedback, score, revenue FROM video_feedback WHERE id IN %s", (tuple(valid_ids),))
                    matches = cursor.fetchall()
                    return "\n".join([f"Past: {m[0]} (Score: {m[1]}, Revenue: ${m[2]})" for m in matches]) or ""
                return ""
        except psycopg2.Error as e:
            logging.error(f"Failed to fetch RAG: {str(e)}")
            return ""
        finally:
            self.db_pool.putconn(conn)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_disposable_email(self, page):
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
        async with async_playwright() as p:
            proxy = self.proxy_rotator.get_proxy()
            browser = await p.chromium.launch(headless=True, proxy={"server": proxy})
            page = await browser.new_page()
            email = await self.generate_disposable_email(page)
            password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
            task = f"""
            Navigate to {self.argil_register_url}.
            Fill email field with '{email}'.
            Fill password field with '{password}'.
            Click submit ('Sign Up' or 'Register').
            Wait for verification (URL may include 'verify').
            Return 'Verification needed' if 'verify' in URL, else 'Failed'.
            """
            result = await self.browser_agent.run(task)
            if "Verification needed" in result:
                code = await self.retrieve_verification_code(page)
                if code:
                    verify_task = f"""
                    Fill code field with '{code}'.
                    Click submit ('Verify' or 'Confirm').
                    Wait for dashboard (URL may include 'dashboard').
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
        try:
            await page.goto(self.disposable_email_url)
            await page.wait_for_selector('.message-item', timeout=30000)
            await page.click('.message-item')
            await page.wait_for_selector('.message-body', timeout=10000)
            code_text = await (await page.query_selector('.message-body')).inner_text()
            code = re.search(r'\b\d{6}\b', code_text).group(0) if re.search(r'\b\d{6}\b', code_text) else None
            if code:
                logging.info(f"Extracted code: {code}")
                return code
            logging.error("No code found.")
            return None
        except PlaywrightTimeoutError:
            logging.error("Timeout retrieving code.")
            raise

    async def find_client_images(self, client_data: Dict) -> str:
        query = f"high-conversion images for {client_data.get('industry', 'e-commerce')}"
        cached = self.cache.get(query)
        if cached:
            return cached
        if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
            return "https://via.placeholder.com/1920x1080"
        images = self.research_engine.document_search(query, max_tokens=500)
        image_urls = json.loads(images)
        url = image_urls[0]["url"] if image_urls else "https://via.placeholder.com/1920x1080"
        self.cache.set(query, url)
        cost = (500 / 1_000_000 * 0.80) + (500 / 1_000_000 * 2.40)  # OpenRouter cost
        self.budget_manager.log_usage(input_tokens=500, output_tokens=500)
        logging.info(f"Image search cost: ${cost:.4f}")
        return url

    def edit_video_locally(self, video_path: str, video_type: str) -> str:
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
            logging.info(f"Edited video: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg failed: {str(e)}")
            return video_path

    async def notify_owner_voice(self, video_type: str, video_url: str):
        context = {"response": f"Boss, new video type: {video_type} at {video_url}!"}
        result = self.voice_agent.handle_lead({"phone": self.my_whatsapp_number[9:], "country_code": "US"}, context)
        logging.info(f"Notified owner: {video_type}, cost: ${result['cost']:.4f}")

    def log_cost_and_revenue(self, video_id: str, revenue: float, cost: float):
        try:
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO video_feedback (client_id, video_id, feedback, score, revenue, cost) VALUES (%s, %s, %s, %s, %s, %s)",
                    ("anonymous", video_id, "Generated video", 80, revenue, cost)
                )
            conn.commit()
            self.db_pool.putconn(conn)
            self._load_rag_data()
            logging.info(f"Logged video {video_id}: Revenue ${revenue}, Cost ${cost:.4f}")
        except psycopg2.Error as e:
            logging.error(f"Failed to log: {str(e)}")

    async def run(self, client_data: Dict = None) -> Dict:
        client_data = client_data or {"company": "EcomElite", "industry": "E-commerce"}
        account = await self.create_argil_account()
        if not account["email"]:
            return {"status": "failure", "error": "Account creation failed"}

        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=1000):
            return {"status": "failure", "error": "Budget exceeded"}

        total_cost = 0
        proxy_cost = 0.001  # $0.001 per SmartProxy request
        total_cost += proxy_cost * 2  # Account creation + video gen
        self.budget_manager.log_usage(0, 0, additional_cost=proxy_cost * 2)

        rag_feedback = self._fetch_rag_feedback(client_data)
        prompt = f"""
        Design a high-ROI UGC video script for {client_data['company']}:
        - Industry: {client_data['industry']}
        - Goal: $5000 sale
        - Past Feedback: {rag_feedback or 'None—make it viral!'}
        - Tone: Casual, profit-driven
        Return: {{script: str, type: 'short' or 'long'}}
        """
        response = self.browser_agent.llm.query(prompt, max_tokens=1000)
        video_plan = json.loads(response['choices'][0]['message']['content'])
        token_cost = (1000 / 1_000_000 * 0.80) + (1000 / 1_000_000 * 2.40)
        total_cost += token_cost
        self.budget_manager.log_usage(input_tokens=1000, output_tokens=1000)

        background_url = await self.find_client_images(client_data)
        total_cost += 0.001  # Proxy for image search
        task = f"""
        Log in to https://app.argil.ai with email '{account["email"]}' and password '{account["password"]}'.
        Click video creation ('Create Video' or 'Generate').
        Fill script with: "{video_plan['script']}"
        Set background to '{background_url}'.
        Choose style: 'Casual, human-like, {'30-second' if video_plan['type'] == 'short' else '60-second'} testimonial'.
        Submit ('Generate', 'Create', or 'Submit').
        Wait for download link, return URL.
        """
        video_url = await self.browser_agent.run(task)
        if not video_url or "http" not in video_url:
            return {"status": "failure", "error": "Video generation failed"}

        video_id = f"video_{random.randint(1000, 9999)}"
        video_path = f"{self.video_dir}/{video_id}.mp4"
        with requests.get(video_url, stream=True, proxies={"http": self.proxy_rotator.get_proxy()}) as r:
            r.raise_for_status()
            with open(video_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        total_cost += 0.001  # Proxy for download

        edited_path = self.edit_video_locally(video_path, video_plan['type'])
        final_url = f"{os.getenv('VIDEO_STORAGE_URL', 'https://nexusplan.store/videos')}/{os.path.basename(edited_path)}"

        if video_plan['type'] not in self.video_types_seen:
            self.video_types_seen.add(video_plan['type'])
            await self.notify_owner_voice(video_plan['type'], final_url)
            total_cost += 0.05  # Estimated call cost
        self.log_cost_and_revenue(video_id, 0, total_cost)

        return {"video_id": video_id, "url": final_url, "status": "success", "cost": total_cost}

if __name__ == "__main__":
    agent = ArgilAutomationAgent()
    asyncio.run(agent.run())