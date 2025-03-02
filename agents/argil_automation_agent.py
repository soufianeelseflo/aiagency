# argil_automation_agent.py
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
from integrations.argil_ai import ArgilVideoProducer
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArgilAutomationAgent:
    def __init__(self):
        self.proxy_rotator = ProxyRotator()
        self.argil_register_url = "https://app.argil.ai/register"
        self.disposable_email_url = "https://tempmail.plus/en/"
        self.argil_producer = ArgilVideoProducer()
        self.research_engine = ResearchEngine()
        self.budget_manager = BudgetManager()
        self.cache = CacheManager()
        self.twilio_client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))
        self.twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        self.my_whatsapp_number = os.getenv("WHATSAPP_NUMBER")
        if not all([self.twilio_client, self.my_whatsapp_number]):
            raise ValueError("TWILIO_SID, TWILIO_TOKEN, and WHATSAPP_NUMBER must be set.")
        
        # PostgreSQL setup
        self.db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=5,
            maxconn=20,
            dbname=os.getenv('POSTGRES_DB', 'smma_db'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host=os.getenv('POSTGRES_HOST', 'postgres')
        )
        self._initialize_database()

        # RAG setup
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.faiss_index = faiss.IndexFlatL2(384)  # 384-dim embeddings
        self.feedback_ids = []
        self._load_rag_data()

    def _initialize_database(self):
        """Set up PostgreSQL tables for video tracking and learning."""
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
        """Load past video feedback into FAISS for RAG."""
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
            logging.info(f"Loaded {self.faiss_index.ntotal} RAG entries for video feedback.")
        except psycopg2.Error as e:
            logging.error(f"Failed to load RAG data: {str(e)}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for RAG retrieval."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

    def _fetch_rag_feedback(self, client_data: Dict) -> str:
        """Retrieve relevant past feedback using RAG."""
        query = f"{client_data['company']} {client_data.get('feedback', '')} {client_data.get('score', '')}"
        embedding = self._get_embedding(query)
        D, I = self.faiss_index.search(np.array([embedding]), k=5)  # Top 5 matches
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

    async def generate_disposable_email(self, page):
        """Generate a disposable email for Argil registration."""
        try:
            await page.goto(self.disposable_email_url)
            await page.wait_for_selector('input#email', timeout=10000)
            email = await page.input_value('input#email')
            logging.info(f"Generated disposable email: {email}")
            self.cache.set(f"email_{email}", {"email": email})
            return email
        except PlaywrightTimeoutError:
            logging.error("Failed to load TempMail page.")
            raise

    async def register_argil_account(self, email: str, password: str, page):
        """Register a new Argil account."""
        try:
            await page.goto(self.argil_register_url)
            await page.fill('input[name="email"]', email)
            await page.fill('input[name="password"]', password)
            await page.click('button[type="submit"]')
            await page.wait_for_load_state("networkidle", timeout=30000)
            if "verify" in page.url:
                logging.info("Registration initiated.")
                return True
            logging.error("Registration failed.")
            return False
        except PlaywrightTimeoutError:
            logging.error("Timeout during registration.")
            return False

    async def retrieve_verification_code(self, page):
        """Retrieve the verification code from disposable email."""
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
            return None

    async def complete_verification(self, code: str, page):
        """Complete Argil account verification."""
        try:
            await page.fill('input[name="code"]', code)
            await page.click('button[type="submit"]')
            await page.wait_for_load_state("networkidle", timeout=30000)
            if "dashboard" in page.url:
                logging.info("Verification successful.")
                return True
            logging.error("Verification failed.")
            return False
        except PlaywrightTimeoutError:
            logging.error("Timeout during verification.")
            return False

    async def find_client_images(self, client_data: Dict) -> List[str]:
        """Search for client-related images, using cache for efficiency."""
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=500):
            logging.warning("Budget exceeded for image search.")
            return []
        query = f"images related to {client_data['company']} website or products"
        cached = self.cache.get(query)
        if cached:
            logging.info(f"Cache hit for image query: {query}")
            return cached
        images = self.research_engine.document_search(query, max_tokens=500)
        image_urls = [img['url'] for img in json.loads(images) if 'url' in img] if images else []
        self.cache.set(query, image_urls)
        return image_urls

    def edit_video_locally(self, video_path: str, video_type: str) -> str:
        """Edit video locally with FFmpeg for length optimization."""
        output_path = f"edited_{os.path.basename(video_path)}"
        try:
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-ss", "5" if video_type == "short" else "0",
                "-t", "30" if video_type == "short" else "60",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-y",  # Overwrite output
                output_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info(f"Video edited at {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg failed: {str(e)}")
            return video_path

    async def deliver_video(self, client_data: Dict, video_path: str):
        """Deliver video to client via WhatsApp and log it."""
        try:
            with open(video_path, 'rb') as video_file:
                message = self.twilio_client.messages.create(
                    body=f"Here’s your video, {client_data['company']}!",
                    from_=self.twilio_whatsapp_number,
                    to=client_data.get('whatsapp_number', self.my_whatsapp_number),
                    media_url=[f"file://{os.path.abspath(video_path)}"]
                )
            logging.info(f"Video delivered via WhatsApp: {message.sid}")
            conn = self.db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO video_deliveries (client_id, video_id, url) VALUES (%s, %s, %s)",
                    (client_data.get('email', 'anonymous'), os.path.basename(video_path), f"file://{video_path}")
                )
            conn.commit()
            self.db_pool.putconn(conn)
        except TwilioRestException as e:
            logging.error(f"Failed to deliver video: {str(e)}")

    def learn_from_video(self, client_data: Dict, video_id: str, feedback: str, score: int):
        """Log feedback and analyze it for future improvement."""
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
            # Reload RAG index with new feedback
            self._load_rag_data()
            logging.info(f"Learned from feedback for video {video_id}: {feedback} (Score: {score})")
        except psycopg2.Error as e:
            logging.error(f"Failed to save feedback: {str(e)}")

    async def create_ugc_video(self, client_data: Dict) -> Dict:
        """Create and deliver a UGC video, enhanced with RAG feedback."""
        proxy = self.proxy_rotator.get_proxy()
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(headless=True, proxy={"server": proxy})
                page = await browser.new_page()

                # Register Argil account
                email = await self.generate_disposable_email(page)
                password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
                if not await self.register_argil_account(email, password, page):
                    raise Exception("Registration failed.")
                
                code = await self.retrieve_verification_code(page)
                if not code or not await self.complete_verification(code, page):
                    raise Exception("Verification failed.")

                # Create avatar
                avatar_name = f"Avatar_{client_data['company']}"
                dataset_url = "https://your-vps.com/dataset.mp4"  # Replace with real VPS-hosted file
                consent_url = "https://your-vps.com/consent.mp4"  # Replace with real VPS-hosted file
                avatar = self.argil_producer.create_avatar(avatar_name, dataset_url, consent_url)

                # Fetch background and RAG feedback
                background_images = await self.find_client_images(client_data)
                background_url = background_images[0] if background_images else "https://via.placeholder.com/1920x1080"
                video_type = client_data.get('video_type', 'short')
                rag_feedback = self._fetch_rag_feedback(client_data)

                # Generate video with RAG-enhanced prompt
                prompt = f"""
                Create a UGC video:
                - Avatar: {avatar_name}
                - Script: "Hey {client_data['company']}, boost your {client_data['industry']} with our ads—save time and money!"
                - Background: {background_url}
                - Style: Casual, human-like, {'30-second' if video_type == 'short' else '60-second'} testimonial
                - Past Feedback: {rag_feedback if rag_feedback else 'No prior feedback; aim for high engagement.'}
                """
                if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=2000):
                    raise Exception("Budget exceeded.")
                video = self.argil_producer.generate_video(avatar['video_id'], resolution="1080p")

                # Download and edit video
                video_url = video['url']
                video_path = f"video_{video['id']}.mp4"
                with requests.get(video_url, stream=True) as r:
                    r.raise_for_status()
                    with open(video_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                edited_video_path = self.edit_video_locally(video_path, video_type)
                await self.deliver_video(client_data, edited_video_path)

                # Learn from feedback (default for testing)
                feedback = client_data.get('feedback', "First video, awaiting score")
                score = client_data.get('score', 80)
                self.learn_from_video(client_data, video['id'], feedback, score)

                await browser.close()
                return {"video_id": video['id'], "url": video_url, "status": "success"}

            except Exception as e:
                logging.error(f"Video creation failed: {str(e)}")
                await browser.close()
                return {"status": "failure", "error": str(e)}

    def run(self) -> Dict:
        """Run the agent with sample client data."""
        return asyncio.run(self.create_ugc_video({"company": "EcomElite", "industry": "E-commerce"}))

if __name__ == "__main__":
    agent = ArgilAutomationAgent()
    result = agent.run()
    print(json.dumps(result, indent=2))