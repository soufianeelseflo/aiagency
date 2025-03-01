import os
import logging
import random
import string
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from utils.proxy_rotator import ProxyRotator
from integrations.argil_ai import ArgilVideoProducer
from agents.research_engine import ResearchEngine
from utils.budget_manager import BudgetManager

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArgilAutomationAgent:
    def __init__(self):
        """Initialize the agent with proxy rotator, Argil.ai, and research tools."""
        self.proxy_rotator = ProxyRotator()
        self.argil_register_url = "https://app.argil.ai/register"
        self.disposable_email_url = "https://tempmail.plus/en/"
        self.argil_producer = ArgilVideoProducer()
        self.research_engine = ResearchEngine()
        self.budget_manager = BudgetManager()

    async def generate_disposable_email(self, page):
        """Generate a disposable email using TempMail."""
        try:
            await page.goto(self.disposable_email_url)
            logging.info("Navigated to TempMail.")
            await page.wait_for_selector('input#email', timeout=10000)  # Ref: https://playwright.dev/python/docs/selectors
            email = await page.input_value('input#email')
            logging.info(f"Generated disposable email: {email}")
            return email
        except PlaywrightTimeoutError:
            logging.error("Failed to load TempMail page.")
            raise

    async def register_argil_account(self, email, password, page):
        """Register a new Argil.ai account."""
        try:
            await page.goto(self.argil_register_url)
            logging.info("Navigated to Argil.ai registration page.")
            await page.fill('input[name="email"]', email)  # Ref: https://playwright.dev/python/docs/input
            await page.fill('input[name="password"]', password)
            await page.click('button[type="submit"]')
            logging.info("Submitted registration form.")
            await page.wait_for_load_state("networkidle", timeout=30000)  # Ref: https://playwright.dev/python/docs/network
            if "verify" in page.url:
                logging.info("Registration initiated, awaiting verification.")
                return True
            else:
                logging.error("Registration failed: Did not reach verification page.")
                return False
        except PlaywrightTimeoutError:
            logging.error("Timeout during registration.")
            return False

    async def retrieve_verification_code(self, page):
        """Retrieve the verification code from TempMail inbox."""
        try:
            await page.goto(self.disposable_email_url)
            logging.info("Checking TempMail inbox for verification email.")
            await page.wait_for_selector('.message-item', timeout=30000)
            await page.click('.message-item')  # Click first email
            await page.wait_for_selector('.message-body', timeout=10000)
            code_element = await page.query_selector('.message-body')
            code_text = await code_element.inner_text()
            code = self.extract_code(code_text)
            if code:
                logging.info(f"Extracted verification code: {code}")
                return code
            else:
                logging.error("Verification code not found in email.")
                return None
        except PlaywrightTimeoutError:
            logging.error("Timeout while retrieving verification code.")
            return None

    async def complete_verification(self, code, page):
        """Enter the verification code on Argil.ai."""
        try:
            await page.fill('input[name="code"]', code)
            await page.click('button[type="submit"]')
            logging.info("Submitted verification code.")
            await page.wait_for_load_state("networkidle", timeout=30000)
            if "dashboard" in page.url:
                logging.info("Verification successful, dashboard loaded.")
                return True
            else:
                logging.error("Verification failed: Did not reach dashboard.")
                return False
        except PlaywrightTimeoutError:
            logging.error("Timeout during verification.")
            return False

    def extract_code(self, email_text):
        """Extract a 6-digit verification code from email text."""
        import re
        match = re.search(r'\b\d{6}\b', email_text)
        return match.group(0) if match else None

    async def find_client_images(self, client_data: dict) -> list:
        """Use ResearchEngine to find Google images related to the client."""
        if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=500):
            logging.error("Budget exceeded for image search.")
            return []
        query = f"images related to {client_data['company']} website or products"
        images = self.research_engine.document_search(query, max_tokens=500)
        return [img['url'] for img in json.loads(images) if 'url' in img] if images else []

    async def create_ugc_video(self, client_data: dict) -> dict:
        """Create a UGC video with an AI avatar, client background, and voice."""
        # Generate disposable account for Argil.ai
        proxy = self.proxy_rotator.get_proxy()
        proxy_config = {"server": proxy}

        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(headless=True, proxy=proxy_config)
                page = await browser.new_page()

                email = await self.generate_disposable_email(page)
                password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))

                success = await self.register_argil_account(email, password, page)
                if not success:
                    raise Exception("Argil.ai registration failed.")

                code = await self.retrieve_verification_code(page)
                if not code:
                    raise Exception("Verification code retrieval failed.")

                verified = await self.complete_verification(code, page)
                if not verified:
                    raise Exception("Argil.ai verification failed.")

                # Create avatar and video
                avatar_name = f"Avatar_{client_data['company']}"
                dataset_url = "https://example.com/dataset_video.mp4"  # Placeholder, replace with actual URL
                consent_url = "https://example.com/consent_video.mp4"  # Placeholder, replace with actual URL
                avatar = self.argil_producer.create_avatar(avatar_name, dataset_url, consent_url)

                # Find client-related background images
                background_images = await self.find_client_images(client_data)
                background_url = background_images[0] if background_images else "https://via.placeholder.com/1920x1080"  # Default if no images

                # Generate video with avatar speaking, client background
                video_prompt = f"""
                Create a simple UGC video:
                - Avatar: {avatar_name}
                - Script: "Hey [name], boost your [industry] with our ads—save time and money!"
                - Background: {background_url}
                - Style: Casual, human-like, with the avatar in front, background related to {client_data['company']}
                """
                if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=2000):
                    raise Exception("Budget exceeded for video creation.")
                video = self.argil_producer.generate_video(avatar['video_id'], resolution="1080p")
                await browser.close()
                return {"video_id": video['id'], "url": video['url']}

            except Exception as e:
                logging.error(f"UGC video creation failed: {str(e)}")
                await browser.close()
                return {"status": "failure", "error": str(e)}

    def run(self):
        """Synchronous wrapper for async execution."""
        import asyncio
        return asyncio.run(self.create_ugc_video({"company": "EcomElite", "industry": "E-commerce"}))

if __name__ == "__main__":
    agent = ArgilAutomationAgent()
    result = agent.run()
    print(result)