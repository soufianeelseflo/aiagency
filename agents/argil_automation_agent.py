import os
import logging
import random
import string
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from utils.proxy_rotator import ProxyRotator

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArgilAutomationAgent:
    def __init__(self):
        """Initialize the agent with proxy rotator and URLs."""
        self.proxy_rotator = ProxyRotator()  # Assumes ProxyRotator handles SmartProxy API key
        self.argil_register_url = "https://app.argil.ai/register"
        self.disposable_email_url = "https://tempmail.plus/en/"  # TempMail as disposable email service

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

    async def run_registration(self):
        """Execute the full registration process."""
        proxy = self.proxy_rotator.get_proxy()
        proxy_config = {"server": proxy}

        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(
                    headless=True,
                    proxy=proxy_config  # Ref: https://playwright.dev/python/docs/network#http-proxy
                )
                page = await browser.new_page()

                # Generate disposable email and random password
                email = await self.generate_disposable_email(page)
                password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))

                # Register on Argil.ai
                success = await self.register_argil_account(email, password, page)
                if not success:
                    raise Exception("Registration failed.")

                # Retrieve and verify code
                code = await self.retrieve_verification_code(page)
                if not code:
                    raise Exception("Failed to retrieve verification code.")

                verified = await self.complete_verification(code, page)
                if not verified:
                    raise Exception("Verification failed.")

                await browser.close()
                return {"status": "success", "email": email, "password": password}

            except Exception as e:
                logging.error(f"Registration process failed: {str(e)}")
                await browser.close()
                return {"status": "failure", "error": str(e)}

    def run(self):
        """Synchronous wrapper for async execution."""
        import asyncio
        return asyncio.run(self.run_registration())

if __name__ == "__main__":
    agent = ArgilAutomationAgent()
    result = agent.run()
    print(result)