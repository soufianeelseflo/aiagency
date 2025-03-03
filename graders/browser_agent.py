# soufianeelseflo-aiagency/agents/browser_agent.py
import os
import logging
import asyncio
import json
import random
import string
from typing import Dict, Optional
from browser_use import Agent
from playwright.async_api import async_playwright
from utils.proxy_rotator import ProxyRotator
from utils.budget_manager import BudgetManager
import psycopg2
from psycopg2 import pool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BrowserAgent:
    def __init__(self):
        self.proxy_rotator = ProxyRotator()
        self.budget_manager = BudgetManager(total_budget=50.0)
        self.temp_email_url = "https://tempmail.plus/en/"
        self.argil_register_url = "https://app.argil.ai/register"
        self.db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=5, maxconn=20,
            dbname=os.getenv('POSTGRES_DB', 'smma_db'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host=os.getenv('POSTGRES_HOST', 'postgres')
        )
        self._initialize_database()
        self.agent = Agent(
            llm="deepseek/deepseek-r1",
            chrome_path=os.getenv("CHROME_PATH", "/usr/bin/google-chrome"),
            chrome_user_data=os.getenv("CHROME_USER_DATA", ""),
            persistent_session=True
        )

    def _initialize_database(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS argil_accounts (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE,
            password VARCHAR(255),
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_query)
                conn.commit()
                self.db_pool.putconn(conn)
            logging.info("BrowserAgent DB initialized.")
        except psycopg2.Error as e:
            logging.error(f"DB init failed: {str(e)}")
            raise

    async def get_temp_email(self) -> str:
        task = f"""
        Navigate to {self.temp_email_url}.
        Wait for 'input#email' (timeout 10000ms).
        Get value of 'input#email'.
        Return the email.
        """
        result = await self.agent.run(task)
        if "Success" in result:
            email = result.split("returned: ")[-1].strip()
            logging.info(f"Generated temp email: {email}")
            return email
        raise Exception("Failed to get temp email")

    async def register_argil_account(self, email: str) -> Dict:
        password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
        task = f"""
        Navigate to {self.argil_register_url}.
        Fill 'input[name="email"]' with '{email}'.
        Fill 'input[name="password"]' with '{password}'.
        Click 'button[type="submit"]' or 'Sign Up'.
        Wait for URL to include 'verify' (timeout 30000ms).
        Return 'Verification needed' if 'verify' in URL, else 'Failed'.
        """
        result = await self.agent.run(task)
        if "Verification needed" in result:
            code = await self.get_verification_code()
            if code:
                verify_task = f"""
                Fill 'input[name="code"]' with '{code}'.
                Click 'button[type="submit"]' or 'Verify'.
                Wait for URL to include 'dashboard' (timeout 30000ms).
                Return 'Success' if 'dashboard' in URL, else 'Failed'.
                """
                verify_result = await self.agent.run(verify_task)
                if "Success" in verify_result:
                    return {"email": email, "password": password}
        logging.error("Argil registration failed")
        return {"email": None, "password": None}

    async def get_verification_code(self) -> Optional[str]:
        task = f"""
        Navigate to {self.temp_email_url}.
        Wait for '.message-item' (timeout 30000ms).
        Click '.message-item'.
        Wait for '.message-body' (timeout 10000ms).
        Get text of '.message-body'.
        Extract 6-digit code using regex '\\b\\d{{6}}\\b'.
        Return the code or None if not found.
        """
        result = await self.agent.run(task)
        if "Success" in result and "None" not in result:
            code = result.split("returned: ")[-1].strip()
            logging.info(f"Extracted code: {code}")
            return code
        return None

    def save_account(self, email: str, password: str):
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO argil_accounts (email, password) VALUES (%s, %s) ON CONFLICT (email) DO NOTHING",
                        (email, password)
                    )
                conn.commit()
                self.db_pool.putconn(conn)
            logging.info(f"Saved account: {email}")
        except psycopg2.Error as e:
            logging.error(f"Failed to save account: {str(e)}")

    def get_active_account(self) -> Optional[Dict]:
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT email, password FROM argil_accounts WHERE is_active LIMIT 1")
                    row = cursor.fetchone()
                self.db_pool.putconn(conn)
            if row:
                return {"email": row[0], "password": row[1]}
            return None
        except psycopg2.Error as e:
            logging.error(f"Failed to get account: {str(e)}")
            return None

    def deactivate_account(self, email: str):
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("UPDATE argil_accounts SET is_active = FALSE WHERE email = %s", (email,))
                conn.commit()
                self.db_pool.putconn(conn)
            logging.info(f"Deactivated account: {email}")
        except psycopg2.Error as e:
            logging.error(f"Failed to deactivate: {str(e)}")

    async def ensure_argil_account(self) -> Dict:
        account = self.get_active_account()
        if account and await self.check_trial_status(account["email"]):
            return account
        if account:
            self.deactivate_account(account["email"])
        email = await self.get_temp_email()
        new_account = await self.register_argil_account(email)
        if new_account["email"]:
            self.save_account(new_account["email"], new_account["password"])
            return new_account
        raise Exception("Failed to create Argil account")

    async def check_trial_status(self, email: str) -> bool:
        task = f"""
        Navigate to 'https://app.argil.ai/login'.
        Fill 'input[name="email"]' with '{email}'.
        Fill 'input[name="password"]' with '{self.get_active_account()["password"]}'.
        Click 'button[type="submit"]'.
        Wait for URL to include 'dashboard' (timeout 30000ms).
        Check if '.trial-expired' or 'Buy Now' exists.
        Return 'Expired' if found, else 'Active'.
        """
        result = await self.agent.run(task)
        return "Active" in result

if __name__ == "__main__":
    agent = BrowserAgent()
    account = asyncio.run(agent.ensure_argil_account())
    print(f"Active Argil account: {account}")