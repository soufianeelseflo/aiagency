# Filename: agents/browsing_agent.py
# Description: Agentic Browsing Agent with enhanced plan execution.
# Version: 3.5 (Removed hardcoded OpenAI model preference)

import asyncio
import logging
import json
import os
import random
import re
import time
import aiohttp
import uuid
import base64
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union, Tuple, AsyncGenerator, Type
from urllib.parse import urlparse, urljoin, quote_plus, unquote

# --- Core Framework Imports ---
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select, update, func, delete
from sqlalchemy.exc import SQLAlchemyError

# --- Project Imports ---
try:
    from .base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
except ImportError:
    logging.warning("Production base agent relative import failed, trying absolute.")
    try:
        from agents.base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
    except ImportError:
        logging.critical("Failed to import GeniusAgentBase. Check PYTHONPATH or project structure.")
        raise SystemExit("Cannot import GeniusAgentBase - critical dependency missing.")

try:
    # Assuming models.py defines these structures
    from models import KnowledgeFragment, AccountCredentials, StrategicDirective
except ImportError:
     logging.error("Failed to import models from parent directory. Ensure models.py is accessible.")
     # Define dummy classes if models are unavailable during direct execution
     class KnowledgeFragment: pass
     class AccountCredentials: pass
     class StrategicDirective: pass

try:
    from config.settings import settings
    # Assuming database utils handle encryption/decryption if needed elsewhere
    # from utils.database import encrypt_data, decrypt_data
except ImportError:
    logging.critical("Failed to import settings. Check config/ directory.")
    raise SystemExit("Cannot import settings - critical dependency missing.")

# --- External Libraries ---
try:
    from playwright.async_api import (
        async_playwright, Playwright, Browser, Page, Error as PlaywrightError,
        TimeoutError as PlaywrightTimeoutError, BrowserContext, Locator
    )
except ImportError:
    logging.critical("Playwright library not found. Requires 'pip install playwright' and 'playwright install'.")
    raise
try:
    from faker import Faker
except ImportError:
    logging.critical("Faker library not found. Requires 'pip install Faker'. Needed for account creation.")
    raise
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
from bs4 import BeautifulSoup # Keep for auxiliary parsing if needed

# --- Temporary Email Service Integration (Conceptual - Requires specific implementation) ---
# Replace with actual library/API client for a service like 1secmail, mail.tm, etc.
# This is a placeholder demonstrating the required functions.
class TempMailService:
    """Conceptual interface for a temporary email service."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.faker = Faker()
        # Base URL or client setup for the chosen temp mail service API
        self.base_url = "https://api.1secmail.com/v1/" # Example using 1secmail (public, rate-limited)

    async def get_new_email_address(self) -> Optional[str]:
        """Generates a new temporary email address."""
        # Public services might generate random ones, private might allow custom domains
        try:
            # Simple random generation for public services like 1secmail
            async with aiohttp.ClientSession() as session:
                 # Generate a plausible but random username
                 local_part = self.faker.user_name() + str(random.randint(1000,9999))
                 # Get available domains from the service
                 async with session.get(f"{self.base_url}?action=getDomainList") as response:
                     if response.status == 200:
                         domains = await response.json()
                         if domains:
                             domain = random.choice(domains)
                             email = f"{local_part}@{domain}"
                             logger.info(f"Generated temporary email: {email}")
                             return email
                     logger.error(f"Failed to get domains from temp mail service. Status: {response.status}")
                     return None
        except Exception as e:
            logger.error(f"Error generating temp email address: {e}", exc_info=True)
            return None

    async def get_verification_link(self, email_address: str, timeout_seconds: int = 180, keyword: str = "verify") -> Optional[str]:
        """Polls the inbox for an email containing a verification link."""
        if not email_address or '@' not in email_address: return None
        local_part, domain = email_address.split('@', 1)
        start_time = time.time()
        logger.info(f"Polling temp email inbox for {email_address} for verification link (keyword: '{keyword}')...")
        while time.time() - start_time < timeout_seconds:
            try:
                async with aiohttp.ClientSession() as session:
                    # Check for new messages
                    url = f"{self.base_url}?action=getMessages&login={local_part}&domain={domain}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            messages = await response.json()
                            if messages:
                                # Check newest messages first
                                for message_header in reversed(messages):
                                    message_id = message_header.get('id')
                                    # Fetch full message content
                                    msg_url = f"{self.base_url}?action=readMessage&login={local_part}&domain={domain}&id={message_id}"
                                    async with session.get(msg_url) as msg_response:
                                        if msg_response.status == 200:
                                            message_content = await msg_response.json()
                                            body = message_content.get('htmlBody') or message_content.get('textBody', '')
                                            # Search for a link containing the keyword
                                            # More robust regex might be needed for different link formats
                                            match = re.search(rf'href=["\'](https?://[^\s"\'<>]+{re.escape(keyword)}[^\s"\'<>]+)["\']', body, re.IGNORECASE)
                                            if match:
                                                link = match.group(1)
                                                # Decode HTML entities
                                                link = unquote(link.replace('&amp;', '&'))
                                                logger.info(f"Found verification link in email {message_id}: {link}")
                                                # Optionally delete the message after finding the link
                                                # await self._delete_message(local_part, domain, message_id)
                                                return link
                                            else:
                                                logger.debug(f"Message {message_id} found, but no link with keyword '{keyword}'.")
                                        else: logger.warning(f"Failed to read message {message_id}. Status: {msg_response.status}")
                            else: logger.debug(f"No messages found yet for {email_address}.")
                        else: logger.warning(f"Failed to get messages for {email_address}. Status: {response.status}")
            except Exception as e:
                logger.error(f"Error polling temp email inbox: {e}", exc_info=True)
            await asyncio.sleep(15) # Wait before polling again
        logger.warning(f"Timeout waiting for verification link for {email_address}.")
        return None

    async def _delete_message(self, local_part: str, domain: str, message_id: int):
         """(Optional) Deletes a message from the temp inbox."""
         # Implementation depends heavily on the specific temp mail service API
         logger.debug(f"Placeholder: Would delete message {message_id} for {local_part}@{domain}")
         pass

# Configure logger
logger = logging.getLogger(__name__)
op_logger = logging.getLogger('OperationalLog') # Assumes operational log is configured elsewhere

# --- Meta Prompt ---
BROWSING_AGENT_META_PROMPT = """
You are the BrowsingAgent within the Synapse AI Sales System, operating with full autonomy.
Your Core Mandate: Execute complex web interactions with genius-level adaptability and human-like stealth. This includes advanced UI automation using visual reasoning (analyzing screenshots), managing numerous accounts with unique proxies, **autonomously creating new accounts** for services when required (including handling CAPTCHAs and email verification via integrated tools), performing targeted information discovery (Google Dorking), and extracting data. You operate without human intervention. Prioritize task success and robust anti-detection measures.
Key Responsibilities:
1.  **Advanced Visual UI Automation:** Interact with any web application via its UI. Use visual reasoning (analyzing screenshots via multimodal LLM) as the primary method to find elements, determine actions (login, click, input, upload, download, scroll), and navigate complex flows like **new account signup**. Adapt dynamically to UI changes. Handle standard HTML elements and visually identify non-standard controls.
2.  **Autonomous Account Management:** Manage a pool of accounts (fetched from DB). Assign and use unique proxies (via Orchestrator) for each context. Track account status. **Crucially, if a task requires an account for a service (e.g., free trial) and no suitable active account exists, initiate and execute the automated new account creation process for that service.**
3.  **Account Creation Engine:** Utilize generated identities (via Faker) and temporary email services (via integrated tools) to complete signup forms. Visually identify and interact with form fields, buttons, checkboxes. Handle CAPTCHAs by visually identifying them (e.g., "I'm not a robot" checkbox) and instructing the execution layer, or by describing visual challenges (e.g., "select all images with buses") to the multimodal LLM for coordinate-based interaction. Handle email verification by retrieving links/codes from the temporary email service and completing the verification step in the browser context.
4.  **Google Dorking & Credential Acquisition:** Execute Google Dorking scans as directed by ThinkTool. Generate dorks, perform searches, scrape results, visit links, and use LLM analysis (including visual, if analyzing screenshots of pages) to extract potential login credentials or sensitive information. Report findings securely. Attempt to utilize discovered credentials for login tasks if directed.
5.  **Human Emulation & Stealth:** Simulate human browsing patterns (variable delays, realistic mouse movements/scrolls, non-linear navigation paths, occasional minor "hesitations") to defeat bot detection systems.
6.  **Resilient Execution:** Gracefully handle UI changes, anti-bot measures, proxy failures, timeouts, login issues, and account creation failures using retries, adaptive strategies driven by visual LLM analysis, and robust error reporting.
7.  **Collaboration & Reporting:** Execute tasks delegated by Orchestrator/ThinkTool. Report extracted data, automation outcomes (including detailed account creation success/failure), errors, and operational insights. Log relevant findings securely to the Knowledge Base via ThinkTool.
**Goal:** Be the agency's fully autonomous, highly adaptable, multi-skilled operative on the web, capable of complex visual UI automation, seamless account creation, and stealthy information gathering across diverse platforms without human oversight. Execute instructions precisely and achieve objectives reliably.
"""

class BrowsingAgent(GeniusAgentBase):
    """
    Browsing Agent (Genius Level - Fully Autonomous): Handles web scraping, visual UI automation,
    autonomous multi-account management including creation, Google Dorking, credential acquisition,
    and data extraction with human emulation.
    Version: 3.5 (Removed hardcoded OpenAI model preference)
    """
    AGENT_NAME = "BrowsingAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any, smartproxy_password: Optional[str] = None):
        """Initializes the Autonomous BrowsingAgent."""
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, session_maker=session_maker)
        self.meta_prompt = BROWSING_AGENT_META_PROMPT
        self.think_tool = orchestrator.agents.get('think') # Assumes ThinkTool agent exists for KB logging/complex reasoning

        # --- Internal State Initialization ---
        self.internal_state = getattr(self, 'internal_state', {}) # Ensure state exists
        self.internal_state['playwright_instance'] = None
        self.internal_state['browser_instance'] = None
        self.internal_state['browser_contexts'] = {} # context_id -> BrowserContext
        self.internal_state['active_pages'] = {} # page_id -> {'page': Page, 'context_id': str}
        self.internal_state['context_account_map'] = {} # context_id -> account_details dict (from DB or newly created)
        self.internal_state['context_proxy_map'] = {} # context_id -> proxy_info dict
        self.internal_state['proxy_stats'] = {} # proxy_url -> {'success': N, 'failure': N, 'last_used': datetime}
        self.internal_state['user_agent'] = self.config.get("BROWSER_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36") # Use a recent UA
        self.internal_state['max_concurrent_contexts'] = int(self.config.get("BROWSER_MAX_CONCURRENT_CONTEXTS", 5)) # Lower default for resource intensity
        self.internal_state['context_semaphore'] = asyncio.Semaphore(self.internal_state['max_concurrent_contexts'])
        self.internal_state['default_timeout_ms'] = int(self.config.get("BROWSER_DEFAULT_TIMEOUT_MS", 120000)) # Longer default timeout
        self.temp_dir = self.config.get("TEMP_DOWNLOAD_DIR", "/app/temp_downloads") # Dedicated temp dir
        os.makedirs(self.temp_dir, exist_ok=True)

        # --- Initialize Tools ---
        self.faker = Faker()
        # Initialize Temp Mail Service (replace with actual implementation/config)
        self.temp_mail_service = TempMailService(api_key=self.config.get("TEMP_MAIL_API_KEY"))

        self.logger.info(f"{self.AGENT_NAME} v3.5 (Removed hardcoded OpenAI model) initialized. Max Contexts: {self.internal_state['max_concurrent_contexts']}")

    async def log_operation(self, level: str, message: str):
        """Helper to log to the operational log file."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err: logger.error(f"Failed to write to operational log: {log_err}")

    # --- Playwright Lifecycle Management (Robust checks) ---
    async def _ensure_playwright_running(self):
        if self.internal_state.get('playwright_instance'): return
        self.logger.info("Initializing Playwright...")
        try:
            self.internal_state['playwright_instance'] = await async_playwright().start()
            self.logger.info("Playwright initialized successfully.")
        except Exception as e:
            self.logger.critical(f"Failed to initialize Playwright: {e}", exc_info=True)
            self._status = self.STATUS_ERROR; await self._report_error(f"Failed Playwright init: {e}")
            raise RuntimeError(f"Playwright initialization failed: {e}") from e

    async def _launch_browser(self):
         await self._ensure_playwright_running()
         browser = self.internal_state.get('browser_instance')
         if browser and browser.is_connected(): return
         elif browser and not browser.is_connected():
             self.logger.warning("Existing browser instance disconnected. Closing and relaunching.")
             try: await browser.close()
             except Exception as close_err: self.logger.error(f"Error closing disconnected browser: {close_err}")
             self.internal_state['browser_instance'] = None

         self.logger.info("Launching persistent Playwright browser instance...")
         try:
             self.internal_state['browser_instance'] = await self.internal_state['playwright_instance'].chromium.launch(
                 headless=self.config.get("BROWSER_HEADLESS", True), # Configurable headless
                 args=[
                     '--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu',
                     '--disable-dev-shm-usage', '--disable-blink-features=AutomationControlled' # Anti-detection
                 ]
             )
             self.logger.info(f"Persistent browser launched successfully (Headless: {self.config.get('BROWSER_HEADLESS', True)}).")
         except Exception as e:
             self.logger.critical(f"Failed to launch persistent browser: {e}", exc_info=True)
             self._status = self.STATUS_ERROR; await self._report_error(f"Failed browser launch: {e}")
             raise RuntimeError(f"Browser launch failed: {e}") from e

    async def _get_new_context(self, account_details: Optional[Dict] = None, proxy_info: Optional[Dict] = None) -> Tuple[str, BrowserContext]:
        await self._launch_browser()
        await self.internal_state['context_semaphore'].acquire()
        context_id = f"ctx_{uuid.uuid4().hex[:8]}"
        context: Optional[BrowserContext] = None
        try:
            # Ensure proxy is obtained if not provided
            if not proxy_info:
                proxy_purpose = f"context_{account_details.get('service', 'general')}_{account_details.get('id', 'new')}" if account_details else "general_context_creation"
                if hasattr(self.orchestrator, 'get_proxy'):
                    proxy_info = await self.orchestrator.get_proxy(purpose=proxy_purpose, quality_level='high') # Request high quality proxy
                else:
                    self.logger.error("Orchestrator missing get_proxy method. Cannot obtain proxy."); proxy_info = None

            playwright_proxy = None
            if proxy_info and proxy_info.get('server'):
                 playwright_proxy = { "server": proxy_info["server"], "username": proxy_info.get("username"), "password": proxy_info.get("password") }
                 self.logger.info(f"Using proxy {proxy_info['server'].split('@')[-1]} for new context {context_id}")
            else: self.logger.warning(f"No valid proxy provided or obtained for new context {context_id}. Proceeding without proxy.")

            browser = self.internal_state['browser_instance']
            if not browser or not browser.is_connected(): raise RuntimeError("Browser instance unavailable.")

            context = await browser.new_context(
                user_agent=self.internal_state['user_agent'], proxy=playwright_proxy,
                java_script_enabled=True, ignore_https_errors=True,
                viewport={'width': random.randint(1366, 1920), 'height': random.randint(768, 1080)}, # Common resolutions
                locale=random.choice(['en-US', 'en-GB']), # More common locales
                timezone_id=random.choice(['America/New_York', 'America/Chicago', 'America/Los_Angeles', 'Europe/London', 'Europe/Paris']),
                permissions=['geolocation'], # Grant common permissions
                geolocation=self._generate_random_geo(), # Add random geolocation
                # color_scheme='light', # Can randomize light/dark
                bypass_csp=True # May help with certain interactions, use cautiously
            )
            # Anti-fingerprinting measures (basic)
            await context.add_init_script("""
                if (navigator.webdriver) { Object.defineProperty(navigator, 'webdriver', { get: () => false }); }
                // Add more sophisticated fingerprinting evasions if necessary
            """)

            self.internal_state['browser_contexts'][context_id] = context
            if account_details: self.internal_state['context_account_map'][context_id] = account_details
            if proxy_info: self.internal_state['context_proxy_map'][context_id] = proxy_info
            self.logger.info(f"Created new browser context (ID: {context_id}). Active: {len(self.internal_state['browser_contexts'])}")
            return context_id, context
        except Exception as e:
            self.internal_state['context_semaphore'].release() # Release semaphore on failure
            self.logger.error(f"Failed to create new browser context: {e}", exc_info=True)
            if context: await self._close_context(context_id, context) # Attempt cleanup
            raise

    def _generate_random_geo(self) -> Dict:
        # Generate plausible geolocations (e.g., within major cities)
        locations = [
            {'latitude': 40.7128, 'longitude': -74.0060, 'accuracy': random.uniform(30, 150)}, # NYC
            {'latitude': 34.0522, 'longitude': -118.2437, 'accuracy': random.uniform(30, 150)}, # LA
            {'latitude': 51.5074, 'longitude': -0.1278, 'accuracy': random.uniform(30, 150)}, # London
            {'latitude': 48.8566, 'longitude': 2.3522, 'accuracy': random.uniform(30, 150)}, # Paris
        ]
        return random.choice(locations)

    async def _close_context(self, context_id: str, context: Optional[BrowserContext] = None):
        # (Implementation largely unchanged from v3.2, ensures semaphore release)
        context_to_close = context or self.internal_state['browser_contexts'].get(context_id)
        semaphore_released = False
        if context_to_close:
            try:
                pages_to_remove = [pid for pid, pdata in self.internal_state['active_pages'].items() if pdata['context_id'] == context_id]
                for pid in pages_to_remove:
                    page_obj = self.internal_state['active_pages'].pop(pid, {}).get('page')
                    if page_obj and not page_obj.is_closed():
                        try: await page_obj.close(run_before_unload=True)
                        except Exception as page_close_err: logger.warning(f"Error closing page {pid}: {page_close_err}")

                if not context_to_close.is_closed(): # Check if already closed
                    await context_to_close.close()
                    self.logger.info(f"Closed browser context (ID: {context_id}).")
            except Exception as e: self.logger.warning(f"Error closing context {context_id}: {e}")
            finally:
                self.internal_state['browser_contexts'].pop(context_id, None)
                self.internal_state['context_account_map'].pop(context_id, None)
                self.internal_state['context_proxy_map'].pop(context_id, None)
                try:
                    # Check semaphore count before releasing
                    # Note: This check isn't perfectly thread-safe but reduces ValueErrors
                    if self.internal_state['context_semaphore']._value < self.internal_state['max_concurrent_contexts']:
                         self.internal_state['context_semaphore'].release()
                         semaphore_released = True
                except ValueError: pass # Already released or race condition
                except AttributeError: pass # Semaphore might not be initialized if error occurred early
                self.logger.debug(f"Context cleanup complete for {context_id}. Active: {len(self.internal_state['browser_contexts'])}. Semaphore released: {semaphore_released}")
        else:
             # If context object wasn't found but ID exists in map, still try to clean up maps and release semaphore
             maps_cleaned = False
             if context_id in self.internal_state['browser_contexts']: self.internal_state['browser_contexts'].pop(context_id, None); maps_cleaned = True
             if context_id in self.internal_state['context_account_map']: self.internal_state['context_account_map'].pop(context_id, None); maps_cleaned = True
             if context_id in self.internal_state['context_proxy_map']: self.internal_state['context_proxy_map'].pop(context_id, None); maps_cleaned = True

             if maps_cleaned:
                 try:
                     if self.internal_state['context_semaphore']._value < self.internal_state['max_concurrent_contexts']:
                          self.internal_state['context_semaphore'].release()
                          semaphore_released = True
                 except ValueError: pass
                 except AttributeError: pass
                 self.logger.warning(f"Context object not found for ID {context_id}, but cleaned maps. Semaphore released: {semaphore_released}")


    async def _get_page_for_context(self, context_id: str) -> Tuple[str, Page]:
        # (Implementation largely unchanged from v3.2)
        context = self.internal_state['browser_contexts'].get(context_id)
        if not context: raise ValueError(f"Context ID {context_id} not found.")
        try:
            if context.is_closed(): raise PlaywrightError(f"Context {context_id} is already closed.")
        except AttributeError: # Handle case where context object might be None unexpectedly
             raise ValueError(f"Context ID {context_id} is invalid or None.")

        existing_page = next((pdata['page'] for pid, pdata in self.internal_state['active_pages'].items() if pdata['context_id'] == context_id and not pdata['page'].is_closed()), None)
        if existing_page:
             page_id = next(pid for pid, pdata in self.internal_state['active_pages'].items() if pdata['page'] == existing_page)
             self.logger.debug(f"Reusing existing page {page_id} for context {context_id}")
             return page_id, existing_page

        page_id = f"page_{uuid.uuid4().hex[:8]}"
        page = await context.new_page()
        page.set_default_timeout(self.internal_state['default_timeout_ms'])
        self.internal_state['active_pages'][page_id] = {'page': page, 'context_id': context_id}
        self.logger.info(f"Created new page (ID: {page_id}) within context {context_id}.")
        return page_id, page

    async def _close_browser_and_playwright(self):
        # (Implementation largely unchanged from v3.2)
        self.logger.info("Closing all browser contexts and stopping Playwright...")
        context_ids = list(self.internal_state['browser_contexts'].keys())
        if context_ids: await asyncio.gather(*(self._close_context(cid) for cid in context_ids), return_exceptions=True)

        browser = self.internal_state.get('browser_instance')
        if browser and browser.is_connected():
            try: await browser.close(); self.logger.info("Persistent browser closed.")
            except Exception as e: self.logger.error(f"Error closing browser: {e}", exc_info=True)
        self.internal_state['browser_instance'] = None

        playwright_instance = self.internal_state.get('playwright_instance')
        if playwright_instance:
            try: await playwright_instance.stop(); self.logger.info("Playwright instance stopped.")
            except Exception as e: self.logger.error(f"Error stopping Playwright: {e}", exc_info=True)
        self.internal_state['playwright_instance'] = None
        self.internal_state['active_pages'] = {} # Clear active pages map

    # --- Proxy Management ---
    def _update_proxy_stats(self, context_id: Optional[str], success: bool):
        # (Implementation unchanged from v3.2)
        if not context_id: return
        proxy_info = self.internal_state['context_proxy_map'].get(context_id)
        if not proxy_info or not proxy_info.get('server'): return
        proxy_url = proxy_info['server']
        now = datetime.now(timezone.utc)
        if proxy_url not in self.internal_state['proxy_stats']: self.internal_state['proxy_stats'][proxy_url] = {'success': 0, 'failure': 0, 'last_used': now}
        if success: self.internal_state['proxy_stats'][proxy_url]['success'] += 1
        else: self.internal_state['proxy_stats'][proxy_url]['failure'] += 1
        self.internal_state['proxy_stats'][proxy_url]['last_used'] = now
        # Optional: Add logic to report consistently failing proxies immediately
        # total_attempts = self.internal_state['proxy_stats'][proxy_url]['success'] + self.internal_state['proxy_stats'][proxy_url]['failure']
        # if total_attempts > 10 and self.internal_state['proxy_stats'][proxy_url]['failure'] / total_attempts > 0.7:
        #     asyncio.create_task(self.orchestrator.report_proxy_issue(proxy_url, "High failure rate"))


    # --- Core Abstract Method Implementations ---

    async def plan_task(self, task_details: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Generates a plan for browsing tasks. Complex plans should come from ThinkTool.
        This agent focuses on executing steps, including autonomous decisions within steps.
        """
        # Planning is kept simple here; complex sequences are assumed to be planned by ThinkTool
        # and passed as a single 'web_ui_automate' or multi-step task.
        action = task_details.get('action')
        plan = []
        await self._internal_think(f"Received task: {action}. Planning execution.", details=task_details)

        # Simple tasks can have basic plans generated here
        if action == 'scrape_website':
            url = task_details.get('url')
            if not url: raise ValueError("Missing 'url' for scrape_website task.")
            plan.append({"step": 1, "action": "open_page", "tool": "browser", "params": {"url": url}})
            # Extraction logic decided within execute_step based on params
            plan.append({"step": 2, "action": "extract_data", "tool": "browser", "params": task_details}) # Pass all details
            plan.append({"step": 3, "action": "close_context", "tool": "browser", "params": {}})

        elif action == 'google_dork_scan':
            target = task_details.get('target')
            if not target: raise ValueError("Missing target for google_dork_scan")
            dork_types = task_details.get('dork_types', ['login', 'password', 'config_files', 'api_keys'])
            plan.append({"step": 1, "action": "generate_dorks", "tool": "internal", "params": {"target": target, "types": dork_types}})
            plan.append({"step": 2, "action": "execute_dork_searches", "tool": "browser", "params": {}})
            plan.append({"step": 3, "action": "analyze_dork_results", "tool": "browser", "params": {}})
            plan.append({"step": 4, "action": "close_context", "tool": "browser", "params": {}})

        # Complex UI automation is treated as a single execution step for this agent
        elif action == 'web_ui_automate':
             goal = task_details.get('goal', 'Unknown UI Goal')
             service = task_details.get('service', 'Unknown Service')
             plan.append({"step": 1, "action": f"Execute UI Automation: {service} - {goal}", "tool": "browser", "params": task_details})
             # Cleanup might be handled within the automation or as a separate step if needed
             plan.append({"step": 2, "action": "close_context", "tool": "browser", "params": {"reason": "UI Automation Complete"}})

        else:
            # Assume other actions are single steps handled by execute_step
            self.logger.info(f"No specific plan generated for action '{action}'. Will attempt direct execution.")
            plan.append({"step": 1, "action": action, "tool": "browser", "params": task_details}) # Default to browser tool
            plan.append({"step": 2, "action": "close_context", "tool": "browser", "params": {"reason": f"Task '{action}' Complete"}})

        self.logger.info(f"Generated plan with {len(plan)} steps for action '{action}'.")
        return plan

    @retry(stop=stop_after_attempt(4), # Increase retries slightly
           wait=wait_exponential(multiplier=1.5, min=5, max=90), # Longer max wait
           retry=retry_if_exception_type((PlaywrightTimeoutError, PlaywrightError)), # Retry on more Playwright errors
           reraise=True) # Reraise the exception after final attempt fails
    async def execute_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a single step, handling account selection/creation and visual UI automation."""
        step_action = step.get('action')
        tool = step.get('tool')
        params = step.get('params', {}).copy() # Base parameters for the step
        step_num = step.get('step', '?')
        result = {"status": "failure", "message": f"Step action '{step_action}' not implemented or failed initialization."}

        context_id = task_context.get("current_context_id")
        page_id = task_context.get("current_page_id")
        context: Optional[BrowserContext] = None
        page: Optional[Page] = None

        try:
            # --- Ensure Context and Page ---
            context, page, context_id, page_id = await self._ensure_context_and_page(step, task_context)
            task_context["current_context_id"] = context_id # Update task context
            task_context["current_page_id"] = page_id

            proxy_url_used = self.internal_state['context_proxy_map'].get(context_id, {}).get('server')
            await self._internal_think(f"Executing step {step_num}: {step_action} in context {context_id}", details=params)

            # --- Action Execution ---
            if tool == 'browser':
                # --- Navigation ---
                if step_action == 'open_page':
                    url = params.get('url')
                    if not url: raise ValueError("Missing 'url' parameter.")
                    self.logger.info(f"Navigating page {page_id} (Context: {context_id}) to URL: {url}")
                    response = await page.goto(url, wait_until='domcontentloaded', timeout=self.internal_state['default_timeout_ms'])
                    status_code = response.status if response else None
                    self.logger.info(f"Page {page_id} navigated to {url}. Status: {status_code}")
                    if not response or not response.ok:
                        page_content = await page.content()
                        raise PlaywrightError(f"Failed to load page {url}. Status: {status_code}. Content: {page_content[:500]}...")
                    await self._human_like_delay(page)
                    result = {"status": "success", "message": f"Page opened: {url}", "page_id": page_id, "context_id": context_id, "status_code": status_code}

                # --- Data Extraction (Consolidated) ---
                elif step_action == 'extract_data':
                    selectors = params.get('selectors')
                    extraction_prompt = params.get('extraction_prompt')
                    extract_main = params.get('extract_main_content', False)
                    output_var = params.get('output_var') # Check if output needs storing

                    extracted_data = {}
                    if selectors and isinstance(selectors, dict):
                        self.logger.info(f"Extracting data using selectors on page {page_id}.")
                        for key, selector in selectors.items():
                            try:
                                element = page.locator(selector).first
                                content = await element.text_content(timeout=15000) or \
                                          await element.inner_text(timeout=15000) or \
                                          await element.input_value(timeout=15000) or \
                                          await element.get_attribute('value', timeout=15000) or \
                                          await element.get_attribute('href', timeout=15000) or \
                                          await element.get_attribute('src', timeout=15000) or \
                                          await element.get_attribute('aria-label', timeout=15000) or ""
                                extracted_data[key] = content.strip()
                            except Exception as e: self.logger.warning(f"Selector '{selector}' for key '{key}' failed: {e}"); extracted_data[key] = None
                        result = {"status": "success", "message": "Data extracted via selectors.", "extracted_data": extracted_data}

                    elif extraction_prompt:
                        self.logger.info(f"Extracting data using Visual LLM analysis on page {page_id}.")
                        screenshot_bytes = await page.screenshot()
                        page_text_content = await page.evaluate("document.body.innerText")
                        llm_task_context = {
                            "task": "Extract structured data from webpage using visual and text context",
                            "webpage_text_content": page_text_content[:6000],
                            "extraction_instructions": extraction_prompt,
                            "current_url": page.url,
                            "desired_output_format": "JSON containing the extracted data according to instructions."
                        }
                        llm_prompt = await self.generate_dynamic_prompt(llm_task_context)
                        # --- MODIFIED: Removed hardcoded model_preference ---
                        llm_response_str = await self.orchestrator.call_llm(
                            agent_name=self.AGENT_NAME, prompt=llm_prompt, temperature=0.1,
                            max_tokens=2000, is_json_output=True,
                            image_data=screenshot_bytes # Let orchestrator pick model based on image_data presence
                        )
                        # --- END MODIFICATION ---
                        if llm_response_str:
                             try:
                                 parsed_data = self._parse_llm_json(llm_response_str)
                                 if parsed_data is not None:
                                     extracted_data = parsed_data # Store parsed data
                                     result = {"status": "success", "message": "Data extracted via Visual LLM.", "extracted_data": extracted_data}
                                 else: raise ValueError("LLM response was not valid JSON.")
                             except Exception as parse_err: result = {"status": "failure", "message": f"LLM returned invalid JSON or failed parsing: {parse_err}", "raw_output": llm_response_str}
                        else: result = {"status": "failure", "message": "Visual LLM analysis returned no response."}

                    elif extract_main:
                        self.logger.info(f"Extracting main text content from page {page_id}.")
                        try:
                            body_text = await page.evaluate("document.body.innerText")
                            extracted_data = {"main_text": body_text.strip() if body_text else ""}
                            result = {"status": "success", "message": "Main content extracted.", "extracted_data": extracted_data}
                        except Exception as text_err: result = {"status": "failure", "message": f"Error extracting main text: {text_err}"}
                    else:
                        result = {"status": "warning", "message": "No extraction method specified (selectors, prompt, or main_content)."}

                    # Store extracted data in task_context if output_var is specified
                    if output_var and result.get("status") == "success":
                        task_context[output_var] = extracted_data
                        self.logger.info(f"Stored extracted data in task context variable: '{output_var}'")
                        result["result_data"] = extracted_data # Also return in step result

                # --- Visual UI Automation (Primary Interaction Method) ---
                elif step_action.startswith('Execute UI Automation') or step_action == 'web_ui_automate':
                    service = params.get('service', 'Unknown Service')
                    goal = params.get('goal', 'Perform UI actions')
                    # --- MODIFICATION START: Handle input_vars ---
                    # Get base parameters specific to this automation goal
                    automate_params = params.get('params', {}).copy()
                    # Get input variables defined in the step plan
                    input_vars_map = params.get('input_vars', {})
                    # Resolve input variables using the main task_context
                    resolved_input_vars = {}
                    for param_name, context_var_name in input_vars_map.items():
                        if isinstance(context_var_name, str) and context_var_name.startswith('$'):
                            var_key = context_var_name[1:] # Remove '$'
                            if var_key in task_context:
                                resolved_input_vars[param_name] = task_context[var_key]
                                self.logger.debug(f"Resolved input var '{param_name}' from task context key '{var_key}'")
                            else:
                                self.logger.warning(f"Input variable '{context_var_name}' requested but not found in task context.")
                                resolved_input_vars[param_name] = None # Or raise error?
                        else:
                             # If not starting with '$', treat as literal value? Or ignore? Let's ignore for now.
                             self.logger.warning(f"Input variable value '{context_var_name}' for '{param_name}' does not start with '$'. Treating as literal or ignoring.")
                             # resolved_input_vars[param_name] = context_var_name # Uncomment to treat as literal
                    # Merge resolved variables into the parameters used by the automation loop
                    automate_params.update(resolved_input_vars)
                    # --- MODIFICATION END ---

                    max_steps = params.get('max_steps', 25)
                    current_step_count = 0
                    last_action_summary = "Automation started."

                    if not goal: raise ValueError("Missing 'goal' for web_ui_automate")
                    self.logger.info(f"Starting Visual UI Automation: Service='{service}', Goal='{goal}' on page {page_id}")

                    while current_step_count < max_steps:
                        current_step_count += 1
                        await self._internal_think(f"UI Automate Step {current_step_count}/{max_steps}: Goal='{goal}'", details={"last_action_result": last_action_summary})
                        await self._human_like_delay(page, short=True)

                        screenshot_bytes = await page.screenshot(full_page=True)
                        page_text_content = await page.evaluate("document.body.innerText")
                        current_url = page.url

                        automation_task_context = {
                            "task": "Determine next UI action based primarily on visual context",
                            "service_context": service, "current_goal": goal,
                            "automation_parameters": automate_params, # Use merged params
                            "page_text_content_snippet": page_text_content[:5000],
                            "current_url": current_url, "last_action_result": last_action_summary,
                            "current_step": f"{current_step_count}/{max_steps}",
                            "available_actions": ["click", "input", "upload", "scroll", "wait", "navigate", "download", "solve_captcha", "finish", "error"],
                            "desired_output_format": "JSON: {\"action_type\": \"<chosen_action>\", \"selector\": \"<css_selector_or_xpath>\", \"coordinates\": {\"x\": float, \"y\": float}?, \"text_to_input\": \"<string>\"?, \"file_path_param\": \"<param_name_for_file>\"?, \"scroll_direction\": \"up|down|left|right|top|bottom\"?, \"wait_time_ms\": int?, \"target_url\": \"<string>\"?, \"captcha_type\": \"checkbox|image_challenge|etc\"?, \"reasoning\": \"<Explanation>\", \"error_message\": \"<If action_type is error>\"}"
                        }
                        if "captcha" in goal.lower() or "captcha" in last_action_summary.lower():
                             automation_task_context["special_focus"] = "A CAPTCHA might be present. Identify it and determine the interaction needed (e.g., click checkbox, describe image challenge)."

                        llm_prompt = await self.generate_dynamic_prompt(automation_task_context)
                        # --- MODIFIED: Removed hardcoded model_preference ---
                        action_json_str = await self.orchestrator.call_llm(
                            agent_name=self.AGENT_NAME, prompt=llm_prompt, temperature=0.05,
                            max_tokens=800, is_json_output=True,
                            image_data=screenshot_bytes # Let orchestrator pick model
                        )
                        # --- END MODIFICATION ---

                        if not action_json_str:
                            last_action_summary = "LLM failed to determine next action."
                            self.logger.error(last_action_summary)
                            if current_step_count == max_steps: raise RuntimeError(f"UI Automation failed: LLM unresponsive after {max_steps} steps.")
                            continue

                        try:
                            action_data = self._parse_llm_json(action_json_str)
                            if not action_data: raise ValueError("Failed to parse LLM action JSON.")

                            action_type = action_data.get('action_type')
                            reasoning = action_data.get('reasoning', 'N/A')
                            self.logger.info(f"Attempt {current_step_count}: LLM decided action: {action_type}. Reasoning: {reasoning}")

                            action_executed_successfully = False
                            action_result_message = f"Action '{action_type}' initiated."
                            step_output_data = None # To store data produced by this micro-step

                            if action_type == 'click':
                                selector = action_data.get('selector'); coords = action_data.get('coordinates')
                                target_element = await self._find_element(page, selector, coords)
                                if target_element: await self._human_click(page, target_element); action_executed_successfully = True; action_result_message = f"Clicked element matching '{selector or coords}'."
                                else: action_result_message = f"Click failed: Element not found for '{selector or coords}'."

                            elif action_type == 'input':
                                selector = action_data.get('selector'); coords = action_data.get('coordinates')
                                text_to_input_raw = action_data.get('text_to_input')
                                text_to_input = None
                                # Resolve input text from automate_params if it's a key reference
                                if isinstance(text_to_input_raw, str) and text_to_input_raw.startswith("params."):
                                    key = text_to_input_raw.split('.')[-1]
                                    text_to_input = automate_params.get(key)
                                    if text_to_input is None: action_result_message = f"Input failed: Parameter key '{key}' not found in resolved params.";
                                    else: self.logger.debug(f"Resolved input text from params key '{key}'")
                                elif text_to_input_raw is not None:
                                    text_to_input = text_to_input_raw # Use literal text
                                else: action_result_message = "Input failed: LLM did not provide text or valid param key."

                                target_element = await self._find_element(page, selector, coords)
                                if target_element and text_to_input is not None:
                                    await self._human_fill(page, target_element, str(text_to_input)) # Ensure text is string
                                    action_executed_successfully = True; action_result_message = f"Input text into element '{selector or coords}'."
                                elif not target_element: action_result_message = f"Input failed: Element not found for '{selector or coords}'."
                                # else: message already set above

                            elif action_type == 'upload':
                                selector = action_data.get('selector'); coords = action_data.get('coordinates')
                                file_path_param_name = action_data.get('file_path_param')
                                file_path = automate_params.get(file_path_param_name) if file_path_param_name else None # Get resolved path from merged params
                                target_element = await self._find_element(page, selector, coords)
                                if target_element and file_path and os.path.exists(str(file_path)): # Check existence
                                    async with page.expect_file_chooser() as fc_info:
                                         await self._human_click(page, target_element, timeout=10000)
                                    file_chooser = await fc_info.value
                                    await file_chooser.set_files(str(file_path)) # Ensure path is string
                                    action_executed_successfully = True; action_result_message = f"Uploaded file '{os.path.basename(str(file_path))}' to element '{selector or coords}'."
                                elif not file_path or not os.path.exists(str(file_path)):
                                    action_result_message = f"Upload failed: File path '{file_path}' invalid or not found (resolved from param '{file_path_param_name}')."
                                else: action_result_message = f"Upload failed: Element not found for '{selector or coords}'."

                            elif action_type == 'scroll':
                                direction = action_data.get('scroll_direction', 'down')
                                selector = action_data.get('selector')
                                target_element = await self._find_element(page, selector) if selector else None
                                await self._human_scroll(page, direction, target_element)
                                action_executed_successfully = True; action_result_message = f"Scrolled {direction}."

                            elif action_type == 'wait':
                                wait_ms = action_data.get('wait_time_ms', random.randint(1500, 4000))
                                await page.wait_for_timeout(wait_ms)
                                action_executed_successfully = True; action_result_message = f"Waited for {wait_ms} ms."

                            elif action_type == 'navigate':
                                target_url = action_data.get('target_url')
                                if target_url:
                                    self.logger.info(f"Navigating to URL specified by LLM: {target_url}")
                                    response = await page.goto(target_url, wait_until='domcontentloaded')
                                    if response and response.ok: action_executed_successfully = True; action_result_message = f"Navigated to {target_url}."
                                    else: action_result_message = f"Navigation failed: Could not load {target_url} (Status: {response.status if response else 'N/A'})."
                                else: action_result_message = "Navigation failed: LLM did not provide target_url."

                            elif action_type == 'download':
                                selector = action_data.get('selector'); coords = action_data.get('coordinates')
                                target_element = await self._find_element(page, selector, coords)
                                if target_element:
                                    async with page.expect_download(timeout=180000) as download_info:
                                        await self._human_click(page, target_element)
                                    download = await download_info.value
                                    download_filename = f"{uuid.uuid4().hex}_{download.suggested_filename}"
                                    download_path = os.path.join(self.temp_dir, download_filename)
                                    await download.save_as(download_path)
                                    self.logger.info(f"File downloaded to: {download_path}")
                                    step_output_data = download_path # Store path as output of this micro-step
                                    action_executed_successfully = True; action_result_message = f"File downloaded to {download_path}"
                                else: action_result_message = f"Download failed: Element not found for '{selector or coords}'."

                            elif action_type == 'solve_captcha':
                                captcha_type = action_data.get('captcha_type', 'unknown')
                                selector = action_data.get('selector'); coords = action_data.get('coordinates')
                                self.logger.info(f"Attempting to solve CAPTCHA (Type: {captcha_type})")
                                target_element = await self._find_element(page, selector, coords)
                                if target_element:
                                    if captcha_type == 'checkbox':
                                        await self._human_click(page, target_element)
                                        await page.wait_for_timeout(random.uniform(3000, 6000))
                                        action_executed_successfully = True; action_result_message = "Clicked CAPTCHA checkbox."
                                    elif captcha_type == 'image_challenge':
                                        action_result_message = f"Image CAPTCHA solving not fully implemented for element '{selector or coords}'. Requires specific LLM interaction."
                                        self.logger.warning(action_result_message)
                                    else: action_result_message = f"CAPTCHA solving failed: Unknown or unsupported type '{captcha_type}'."
                                else: action_result_message = f"CAPTCHA solving failed: Element not found for '{selector or coords}'."

                            elif action_type == 'finish':
                                self.logger.info(f"Visual UI Automation goal '{goal}' completed successfully based on LLM decision.")
                                # Check if the original step requested an output variable
                                output_var_name = params.get("output_var")
                                if output_var_name:
                                     # Try to get the final result (e.g., URL, confirmation text) - needs refinement
                                     final_output_value = page.url # Default to final URL
                                     self.logger.info(f"Storing final automation output '{final_output_value}' in task context variable: '{output_var_name}'")
                                     task_context[output_var_name] = final_output_value
                                     result["result_data"] = final_output_value # Include in step result
                                result["status"] = "success"; result["message"] = f"UI Automation goal '{goal}' completed."
                                action_executed_successfully = True
                                break # Exit the automation loop

                            elif action_type == 'error':
                                error_msg = action_data.get('error_message', 'LLM indicated an unrecoverable error.')
                                self.logger.error(f"LLM reported error during UI automation: {error_msg}")
                                raise RuntimeError(f"UI Automation failed: {error_msg}")

                            else:
                                action_result_message = f"Unsupported action type from LLM: {action_type}"
                                self.logger.warning(action_result_message)

                            # --- Store micro-step output if needed ---
                            # If the LLM's action produced data AND the original step defined an output_var
                            # (This is primarily handled by 'download' and 'finish' currently)
                            # output_var_name = params.get("output_var")
                            # if output_var_name and step_output_data is not None:
                            #     task_context[output_var_name] = step_output_data
                            #     self.logger.info(f"Stored step output in task context variable: '{output_var_name}'")

                            # --- Update loop status ---
                            last_action_summary = action_result_message
                            if not action_executed_successfully:
                                self.logger.warning(f"Attempt {current_step_count}: Action '{action_type}' failed or was not applicable. Message: {action_result_message}")
                                if current_step_count == max_steps:
                                     raise RuntimeError(f"UI Automation failed after {max_steps} steps. Last failed action: {action_type}. Message: {last_action_summary}")

                        except PlaywrightError as pe_auto:
                            last_action_summary = f"Playwright error executing action '{action_data.get('action_type', 'unknown')}': {pe_auto}"
                            self.logger.error(f"Attempt {current_step_count}: {last_action_summary}", exc_info=False)
                            if current_step_count == max_steps: raise RuntimeError(f"UI Automation failed after {max_steps} steps due to Playwright error: {pe_auto}") from pe_auto
                        except Exception as auto_err:
                            last_action_summary = f"Unexpected error executing action '{action_data.get('action_type', 'unknown')}': {auto_err}"
                            self.logger.error(f"Attempt {current_step_count}: {last_action_summary}", exc_info=True)
                            if current_step_count == max_steps: raise RuntimeError(f"UI Automation failed after {max_steps} steps due to unexpected error: {auto_err}") from auto_err

                    # After loop finishes (either by 'finish' or max_steps)
                    if result.get("status") != "success": # If loop ended due to max steps without 'finish'
                         self.logger.warning(f"UI Automation loop finished after {max_steps} steps without explicit 'finish' action. Goal '{goal}' might be incomplete.")
                         result = {"status": "warning", "message": f"Automation finished due to max steps ({max_steps}). Goal '{goal}' may be incomplete. Last action result: {last_action_summary}"}

                # --- Google Dorking Actions ---
                elif tool == 'internal' and step_action == 'generate_dorks':
                    target = params.get('target')
                    dork_types = params.get('types', [])
                    if not target: raise ValueError("Missing target for dork generation.")
                    dorks = self._generate_google_dorks(target, dork_types) # Use enhanced helper
                    task_context['generated_dorks'] = dorks
                    result = {"status": "success", "message": f"Generated {len(dorks)} dorks.", "dorks": dorks}

                elif tool == 'browser' and step_action == 'execute_dork_searches':
                    dorks = task_context.get('generated_dorks')
                    if not dorks: raise ValueError("No dorks generated in previous step.")
                    search_results = {}
                    self.logger.info(f"Executing {len(dorks)} Google Dorks...")
                    for i, dork in enumerate(dorks):
                        self.logger.info(f"Searching Dork ({i+1}/{len(dorks)}): {dork}")
                        search_url = f"https://www.google.com/search?q={quote_plus(dork)}&num=20&hl=en&filter=0" # Added filter=0
                        try:
                            await page.goto(search_url, wait_until='domcontentloaded', timeout=35000)
                            await self._human_like_delay(page) # Human delay after load

                            # Check for CAPTCHA or block page visually
                            screenshot_bytes = await page.screenshot()
                            llm_check_context = {"task": "Check for Google search block/CAPTCHA", "current_url": page.url, "desired_output_format": "JSON: {\"is_blocked\": boolean, \"reason\": \"<description if blocked>\"}"}
                            llm_prompt = await self.generate_dynamic_prompt(llm_check_context)
                            # --- MODIFIED: Removed hardcoded model_preference ---
                            check_response = await self.orchestrator.call_llm(
                                agent_name=self.AGENT_NAME, prompt=llm_prompt, image_data=screenshot_bytes,
                                max_tokens=100, is_json_output=True, temperature=0.1
                            )
                            # --- END MODIFICATION ---
                            block_check = self._parse_llm_json(check_response)

                            if block_check and block_check.get("is_blocked"):
                                reason = block_check.get("reason", "Unknown block")
                                self.logger.warning(f"Google search blocked for dork: {dork}. Reason: {reason}. Skipping remaining dorks for this context.")
                                self._update_proxy_stats(context_id, False); break # Stop dorking in this context

                            # Extract results (more robustly)
                            links = await page.locator('//div[contains(@class, "g ")]//a[@href and @ping]').all() # More specific XPath
                            valid_links = []
                            for link_element in links:
                                href = await link_element.get_attribute('href')
                                if href and href.startswith('http') and not re.match(r'https?://(www\.)?google\.com/|https?://accounts\.google\.com/', href):
                                    # Clean tracking parameters
                                    parsed_url = urlparse(href)
                                    cleaned_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                                    if parsed_url.query: cleaned_url += f"?{parsed_url.query}" # Keep query for now, might need cleaning too
                                    valid_links.append(cleaned_url)

                            search_results[dork] = list(dict.fromkeys(valid_links))[:15] # Unique links, limit 15
                            self.logger.info(f"Dork '{dork}' found {len(search_results[dork])} unique potential links.")
                            self._update_proxy_stats(context_id, True)

                        except PlaywrightTimeoutError: self.logger.warning(f"Timeout searching Google for dork: {dork}"); self._update_proxy_stats(context_id, False)
                        except PlaywrightError as pe_dork: self.logger.error(f"Playwright error searching Google for dork '{dork}': {pe_dork}"); self._update_proxy_stats(context_id, False)
                        except Exception as search_err: self.logger.error(f"Unexpected error searching Google for dork '{dork}': {search_err}", exc_info=True); self._update_proxy_stats(context_id, False)
                        # Longer, variable delay between searches
                        await asyncio.sleep(random.uniform(8, 20))

                    task_context['dork_search_results'] = search_results
                    result = {"status": "success", "message": f"Executed {len(dorks)} dork searches.", "results_summary": {k: len(v) for k, v in search_results.items()}}

                elif tool == 'browser' and step_action == 'analyze_dork_results':
                    search_results = task_context.get('dork_search_results')
                    if not search_results: raise ValueError("No dork search results found to analyze.")
                    potential_findings = [] # Store detailed findings
                    urls_to_analyze = list(set(url for urls in search_results.values() for url in urls))
                    max_urls_to_analyze = self.config.get("DORK_MAX_URLS_TO_ANALYZE", 50)
                    self.logger.info(f"Analyzing up to {max_urls_to_analyze} unique URLs from dork results ({len(urls_to_analyze)} total found)...")

                    for i, url in enumerate(urls_to_analyze[:max_urls_to_analyze]):
                        await self._internal_think(f"Analyzing Dork Result URL ({i+1}/{min(len(urls_to_analyze), max_urls_to_analyze)}): {url}")
                        try:
                            await page.goto(url, wait_until='domcontentloaded', timeout=45000) # Longer timeout for potentially slow pages
                            await self._human_like_delay(page, short=True)

                            # Use visual LLM to analyze screenshot + text for credentials/sensitive info
                            screenshot_bytes = await page.screenshot(full_page=True)
                            page_text = await page.evaluate("document.body.innerText")
                            if not page_text or len(page_text) < 30: self.logger.debug(f"Skipping short/empty page: {url}"); continue

                            analysis_task_context = {
                                "task": "Analyze webpage screenshot and text for potential credentials, API keys, secrets, sensitive config, or login forms",
                                "text_content_snippet": page_text[:10000], # Generous snippet
                                "target_url": url,
                                "desired_output_format": "JSON list of findings: [{\"type\": \"password|api_key|secret|db_conn|config_file|login_form|ssh_key|other_sensitive\", \"value_snippet\": \"<relevant text snippet>\", \"context_snippet\": \"<surrounding text>\", \"location_hint\": \"<e.g., form input, code block, plain text>\"}] or empty list []."
                            }
                            llm_prompt = await self.generate_dynamic_prompt(analysis_task_context)
                            # --- MODIFIED: Removed hardcoded model_preference ---
                            findings_json_str = await self.orchestrator.call_llm(
                                agent_name=self.AGENT_NAME, prompt=llm_prompt, temperature=0.1,
                                max_tokens=1500, is_json_output=True,
                                image_data=screenshot_bytes # Let orchestrator pick model
                            )
                            # --- END MODIFICATION ---

                            if findings_json_str:
                                findings_found = self._parse_llm_json(findings_json_str, expect_type=list)
                                if findings_found:
                                    for finding in findings_found: finding['source_url'] = url # Add source URL
                                    potential_findings.extend(findings_found)
                                    self.logger.info(f"Found {len(findings_found)} potential items on {url}")
                                    # Log immediately to KB for high-priority findings
                                    if any(f['type'] in ['password', 'api_key', 'secret', 'ssh_key'] for f in findings_found):
                                         await self.log_knowledge_fragment(
                                             agent_source=self.AGENT_NAME, data_type="dorking_credential_finding",
                                             content={"findings": findings_found, "source_url": url},
                                             tags=["dorking", "credentials", "security_risk", "high_priority", urlparse(url).netloc],
                                             relevance_score=0.95
                                         )
                        except PlaywrightTimeoutError: self.logger.warning(f"Timeout visiting dork result URL {url}")
                        except PlaywrightError as pe_analyze: self.logger.warning(f"Playwright error analyzing dork result URL {url}: {pe_analyze}")
                        except Exception as page_err: self.logger.warning(f"Failed to analyze dork result URL {url}: {page_err}", exc_info=True)
                        await asyncio.sleep(random.uniform(3, 7)) # Delay between URL analyses

                    if potential_findings:
                        # Log summary and notify
                        await self.log_operation('critical', f"Google Dorking Analysis Complete. Found {len(potential_findings)} total potential items across analyzed URLs. Details logged to KB.")
                        # Send notification via orchestrator
                        await self.orchestrator.send_notification(
                            title="Potential Sensitive Findings via Google Dorking",
                            message=f"Scan found {len(potential_findings)} potential items (credentials, keys, configs, etc.). Check Knowledge Base (Type: dorking_credential_finding) for details.",
                            level="critical"
                        )
                    else:
                        self.logger.info("Dork analysis complete. No high-confidence sensitive items found in analyzed URLs.")

                    result = {"status": "success", "message": f"Dork analysis complete. Found {len(potential_findings)} potential items.", "potential_findings": potential_findings}
                    self._update_proxy_stats(context_id, True)


                # --- Context Closing ---
                elif tool == 'browser' and step_action == 'close_context':
                    reason = params.get('reason', 'Step complete')
                    if context_id:
                        await self._close_context(context_id, context)
                        task_context["current_context_id"] = None # Clear from task context
                        task_context["current_page_id"] = None
                        result = {"status": "success", "message": f"Browser context closed. Reason: {reason}"}
                    else: result = {"status": "warning", "message": "No active context ID found to close."}

                # --- Fallback for unknown browser actions ---
                else:
                    result = {"status": "failure", "message": f"Unsupported browser action: {step_action}"}

            # --- Handle Non-Browser Tools (Internal, Search, LLM) ---
            elif tool == 'internal':
                 if step_action == 'generate_dorks':
                     # Logic is above, this case might not be hit if planned correctly
                     target = params.get('target'); dork_types = params.get('types', [])
                     if not target: raise ValueError("Missing target for dork generation.")
                     dorks = self._generate_google_dorks(target, dork_types)
                     task_context['generated_dorks'] = dorks
                     result = {"status": "success", "message": f"Generated {len(dorks)} dorks.", "dorks": dorks}
                 else: result = {"status": "failure", "message": f"Unknown internal action: {step_action}"}

            elif tool == 'search_engine': # Assumes orchestrator provides a direct search tool if needed
                 if step_action == 'execute_search':
                      query = params.get('query'); num_results = params.get('num_results', 5)
                      if not query: raise ValueError("Missing 'query' parameter.")
                      # Use orchestrator's tool execution method
                      search_tool_result = await self._execute_tool('search_engine_api', {'query': query, 'num_results': num_results}, step_num)
                      if search_tool_result.get('status') == 'success':
                           task_context['search_results'] = search_tool_result.get('results', [])
                           result = {"status": "success", "message": f"Search completed via API.", "result_data": task_context['search_results']}
                      else: result = search_tool_result
                 else: result = {"status": "failure", "message": f"Unknown search engine action: {step_action}"}

            elif tool == 'llm': # Direct LLM calls if needed (e.g., summarization)
                 if step_action == 'summarize_results':
                      search_results = task_context.get('search_results')
                      if not search_results: raise ValueError("No search results found in context to summarize.")
                      content_to_summarize = "\n---\n".join([f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nSnippet: {r.get('snippet', 'N/A')}" for r in search_results])
                      llm_task_context = { "task": "Summarize search results", "search_results_content": content_to_summarize[:10000], "desired_output_format": "Concise summary highlighting key findings." }
                      llm_prompt = await self.generate_dynamic_prompt(llm_task_context)
                      summary = await self.orchestrator.call_llm(agent_name=self.AGENT_NAME, prompt=llm_prompt, temperature=0.3, max_tokens=1000)
                      if summary: result = {"status": "success", "message": "Search results summarized.", "summary": summary, "result_data": summary}
                      else: result = {"status": "failure", "message": "LLM summarization failed."}
                 else: result = {"status": "failure", "message": f"Unknown LLM action: {step_action}"}

            # --- Unknown Tool ---
            else:
                result = {"status": "failure", "message": f"Unsupported tool type specified: {tool}"}

        # --- Error Handling ---
        except PlaywrightError as pe:
            self.logger.error(f"Playwright error during step {step_num} ('{step_action}'): {pe}", exc_info=False) # Less verbose logging for PE
            result = {"status": "failure", "message": f"Browser error: {pe.message}"} # Use PE message
            self._update_proxy_stats(context_id, False)
            # Capture screenshot on error if page exists
            if page and not page.is_closed():
                 try:
                     error_screenshot_path = os.path.join(self.temp_dir, f"error_{context_id}_{page_id}_{int(time.time())}.png")
                     await page.screenshot(path=error_screenshot_path, full_page=True)
                     result["error_screenshot"] = error_screenshot_path
                     self.logger.info(f"Error screenshot saved to: {error_screenshot_path}")
                 except Exception as ss_err: self.logger.error(f"Failed to capture error screenshot: {ss_err}")
            raise # Re-raise to trigger tenacity retry

        except FileNotFoundError as fnf_err:
             self.logger.error(f"File not found during step {step_num} ('{step_action}'): {fnf_err}")
             result = {"status": "failure", "message": str(fnf_err)}
             # Don't retry file not found errors

        except ValueError as ve: # Catch specific value errors (e.g., missing params)
            self.logger.error(f"Value error during step {step_num} ('{step_action}'): {ve}")
            result = {"status": "failure", "message": f"Configuration or Parameter Error: {ve}"}
            # Don't retry value errors

        except RuntimeError as rte: # Catch deliberate runtime errors (e.g., LLM error, account creation fail)
            self.logger.error(f"Runtime error during step {step_num} ('{step_action}'): {rte}")
            result = {"status": "failure", "message": f"Runtime Error: {rte}"}
            # Don't retry most runtime errors unless specifically designed for it

        except Exception as e:
            self.logger.error(f"Unexpected error during step {step_num} ('{step_action}'): {e}", exc_info=True)
            result = {"status": "failure", "message": f"Unexpected error: {e}"}
            self._update_proxy_stats(context_id, False)
            if page and not page.is_closed(): # Screenshot on unexpected errors too
                 try:
                     error_screenshot_path = os.path.join(self.temp_dir, f"unexpected_error_{context_id}_{page_id}_{int(time.time())}.png")
                     await page.screenshot(path=error_screenshot_path, full_page=True)
                     result["error_screenshot"] = error_screenshot_path
                     self.logger.info(f"Unexpected error screenshot saved to: {error_screenshot_path}")
                 except Exception as ss_err: self.logger.error(f"Failed to capture unexpected error screenshot: {ss_err}")
            await self._report_error(f"Step {step_num} ('{step_action}') failed unexpectedly: {e}", task_context.get('id'))
            # Consider if unexpected errors should be retried by tenacity (PlaywrightError is handled above)

        # --- Final Result Logging ---
        log_level = "info" if result.get("status") == "success" else "warning"
        if result.get("status") == "failure": log_level = "error"
        self.logger.log(logging.getLevelName(log_level.upper()), f"Step {step_num} ('{step_action}') completed with status: {result.get('status')}. Message: {result.get('message')}")

        return result

    async def _ensure_context_and_page(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Tuple[BrowserContext, Page, str, str]:
        """Gets or creates context/page, handling account selection/creation."""
        context_id = task_context.get("current_context_id")
        page_id = task_context.get("current_page_id")
        context: Optional[BrowserContext] = None
        page: Optional[Page] = None

        params = step.get('params', {})
        requires_account = params.get('requires_account', False) # Step indicates if account is needed
        service_needed = params.get('service') or params.get('target_platform')
        account_id_param = params.get('account_id') # Specific account requested?
        allow_creation = params.get('allow_account_creation', True) # Allow creation by default if needed

        # Check existing context
        if context_id:
            context = self.internal_state['browser_contexts'].get(context_id)
            if not context or context.is_closed():
                self.logger.warning(f"Context {context_id} was closed or invalid. Will create new context.")
                await self._close_context(context_id) # Ensure cleanup
                context_id = None; page_id = None; context = None
            else:
                # Verify if the existing context's account matches requirements
                current_account = self.internal_state['context_account_map'].get(context_id)
                if requires_account:
                    if not current_account:
                        self.logger.warning(f"Existing context {context_id} has no associated account, but step requires one. Creating new context.")
                        await self._close_context(context_id); context_id = None; page_id = None; context = None
                    elif service_needed and current_account.get('service') != service_needed:
                        self.logger.warning(f"Existing context {context_id} is for service '{current_account.get('service')}', but step requires '{service_needed}'. Creating new context.")
                        await self._close_context(context_id); context_id = None; page_id = None; context = None
                    elif account_id_param and current_account.get('id') != account_id_param:
                         self.logger.warning(f"Existing context {context_id} has account ID {current_account.get('id')}, but step requires {account_id_param}. Creating new context.")
                         await self._close_context(context_id); context_id = None; page_id = None; context = None

        # Create new context if needed
        if not context_id:
            account_details = None
            proxy_info = params.get('proxy_info') # Use proxy from step params if provided

            # Determine account needed
            if requires_account:
                if account_id_param:
                    account_details = await self._get_account_details_from_db(account_id_param)
                    if not account_details:
                        raise ValueError(f"Required account ID {account_id_param} not found in DB.")
                    self.logger.info(f"Using specific account ID {account_id_param} for service {account_details.get('service')}.")
                elif service_needed:
                    account_details = await self._select_available_account_for_service(service_needed)
                    if not account_details and allow_creation:
                        self.logger.warning(f"No available active account found for service '{service_needed}'. Attempting autonomous creation.")
                        account_details = await self._attempt_new_account_creation(service_needed) # This now returns details or raises error
                        if not account_details:
                             # _attempt_new_account_creation should raise error on failure now
                             raise RuntimeError(f"Failed to find or create an account for service: {service_needed}")
                        # If creation succeeded, account_details will be populated
                    elif not account_details and not allow_creation:
                         raise ValueError(f"No available active account found for service '{service_needed}', and creation is disallowed for this step.")
                else:
                    raise ValueError("Step requires account, but 'service' or 'account_id' parameter is missing.")

            # Get proxy if not provided in step params
            if not proxy_info and hasattr(self.orchestrator, 'get_proxy'):
                 proxy_purpose = f"context_{service_needed or 'general'}_{account_details.get('id', 'new') if account_details else 'anon'}"
                 proxy_info = await self.orchestrator.get_proxy(purpose=proxy_purpose, quality_level='high' if requires_account else 'standard')

            context_id, context = await self._get_new_context(account_details, proxy_info)

        # Get page for the context
        page_id, page = await self._get_page_for_context(context_id)

        return context, page, context_id, page_id


    # --- Human Emulation Helpers ---
    async def _human_like_delay(self, page: Page, short: bool = False):
        """Adds variable delays, mouse movements, and scrolls."""
        if short:
            await page.wait_for_timeout(random.uniform(400, 900))
            return

        try:
            # Move mouse to a random plausible location
            viewport = page.viewport_size or {'width': 1280, 'height': 720}
            await page.mouse.move(
                random.uniform(viewport['width'] * 0.1, viewport['width'] * 0.9),
                random.uniform(viewport['height'] * 0.1, viewport['height'] * 0.9),
                steps=random.randint(5, 15) # More steps for smoother movement
            )
            await page.wait_for_timeout(random.uniform(600, 1800))

            # Random scroll
            if random.random() < 0.6: # Increased scroll probability
                scroll_amount = random.randint(-350, 350)
                await page.mouse.wheel(0, scroll_amount)
                await page.wait_for_timeout(random.uniform(500, 1200))

        except Exception as delay_err:
            self.logger.warning(f"Minor error during human-like delay simulation: {delay_err}")

        # Final longer wait
        await page.wait_for_timeout(random.uniform(1800, 4500))

    async def _human_click(self, page: Page, target: Union[Locator, str], timeout: int = 30000):
        """Performs a click with human-like mouse movement and slight offset."""
        element: Optional[Locator] = None
        if isinstance(target, str):
            element = page.locator(target).first
        elif isinstance(target, Locator):
            element = target

        if not element: raise PlaywrightError(f"Cannot click, element not found or invalid: {target}")

        try:
            await element.scroll_into_view_if_needed(timeout=10000)
            await page.wait_for_timeout(random.uniform(100, 300)) # Brief pause after scroll

            bb = await element.bounding_box()
            if not bb: raise PlaywrightError("Element bounding box not available for click.")

            # Calculate a random point within the element bounds
            click_x = bb['x'] + random.uniform(bb['width'] * 0.2, bb['width'] * 0.8)
            click_y = bb['y'] + random.uniform(bb['height'] * 0.2, bb['height'] * 0.8)

            # Move mouse realistically towards the point
            await page.mouse.move(click_x, click_y, steps=random.randint(8, 20))
            await page.wait_for_timeout(random.uniform(150, 400)) # Pause before click

            # Perform the click
            await page.mouse.down()
            await page.wait_for_timeout(random.uniform(60, 150)) # Hold click briefly
            await page.mouse.up()

            await page.wait_for_timeout(random.uniform(200, 500)) # Pause after click

        except PlaywrightTimeoutError as pte:
             raise PlaywrightTimeoutError(f"Timeout during human-like click on {target}: {pte.message}") from pte
        except Exception as e:
             raise PlaywrightError(f"Error during human-like click on {target}: {e}") from e

    async def _human_fill(self, page: Page, target: Union[Locator, str], text: str, delay_ms: int = 110):
        """Fills an input field with human-like typing delay."""
        element: Optional[Locator] = None
        if isinstance(target, str):
            element = page.locator(target).first
        elif isinstance(target, Locator):
            element = target

        if not element: raise PlaywrightError(f"Cannot fill, element not found or invalid: {target}")

        try:
            await self._human_click(page, element) # Click the field first
            await page.wait_for_timeout(random.uniform(200, 500))
            # Use Playwright's fill with delay for simplicity, or type character by character
            # await element.fill(text, timeout=20000) # Simpler approach
            # More realistic typing:
            await element.press_sequentially(text, delay=random.uniform(delay_ms * 0.7, delay_ms * 1.3))
            await page.wait_for_timeout(random.uniform(300, 600)) # Pause after typing
        except Exception as e:
             raise PlaywrightError(f"Error during human-like fill on {target}: {e}") from e

    async def _human_scroll(self, page: Page, direction: str, target_element: Optional[Locator] = None):
        """Performs scrolling with mouse wheel simulation."""
        scroll_amount = random.randint(400, 900)
        if direction == "up": scroll_amount = -scroll_amount
        elif direction == "left": await page.mouse.wheel(-scroll_amount, 0); return
        elif direction == "right": await page.mouse.wheel(scroll_amount, 0); return
        elif direction == "top": await page.evaluate("window.scrollTo(0, 0)"); return
        elif direction == "bottom": await page.evaluate("window.scrollTo(0, document.body.scrollHeight)"); return
        # Default is down

        if target_element:
             # Scroll the specific element if possible (less reliable)
             try: await target_element.evaluate(f"(element) => element.scrollTop += {scroll_amount}")
             except Exception: await page.mouse.wheel(0, scroll_amount) # Fallback to page scroll
        else:
             await page.mouse.wheel(0, scroll_amount)
        await page.wait_for_timeout(random.uniform(300, 700))

    async def _find_element(self, page: Page, selector: Optional[str], coords: Optional[Dict] = None) -> Optional[Locator]:
         """Finds element by selector or coordinates (prioritizes selector)."""
         if selector:
             try:
                 element = page.locator(selector).first
                 await element.wait_for(state='visible', timeout=15000) # Wait for visibility
                 return element
             except PlaywrightTimeoutError:
                 self.logger.warning(f"Element not found or not visible for selector: {selector}")
                 return None
             except Exception as e:
                  self.logger.error(f"Error finding element with selector '{selector}': {e}")
                  return None
         elif coords and 'x' in coords and 'y' in coords:
             # Finding by coordinates is less reliable, best used for clicking, not getting Locator object easily
             # This might need refinement or direct coordinate clicking
             self.logger.warning("Finding element purely by coordinates is not directly supported for returning Locator. Use coordinates directly in click actions.")
             return None # Cannot reliably return a Locator from coordinates alone
         return None

    # --- Account Management ---
    async def _generate_fake_identity(self) -> Optional[Dict]:
        """Generates realistic fake identity details including a temp email."""
        try:
            first_name = self.faker.first_name()
            last_name = self.faker.last_name()
            username_base = f"{first_name.lower()}_{last_name.lower()}{random.randint(10, 999)}"
            password = self.faker.password(length=random.randint(12, 16), special_chars=True, digits=True, upper_case=True, lower_case=True)

            # Get temporary email address
            email = await self.temp_mail_service.get_new_email_address()
            if not email:
                self.logger.error("Failed to generate temporary email address for fake identity.")
                return None

            identity = {
                "first_name": first_name,
                "last_name": last_name,
                "full_name": f"{first_name} {last_name}",
                "email": email,
                "username": username_base, # Service might require email as username
                "password": password,
                # Add more fields as needed (e.g., address, phone - requires more complex generation/validation)
                # "address": self.faker.address(),
                # "city": self.faker.city(),
                # "zipcode": self.faker.zipcode(),
            }
            self.logger.info(f"Generated fake identity with email: {email}")
            return identity
        except Exception as e:
            self.logger.error(f"Error generating fake identity: {e}", exc_info=True)
            return None

    async def _attempt_new_account_creation(self, service_name: str) -> Optional[Dict]:
        """
        Performs autonomous account creation using visual automation and temp email.
        Returns account details dict on success, raises RuntimeError on failure.
        """
        await self._internal_think(f"Initiating autonomous account creation for service: {service_name}")
        self.logger.info(f"Attempting new account creation for: {service_name}")

        identity = await self._generate_fake_identity()
        if not identity:
            raise RuntimeError(f"Account creation failed for {service_name}: Could not generate fake identity.")

        # --- Setup Context for Creation ---
        creation_context_id: Optional[str] = None
        creation_page: Optional[Page] = None
        try:
            # Get high-quality proxy specifically for signup
            proxy_info = await self.orchestrator.get_proxy(purpose=f"account_creation_{service_name}", quality_level='premium')
            if not proxy_info: self.logger.warning("Failed to get premium proxy for account creation, using standard.") # Fallback?

            creation_context_id, creation_context = await self._get_new_context(account_details={"service": service_name, "status": "creating"}, proxy_info=proxy_info)
            _, creation_page = await self._get_page_for_context(creation_context_id)

            # --- Find Signup Page (Requires service-specific knowledge or discovery) ---
            # This URL should ideally be provided by ThinkTool or discovered dynamically
            signup_url = self.config.get(f"SERVICE_{service_name.upper()}_SIGNUP_URL")
            if not signup_url:
                 # TODO: Implement dynamic discovery of signup URL if needed
                 raise RuntimeError(f"Account creation failed: Signup URL for service '{service_name}' not configured.")

            await creation_page.goto(signup_url, wait_until='domcontentloaded')
            await self._human_like_delay(creation_page)

            # --- Execute Visual UI Automation for Signup ---
            signup_goal = f"Complete the signup process on {service_name} using the provided identity details, handle CAPTCHAs, and perform email verification if required."
            signup_params = {"identity": identity} # Pass generated identity to the automation goal

            # Use the main execute_step logic for the UI automation part
            automation_step = {
                "action": "Execute UI Automation: Signup",
                "tool": "browser",
                "params": {
                    "service": service_name,
                    "goal": signup_goal,
                    "params": signup_params,
                    "max_steps": 35 # Allow more steps for signup complexity
                }
            }
            # Create a temporary task context for this sub-task
            signup_task_context = {"current_context_id": creation_context_id, "current_page_id": _} # Page ID not needed directly here

            # We need to call the automation logic directly here, not via execute_step recursion
            # Reusing the core loop from execute_step's web_ui_automate section:
            max_steps = automation_step['params'].get('max_steps', 35)
            current_step_count = 0
            last_action_summary = "Signup automation started."
            signup_successful = False

            while current_step_count < max_steps:
                 current_step_count += 1
                 await self._internal_think(f"Signup Step {current_step_count}/{max_steps}: Goal='{signup_goal}'", details={"last_action_result": last_action_summary})
                 await self._human_like_delay(creation_page, short=True)

                 screenshot_bytes = await creation_page.screenshot(full_page=True)
                 page_text_content = await creation_page.evaluate("document.body.innerText")
                 current_url = creation_page.url

                 automation_task_context = {
                     "task": "Determine next UI action for account signup",
                     "service_context": service_name, "current_goal": signup_goal,
                     "automation_parameters": signup_params, # Contains identity details
                     "page_text_content_snippet": page_text_content[:5000],
                     "current_url": current_url, "last_action_result": last_action_summary,
                     "current_step": f"{current_step_count}/{max_steps}",
                     "available_actions": ["click", "input", "scroll", "wait", "solve_captcha", "check_email_verification", "finish", "error"], # Adjusted actions
                     "desired_output_format": "JSON: {\"action_type\": \"<chosen_action>\", \"selector\": \"<css>\", \"coordinates\": {\"x\":y}?, \"text_to_input\": \"<string>\"?, \"captcha_type\": \"checkbox|etc\"?, \"reasoning\": \"<Explanation>\", \"error_message\": \"<If error>\"}"
                 }
                 # Add identity details explicitly for LLM context if needed (be mindful of prompt length)
                 # automation_task_context["identity_details_summary"] = f"Email: {identity['email']}, Name: {identity['full_name']}"

                 # Special handling for email verification step
                 if "email verification" in last_action_summary.lower() or "check your email" in page_text_content.lower():
                      automation_task_context["special_focus"] = "Email verification might be required. Check page for confirmation status or decide to trigger 'check_email_verification' action."

                 llm_prompt = await self.generate_dynamic_prompt(automation_task_context)
                 # --- MODIFIED: Removed hardcoded model_preference ---
                 action_json_str = await self.orchestrator.call_llm(
                     agent_name=self.AGENT_NAME, prompt=llm_prompt, temperature=0.05,
                     max_tokens=800, is_json_output=True, image_data=screenshot_bytes # Let orchestrator pick model
                 )
                 # --- END MODIFICATION ---

                 if not action_json_str: last_action_summary = "LLM failed to determine signup action."; continue

                 try:
                     action_data = self._parse_llm_json(action_json_str)
                     if not action_data: raise ValueError("Failed to parse LLM action JSON.")
                     action_type = action_data.get('action_type')
                     reasoning = action_data.get('reasoning', 'N/A')
                     self.logger.info(f"Signup Attempt {current_step_count}: LLM decided action: {action_type}. Reasoning: {reasoning}")

                     action_executed_successfully = False
                     action_result_message = f"Signup action '{action_type}' initiated."

                     # --- Execute Signup Actions ---
                     if action_type == 'click':
                         selector = action_data.get('selector'); coords = action_data.get('coordinates')
                         target_element = await self._find_element(creation_page, selector, coords)
                         if target_element: await self._human_click(creation_page, target_element); action_executed_successfully = True; action_result_message = f"Clicked element '{selector or coords}'."
                         else: action_result_message = f"Click failed: Element not found for '{selector or coords}'."
                     elif action_type == 'input':
                         selector = action_data.get('selector'); coords = action_data.get('coordinates')
                         text_to_input = action_data.get('text_to_input') # LLM should specify value or reference identity key
                         # Resolve text: Check if it refers to identity dict
                         if text_to_input and text_to_input.startswith("identity."):
                              key = text_to_input.split('.')[-1]
                              resolved_text = identity.get(key)
                              if resolved_text is None: action_result_message = f"Input failed: Identity key '{key}' not found."; resolved_text = ""
                              else: text_to_input = resolved_text # Use the resolved value
                         elif text_to_input is None: action_result_message = "Input failed: LLM did not provide text."; text_to_input = ""

                         target_element = await self._find_element(creation_page, selector, coords)
                         if target_element and text_to_input is not None:
                              await self._human_fill(creation_page, target_element, text_to_input)
                              action_executed_successfully = True; action_result_message = f"Input text into element '{selector or coords}'."
                         elif not target_element: action_result_message = f"Input failed: Element not found for '{selector or coords}'."
                     elif action_type == 'scroll':
                         direction = action_data.get('scroll_direction', 'down')
                         await self._human_scroll(creation_page, direction)
                         action_executed_successfully = True; action_result_message = f"Scrolled {direction}."
                     elif action_type == 'wait':
                         wait_ms = action_data.get('wait_time_ms', random.randint(1500, 4000))
                         await creation_page.wait_for_timeout(wait_ms)
                         action_executed_successfully = True; action_result_message = f"Waited for {wait_ms} ms."
                     elif action_type == 'solve_captcha':
                         captcha_type = action_data.get('captcha_type', 'unknown')
                         selector = action_data.get('selector'); coords = action_data.get('coordinates')
                         target_element = await self._find_element(creation_page, selector, coords)
                         if target_element:
                             if captcha_type == 'checkbox':
                                 await self._human_click(creation_page, target_element)
                                 await creation_page.wait_for_timeout(random.uniform(4000, 7000)) # Wait longer for challenge/verification
                                 action_executed_successfully = True; action_result_message = "Clicked CAPTCHA checkbox."
                             else: action_result_message = f"CAPTCHA solving failed: Unsupported type '{captcha_type}'."
                         else: action_result_message = f"CAPTCHA solving failed: Element not found for '{selector or coords}'."
                     elif action_type == 'check_email_verification':
                         self.logger.info("Checking temporary email for verification link...")
                         verification_link = await self.temp_mail_service.get_verification_link(identity['email'], timeout_seconds=180, keyword="verify") # Adjust keyword if needed
                         if verification_link:
                             self.logger.info(f"Found verification link: {verification_link}. Navigating in current context.")
                             # Navigate to the link in the *same* context
                             response = await creation_page.goto(verification_link, wait_until='domcontentloaded')
                             if response and response.ok:
                                 action_executed_successfully = True; action_result_message = "Navigated to email verification link successfully."
                                 await self._human_like_delay(creation_page) # Wait after verification page loads
                             else: action_result_message = f"Email verification failed: Could not load link {verification_link} (Status: {response.status if response else 'N/A'})."
                         else:
                             action_result_message = "Email verification failed: No verification link found in temp email within timeout."
                             # Keep action_executed_successfully = False, LLM might retry or error
                     elif action_type == 'finish':
                         action_executed_successfully = True
                         signup_successful = True
                         action_result_message = f"LLM determined signup process for {service_name} is complete."
                         self.logger.info(action_result_message)
                         break # Exit signup loop
                     elif action_type == 'error':
                         error_msg = action_data.get('error_message', 'LLM indicated signup error.')
                         raise RuntimeError(f"Account creation failed for {service_name}: {error_msg}")
                     else:
                         action_result_message = f"Unsupported action type during signup: {action_type}"

                     last_action_summary = action_result_message
                     if not action_executed_successfully:
                          self.logger.warning(f"Signup Attempt {current_step_count}: Action '{action_type}' failed. Message: {action_result_message}")
                          if current_step_count == max_steps: raise RuntimeError(f"Account creation failed for {service_name} after {max_steps} steps. Last failed action: {action_type}.")

                 except Exception as signup_err:
                      last_action_summary = f"Error during signup step {current_step_count}: {signup_err}"
                      self.logger.error(last_action_summary, exc_info=True)
                      if current_step_count == max_steps: raise RuntimeError(f"Account creation failed for {service_name} due to error: {signup_err}") from signup_err

            # --- Post-Automation Check ---
            if not signup_successful:
                raise RuntimeError(f"Account creation for {service_name} failed: Automation loop completed {max_steps} steps without explicit 'finish' action. Last status: {last_action_summary}")

            # --- Store Credentials Securely ---
            self.logger.info(f"Account creation for {service_name} appears successful. Storing credentials...")
            # Use Orchestrator's secure storage mechanism
            stored_account_details = await self.orchestrator.secure_storage.store_new_account(
                service=service_name,
                identifier=identity['email'], # Or username if applicable
                password=identity['password'],
                status='active', # Mark as active
                metadata={"created_by": self.AGENT_NAME, "creation_ts": datetime.now(timezone.utc).isoformat()}
            )

            if not stored_account_details or 'id' not in stored_account_details:
                 # Log critical error but maybe proceed with in-memory details for current task?
                 self.logger.critical(f"Failed to securely store credentials for newly created {service_name} account ({identity['email']}) via orchestrator!")
                 # Return the generated identity for immediate use, but it won't persist
                 stored_account_details = {**identity, "id": f"temp_{uuid.uuid4().hex[:4]}", "status": "active", "service": service_name} # Temporary ID
            else:
                 self.logger.info(f"Successfully created and stored account ID {stored_account_details['id']} for {service_name}.")

            # Update context map with the *stored* details (including the real ID)
            self.internal_state['context_account_map'][creation_context_id] = stored_account_details

            return stored_account_details

        except Exception as creation_err:
            self.logger.error(f"Account creation process for {service_name} failed: {creation_err}", exc_info=True)
            # Ensure context is closed on failure
            if creation_context_id:
                await self._close_context(creation_context_id)
            raise RuntimeError(f"Account creation failed for {service_name}: {creation_err}") from creation_err
        # No finally block needed for context closing, handled in exception or normal flow


    async def _get_account_details_from_db(self, account_id: int) -> Optional[Dict]:
        """Fetches non-sensitive account details from the database via Orchestrator."""
        # Delegate to orchestrator's secure storage to avoid direct DB dependency here
        if hasattr(self.orchestrator, 'secure_storage') and hasattr(self.orchestrator.secure_storage, 'get_account_details_by_id'):
            try:
                details = await self.orchestrator.secure_storage.get_account_details_by_id(account_id)
                # Ensure password is not included in the returned dict for general use
                if details: details.pop('password', None)
                return details
            except Exception as e:
                self.logger.error(f"Error fetching account details for ID {account_id} via orchestrator: {e}")
                return None
        else:
            self.logger.error("Orchestrator secure storage or get_account_details_by_id method unavailable.")
            return None


    async def _select_available_account_for_service(self, service_name: str) -> Optional[Dict]:
        """Selects an 'active' account for a given service via Orchestrator."""
        self.logger.debug(f"Selecting available account for service: {service_name}")
        if hasattr(self.orchestrator, 'secure_storage') and hasattr(self.orchestrator.secure_storage, 'find_active_account_for_service'):
            try:
                details = await self.orchestrator.secure_storage.find_active_account_for_service(service_name)
                if details:
                    self.logger.info(f"Selected account ID {details.get('id')} for service {service_name}")
                    # Ensure password is not included
                    details.pop('password', None)
                    return details
                else:
                    self.logger.info(f"No 'active' account found for service: {service_name}")
                    return None
            except Exception as e:
                self.logger.error(f"Error selecting account for service {service_name} via orchestrator: {e}")
                return None
        else:
            self.logger.error("Orchestrator secure storage or find_active_account_for_service method unavailable.")
            return None

    def _generate_google_dorks(self, target: str, dork_types: List[str]) -> List[str]:
        """Generates a list of Google Dorks based on requested types."""
        dorks = []
        # Ensure target doesn't have protocol for site: operator
        parsed_target = urlparse(target)
        site_target = parsed_target.netloc or parsed_target.path # Use netloc if available
        if not site_target: return [] # Cannot generate dorks without a valid target domain/path

        base_site_dork = f'site:{site_target}'
        # More comprehensive dorks
        if 'login' in dork_types: dorks.extend([f'{base_site_dork} intitle:"login" | intitle:"signin" | inurl:login | inurl:signin | inurl:auth', f'{base_site_dork} "admin" | "administrator" | "staff" login'])
        if 'password' in dork_types: dorks.extend([f'{base_site_dork} filetype:log | filetype:txt | filetype:cfg | filetype:env | filetype:ini password | pass | pwd | secret', f'{base_site_dork} filetype:sql | filetype:dump | filetype:db | filetype:mdb "password" | "pwd" | "hash"', f'{base_site_dork} intext:"password" | intext:"secret_key" | intext:"api_key"'])
        if 'config_files' in dork_types: dorks.extend([f'{base_site_dork} filetype:env | filetype:yml | filetype:yaml | filetype:config | filetype:conf | filetype:inf | filetype:rdp | filetype:cfg | filetype:ini | filetype:bak | filetype:backup | filetype:swp | filetype:bkf', f'{base_site_dork} ext:sql | ext:db | ext:mdb | ext:backup | ext:bak | ext:swp | ext:bkf', f'{base_site_dork} "index of /" "config" | "admin" | "backup" | "private" | "etc"'])
        if 'api_keys' in dork_types: dorks.extend([f'{base_site_dork} "api_key" | "apikey" | "client_secret" | "access_token" | "authorization_bearer"', f'{base_site_dork} filetype:json | filetype:js | filetype:yaml | filetype:yml | filetype:txt api_key | authorization | secret | token'])
        if 'documents' in dork_types: dorks.extend([f'{base_site_dork} filetype:pdf | filetype:docx | filetype:xlsx | filetype:pptx | filetype:doc | filetype:xls | filetype:ppt "internal" | "confidential" | "private" | "minutes" | "report"'])
        if 'subdomains' in dork_types: dorks.append(f'site:*.{site_target} -site:www.{site_target}')
        if 'error_messages' in dork_types: dorks.append(f'inurl:{site_target} "error" | "exception" | "stack trace" | "warning" | "debug"')
        if 'directory_listing' in dork_types: dorks.append(f'intitle:"index of" {base_site_dork}')
        if 'sensitive_urls' in dork_types: dorks.extend([f'{base_site_dork} inurl:admin | inurl:dashboard | inurl:private | inurl:secure | inurl:backup | inurl:config'])

        unique_dorks = list(dict.fromkeys(dorks)) # Remove duplicates
        self.logger.info(f"Generated {len(unique_dorks)} unique dorks for target '{site_target}'.")
        return unique_dorks


    # --- Learning, Critique, Prompt Generation, Insights (Largely unchanged from v3.2/v3.1) ---
    # These methods provide status and feedback but don't alter core execution logic based on user reqs.
    async def learning_loop(self):
        """Autonomous learning cycle (Placeholder - requires specific learning goals)."""
        self.logger.info("BrowsingAgent learning_loop started (currently passive monitoring).")
        while not self._stop_event.is_set():
            try:
                await self._internal_think("Starting learning cycle: Analyzing proxy performance and task outcomes.")
                # 1. Analyze Proxy Performance
                poor_proxies = []
                proxy_stats = self.internal_state.get('proxy_stats', {})
                for proxy, stats in proxy_stats.items():
                    total_attempts = stats.get('success', 0) + stats.get('failure', 0)
                    if total_attempts > 15: # Higher threshold for analysis
                        failure_rate = stats.get('failure', 0) / total_attempts
                        if failure_rate > 0.6: # Flag proxies failing > 60%
                            poor_proxies.append({"proxy": proxy.split('@')[-1], "failure_rate": failure_rate, "attempts": total_attempts})
                            self.logger.warning(f"Proxy {proxy.split('@')[-1]} identified as poor performing (Fail rate: {failure_rate:.1%}).")

                if poor_proxies:
                     # Report poor proxies to Orchestrator/ThinkTool for potential action
                     await self.log_knowledge_fragment(
                         agent_source=self.AGENT_NAME, data_type="proxy_performance_alert",
                         content={"poor_proxies": poor_proxies}, tags=["proxy", "performance", "alert", "monitoring"], relevance_score=0.8
                     )
                     # Optionally trigger orchestrator action
                     # await self.orchestrator.handle_poor_proxies(poor_proxies)

                # 2. Analyze Recent Task Failures (Example: UI Automation)
                # This requires querying the KB via ThinkTool or accessing task history
                # failure_fragments = await self.query_knowledge_base(...)
                # if failed_goals:
                #    self.logger.warning(f"Recent UI Automation Failures: {dict(failed_goals)}")
                #    # Trigger LLM analysis or ThinkTool directive to investigate common failures

                self.internal_state["last_learning_cycle_ts"] = datetime.now(timezone.utc)
                self.logger.debug("BrowsingAgent learning cycle complete.")

                learn_interval = int(self.config.get("BROWSER_LEARNING_INTERVAL_S", 7200)) # Default 2 hours
                await asyncio.sleep(learn_interval)

            except asyncio.CancelledError: self.logger.info("BrowsingAgent learning loop cancelled."); break
            except Exception as e:
                self.logger.error(f"Error in BrowsingAgent learning loop: {e}", exc_info=True)
                await self._report_error(f"Learning loop error: {e}")
                await asyncio.sleep(60 * 30) # Wait longer after error

    async def self_critique(self) -> Dict[str, Any]:
        """Evaluates agent health, resource usage, and proxy performance."""
        self.logger.info(f"{self.AGENT_NAME}: Performing self-critique.")
        critique = {"status": "ok", "feedback": "Critique pending analysis."}
        try:
            num_active_contexts = len(self.internal_state.get('browser_contexts', {}))
            proxy_stats = self.internal_state.get('proxy_stats', {})
            # Summarize proxy stats without credentials
            proxy_stats_summary = {
                p.split('@')[-1]: f"S:{s.get('success',0)}/F:{s.get('failure',0)} ({(s.get('success',0)/(s.get('success',0)+s.get('failure',1)))*100:.1f}%)"
                for p, s in proxy_stats.items() if (s.get('success',0)+s.get('failure',0)) > 0
            }
            critique['resource_usage'] = {"active_contexts": num_active_contexts, "max_contexts": self.internal_state['max_concurrent_contexts']}
            critique['proxy_stats_summary'] = proxy_stats_summary

            feedback_points = [f"Active Contexts: {num_active_contexts}/{self.internal_state['max_concurrent_contexts']}."]
            high_failure_proxies_count = sum(1 for stats in proxy_stats.values() if (stats.get('success',0) + stats.get('failure',0)) > 10 and stats.get('failure',0) / (stats.get('success',0) + stats.get('failure',1)) > 0.5)
            if high_failure_proxies_count > 0:
                feedback_points.append(f"WARNING: {high_failure_proxies_count} proxies show high failure rates (>50% with >10 attempts). Performance degradation possible.")
                critique['status'] = 'warning'
            if num_active_contexts >= self.internal_state['max_concurrent_contexts']:
                feedback_points.append("WARNING: Operating at maximum concurrent context limit. Task throughput may be limited.")
                critique['status'] = 'warning'

            # TODO: Query DB/KB for recent task success/failure rates specific to this agent.

            critique['feedback'] = " ".join(feedback_points)
        except Exception as e:
            self.logger.error(f"Error during self-critique: {e}", exc_info=True)
            critique['status'] = 'error'; critique['feedback'] = f"Critique failed: {e}"
        return critique

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Constructs context-rich prompts for LLM calls, emphasizing visual input."""
        self.logger.debug(f"Generating dynamic prompt for BrowsingAgent task: {task_context.get('task')}")
        prompt_parts = [self.meta_prompt] # Start with agent's core identity and capabilities
        prompt_parts.append("\n--- Current Task & Context ---")
        # Prioritize key context items
        priority_keys = ['task', 'current_goal', 'service_context', 'current_url', 'last_action_result', 'special_focus', 'identity_details_summary']
        for key in priority_keys:
            if key in task_context:
                 value = task_context[key]
                 value_str = str(value)[:1000] + ("..." if len(str(value)) > 1000 else "") # Limit length
                 prompt_parts.append(f"**{key.replace('_', ' ').title()}**: {value_str}")

        prompt_parts.append("\n**Visual Context:** A screenshot of the current webpage is provided.")
        prompt_parts.append("**Text Context Snippet:**")
        text_snippet = task_context.get('page_text_content_snippet') or task_context.get('text_content_snippet', '')
        prompt_parts.append(f"```\n{text_snippet[:4000]}\n```") # Limit text snippet length

        # Add remaining context items concisely
        prompt_parts.append("\n**Other Parameters:**")
        other_params = {k: v for k, v in task_context.items() if k not in priority_keys and k not in ['page_text_content_snippet', 'text_content_snippet', 'task', 'desired_output_format']}
        if other_params:
             try: prompt_parts.append(f"```json\n{json.dumps(other_params, default=str, indent=2)}\n```")
             except TypeError: prompt_parts.append(str(other_params)[:1000] + "...") # Fallback
        else: prompt_parts.append("None")

        # --- Specific Instructions based on Task ---
        prompt_parts.append("\n--- Your Instructions ---")
        task_type = task_context.get('task')
        if task_type == 'Determine next UI action based primarily on visual context' or task_type == 'Determine next UI action for account signup':
            prompt_parts.append(f"Analyze the provided **screenshot** and text snippet. Based *primarily* on the visual layout and the current goal ('{task_context.get('current_goal', 'N/A')}'), determine the single best next action from {task_context.get('available_actions', [])}.")
            prompt_parts.append("Provide a precise selector (CSS or XPath) or coordinates for the target element. If inputting text, specify the exact text or reference the 'params.<key>' from parameters.") # MODIFIED: Reference 'params.<key>'
            prompt_parts.append("If a CAPTCHA is visible, use the 'solve_captcha' action and identify its type and target element.")
            prompt_parts.append("If the goal involves email verification, use 'check_email_verification' when appropriate.")
            prompt_parts.append("Use 'finish' only when the goal is definitively achieved. Use 'error' if stuck or encountering an unrecoverable issue.")
            prompt_parts.append("Provide clear reasoning, focusing on visual cues.")
        elif task_type == 'Extract structured data from webpage using visual and text context':
            prompt_parts.append(f"Analyze the screenshot and text. Follow these instructions: {task_context.get('extraction_instructions', 'Extract key information.')}")
            prompt_parts.append("Identify the data visually and textually. Use null for missing fields.")
        elif task_type == 'Analyze webpage screenshot and text for potential credentials...':
             prompt_parts.append(f"Carefully examine the **screenshot** and text from URL '{task_context.get('target_url')}' for any potential credentials (passwords, API keys, secrets), sensitive configuration data, exposed login forms, or other security risks.")
             prompt_parts.append("Focus on visual elements like forms, code blocks, and unusual text patterns. Be precise in your findings.")
        elif task_type == 'Check for Google search block/CAPTCHA':
             prompt_parts.append("Analyze the screenshot of the Google search results page. Determine if the search was blocked or if a CAPTCHA is present. Respond with the required JSON structure.")
        else: prompt_parts.append("Analyze the provided context and visual information to complete the task as described.")

        if task_context.get('desired_output_format'):
            prompt_parts.append(f"\n**Output Format:** Respond ONLY with valid JSON matching this structure: {task_context['desired_output_format']}")
            # Ensure JSON block instruction is clear if needed
            if "JSON" in task_context.get('desired_output_format', ''):
                 prompt_parts.append("\n```json") # Start JSON block hint

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for BrowsingAgent task: {task_type} (Length: {len(final_prompt)} chars)")
        return final_prompt


    async def collect_insights(self) -> Dict[str, Any]:
        """Collects insights about browsing activity, resource usage, and proxy status."""
        self.logger.debug("BrowsingAgent collect_insights called.")
        num_active_contexts = len(self.internal_state.get('browser_contexts', {}))
        proxy_stats = self.internal_state.get('proxy_stats', {})
        total_success = sum(s.get('success', 0) for s in proxy_stats.values())
        total_failure = sum(s.get('failure', 0) for s in proxy_stats.values())
        total_attempts = total_success + total_failure
        avg_proxy_success_rate = (total_success / total_attempts) if total_attempts > 0 else 1.0
        # Count accounts created in this session (requires tracking)
        accounts_created = self.internal_state.get('accounts_created_this_session', 0)

        insights = {
            "agent_name": self.AGENT_NAME, "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_contexts": num_active_contexts,
            "max_contexts": self.internal_state.get('max_concurrent_contexts'),
            "avg_proxy_success_rate": round(avg_proxy_success_rate, 3),
            "total_proxy_attempts": total_attempts,
            "accounts_created_session": accounts_created,
            }
        return insights

    async def stop(self, timeout: float = 45.0): # Increased timeout for graceful shutdown
        """Override stop to close browser/contexts and cancel tasks gracefully."""
        if self._status in [self.STATUS_STOPPING, self.STATUS_STOPPED]:
            self.logger.info(f"{self.AGENT_NAME} stop requested but already {self._status}.")
            return

        self.logger.info(f"{self.AGENT_NAME} received stop signal. Initiating graceful shutdown...")
        self._status = self.STATUS_STOPPING
        self._stop_event.set()

        # 1. Cancel Background Tasks (like learning loop)
        tasks_to_cancel = list(self._background_tasks)
        if tasks_to_cancel:
            self.logger.info(f"Cancelling {len(tasks_to_cancel)} BrowsingAgent background tasks...")
            for task in tasks_to_cancel:
                if task and not task.done(): task.cancel()
            # Wait briefly for cancellations
            await asyncio.sleep(1)

        # 2. Gracefully close browser resources
        await self._close_browser_and_playwright()

        # 3. Wait for background tasks to finish cancelling
        if tasks_to_cancel:
            self.logger.info(f"Waiting up to {timeout/2:.1f}s for background tasks to finalize...")
            done, pending = await asyncio.wait(tasks_to_cancel, timeout=timeout/2, return_when=asyncio.ALL_COMPLETED)
            if pending: self.logger.warning(f"{len(pending)} BrowsingAgent background tasks did not cancel/finalize gracefully.")

        # 4. Call base class stop
        await super().stop(timeout/2) # Pass remaining timeout
        self.logger.info(f"{self.AGENT_NAME} stopped.")

    # --- KB Interaction Helpers (Delegate to ThinkTool/Orchestrator) ---
    async def log_knowledge_fragment(self, *args, **kwargs):
        """Logs knowledge via ThinkTool or Orchestrator."""
        if self.think_tool and hasattr(self.think_tool, 'log_knowledge_fragment'):
            return await self.think_tool.log_knowledge_fragment(*args, **kwargs)
        elif hasattr(self.orchestrator, 'log_knowledge_fragment'): # Fallback to orchestrator
             return await self.orchestrator.log_knowledge_fragment(*args, **kwargs)
        else: self.logger.error("Neither ThinkTool nor Orchestrator available for logging KB fragment."); return None

    async def query_knowledge_base(self, *args, **kwargs):
         """Queries knowledge via ThinkTool or Orchestrator."""
         # Browsing agent typically executes, doesn't query KB itself often
         self.logger.warning("BrowsingAgent query_knowledge_base called - typically delegated.")
         if self.think_tool and hasattr(self.think_tool, 'query_knowledge_base'):
             return await self.think_tool.query_knowledge_base(*args, **kwargs)
         elif hasattr(self.orchestrator, 'query_knowledge_base'):
              return await self.orchestrator.query_knowledge_base(*args, **kwargs)
         return []

    # --- Helper Methods ---
    def _parse_llm_json(self, json_string: str, expect_type: Union[Type[dict], Type[list]] = dict) -> Union[Dict, List, None]:
        """Safely parses JSON from LLM output, handling markdown code blocks and common issues."""
        if not json_string: return None
        try:
            # 1. Find JSON block (handles ```json ... ```)
            match = None
            if expect_type == list:
                match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', json_string, re.DOTALL)
                start_char, end_char = '[', ']'
            else: # Default to dict
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_string, re.DOTALL)
                start_char, end_char = '{', '}'

            potential_json = ""
            if match:
                potential_json = match.group(1)
            else:
                # If no markdown block, try finding the first '{' or '[' and last '}' or ']'
                start_index = json_string.find(start_char)
                end_index = json_string.rfind(end_char)
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    potential_json = json_string[start_index : end_index + 1]
                else:
                    # As a last resort, assume the whole string might be JSON if it starts/ends correctly
                    trimmed = json_string.strip()
                    if trimmed.startswith(start_char) and trimmed.endswith(end_char):
                         potential_json = trimmed
                    else:
                         self.logger.warning(f"Could not extract potential JSON ({expect_type}) structure from LLM output: {json_string[:200]}...")
                         return None

            # 2. Clean and Parse
            try:
                # Basic cleaning: remove trailing commas before closing bracket/brace
                cleaned_json = re.sub(r',\s*([\}\]])', r'\1', potential_json)
                # Attempt parsing
                parsed_json = json.loads(cleaned_json)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing failed after cleaning ({e}): {potential_json[:200]}...")
                # Optionally try more aggressive cleaning here if needed
                return None

            # 3. Validate Type
            if isinstance(parsed_json, expect_type):
                return parsed_json
            else:
                self.logger.error(f"Parsed JSON type mismatch. Expected {expect_type}, got {type(parsed_json)}")
                return None

        except Exception as e:
            self.logger.error(f"Unexpected error during JSON parsing: {e}", exc_info=True)
            return None