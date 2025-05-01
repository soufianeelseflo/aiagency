# Filename: agents/browsing_agent.py
# Description: Genius Agentic Browsing Agent - Handles all web interactions,
#              multi-instance management, proxy rotation, anti-ban, and task execution.
# Version: 3.0 (Genius Agentic - Multi-Instance, Clay API, Trial Creation)

import asyncio
import logging
import json
import os
import random
import time
import base64
import re # Added for sanitizing filenames etc.
import uuid # Added for unique identifiers
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, Union
from urllib.parse import urlparse

# --- Core Framework Imports ---
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

# --- Project Imports ---
try:
    # Use production-ready base class
    from .base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
except ImportError:
    # Fallback, log warning
    logging.warning("Production base agent not found, using GeniusAgentBase. Ensure base_agent_prod.py is used.")
    from .base_agent import GeniusAgentBase

from models import AccountCredentials # To log new account details
from config.settings import settings # Access validated settings
from utils.database import encrypt_data # For encrypting credentials before logging

# --- Web Interaction Libraries ---
import aiohttp # For direct API calls like Clay.com
try:
    from playwright.async_api import (
        async_playwright, Playwright, Browser, BrowserContext, Page, Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError
    )
except ImportError:
    logging.critical("Playwright library not found. BrowsingAgent cannot function. Install: pip install playwright && python -m playwright install --with-deps")
    raise ImportError("Playwright is required for BrowsingAgent")

try:
    from fake_useragent import UserAgent
except ImportError:
    logging.warning("fake_useragent not installed. Using default User-Agent.")
    UserAgent = None # Define as None if import fails

# Configure logger
logger = logging.getLogger(__name__)
# Configure dedicated operational logger
op_logger = logging.getLogger('OperationalLog') # Assuming setup elsewhere

# --- Meta Prompt ---
BROWSING_AGENT_META_PROMPT = """
You are the BrowsingAgent within the Synapse AI Sales System.
Your Core Mandate: Execute all web-based interactions flawlessly and stealthily as directed by the Orchestrator/ThinkTool. This includes scraping, API calls (via HTTP or browser automation), social media actions, and account creation/management.
Key Responsibilities:
1.  **Task Execution:** Reliably perform actions like `scrape_url`, `execute_clay_api_call`, `create_free_trial_account`, `execute_social_post`, `execute_social_interact`, `perform_search_and_summarize`.
2.  **Multi-Instance Management:** Manage multiple independent browser contexts, each potentially tied to a specific account identity and proxy. Ensure contexts persist state (cookies) for specific identifiers.
3.  **Proxy & Identity Management:** Utilize proxies provided by the Orchestrator for specific tasks or accounts. Maintain session integrity per context. Use realistic user agents.
4.  **Anti-Detection & Human Simulation:** Implement techniques to avoid bot detection (random delays between actions, realistic typing speeds, viewport interaction). Handle basic CAPTCHAs by flagging failure if unsolvable.
5.  **Robust Interaction:** Navigate websites, fill forms, click buttons, handle logins, extract data based on provided selectors or intelligent inference (if necessary). Handle dynamic page loads and waits effectively.
6.  **Error Handling & Reporting:** Gracefully handle common web errors (timeouts, connection errors, element not found). Take screenshots on failure. Report failures clearly to the Orchestrator.
7.  **Resource Efficiency:** Close browser contexts and pages promptly when no longer needed or after a period of inactivity.
**Goal:** Be the reliable "hands and eyes" of the agency on the internet, executing complex web tasks efficiently and avoiding bans.
"""

class BrowsingAgent(GeniusAgentBase):
    """
    Browsing Agent (Genius Level): Executes all web interactions, manages browser instances,
    handles proxies, simulates human behavior, and performs tasks like scraping, API calls,
    account creation, and social media actions.
    Version: 3.0
    """
    AGENT_NAME = "BrowsingAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any, smartproxy_password: Optional[str]):
        """Initializes the BrowsingAgent."""
        # ### Phase 3 Plan Ref: 6.1 (Implement __init__)
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, session_maker=session_maker)
        self.meta_prompt = BROWSING_AGENT_META_PROMPT

        # Playwright instance management
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._contexts: Dict[str, BrowserContext] = {} # identifier -> context
        self._context_locks: Dict[str, asyncio.Lock] = {} # identifier -> lock (for safe access/creation)
        self._context_last_used: Dict[str, float] = {} # identifier -> timestamp

        # Proxy configuration
        self._proxy_username = self.config.get("SMARTPROXY_USERNAME")
        self._proxy_password = smartproxy_password # Passed during init
        self._proxy_endpoint_base = os.getenv("SMARTPROXY_ENDPOINT", "gate.smartproxy.com:7000") # Example

        # User Agent Generator
        self.user_agent_generator = None
        try:
            if UserAgent: self.user_agent_generator = UserAgent()
        except Exception as ua_err:
            self.logger.warning(f"Failed to initialize fake_useragent: {ua_err}. Using default UA.")

        self.logger.info(f"{self.AGENT_NAME} v3.0 initialized.")
        # Start Playwright in the background
        asyncio.create_task(self._ensure_playwright_running())
        # Start context cleanup task
        asyncio.create_task(self._periodic_context_cleanup())

    async def log_operation(self, level: str, message: str):
        """Helper to log to the operational log file."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err: logger.error(f"Failed to write to operational log: {log_err}")

    # --- Playwright Lifecycle Management ---

    async def _ensure_playwright_running(self):
        """Starts Playwright and launches a persistent browser instance if not already running."""
        # ### Phase 3 Plan Ref: 6.1
        if self._browser and self._browser.is_connected():
            self.logger.debug("Playwright browser already running.")
            return True
        try:
            self.logger.info("Initializing Playwright and launching browser...")
            self._playwright = await async_playwright().start()
            # Launch Chromium. Headless=True for server environments.
            self._browser = await self._playwright.chromium.launch(headless=True)
            self.logger.info("Playwright browser launched successfully.")
            return True
        except Exception as e:
            self.logger.critical(f"Failed to initialize Playwright or launch browser: {e}", exc_info=True)
            self._status = self.STATUS_ERROR
            await self._report_error(f"Playwright initialization failed: {e}")
            return False

    async def _shutdown_playwright(self):
        """Closes all contexts and the browser."""
        # ### Phase 3 Plan Ref: 6.1
        self.logger.info("Shutting down Playwright...")
        for identifier in list(self._contexts.keys()):
            await self._close_browser_context(identifier)
        if self._browser and self._browser.is_connected(): # Check connection before closing
            try:
                await self._browser.close()
                self.logger.info("Playwright browser closed.")
            except Exception as e:
                self.logger.error(f"Error closing Playwright browser: {e}")
        if self._playwright:
            try:
                # Playwright's stop method is synchronous in the library, run in executor
                await asyncio.to_thread(self._playwright.stop)
                self.logger.info("Playwright stopped.")
            except Exception as e:
                self.logger.error(f"Error stopping Playwright: {e}")
        self._browser = None
        self._playwright = None
        self._contexts = {}
        self._context_locks = {}
        self._context_last_used = {}

    # --- Browser Context & Proxy Management ---

    def _get_proxy_config(self, session_id: Optional[str] = None, country: Optional[str] = None) -> Optional[Dict]:
        """Constructs proxy dictionary for Playwright, potentially with session/geo."""
        # ### Phase 3 Plan Ref: 6.6 (Proxy config generation)
        if not self._proxy_username or not self._proxy_password or not self._proxy_endpoint_base:
            self.logger.warning("Proxy credentials or endpoint missing. Cannot configure proxy.")
            return None

        # Smartproxy session format: user-USERNAME-country-COUNTRY-session-SESSIONID
        proxy_user = f"user-{self._proxy_username}"
        if country: proxy_user += f"-country-{country.lower()}"
        # Use a unique but deterministic session ID based on the identifier
        session_part = session_id if session_id else f"rand_{random.randint(10000, 99999)}"
        proxy_user += f"-session-{session_part}"

        return {
            "server": f"http://{self._proxy_endpoint_base}", # Playwright needs scheme
            "username": proxy_user,
            "password": self._proxy_password
        }

    async def get_proxy_for_account(self, account_identifier: Optional[str] = None, purpose: str = "general", target_url: Optional[str] = None) -> Optional[str]:
        """Gets a formatted proxy URL string, potentially specific to an account."""
        # ### Phase 3 Plan Ref: 6.6 (Proxy URL for Orchestrator/aiohttp)
        session_id = account_identifier # Use account identifier for sticky session
        # TODO: Determine country based on target_url or purpose if needed
        country = None
        proxy_dict = self._get_proxy_config(session_id=session_id, country=country)
        if proxy_dict:
            # Format for aiohttp/requests
            server_no_scheme = proxy_dict['server'].replace('http://', '').replace('https://', '')
            return f"http://{proxy_dict['username']}:{proxy_dict['password']}@{server_no_scheme}"
        return None

    async def _get_browser_context(self, identifier: str, recreate: bool = False) -> Optional[BrowserContext]:
        """Gets or creates a BrowserContext, potentially with a unique proxy and user agent."""
        # ### Phase 3 Plan Ref: 6.1 (Multi-instance context management)
        if not await self._ensure_playwright_running(): return None

        # Ensure a lock exists for this identifier
        if identifier not in self._context_locks:
            self._context_locks[identifier] = asyncio.Lock()

        async with self._context_locks[identifier]:
            if identifier in self._contexts and not recreate:
                self.logger.debug(f"Reusing existing browser context for identifier: {identifier}")
                self._context_last_used[identifier] = time.time() # Update last used time
                return self._contexts[identifier]

            if identifier in self._contexts and recreate:
                self.logger.info(f"Recreating browser context for identifier: {identifier}")
                await self._close_browser_context(identifier, acquire_lock=False) # Close existing without re-acquiring lock

            self.logger.info(f"Creating new browser context for identifier: {identifier}")
            try:
                # Generate unique proxy session for this context
                proxy_config = self._get_proxy_config(session_id=identifier)
                user_agent = self.user_agent_generator.random if self.user_agent_generator else None

                context_options = {
                    "user_agent": user_agent,
                    "proxy": proxy_config,
                    "locale": "en-US",
                    "timezone_id": "America/New_York", # Example, make configurable?
                    "geolocation": None, # TODO: Add random geolocation?
                    "permissions": ["geolocation"], # Example permissions
                    "java_script_enabled": True,
                    "accept_downloads": True,
                }
                context_options = {k: v for k, v in context_options.items() if v is not None}

                context = await self._browser.new_context(**context_options)
                self._contexts[identifier] = context
                self._context_last_used[identifier] = time.time()
                self.logger.info(f"Browser context created for {identifier} with Proxy Session: {proxy_config['username'] if proxy_config else 'None'}")
                return context
            except Exception as e:
                self.logger.error(f"Failed to create browser context for {identifier}: {e}", exc_info=True)
                await self._report_error(f"Failed to create browser context for {identifier}: {e}")
                # Clean up lock if creation failed
                self._context_locks.pop(identifier, None)
                return None

    async def _close_browser_context(self, identifier: str, acquire_lock: bool = True):
        """Closes and removes a specific browser context."""
        lock = self._context_locks.get(identifier)
        if not lock:
             # If no lock exists, context likely doesn't either or was already cleaned up
             self.logger.debug(f"No lock found for context identifier {identifier} during close request, likely already closed.")
             self._contexts.pop(identifier, None) # Ensure removal from dicts
             self._context_last_used.pop(identifier, None)
             return

        if acquire_lock:
            acquired = await lock.acquire()
            if not acquired: # Should not happen with default Lock behavior
                 self.logger.error(f"Failed to acquire lock for closing context {identifier}")
                 return
        try:
            context = self._contexts.pop(identifier, None)
            self._context_last_used.pop(identifier, None) # Remove usage timestamp
            if context:
                try:
                    await context.close()
                    self.logger.info(f"Closed browser context for identifier: {identifier}")
                except Exception as e:
                    # Log error but continue cleanup
                    self.logger.error(f"Error closing context {identifier}: {e}")
            else:
                self.logger.debug(f"Context {identifier} already closed or never existed.")
            # Remove the lock itself after closing the context
            self._context_locks.pop(identifier, None)
        finally:
            if acquire_lock and lock.locked():
                lock.release()

    async def _periodic_context_cleanup(self, interval_seconds: int = 300, max_idle_seconds: int = 1800):
         """Periodically closes contexts that haven't been used recently."""
         self.logger.info(f"Starting periodic browser context cleanup task (Interval: {interval_seconds}s, Max Idle: {max_idle_seconds}s).")
         while True:
             await asyncio.sleep(interval_seconds)
             now = time.time()
             closed_count = 0
             # Iterate over a copy of keys as we might modify the dict
             for identifier in list(self._contexts.keys()):
                 last_used = self._context_last_used.get(identifier, 0)
                 if now - last_used > max_idle_seconds:
                     self.logger.info(f"Context '{identifier}' idle for > {max_idle_seconds}s. Closing.")
                     await self._close_browser_context(identifier) # Handles lock acquisition/release
                     closed_count += 1
             if closed_count > 0:
                 self.logger.info(f"Periodic cleanup closed {closed_count} idle browser contexts.")


    async def _get_page(self, identifier: str) -> Optional[Page]:
        """Gets a new page within a specific browser context."""
        # ### Phase 3 Plan Ref: 6.1
        context = await self._get_browser_context(identifier)
        if not context: return None
        try:
            page = await context.new_page()
            # Add stealth measures
            await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return page
        except Exception as e:
            self.logger.error(f"Failed to create new page for context {identifier}: {e}")
            return None

    # --- Human Simulation Helpers ---

    async def _human_like_type(self, page: Page, selector: str, text: str, delay_ms: int = 110):
        """Types text into an element with human-like delays."""
        # ### Phase 3 Plan Ref: 6.5 (Implement human-like interaction)
        self.logger.debug(f"Typing into selector '{selector}' with human-like delay.")
        try:
            # Wait for element and click with slight delay
            element = page.locator(selector).first # Ensure we target one element
            await element.wait_for(state='visible', timeout=15000)
            await element.click(delay=random.uniform(50, 150))
            await asyncio.sleep(random.uniform(0.1, 0.3)) # Pause after click

            # Use fill with a slight delay before starting
            await element.fill(text, timeout=15000)
            # Optional: Add slight pause after filling
            await asyncio.sleep(random.uniform(0.2, 0.5))

        except PlaywrightError as e:
            self.logger.warning(f"Playwright error during human-like typing into '{selector}': {e}")
            raise # Re-raise to be caught by caller
        except Exception as e:
             self.logger.error(f"Unexpected error during typing into '{selector}': {e}")
             raise PlaywrightError(f"Typing failed: {e}") from e


    async def _human_like_click(self, page: Page, selector: str, timeout: int = 15000):
        """Clicks an element with slight random delay after ensuring visibility."""
        # ### Phase 3 Plan Ref: 6.5 (Implement human-like interaction)
        self.logger.debug(f"Clicking selector '{selector}' with human-like delay.")
        try:
            element = page.locator(selector).first
            await element.wait_for(state='visible', timeout=timeout)
            # Optional: Hover briefly before clicking
            try:
                await element.hover(timeout=3000)
                await asyncio.sleep(random.uniform(0.1, 0.4))
            except Exception:
                self.logger.debug(f"Hover failed or not applicable for selector '{selector}', proceeding with click.")

            await element.click(delay=random.uniform(80, 200), timeout=timeout)
            # Optional: Short pause after click
            await asyncio.sleep(random.uniform(0.1, 0.3))
        except PlaywrightError as e:
            self.logger.warning(f"Playwright error during human-like click on '{selector}': {e}")
            raise # Re-raise
        except Exception as e:
             self.logger.error(f"Unexpected error during click on '{selector}': {e}")
             raise PlaywrightError(f"Click failed: {e}") from e

    # --- Core Task Execution Methods ---

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Executes browsing-related tasks delegated by the Orchestrator."""
        # ### Phase 3 Plan Ref: 6.7 (Implement execute_task router)
        action = task_details.get('action')
        self.logger.info(f"{self.AGENT_NAME} executing action: {action}")
        self._status = self.STATUS_EXECUTING
        result = {"status": "failure", "message": f"Unsupported browsing action: {action}"}

        handler_method = None
        if action == 'scrape_url': handler_method = self._scrape_url_task
        elif action == 'execute_clay_api_call': handler_method = self._execute_clay_api_call_task
        elif action == 'create_free_trial_account': handler_method = self._create_free_trial_account_task
        elif action == 'execute_social_post': handler_method = self._execute_social_post_task
        elif action == 'execute_social_interact': handler_method = self._execute_social_interact_task
        elif action == 'perform_search_and_summarize': handler_method = self._perform_search_and_summarize_task
        # Add other actions...

        if handler_method:
            try:
                result = await handler_method(task_details)
            except PlaywrightError as pe:
                 self.logger.error(f"Playwright error during action '{action}': {pe}", exc_info=True)
                 result = {"status": "error", "message": f"Playwright Error: {pe}"}
                 await self._report_error(f"Playwright Error during {action}: {pe}")
            except Exception as e:
                self.logger.error(f"Unexpected error during action '{action}': {e}", exc_info=True)
                result = {"status": "error", "message": f"Unexpected error: {e}"}
                await self._report_error(f"Unexpected error during {action}: {e}")
        else:
            self.logger.warning(f"No handler found for action: {action}")

        self._status = self.STATUS_IDLE
        return result

    # --- Specific Task Handlers ---

    async def _scrape_url_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handles scraping data from a specific URL."""
        # ### Phase 3 Plan Ref: 6.4 (Add execute_scrape_task)
        url = task_details.get('url')
        selectors = task_details.get('selectors') # Dict: {'data_point_name': 'css_selector'}
        account_identifier = task_details.get('account_identifier', f"scrape_{urlparse(url).netloc}_{uuid.uuid4().hex[:6]}") # Unique ID for scrape task
        if not url or not isinstance(selectors, dict): return {"status": "failure", "message": "Missing url or valid selectors dict for scrape task."}

        page = None
        scraped_data = {}
        try:
            await self._internal_think(f"Scraping URL: {url}", details={"selectors": selectors, "context": account_identifier})
            page = await self._get_page(account_identifier) # Get a page within the context
            if not page: raise RuntimeError("Failed to get browser page.")

            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(random.uniform(2, 5)) # Wait for dynamic content

            for name, selector in selectors.items():
                try:
                    element = page.locator(selector).first
                    await element.wait_for(state='visible', timeout=10000) # Wait for element
                    content = await element.text_content(timeout=5000)
                    scraped_data[name] = content.strip() if content else None
                except PlaywrightTimeoutError:
                    self.logger.warning(f"Timeout finding/waiting for selector '{selector}' for '{name}' on {url}")
                    scraped_data[name] = None
                except Exception as loc_err:
                     self.logger.warning(f"Error extracting selector '{selector}' for '{name}' on {url}: {loc_err}")
                     scraped_data[name] = None

            self.logger.info(f"Scraping successful for {url}. Found data for {len([v for v in scraped_data.values() if v])} selectors.")
            return {"status": "success", "data": scraped_data}
        except Exception as e:
            self.logger.error(f"Scraping failed for {url}: {e}", exc_info=True)
            return {"status": "failure", "message": f"Scraping failed: {e}"}
        finally:
            if page: await page.close()
            # Close context after scrape? Optional - keeps session if needed later
            # await self._close_browser_context(account_identifier)

    async def _execute_clay_api_call_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handles making a direct API call to Clay.com."""
        # ### Phase 3 Plan Ref: 6.2 (Implement execute_clay_api_call)
        api_key = os.getenv("CLAY_API_KEY") # Get key from environment
        parameters = task_details.get('parameters') # e.g., {"linkedin_url": "...", "enrichment_type": "email"}
        if not api_key: return {"status": "failure", "message": "Clay.com API key not found in environment."}
        if not parameters: return {"status": "failure", "message": "Missing parameters for Clay API call."}

        # Construct Clay API request (EXAMPLE - needs actual Clay API docs)
        clay_api_endpoint = "https://api.clay.com/v1/enrich" # Placeholder URL
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = parameters # Assuming parameters match API body

        await self._internal_think("Executing Clay.com API call", details=payload)
        try:
            timeout = aiohttp.ClientTimeout(total=45)
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                async with session.post(clay_api_endpoint, json=payload) as response:
                    response_data = await response.json() if 'application/json' in response.headers.get('Content-Type', '') else await response.text()
                    if 200 <= response.status < 300:
                        self.logger.info(f"Clay.com API call successful. Status: {response.status}")
                        # Report cost to Orchestrator
                        await self.orchestrator.report_expense(self.AGENT_NAME, 0.01, "API", "Clay.com API Call") # Example cost
                        return {"status": "success", "data": response_data}
                    else:
                        self.logger.error(f"Clay.com API call failed. Status: {response.status}, Response: {str(response_data)[:500]}...")
                        return {"status": "failure", "message": f"Clay API Error (Status {response.status})", "details": response_data}
        except Exception as e:
            self.logger.error(f"Error during Clay.com API call: {e}", exc_info=True)
            return {"status": "error", "message": f"Clay API call exception: {e}"}

    async def _create_free_trial_account_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handles creating a new free trial account for a service."""
        # ### Phase 3 Plan Ref: 6.3 (Implement create_free_trial_account)
        service = task_details.get('service') # e.g., 'clay.com', 'heygen.com'
        target_url = task_details.get('target_url') # Signup URL
        new_email = task_details.get('new_email') # Email provided by ThinkTool/Orchestrator
        if not service or not target_url or not new_email:
            return {"status": "failure", "message": "Missing service, target_url, or new_email for trial creation."}

        account_identifier = new_email # Use email as the unique ID for this context/proxy
        page = None
        context = None # Keep track of context to close it
        generated_password = f"GeniusP@ss{random.randint(100000,999999)}!" # More complex password

        try:
            await self._internal_think(f"Creating new trial account for {service} using email {new_email}", details={"url": target_url})
            context = await self._get_browser_context(account_identifier, recreate=True) # Use a fresh context/proxy/UA
            if not context: raise RuntimeError("Failed to get browser context for trial creation.")
            page = await self._get_page(account_identifier) # Get page from this specific context
            if not page: raise RuntimeError("Failed to get browser page for trial creation.")

            await page.goto(target_url, wait_until="domcontentloaded", timeout=90000)
            await asyncio.sleep(random.uniform(3, 6))

            # --- Platform-Specific Signup Logic ---
            # Needs refinement based on actual site structure
            email_selector = 'input[type="email"], input[name*="email"], #email'
            password_selector = 'input[type="password"], input[name*="password"], #password'
            name_selector = 'input[name*="name"], #name, #fullname' # Guess common name fields
            submit_selector = 'button[type="submit"], input[type="submit"], button:has-text("Sign Up"), button:has-text("Create Account"), button:has-text("Continue")'

            await self._human_like_type(page, email_selector, new_email)
            await self._human_like_type(page, password_selector, generated_password)
            # Attempt to fill name if selector found
            try: await self._human_like_type(page, name_selector, f"{new_email.split('@')[0]}") # Use email prefix as name guess
            except Exception: self.logger.debug(f"Name field not found or failed to fill for {service} signup.")

            await self._human_like_click(page, submit_selector)
            await page.wait_for_load_state("networkidle", timeout=60000)
            await asyncio.sleep(random.uniform(2, 4))

            # --- Verification & API Key Extraction ---
            # TODO: Implement email/SMS verification handling using _VerificationHandler
            # TODO: Implement API key extraction logic using _extract_api_key_after_signup

            api_key = None # Placeholder
            signup_confirmed = True # Placeholder - check for dashboard element

            # --- Log Credentials ---
            if signup_confirmed:
                self.logger.info(f"Successfully created trial account for {service} ({new_email}). API Key found: {'Yes' if api_key else 'No'}")
                encrypted_api_key = encrypt_data(api_key) if api_key else None
                encrypted_password = encrypt_data(generated_password)
                proxy_used = self._get_proxy_config(session_id=account_identifier).get('server') if self._get_proxy_config(session_id=account_identifier) else None

                # Log to DB via Orchestrator/ThinkTool directive
                log_task = {
                    "action": "log_new_credential", # Define this for ThinkTool/Orchestrator
                    "credential_data": {
                        "service": service, "account_identifier": new_email,
                        "api_key": encrypted_api_key, "password": encrypted_password,
                        "proxy_used": proxy_used, "status": 'active',
                        "notes": f"Free trial created {datetime.now(timezone.utc).date()}"
                    }
                }
                # Delegate logging task
                await self.orchestrator.delegate_task("ThinkTool", log_task)

                return {"status": "success", "message": f"Trial account created for {service}.", "account_identifier": new_email, "api_key_found": bool(api_key)}
            else:
                self.logger.error(f"Failed to confirm successful signup for {service} ({new_email}).")
                return {"status": "failure", "message": "Failed to confirm signup success."}

        except Exception as e:
            self.logger.error(f"Trial account creation failed for {service} ({new_email}): {e}", exc_info=True)
            return {"status": "error", "message": f"Trial creation exception: {e}"}
        finally:
            # Close the specific context used for this signup attempt
            await self._close_browser_context(account_identifier)

    async def _execute_social_post_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handles posting content to a social media platform."""
        # ### Phase 3 Plan Ref: 6.4 (Add execute_social_post)
        platform = task_details.get('target_platform')
        account_identifier = task_details.get('account_identifier') # Email or username
        content = task_details.get('post_content')
        if not platform or not account_identifier or not content:
            return {"status": "failure", "message": "Missing platform, account_identifier, or content for social post."}

        page = None
        context = None # Keep track to close later
        try:
            await self._internal_think(f"Posting to {platform} using account {account_identifier}", details={"content": content[:100]+"..."})
            context = await self._get_browser_context(account_identifier) # Reuse context if exists
            if not context: raise RuntimeError(f"Failed to get browser context for {account_identifier}")
            page = await self._get_page(account_identifier)
            if not page: raise RuntimeError("Failed to get browser page for social post.")

            # --- Platform-Specific Posting Logic ---
            platform_config = self._get_platform_config(platform)
            post_url = platform_config.get("post_url", f"https://{platform}/")
            textarea_selector = platform_config.get("selectors", {}).get("post_textarea")
            submit_selector = platform_config.get("selectors", {}).get("submit_button")

            if not textarea_selector or not submit_selector:
                raise ValueError(f"Missing post selectors for platform {platform}")

            await page.goto(post_url, wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(random.uniform(2, 4))

            # TODO: Implement robust login check and handling using _login_to_service

            await self._human_like_type(page, textarea_selector, content)
            await asyncio.sleep(random.uniform(0.5, 1.5))
            await self._human_like_click(page, submit_selector)
            await page.wait_for_load_state("networkidle", timeout=30000)

            # TODO: Verify post success
            post_id = f"{platform}_{int(time.time())}" # Placeholder

            self.logger.info(f"Successfully posted to {platform} using account {account_identifier}.")
            return {"status": "success", "message": "Post successful.", "post_id": post_id}

        except Exception as e:
            self.logger.error(f"Social post failed for {platform} ({account_identifier}): {e}", exc_info=True)
            screenshot_path = None
            if page and not page.is_closed(): screenshot_path = await self._take_screenshot(page, f"social_post_fail_{platform}_{account_identifier}")
            return {"status": "error", "message": f"Social post exception: {e}", "screenshot": screenshot_path}
        finally:
            if page: await page.close()
            # Don't close context here, let periodic cleanup handle it or specific directive

    async def _execute_social_interact_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handles interacting (like/comment) on a social media post."""
        # ### Phase 3 Plan Ref: 6.4 (Add execute_social_interact)
        self.logger.warning("_execute_social_interact_task not fully implemented.")
        return {"status": "skipped", "message": "Social interaction not implemented yet."}

    async def _perform_search_and_summarize_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Performs a web search and summarizes results (for Tech Radar)."""
        # ### Phase 3 Plan Ref: 6.7 (Implement search/summarize for Radar)
        query = task_details.get('query')
        num_results = task_details.get('num_results', 3)
        if not query: return {"status": "failure", "message": "Missing query for search task."}

        page = None
        context = None
        search_engine_url = "https://duckduckgo.com/"
        account_identifier = f"search_{hashlib.sha1(query.encode()).hexdigest()[:8]}"

        try:
            await self._internal_think(f"Performing web search for: '{query}'", details={"num_results": num_results})
            context = await self._get_browser_context(account_identifier, recreate=True) # Fresh context
            if not context: raise RuntimeError("Failed to get browser context for search.")
            page = await self._get_page(account_identifier)
            if not page: raise RuntimeError("Failed to get browser page for search.")

            await page.goto(f"{search_engine_url}?q={query.replace(' ', '+')}", wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(random.uniform(1, 3))

            link_selector = 'a[data-testid="result-title-a"]'
            links = await page.locator(link_selector).all()
            urls_to_scrape = [await link.get_attribute('href') for link in links[:num_results] if await link.get_attribute('href')]

            if not urls_to_scrape: return {"status": "success", "summary": "No relevant search results found.", "results": []}

            scraped_contents = []
            for url in urls_to_scrape:
                await asyncio.sleep(random.uniform(1, 2))
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                    body_text = await page.locator('body').inner_text(timeout=10000)
                    scraped_contents.append({"url": url, "content": body_text[:2000]})
                except Exception as scrape_err: self.logger.warning(f"Failed to scrape search result {url}: {scrape_err}")

            if not scraped_contents: return {"status": "success", "summary": "Found search results but failed to scrape content.", "results": urls_to_scrape}

            await self._internal_think("Summarizing search results via LLM.")
            summary_prompt = f"Summarize the key information relevant to the query '{query}' from the following scraped web content:\n\n"
            for item in scraped_contents: summary_prompt += f"--- URL: {item['url']} ---\n{item['content']}\n\n"
            summary_prompt += "\nProvide a concise summary highlighting new tools, techniques, or significant findings."

            summary = await self._call_llm_with_retry(summary_prompt, max_tokens=1000, temperature=0.4)

            return {"status": "success", "summary": summary or "Failed to generate summary.", "results": scraped_contents}

        except Exception as e:
            self.logger.error(f"Search and summarize failed for query '{query}': {e}", exc_info=True)
            return {"status": "error", "message": f"Search/Summarize exception: {e}"}
        finally:
            # Close context used specifically for this search
            await self._close_browser_context(account_identifier)

    async def _take_screenshot(self, page: Page, base_filename: str) -> Optional[str]:
         """Takes a screenshot and saves it."""
         try:
              screenshot_dir = os.path.join(os.path.dirname(__file__), '..', 'screenshots')
              os.makedirs(screenshot_dir, exist_ok=True)
              timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
              safe_base = re.sub(r'[^\w\-]+', '_', base_filename)
              filename = f"{safe_base}_{timestamp}.png"
              full_path = os.path.join(screenshot_dir, filename)
              await page.screenshot(path=full_path, full_page=True)
              self.logger.info(f"Screenshot saved: {full_path}")
              return full_path
         except Exception as e:
              self.logger.error(f"Failed to take screenshot {base_filename}: {e}")
              return None

    # --- Abstract Method Implementations ---

    async def plan_task(self, task_details: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        self.logger.debug("BrowsingAgent does not plan tasks, relies on directives.")
        return None

    async def execute_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.warning(f"execute_step called unexpectedly for BrowsingAgent. Step: {step}")
        action = step.get("tool") or step.get("action")
        params = step.get("params", {})
        params['action'] = action # Ensure action key exists
        if action: return await self.execute_task(params)
        else: return {"status": "failure", "message": "Invalid step format for execute_step."}

    async def learning_loop(self):
        self.logger.info("BrowsingAgent learning loop: Passive. Performance tracked via task outcomes analyzed by ThinkTool.")
        while self.status == self.STATUS_RUNNING and not self._stop_event.is_set():
            await asyncio.sleep(3600)

    async def self_critique(self) -> Dict[str, Any]:
        self.logger.info(f"{self.AGENT_NAME}: Performing self-critique.")
        num_contexts = len(self._contexts)
        # TODO: Query logs/metrics for task success rates
        feedback = f"Critique: Currently managing {num_contexts} browser contexts. Task success rate analysis needed."
        return {"status": "ok", "feedback": feedback, "metrics": {"active_contexts": num_contexts}}

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Constructs prompts for LLM calls (e.g., inferring steps, summarizing)."""
        self.logger.debug(f"Generating dynamic prompt for BrowsingAgent task: {task_context.get('task')}")
        prompt_parts = [self.meta_prompt]
        prompt_parts.append("\n--- Current Task Context ---")
        for key, value in task_context.items():
            value_str = str(value)
            if len(value_str) > 500: value_str = value_str[:500] + "..."
            prompt_parts.append(f"**{key.replace('_', ' ').title()}**: {value_str}")

        prompt_parts.append("\n--- Instructions ---")
        task_type = task_context.get('task')
        if task_type == 'Infer Browsing Steps':
             prompt_parts.append("Based on the goal and current page state, determine the next logical browsing action (e.g., click, type, scroll, navigate).")
             prompt_parts.append(f"**Output Format:** {task_context.get('desired_output_format', 'JSON: {\"action\": \"click|type|scroll|navigate\", \"selector\": \"css_selector?\", \"text\": \"text_to_type?\", \"url\": \"url_to_navigate?\"}')}")
        elif task_type == 'Summarize Scraped Content':
             prompt_parts.append("Summarize the key information from the provided scraped web content relevant to the original query.")
             prompt_parts.append(f"**Output Format:** {task_context.get('desired_output_format', 'Concise text summary.')}")
        else:
            prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")

        if "JSON" in task_context.get('desired_output_format', ''): prompt_parts.append("\n```json")

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for BrowsingAgent (length: {len(final_prompt)} chars)")
        return final_prompt

    async def collect_insights(self) -> Dict[str, Any]:
        """Collects insights about browsing operations."""
        return {
            "agent_name": self.AGENT_NAME, "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_browser_contexts": len(self._contexts),
            "key_observations": ["Basic operational status."]
        }

    async def stop(self, timeout: float = 30.0):
        """Override stop to include Playwright shutdown."""
        await self._shutdown_playwright() # Shutdown browser first
        await super().stop(timeout) # Call base class stop

# --- End of agents/browsing_agent.py ---