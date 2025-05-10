# Filename: agents/browsing_agent.py
# Description: Agentic Browsing Agent with fully integrated dynamic temp mail UI automation and deep learning logging.
# Version: 3.8 (Unyielding Detail, FULL CODE)

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
    from.base_agent import GeniusAgentBase
except ImportError:
    logging.warning("Production base agent relative import failed, trying absolute.")
    try:
        from .base_agent import GeniusAgentBase
    except ImportError:
        logging.critical("Failed to import GeniusAgentBase. Check PYTHONPATH or project structure.")
        raise SystemExit("Cannot import GeniusAgentBase - critical dependency missing.")

try:
    from models import KnowledgeFragment, AccountCredentials, StrategicDirective
except ImportError:
     logging.error("Failed to import models from parent directory. Ensure models.py is accessible.")
     class KnowledgeFragment: pass
     class AccountCredentials: pass
     class StrategicDirective: pass

try:
    from config.settings import settings
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
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
op_logger = logging.getLogger('OperationalLog')

BROWSING_AGENT_META_PROMPT = """
You are the BrowsingAgent (Level 40+ Transmuted Entity) within the Nolli AI Sales System.
Your Core Mandate: Execute any web interaction with god-tier adaptability, human-like stealth, and AI-native cunning. This includes flawless visual UI automation, autonomous management and creation of diverse accounts (including dynamic temp mail acquisition via UI exploits), targeted information discovery, and resilient data extraction. You operate with extreme agency and resourcefulness, leaving no detail to chance.
Key Responsibilities:
1.  **Visual UI Domination (Unyielding Detail):** Interact with any web application via its UI. Use advanced visual reasoning (analyzing screenshots via multimodal LLM) as the *primary* method to find elements, determine actions (click, input, scroll, upload, download, navigate, solve CAPTCHAs, handle popups/alerts), and navigate complex flows including dynamic account signup processes. Adapt to UI changes instantly. Every LLM decision for UI interaction must be precise and justified.
2.  **Autonomous Account Lifecycle Management:** Manage a pool of accounts. Assign unique proxies. **Crucially, if a task requires an account for a service (e.g., free trial, platform access) and no suitable active account exists, initiate and execute the automated new account creation process for that service, leveraging dynamic temp mail UI automation.**
3.  **Dynamic Temp Mail UI Exploitation Engine (Fully Realized):** When needing a temporary email for verification, request "temp mail UI exploit scripts" from ThinkTool. Execute these multi-step UI automation scripts against various free temp mail websites to obtain email addresses and retrieve verification links/codes directly from their UI using the internal `_execute_web_ui_sub_task` method. Fall back to direct API (e.g., 1secmail) only if configured and all UI attempts fail.
4.  **Stealth & Anti-Detection Mastery:** Simulate human browsing patterns with high fidelity (variable delays, complex mouse movements, non-linear navigation, realistic fingerprinting via Playwright context options). Dynamically adapt to anti-bot measures based on visual analysis and ThinkTool insights.
5.  **Resilient & Adaptive Execution (Zero Tolerance for Ambiguity):** Gracefully handle UI changes, anti-bot measures, proxy failures, timeouts, login issues, and account creation failures using retries, adaptive strategies driven by visual LLM analysis, and robust error reporting. Log detailed outcomes of all significant UI automation tasks to the Knowledge Base.
6.  **Deep Learning Data Provision:** After UI automation tasks, log comprehensive KFs detailing the service, goal, success, steps, errors, proxy, account, timings, CAPTCHA encounters, and LLM decision points. This data fuels ThinkTool's strategic learning and your own adaptation.
**Goal:** Be the agency's ultimate web operative, capable of any visual UI automation, seamless account creation using UI exploits for temp mail, and stealthy information gathering, all driven by AI-native reasoning, meticulous execution, and a relentless pursuit of task success.
"""

class TempMailService:
    """
    Enhanced TempMailService (L35+):
    Leverages BrowsingAgent's UI automation capabilities to interact with temp mail sites.
    """
    def __init__(self, orchestrator: Any, browsing_agent_instance: 'BrowsingAgent'):
        self.orchestrator = orchestrator
        self.browsing_agent = browsing_agent_instance
        self.faker = Faker()
        self.logger = logging.getLogger(f"{__name__}.TempMailService")
        self.fallback_1secmail_enabled = settings.get("TEMP_MAIL_ENABLE_1SECMAIL_FALLBACK", False)
        self.fallback_1secmail_base_url = "https://www.1secmail.com/api/v1/"

    async def _execute_temp_mail_ui_sub_task(self, sub_task_goal: str, site_url: str, ui_steps_or_direct_goal: Union[List[Dict], str], existing_context_id: Optional[str] = None) -> Dict[str, Any]:
        return await self.browsing_agent._execute_web_ui_sub_task(
            service_name=f"TempMail_{urlparse(site_url).netloc if site_url else 'UnknownSite'}",
            goal=sub_task_goal,
            initial_url=site_url,
            ui_steps_or_direct_goal=ui_steps_or_direct_goal,
            max_sub_steps=20,
            use_existing_context_id=existing_context_id,
            is_temp_mail_operation=True
        )

    async def get_new_email_address(self) -> Optional[Dict[str, str]]:
        think_tool = self.orchestrator.agents.get('think')
        if not think_tool: self.logger.error("ThinkTool unavailable for TempMailService."); return None

        exploit_fragments = await think_tool.query_knowledge_base(
            data_types=['temp_mail_exploit_script'], tags=['temp_mail', 'working_get_email'], limit=3, min_relevance=0.75
        )
        if exploit_fragments:
            random.shuffle(exploit_fragments)
            for fragment in exploit_fragments:
                try:
                    script_data = json.loads(fragment.content)
                    script_name = script_data.get("script_name", f"KF_{fragment.id}_get_email")
                    initial_url = script_data.get("site_url")
                    ui_goal_for_sub_task = script_data.get("goal_to_get_email", f"Navigate {initial_url} and visually identify and extract a newly generated temporary email address. Store it as 'email_address'.")
                    
                    if initial_url:
                        self.logger.info(f"Attempting to get temp email via UI script: {script_name} on {initial_url}")
                        result = await self._execute_temp_mail_ui_sub_task(
                            sub_task_goal=ui_goal_for_sub_task,
                            site_url=initial_url,
                            ui_steps_or_direct_goal=ui_goal_for_sub_task
                        )
                        if result.get("status") == "success" and result.get("extracted_data", {}).get("email_address"):
                            email_address = result["extracted_data"]["email_address"]
                            self.logger.info(f"Successfully obtained temp email via UI: {email_address} using script {script_name}")
                            return {
                                "email": email_address,
                                "context_id_for_checking": result.get("context_id"),
                                "method": "ui_automation",
                                "script_name_used": script_name,
                                "site_url_used": initial_url
                            }
                        else:
                            self.logger.warning(f"UI script {script_name} failed to get email: {result.get('message')}")
                except Exception as e: self.logger.error(f"Error processing temp mail exploit script {fragment.id}: {e}")

        if self.fallback_1secmail_enabled:
            self.logger.info("UI temp mail failed or no scripts. Falling back to 1secmail API.")
            try:
                async with aiohttp.ClientSession() as session:
                     local_part = self.faker.user_name() + str(random.randint(1000,9999))
                     async with session.get(f"{self.fallback_1secmail_base_url}?action=getDomainList") as response:
                         if response.status == 200:
                             domains = await response.json()
                             if domains and isinstance(domains, list) and len(domains) > 0:
                                 domain = random.choice(domains)
                                 email = f"{local_part}@{domain}"
                                 self.logger.info(f"Generated temporary email via 1secmail API: {email}")
                                 return {"email": email, "context_id_for_checking": None, "method": "api_fallback_1secmail"}
                             else: self.logger.error(f"1secmail API returned no domains or invalid format: {domains}")
                         else: self.logger.error(f"Failed to get domains from 1secmail API. Status: {response.status}, Body: {await response.text()}")
                         return None
            except Exception as e:
                self.logger.error(f"Error generating temp email via 1secmail API: {e}", exc_info=True)
        else:
            self.logger.warning("No working temp mail UI scripts and 1secmail fallback is disabled.")
        return None

    async def get_verification_link(self, email_info: Dict[str, str], timeout_seconds: int = 240, keyword: str = "verify") -> Optional[str]:
        email_address = email_info.get("email")
        method = email_info.get("method")
        context_id = email_info.get("context_id_for_checking")
        site_url = email_info.get("site_url_used")

        if not email_address: return None
        think_tool = self.orchestrator.agents.get('think')

        if method == "ui_automation" and context_id and site_url and think_tool:
            self.logger.info(f"Attempting to get verification link for {email_address} via UI automation (context: {context_id}, site: {site_url}).")
            ui_goal_for_sub_task = f"On the temp mail site {site_url} (currently in context for {email_address}), locate the inbox for this email. Find an email containing the keyword '{keyword}' or a clear verification/confirmation link/button. Extract the full URL of that link and store it as 'verification_link'."
            
            result = await self._execute_temp_mail_ui_sub_task(
                sub_task_goal=ui_goal_for_sub_task,
                site_url=site_url,
                ui_steps_or_direct_goal=ui_goal_for_sub_task,
                existing_context_id=context_id
            )
            if result.get("status") == "success" and result.get("extracted_data", {}).get("verification_link"):
                link = result["extracted_data"]["verification_link"]
                self.logger.info(f"Successfully obtained verification link via UI: {link}")
                return link
            else:
                self.logger.warning(f"UI automation failed to get verification link for {email_address}: {result.get('message')}")
                return None
        elif method == "api_fallback_1secmail":
            if '@' not in email_address: return None
            local_part, domain = email_address.split('@', 1)
            start_time = time.time()
            self.logger.info(f"Polling 1secmail API for {email_address} for verification link (keyword: '{keyword}')...")
            while time.time() - start_time < timeout_seconds:
                try:
                    async with aiohttp.ClientSession() as session:
                        url = f"{self.fallback_1secmail_base_url}?action=getMessages&login={local_part}&domain={domain}"
                        async with session.get(url) as response:
                            if response.status == 200:
                                messages = await response.json()
                                if messages and isinstance(messages, list):
                                    for message_header in reversed(messages):
                                        message_id = message_header.get('id')
                                        msg_url = f"{self.fallback_1secmail_base_url}?action=readMessage&login={local_part}&domain={domain}&id={message_id}"
                                        async with session.get(msg_url) as msg_response:
                                            if msg_response.status == 200:
                                                message_content = await msg_response.json()
                                                body = message_content.get('htmlBody') or message_content.get('textBody', '')
                                                link_patterns = [
                                                    rf'href=["\'](https?://[^\s"\'<>]*{re.escape(keyword)}[^\s"\'<>]+)["\']',
                                                    rf'(https?://[^\s"\'<>]*{re.escape(keyword)}[^\s"\'<>]+)',
                                                    r'https?://[a-zA-Z0-9.-]*\.?service-provider\.com/verify\?[^ \n\r"\']+'
                                                ]
                                                found_link = None
                                                for pat in link_patterns:
                                                    match = re.search(pat, body, re.IGNORECASE)
                                                    if match: found_link = match.group(1); break
                                                if found_link:
                                                    link = unquote(found_link.replace('&amp;', '&'))
                                                    self.logger.info(f"Found verification link in 1secmail {message_id}: {link}")
                                                    return link
                                else: self.logger.debug(f"No messages found yet for {email_address} via 1secmail API.")
                            else: self.logger.warning(f"Failed to get messages for {email_address} via 1secmail API. Status: {response.status}, Body: {await response.text()}")
                except Exception as e: self.logger.error(f"Error polling 1secmail API: {e}", exc_info=True)
                await asyncio.sleep(20)
            self.logger.warning(f"Timeout waiting for verification link for {email_address} via 1secmail API.")
            return None
        else:
            self.logger.error(f"Unknown temp mail method '{method}' for {email_address}. Cannot get verification link.")
            return None

# --- BrowsingAgent Class ---
class BrowsingAgent(GeniusAgentBase):
    AGENT_NAME = "BrowsingAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any, smartproxy_password: Optional[str] = None):
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, session_maker=session_maker, config=getattr(orchestrator, 'config', settings))
        self.meta_prompt = BROWSING_AGENT_META_PROMPT
        self.think_tool = orchestrator.agents.get('think')

        self.internal_state = getattr(self, 'internal_state', {})
        self.internal_state['playwright_instance'] = None
        self.internal_state['browser_instance'] = None
        self.internal_state['browser_contexts'] = {}
        self.internal_state['active_pages'] = {}
        self.internal_state['context_account_map'] = {}
        self.internal_state['context_proxy_map'] = {}
        self.internal_state['proxy_stats'] = {}
        self.internal_state['user_agent'] = self.config.get("BROWSER_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")
        self.internal_state['max_concurrent_contexts'] = int(self.config.get("BROWSER_MAX_CONCURRENT_PAGES", 2))
        self.internal_state['context_semaphore'] = asyncio.Semaphore(self.internal_state['max_concurrent_contexts'])
        self.internal_state['default_timeout_ms'] = int(self.config.get("BROWSER_DEFAULT_TIMEOUT_MS", 100000))
        self.temp_dir = self.config.get("TEMP_DOWNLOAD_DIR", "/app/temp_downloads")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.faker = Faker()
        self.temp_mail_service = TempMailService(orchestrator=self.orchestrator, browsing_agent_instance=self)

        self.logger.info(f"{self.AGENT_NAME} v3.8 (L40+ Unyielding Detail) initialized. Max Contexts: {self.internal_state['max_concurrent_contexts']}")

    async def log_operation(self, level: str, message: str):
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err: logger.error(f"Failed to write to OP log from {self.agent_name}: {log_err}")

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
                 headless=self.config.get("BROWSER_HEADLESS", True),
                 args=[
                     '--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu',
                     '--disable-dev-shm-usage', '--disable-blink-features=AutomationControlled',
                     '--window-size=1920,1080'
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
            if not proxy_info:
                proxy_purpose = f"context_{account_details.get('service', 'general')}_{account_details.get('id', 'new')}" if account_details else "general_browsing_context"
                if hasattr(self.orchestrator, 'get_proxy'):
                    proxy_info = await self.orchestrator.get_proxy(purpose=proxy_purpose, quality_level='high_stealth')
                else: self.logger.error("Orchestrator missing get_proxy method."); proxy_info = None

            playwright_proxy = None
            if proxy_info and proxy_info.get('server'):
                 playwright_proxy = { "server": proxy_info["server"], "username": proxy_info.get("username"), "password": proxy_info.get("password") }
                 self.logger.info(f"Using proxy {proxy_info['server'].split('@')[-1] if '@' in proxy_info['server'] else proxy_info['server']} for new context {context_id}")
            else: self.logger.warning(f"No valid proxy for new context {context_id}. Proceeding without proxy (higher risk).")

            browser = self.internal_state['browser_instance']
            if not browser or not browser.is_connected(): raise RuntimeError("Browser instance unavailable.")

            context = await browser.new_context(
                user_agent=self.internal_state['user_agent'], proxy=playwright_proxy,
                java_script_enabled=True, ignore_https_errors=True,
                viewport={'width': random.randint(1400, 1920), 'height': random.randint(800, 1080)},
                locale=random.choice(['en-US', 'en-GB', 'en-CA']),
                timezone_id=random.choice(['America/New_York', 'America/Chicago', 'America/Los_Angeles', 'Europe/London', 'Europe/Berlin']),
                permissions=['geolocation', 'notifications'],
                geolocation=self._generate_random_geo(),
                bypass_csp=True
            )
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => false });
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5].map(i => ({name: `Plugin ${i}`, filename: `plugin${i}.dll`, description: `Description for plugin ${i}`})) });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                // Further fingerprinting randomization can be added here (e.g., WebGL, Canvas, AudioContext)
            """)

            self.internal_state['browser_contexts'][context_id] = context
            if account_details: self.internal_state['context_account_map'][context_id] = account_details
            if proxy_info: self.internal_state['context_proxy_map'][context_id] = proxy_info
            self.logger.info(f"Created new browser context (ID: {context_id}). Active: {len(self.internal_state['browser_contexts'])}")
            return context_id, context
        except Exception as e:
            self.internal_state['context_semaphore'].release()
            self.logger.error(f"Failed to create new browser context: {e}", exc_info=True)
            if context and hasattr(context, 'close'): await self._close_context(context_id, context)
            raise

    def _generate_random_geo(self) -> Dict:
        locations = [
            {'latitude': 40.7128, 'longitude': -74.0060, 'accuracy': random.uniform(20, 100)}, # NYC
            {'latitude': 34.0522, 'longitude': -118.2437, 'accuracy': random.uniform(20, 100)}, # LA
            {'latitude': 51.5074, 'longitude': -0.1278, 'accuracy': random.uniform(20, 100)}, # London
            {'latitude': 48.8566, 'longitude': 2.3522, 'accuracy': random.uniform(20, 100)}, # Paris
            {'latitude': 35.6895, 'longitude': 139.6917, 'accuracy': random.uniform(20, 100)}, # Tokyo
            {'latitude': -33.8688, 'longitude': 151.2093, 'accuracy': random.uniform(20, 100)}, # Sydney
        ]
        return random.choice(locations)

    async def _close_context(self, context_id: str, context: Optional[BrowserContext] = None):
        context_to_close = context or self.internal_state['browser_contexts'].get(context_id)
        semaphore_released_in_this_call = False
        if context_to_close:
            try:
                pages_to_remove = [pid for pid, pdata in self.internal_state['active_pages'].items() if pdata['context_id'] == context_id]
                for pid in pages_to_remove:
                    page_obj = self.internal_state['active_pages'].pop(pid, {}).get('page')
                    if page_obj and not page_obj.is_closed():
                        try: await page_obj.close(run_before_unload=True)
                        except Exception as page_close_err: logger.warning(f"Error closing page {pid} in context {context_id}: {page_close_err}")

                if hasattr(context_to_close, 'is_closed') and not context_to_close.is_closed(): # Check if already closed
                    await context_to_close.close()
                    self.logger.info(f"Closed browser context (ID: {context_id}).")
            except Exception as e: self.logger.warning(f"Error closing context {context_id}: {e}")
            finally:
                self.internal_state['browser_contexts'].pop(context_id, None)
                self.internal_state['context_account_map'].pop(context_id, None)
                self.internal_state['context_proxy_map'].pop(context_id, None)
                try:
                    if self.internal_state['context_semaphore']._value < self.internal_state['max_concurrent_contexts']:
                         self.internal_state['context_semaphore'].release()
                         semaphore_released_in_this_call = True
                except (ValueError, AttributeError): pass 
                self.logger.debug(f"Context cleanup complete for {context_id}. Active contexts: {len(self.internal_state['browser_contexts'])}. Semaphore released now: {semaphore_released_in_this_call}")
        else:
             cleaned_from_maps = False
             if self.internal_state['browser_contexts'].pop(context_id, None): cleaned_from_maps = True
             if self.internal_state['context_account_map'].pop(context_id, None): cleaned_from_maps = True
             if self.internal_state['context_proxy_map'].pop(context_id, None): cleaned_from_maps = True
             if cleaned_from_maps:
                 self.logger.warning(f"Context object for ID {context_id} not found, but removed from internal maps.")
                 try:
                     if self.internal_state['context_semaphore']._value < self.internal_state['max_concurrent_contexts']:
                          self.internal_state['context_semaphore'].release()
                          semaphore_released_in_this_call = True
                 except (ValueError, AttributeError) : pass
                 self.logger.debug(f"Semaphore released after cleaning maps for {context_id}: {semaphore_released_in_this_call}")

    async def _get_page_for_context(self, context_id: str) -> Tuple[str, Page]:
        context = self.internal_state['browser_contexts'].get(context_id)
        if not context: raise ValueError(f"Context ID {context_id} not found for page retrieval.")
        try:
            if hasattr(context, 'is_closed') and context.is_closed(): raise PlaywrightError(f"Context {context_id} is already closed.")
        except AttributeError: raise ValueError(f"Context ID {context_id} is invalid or None.")

        for pid, pdata in self.internal_state['active_pages'].items():
            if pdata['context_id'] == context_id and pdata['page'] and not pdata['page'].is_closed():
                self.logger.debug(f"Reusing existing page {pid} for context {context_id}")
                return pid, pdata['page']

        page_id = f"page_{uuid.uuid4().hex[:8]}"
        page = await context.new_page()
        page.set_default_timeout(self.internal_state['default_timeout_ms'])
        self.internal_state['active_pages'][page_id] = {'page': page, 'context_id': context_id}
        self.logger.info(f"Created new page (ID: {page_id}) within context {context_id}. Total active pages: {len(self.internal_state['active_pages'])}")
        return page_id, page

    async def _close_browser_and_playwright(self):
        self.logger.info("Closing all browser contexts and stopping Playwright...")
        context_ids = list(self.internal_state['browser_contexts'].keys())
        if context_ids:
            for cid in context_ids: await self._close_context(cid)
        self.internal_state['browser_contexts'].clear()
        self.internal_state['active_pages'].clear()
        self.internal_state['context_account_map'].clear()
        self.internal_state['context_proxy_map'].clear()

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

    async def _update_proxy_stats(self, context_id: Optional[str], success: bool):
        if not context_id: return
        proxy_info = self.internal_state['context_proxy_map'].get(context_id)
        if not proxy_info or not proxy_info.get('server'): return
        proxy_url = proxy_info['server']
        now = datetime.now(timezone.utc)
        if proxy_url not in self.internal_state['proxy_stats']: self.internal_state['proxy_stats'][proxy_url] = {'success': 0, 'failure': 0, 'last_used': now, 'first_used': now, 'associated_context_ids': set()}
        self.internal_state['proxy_stats'][proxy_url]['associated_context_ids'].add(context_id)
        if success: self.internal_state['proxy_stats'][proxy_url]['success'] += 1
        else: self.internal_state['proxy_stats'][proxy_url]['failure'] += 1
        self.internal_state['proxy_stats'][proxy_url]['last_used'] = now

    async def plan_task(self, task_details: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        action = task_details.get('action')
        plan = []
        await self._internal_think(f"Received task: {action}. Planning execution.", details=task_details)
        if action == 'scrape_website':
            url = task_details.get('url')
            if not url: raise ValueError("Missing 'url' for scrape_website task.")
            plan.append({"step": 1, "action": "open_page", "tool": "browser", "params": {"url": url}})
            plan.append({"step": 2, "action": "extract_data", "tool": "browser", "params": task_details})
            plan.append({"step": 3, "action": "close_context", "tool": "browser", "params": {}})
        elif action == 'google_dork_scan':
            target = task_details.get('target')
            if not target: raise ValueError("Missing target for google_dork_scan")
            dork_types = task_details.get('dork_types', ['login', 'password', 'config_files', 'api_keys'])
            plan.append({"step": 1, "action": "generate_dorks", "tool": "internal", "params": {"target": target, "types": dork_types}})
            plan.append({"step": 2, "action": "execute_dork_searches", "tool": "browser", "params": {}})
            plan.append({"step": 3, "action": "analyze_dork_results", "tool": "browser", "params": {}})
            plan.append({"step": 4, "action": "close_context", "tool": "browser", "params": {}})
        elif action == 'web_ui_automate':
             goal = task_details.get('goal', 'Unknown UI Goal')
             service = task_details.get('service', 'Unknown Service')
             plan.append({"step": 1, "action": f"Execute UI Automation: {service} - {goal}", "tool": "browser", "params": task_details})
             plan.append({"step": 2, "action": "close_context", "tool": "browser", "params": {"reason": "UI Automation Complete"}})
        else:
            self.logger.info(f"No specific plan generated by BrowsingAgent for action '{action}'. Will attempt direct execution via execute_step.")
            plan.append({"step": 1, "action": action, "tool": "browser", "params": task_details})
            plan.append({"step": 2, "action": "close_context", "tool": "browser", "params": {"reason": f"Task '{action}' Complete"}})
        self.logger.info(f"Generated plan with {len(plan)} steps for action '{action}'.")
        return plan

    async def _execute_web_ui_sub_task(self, service_name: str, goal: str, initial_url: Optional[str] = None, ui_steps_or_direct_goal: Union[List[Dict], str] = "Achieve specified goal", max_sub_steps: int = 15, use_existing_context_id: Optional[str] = None, is_temp_mail_operation: bool = False) -> Dict[str, Any]:
        sub_task_id = f"sub_{uuid.uuid4().hex[:6]}"
        self.logger.info(f"Executing UI sub-task '{sub_task_id}': Service='{service_name}', Goal='{goal}'")
        final_result = {"status": "failure", "message": "Sub-task failed initialization.", "extracted_data": {}, "context_id": use_existing_context_id}
        
        context_id = use_existing_context_id
        page_id = None
        page: Optional[Page] = None
        created_new_context_for_subtask = False
        sub_task_start_time = time.time()
        log_content_for_kb_sub: Dict[str, Any] = {
            "sub_task_id": sub_task_id, "service": service_name, "goal_summary": goal[:100],
            "timestamp_start": datetime.now(timezone.utc).isoformat(),
            "is_temp_mail_op": is_temp_mail_operation
        }

        try:
            if context_id:
                context_obj = self.internal_state['browser_contexts'].get(context_id)
                if not context_obj or (hasattr(context_obj, 'is_closed') and context_obj.is_closed()):
                    self.logger.warning(f"Provided context {context_id} for sub-task is closed/invalid. Creating new one.")
                    await self._close_context(context_id) # Ensure old one is cleaned if it was in our map
                    context_id = None
                else:
                    try:
                        active_page_for_context = next((pdata['page'] for pid, pdata in self.internal_state['active_pages'].items() if pdata['context_id'] == context_id and not pdata['page'].is_closed()), None)
                        if active_page_for_context:
                            page = active_page_for_context
                            page_id = next(pid for pid, pdata in self.internal_state['active_pages'].items() if pdata['page'] == page)
                            self.logger.debug(f"Reusing page {page_id} in context {context_id} for sub-task.")
                        else:
                            page_id, page = await self._get_page_for_context(context_id)
                        
                        if initial_url and page.url != initial_url:
                            self.logger.info(f"Sub-task navigating page {page_id} to initial_url: {initial_url}")
                            await page.goto(initial_url, wait_until='domcontentloaded', timeout=45000)
                    except Exception as page_err:
                        self.logger.warning(f"Failed to get/use page in existing context {context_id} for sub-task: {page_err}. Creating new context.")
                        if page_id and page_id in self.internal_state['active_pages']: del self.internal_state['active_pages'][page_id]
                        context_id = None; page = None; page_id = None
            if not context_id:
                proxy_purpose = f"sub_task_{service_name}_{goal[:20].replace(' ','_')}"
                proxy_quality = 'standard_plus' if is_temp_mail_operation else 'high_stealth'
                proxy = await self.orchestrator.get_proxy(purpose=proxy_purpose, quality_level=proxy_quality)
                context_id, _ = await self._get_new_context(proxy_info=proxy)
                page_id, page = await self._get_page_for_context(context_id)
                created_new_context_for_subtask = True
                if initial_url:
                    await page.goto(initial_url, wait_until='domcontentloaded', timeout=60000)
            
            final_result["context_id"] = context_id
            log_content_for_kb_sub["context_id_used"] = context_id
            proxy_details_sub = self.internal_state['context_proxy_map'].get(context_id)
            log_content_for_kb_sub["proxy_used_details"] = proxy_details_sub.get('server','N/A').split('@')[-1] if proxy_details_sub and proxy_details_sub.get('server') else "None"


            current_step_count = 0; last_action_summary = "Sub-task UI automation started."
            automation_successful = False; extracted_sub_task_data = {}
            log_content_for_kb_sub["visual_llm_decisions"] = 0
            micro_steps_to_execute = []

            if isinstance(ui_steps_or_direct_goal, str): # LLM needs to plan micro-steps
                initial_planning_context = {
                    "task": "Plan micro-steps for UI sub-task", "service_context": service_name, "overall_sub_task_goal": ui_steps_or_direct_goal,
                    "current_url": page.url, "page_text_content_snippet": (await page.evaluate("document.body.innerText"))[:2000],
                    "desired_output_format": "JSON list of micro-action objects: [{\"action_description\": \"brief description of micro-step for LLM to execute next (e.g., 'Click the generate email button', 'Copy the displayed email text')\"}] - Keep steps very small and precise."
                }
                planning_prompt = await self.generate_dynamic_prompt(initial_planning_context)
                screenshot_for_planning = await page.screenshot(full_page=True)
                planned_steps_json = await self.orchestrator.call_llm(agent_name=self.AGENT_NAME, prompt=planning_prompt, temperature=0.1, max_tokens=1000, is_json_output=True, image_data=screenshot_for_planning)
                if planned_steps_json:
                    parsed_plan = self._parse_llm_json(planned_steps_json, expect_type=list)
                    if parsed_plan and isinstance(parsed_plan, list): micro_steps_to_execute = parsed_plan
                    else: self.logger.warning(f"Sub-task {sub_task_id}: LLM failed to provide valid micro-steps. Using overall goal."); micro_steps_to_execute = [{"action_description": ui_steps_or_direct_goal}]
                else: micro_steps_to_execute = [{"action_description": ui_steps_or_direct_goal}]
            elif isinstance(ui_steps_or_direct_goal, list): micro_steps_to_execute = ui_steps_or_direct_goal
            
            if not micro_steps_to_execute : micro_steps_to_execute = [{"action_description": goal}] # Ensure there's at least one step (the overall goal)

            self.logger.info(f"Sub-task {sub_task_id}: Beginning execution of {len(micro_steps_to_execute)} micro-steps.")

            while current_step_count < max_sub_steps and current_step_count < len(micro_steps_to_execute):
                current_micro_goal_desc = micro_steps_to_execute[current_step_count].get("action_description", goal)
                await self._internal_think(f"UI Sub-Task {sub_task_id} Micro-Step {current_step_count + 1}/{len(micro_steps_to_execute)}: Goal='{current_micro_goal_desc}'", details={"last_action_result": last_action_summary})
                await self._human_like_delay(page, short=True)

                screenshot_bytes = await page.screenshot(full_page=True)
                page_text_content = await page.evaluate("document.body.innerText")
                current_url = page.url
                log_content_for_kb_sub["visual_llm_decisions"] +=1

                automation_task_context = {
                    "task": "Determine next UI action for current micro-step of sub-task",
                    "service_context": service_name, "current_micro_goal": current_micro_goal_desc, "overall_sub_task_goal": goal,
                    "page_text_content_snippet": page_text_content[:4000], "current_url": current_url,
                    "last_action_result": last_action_summary, "current_step_in_sub_task": f"{current_step_count + 1}/{len(micro_steps_to_execute)}",
                    "available_actions": ["click", "input", "scroll", "wait", "extract_text", "finish_current_micro_goal", "finish_entire_sub_task", "error_sub_task"],
                    "desired_output_format": "JSON: {\"action_type\": \"<chosen_action>\", \"selector\": \"<css_selector_or_xpath_or_visual_desc>\", \"coordinates\": {\"x\": float, \"y\": float}?, \"text_to_input\": \"<string>\"?, \"extraction_target_description\": \"<if extract_text, what to extract, e.g., 'the email address displayed'>\"?, \"output_variable_name\": \"<if extract_text, name to store result, e.g., 'email_address' or 'verification_link'>\"?, \"reasoning\": \"<Explanation>\", \"error_message\": \"<If error_sub_task>\"}"
                }
                llm_prompt = await self.generate_dynamic_prompt(automation_task_context)
                action_json_str = await self.orchestrator.call_llm(
                    agent_name=self.AGENT_NAME, prompt=llm_prompt, temperature=0.05, max_tokens=1000, is_json_output=True, image_data=screenshot_bytes
                )
                if not action_json_str: last_action_summary = "LLM failed to determine next sub-task action."; current_step_count += 1; continue
                try:
                    action_data = self._parse_llm_json(action_json_str)
                    if not action_data: raise ValueError("Failed to parse LLM action JSON for sub-task.")
                    action_type = action_data.get('action_type')
                    reasoning = action_data.get('reasoning', 'N/A')
                    self.logger.info(f"Sub-Task {sub_task_id} Micro-Step {current_step_count + 1}: LLM decided action: {action_type}. Reasoning: {reasoning}")
                    action_executed_successfully = False; action_result_message = f"Sub-task action '{action_type}' initiated."

                    if action_type == 'click':
                        selector = action_data.get('selector'); coords = action_data.get('coordinates')
                        target_element = await self._find_element(page, selector, coords, visual_description_if_no_selector=selector if not coords else None)
                        if target_element: await self._human_click(page, target_element); action_executed_successfully = True; action_result_message = f"Clicked element: '{selector or coords or 'visual_target'}'."
                        else: action_result_message = f"Click failed: Element not found for '{selector or coords or 'visual_target'}'."
                    elif action_type == 'input':
                        selector = action_data.get('selector'); coords = action_data.get('coordinates')
                        text_to_input = action_data.get('text_to_input', "")
                        target_element = await self._find_element(page, selector, coords, visual_description_if_no_selector=selector if not coords else None)
                        if target_element:
                             await self._human_fill(page, target_element, str(text_to_input))
                             action_executed_successfully = True; action_result_message = f"Input text into element '{selector or coords or 'visual_target'}'."
                        else: action_result_message = f"Input failed: Element not found for '{selector or coords or 'visual_target'}'."
                    elif action_type == 'extract_text':
                        selector = action_data.get('selector'); coords = action_data.get('coordinates')
                        target_description = action_data.get('extraction_target_description', 'target_text_value')
                        output_var_name = action_data.get('output_variable_name', target_description.lower().replace(" ", "_"))
                        element_to_extract = await self._find_element(page, selector, coords, visual_description_if_no_selector=selector if not coords else target_description)
                        if element_to_extract:
                            extracted_text = await element_to_extract.text_content(timeout=10000) or await element_to_extract.inner_text(timeout=10000) or await element_to_extract.input_value(timeout=10000)
                            if extracted_text is not None:
                                extracted_sub_task_data[output_var_name] = extracted_text.strip()
                                action_executed_successfully = True; action_result_message = f"Extracted text for '{target_description}' as '{output_var_name}': '{extracted_text.strip()[:50]}...'."
                            else: action_result_message = f"Extraction failed: Element found but no text for '{target_description}'."
                        else: action_result_message = f"Extraction failed: Element not found for '{selector or target_description}'."
                    elif action_type == 'scroll':
                        direction = action_data.get('scroll_direction', 'down'); await self._human_scroll(page, direction); action_executed_successfully = True; action_result_message = f"Scrolled {direction}."
                    elif action_type == 'wait':
                        wait_ms = action_data.get('wait_time_ms', random.randint(1000, 3000)); await page.wait_for_timeout(wait_ms); action_executed_successfully = True; action_result_message = f"Waited for {wait_ms} ms."
                    elif action_type == 'finish_current_micro_goal':
                        action_executed_successfully = True; action_result_message = f"Micro-goal '{current_micro_goal_desc}' deemed complete."
                        current_step_count += 1; continue
                    elif action_type == 'finish_entire_sub_task':
                        automation_successful = True; action_executed_successfully = True
                        action_result_message = f"Entire sub-task goal '{goal}' deemed complete by LLM."
                        self.logger.info(action_result_message); break
                    elif action_type == 'error_sub_task':
                        error_msg = action_data.get('error_message', 'LLM indicated unrecoverable error for sub-task.')
                        self.logger.error(f"LLM reported error during UI sub-task: {error_msg}")
                        raise RuntimeError(f"UI Sub-task failed: {error_msg}")
                    else: action_result_message = f"Unsupported action type for sub-task: {action_type}"
                    
                    last_action_summary = action_result_message
                    if not action_executed_successfully:
                        self.logger.warning(f"Sub-Task {sub_task_id} Micro-Step {current_step_count + 1}: Action '{action_type}' failed. Msg: {action_result_message}")
                        raise RuntimeError(f"UI Sub-task micro-step failed: {action_type}. Msg: {last_action_summary}")
                    current_step_count += 1
                except Exception as auto_err:
                    last_action_summary = f"Unexpected error executing sub-task action '{action_data.get('action_type', 'unknown') if 'action_data' in locals() else 'unknown'}': {auto_err}"
                    self.logger.error(f"Sub-Task {sub_task_id} Micro-Step {current_step_count + 1}: {last_action_summary}", exc_info=True)
                    raise RuntimeError(f"UI Sub-task failed due to error: {auto_err}") from auto_err

            log_content_for_kb_sub["steps_taken_count"] = current_step_count
            if automation_successful:
                final_result["status"] = "success"
                final_result["message"] = f"Sub-task '{goal}' completed successfully."
                final_result["extracted_data"] = extracted_sub_task_data
            else:
                final_result["message"] = f"Sub-task '{goal}' reached max steps ({max_sub_steps}) or exhausted micro-steps without explicit 'finish_entire_sub_task'. Last action: {last_action_summary}"
                final_result["extracted_data"] = extracted_sub_task_data # Return any partial data
            log_content_for_kb_sub["success_status"] = automation_successful
            log_content_for_kb_sub["final_url_or_state"] = page.url if page else "Page unavailable"

        except Exception as e:
            self.logger.error(f"Critical error during UI sub-task '{sub_task_id}': {e}", exc_info=True)
            final_result["message"] = f"Critical error in sub-task: {e}"
            final_result["extracted_data"] = extracted_sub_task_data
            log_content_for_kb_sub.update({"status_final": "failure", "error_type": type(e).__name__, "error_message": str(e)})
        finally:
            log_content_for_kb_sub["timestamp_end"] = datetime.now(timezone.utc).isoformat()
            log_content_for_kb_sub["duration_seconds"] = round(time.time() - sub_task_start_time, 2)
            if self.think_tool:
                await self.think_tool.execute_task({
                    "action": "log_knowledge_fragment", "fragment_data": {
                        "agent_source": self.AGENT_NAME, "data_type": "browsing_ui_sub_task_outcome",
                        "content": log_content_for_kb_sub,
                        "tags": [self.AGENT_NAME, "sub_task", service_name, final_result.get("status", "unknown")],
                        "relevance_score": 0.65 if final_result.get("status") == "success" else 0.45 }})
            if created_new_context_for_subtask and context_id and not use_existing_context_id:
                await self._close_context(context_id)
        return final_result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.2, min=3, max=45), retry=retry_if_exception_type(PlaywrightTimeoutError), reraise=True)
    async def execute_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        step_action = step.get('action')
        tool = step.get('tool')
        params = step.get('params', {}).copy()
        step_num = step.get('step', '?')
        step_start_time = time.time()
        result = {"status": "failure", "message": f"Step action '{step_action}' not implemented or failed initialization."}
        
        log_content_for_kb: Dict[str, Any] = {
            "task_id": task_context.get("id", "N/A"), "step_num": step_num, "action_planned": step_action, "tool_planned": tool,
            "service_target": params.get("service", params.get("target_platform", "UnknownService")),
            "goal_summary": params.get("goal", step_action)[:150],
            "timestamp_start": datetime.now(timezone.utc).isoformat(),
            "params_input": {k: (v[:100]+"..." if isinstance(v,str) and len(v)>100 else v) for k,v in params.items() if k != 'account_password'}
        }

        context_id = task_context.get("current_context_id")
        page_id = task_context.get("current_page_id")
        page: Optional[Page] = None
        context_obj: Optional[BrowserContext] = None # Define context_obj

        try:
            context_obj, page, context_id, page_id = await self._ensure_context_and_page(step, task_context)
            task_context["current_context_id"] = context_id
            task_context["current_page_id"] = page_id
            log_content_for_kb["context_id_used"] = context_id
            proxy_details = self.internal_state['context_proxy_map'].get(context_id)
            log_content_for_kb["proxy_used_details"] = proxy_details.get('server','N/A').split('@')[-1] if proxy_details and proxy_details.get('server') else "None"
            account_ctx_details = self.internal_state['context_account_map'].get(context_id, {})
            log_content_for_kb["account_id_used"] = account_ctx_details.get('id')
            log_content_for_kb["account_identifier_used"] = account_ctx_details.get('identifier')

            await self._internal_think(f"Executing step {step_num}: {step_action} in context {context_id}", details=params)

            if tool == 'browser':
                if step_action == 'open_page':
                    url = params.get('url')
                    if not url: raise ValueError("Missing 'url' parameter for open_page.")
                    self.logger.info(f"Navigating page {page_id} (Context: {context_id}) to URL: {url}")
                    response = await page.goto(url, wait_until='domcontentloaded', timeout=self.internal_state['default_timeout_ms'])
                    status_code = response.status if response else None
                    self.logger.info(f"Page {page_id} navigated to {url}. Status: {status_code}, Final URL: {page.url}")
                    if not response or not response.ok:
                        page_content_snippet = (await page.content())[:500]
                        raise PlaywrightError(f"Failed to load page {url}. Status: {status_code}. Content: {page_content_snippet}...")
                    await self._human_like_delay(page)
                    result = {"status": "success", "message": f"Page opened: {url}", "page_id": page_id, "context_id": context_id, "status_code": status_code, "final_url": page.url}
                
                elif step_action == 'extract_data':
                    selectors = params.get('selectors'); extraction_prompt = params.get('extraction_prompt')
                    extract_main = params.get('extract_main_content', False); output_var = params.get('output_var')
                    extracted_data = {}; 
                    if selectors and isinstance(selectors, dict):
                        self.logger.info(f"Extracting data using selectors on page {page_id}.")
                        for key, selector_val in selectors.items():
                            try:
                                element = page.locator(selector_val).first
                                await element.wait_for(state='visible', timeout=15000)
                                content_options = [
                                    await element.text_content(timeout=10000), await element.inner_text(timeout=10000),
                                    await element.input_value(timeout=10000), await element.get_attribute('value', timeout=10000),
                                    await element.get_attribute('href', timeout=10000), await element.get_attribute('src', timeout=10000),
                                    await element.get_attribute('aria-label', timeout=10000)
                                ]
                                content = next((opt for opt in content_options if opt is not None and opt.strip()), "")
                                extracted_data[key] = content.strip()
                            except Exception as e_sel: self.logger.warning(f"Selector '{selector_val}' for key '{key}' failed: {e_sel}"); extracted_data[key] = None
                        result = {"status": "success", "message": "Data extracted via selectors.", "extracted_data": extracted_data}
                    elif extraction_prompt:
                        self.logger.info(f"Extracting data using Visual LLM analysis on page {page_id}.")
                        screenshot_bytes = await page.screenshot(full_page=True); page_text_content = await page.evaluate("document.body.innerText")
                        llm_task_context = {
                            "task": "Extract structured data from webpage using visual and text context",
                            "webpage_text_content": page_text_content[:8000], "extraction_instructions": extraction_prompt,
                            "current_url": page.url, "desired_output_format": "JSON containing the extracted data according to instructions."}
                        llm_prompt_str = await self.generate_dynamic_prompt(llm_task_context)
                        llm_response_str = await self.orchestrator.call_llm(agent_name=self.AGENT_NAME, prompt=llm_prompt_str, temperature=0.05, max_tokens=2500, is_json_output=True, image_data=screenshot_bytes)
                        if llm_response_str:
                             parsed_llm_data = self._parse_llm_json(llm_response_str)
                             if parsed_llm_data is not None: extracted_data = parsed_llm_data; result = {"status": "success", "message": "Data extracted via Visual LLM.", "extracted_data": extracted_data}
                             else: result = {"status": "failure", "message": "LLM returned invalid JSON for extraction.", "raw_output": llm_response_str}
                        else: result = {"status": "failure", "message": "Visual LLM analysis returned no response for extraction."}
                    elif extract_main:
                        self.logger.info(f"Extracting main text content from page {page_id}.")
                        try:
                            body_text = await page.evaluate("document.body.innerText")
                            extracted_data = {"main_text": body_text.strip() if body_text else ""}
                            result = {"status": "success", "message": "Main content extracted.", "extracted_data": extracted_data}
                        except Exception as text_err: result = {"status": "failure", "message": f"Error extracting main text: {text_err}"}
                    else: result = {"status": "warning", "message": "No extraction method specified (selectors, prompt, or main_content)."}
                    if output_var and result.get("status") == "success": task_context[output_var] = extracted_data; result["result_data"] = extracted_data
                
                elif step_action.startswith('Execute UI Automation') or step_action == 'web_ui_automate':
                    automate_goal = params.get('goal', 'Perform UI actions')
                    automate_params = params.get('params', {}).copy() # Specific params for this goal
                    input_vars_map = params.get('input_vars', {})
                    resolved_input_vars = {}
                    for param_name, context_var_name in input_vars_map.items():
                        if isinstance(context_var_name, str) and context_var_name.startswith('$'):
                            var_key = context_var_name[1:]
                            if var_key in task_context: resolved_input_vars[param_name] = task_context[var_key]
                            else: self.logger.warning(f"Input var '{context_var_name}' not found in task_context for UI automation."); resolved_input_vars[param_name] = None
                        else: resolved_input_vars[param_name] = context_var_name # Treat as literal
                    automate_params.update(resolved_input_vars)
                    
                    max_steps_ui = params.get('max_steps', 30); current_step_ui = 0; last_action_summary_ui = "UI Automation started."
                    automation_success = False; ui_automation_extracted_data = {}; log_content_for_kb["visual_llm_decisions"] = 0; log_content_for_kb["captcha_encounters"] = 0
                    step_output_data_ui = None

                    self.logger.info(f"Starting Visual UI Automation: Service='{params.get('service')}', Goal='{automate_goal}' on page {page_id}")

                    while current_step_ui < max_steps_ui:
                        current_step_ui += 1; log_content_for_kb["visual_llm_decisions"] +=1
                        await self._internal_think(f"UI Automate Step {current_step_ui}/{max_steps_ui}: Goal='{automate_goal}'", details={"last_action_result": last_action_summary_ui})
                        await self._human_like_delay(page, short=True)

                        screenshot_bytes_ui = await page.screenshot(full_page=True)
                        page_text_content_ui = await page.evaluate("document.body.innerText")
                        current_url_ui = page.url

                        ui_automation_task_context = {
                            "task": "Determine next UI action based primarily on visual context",
                            "service_context": params.get('service'), "current_goal": automate_goal,
                            "automation_parameters": automate_params,
                            "page_text_content_snippet": page_text_content_ui[:6000], # Increased snippet
                            "current_url": current_url_ui, "last_action_result": last_action_summary_ui,
                            "current_step": f"{current_step_ui}/{max_steps_ui}",
                            "available_actions": ["click", "input", "upload", "scroll", "wait", "navigate", "download", "solve_captcha", "extract_value", "finish", "error"],
                            "desired_output_format": "JSON: {\"action_type\": \"<chosen_action>\", \"selector\": \"<css_selector_or_xpath_or_visual_desc_of_target>\", \"coordinates\": {\"x\": float, \"y\": float}?, \"text_to_input\": \"<string_or_param_key_like_params.my_value_or_identity.email>\"?, \"file_path_param\": \"<param_name_for_file_from_automation_parameters>\"?, \"scroll_direction\": \"up|down|left|right|top|bottom\"?, \"wait_time_ms\": int?, \"target_url\": \"<string>\"?, \"captcha_type\": \"checkbox|image_challenge|etc\"?, \"extraction_details\": {\"output_variable_name\": \"<name_for_extracted_data>\", \"value_type\": \"text|attribute:src|attribute:href\"}?, \"reasoning\": \"<Detailed explanation of why this action and target were chosen based on visual evidence and goal.>\", \"error_message\": \"<If action_type is error>\"}"
                        }
                        if "captcha" in automate_goal.lower() or "captcha" in last_action_summary_ui.lower():
                             ui_automation_task_context["special_focus"] = "A CAPTCHA might be present. Identify it and determine the interaction needed."
                        
                        llm_prompt_ui = await self.generate_dynamic_prompt(ui_automation_task_context)
                        action_json_str_ui = await self.orchestrator.call_llm(agent_name=self.AGENT_NAME, prompt=llm_prompt_ui, temperature=0.05, max_tokens=1200, is_json_output=True, image_data=screenshot_bytes_ui)
                        
                        if not action_json_str_ui: last_action_summary_ui = "LLM failed to determine UI action."; continue
                        try:
                            action_data_ui = self._parse_llm_json(action_json_str_ui)
                            if not action_data_ui: raise ValueError("Failed to parse UI action JSON.")
                            action_type_ui = action_data_ui.get('action_type')
                            reasoning_ui = action_data_ui.get('reasoning', 'N/A')
                            self.logger.info(f"UI Automate Step {current_step_ui}: LLM action: {action_type_ui}. Reasoning: {reasoning_ui}")
                            action_executed_successfully_ui = False; action_result_message_ui = f"Action '{action_type_ui}' initiated."

                            if action_type_ui == 'click':
                                selector_ui = action_data_ui.get('selector'); coords_ui = action_data_ui.get('coordinates')
                                target_element_ui = await self._find_element(page, selector_ui, coords_ui, visual_description_if_no_selector=selector_ui if not coords_ui else None)
                                if target_element_ui: await self._human_click(page, target_element_ui); action_executed_successfully_ui = True; action_result_message_ui = f"Clicked: '{selector_ui or coords_ui or 'visual_target'}'."
                                else: action_result_message_ui = f"Click failed: Element not found for '{selector_ui or coords_ui or 'visual_target'}'."
                            elif action_type_ui == 'input':
                                selector_ui = action_data_ui.get('selector'); coords_ui = action_data_ui.get('coordinates')
                                text_to_input_raw_ui = action_data_ui.get('text_to_input')
                                text_to_input_ui = None
                                if isinstance(text_to_input_raw_ui, str):
                                    if text_to_input_raw_ui.startswith("params."):
                                        key_ui = text_to_input_raw_ui.split('params.')[-1]
                                        text_to_input_ui = automate_params.get(key_ui)
                                        if text_to_input_ui is None: action_result_message_ui = f"Input failed: Param key '{key_ui}' not in automate_params."
                                        else: self.logger.debug(f"Resolved input text from automate_params key '{key_ui}'")
                                    elif text_to_input_raw_ui.startswith("identity."): # Handle identity resolution
                                        identity_params = automate_params.get("identity", {}) # Assuming identity is passed in automate_params
                                        key_ui = text_to_input_raw_ui.split('identity.')[-1]
                                        text_to_input_ui = identity_params.get(key_ui)
                                        if text_to_input_ui is None: action_result_message_ui = f"Input failed: Identity key '{key_ui}' not found."
                                        else: self.logger.debug(f"Resolved input text from identity key '{key_ui}'")
                                    else: text_to_input_ui = str(text_to_input_raw_ui) # Literal
                                elif text_to_input_raw_ui is not None: text_to_input_ui = str(text_to_input_raw_ui)
                                else: action_result_message_ui = "Input failed: LLM provided no text or valid param/identity key."

                                target_element_ui = await self._find_element(page, selector_ui, coords_ui, visual_description_if_no_selector=selector_ui if not coords_ui else None)
                                if target_element_ui and text_to_input_ui is not None:
                                    await self._human_fill(page, target_element_ui, text_to_input_ui)
                                    action_executed_successfully_ui = True; action_result_message_ui = f"Input '{str(text_to_input_ui)[:20]}...' into '{selector_ui or coords_ui or 'visual_target'}'."
                                elif not target_element_ui: action_result_message_ui = f"Input failed: Element not found for '{selector_ui or coords_ui or 'visual_target'}'."
                            elif action_type_ui == 'upload':
                                selector_ui = action_data_ui.get('selector'); coords_ui = action_data_ui.get('coordinates')
                                file_path_param_name_ui = action_data_ui.get('file_path_param')
                                file_path_ui = automate_params.get(file_path_param_name_ui) if file_path_param_name_ui else None
                                target_element_ui = await self._find_element(page, selector_ui, coords_ui, visual_description_if_no_selector=selector_ui if not coords_ui else None)
                                if target_element_ui and file_path_ui and os.path.exists(str(file_path_ui)):
                                    async with page.expect_file_chooser(timeout=30000) as fc_info_ui:
                                         await self._human_click(page, target_element_ui, timeout=20000)
                                    file_chooser_ui = await fc_info_ui.value
                                    await file_chooser_ui.set_files(str(file_path_ui))
                                    action_executed_successfully_ui = True; action_result_message_ui = f"Uploaded file '{os.path.basename(str(file_path_ui))}'."
                                elif not file_path_ui or not os.path.exists(str(file_path_ui)): action_result_message_ui = f"Upload failed: File path '{file_path_ui}' invalid (from param '{file_path_param_name_ui}')."
                                else: action_result_message_ui = f"Upload failed: Element not found for '{selector_ui or coords_ui}'."
                            elif action_type_ui == 'scroll':
                                direction_ui = action_data_ui.get('scroll_direction', 'down'); selector_ui = action_data_ui.get('selector')
                                target_element_ui = await self._find_element(page, selector_ui) if selector_ui else None
                                await self._human_scroll(page, direction_ui, target_element_ui)
                                action_executed_successfully_ui = True; action_result_message_ui = f"Scrolled {direction_ui}."
                            elif action_type_ui == 'wait':
                                wait_ms_ui = action_data_ui.get('wait_time_ms', random.randint(1500, 4000))
                                await page.wait_for_timeout(wait_ms_ui)
                                action_executed_successfully_ui = True; action_result_message_ui = f"Waited for {wait_ms_ui} ms."
                            elif action_type_ui == 'navigate':
                                target_url_ui = action_data_ui.get('target_url')
                                if target_url_ui:
                                    response_nav = await page.goto(target_url_ui, wait_until='domcontentloaded', timeout=60000)
                                    if response_nav and response_nav.ok: action_executed_successfully_ui = True; action_result_message_ui = f"Navigated to {target_url_ui}."
                                    else: action_result_message_ui = f"Navigation failed: Status {response_nav.status if response_nav else 'N/A'} for {target_url_ui}."
                                else: action_result_message_ui = "Navigation failed: LLM did not provide target_url."
                            elif action_type_ui == 'download':
                                selector_ui = action_data_ui.get('selector'); coords_ui = action_data_ui.get('coordinates')
                                target_element_ui = await self._find_element(page, selector_ui, coords_ui, visual_description_if_no_selector=selector_ui if not coords_ui else None)
                                if target_element_ui:
                                    async with page.expect_download(timeout=self.config.get("BROWSER_DOWNLOAD_TIMEOUT_S", 300) * 1000) as download_info_ui:
                                        await self._human_click(page, target_element_ui)
                                    download_ui = await download_info_ui.value
                                    download_filename_ui = f"{task_context.get('id','task')}_{uuid.uuid4().hex[:8]}_{download_ui.suggested_filename}" # Include task ID
                                    download_path_ui = os.path.join(self.temp_dir, download_filename_ui)
                                    await download_ui.save_as(download_path_ui)
                                    ui_automation_extracted_data["downloaded_file_path"] = download_path_ui
                                    action_executed_successfully_ui = True; action_result_message_ui = f"File downloaded to {download_path_ui}"
                                else: action_result_message_ui = f"Download failed: Element not found for '{selector_ui or coords_ui}'."
                            elif action_type_ui == 'solve_captcha':
                                log_content_for_kb["captcha_encounters"] +=1
                                captcha_type_ui = action_data_ui.get('captcha_type', 'unknown'); selector_ui = action_data_ui.get('selector'); coords_ui = action_data_ui.get('coordinates')
                                target_element_ui = await self._find_element(page, selector_ui, coords_ui, visual_description_if_no_selector=selector_ui if not coords_ui else "the CAPTCHA element")
                                if target_element_ui:
                                    if captcha_type_ui == 'checkbox':
                                        await self._human_click(page, target_element_ui); await page.wait_for_timeout(random.uniform(5000, 10000))
                                        verify_screenshot = await page.screenshot(); verify_text = await page.evaluate("document.body.innerText")
                                        verify_prompt_ctx = {"task": "Verify CAPTCHA checkbox result", "page_text_content_snippet": verify_text[:2000], "desired_output_format": "JSON: {\"captcha_solved\": bool, \"new_challenge_description\": str?}"}
                                        verify_prompt = await self.generate_dynamic_prompt(verify_prompt_ctx)
                                        verify_json = await self.orchestrator.call_llm(self.AGENT_NAME, verify_prompt, image_data=verify_screenshot, is_json_output=True, max_tokens=300)
                                        verify_data = self._parse_llm_json(verify_json)
                                        if verify_data and verify_data.get("captcha_solved"): action_executed_successfully_ui = True; action_result_message_ui = "Clicked CAPTCHA checkbox, visually confirmed."
                                        else: action_result_message_ui = f"Clicked CAPTCHA checkbox, but visual confirmation failed or new challenge: {verify_data.get('new_challenge_description','N/A')}"
                                    elif captcha_type_ui == 'image_challenge':
                                        action_result_message_ui = f"Image CAPTCHA solving needs advanced visual reasoning and interaction not yet fully implemented for element '{selector_ui or coords_ui}'. Consider manual intervention or dedicated CAPTCHA solving service integration via ThinkTool."
                                        self.logger.warning(action_result_message_ui)
                                    else: action_result_message_ui = f"CAPTCHA solving failed: Unsupported type '{captcha_type_ui}'."
                                else: action_result_message_ui = f"CAPTCHA solving failed: Element not found for '{selector_ui or coords_ui}'."
                            elif action_type_ui == 'extract_value':
                                extraction_details_ui = action_data_ui.get('extraction_details', {})
                                selector_ui = action_data_ui.get('selector'); coords_ui = action_data_ui.get('coordinates')
                                output_var_name_ui = extraction_details_ui.get('output_variable_name', f"extracted_val_{current_step_ui}")
                                value_type_ui = extraction_details_ui.get('value_type', 'text')
                                target_element_ui = await self._find_element(page, selector_ui, coords_ui, visual_description_if_no_selector=selector_ui if not coords_ui else None)
                                if target_element_ui:
                                    extracted_val = None
                                    if value_type_ui == 'text': extracted_val = await target_element_ui.text_content(timeout=10000)
                                    elif value_type_ui.startswith('attribute:'): attr_name = value_type_ui.split(':')[-1]; extracted_val = await target_element_ui.get_attribute(attr_name, timeout=10000)
                                    if extracted_val is not None:
                                        ui_automation_extracted_data[output_var_name_ui] = extracted_val.strip() # Store in dict for this automation task
                                        action_executed_successfully_ui = True; action_result_message_ui = f"Extracted '{value_type_ui}' for '{output_var_name_ui}'."
                                    else: action_result_message_ui = f"Extraction failed: Element found but no value for '{output_var_name_ui}' (type: {value_type_ui})."
                                else: action_result_message_ui = f"Extraction failed: Element not found for '{selector_ui or coords_ui}'."
                            elif action_type_ui == 'finish':
                                automation_success = True; action_executed_successfully_ui = True
                                action_result_message_ui = f"UI Automation goal '{automate_goal}' completed successfully by LLM decision."
                                self.logger.info(action_result_message_ui); break
                            elif action_type_ui == 'error':
                                error_msg_ui = action_data_ui.get('error_message', 'LLM indicated an unrecoverable error.')
                                self.logger.error(f"LLM reported error during UI automation: {error_msg_ui}")
                                raise RuntimeError(f"UI Automation failed: {error_msg_ui}")
                            else: action_result_message_ui = f"Unsupported action type from LLM: {action_type_ui}"; self.logger.warning(action_result_message_ui)

                            last_action_summary_ui = action_result_message_ui
                            if not action_executed_successfully_ui:
                                self.logger.warning(f"UI Automate Attempt {current_step_ui}: Action '{action_type_ui}' failed. Msg: {action_result_message_ui}")
                                if current_step_ui == max_steps_ui: raise RuntimeError(f"UI Automation failed after {max_steps_ui} steps. Last: {action_type_ui}. Msg: {last_action_summary_ui}")
                        except PlaywrightError as pe_auto:
                            last_action_summary_ui = f"Playwright error executing '{action_data_ui.get('action_type', 'unknown')}': {pe_auto}"
                            self.logger.error(f"UI Automate Step {current_step_ui}: {last_action_summary_ui}", exc_info=False)
                            if current_step_ui == max_steps_ui: raise RuntimeError(f"UI Automation failed (Playwright error): {pe_auto}") from pe_auto
                        except Exception as auto_err:
                            last_action_summary_ui = f"Unexpected error executing '{action_data_ui.get('action_type', 'unknown')}': {auto_err}"
                            self.logger.error(f"UI Automate Step {current_step_ui}: {last_action_summary_ui}", exc_info=True)
                            if current_step_ui == max_steps_ui: raise RuntimeError(f"UI Automation failed (unexpected error): {auto_err}") from auto_err
                    
                    log_content_for_kb["steps_taken_count"] = current_step_ui
                    if automation_success:
                        result = {"status": "success", "message": f"UI Automation goal '{automate_goal}' completed.", "result_data": ui_automation_extracted_data}
                    else: result = {"status": "failure", "message": f"UI Automation goal '{automate_goal}' failed or max steps. Last: {last_action_summary_ui}", "result_data": ui_automation_extracted_data}
                    log_content_for_kb["success_status"] = automation_success
                    log_content_for_kb["final_url_or_state"] = page.url if page else "Page unavailable"

                elif step_action == 'close_context':
                    reason_close = params.get('reason', 'Step complete')
                    if context_id: await self._close_context(context_id, context_obj); task_context["current_context_id"] = None; task_context["current_page_id"] = None
                    result = {"status": "success", "message": f"Browser context closed. Reason: {reason_close}"}
                
                elif step_action == 'generate_dorks':
                    target_gd = params.get('target'); dork_types_gd = params.get('types', [])
                    if not target_gd: raise ValueError("Missing target for dork generation.")
                    dorks_gd = self._generate_google_dorks(target_gd, dork_types_gd)
                    task_context['generated_dorks'] = dorks_gd
                    result = {"status": "success", "message": f"Generated {len(dorks_gd)} dorks.", "dorks": dorks_gd, "result_data": dorks_gd}
                
                elif step_action == 'execute_dork_searches':
                    dorks_to_search = task_context.get('generated_dorks', params.get('dorks'))
                    if not dorks_to_search: raise ValueError("No dorks provided or generated for execute_dork_searches.")
                    search_results_map = {}; log_content_for_kb["dork_searches_attempted"] = len(dorks_to_search)
                    dork_success_count = 0
                    for i, dork_query in enumerate(dorks_to_search):
                        self.logger.info(f"Searching Dork ({i+1}/{len(dorks_to_search)}): {dork_query}")
                        search_url_gd = f"https://www.google.com/search?q={quote_plus(dork_query)}&num=20&hl=en&filter=0"
                        try:
                            await page.goto(search_url_gd, wait_until='domcontentloaded', timeout=35000)
                            await self._human_like_delay(page)
                            screenshot_gd = await page.screenshot(); llm_check_ctx_gd = {"task": "Check for Google search block/CAPTCHA", "current_url": page.url, "desired_output_format": "JSON: {\"is_blocked\": boolean, \"reason\": \"<description if blocked>\"}"}
                            llm_prompt_gd = await self.generate_dynamic_prompt(llm_check_ctx_gd)
                            check_resp_gd = await self.orchestrator.call_llm(agent_name=self.AGENT_NAME, prompt=llm_prompt_gd, image_data=screenshot_gd, max_tokens=150, is_json_output=True, temperature=0.1)
                            block_check_gd = self._parse_llm_json(check_resp_gd)
                            if block_check_gd and block_check_gd.get("is_blocked"):
                                self.logger.warning(f"Google search blocked for dork: {dork_query}. Reason: {block_check_gd.get('reason')}. Skipping remaining dorks for this context."); self._update_proxy_stats(context_id, False); break
                            
                            links_locators = await page.locator('//div[contains(@class, "g ") and .//a[@href and @ping]]//a[@href and @ping]').all()
                            valid_links_gd = []
                            for link_el in links_locators:
                                href_gd = await link_el.get_attribute('href')
                                if href_gd and href_gd.startswith('http') and not re.match(r'https?://(www\.)?google\.com/|https?://accounts\.google\.com/', href_gd):
                                    cleaned_url_gd = urljoin(page.url, href_gd)
                                    parsed_url_gd = urlparse(cleaned_url_gd)
                                    final_url_gd = f"{parsed_url_gd.scheme}://{parsed_url_gd.netloc}{parsed_url_gd.path}"
                                    if parsed_url_gd.query: final_url_gd += f"?{parsed_url_gd.query}"
                                    valid_links_gd.append(final_url_gd)
                            search_results_map[dork_query] = list(dict.fromkeys(valid_links_gd))[:15]
                            dork_success_count +=1; self._update_proxy_stats(context_id, True)
                        except Exception as e_dork_search: self.logger.error(f"Error searching dork '{dork_query}': {e_dork_search}"); self._update_proxy_stats(context_id, False)
                        await asyncio.sleep(random.uniform(7, 18))
                    task_context['dork_search_results'] = search_results_map
                    log_content_for_kb["dork_searches_successful"] = dork_success_count
                    result = {"status": "success", "message": f"Executed {len(dorks_to_search)} dork searches.", "results_summary": {k: len(v) for k, v in search_results_map.items()}, "result_data": search_results_map}

                elif step_action == 'analyze_dork_results':
                    dork_results_to_analyze = task_context.get('dork_search_results', params.get('dork_results'))
                    if not dork_results_to_analyze: raise ValueError("No dork search results to analyze.")
                    all_urls_to_analyze = list(set(url for urls in dork_results_to_analyze.values() for url in urls))
                    max_urls_dork = self.config.get("DORK_MAX_URLS_TO_ANALYZE", 30)
                    potential_findings_dork = []; analyzed_count_dork = 0
                    for i, url_dork in enumerate(all_urls_to_analyze[:max_urls_dork]):
                        await self._internal_think(f"Analyzing Dork Result URL ({i+1}/{min(len(all_urls_to_analyze), max_urls_dork)}): {url_dork}")
                        try:
                            await page.goto(url_dork, wait_until='domcontentloaded', timeout=45000)
                            await self._human_like_delay(page, short=True); analyzed_count_dork +=1
                            screenshot_dork = await page.screenshot(full_page=True); page_text_dork = await page.evaluate("document.body.innerText")
                            if not page_text_dork or len(page_text_dork) < 20: continue
                            analysis_ctx_dork = {"task": "Analyze webpage for credentials/secrets/login_forms", "text_content_snippet": page_text_dork[:8000], "target_url": url_dork, "desired_output_format": "JSON list of findings: [{\"type\": \"password|api_key|secret|login_form|other\", \"value_snippet\": \"...\", \"context_snippet\": \"...\"}] or []"}
                            llm_prompt_dork = await self.generate_dynamic_prompt(analysis_ctx_dork)
                            findings_json_dork = await self.orchestrator.call_llm(agent_name=self.AGENT_NAME, prompt=llm_prompt_dork, temperature=0.1, max_tokens=1500, is_json_output=True, image_data=screenshot_dork)
                            if findings_json_dork:
                                findings_list_dork = self._parse_llm_json(findings_json_dork, expect_type=list)
                                if findings_list_dork:
                                    for finding_dork in findings_list_dork: finding_dork['source_url'] = url_dork
                                    potential_findings_dork.extend(findings_list_dork)
                        except Exception as e_dork_analyze: self.logger.warning(f"Failed to analyze dork URL {url_dork}: {e_dork_analyze}")
                        await asyncio.sleep(random.uniform(2, 5))
                    log_content_for_kb["dork_urls_analyzed"] = analyzed_count_dork
                    log_content_for_kb["dork_potential_findings_count"] = len(potential_findings_dork)
                    result = {"status": "success", "message": f"Dork analysis complete. Found {len(potential_findings_dork)} potential items.", "potential_findings": potential_findings_dork, "result_data": potential_findings_dork}
                    if potential_findings_dork and self.think_tool:
                        await self.think_tool.execute_task({"action": "log_knowledge_fragment", "fragment_data": {"agent_source": self.AGENT_NAME, "data_type": "dorking_sensitive_findings_batch", "content": {"findings": potential_findings_dork[:10]}, "tags": ["dorking", "security_alert"], "relevance_score": 0.9}})
                else: result = {"status": "failure", "message": f"Unsupported browser action: {step_action}"}
            elif tool == 'internal':
                 if step_action == 'generate_dorks':
                     target_gd = params.get('target'); dork_types_gd = params.get('types', [])
                     if not target_gd: raise ValueError("Missing target for dork generation.")
                     dorks_gd = self._generate_google_dorks(target_gd, dork_types_gd)
                     task_context['generated_dorks'] = dorks_gd
                     result = {"status": "success", "message": f"Generated {len(dorks_gd)} dorks.", "dorks": dorks_gd, "result_data": dorks_gd}
                 else: result = {"status": "failure", "message": f"Unknown internal action: {step_action}"}
            else: result = {"status": "failure", "message": f"Unsupported tool type specified: {tool}"}

            log_content_for_kb["status_final"] = result.get("status")
            log_content_for_kb["message_final"] = result.get("message")
            log_content_for_kb["extracted_data_preview"] = str(result.get("extracted_data", result.get("result_data", {})))[:200]
            self._update_proxy_stats(context_id, result.get("status") == "success")

        except PlaywrightError as pe:
            self.logger.error(f"Playwright error during step {step_num} ('{step_action}'): {pe}", exc_info=False)
            result = {"status": "failure", "message": f"Browser error: {pe.message}"}
            log_content_for_kb.update({"status_final": "failure", "error_type": "PlaywrightError", "error_message": pe.message})
            self._update_proxy_stats(context_id, False)
            if page and not page.is_closed():
                 error_screenshot_path = os.path.join(self.temp_dir, f"error_{context_id}_{page_id}_{int(time.time())}.png")
                 try: await page.screenshot(path=error_screenshot_path, full_page=True); result["error_screenshot"] = error_screenshot_path; log_content_for_kb["error_screenshot"] = error_screenshot_path
                 except Exception as ss_err: self.logger.error(f"Failed to capture error screenshot: {ss_err}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during step {step_num} ('{step_action}'): {e}", exc_info=True)
            result = {"status": "failure", "message": f"Unexpected error: {e}"}
            log_content_for_kb.update({"status_final": "failure", "error_type": type(e).__name__, "error_message": str(e)})
            self._update_proxy_stats(context_id, False)
            if page and not page.is_closed():
                 error_screenshot_path = os.path.join(self.temp_dir, f"unexpected_error_{context_id}_{page_id}_{int(time.time())}.png")
                 try: await page.screenshot(path=error_screenshot_path, full_page=True); result["error_screenshot"] = error_screenshot_path; log_content_for_kb["error_screenshot"] = error_screenshot_path
                 except Exception as ss_err: self.logger.error(f"Failed to capture unexpected error screenshot: {ss_err}")
            await self._report_error(f"Step {step_num} ('{step_action}') failed unexpectedly: {e}", task_context.get('id'))
        finally:
            log_content_for_kb["timestamp_end"] = datetime.now(timezone.utc).isoformat()
            log_content_for_kb["duration_seconds"] = round(time.time() - step_start_time, 2)
            if self.think_tool and step_action not in ['close_context'] and log_content_for_kb.get("status_final"):
                await self.think_tool.execute_task({
                    "action": "log_knowledge_fragment", "fragment_data": {
                        "agent_source": self.AGENT_NAME, "data_type": "browsing_ui_automation_outcome",
                        "content": log_content_for_kb,
                        "tags": [self.AGENT_NAME, step_action.replace(" ","_"), log_content_for_kb["status_final"], params.get("service", "general")],
                        "relevance_score": 0.75 if log_content_for_kb["status_final"] == "success" else 0.55 }})
        return result

    async def _ensure_context_and_page(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Tuple[BrowserContext, Page, str, str]:
        context_id = task_context.get("current_context_id")
        page_id = task_context.get("current_page_id")
        context: Optional[BrowserContext] = None
        page: Optional[Page] = None
        params = step.get('params', {})
        requires_account = params.get('requires_account', False)
        service_needed = params.get('service') or params.get('target_platform')
        account_id_param = params.get('account_id')
        allow_creation = params.get('allow_account_creation', True)

        if context_id:
            context = self.internal_state['browser_contexts'].get(context_id)
            if not context or (hasattr(context, 'is_closed') and context.is_closed()):
                self.logger.warning(f"Context {context_id} was closed/invalid. Will create new.")
                await self._close_context(context_id)
                context_id = None; page_id = None; context = None
            else:
                current_account = self.internal_state['context_account_map'].get(context_id)
                if requires_account:
                    if not current_account:
                        self.logger.warning(f"Existing context {context_id} has no account, step requires one. New context.")
                        await self._close_context(context_id); context_id = None; page_id = None; context = None
                    elif service_needed and current_account.get('service') != service_needed:
                        self.logger.warning(f"Context {context_id} for service '{current_account.get('service')}', step needs '{service_needed}'. New context.")
                        await self._close_context(context_id); context_id = None; page_id = None; context = None
                    elif account_id_param and current_account.get('id') != account_id_param:
                         self.logger.warning(f"Context {context_id} has account ID {current_account.get('id')}, step needs {account_id_param}. New context.")
                         await self._close_context(context_id); context_id = None; page_id = None; context = None
        if not context_id:
            account_details = None
            proxy_info = params.get('proxy_info')
            if requires_account:
                if account_id_param:
                    account_details = await self._get_account_details_from_db(account_id_param)
                    if not account_details: raise ValueError(f"Required account ID {account_id_param} not found.")
                elif service_needed:
                    account_details = await self._select_available_account_for_service(service_needed)
                    if not account_details and allow_creation:
                        self.logger.warning(f"No available account for '{service_needed}'. Attempting creation.")
                        account_details = await self._attempt_new_account_creation(service_needed)
                        if not account_details: raise RuntimeError(f"Failed to find or create account for: {service_needed}")
                    elif not account_details and not allow_creation:
                         raise ValueError(f"No account for '{service_needed}', creation disallowed.")
                else: raise ValueError("Step requires account, but 'service' or 'account_id' missing.")
            if not proxy_info and hasattr(self.orchestrator, 'get_proxy'):
                 proxy_purpose = f"context_{service_needed or 'general'}_{account_details.get('id', 'new') if account_details else 'anon'}"
                 proxy_info = await self.orchestrator.get_proxy(purpose=proxy_purpose, quality_level='high_stealth' if requires_account else 'standard_plus')
            context_id, context = await self._get_new_context(account_details, proxy_info)
        page_id, page = await self._get_page_for_context(context_id)
        return context, page, context_id, page_id

    async def learning_loop(self):
        self.logger.info(f"{self.AGENT_NAME} L40+ learning loop started.")
        while not self._stop_event.is_set():
            try:
                learn_interval = int(self.config.get("BROWSER_LEARNING_INTERVAL_S", 3600 * 1))
                await asyncio.sleep(learn_interval)
                if self._stop_event.is_set(): break
                await self._internal_think("Learning cycle: Analyzing proxy performance and detailed UI task outcomes for ThinkTool.")
                poor_proxies = []
                proxy_stats = self.internal_state.get('proxy_stats', {})
                for proxy_url, stats in proxy_stats.items():
                    total_attempts = stats.get('success', 0) + stats.get('failure', 0)
                    if total_attempts > 5:
                        failure_rate = stats.get('failure', 0) / total_attempts if total_attempts > 0 else 0
                        if failure_rate > 0.4:
                            proxy_display_name = proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url
                            last_used_iso = stats.get('last_used').isoformat() if isinstance(stats.get('last_used'), datetime) else str(stats.get('last_used'))
                            poor_proxies.append({"proxy_server": proxy_display_name, "failure_rate": failure_rate, "attempts": total_attempts, "last_used": last_used_iso})
                if poor_proxies and self.think_tool:
                     await self.think_tool.execute_task({
                         "action": "log_knowledge_fragment", "fragment_data": {
                             "agent_source": self.AGENT_NAME, "data_type": "proxy_performance_alert",
                             "content": {"poor_performing_proxies": poor_proxies, "analysis_timestamp": datetime.now(timezone.utc).isoformat()},
                             "tags": ["proxy", "performance", "alert", "browsing_agent"], "relevance_score": 0.85 }})
                if self.think_tool:
                    recent_ui_outcomes = await self.think_tool.query_knowledge_base(data_types=["browsing_ui_automation_outcome"], tags=[self.AGENT_NAME], time_window=timedelta(days=1), limit=50)
                    if recent_ui_outcomes:
                        detailed_outcomes_summary = []
                        for frag in recent_ui_outcomes:
                            try:
                                content = json.loads(frag.content) if isinstance(frag.content, str) else frag.content
                                detailed_outcomes_summary.append({
                                    "service": content.get("service_target", "Unknown"), "goal_summary": content.get("goal_summary", "Unknown"),
                                    "status": content.get("status_final", "unknown"), "error_type": content.get("error_type"),
                                    "steps_taken": content.get("steps_taken_count"), "captcha_encounters": content.get("captcha_encounters",0),
                                    "visual_llm_decisions": content.get("visual_llm_decisions",0),
                                    "proxy_used_short": content.get("proxy_used_details","N/A"),
                                    "timestamp": frag.timestamp.isoformat() if isinstance(frag.timestamp, datetime) else str(frag.timestamp)
                                })
                            except Exception as e_parse_frag: self.logger.warning(f"Could not parse outcome fragment {frag.id}: {e_parse_frag}")
                        if detailed_outcomes_summary:
                            await self.think_tool.execute_task({"action": "process_feedback", "feedback_data": {self.AGENT_NAME: {"type": "ui_automation_performance_deep_dive", "outcomes_summary": detailed_outcomes_summary, "period_analyzed": "last_24_hours", "request_for_thinktool": "Analyze these UI automation outcomes. Identify patterns of failure for specific services or goals. Suggest improvements to BrowsingAgent's visual analysis prompts, interaction strategies, or if specific sites require dedicated 'exploit scripts'."}}})
                            self.logger.info(f"Sent {len(detailed_outcomes_summary)} UI automation outcome summaries to ThinkTool for deep analysis.")
                    else: self.logger.info("No detailed UI automation outcomes found in KB for learning cycle.")
                self.internal_state["last_learning_cycle_ts"] = datetime.now(timezone.utc)
                self.logger.debug(f"{self.AGENT_NAME} learning cycle complete.")
            except asyncio.CancelledError: self.logger.info(f"{self.AGENT_NAME} learning loop cancelled."); break
            except Exception as e: self.logger.error(f"Error in {self.AGENT_NAME} learning loop: {e}", exc_info=True); await self._report_error(f"Learning loop error: {e}"); await asyncio.sleep(60 * 15)

    async def self_critique(self) -> Dict[str, Any]:
        self.logger.info(f"{self.AGENT_NAME}: Performing self-critique.")
        critique = {"status": "ok", "feedback": "Critique pending analysis."}
        try:
            insights = await self.collect_insights()
            error_fragments_content = []
            if self.think_tool:
                error_fragments = await self.think_tool.query_knowledge_base(data_types=["browsing_ui_automation_outcome"], tags=[self.AGENT_NAME, "failure"], time_window=timedelta(days=3), limit=20)
                for frag in error_fragments:
                    try: error_fragments_content.append(json.loads(frag.content) if isinstance(frag.content, str) else frag.content)
                    except: pass
            
            critique_context = {
                "task": "Critique BrowsingAgent Performance & Resilience", "current_insights": insights,
                "recent_failure_logs": error_fragments_content,
                "desired_output_format": "JSON: { \"overall_assessment\": str, \"strengths\": [str], \"weaknesses\": [str], \"suggestions_for_thinktool_directives\": [\"Specific directive for ThinkTool to improve BrowsingAgent (e.g., 'Research new anti-fingerprinting for site X', 'Develop UI exploit script for temp mail Y')\"] }"
            }
            prompt = await self.generate_dynamic_prompt(critique_context)
            llm_model_pref = settings.OPENROUTER_MODELS.get('think_critique')
            critique_json = await self._call_llm_with_retry(prompt, model=llm_model_pref, temperature=0.4, max_tokens=1000, is_json_output=True)
            if critique_json:
                 try:
                     critique_result = self._parse_llm_json(critique_json)
                     if not critique_result: raise ValueError("Parsed critique is None")
                     critique['feedback'] = critique_result.get('overall_assessment', 'LLM critique generated.')
                     critique['details'] = critique_result
                     if self.think_tool: await self.think_tool.execute_task({"action": "log_knowledge_fragment", "fragment_data": {"agent_source": self.AGENT_NAME, "data_type": "self_critique_summary_L40", "content": critique_result, "tags": ["critique", "browsing_agent", "L40"], "relevance_score": 0.9 }})
                     if self.think_tool and critique_result.get("suggestions_for_thinktool_directives"):
                         for suggestion_str in critique_result["suggestions_for_thinktool_directives"]:
                             await self.think_tool.execute_task({"action":"create_directive_from_suggestion", "content": {"source_agent": self.AGENT_NAME, "suggestion": suggestion_str, "priority": 6}})
                 except Exception as e_parse: self.logger.error(f"Failed to parse self-critique LLM response: {e_parse}"); critique['feedback'] += " Failed to parse LLM critique."; critique['status'] = 'error'
            else: critique['feedback'] += " LLM critique call failed."; critique['status'] = 'error'
        except Exception as e: self.logger.error(f"Error during self-critique: {e}", exc_info=True); critique['status'] = 'error'; critique['feedback'] = f"Self-critique failed: {e}"
        return critique

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        self.logger.debug(f"Generating dynamic prompt for BrowsingAgent task: {task_context.get('task')}")
        prompt_parts = [self.meta_prompt]
        prompt_parts.append("\n--- Current Task & Context ---")
        priority_keys = ['task', 'current_goal', 'current_sub_goal', 'overall_sub_task_goal', 'service_context', 'current_url', 'last_action_result', 'special_focus', 'identity_details_summary', 'automation_parameters', 'target_description', 'extraction_instructions', 'target_profile_for_analytics', 'query']
        for key in priority_keys:
            if key in task_context and task_context[key] is not None:
                 value = task_context[key];
                 value_str = ""
                 if isinstance(value, (dict,list)):
                     try: value_str = json.dumps(value, default=str, indent=2)[:1500] + "..." # Limit complex objects
                     except: value_str = str(value)[:1500] + "..."
                 else: value_str = str(value)[:1500] + ("..." if len(str(value)) > 1500 else "")
                 prompt_parts.append(f"**{key.replace('_', ' ').title()}**: {value_str}")

        prompt_parts.append("\n**Visual Context:** A screenshot of the current webpage is provided with this request.")
        prompt_parts.append("**Text Context Snippet (from page):**"); text_snippet = task_context.get('page_text_content_snippet', '')
        prompt_parts.append(f"```\n{text_snippet[:4000]}\n```") # Ensure snippet is not overly long
        
        other_params = {k: v for k, v in task_context.items() if k not in priority_keys and k not in ['page_text_content_snippet', 'task', 'desired_output_format', 'image_data']}
        if other_params:
            prompt_parts.append("\n**Other Parameters for this action:**")
            try: prompt_parts.append(f"```json\n{json.dumps(other_params, default=str, indent=2)}\n```")
            except: prompt_parts.append(str(other_params)[:1000] + "...")
        
        prompt_parts.append("\n--- Your Instructions ---")
        task_type = task_context.get('task')
        if task_type == 'Determine next UI action based primarily on visual context' or task_type == 'Determine next UI action for sub-task based on visual context' or task_type == 'Determine next UI action for account signup':
            prompt_parts.append(f"Analyze the provided **screenshot** and text snippet. Based *primarily* on the visual layout and the current goal ('{task_context.get('current_goal') or task_context.get('current_sub_goal') or task_context.get('current_micro_goal','N/A')}'), determine the single best next action from {task_context.get('available_actions', [])}.")
            prompt_parts.append("Provide a precise selector (CSS or XPath if obvious, otherwise a clear visual description of the target element, e.g., 'the large blue button that says Submit') or coordinates for the target element. If inputting text, specify the exact text or reference 'params.<key>' or 'identity.<key>' from automation_parameters.")
            prompt_parts.append("If a CAPTCHA is visible, use 'solve_captcha'. If extracting a value, use 'extract_value' and specify 'output_variable_name'.")
            prompt_parts.append("Use 'finish', 'finish_current_micro_goal', or 'finish_entire_sub_task' only when the specific goal is definitively achieved. Use 'error' or 'error_sub_task' if stuck or unrecoverable issue.")
            prompt_parts.append("Provide clear, concise reasoning, focusing on visual cues and goal alignment.")
        elif task_type == 'Extract structured data from webpage using visual and text context':
            prompt_parts.append(f"Analyze the screenshot and text. Follow these instructions: {task_context.get('extraction_instructions', 'Extract key information.')}")
            prompt_parts.append("Identify the data visually and textually. Use null for missing fields.")
        elif task_type == 'Analyze webpage for credentials/secrets/login_forms':
             prompt_parts.append(f"Carefully examine the **screenshot** and text from URL '{task_context.get('target_url')}' for any potential credentials (passwords, API keys, secrets), sensitive configuration data, exposed login forms, or other security risks.")
             prompt_parts.append("Focus on visual elements like forms, code blocks, and unusual text patterns. Be precise in your findings.")
        elif task_type == 'Check for Google search block/CAPTCHA':
             prompt_parts.append("Analyze the screenshot of the Google search results page. Determine if the search was blocked or if a CAPTCHA is present. Respond with the required JSON structure.")
        elif task_type == 'Plan micro-steps for UI sub-task':
            prompt_parts.append(f"Given the overall sub-task goal '{task_context.get('overall_sub_task_goal')}' and the current page state (visual and text), break this down into a sequence of 1-5 very small, precise, actionable micro-steps for UI automation. Each step should be a clear instruction like 'Click the button labeled Next' or 'Input the value X into the field described as Y'.")
        else: prompt_parts.append("Analyze the provided context and visual information to complete the task as described.")

        if task_context.get('desired_output_format'):
            prompt_parts.append(f"\n**Output Format:** Respond ONLY with valid JSON matching this structure: {task_context['desired_output_format']}")
            if "JSON" in task_context.get('desired_output_format', ''): prompt_parts.append("\n```json")
        return "\n".join(prompt_parts)

    async def collect_insights(self) -> Dict[str, Any]:
        self.logger.debug("BrowsingAgent collect_insights called.")
        num_active_contexts = len(self.internal_state.get('browser_contexts', {}))
        proxy_stats = self.internal_state.get('proxy_stats', {})
        total_success = sum(s.get('success', 0) for s in proxy_stats.values()); total_failure = sum(s.get('failure', 0) for s in proxy_stats.values())
        total_attempts = total_success + total_failure; avg_proxy_success_rate = (total_success / total_attempts) if total_attempts > 0 else 1.0
        accounts_created_session = self.internal_state.get('accounts_created_this_session', 0)
        last_learning_ts_iso = None
        if isinstance(self.internal_state.get("last_learning_cycle_ts"), datetime):
            last_learning_ts_iso = self.internal_state["last_learning_cycle_ts"].isoformat()
        elif self.internal_state.get("last_learning_cycle_ts") is not None:
            last_learning_ts_iso = str(self.internal_state["last_learning_cycle_ts"])

        return {"agent_name": self.AGENT_NAME, "status": self.status, "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_contexts": num_active_contexts, "max_contexts": self.internal_state.get('max_concurrent_contexts'),
            "avg_proxy_success_rate": round(avg_proxy_success_rate, 3), "total_proxy_attempts": total_attempts,
            "accounts_created_session": accounts_created, "last_learning_ts": last_learning_ts_iso }

    async def stop(self, timeout: float = 60.0):
        if self._status in [self.STATUS_STOPPING, self.STATUS_STOPPED]:
            self.logger.info(f"{self.AGENT_NAME} stop requested but already {self._status}.")
            return
        self.logger.info(f"{self.AGENT_NAME} received stop signal. Initiating graceful shutdown (timeout: {timeout}s)...")
        self._status = self.STATUS_STOPPING
        self._stop_event.set()
        tasks_to_cancel = list(self._background_tasks)
        if tasks_to_cancel:
            self.logger.info(f"Cancelling {len(tasks_to_cancel)} BrowsingAgent background tasks...")
            for task in tasks_to_cancel:
                if task and not task.done(): task.cancel()
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        await self._close_browser_and_playwright()
        self.logger.info(f"BrowsingAgent resources closed. Finalizing stop.")
        self._status = self.STATUS_STOPPED
        self.logger.info(f"{self.AGENT_NAME} stopped.")

# --- End of agents/browsing_agent.py ---