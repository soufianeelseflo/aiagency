# Filename: agents/browsing_agent.py
# Description: Genius Agentic Browsing Agent - Handles web scraping, searching,
#              data extraction, proxy management, and Clay.com API integration.
# Version: 2.0 (Genius Agentic - Production Ready with Clay API)

import asyncio
import logging
import json
import os
import random
import re
import time
import aiohttp # For Clay.com API calls
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union, Tuple, AsyncGenerator, Type
from urllib.parse import urlparse, urljoin

# --- Core Framework Imports ---
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select, update, func
from sqlalchemy.exc import SQLAlchemyError

# --- Project Imports ---
try:
    from .base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
except ImportError:
    logging.warning("Production base agent not found, using GeniusAgentBase. Ensure base_agent_prod.py is used.")
    from .base_agent import GeniusAgentBase # Fallback

from models import KnowledgeFragment, AccountCredentials, StrategicDirective # Add relevant models
from config.settings import settings # Use validated settings
from utils.database import encrypt_data, decrypt_data # Use DB utils

# --- External Libraries ---
try:
    from playwright.async_api import async_playwright, Playwright, Browser, Page, Error as PlaywrightError
except ImportError:
    logging.critical("Playwright library not found. BrowsingAgent requires 'pip install playwright' and 'playwright install'.")
    raise
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
from bs4 import BeautifulSoup # For basic HTML parsing if needed

# Configure logger
logger = logging.getLogger(__name__)
# Configure dedicated operational logger
op_logger = logging.getLogger('OperationalLog') # Assuming setup elsewhere

# --- Meta Prompt ---
BROWSING_AGENT_META_PROMPT = """
You are the BrowsingAgent within the Synapse AI Sales System.
Your Core Mandate: Intelligently navigate the web, extract critical information, manage proxies effectively, and integrate with Clay.com API to support the agency's sales and data enrichment goals ($10k+/day -> $100M/8mo).
Key Responsibilities:
1.  **Intelligent Web Navigation & Scraping:** Access URLs, perform searches, analyze page structures (using LLM if needed), extract specific data points (text, links, structured data), and handle dynamic content (JavaScript rendering).
2.  **Proxy Management & Rotation:** Utilize and manage a pool of proxies (e.g., Smartproxy) for requests. Implement rotation strategies, handle authentication, and track proxy performance/bans. Provide proxies to other agents via Orchestrator if requested.
3.  **Data Extraction & Structuring:** Extract relevant information based on task requirements or LLM analysis. Structure the extracted data (e.g., into JSON) for downstream use.
4.  **Clay.com API Integration:** Directly call Clay.com API endpoints (as tasked by ThinkTool/Orchestrator) for data enrichment or workflow triggers using scraped data. Process and return Clay API results.
5.  **Error Handling & Resilience:** Gracefully handle website changes, anti-scraping measures, proxy failures, and timeouts using retries and fallback strategies.
6.  **Learning & Adaptation:** Learn effective scraping selectors, identify reliable proxy patterns, adapt to website structure changes, and potentially learn optimal times/methods for accessing specific sites.
7.  **Collaboration & Reporting:** Execute browsing/scraping/Clay tasks delegated by the Orchestrator. Report extracted data, summaries, errors, and operational insights. Log relevant findings to the Knowledge Base via ThinkTool.
**Goal:** Be the agency's reliable eyes and hands on the web and its interface to Clay.com, providing accurate data and enabling automated workflows while navigating complexities and maintaining operational integrity.
"""

class BrowsingAgent(GeniusAgentBase):
    """
    Browsing Agent (Genius Level): Navigates the web, scrapes data, manages proxies,
    and integrates directly with the Clay.com API.
    Version: 2.0
    """
    AGENT_NAME = "BrowsingAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any, smartproxy_password: Optional[str] = None):
        """Initializes the BrowsingAgent."""
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, session_maker=session_maker)
        self.meta_prompt = BROWSING_AGENT_META_PROMPT
        self.think_tool = orchestrator.agents.get('think') # Reference ThinkTool if needed

        # Store secrets passed directly
        self._smartproxy_password = smartproxy_password
        self._smartproxy_user = self.config.get("SMARTPROXY_USER")
        self._smartproxy_host = self.config.get("SMARTPROXY_HOST")
        self._smartproxy_port = self.config.get("SMARTPROXY_PORT")

        # --- Internal State Initialization ---
        self.internal_state = getattr(self, 'internal_state', {})
        self.internal_state['playwright_instance'] = None # type: Optional[Playwright]
        self.internal_state['browser_instance'] = None # type: Optional[Browser]
        self.internal_state['active_pages'] = {} # page_id -> Page object
        self.internal_state['proxy_pool'] = self._load_proxies() # Load initial proxy config
        self.internal_state['current_proxy_index'] = 0
        self.internal_state['proxy_stats'] = {} # proxy_url -> {'success': N, 'failure': N, 'last_used': datetime}
        self.internal_state['user_agent'] = self.config.get("BROWSER_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36") # Example User Agent
        self.internal_state['max_concurrent_pages'] = int(self.config.get("BROWSER_MAX_CONCURRENT_PAGES", 5))
        self.internal_state['page_semaphore'] = asyncio.Semaphore(self.internal_state['max_concurrent_pages'])
        self.internal_state['default_timeout_ms'] = int(self.config.get("BROWSER_DEFAULT_TIMEOUT_MS", 60000)) # 60 seconds

        self.logger.info(f"{self.AGENT_NAME} v2.0 initialized. Max Pages: {self.internal_state['max_concurrent_pages']}")

    def _load_proxies(self) -> List[Dict]:
        """Loads proxy configurations."""
        proxies = []
        # Example: Load Smartproxy rotating residential proxy
        if self._smartproxy_user and self._smartproxy_password and self._smartproxy_host and self._smartproxy_port:
            # Format for Playwright proxy setting
            proxy_server = f"http://{self._smartproxy_host}:{self._smartproxy_port}"
            proxies.append({
                "server": proxy_server,
                "username": self._smartproxy_user,
                "password": self._smartproxy_password,
                "type": "smartproxy_rotating" # Identifier
            })
            self.logger.info(f"Loaded Smartproxy configuration: {proxy_server}")
        else:
            self.logger.warning("Smartproxy credentials/host/port missing. Proxy functionality limited.")
        # TODO: Extend to load other proxy types or lists from config/DB if needed
        return proxies

    async def log_operation(self, level: str, message: str):
        """Helper to log to the operational log file."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err: logger.error(f"Failed to write to operational log: {log_err}")

    # --- Playwright Lifecycle Management ---

    async def _ensure_browser_running(self):
        """Ensures Playwright and a browser instance are running."""
        if self.internal_state.get('browser_instance') and self.internal_state['browser_instance'].is_connected():
            return # Already running

        self.logger.info("Initializing Playwright and launching browser...")
        try:
            if not self.internal_state.get('playwright_instance'):
                self.internal_state['playwright_instance'] = await async_playwright().start()

            # Select proxy for browser launch (optional, can be set per context/page)
            # proxy_config = self._get_proxy_config() # Get proxy settings if needed for the whole browser

            self.internal_state['browser_instance'] = await self.internal_state['playwright_instance'].chromium.launch(
                headless=True, # Run headless for server environment
                # proxy=proxy_config # Apply proxy globally if needed
                args=['--no-sandbox', '--disable-setuid-sandbox'] # Common args for Linux/Docker
            )
            self.logger.info("Playwright browser launched successfully.")
        except Exception as e:
            self.logger.critical(f"Failed to launch Playwright browser: {e}", exc_info=True)
            self._status = self.STATUS_ERROR
            await self._report_error(f"Failed to launch browser: {e}")
            raise # Re-raise to indicate critical failure

    async def _close_browser(self):
        """Closes the browser and stops Playwright if running."""
        self.logger.info("Attempting to close Playwright browser...")
        browser = self.internal_state.get('browser_instance')
        if browser and browser.is_connected():
            try:
                await browser.close()
                self.logger.info("Playwright browser closed.")
            except Exception as e:
                self.logger.error(f"Error closing browser: {e}", exc_info=True)
        self.internal_state['browser_instance'] = None

        playwright_instance = self.internal_state.get('playwright_instance')
        if playwright_instance:
            try:
                await playwright_instance.stop()
                self.logger.info("Playwright instance stopped.")
            except Exception as e:
                self.logger.error(f"Error stopping Playwright: {e}", exc_info=True)
        self.internal_state['playwright_instance'] = None
        self.internal_state['active_pages'] = {} # Clear active pages

    async def _get_new_page(self) -> Tuple[str, Page]:
        """Gets a new browser page, respecting concurrency limits."""
        await self._ensure_browser_running() # Make sure browser is up
        browser = self.internal_state.get('browser_instance')
        if not browser: raise RuntimeError("Browser instance is not available.")

        await self.internal_state['page_semaphore'].acquire() # Wait for available slot
        page_id = str(uuid.uuid4())
        page = None
        try:
            # --- Proxy Configuration Per Context ---
            proxy_config = self._get_proxy_config() # Get proxy for this specific context

            context = await browser.new_context(
                user_agent=self.internal_state['user_agent'],
                proxy=proxy_config, # Apply proxy per-context
                java_script_enabled=True,
                ignore_https_errors=True, # Be cautious with this in production
                # viewport={'width': 1920, 'height': 1080} # Set viewport if needed
            )
            page = await context.new_page()
            page.set_default_timeout(self.internal_state['default_timeout_ms'])
            self.internal_state['active_pages'][page_id] = page
            self.logger.info(f"Created new browser page (ID: {page_id}). Active pages: {len(self.internal_state['active_pages'])}")
            return page_id, page
        except Exception as e:
            self.internal_state['page_semaphore'].release() # Release semaphore on error
            self.logger.error(f"Failed to create new browser page: {e}", exc_info=True)
            if page: await self._close_page(page_id, page) # Attempt cleanup
            raise # Re-raise

    async def _close_page(self, page_id: str, page: Optional[Page] = None):
        """Closes a specific browser page and releases semaphore."""
        page_to_close = page or self.internal_state['active_pages'].get(page_id)
        if page_to_close:
            try:
                await page_to_close.close()
                self.logger.info(f"Closed browser page (ID: {page_id}).")
            except Exception as e:
                self.logger.warning(f"Error closing page {page_id}: {e}")
            finally:
                self.internal_state['active_pages'].pop(page_id, None)
                self.internal_state['page_semaphore'].release() # Release semaphore
                self.logger.debug(f"Semaphore released. Active pages: {len(self.internal_state['active_pages'])}")
        else:
             # If page wasn't found but ID exists, ensure semaphore is released
             if page_id in self.internal_state['active_pages']:
                 self.internal_state['active_pages'].pop(page_id, None)
                 self.internal_state['page_semaphore'].release()
                 self.logger.warning(f"Page {page_id} not found in active list, but released semaphore.")

    # --- Proxy Management ---

    def _get_proxy_config(self) -> Optional[Dict]:
        """Selects the next proxy from the pool for Playwright."""
        if not self.internal_state['proxy_pool']:
            self.logger.debug("No proxies configured.")
            return None

        num_proxies = len(self.internal_state['proxy_pool'])
        self.internal_state['current_proxy_index'] = (self.internal_state['current_proxy_index']) % num_proxies
        selected_proxy = self.internal_state['proxy_pool'][self.internal_state['current_proxy_index']]
        self.internal_state['current_proxy_index'] += 1 # Rotate for next time

        proxy_url = selected_proxy.get("server")
        self.logger.info(f"Selected proxy for next context/page: {proxy_url}")

        # Update stats (can be enhanced to track success/failure)
        now = datetime.now(timezone.utc)
        if proxy_url not in self.internal_state['proxy_stats']:
             self.internal_state['proxy_stats'][proxy_url] = {'success': 0, 'failure': 0, 'last_used': now}
        else: self.internal_state['proxy_stats'][proxy_url]['last_used'] = now

        # Return format expected by Playwright's context/launch `proxy` option
        return {
            "server": selected_proxy["server"],
            "username": selected_proxy.get("username"),
            "password": selected_proxy.get("password")
        }

    async def get_proxy_for_account(self, account_identifier: Optional[str] = None, purpose: str = "general", target_url: Optional[str] = None) -> Optional[str]:
        """Provides a proxy URL, potentially associated with an account."""
        # This method is called by Orchestrator.get_proxy
        # Simple implementation: return next available proxy URL string
        # TODO: Implement more sophisticated logic if proxies need to be pinned to accounts
        # or selected based on target_url/purpose.
        selected_proxy_config = self._get_proxy_config()
        if selected_proxy_config:
            # Construct URL string if username/password exist
            url_parts = urlparse(selected_proxy_config["server"])
            user = selected_proxy_config.get("username")
            pwd = selected_proxy_config.get("password")
            if user and pwd:
                proxy_url_str = f"{url_parts.scheme}://{user}:{pwd}@{url_parts.netloc}"
            else:
                proxy_url_str = selected_proxy_config["server"]
            self.logger.info(f"Providing proxy URL: {proxy_url_str} for purpose: {purpose}")
            return proxy_url_str
        else:
            self.logger.warning(f"No proxy available to provide for purpose: {purpose}")
            return None

    def _update_proxy_stats(self, proxy_url: Optional[str], success: bool):
        """Updates success/failure count for a used proxy."""
        if not proxy_url or proxy_url not in self.internal_state['proxy_stats']:
            return
        if success:
            self.internal_state['proxy_stats'][proxy_url]['success'] += 1
        else:
            self.internal_state['proxy_stats'][proxy_url]['failure'] += 1
        # TODO: Add logic to temporarily disable failing proxies

    # --- Clay.com API Integration ---

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(aiohttp.ClientError))
    async def call_clay_api(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Makes a direct API call to the specified Clay.com endpoint.
        (Moved from ThinkTool as per user request)
        """
        api_key = self.config.get_secret("CLAY_API_KEY")
        if not api_key:
            self.logger.error("Clay.com API key (CLAY_API_KEY) not found.")
            return {"status": "failure", "message": "Clay API key not configured."}
        if not endpoint.startswith('/'): endpoint = '/' + endpoint

        clay_url = f"https://api.clay.com{endpoint}"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}

        await self._internal_think(f"Calling Clay API via BrowsingAgent: {endpoint}", details=data)
        await self.log_operation('debug', f"Calling Clay API endpoint: {endpoint}")

        try:
            timeout = aiohttp.ClientTimeout(total=60) # 60 second timeout
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                async with session.post(clay_url, json=data) as response:
                    response_status = response.status
                    try: response_data = await response.json(content_type=None) # Try parsing JSON regardless of type
                    except Exception: response_data = await response.text() # Fallback to text

                    if 200 <= response_status < 300:
                        self.logger.info(f"Clay API call to {endpoint} successful (Status: {response_status}).")
                        # Report expense via orchestrator
                        await self.orchestrator.report_expense(self.AGENT_NAME, 0.02, "API_Clay", f"Clay API Call: {endpoint}") # Example cost
                        return {"status": "success", "data": response_data}
                    else:
                        self.logger.error(f"Clay API call to {endpoint} failed. Status: {response_status}, Response: {str(response_data)[:500]}...")
                        return {"status": "failure", "message": f"Clay API Error (Status {response_status})", "details": response_data}
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout calling Clay API endpoint: {endpoint}")
            return {"status": "error", "message": f"Clay API call timed out"}
        except aiohttp.ClientError as e:
            self.logger.error(f"Network/Connection error calling Clay API endpoint {endpoint}: {e}")
            raise # Re-raise for tenacity retry
        except Exception as e:
            self.logger.error(f"Unexpected error during Clay API call to {endpoint}: {e}", exc_info=True)
            return {"status": "error", "message": f"Clay API call exception: {e}"}

    async def _process_clay_result(self, clay_data: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes data returned from Clay API within the browsing agent's context.
        (Moved from ThinkTool as per user request)
        """
        self.logger.info(f"Processing Clay API result within BrowsingAgent.")
        await self._internal_think("Processing Clay API result", details=clay_data)
        processed_result = {"status": "success", "message": "Clay result processed.", "processed_data": None}

        # Example: Extract key info and maybe log it or return it
        # This depends heavily on the task and the expected Clay response structure
        extracted_info = {}
        if isinstance(clay_data, dict):
            extracted_info['email'] = clay_data.get('email') or clay_data.get('person', {}).get('email')
            extracted_info['linkedin_url'] = clay_data.get('linkedin_url')
            extracted_info['job_title'] = clay_data.get('job_title') or clay_data.get('person', {}).get('title')
            extracted_info['company_name'] = clay_data.get('company', {}).get('name')
            # Add more extraction logic as needed based on the Clay endpoint used

            processed_result["processed_data"] = extracted_info
            self.logger.info(f"Extracted from Clay result: {extracted_info}")

            # Optionally log to KB via ThinkTool if significant insight gained
            if self.think_tool and extracted_info.get('email'):
                try:
                    log_task = {
                        "action": "log_knowledge_fragment",
                        "fragment_data": {
                            "agent_source": self.AGENT_NAME, "data_type": "clay_enrichment_result",
                            "content": extracted_info, "tags": ["clay", "enrichment", "lead_data", "browsing_agent"],
                            "relevance_score": 0.8, "source_reference": f"BrowsingTask_{task_context.get('id', 'N/A')}"
                        }
                    }
                    await self.orchestrator.delegate_task("ThinkTool", log_task)
                except Exception as log_err:
                    self.logger.warning(f"Failed to log Clay result to KB via ThinkTool: {log_err}")

        else:
             self.logger.warning(f"Received non-dict data for Clay result processing: {type(clay_data)}")
             processed_result = {"status": "warning", "message": "Received non-dictionary Clay data.", "processed_data": clay_data}

        return processed_result

    # --- Core Abstract Method Implementations ---

    async def plan_task(self, task_details: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Generates a plan for browsing, scraping, or Clay API tasks."""
        action = task_details.get('action')
        plan = []
        await self._internal_think(f"Planning task: {action}", details=task_details)

        if action == 'scrape_website':
            url = task_details.get('url')
            selectors = task_details.get('selectors') # e.g., {"title": "h1", "description": ".desc"}
            extraction_prompt = task_details.get('extraction_prompt') # For LLM-based extraction
            if not url: raise ValueError("Missing 'url' for scrape_website task.")

            plan.append({"step": 1, "action": "open_page", "tool": "browser", "params": {"url": url}})
            if selectors:
                plan.append({"step": 2, "action": "extract_by_selectors", "tool": "browser", "params": {"selectors": selectors}})
            elif extraction_prompt:
                plan.append({"step": 2, "action": "extract_with_llm", "tool": "browser", "params": {"prompt": extraction_prompt}})
            else:
                # Default: Extract main content text if no specific instructions
                plan.append({"step": 2, "action": "extract_main_content", "tool": "browser", "params": {}})
            plan.append({"step": 3, "action": "close_page", "tool": "browser", "params": {}}) # Ensure cleanup

        elif action == 'perform_search_and_summarize':
            query = task_details.get('query')
            num_results = task_details.get('num_results', 3)
            if not query: raise ValueError("Missing 'query' for search task.")
            # Simple plan: Use an external search tool/API via orchestrator (if available)
            # Or implement basic browser-based search
            plan.append({"step": 1, "action": "execute_search", "tool": "search_engine", "params": {"query": query, "num_results": num_results}})
            plan.append({"step": 2, "action": "summarize_results", "tool": "llm", "params": {}}) # Use LLM to summarize

        elif action == 'call_clay_api_via_browser_agent': # Specific action name
            endpoint = task_details.get('endpoint')
            payload = task_details.get('payload')
            if not endpoint or not payload: raise ValueError("Missing 'endpoint' or 'payload' for Clay API call task.")
            plan.append({"step": 1, "action": "execute_clay_call", "tool": "clay_api", "params": {"endpoint": endpoint, "data": payload}})
            plan.append({"step": 2, "action": "process_clay_response", "tool": "internal", "params": {}})

        # Add more complex planning logic here if needed
        # e.g., multi-page navigation, form filling

        else:
            self.logger.warning(f"No specific plan generated for action '{action}'. Assuming direct execution if possible.")
            # Return None or a single step plan if execute_step can handle it directly
            return None # Let execute_task handle direct actions

        self.logger.info(f"Generated plan with {len(plan)} steps for action '{action}'.")
        return plan

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=3, max=10), retry=retry_if_exception_type(PlaywrightError))
    async def execute_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a single step of a browsing/scraping/Clay plan."""
        step_action = step.get('action')
        tool = step.get('tool')
        params = step.get('params', {})
        step_num = step.get('step', '?')
        result = {"status": "failure", "message": f"Step action '{step_action}' not implemented."}
        page_id = task_context.get("current_page_id") # Track current page across steps
        page = self.internal_state['active_pages'].get(page_id) if page_id else None
        proxy_url = page.context.proxy.server if page and page.context.proxy else None # Get proxy used for this page

        await self._internal_think(f"Executing step {step_num}: {step_action}", details=params)

        try:
            # --- Browser Actions ---
            if tool == 'browser':
                if step_action == 'open_page':
                    url = params.get('url')
                    if not url: raise ValueError("Missing 'url' parameter.")
                    if page_id and page: await self._close_page(page_id, page) # Close previous page if any
                    page_id, page = await self._get_new_page()
                    task_context["current_page_id"] = page_id # Store for subsequent steps
                    self.logger.info(f"Navigating page {page_id} to URL: {url}")
                    response = await page.goto(url, wait_until='domcontentloaded') # Use 'domcontentloaded' or 'load' or 'networkidle'
                    status_code = response.status if response else None
                    self.logger.info(f"Page {page_id} navigated to {url}. Status: {status_code}")
                    if not response or not response.ok:
                         raise PlaywrightError(f"Failed to load page {url}. Status: {status_code}")
                    # Optional: Add delay or wait for specific element after load
                    await page.wait_for_timeout(random.uniform(1500, 3000)) # Small random delay
                    result = {"status": "success", "message": f"Page opened: {url}", "page_id": page_id, "status_code": status_code}
                    self._update_proxy_stats(proxy_url, True)

                elif step_action == 'extract_by_selectors':
                    if not page: raise RuntimeError("No active page found for extraction.")
                    selectors = params.get('selectors')
                    if not selectors or not isinstance(selectors, dict): raise ValueError("Invalid 'selectors' parameter.")
                    extracted_data = {}
                    self.logger.info(f"Extracting data using selectors on page {page_id}.")
                    for key, selector in selectors.items():
                        try:
                            element = page.locator(selector).first # Take the first match
                            # Try different ways to get text content
                            content = await element.text_content(timeout=5000) or await element.inner_text(timeout=5000) or await element.get_attribute('value', timeout=5000) or ""
                            extracted_data[key] = content.strip()
                        except PlaywrightError as pe: self.logger.warning(f"Selector '{selector}' for key '{key}' failed on page {page_id}: {pe}"); extracted_data[key] = None
                        except Exception as e: self.logger.warning(f"Error extracting selector '{selector}' for key '{key}': {e}"); extracted_data[key] = None
                    result = {"status": "success", "message": "Data extracted via selectors.", "extracted_data": extracted_data}
                    self._update_proxy_stats(proxy_url, True)

                elif step_action == 'extract_with_llm':
                    if not page: raise RuntimeError("No active page found for LLM extraction.")
                    extraction_prompt = params.get('prompt')
                    if not extraction_prompt: raise ValueError("Missing 'prompt' for LLM extraction.")
                    self.logger.info(f"Extracting data using LLM analysis on page {page_id}.")
                    try:
                        # Get page content (consider simplifying HTML)
                        # page_content = await page.content() # Full HTML
                        # Use evaluate to get text content, potentially cleaner
                        body_text = await page.evaluate("document.body.innerText")
                        max_len = 8000 # Limit context for LLM
                        content_snippet = body_text[:max_len] + ("..." if len(body_text) > max_len else "")

                        llm_task_context = {
                            "task": "Extract structured data from webpage content based on prompt",
                            "webpage_content_snippet": content_snippet,
                            "extraction_instructions": extraction_prompt,
                            "current_url": page.url,
                            "desired_output_format": "JSON containing the extracted data as requested by the instructions."
                        }
                        llm_prompt = await self.generate_dynamic_prompt(llm_task_context)
                        llm_response_str = await self.orchestrator.call_llm(
                            agent_name=self.AGENT_NAME, prompt=llm_prompt, temperature=0.2,
                            max_tokens=1500, is_json_output=True
                        )
                        if llm_response_str:
                             # Basic JSON parsing (improve if needed)
                             try:
                                 extracted_data = json.loads(llm_response_str[llm_response_str.find('{'):llm_response_str.rfind('}')+1])
                                 result = {"status": "success", "message": "Data extracted via LLM.", "extracted_data": extracted_data}
                             except json.JSONDecodeError: result = {"status": "failure", "message": "LLM returned non-JSON data.", "raw_output": llm_response_str}
                        else: result = {"status": "failure", "message": "LLM analysis returned no response."}
                    except Exception as llm_err: result = {"status": "failure", "message": f"Error during LLM extraction: {llm_err}"}
                    self._update_proxy_stats(proxy_url, True) # Assume proxy worked if page loaded

                elif step_action == 'extract_main_content':
                     if not page: raise RuntimeError("No active page found for extraction.")
                     self.logger.info(f"Extracting main text content from page {page_id}.")
                     try:
                         body_text = await page.evaluate("document.body.innerText")
                         result = {"status": "success", "message": "Main content extracted.", "extracted_data": {"main_text": body_text.strip()}}
                     except Exception as text_err: result = {"status": "failure", "message": f"Error extracting main text: {text_err}"}
                     self._update_proxy_stats(proxy_url, True)

                elif step_action == 'close_page':
                    if page_id:
                        await self._close_page(page_id, page)
                        task_context["current_page_id"] = None # Clear page ID from context
                        result = {"status": "success", "message": "Page closed."}
                    else: result = {"status": "warning", "message": "No active page ID found to close."}

                # Add other browser actions: click, fill_form, screenshot, etc.

            # --- Search Engine Action ---
            elif tool == 'search_engine':
                 if step_action == 'execute_search':
                      query = params.get('query')
                      num_results = params.get('num_results', 3)
                      if not query: raise ValueError("Missing 'query' parameter.")
                      self.logger.info(f"Executing search for: '{query}'")
                      # Delegate to an external search tool/API via orchestrator
                      # Example: Using a hypothetical 'google_search' tool
                      search_tool_result = await self._execute_tool('google_search', {'query': query, 'num_results': num_results}, step_num)
                      if search_tool_result.get('status') == 'success':
                           task_context['search_results'] = search_tool_result.get('results', []) # Store for next step
                           result = {"status": "success", "message": f"Search completed. Found {len(task_context['search_results'])} results.", "result_data": task_context['search_results']}
                      else: result = search_tool_result # Propagate failure
                 else: result = {"status": "failure", "message": f"Unknown search engine action: {step_action}"}

            # --- LLM Action (e.g., Summarization) ---
            elif tool == 'llm':
                 if step_action == 'summarize_results':
                      search_results = task_context.get('search_results')
                      if not search_results: raise ValueError("No search results found in context to summarize.")
                      self.logger.info(f"Summarizing {len(search_results)} search results using LLM.")
                      # Prepare content for LLM
                      content_to_summarize = "\n---\n".join([f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nSnippet: {r.get('snippet', 'N/A')}" for r in search_results])
                      max_len = 8000
                      content_snippet = content_to_summarize[:max_len] + ("..." if len(content_to_summarize) > max_len else "")

                      llm_task_context = {
                          "task": "Summarize the key information from the provided search results",
                          "search_results_content": content_snippet,
                          "desired_output_format": "A concise bulleted list summarizing the main findings."
                      }
                      llm_prompt = await self.generate_dynamic_prompt(llm_task_context)
                      summary = await self.orchestrator.call_llm(
                          agent_name=self.AGENT_NAME, prompt=llm_prompt, temperature=0.3, max_tokens=1000
                      )
                      if summary: result = {"status": "success", "message": "Search results summarized.", "summary": summary, "result_data": summary}
                      else: result = {"status": "failure", "message": "LLM summarization failed."}
                 else: result = {"status": "failure", "message": f"Unknown LLM action: {step_action}"}

            # --- Clay API Actions ---
            elif tool == 'clay_api':
                 if step_action == 'execute_clay_call':
                      endpoint = params.get('endpoint')
                      data = params.get('data')
                      if not endpoint or not data: raise ValueError("Missing endpoint/data for Clay API call.")
                      clay_result = await self.call_clay_api(endpoint, data) # Call the method within this agent
                      task_context['last_clay_response'] = clay_result # Store response for next step
                      result = clay_result # Pass the result directly
                 else: result = {"status": "failure", "message": f"Unknown Clay API action: {step_action}"}

            # --- Internal Processing Actions ---
            elif tool == 'internal':
                 if step_action == 'process_clay_response':
                      clay_response = task_context.get('last_clay_response')
                      if not clay_response: raise ValueError("No Clay API response found in context.")
                      if clay_response.get("status") == "success":
                           processed_data = await self._process_clay_result(clay_response.get("data"), task_context)
                           result = {"status": "success", "message": "Clay response processed.", "result_data": processed_data.get("processed_data")}
                      else:
                           # Pass through the failure/error from the API call step
                           result = {"status": clay_response.get("status", "failure"), "message": f"Clay API call failed: {clay_response.get('message', 'Unknown error')}", "details": clay_response.get("details")}
                 else: result = {"status": "failure", "message": f"Unknown internal action: {step_action}"}

            # --- Unknown Tool ---
            else:
                result = {"status": "failure", "message": f"Unsupported tool type: {tool}"}

        except PlaywrightError as pe:
            self.logger.error(f"Playwright error during step {step_num} ('{step_action}'): {pe}", exc_info=True)
            result = {"status": "failure", "message": f"Browser error: {pe}"}
            # Update proxy stats on failure if proxy was involved
            self._update_proxy_stats(proxy_url, False)
            # Attempt to close the problematic page
            if page_id: await self._close_page(page_id, page)
            task_context["current_page_id"] = None
            raise # Re-raise Playwright errors to trigger tenacity retry if configured at task level
        except Exception as e:
            self.logger.error(f"Unexpected error during step {step_num} ('{step_action}'): {e}", exc_info=True)
            result = {"status": "failure", "message": f"Unexpected error: {e}"}
            self._update_proxy_stats(proxy_url, False) # Assume failure if exception occurs
            if page_id: await self._close_page(page_id, page) # Cleanup on error
            task_context["current_page_id"] = None
            await self._report_error(f"Step {step_num} ('{step_action}') failed: {e}", task_context.get('id'))

        return result

    async def learning_loop(self):
        """Autonomous learning cycle for the browsing agent."""
        self.logger.info("BrowsingAgent learning_loop started.")
        while not self._stop_event.is_set():
            try:
                await self._internal_think("Starting learning cycle: Analyzing proxy performance and website structures.")

                # 1. Analyze Proxy Performance
                poor_proxies = []
                for proxy, stats in self.internal_state['proxy_stats'].items():
                    total_attempts = stats['success'] + stats['failure']
                    if total_attempts > 10 and (stats['failure'] / total_attempts) > 0.5:
                        poor_proxies.append(proxy)
                        self.logger.warning(f"Proxy {proxy} identified as potentially poor performing (Fail rate: {stats['failure']/total_attempts:.1%}).")
                # TODO: Implement logic to temporarily disable or deprioritize poor proxies.

                # 2. Analyze Recent Scraping Failures (Needs logging mechanism)
                # - Query KB for recent scraping failures logged by this agent.
                # - Identify common failing URLs or selectors.
                # - Trigger LLM analysis: "Website structure for [URL] likely changed. Analyze current structure and suggest new selectors for [target data]."
                # - Update KB or internal state with new potential selectors.

                # 3. Identify Opportunities for Clay Enrichment
                # - Query KB for recently scraped company/person data lacking key info (e.g., email).
                # - Generate a StrategicDirective for ThinkTool: "Found potential leads [list] needing enrichment via Clay. Please prioritize."

                self.internal_state["last_learning_cycle_ts"] = datetime.now(timezone.utc)
                self.logger.info("BrowsingAgent learning cycle complete.")

                # Sleep for a configured interval
                learn_interval = int(self.config.get("BROWSER_LEARNING_INTERVAL_S", 3600 * 4)) # Default 4 hours
                await asyncio.sleep(learn_interval)

            except asyncio.CancelledError:
                self.logger.info("BrowsingAgent learning loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in BrowsingAgent learning loop: {e}", exc_info=True)
                await self._report_error(f"Learning loop error: {e}")
                await asyncio.sleep(60 * 15) # Wait longer after error

    async def self_critique(self) -> Dict[str, Any]:
        """Evaluates browsing performance, proxy health, and resource usage."""
        self.logger.info(f"{self.AGENT_NAME}: Performing self-critique.")
        critique = {"status": "ok", "feedback": "Critique pending analysis."}
        try:
            num_active_pages = len(self.internal_state['active_pages'])
            proxy_stats_summary = {p: f"S:{s['success']}/F:{s['failure']}" for p, s in self.internal_state['proxy_stats'].items()}

            critique['resource_usage'] = {"active_pages": num_active_pages, "max_pages": self.internal_state['max_concurrent_pages']}
            critique['proxy_stats'] = proxy_stats_summary

            feedback_points = [f"Active Pages: {num_active_pages}/{self.internal_state['max_concurrent_pages']}."]
            high_failure_proxies = sum(1 for stats in self.internal_state['proxy_stats'].values() if (stats['success'] + stats['failure']) > 5 and stats['failure'] / (stats['success'] + stats['failure']) > 0.4)
            if high_failure_proxies > 0:
                feedback_points.append(f"WARNING: {high_failure_proxies} proxies show high failure rates. Check proxy provider/config.")
                critique['status'] = 'warning'
            if num_active_pages >= self.internal_state['max_concurrent_pages']:
                 feedback_points.append("INFO: Approaching maximum concurrent page limit.")

            # TODO: Query DB/KB for recent task success/failure rates specific to this agent.

            critique['feedback'] = " ".join(feedback_points)

        except Exception as e:
            self.logger.error(f"Error during self-critique: {e}", exc_info=True)
            critique['status'] = 'error'; critique['feedback'] = f"Critique failed: {e}"
        return critique

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Constructs context-rich prompts for LLM calls related to browsing."""
        self.logger.debug(f"Generating dynamic prompt for BrowsingAgent task: {task_context.get('task')}")
        prompt_parts = [self.meta_prompt] # Start with BrowsingAgent's meta-prompt

        prompt_parts.append("\n--- Current Task Context ---")
        # Add specific task details, limiting length of large items
        for key, value in task_context.items():
            value_str = ""
            max_len = 4000 # Allow more context for webpage content
            if key == 'webpage_content_snippet': max_len = 8000
            if isinstance(value, str): value_str = value[:max_len] + ("..." if len(value) > max_len else "")
            elif isinstance(value, (int, float, bool)): value_str = str(value)
            elif isinstance(value, dict): value_str = json.dumps(value, default=str)[:max_len] + "..."
            elif isinstance(value, list): value_str = json.dumps(value, default=str)[:max_len] + "..."
            else: value_str = str(value)[:max_len] + "..."
            prompt_parts.append(f"**{key.replace('_', ' ').title()}**: {value_str}")

        prompt_parts.append("\n--- Instructions ---")
        task_type = task_context.get('task')
        if task_type == 'Extract structured data from webpage content based on prompt':
            prompt_parts.append(f"Analyze the 'Webpage Content Snippet' from URL '{task_context.get('current_url', 'N/A')}'.")
            prompt_parts.append(f"Follow these extraction instructions precisely: {task_context.get('extraction_instructions', 'Extract key information.')}")
            prompt_parts.append("If data is missing, use null or an empty string for that field.")
            prompt_parts.append(f"**Output Format:** {task_context.get('desired_output_format')}")
        elif task_type == 'Summarize the key information from the provided search results':
             prompt_parts.append("Review the 'Search Results Content'.")
             prompt_parts.append("Identify the most relevant and important findings related to the original query.")
             prompt_parts.append(f"**Output Format:** {task_context.get('desired_output_format')}")
        # Add prompts for other LLM-assisted browsing tasks (e.g., deciding next navigation step)
        else:
            prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")

        if "JSON" in task_context.get('desired_output_format', ''): prompt_parts.append("\n```json")

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for BrowsingAgent (length: {len(final_prompt)} chars)")
        return final_prompt

    async def collect_insights(self) -> Dict[str, Any]:
        """Collects insights about browsing activity and proxy status."""
        self.logger.debug("BrowsingAgent collect_insights called.")
        num_active_pages = len(self.internal_state.get('active_pages', {}))
        # Calculate overall proxy success rate (simple average)
        total_success = sum(s['success'] for s in self.internal_state['proxy_stats'].values())
        total_failure = sum(s['failure'] for s in self.internal_state['proxy_stats'].values())
        total_attempts = total_success + total_failure
        avg_proxy_success_rate = (total_success / total_attempts) if total_attempts > 0 else 1.0

        insights = {
            "agent_name": self.AGENT_NAME, "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_pages": num_active_pages,
            "max_pages": self.internal_state.get('max_concurrent_pages'),
            "proxy_pool_size": len(self.internal_state.get('proxy_pool', [])),
            "avg_proxy_success_rate": round(avg_proxy_success_rate, 3),
            "total_proxy_attempts": total_attempts,
            # Add metrics like pages_scraped_session, data_points_extracted_session if tracked
        }
        return insights

    async def stop(self, timeout: float = 30.0):
        """Override stop to close browser and cancel tasks."""
        self.logger.info(f"{self.AGENT_NAME} received stop signal.")
        # Signal loops to stop
        self._stop_event.set()
        # Close browser first
        await self._close_browser()
        # Cancel background tasks specific to this agent
        tasks_to_cancel = list(self._background_tasks)
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
        if tasks_to_cancel:
            self.logger.info(f"Waiting for {len(tasks_to_cancel)} BrowsingAgent background tasks to cancel...")
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            self.logger.info("BrowsingAgent background tasks cancellation complete.")
        # Call base class stop for any common cleanup
        await super().stop(timeout)

    async def run(self):
        """Main run loop: Processes task queue and runs learning loop."""
        if await self._run_lock.acquire(blocking=False): # Use non-blocking acquire
            try:
                if self._status == self.STATUS_RUNNING:
                    self.logger.warning("Run requested but agent is already running.")
                    return
                if self._status == self.STATUS_STOPPING:
                    self.logger.warning("Run requested but agent is currently stopping.")
                    return

                self._status = self.STATUS_RUNNING
                self._stop_event.clear()
                self.internal_state["errors_encountered_session"] = 0
                self.internal_state["tasks_processed_session"] = 0
                self.internal_state["last_error_ts"] = None
                self.internal_state["last_error_details"] = None

                # Ensure browser is ready before starting loops that might need it
                await self._ensure_browser_running()

                # Create and store the main run task (which starts sub-tasks)
                run_task = asyncio.create_task(self._run_main_loop(), name=f"{self.agent_name}_MainLoop")
                self._background_tasks.add(run_task)
                self.logger.info("Main run loop initiated.")

            except Exception as e:
                 self.logger.critical(f"Failed to start BrowsingAgent run loop: {e}", exc_info=True)
                 self._status = self.STATUS_ERROR
                 await self._report_error(f"Failed to start run loop: {e}")
            finally:
                self._run_lock.release()
        else:
            self.logger.warning("Run lock already held, skipping run initiation.")


# --- End of agents/browsing_agent.py ---