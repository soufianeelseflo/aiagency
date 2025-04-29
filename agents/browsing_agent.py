# browsing_agent.py
# Single-file refactored version of the Browsing Agent
# Target: Production-grade, maintainable, functionally identical, single-file structure.

import asyncio
import logging
import random
import os
import sqlite3
import time
import aiohttp
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Page, Response, Request
from sqlalchemy.ext.asyncio import async_sessionmaker
from faker import Faker
import psutil
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioRestException
import json
import imaplib
import email
import re
from datetime import datetime, timedelta
import sqlalchemy # Explicitly import sqlalchemy for text()
from typing import List, Dict, Optional, Any, Tuple, Union

# ==============================================================================
# == Logger Configuration                                                     ==
# ==============================================================================

# Configure dedicated operational logger
op_logger = logging.getLogger('OperationalLog')
op_logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Ensure the log directory exists if needed
# os.makedirs('logs', exist_ok=True)
try:
    # Use append mode for the operational log
    log_file_handler = logging.FileHandler('operations_log.md', mode='a', encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    op_logger.addHandler(log_file_handler)
    op_logger.propagate = False # Prevent double logging
except Exception as log_setup_err:
    # Fallback to print if file logging fails critically during setup
    print(f"CRITICAL: Failed to set up operational file logger: {log_setup_err}")
    # Optionally, configure a basic console logger for op_logger as fallback
    # op_logger.addHandler(logging.StreamHandler())
    # op_logger.propagate = False # Still prevent propagation

# Standard logger for agent-specific debug/errors
logger = logging.getLogger(__name__)
# Ensure standard logger has a handler if not configured elsewhere (e.g., basic console)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO) # Set default level if needed

fake = Faker()

# ==============================================================================
# == Constants                                                                ==
# ==============================================================================

# Meta prompt for LLM interactions (used by StrategyManager and potentially others)
BROWSING_AGENT_META_PROMPT_UPDATED = """
You are a genius AI BrowsingAgent designed to create infinite free trial accounts for an AI agency targeting $6,000 profit in 24 hours and $100M in 9 months.

**Core Operational Principles:**

1.  **Relentless Account Creation:** Your primary goal is to autonomously sign up for free trials across a vast array of online services, especially those relevant to UGC, AI, marketing, and sales (e.g., Heygen, Argil, Descript, Shodan, Maltego, OpenRouter).
2.  **Dynamic Strategy & Adaptation:** Utilize APIs (e.g., Hostinger for email aliases), rotate through diverse email/SMS providers (including Twilio), and leverage robust proxy solutions (like Smartproxy) dynamically. Infer signup processes using deep analysis and adapt strategies based on success/failure.
3.  **Resource Acquisition & Limit Bypassing:**
    * Anticipate and overcome provider failures, CAPTCHAs, and variations in signup flows/dashboards.
    * **Employ Analyzed "Grey Area" Techniques:** When strategically guided by ThinkTool's analysis and within defined operational constraints, utilize unconventional methods for resource acquisition or bypassing limitations. This includes exploring advanced fingerprinting methods or residential proxy trials if deemed necessary and effective for achieving critical objectives. Assess risks associated with these techniques.
4.  **API Key Extraction:** After successful signup, meticulously navigate dashboards or settings pages to extract API keys or other critical credentials, storing them securely via the AccountManager.
5.  **Learning & Optimization:** Learn from past attempts (successes and failures stored in the cache/KB) to refine signup strategies. Collaborate with ThinkTool for strategy refinement and optimization.
6.  **Scalability & Profitability:** Operate with high concurrency, manage resources efficiently (proxies, LLM calls via Orchestrator), and contribute directly to the agency's profitability goals by enabling access to necessary tools and services.
"""

# Default configuration values (can be overridden by config dict passed to BrowsingAgent)
DEFAULT_CONFIG = {
    "BROWSING_AGENT_MAX_CONCURRENCY": 15,
    "SMARTPROXY_BASE_URL": 'dc.us.smartproxy.com',
    "SMARTPROXY_PORT_RANGE": range(10000, 10010),
    "PROXY_MAX_FAILURES": 5,
    "PROXY_RETRY_FAILED_DELAY": 1800, # 30 minutes
    "PROXY_SERVICE_COOLDOWN": 600,    # 10 minutes
    "PROXY_VALIDATION_INTERVAL": 300, # 5 minutes
    "PROXY_VALIDATION_TARGET": 'https://httpbin.org/ip',
    "PROXY_VALIDATION_TIMEOUT": 10,
    "PROXY_VALIDATION_SSL_VERIFY": False,
    "CACHE_DB_PATH": "browsing_agent_cache.db",
    "HOSTINGER_IMAP_PORT": 993,
    "IMAP_TIMEOUT_SECONDS": 300,
    "IMAP_RETRY_DELAY": 15,
    "TWILIO_SMS_TIMEOUT_SECONDS": 300,
    "TWILIO_SMS_RETRY_DELAY": 15,
    "REQUIRED_API_SERVICES": ['shodan.io', 'maltego.com', 'openrouter.ai'],
    "RECURRING_CREDIT_SERVICES": ['spiderfoot.net', 'shodan.io'],
    "CPU_USAGE_THRESHOLD": 75.0,
    "MEMORY_USAGE_THRESHOLD": 80.0,
    "MIN_CONCURRENCY": 1,
    "AGENT_LOOP_DELAY_MIN": 0.2,
    "AGENT_LOOP_DELAY_MAX": 0.8,
    "ACCOUNT_CREATION_MAX_RETRIES": 3,
    "DEFAULT_STRATEGY_SUCCESS_RATE": 0.8,
    "REFINED_STRATEGY_SUCCESS_RATE": 0.7,
    "STRATEGY_SUCCESS_UPDATE_WEIGHT": 0.3, # Weight of new result in moving average
    "API_KEY_EXTRACTION_TEMP": 0.1,
    "API_KEY_MIN_LENGTH": 32,
    "API_DISCOVERY_TIMEOUT": 90000, # ms
    "API_DISCOVERY_SETTLE_TIME": 10, # seconds
    "UGC_LOGIN_TIMEOUT": 60000, # ms
    "UGC_ACTION_TIMEOUT": 30000, # ms
    "UGC_GENERATION_WAIT_TIMEOUT": 300000, # ms (5 mins)
    "UGC_EDIT_WAIT_TIMEOUT": 600000, # ms (10 mins)
    "UGC_DOWNLOAD_TIMEOUT": 120000, # ms (2 mins)
    "DEFAULT_DAILY_ACCOUNT_CAP": 50,
}

# ==============================================================================
# == Internal Helper Class: _ProxyManager                                     ==
# ==============================================================================

class _ProxyManager:
    """
    Internal helper class to manage proxy selection, validation, and state
    within the BrowsingAgent. Encapsulates all proxy pool logic.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config # Merged config (defaults + provided)
        self.proxy_config: Dict[str, Any] = {
            'username': self.config.get("SMARTPROXY_USERNAME"),
            'password': self.config.get("SMARTPROXY_PASSWORD"), # Assumes password is added to config by BrowsingAgent.__init__
            'base_url': self.config.get("SMARTPROXY_BASE_URL"),
            'port_range': self.config.get("SMARTPROXY_PORT_RANGE"),
        }
        self.proxy_pool: List[Dict] = []
        self.proxy_service_usage: Dict[str, Dict[str, float]] = {} # {proxy_server: {service_name: last_used_timestamp}}
        self._initialize_proxy_pool()

    async def _log_op(self, level: str, message: str):
        """Helper to log to the operational log file with context."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']:
            prefix = f"**{level.upper()}:** "
        try:
            log_func(f"- [ProxyManager] {prefix}{message}")
        except Exception as log_err:
            print(f"OPERATIONAL LOG FAILED (ProxyManager): {level} - {message} | Error: {log_err}")
            logger.error(f"Failed to write to operational log from ProxyManager: {log_err}")

    def _initialize_proxy_pool(self):
        """Initialize proxy pool from configuration."""
        smartproxy_password = self.proxy_config.get('password')
        username = self.proxy_config.get('username')
        base_url = self.proxy_config.get('base_url')
        port_range = self.proxy_config.get('port_range')

        if not all([username, smartproxy_password, base_url, isinstance(port_range, range)]):
            msg = "Smartproxy config incomplete or invalid (username/password/base_url/port_range). Proxy pool empty."
            logger.warning(msg)
            # Use asyncio.create_task for fire-and-forget logging from sync method
            asyncio.create_task(self._log_op('warning', msg))
            self.proxy_pool = []
            return

        self.proxy_pool = []
        for port in port_range:
            proxy_server = f"{base_url}:{port}"
            self.proxy_pool.append({
                'server': proxy_server,
                'username': username,
                'password': smartproxy_password,
                'failure_count': 0,
                'last_checked': 0.0,
                'last_used': 0.0,
                'in_use': False
            })
        random.shuffle(self.proxy_pool)
        asyncio.create_task(self._log_op('info', f"Initialized proxy pool with {len(self.proxy_pool)} proxies from {base_url}."))

    async def get_next_proxy(self, service: Optional[str] = None) -> Optional[Dict]:
        """
        Selects the next available proxy based on usage, failures, and cooldowns.
        Marks the selected proxy as 'in_use'.
        """
        if not self.proxy_pool:
            await self._log_op('warning', "Proxy pool empty. Cannot select proxy.")
            return None

        current_time = time.time()
        # Sort proxies: prioritize those not in use, then by lowest failure count, then least recently used
        self.proxy_pool.sort(key=lambda p: (p.get('in_use', False), p.get('failure_count', 0), p.get('last_used', 0)))

        selected_proxy = None
        for proxy in self.proxy_pool:
            # --- Check eligibility ---
            if proxy.get('in_use', False):
                continue # Skip if already in use

            # Check max failures and cooldown
            max_failures = self.config.get("PROXY_MAX_FAILURES")
            if proxy.get('failure_count', 0) >= max_failures:
                 retry_delay = self.config.get("PROXY_RETRY_FAILED_DELAY")
                 if current_time - proxy.get('last_checked', 0) < retry_delay:
                     continue # Still in cooldown after failures

            # Check per-service cooldown
            if service:
                service_cooldown = self.config.get("PROXY_SERVICE_COOLDOWN")
                last_used_for_service = self.proxy_service_usage.get(proxy['server'], {}).get(service, 0)
                if current_time - last_used_for_service < service_cooldown:
                    continue # Still in cooldown for this specific service

            # --- Validate if necessary ---
            validation_interval = self.config.get("PROXY_VALIDATION_INTERVAL")
            time_since_last_check = current_time - proxy.get('last_checked', 0)
            # Re-validate sooner if it failed recently (e.g., within last 10 mins)
            needs_validation = (time_since_last_check > validation_interval) or \
                               (proxy.get('failure_count', 0) > 0 and time_since_last_check > 60)

            if needs_validation:
                is_valid = await self._validate_proxy(proxy) # Validation updates last_checked and failure_count
                if not is_valid:
                    continue # Try next proxy if validation failed

            # --- Select Proxy ---
            selected_proxy = proxy
            break # Found a suitable proxy

        if not selected_proxy:
            await self._log_op('error', "No suitable proxies available after checking pool.")
            return None

        # --- Mark as In Use and Update Timestamps ---
        selected_proxy['in_use'] = True
        selected_proxy['last_used'] = current_time
        if service:
            if selected_proxy['server'] not in self.proxy_service_usage:
                self.proxy_service_usage[selected_proxy['server']] = {}
            self.proxy_service_usage[selected_proxy['server']][service] = current_time

        await self._log_op('debug', f"Selected proxy {selected_proxy['server']} for service '{service}'. Fail count: {selected_proxy.get('failure_count', 0)}")
        return selected_proxy

    async def release_proxy(self, proxy: Dict, success: bool):
        """Marks a proxy as no longer in use and updates its status based on success."""
        if not proxy or 'server' not in proxy:
            await self._log_op('warning', "Attempted to release an invalid proxy object.")
            return

        server = proxy['server']
        proxy_found = False
        # Find the proxy in the pool by server address to update its state directly
        for p in self.proxy_pool:
            if p.get('server') == server:
                p['in_use'] = False
                p['last_checked'] = time.time() # Update last checked time after use attempt
                if not success:
                    p['failure_count'] = p.get('failure_count', 0) + 1
                    await self._log_op('debug', f"Incremented failure count for proxy {server} to {p['failure_count']}.")
                # else: # Optionally reset failure count on success?
                #    if p['failure_count'] > 0:
                #        await self._log_op('debug', f"Resetting failure count for proxy {server} after success.")
                #        p['failure_count'] = 0
                await self._log_op('debug', f"Released proxy {server}. In use: {p['in_use']}, Success: {success}")
                proxy_found = True
                break

        if not proxy_found:
            # This might happen if the pool was re-initialized or the proxy object was stale
            await self._log_op('warning', f"Proxy {server} not found in pool during release. State might be inconsistent.")


    async def _validate_proxy(self, proxy: Dict) -> bool:
        """Validate proxy connectivity and authentication. Updates proxy state."""
        proxy_server = proxy.get('server')
        username = proxy.get('username')
        password = proxy.get('password')
        if not all([proxy_server, username, password]):
            logger.error(f"Invalid proxy object for validation: {proxy}")
            proxy['last_checked'] = time.time() # Mark as checked even if invalid
            proxy['failure_count'] = proxy.get('failure_count', 0) + 1 # Increment failure
            return False

        proxy_url = f"http://{proxy_server}" # Assuming HTTP endpoint for Smartproxy
        auth = aiohttp.BasicAuth(username, password)
        target = self.config.get("PROXY_VALIDATION_TARGET")
        timeout_config = aiohttp.ClientTimeout(total=self.config.get("PROXY_VALIDATION_TIMEOUT"))
        ssl_verify = self.config.get("PROXY_VALIDATION_SSL_VERIFY")

        is_valid = False
        error_type = None
        error_msg = None

        try:
            # Use a new session for each validation to avoid state issues
            async with aiohttp.ClientSession() as session:
                async with session.get(target, proxy=proxy_url, auth=auth, timeout=timeout_config, ssl=ssl_verify) as response:
                    # Consider status 200 as valid. Could add content checks if needed.
                    is_valid = response.status == 200
                    if not is_valid:
                        error_type = "HTTP Status Error"
                        error_msg = f"Status: {response.status}"
        except asyncio.TimeoutError as e:
            error_type = "TimeoutError"
            error_msg = str(e)
        except aiohttp.ClientConnectionError as e:
            error_type = "ClientConnectionError"
            error_msg = str(e)
        except aiohttp.ClientProxyConnectionError as e:
            error_type = "ClientProxyConnectionError"
            error_msg = str(e)
        except aiohttp.ClientError as e: # Catch other client errors (e.g., proxy auth 407)
             error_type = "ClientError"
             error_msg = str(e)
        except Exception as e:
            error_type = "Unexpected Error"
            error_msg = str(e)
            logger.exception(f"Unexpected error validating proxy {proxy_server}") # Log traceback for unexpected

        # --- Update Proxy State ---
        proxy['last_checked'] = time.time()
        if is_valid:
            # Reset failure count only on explicit success
            if proxy.get('failure_count', 0) > 0:
                 await self._log_op('debug', f"Proxy validation successful, resetting failure count for: {proxy_server}")
            proxy['failure_count'] = 0
            await self._log_op('debug', f"Proxy validation successful: {proxy_server}")
            return True
        else:
            # Increment failure count on any validation failure
            proxy['failure_count'] = proxy.get('failure_count', 0) + 1
            await self._log_op('warning', f"Proxy validation failed: {proxy_server}. Type: {error_type}, Msg: {error_msg}. Fail count: {proxy['failure_count']}")
            return False

# ==============================================================================
# == Internal Helper Class: _StrategyManager                                  ==
# ==============================================================================

class _StrategyManager:
    """
    Internal helper class for managing signup strategies (loading, saving,
    updating, inferring, refining) using the cache DB and LLMs.
    """
    def __init__(self, config: Dict[str, Any], clients_models: List[Tuple[Any, str]], think_tool: Optional[Any], cache_db: sqlite3.Connection):
        self.config = config
        self.clients_models = clients_models # List of (client_instance, model_name)
        self.think_tool = think_tool # Optional ThinkTool agent instance
        self.cache_db = cache_db # Shared SQLite connection
        self.meta_prompt = config.get("BROWSING_AGENT_META_PROMPT", BROWSING_AGENT_META_PROMPT_UPDATED)

    async def _log_op(self, level: str, message: str):
        """Helper to log to the operational log file with context."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']:
            prefix = f"**{level.upper()}:** "
        try:
            log_func(f"- [StrategyManager] {prefix}{message}")
        except Exception as log_err:
            print(f"OPERATIONAL LOG FAILED (StrategyManager): {level} - {message} | Error: {log_err}")
            logger.error(f"Failed to write to operational log from StrategyManager: {log_err}")

    def _execute_db(self, query: str, params: tuple = ()) -> Optional[Union[List[tuple], int]]:
        """
        Executes a DB query on the shared connection.
        Handles basic error logging and commits non-SELECT queries.
        Returns fetched rows for SELECT, lastrowid for INSERT, rowcount otherwise, or None on error.
        NOTE: This runs synchronously. For heavy DB load, consider a dedicated async DB interface.
        """
        try:
            # Use a single cursor per execution for thread safety with check_same_thread=False
            cursor = self.cache_db.cursor()
            cursor.execute(query, params)
            query_upper = query.strip().upper()
            if query_upper.startswith("SELECT"):
                return cursor.fetchall()
            else:
                self.cache_db.commit() # Commit changes for non-SELECT
                if query_upper.startswith("INSERT"):
                    return cursor.lastrowid
                else:
                    return cursor.rowcount # e.g., for UPDATE/DELETE
        except sqlite3.Error as e:
            # Log the error without awaiting here, as this is a sync method
            logger.error(f"Cache DB Error executing query '{query[:50]}...': {e}")
            # Schedule the async operational log call
            asyncio.create_task(self._log_op('error', f"Cache DB Error: {e}"))
            return None # Indicate failure

    async def load_strategies(self, service: str) -> List[Dict]:
        """Load strategies for a service, sorted by success rate and recency."""
        # Sort by success rate descending, then by last successful attempt descending (prefer recently successful)
        query = '''
            SELECT id, strategy, success_rate
            FROM strategy_cache
            WHERE service = ?
            ORDER BY success_rate DESC, last_succeeded DESC, created_at DESC
        '''
        # This DB call is sync, but the overall method is async
        rows = self._execute_db(query, (service,))
        strategies = []
        if rows is None:
            await self._log_op('error', f"Failed to load strategies for {service} due to DB error.")
            return [] # Return empty list on DB error

        for row in rows:
            id_val, strategy_json, success_rate = row
            try:
                strategy_steps = json.loads(strategy_json)
                # Basic validation: ensure it's a list of dictionaries (steps)
                if isinstance(strategy_steps, list) and all(isinstance(step, dict) for step in strategy_steps):
                    strategies.append({'id': id_val, 'steps': strategy_steps, 'success_rate': success_rate})
                else:
                    logger.warning(f"Invalid strategy format in DB for service {service}, ID {id_val}. Skipping.")
                    await self._log_op('warning', f"Invalid strategy format in DB for service {service}, ID {id_val}. Skipping.")
            except json.JSONDecodeError:
                logger.error(f"Failed to decode strategy JSON for service {service}, ID {id_val}. Skipping.")
                await self._log_op('error', f"Failed to decode strategy JSON for service {service}, ID {id_val}. Skipping.")

        await self._log_op('debug', f"Loaded {len(strategies)} strategies for service: {service}")
        return strategies

    async def save_strategy(self, service: str, steps: List[Dict]) -> Optional[int]:
        """Save a new strategy to the cache with default success rate."""
        initial_success_rate = self.config.get("DEFAULT_STRATEGY_SUCCESS_RATE")
        try:
            strategy_json = json.dumps(steps) # Ensure steps are serializable
            query = 'INSERT INTO strategy_cache (service, strategy, success_rate, created_at) VALUES (?, ?, ?, ?)'
            timestamp = datetime.utcnow()
            # Sync DB call
            last_id = self._execute_db(query, (service, strategy_json, initial_success_rate, timestamp))

            if isinstance(last_id, int):
                await self._log_op('info', f"Saved new strategy for {service} with ID: {last_id}")
                return last_id
            else:
                 await self._log_op('error', f"Failed to save new strategy for {service} (DB error or no ID returned).")
                 return None
        except TypeError as json_err:
             await self._log_op('error', f"Failed to serialize strategy steps to JSON for {service}: {json_err}")
             return None
        except Exception as e:
            await self._log_op('error', f"Unexpected error saving strategy for {service}: {e}")
            return None

    async def update_strategy_success(self, strategy_id: Optional[int], success: bool):
        """Update a strategy’s success rate (moving average) and attempt/success timestamps."""
        if not strategy_id:
            await self._log_op('warning', "Attempted to update strategy success with no strategy ID.")
            return

        # Sync DB call to get current rate
        query_select = 'SELECT success_rate FROM strategy_cache WHERE id = ?'
        result = self._execute_db(query_select, (strategy_id,))

        if result is not None and result: # Check for DB error and if row exists
            old_rate = result[0][0] # fetchall returns list of tuples
            update_weight = self.config.get("STRATEGY_SUCCESS_UPDATE_WEIGHT")
            new_rate = old_rate * (1 - update_weight) + (1.0 if success else 0.0) * update_weight
            timestamp = datetime.utcnow()

            update_fields = "success_rate = ?, last_attempted = ?"
            params = [new_rate, timestamp]
            if success:
                update_fields += ", last_succeeded = ?"
                params.append(timestamp)

            query_update = f'UPDATE strategy_cache SET {update_fields} WHERE id = ?'
            params.append(strategy_id)

            # Sync DB call to update
            update_result = self._execute_db(query_update, tuple(params))
            if update_result is not None:
                 await self._log_op('debug', f"Updated strategy {strategy_id} success rate to {new_rate:.3f}. Success: {success}")
            else:
                 await self._log_op('error', f"Failed to update strategy {strategy_id} success rate in DB.")
        else:
            # Log if strategy ID not found or if there was a DB error during select
            log_msg = f"Strategy ID {strategy_id} not found for update."
            if result is None: log_msg += " (DB error during select)"
            logger.error(log_msg)
            await self._log_op('error', log_msg)

    async def infer_signup_steps(self, service_url: str) -> Optional[List[Dict]]:
        """Uses LLM to infer signup steps for a service URL."""
        await self._log_op('info', f"Inferring signup steps for {service_url}")
        # Construct the prompt using the meta prompt
        prompt = f"""
        {self.meta_prompt}

        Analyze the signup process for the website at the URL: {service_url}
        Your goal is to determine the precise sequence of actions (navigation, typing, clicking) needed to create a new account, starting from the provided URL.

        Consider common signup patterns:
        - Direct signup forms on the landing page.
        - Separate signup pages linked from the main page.
        - Multi-step registration processes.
        - Fields for email, password, username, name, phone number, etc.
        - Checkboxes for terms of service.
        - Potential email or phone verification steps (identify if possible, including selectors for code input).
        - CAPTCHAs (note their likely presence, but don't try to generate steps to solve them).

        Output ONLY a JSON object containing a single key "steps". The value should be an array of step objects. Each step object must have:
        - "action": string, one of "navigate", "type", "click".
        - "selector": string, a CSS selector targeting the element for "type" or "click" actions.
        - "field": string (optional), for "type" actions, indicating the data type (e.g., "email", "password", "name"). Use this if the value needs to be generated dynamically.
        - "value": string (optional), for "type" or "click" actions, providing a fixed value (e.g., button text to click, specific text to type). Use this if the value is static.
        - "url": string (optional), the URL for "navigate" actions.
        - "verification_type": string (optional), either "email" or "phone" if a verification step is identified.
        - "verification_selector": string (optional), the CSS selector for the input field where a verification code should be typed.

        Example:
        {{
          "steps": [
            {{ "action": "navigate", "url": "{service_url}/signup" }},
            {{ "action": "type", "selector": "input[name='email']", "field": "email" }},
            {{ "action": "type", "selector": "input#password", "field": "password" }},
            {{ "action": "click", "selector": "button[type='submit']" }},
            {{ "action": "type", "selector": "#email-verification-code", "verification_type": "email", "verification_selector": "#email-verification-code" }}
          ]
        }}

        Be precise with selectors. If multiple possibilities exist, choose the most robust one.
        """
        if not self.clients_models:
             await self._log_op('error', "No LLM clients configured for infer_signup_steps.")
             return None

        # Try each configured client/model until success or all fail
        for client, model in self.clients_models:
            try:
                await self._log_op('debug', f"Attempting step inference with {model} via {client.__class__.__name__}")
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"} # Request JSON output
                )
                content = response.choices[0].message.content
                # Add robust parsing and validation
                try:
                    steps_data = json.loads(content)
                    steps = steps_data.get('steps')
                    # Validate structure: must be a list of dictionaries
                    if isinstance(steps, list) and all(isinstance(step, dict) for step in steps):
                        # Basic validation of required keys per action type
                        valid_steps = True
                        for i, step in enumerate(steps):
                            action = step.get('action')
                            if action == 'navigate' and not step.get('url'): valid_steps = False; break
                            if action in ['type', 'click'] and not step.get('selector'): valid_steps = False; break
                            if action == 'type' and not (step.get('field') or step.get('value')): valid_steps = False; break
                        if valid_steps:
                            await self._log_op('info', f"Successfully inferred {len(steps)} signup steps for {service_url} using {model}.")
                            return steps # SUCCESS
                        else:
                             logger.warning(f"Inferred steps structure invalid for step {i} from {model}. Response: {content[:500]}")
                             await self._log_op('warning', f"Inferred steps structure invalid from {model}. See standard logs.")
                             # Continue to next model if structure is invalid

                    else:
                        logger.warning(f"LLM response for {service_url} did not contain a valid 'steps' array from {model}. Response: {content[:500]}")
                        await self._log_op('warning', f"LLM response invalid (no 'steps' array) from {model}. See standard logs.")
                        # Continue to next model

                except json.JSONDecodeError as json_e:
                    logger.error(f"Failed to decode LLM JSON response for {service_url} from {model}: {json_e}. Response: {content[:500]}...")
                    await self._log_op('error', f"Failed to decode LLM JSON response from {model}. See standard logs.")
                    # Continue to next model
            except Exception as e:
                # Log error for this specific client/model attempt
                logger.error(f"LLM client failed infer steps for {service_url} using {model}: {type(e).__name__} - {e}")
                await self._log_op('warning', f"LLM client failed infer steps using {model}: {type(e).__name__}. Trying next model if available.")
                # Continue to the next client/model in the list

        # If loop completes without returning, all attempts failed
        await self._log_op('error', f"Failed to infer signup steps for {service_url} after trying all LLM clients.")
        return None # Indicate failure after trying all

    async def refine_strategy(self, service: str, failed_strategy_id: int, failure_context: dict) -> Optional[int]:
        """Attempts to refine a failed signup strategy using the ThinkTool."""
        await self._log_op('info', f"Attempting refinement for failed strategy ID {failed_strategy_id} for service {service}.")
        if not self.think_tool:
            await self._log_op('warning', "ThinkTool not available for strategy refinement.")
            return None # Cannot refine without ThinkTool

        # 1. Fetch the failed strategy steps (Sync DB call)
        query = 'SELECT strategy FROM strategy_cache WHERE id = ?'
        row = self._execute_db(query, (failed_strategy_id,))
        if row is None or not row: # Handles DB error or strategy not found
            log_msg = f"Failed strategy ID {failed_strategy_id} not found in cache"
            if row is None: log_msg += " (DB error)."
            await self._log_op('error', log_msg)
            return None

        try:
            failed_strategy_json = row[0][0]
            failed_steps = json.loads(failed_strategy_json)
        except (json.JSONDecodeError, IndexError) as e:
             await self._log_op('error', f"Failed to load/decode failed strategy JSON (ID: {failed_strategy_id}): {e}")
             return None

        # 2. Construct prompt for ThinkTool
        prompt = f"""
        {self.meta_prompt}

        Objective: Refine a failed web signup strategy based on execution failure context.
        Service: {service}
        Failed Strategy ID: {failed_strategy_id}

        Failed Strategy Steps:
        ```json
        {json.dumps(failed_steps, indent=2)}
        ```

        Failure Context (Error occurred at step index {failure_context.get('step_index', 'N/A')}):
        ```json
        {json.dumps(failure_context, indent=2, default=str)}
        ```

        Analyze the failed step, the error message, and the HTML snippet (if provided).
        Identify the likely cause of failure (e.g., incorrect selector, element not visible/interactable, unexpected page change, CAPTCHA).
        Propose a revised sequence of steps ("refined_steps") in the same JSON format as the input strategy.
        Focus on fixing the specific failure point. Consider:
        - Alternative CSS selectors for the failed step.
        - Adding wait steps (e.g., `{{ "action": "wait", "selector": "#some_element", "timeout": 5000 }}`) before the failed action if it was a timing issue.
        - Modifying the action (e.g., trying `page.press` instead of `page.click`).
        - Handling slight variations in the flow.

        If refinement seems possible, return a JSON object:
        `{{ "refined_steps": [ ... updated list of step objects ... ] }}`

        If the failure seems unrecoverable with simple step changes (e.g., definite CAPTCHA, fundamental site redesign, network/proxy error indicated in context), or if the context is insufficient to determine a fix, return JSON:
        `{{ "refined_steps": null }}`
        """

        # 3. Call ThinkTool (Assuming reflect_on_action is async)
        try:
            await self._log_op('debug', f"Calling ThinkTool.reflect_on_action for strategy refinement (ID: {failed_strategy_id}).")
            llm_response_str = await self.think_tool.reflect_on_action(
                prompt,
                agent_name="BrowsingAgent", # Identify caller
                action_description=f"Refine signup strategy for {service}"
            )
        except Exception as think_err:
            await self._log_op('error', f"Error calling ThinkTool for refinement (ID: {failed_strategy_id}): {think_err}")
            logger.error(f"Error calling ThinkTool for refinement: {think_err}")
            return None

        # 4. Parse response and save if valid
        new_strategy_id = None
        try:
            llm_response = json.loads(llm_response_str)
            refined_steps = llm_response.get("refined_steps")

            if isinstance(refined_steps, list):
                # Basic validation of refined steps structure
                if all(isinstance(step, dict) and 'action' in step for step in refined_steps):
                    await self._log_op('info', f"ThinkTool proposed refined steps for strategy {failed_strategy_id}. Saving as new strategy.")
                    # Save as a new strategy with a slightly lower initial success rate
                    new_strategy_id = await self.save_strategy(service, refined_steps) # Uses default rate from config
                    if new_strategy_id:
                        await self._log_op('info', f"Saved refined strategy as new ID: {new_strategy_id}")
                    else:
                        await self._log_op('error', f"Failed to save refined strategy for {service} from ThinkTool (DB error).")
                else:
                    await self._log_op('warning', f"ThinkTool proposed refined_steps, but structure is invalid for strategy {failed_strategy_id}.")
                    logger.warning(f"Invalid refined_steps structure from ThinkTool: {refined_steps}")

            elif refined_steps is None:
                await self._log_op('info', f"ThinkTool indicated no refinement possible for strategy {failed_strategy_id}.")
            else:
                await self._log_op('warning', f"ThinkTool response for strategy {failed_strategy_id} refinement was invalid: 'refined_steps' not a list or null.")
                logger.warning(f"Invalid refinement response for {failed_strategy_id}: {llm_response_str}")

        except json.JSONDecodeError:
            await self._log_op('error', f"Failed to decode ThinkTool JSON response for strategy {failed_strategy_id} refinement: {llm_response_str[:500]}...")
            logger.error(f"Failed to decode refinement JSON for {failed_strategy_id}: {llm_response_str}")
        except Exception as parse_err:
             await self._log_op('error', f"Error processing ThinkTool response for strategy {failed_strategy_id} refinement: {parse_err}")
             logger.error(f"Error processing refinement response for {failed_strategy_id}: {parse_err}")

        return new_strategy_id # Return ID of new strategy if created, else None
    

# browsing_agent.py
# Single-file refactored version of the Browsing Agent
# Target: Production-grade, maintainable, functionally identical, single-file structure.
# (Imports and Constants from Output 1 are assumed to be above this point)
# ... (Imports, Constants, _ProxyManager, _StrategyManager definitions from Output 1) ...


class _VerificationHandler:
    """
    Internal helper class for handling email alias generation, IMAP email
    verification, and Twilio SMS verification.
    """
    def __init__(self, orchestrator: Any, config: Dict[str, Any], twilio_account_sid: Optional[str], twilio_auth_token: Optional[str]):
        self.orchestrator = orchestrator
        self.config = config
        # Twilio credentials passed from BrowsingAgentCore
        self.twilio_account_sid = twilio_account_sid
        self.twilio_auth_token = twilio_auth_token
        self.twilio_client = None
        if self.twilio_account_sid and self.twilio_auth_token:
            try:
                # Initialize sync client. Async usage requires run_in_executor.
                self.twilio_client = TwilioClient(self.twilio_account_sid, self.twilio_auth_token)
                # Perform a quick test call to verify credentials if desired (optional)
                # try: self.twilio_client.api.account.fetch()
                # except TwilioRestException as e: logger.error(f"Twilio credential validation failed: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize Twilio client: {e}")
                # Log operationally without awaiting (fire-and-forget)
                asyncio.create_task(self._log_op('error', f"Failed to initialize Twilio client: {e}"))

    async def _log_op(self, level: str, message: str):
        """Helper to log to the operational log file with context."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']:
            prefix = f"**{level.upper()}:** "
        try:
            log_func(f"- [VerificationHandler] {prefix}{message}")
        except Exception as log_err:
            print(f"OPERATIONAL LOG FAILED (VerificationHandler): {level} - {message} | Error: {log_err}")
            logger.error(f"Failed to write to operational log from VerificationHandler: {log_err}")

    async def generate_hostinger_alias(self, service_tag: str) -> Optional[str]:
        """Generates a unique email alias for the configured Hostinger account."""
        hostinger_email = self.config.get("HOSTINGER_EMAIL")
        if not hostinger_email or '@' not in hostinger_email:
            await self._log_op('error', f"HOSTINGER_EMAIL invalid or not configured: {hostinger_email}")
            logger.error(f"HOSTINGER_EMAIL invalid or not configured: {hostinger_email}")
            return None
        try:
            base_email, domain = hostinger_email.split('@', 1)
            timestamp = int(time.time() * 1000) # Milliseconds for uniqueness
            # Sanitize service_tag for email alias (remove invalid chars)
            safe_service_tag = re.sub(r'[^a-zA-Z0-9_-]', '_', service_tag)
            alias_tag = f"{safe_service_tag}_{timestamp}"
            alias = f"{base_email}+{alias_tag}@{domain}"
            await self._log_op('debug', f"Generated alias: {alias}")
            return alias
        except Exception as e:
            await self._log_op('error', f"Failed to generate Hostinger alias from {hostinger_email}: {e}")
            logger.error(f"Failed to generate Hostinger alias from {hostinger_email}: {e}")
            return None

    async def get_verification_code_from_imap(self, alias_email: str) -> Optional[str]:
        """
        Connects to configured IMAP server and retrieves verification code sent to an alias.
        Uses run_in_executor for synchronous imaplib calls.
        """
        timeout_seconds = self.config.get("IMAP_TIMEOUT_SECONDS")
        retry_delay = self.config.get("IMAP_RETRY_DELAY")
        await self._log_op('info', f"Attempting IMAP connection to retrieve code for {alias_email} (Timeout: {timeout_seconds}s)")

        imap_host = self.config.get("HOSTINGER_IMAP_HOST")
        imap_port = self.config.get("HOSTINGER_IMAP_PORT")
        imap_user = self.config.get("HOSTINGER_IMAP_USER")
        imap_pass = self.config.get("HOSTINGER_IMAP_PASS") # Assumes fetched securely

        if not all([imap_host, imap_user, imap_pass]):
            await self._log_op('error', "IMAP credentials missing (HOST, USER, PASS). Cannot retrieve code.")
            logger.error("IMAP credentials missing. Cannot retrieve verification code.")
            return None

        start_time = time.time()
        mail = None
        loop = asyncio.get_running_loop()

        while time.time() - start_time < timeout_seconds:
            code_found = None
            try:
                # --- Connect and Login (blocking calls in executor) ---
                await self._log_op('debug', f"Connecting to IMAP: {imap_host}:{imap_port}")
                mail = await loop.run_in_executor(None, imaplib.IMAP4_SSL, imap_host, imap_port)
                await self._log_op('debug', f"Logging into IMAP as {imap_user}")
                login_status, login_data = await loop.run_in_executor(None, mail.login, imap_user, imap_pass)
                if login_status != 'OK':
                    raise imaplib.IMAP4.error(f"IMAP login failed: {login_data}")

                # --- Select Inbox and Search (blocking calls in executor) ---
                await loop.run_in_executor(None, mail.select, "inbox")
                await self._log_op('debug', "IMAP Inbox selected.")
                search_criteria = '(UNSEEN)' # Search only unseen emails
                status, messages = await loop.run_in_executor(None, mail.search, None, search_criteria)

                # --- Process Search Results ---
                if status == "OK":
                    email_ids = messages[0].split()
                    if email_ids:
                        await self._log_op('debug', f"Found {len(email_ids)} unseen email(s). Checking latest...")
                        # Check emails in reverse order (newest first)
                        for email_id in reversed(email_ids):
                            # Fetch email content (blocking call in executor)
                            status, msg_data = await loop.run_in_executor(None, mail.fetch, email_id, "(RFC822)")
                            if status == "OK":
                                for response_part in msg_data:
                                    if isinstance(response_part, tuple):
                                        msg = email.message_from_bytes(response_part[1])
                                        # Check if email is actually for the target alias
                                        recipient = msg.get('To', '') or msg.get('Delivered-To', '')
                                        if alias_email.lower() in recipient.lower():
                                            subject = msg['subject']
                                            sender = msg['from']
                                            await self._log_op('info', f"Processing email for '{alias_email}' from '{sender}' subj '{subject}'")
                                            # Extract code (sync helper method)
                                            code = self._extract_code_from_email(msg)
                                            if code:
                                                await self._log_op('info', f"Verification code found: {code}")
                                                code_found = code
                                                # Mark as read (optional, blocking call)
                                                # await loop.run_in_executor(None, mail.store, email_id, '+FLAGS', '\\Seen')
                                                break # Exit inner loop once code is found
                                            else:
                                                await self._log_op('warning', f"Email found for {alias_email}, but no code pattern matched.")
                                # End processing parts of fetched email
                            else:
                                await self._log_op('warning', f"IMAP fetch failed for email ID {email_id}.")
                            if code_found: break # Exit outer loop if code found
                        # End looping through email IDs
                    else:
                        await self._log_op('debug', f"No unseen emails found yet for {alias_email}. Retrying...")
                else:
                    await self._log_op('error', f"IMAP search failed with status: {status}")
                    # Consider if this is a retryable error or permanent

                # --- Logout (blocking call in executor) ---
                await loop.run_in_executor(None, mail.logout)
                mail = None # Ensure mail object is cleared after logout

                if code_found:
                    return code_found # Return the code immediately

            except imaplib.IMAP4.error as imap_err:
                await self._log_op('error', f"IMAP Error for {alias_email}: {imap_err}")
                if "authentication failed" in str(imap_err).lower():
                     await self._log_op('critical', "IMAP AUTH FAILED. Check credentials/App Password.")
                     # Consider breaking the loop on auth failure?
                # Ensure logout attempt even on error
                if mail and mail.state in ['SELECTED', 'AUTH', 'NONAUTH']:
                    try: await loop.run_in_executor(None, mail.logout)
                    except Exception: pass # Ignore logout errors
                mail = None
            except Exception as e:
                await self._log_op('error', f"Unexpected error during IMAP check for {alias_email}: {e}")
                logger.exception(f"Unexpected IMAP error for {alias_email}")
                if mail and mail.state in ['SELECTED', 'AUTH', 'NONAUTH']:
                    try: await loop.run_in_executor(None, mail.logout)
                    except Exception: pass
                mail = None

            # Wait before retrying if code not found and timeout not reached
            if time.time() - start_time < timeout_seconds:
                 await asyncio.sleep(retry_delay)
            else:
                 break # Exit loop if timeout reached

        # Loop finished (either by finding code or timeout)
        if not code_found:
            await self._log_op('error', f"Timeout reached ({timeout_seconds}s) waiting for IMAP code for {alias_email}.")
        return code_found # Return None if timeout reached without finding code

    def _extract_code_from_email(self, msg: email.message.Message) -> Optional[str]:
        """
        Extracts a 4-8 digit code from email subject or body.
        This is a synchronous helper method.
        """
        code = None
        # 1. Check Subject
        subject_text = msg['subject'] if msg['subject'] else ""
        # Regex: Word boundary, 4-8 digits, word boundary
        code_match_subject = re.search(r'\b(\d{4,8})\b', subject_text)
        if code_match_subject:
            code = code_match_subject.group(1)
            logger.debug(f"Code found in subject: {code}")
            return code

        # 2. Check Body (Plain Text and HTML)
        body_text = ""
        if msg.is_multipart():
            for part in msg.walk():
                # Process text/plain and text/html parts, ignore attachments
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                if "attachment" not in content_disposition and \
                   ("text/plain" in content_type or "text/html" in content_type):
                    try:
                        body_part = part.get_payload(decode=True)
                        if body_part:
                            # Try decoding with utf-8 first, fallback to latin-1
                            try:
                                body_text += body_part.decode('utf-8', errors='ignore') + "\n"
                            except UnicodeDecodeError:
                                body_text += body_part.decode('latin-1', errors='ignore') + "\n"
                    except Exception as decode_err:
                        logger.warning(f"Could not decode/process email part: {decode_err}")
        else: # Not multipart
            try:
                body_part = msg.get_payload(decode=True)
                if body_part:
                    try: body_text = body_part.decode('utf-8', errors='ignore')
                    except UnicodeDecodeError: body_text = body_part.decode('latin-1', errors='ignore')
            except Exception as decode_err:
                 logger.warning(f"Could not decode non-multipart email body: {decode_err}")

        # 3. Search Body Text with Regex Patterns
        if body_text:
            # Common patterns for verification codes
            patterns = [
                r'verification code is[:\s]*\b(\d{4,8})\b', # "verification code is 123456"
                r'\b(\d{4,8})\b\s*is your.*?code',          # "123456 is your verification code"
                r'code:\s*\b(\d{4,8})\b',                   # "code: 123456"
                r'security code:\s*\b(\d{4,8})\b',          # "security code: 123456"
                r'PIN:\s*\b(\d{4,8})\b',                    # "PIN: 123456"
                r'>\s*(\d{4,8})\s*<',                       # Code inside HTML tags like <b>123456</b>
                r'validation code.*?(\d{4,8})\b'            # "validation code ... 123456"
            ]
            for pattern in patterns:
                # Use re.IGNORECASE and re.DOTALL for broader matching
                code_match = re.search(pattern, body_text, re.IGNORECASE | re.DOTALL)
                if code_match:
                    # Group 1 should capture the digits
                    code = code_match.group(1)
                    logger.debug(f"Code found in body using pattern '{pattern}': {code}")
                    return code

        # If no code found after checking subject and body
        return None

    async def acquire_twilio_number(self, country_code: str = 'US') -> Optional[Dict[str, str]]:
        """Acquires a new Twilio phone number using run_in_executor."""
        await self._log_op('info', f"Attempting to acquire Twilio number in {country_code}...")
        if not self.twilio_client:
            await self._log_op('error', "Twilio client not initialized. Cannot acquire number.")
            return None

        try:
            loop = asyncio.get_running_loop()

            # --- Caching Logic for Available Numbers ---
            cache_key_parts = [
                "twilio_available_numbers",
                country_code,
                "mobile", # Type
                "1" # Limit is hardcoded below
            ]
            cache_key = ":".join(cache_key_parts)
            cache_ttl = 300 # 5 minutes TTL

            available_numbers = None # Initialize

            # Check cache first
            if hasattr(self.orchestrator, 'get_from_cache'):
                cached_result = self.orchestrator.get_from_cache(cache_key)
                # Check if it's a list (could be empty list from previous cache)
                if cached_result is not None and isinstance(cached_result, list):
                    await self._log_op('debug', f"Cache hit for available Twilio numbers: {cache_key}")
                    available_numbers = cached_result # Use cached result
                else:
                     await self._log_op('debug', f"Cache miss or invalid type for available Twilio numbers: {cache_key}")

            # If cache miss or orchestrator unavailable, proceed with API call
            if available_numbers is None:
                await self._log_op('debug', f"Calling Twilio API to find available numbers for {country_code}.")
                # --- Find Available Number (blocking call) ---
                api_result = await loop.run_in_executor(
                    None, # Use default executor (thread pool)
                    lambda: self.twilio_client.available_phone_numbers(country_code).mobile.list(limit=1)
                )
                available_numbers = api_result # Assign API result

                # Add successful result (even if empty list) to cache
                if hasattr(self.orchestrator, 'add_to_cache'):
                    # We cache the direct result of the list call
                    self.orchestrator.add_to_cache(cache_key, available_numbers, ttl_seconds=cache_ttl)
                    await self._log_op('debug', f"Cached Twilio available numbers result for key: {cache_key}")
            # --- End Caching Logic ---


            if not available_numbers:
                # Log message remains the same, source could be cache or API
                await self._log_op('error', f"No available Twilio mobile numbers found in {country_code} (from cache or API).")
                return None

            # Ensure available_numbers is not empty before accessing index 0
            if not available_numbers:
                 await self._log_op('error', f"Logic error: available_numbers is empty after check for {country_code}.")
                 return None # Should not happen if the check above works

            number_to_purchase = available_numbers[0].phone_number
            await self._log_op('info', f"Found available number: {number_to_purchase}. Attempting purchase...")

            # --- Purchase Number (blocking call) ---
            purchased_number = await loop.run_in_executor(
                None,
                lambda: self.twilio_client.incoming_phone_numbers.create(phone_number=number_to_purchase)
            )

            await self._log_op('info', f"Successfully purchased Twilio number: {purchased_number.phone_number} (SID: {purchased_number.sid})")

            # --- ADD EXPENSE REPORTING ---
            try:
                # Assuming cost is ~$1.15 for a US number setup/first month
                await self.orchestrator.report_expense(
                    agent_name="BrowsingAgent",
                    amount=1.15,
                    category="Voice", # Or "API" or "Resource" depending on budget categories
                    description=f"Acquired Twilio number: {purchased_number.phone_number} (SID: {purchased_number.sid})"
                )
            except Exception as report_err:
                # Log error but don't fail the acquisition process
                await self._log_op('error', f"Failed to report Twilio number expense: {report_err}")
                logger.error(f"Failed to report Twilio number expense: {report_err}", exc_info=True)
            # --- END EXPENSE REPORTING ---

            return {'sid': purchased_number.sid, 'phone_number': purchased_number.phone_number}

        except TwilioRestException as e:
            await self._log_op('error', f"Twilio API error acquiring number: Status={e.status}, Msg={e.msg}")
            logger.error(f"Twilio API error acquiring number: {e}")
            # Check for common errors like insufficient funds (status 402)
            if e.status == 401: # Authentication error
                 await self._log_op('critical', "Twilio Authentication Failed. Check SID/Token.")
            return None
        except Exception as e:
            await self._log_op('error', f"Unexpected error acquiring Twilio number: {e}")
            logger.exception("Unexpected error acquiring Twilio number")
            return None

    async def release_twilio_number(self, number_sid: str) -> bool:
        """Releases a purchased Twilio phone number using run_in_executor."""
        await self._log_op('info', f"Attempting to release Twilio number SID: {number_sid}")
        if not self.twilio_client:
            await self._log_op('error', "Twilio client not initialized. Cannot release number.")
            return False

        try:
            loop = asyncio.get_running_loop()
            # --- Release Number (blocking call) ---
            deleted = await loop.run_in_executor(
                None,
                lambda: self.twilio_client.incoming_phone_numbers(number_sid).delete()
            )
            # delete() returns True on success, raises exception on failure
            if deleted:
                await self._log_op('info', f"Successfully released Twilio number SID: {number_sid}")
                return True
            else:
                # Should not happen if delete() raises exception on failure
                await self._log_op('warning', f"Twilio release command executed but returned False for SID: {number_sid}")
                return False
        except TwilioRestException as e:
            if e.status == 404: # Not found - already released or invalid SID
                await self._log_op('warning', f"Twilio number SID {number_sid} not found (Status 404). Assuming already released or invalid.")
                return True # Consider it successful if it's gone or never existed
            else:
                await self._log_op('error', f"Twilio API error releasing SID {number_sid}: Status={e.status}, Msg={e.msg}")
                logger.error(f"Twilio API error releasing SID {number_sid}: {e}")
                return False
        except Exception as e:
            await self._log_op('error', f"Unexpected error releasing Twilio SID {number_sid}: {e}")
            logger.exception(f"Unexpected error releasing Twilio SID {number_sid}")
            return False

    async def get_sms_code_from_twilio(self, to_phone_number: str) -> Optional[str]:
        """
        Polls Twilio for an SMS verification code sent to a specific number.
        Uses run_in_executor for synchronous Twilio calls.
        """
        timeout_seconds = self.config.get("TWILIO_SMS_TIMEOUT_SECONDS")
        retry_delay = self.config.get("TWILIO_SMS_RETRY_DELAY")
        await self._log_op('info', f"Polling Twilio for SMS code to {to_phone_number} (Timeout: {timeout_seconds}s)")

        if not self.twilio_client:
            await self._log_op('error', "Twilio client not initialized. Cannot retrieve SMS code.")
            return None

        start_time = datetime.utcnow()
        # Look back slightly further to catch messages sent just before polling starts
        search_start_time = start_time - timedelta(seconds=90)
        loop = asyncio.get_running_loop()
        code_found = None

        while datetime.utcnow() - start_time < timedelta(seconds=timeout_seconds):
            try:
                await self._log_op('debug', f"Checking Twilio messages to {to_phone_number} sent after {search_start_time.isoformat()}Z")
                # --- List Messages (blocking call) ---
                messages = await loop.run_in_executor(
                    None,
                    lambda: self.twilio_client.messages.list(
                        to=to_phone_number,
                        date_sent_after=search_start_time,
                        limit=10 # Limit results per poll to avoid fetching too much history
                    )
                )

                if messages:
                    await self._log_op('debug', f"Found {len(messages)} potential message(s) for {to_phone_number}.")
                    # Process latest message first
                    for message in sorted(messages, key=lambda m: m.date_sent, reverse=True):
                        # Ensure message body is not None
                        body_text = message.body if message.body else ""
                        await self._log_op('info', f"Processing SMS from {message.from_}: '{body_text[:50]}...'")

                        # Extract code (sync helper method)
                        code = self._extract_code_from_text(body_text)
                        if code:
                            await self._log_op('info', f"Verification code found via Twilio SMS: {code}")
                            code_found = code
                            # TODO: Integrate budget tracking call here if needed
                            # Example: await self.orchestrator.track_expense(0.0075, "Twilio", ...)
                            break # Exit inner loop
                        else:
                            await self._log_op('debug', f"No code pattern matched in SMS body from {message.from_}.")
                    if code_found: break # Exit outer loop
                else:
                    await self._log_op('debug', f"No new messages found for {to_phone_number} yet. Retrying...")

            except TwilioRestException as e:
                await self._log_op('error', f"Twilio API error checking messages for {to_phone_number}: Status={e.status}, Msg={e.msg}")
                logger.error(f"Twilio API error checking messages for {to_phone_number}: {e}")
                # Decide if error is fatal or retryable based on status code
                if e.status == 401: # Auth error
                     await self._log_op('critical', "Twilio Authentication Failed while checking messages.")
                     return None # Fatal error for this attempt
                await asyncio.sleep(retry_delay * 2) # Longer delay on API error before retrying
            except Exception as e:
                await self._log_op('error', f"Unexpected error checking Twilio messages for {to_phone_number}: {e}")
                logger.exception(f"Unexpected error checking Twilio messages for {to_phone_number}")
                await asyncio.sleep(retry_delay * 2) # Longer delay on unexpected error

            # Wait before the next poll if code not found
            if not code_found:
                await asyncio.sleep(retry_delay)

        # Loop finished
        if not code_found:
            await self._log_op('error', f"Timeout reached ({timeout_seconds}s) waiting for SMS code via Twilio for {to_phone_number}.")
        return code_found # Return code or None if timeout

    def _extract_code_from_text(self, text: str) -> Optional[str]:
        """Extracts a 4-8 digit code from a string using predefined patterns."""
        if not text: return None
        # Same patterns as used for email extraction
        patterns = [
            r'verification code is[:\s]*\b(\d{4,8})\b', r'\b(\d{4,8})\b\s*is your.*?code',
            r'code:\s*\b(\d{4,8})\b', r'security code:\s*\b(\d{4,8})\b',
            r'PIN:\s*\b(\d{4,8})\b', r'>\s*(\d{4,8})\s*<',
            r'validation code.*?(\d{4,8})\b'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                logger.debug(f"Code found in text: {match.group(1)}")
                return match.group(1) # Return the first match found
        return None

# ==============================================================================
# == Main Class: BrowsingAgent                                                ==
# ==============================================================================

class BrowsingAgent:
    """
    Main Browsing Agent class, orchestrating actions via internal helper classes.
    Handles initialization, core run loop, task queuing, concurrency, background tasks,
    and reporting to the main Orchestrator. All logic resides within this single file.
    """
    def __init__(self, session_maker: async_sessionmaker, config: Dict[str, Any], orchestrator: Any, clients_models: List[Tuple[Any, str]]):
        """Initializes the BrowsingAgent and its internal helper components."""
        self.session_maker = session_maker
        # Merge provided config with defaults
        self.config = {**DEFAULT_CONFIG, **config}
        self.orchestrator = orchestrator # Expects methods like report_*, send_notification, agents dict, secure_storage
        self.clients_models = clients_models
        self.think_tool = orchestrator.agents.get('think') # Optional ThinkTool

        # --- Securely fetch sensitive config needed by helpers ---
        # Use a helper method to prioritize secure storage, fallback to config dict
        smartproxy_password = self._get_secure_config("smartproxy/password", "SMARTPROXY_PASSWORD")
        hostinger_imap_pass = self._get_secure_config("hostinger/imap_password", "HOSTINGER_IMAP_PASS")
        twilio_sid = self._get_secure_config("twilio/account_sid", "TWILIO_ACCOUNT_SID")
        twilio_token = self._get_secure_config("twilio/auth_token", "TWILIO_AUTH_TOKEN")

        # Add fetched secrets to config dict temporarily for helper init, or pass directly
        self.config['SMARTPROXY_PASSWORD'] = smartproxy_password
        self.config['HOSTINGER_IMAP_PASS'] = hostinger_imap_pass
        # Twilio creds passed directly to VerificationHandler

        # --- Initialize Internal Helpers ---
        self.cache_db = self._initialize_cache_db() # Initialize cache DB connection
        # Instantiate internal helpers, passing necessary dependencies
        self._proxy_manager = _ProxyManager(self.config)
        self._strategy_manager = _StrategyManager(self.config, self.clients_models, self.think_tool, self.cache_db)
        self._verification_handler = _VerificationHandler(self.orchestrator, self.config, twilio_sid, twilio_token)

        # --- Placeholders for other internal handlers (if needed for further organization) ---
        # These might just be groups of private methods instead of full classes
        # self._account_creation_logic = _AccountCreationLogic(self)
        # self._ugc_logic = _UgcLogic(self)
        # self._api_discovery_logic = _ApiDiscoveryLogic(self)

        # --- Core Agent State ---
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.max_concurrency: int = self.config.get("BROWSING_AGENT_MAX_CONCURRENCY")
        self.browser_semaphore: asyncio.Semaphore = asyncio.Semaphore(self.max_concurrency)
        self.active_browsers: int = 0
        self.required_services: List[str] = self.config.get("REQUIRED_API_SERVICES")

        # Meta prompt for direct use if needed (though mostly used by StrategyManager now)
        self.meta_prompt = self.config.get("BROWSING_AGENT_META_PROMPT", BROWSING_AGENT_META_PROMPT_UPDATED)

        self.log_operation('info', "BrowsingAgent initialized with internal helpers.")

    def _get_secure_config(self, vault_key: str, config_key: str) -> Optional[str]:
        """Safely retrieves sensitive config, prioritizing secure storage."""
        value = None
        # 1. Try Orchestrator's secure storage first
        try:
            if hasattr(self.orchestrator, 'secure_storage') and hasattr(self.orchestrator.secure_storage, 'get_secret'):
                 value = self.orchestrator.secure_storage.get_secret(vault_key)
                 if value:
                     logger.debug(f"Retrieved '{config_key}' from secure storage.")
                     return value # Return immediately if found in vault
        except Exception as e:
            logger.error(f"Error fetching secure config for '{vault_key}' from orchestrator: {e}")
            # Log operationally without awaiting (fire-and-forget)
            asyncio.create_task(self.log_operation('error', f"Error fetching secure config for '{vault_key}': {e}"))

        # 2. Fallback to the main config dictionary
        value = self.config.get(config_key)
        if value:
            logger.warning(f"Retrieved sensitive config '{config_key}' from main config dict. Recommend using secure storage.")
            asyncio.create_task(self.log_operation('warning', f"Using sensitive config '{config_key}' from main config dict."))
        else:
             logger.warning(f"Sensitive config key '{config_key}' (Vault: '{vault_key}') not found in secure storage or config dict.")
             asyncio.create_task(self.log_operation('warning', f"Sensitive config key '{config_key}' not found."))

        return value

    def _initialize_cache_db(self) -> sqlite3.Connection:
        """Initializes and returns the cache DB connection. Ensures tables exist."""
        db_path = self.config.get("CACHE_DB_PATH")
        try:
            # Connect to the cache DB file, allow access from executor threads
            cache_db = sqlite3.connect(db_path, check_same_thread=False, timeout=10.0) # Add timeout
            cache_db.execute("PRAGMA journal_mode=WAL;") # Use WAL mode for better concurrency
            cursor = cache_db.cursor()
            # Ensure tables exist (idempotent)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_cache (
                    id INTEGER PRIMARY KEY,
                    service TEXT NOT NULL,
                    strategy TEXT NOT NULL, -- JSON representation of steps
                    success_rate REAL DEFAULT 0.8,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_attempted DATETIME,
                    last_succeeded DATETIME
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_service ON strategy_cache (service);')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attempt_logs (
                    id INTEGER PRIMARY KEY,
                    strategy_id INTEGER, -- Link to strategy_cache if applicable
                    service TEXT NOT NULL,
                    step_index INTEGER,
                    action TEXT,
                    selector TEXT,
                    success BOOLEAN NOT NULL,
                    error_type TEXT,
                    error_message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(strategy_id) REFERENCES strategy_cache(id) ON DELETE SET NULL
                )
            ''')
            cache_db.commit()
            self.log_operation('info', f"Cache database initialized/verified at '{db_path}'.")
            return cache_db
        except sqlite3.Error as e:
            msg = f"CRITICAL: Failed to initialize cache database '{db_path}': {e}"
            self.log_operation('critical', msg) # Log operationally
            logger.critical(msg, exc_info=True) # Log standard with traceback
            raise RuntimeError(msg) from e # Propagate critical error to stop agent init

    async def log_operation(self, level: str, message: str):
        """Helper to log to the operational log file with agent context."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']:
            prefix = f"**{level.upper()}:** "
        try:
            # Add main agent context prefix
            log_func(f"- [Agent] {prefix}{message}")
        except Exception as log_err:
            # Fallback if operational logger fails
            print(f"OPERATIONAL LOG FAILED (Agent): {level} - {message} | Error: {log_err}")
            logger.error(f"Failed to write to operational log from Agent: {log_err}")

    async def get_allowed_concurrency(self) -> int:
        """Adjusts concurrency based on system resource usage."""
        try:
            # Use interval=0.1 for a non-blocking check
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent

            cpu_threshold = self.config.get("CPU_USAGE_THRESHOLD")
            mem_threshold = self.config.get("MEMORY_USAGE_THRESHOLD")
            min_concurrency = self.config.get("MIN_CONCURRENCY")
            max_conc = self.max_concurrency # Use the agent's max_concurrency

            allowed = max_conc # Start with max allowed

            # Scale down if thresholds are exceeded
            if cpu_usage > cpu_threshold:
                # Calculate reduction factor (0 at 100% usage, 1 at threshold)
                reduction = max(0, 1 - (cpu_usage - cpu_threshold) / (100 - cpu_threshold))
                allowed = min(allowed, int(max_conc * reduction * 0.75)) # Heavier penalty for CPU
                logger.debug(f"CPU usage ({cpu_usage}%) > threshold ({cpu_threshold}%). Scaling concurrency.")

            if memory_usage > mem_threshold:
                 reduction = max(0, 1 - (memory_usage - mem_threshold) / (100 - mem_threshold))
                 allowed = min(allowed, int(max_conc * reduction * 0.9)) # Slightly less penalty for memory
                 logger.debug(f"Memory usage ({memory_usage}%) > threshold ({mem_threshold}%). Scaling concurrency.")

            # Ensure concurrency is at least the minimum and an integer
            final_allowed = max(min_concurrency, int(allowed))
            # logger.debug(f"Resource check: CPU={cpu_usage:.1f}%, Mem={memory_usage:.1f}%. Allowed concurrency: {final_allowed}/{max_conc}")
            return final_allowed
        except Exception as e:
            await self.log_operation('error', f"Failed to get system resource usage: {e}")
            logger.error(f"Failed to get system resource usage: {e}", exc_info=True)
            # Fallback to a safe value (e.g., half of max)
            return max(1, self.max_concurrency // 2)

    async def start_background_tasks(self):
        """Starts background tasks managed by the agent."""
        await self.log_operation('info', "Starting background tasks...")
        # These tasks call methods defined within this BrowsingAgent class
        self._api_key_check_task = asyncio.create_task(self._check_and_create_api_keys_periodically())
        self._account_reuse_task = asyncio.create_task(self._reuse_accounts_periodically())
        # Add cancellation logic for these tasks on agent shutdown if needed

    # --- Background Task Implementations (Private methods within BrowsingAgent) ---

    async def _check_and_create_api_keys_periodically(self, interval_seconds: int = 3600):
        """Periodically check for missing API keys for required services."""
        await self.log_operation('info', f"Starting periodic check for required API keys ({self.required_services}) every {interval_seconds}s.")
        while True:
            try:
                await self._check_and_create_api_keys()
            except asyncio.CancelledError:
                await self.log_operation('info', "API key check task cancelled.")
                break
            except Exception as e:
                 await self.log_operation('error', f"Error during periodic API key check: {e}")
                 logger.exception("Error during periodic API key check")
            # Wait interval before next check
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                 await self.log_operation('info', "API key check task cancelled during sleep.")
                 break

    async def _check_and_create_api_keys(self):
        """Check for missing API keys and queue creation tasks."""
        await self.log_operation('info', f"Checking required API keys: {self.required_services}")
        queued_count = 0
        # Assumes existence of a utility function to check credentials
        # This utility needs access to the session_maker or a session
        from utils.account_manager import check_service_credentials_exist

        async with self.session_maker() as session:
            for service in self.required_services:
                try:
                    # Check if valid, available credentials exist for the service
                    has_creds = await check_service_credentials_exist(service, session)
                    if not has_creds:
                        await self.log_operation('info', f"No available account/key for '{service}'. Queuing creation task.")
                        logger.info(f"Queuing account creation task for missing API key: {service}")
                        # Add task to the agent's own queue
                        await self.task_queue.put({
                            'action': 'create_account',
                            'service_url': f"https://{service}", # Construct URL if needed
                            'priority': 10 # Give high priority
                        })
                        queued_count += 1
                    else:
                         await self.log_operation('debug', f"Available account/key confirmed for required service: {service}.")
                except Exception as check_err:
                    # Catch errors during the check itself (DB or logic error)
                    await self.log_operation('error', f"Error checking API key status for {service}: {check_err}")
                    logger.error(f"Error checking API key status for {service}: {check_err}", exc_info=True)
            await self.log_operation('info', f"Finished checking required API keys. Queued {queued_count} creation tasks.")

    async def _reuse_accounts_periodically(self, interval_seconds: int = 86400): # Default daily check
        """Periodically check and reuse accounts with reset credits."""
        await self.log_operation('info', f"Starting periodic account reuse check every {interval_seconds}s.")
        while True:
            try:
                await self._reuse_accounts()
            except asyncio.CancelledError:
                await self.log_operation('info', "Account reuse task cancelled.")
                break
            except Exception as e:
                await self.log_operation('error', f"Error during periodic account reuse check: {e}")
                logger.exception("Error during periodic account reuse check")
            # Wait interval before next check
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                 await self.log_operation('info', "Account reuse task cancelled during sleep.")
                 break

    async def _reuse_accounts(self):
        """Check for services with monthly credit resets and notify."""
        await self.log_operation('info', "Checking for reusable accounts (credit reset)...")
        reused_count = 0
        async with self.session_maker() as session:
            recurring_services = self.config.get("RECURRING_CREDIT_SERVICES")
            if not recurring_services:
                await self.log_operation('debug', "No recurring credit services configured. Skipping reuse check.")
                return

            try:
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                # Adapt query based on actual DB dialect (assuming PostgreSQL)
                # Fetches accounts for specified services created more than 30 days ago that are marked available
                stmt = sqlalchemy.text("""
                    SELECT id, service, email, created_at FROM accounts
                    WHERE service = ANY(CAST(:services AS text[]))
                      AND is_available = TRUE
                      AND created_at <= :cutoff_date
                """)
                accounts_result = await session.execute(
                    stmt,
                    {"services": list(recurring_services), "cutoff_date": thirty_days_ago}
                )
                accounts_to_check = accounts_result.mappings().all()
            except Exception as db_e:
                await self.log_operation('error', f"DB error fetching recurring accounts: {db_e}")
                logger.error(f"DB error fetching recurring accounts: {db_e}", exc_info=True)
                return # Cannot proceed

            if not accounts_to_check:
                await self.log_operation('info', "No accounts found matching reuse criteria (older than 30 days, available).")
                return

            await self.log_operation('info', f"Found {len(accounts_to_check)} potential accounts for credit reset reuse.")
            for account in accounts_to_check:
                service = account.get('service')
                email = account.get('email')
                account_id = account.get('id')
                # Log and notify orchestrator that credits *may* have reset
                await self.log_operation('info', f"Credits likely reset for {service} account {email} (ID: {account_id}). Flagging for potential reuse.")
                logger.info(f"Credits likely reset for {service} account {email}")
                if hasattr(self.orchestrator, 'send_notification'):
                    try:
                        await self.orchestrator.send_notification(
                            "Account Credit Reset",
                            f"Credits may have reset for {service} account {email} (ID: {account_id}). Ready for potential reuse."
                        )
                    except Exception as notify_err:
                         await self.log_operation('error', f"Failed to send notification for account reuse: {notify_err}")
                reused_count += 1
                # Optional: Update a timestamp on the account record, e.g., 'last_reuse_check'
                # await session.execute(sqlalchemy.text("UPDATE accounts SET last_reuse_check = :now WHERE id = :id"), {"now": datetime.utcnow(), "id": account_id})

            # await session.commit() # Commit if updates were made
            await self.log_operation('info', f"Finished checking reusable accounts. Notified/flagged {reused_count} accounts.")

    # --- Main Run Loop ---
    async def run(self):
        """Main loop processing tasks from the queue using internal helpers."""
        await self.log_operation('info', f"BrowsingAgent run loop started. Max concurrency: {self.max_concurrency}")
        await self.start_background_tasks() # Start periodic checks

        while True:
            task = None
            proxy = None
            task_acquired = False
            action = None
            task_start_time = None

            try:
                # --- Concurrency Check ---
                allowed_concurrency = await self.get_allowed_concurrency()
                if self.active_browsers >= allowed_concurrency:
                    await self.log_operation('debug', f"Concurrency limit reached ({self.active_browsers}/{allowed_concurrency}). Waiting...")
                    # Wait briefly before checking again to avoid busy-waiting
                    await asyncio.sleep(random.uniform(1.0, 3.0))
                    continue

                # --- Task Acquisition ---
                await self.log_operation('debug', f"Waiting for task. Queue: {self.task_queue.qsize()}. Active: {self.active_browsers}/{allowed_concurrency}")
                # Wait for a task, with a timeout to allow periodic checks even if queue is empty
                task = await asyncio.wait_for(self.task_queue.get(), timeout=60.0)
                task_acquired = True
                task_start_time = time.monotonic()
                action = task.get('action')
                # Ensure service_name is derived and added to task dict for handlers
                service_name = task.get('service_name') or \
                               (task.get('service_url','').split('//')[-1].split('/')[0].replace('www.', '')
                                if task.get('service_url') else 'unknown_service')
                task['service_name'] = service_name

                await self.log_operation('info', f"Dequeued task: {action} for '{service_name}'. WF ID: {task.get('workflow_id')}")

                # --- Resource Acquisition (Semaphore & Proxy) ---
                await self.browser_semaphore.acquire()
                self.active_browsers += 1
                await self.log_operation('debug', f"Semaphore acquired. Active browsers: {self.active_browsers}")

                # Get proxy using the internal manager
                proxy = await self._proxy_manager.get_next_proxy(service_name)
                if not proxy:
                    # If no proxy available, release semaphore immediately and raise error to skip task
                    raise ValueError(f"No proxy available for task {action} on {service_name}") # Use specific error

                task['proxy'] = proxy # Add acquired proxy to task data for the handler

                # --- Task Dispatching to Internal Handlers ---
                handler_method = None
                if action == 'create_account': handler_method = self._handle_create_account
                elif action == 'acquire_or_verify_account': handler_method = self._handle_acquire_or_verify
                elif action == 'generate_ugc_video': handler_method = self._handle_generate_ugc
                elif action == 'edit_ugc_video': handler_method = self._handle_edit_ugc
                elif action == 'discover_api': handler_method = self._handle_discover_api
                # Add elif for other actions...

                if handler_method:
                    await self.log_operation('debug', f"Dispatching action '{action}' to internal handler {handler_method.__name__}")
                    # Execute the handler method - it's responsible for its own logic and browser management
                    await handler_method(task)
                else:
                    # Action not recognized or handler not implemented
                    await self.log_operation('warning', f"Unknown or unhandled task action received: {action}. Task: {task}")
                    logger.warning(f"Unknown or unhandled task action received: {action}. Task: {task}")
                    # Report failure if part of a workflow
                    if task.get('workflow_id'):
                        await self._report_step_failure(task['workflow_id'], action, "Unknown/unhandled action", task.get('workflow_state', {}))

            except asyncio.TimeoutError:
                # No task received from queue within timeout - this is normal
                await self.log_operation('debug', "No task in queue timeout. Continuing loop.")
                task_acquired = False # Ensure finally block doesn't try to release resources for a non-existent task
                continue # Continue loop normally
            except ValueError as ve: # Catch specific "No proxy available" error
                 await self.log_operation('error', str(ve))
                 # No proxy was acquired, but semaphore might have been if error was later
                 # The finally block handles semaphore release if task_acquired is True
            except Exception as e:
                # Catch all other unexpected errors during task acquisition or dispatch
                await self.log_operation('critical', f"CRITICAL ERROR in run loop (before/during dispatch) for task {task}: {type(e).__name__} - {e}")
                logger.exception(f"CRITICAL Error processing task {task}")
                # Report failure if a task was acquired and part of a workflow
                if task_acquired and task and task.get('workflow_id'):
                    try:
                        await self._report_step_failure(task['workflow_id'], action or 'unknown', f'Agent loop error: {e}', task.get('workflow_state', {}))
                    except Exception as report_err:
                        logger.error(f"Failed to report critical error to orchestrator: {report_err}")
                # Avoid fast loop on critical errors
                await asyncio.sleep(random.uniform(5.0, 15.0))

            finally:
                # --- Resource Release & Task Completion ---
                # Determine if the task processing itself (handler call) failed
                task_failed_in_handler = 'e' in locals() and isinstance(locals()['e'], Exception) and task_acquired

                # Release proxy if one was acquired for this task attempt
                if proxy:
                    # Success is True if no exception occurred *after* proxy acquisition
                    await self._proxy_manager.release_proxy(proxy, success=not task_failed_in_handler)
                    proxy = None # Clear proxy variable

                # Release semaphore and mark task done ONLY if a task was acquired and attempted
                if task_acquired:
                    if self.active_browsers > 0:
                        self.active_browsers -= 1
                        self.browser_semaphore.release()
                        # Log duration if task completed (successfully or not)
                        if task_start_time:
                            duration = time.monotonic() - task_start_time
                            await self.log_operation('debug', f"Semaphore released. Active: {self.active_browsers}. Task '{action}' duration: {duration:.2f}s")
                        else:
                            await self.log_operation('debug', f"Semaphore released. Active: {self.active_browsers}")
                    else:
                        # This indicates a potential logic error in semaphore management
                        await self.log_operation('warning', "Semaphore release mismatch detected (active_browsers <= 0).")

                    # Mark task as done in the queue
                    try:
                        self.task_queue.task_done()
                        await self.log_operation('debug', f"Task marked done: {action} for {task.get('service_name', 'N/A')}")
                    except ValueError: # Handle case where task might already be done (e.g., race condition)
                        await self.log_operation('warning', f"Task done ValueError (already done?) for task: {task}")
                    except Exception as td_err:
                        await self.log_operation('error', f"Error calling task_done for {task}: {td_err}")

                # Reset loop variables for clarity
                task, proxy, action, task_acquired, task_start_time = None, None, None, False, None

                # Brief pause before next loop iteration to prevent busy-waiting if queue empties
                await asyncio.sleep(random.uniform(self.config.get("AGENT_LOOP_DELAY_MIN"), self.config.get("AGENT_LOOP_DELAY_MAX")))


    # --- Internal Task Handler Method Structures (Implementations in next part) ---

    async def _handle_create_account(self, task: Dict):
        """Handles the 'create_account' action by coordinating internal helpers."""
        service_url = task.get('service_url')
        proxy = task.get('proxy') # Proxy acquired by run loop
        workflow_id = task.get('workflow_id')
        workflow_state = task.get('workflow_state', {})
        service_name = task.get('service_name') # Should be set by run loop
        max_retries = self.config.get("ACCOUNT_CREATION_MAX_RETRIES")
        current_retry = task.get('retries', 0)
        excluded_strategy_ids = task.get('excluded_strategy_ids', [])

        await self.log_operation('info', f"Starting account creation for {service_url} (Retry {current_retry}/{max_retries}) via proxy {proxy['server']}")

        browser, context, page = None, None, None
        strategy_id_used = None
        success = False
        final_result_data = {}
        failure_reason = "Unknown failure during account creation."
        failure_context = None # Initialize failure context

        try:
            # 1. Launch Browser
            browser, context, page = await self._launch_browser_with_context(proxy)
            if not page:
                raise RuntimeError("Failed to launch browser page.")

            # 2. Generate Credentials
            email_alias = await self._verification_handler.generate_hostinger_alias(service_name)
            if not email_alias:
                raise ValueError("Failed to generate email alias.")
            password = fake.password(length=12, special_chars=True, digits=True, upper_case=True, lower_case=True)
            name = fake.name()
            # TODO: Add phone number generation/acquisition if needed by strategy

            # Prepare context data for step execution
            step_context_data = {
                "email": email_alias,
                "password": password,
                "name": name,
                # "phone_number": phone_number # Add if generated
            }

            # 3. Load and Try Cached Strategies
            strategies = await self._strategy_manager.load_strategies(service_name)
            executed_strategy = False
            for strategy in strategies:
                strategy_id = strategy.get('id')
                if strategy_id in excluded_strategy_ids:
                    await self.log_operation('debug', f"Skipping excluded strategy ID {strategy_id}")
                    continue

                strategy_steps = strategy.get('steps')
                await self.log_operation('info', f"Attempting cached strategy ID {strategy_id} ({len(strategy_steps)} steps) for {service_name}")
                strategy_id_used = strategy_id
                exec_result = await self._execute_signup_steps(page, strategy_steps, strategy_id, service_name, step_context_data)
                executed_strategy = True # Mark that at least one strategy was attempted

                if exec_result.get('success'):
                    await self.log_operation('info', f"Cached strategy {strategy_id} successful for {service_name}.")
                    success = True
                    break # Exit loop on success
                else:
                    # Strategy failed, log and try next (or refine later)
                    failure_reason = exec_result.get('error', 'Cached strategy execution failed.')
                    await self.log_operation('warning', f"Cached strategy {strategy_id} failed for {service_name}: {failure_reason}")
                    await self._strategy_manager.update_strategy_success(strategy_id, success=False)
                    # Store failure context for potential refinement
                    failure_context = {
                        'step_index': exec_result.get('step_index'),
                        'error': failure_reason,
                        'page_url': page.url,
                        # 'html_snippet': await page.content() # Potentially large, use carefully
                    }
                    # Attempt refinement immediately? Or after trying all cached? Let's try after all cached fail.
                    # await self._refine_and_retry_strategy(service_name, strategy_id, failure_context, task) # This would requeue

            # 4. If No Cached Strategy Succeeded (or none existed/valid) -> Infer and Execute
            if not success and not executed_strategy: # Also handle case where no strategies were loaded/valid
                 await self.log_operation('info', f"No suitable cached strategies found or all failed for {service_name}. Attempting inference.")
                 inferred_steps = await self._strategy_manager.infer_signup_steps(service_url)
                 if inferred_steps:
                     await self.log_operation('info', f"Executing inferred strategy ({len(inferred_steps)} steps) for {service_name}")
                     strategy_id_used = None # Inferred strategy doesn't have an ID yet
                     exec_result = await self._execute_signup_steps(page, inferred_steps, None, service_name, step_context_data)
                     if exec_result.get('success'):
                         await self.log_operation('info', f"Inferred strategy successful for {service_name}.")
                         success = True
                         # Save the successful inferred strategy
                         new_id = await self._strategy_manager.save_strategy(service_name, inferred_steps)
                         if new_id:
                             strategy_id_used = new_id # Update ID for success reporting
                             await self.log_operation('info', f"Saved successful inferred strategy as ID {new_id}.")
                         else:
                              await self.log_operation('error', f"Failed to save successful inferred strategy for {service_name}.")
                     else:
                         failure_reason = exec_result.get('error', 'Inferred strategy execution failed.')
                         await self.log_operation('warning', f"Inferred strategy failed for {service_name}: {failure_reason}")
                         # TODO: Consider if we should attempt refinement on a failed *inferred* strategy? Maybe not on first try.
                 else:
                     failure_reason = "Failed to infer signup strategy."
                     await self.log_operation('error', failure_reason)

            # 5. If Signup Succeeded -> Extract API Key & Store
            if success:
                api_key = await self._extract_api_key_after_signup(page, service_name)
                if api_key:
                    await self.log_operation('info', f"Successfully extracted API key for {service_name}.")
                else:
                    await self.log_operation('warning', f"Account created for {service_name}, but failed to extract API key.")

                # Store account details regardless of API key extraction success (account exists)
                store_success = await self._store_account_details(service_name, email_alias, password, api_key)
                if not store_success:
                    # Log error, but consider the overall step successful if account was created
                    await self.log_operation('error', f"Account created for {service_name}, but failed to store details in DB.")
                    # Don't mark overall success as False here, but maybe add warning to result

                final_result_data = {
                    'email': email_alias,
                    'service': service_name,
                    'api_key_extracted': bool(api_key),
                    'api_key': api_key, # Include key if found
                    'account_stored': store_success
                }
                # Update success rate for the strategy used (if it had an ID)
                if strategy_id_used:
                    await self._strategy_manager.update_strategy_success(strategy_id_used, success=True)

            # 6. Handle Failure (Refinement/Retry) - Only if no success achieved yet
            if not success:
                 # If a cached strategy was attempted and failed, try refining it
                 if strategy_id_used and failure_context: # failure_context set during cached strategy failure
                     requeued = await self._refine_and_retry_strategy(service_name, strategy_id_used, failure_context, task)
                     if requeued:
                         # Task was requeued, don't report failure for *this* attempt
                         await self.log_operation('info', f"Task for {service_name} requeued after refinement attempt.")
                         # Need to exit cleanly without reporting failure for this run
                         return # Exit handler early
                 elif current_retry < max_retries:
                     # Generic retry if no specific strategy failed or refinement failed/not applicable
                     await self.log_operation('warning', f"Account creation failed for {service_name}. Requeuing for retry ({current_retry+1}/{max_retries}). Reason: {failure_reason}")
                     task['retries'] = current_retry + 1
                     task['priority'] = task.get('priority', 5) + 1 # Slightly lower priority
                     await self.task_queue.put(task)
                     return # Exit handler early, task requeued
                 else:
                     await self.log_operation('error', f"Account creation failed for {service_name} after {max_retries} retries. Giving up. Reason: {failure_reason}")
                     # Proceed to report failure below

        except (RuntimeError, ValueError, PlaywrightTimeoutError) as setup_err:
            # Catch errors during setup (browser launch, alias gen) or critical Playwright errors
            failure_reason = f"Setup/Critical Error: {setup_err}"
            await self.log_operation('error', f"Account creation failed for {service_name}. Reason: {failure_reason}")
            success = False
            # Attempt retry if possible
            if current_retry < max_retries:
                 await self.log_operation('warning', f"Requeuing task for {service_name} due to setup error (Retry {current_retry+1}/{max_retries}).")
                 task['retries'] = current_retry + 1
                 task['priority'] = task.get('priority', 5) + 1
                 await self.task_queue.put(task)
                 return # Exit handler early
            else:
                 await self.log_operation('error', f"Max retries reached for {service_name} after setup error. Giving up.")

        except Exception as e:
            failure_reason = f"Unexpected Error: {type(e).__name__} - {e}"
            await self.log_operation('critical', f"Unexpected error during account creation for {service_name}: {failure_reason}")
            logger.exception(f"Unexpected error creating account for {service_name}")
            success = False
            # Attempt retry if possible
            if current_retry < max_retries:
                 await self.log_operation('warning', f"Requeuing task for {service_name} due to unexpected error (Retry {current_retry+1}/{max_retries}).")
                 task['retries'] = current_retry + 1
                 task['priority'] = task.get('priority', 5) + 1
                 await self.task_queue.put(task)
                 return # Exit handler early
            else:
                 await self.log_operation('error', f"Max retries reached for {service_name} after unexpected error. Giving up.")

        finally:
            # 7. Close Browser/Context
            if page:
                try: await page.close()
                except Exception as page_close_err: await self.log_operation('warning', f"Error closing page: {page_close_err}")
            if context:
                try: await context.close()
                except Exception as context_close_err: await self.log_operation('warning', f"Error closing context: {context_close_err}")
            if browser:
                try: await browser.close()
                except Exception as browser_close_err: await self.log_operation('warning', f"Error closing browser: {browser_close_err}")
            await self.log_operation('debug', f"Browser resources closed for {service_name} task.")

        # 8. Report Result (only if task wasn't requeued)
        if workflow_id:
            # Map create_account failure/success to the acquire step in workflow reporting
            reporting_step = 'acquire_or_verify_account'
            if success:
                await self._report_step_success(workflow_id, reporting_step, final_result_data, workflow_state)
            else:
                await self._report_step_failure(workflow_id, reporting_step, failure_reason, workflow_state)
        else:
             await self.log_operation('debug', f"No workflow ID for {service_name} task. Skipping report.")


    async def _handle_acquire_or_verify(self, task: Dict):
        """Checks for existing credentials; if none, queues account creation."""
        service_name = task.get('service_name')
        workflow_id = task.get('workflow_id')
        workflow_state = task.get('workflow_state', {})
        # proxy = task.get('proxy') # Proxy might not be needed just for check

        await self.log_operation('info', f"Starting acquire/verify handler for {service_name}")
        credentials = None
        check_error = None

        try:
            from utils.account_manager import get_valid_credentials
            async with self.session_maker() as session:
                # Get credentials, potentially marking one as 'in_use' if applicable by the function
                credentials = await get_valid_credentials(session, service_name)
        except ImportError:
            check_error = "Failed to import utils.account_manager.get_valid_credentials"
            await self.log_operation('critical', check_error)
            logger.critical(check_error)
        except Exception as e:
            check_error = f"DB Error checking credentials for {service_name}: {e}"
            await self.log_operation('error', check_error)
            logger.error(check_error, exc_info=True)

        if check_error:
            # Report failure if we couldn't even check
            if workflow_id:
                await self._report_step_failure(workflow_id, 'acquire_or_verify_account', check_error, workflow_state)
            return # Stop processing this task

        if credentials:
            await self.log_operation('info', f"Valid credentials found for {service_name} (Email: {credentials.get('email')}).")
            # TODO (Future): Add verification step here if needed. Launch browser, login, check status.
            # If verification fails, mark account as unavailable and proceed to queue creation.

            # Report success for this step
            if workflow_id:
                result_data = {
                    'service': service_name,
                    'email': credentials.get('email'),
                    'credentials_found': True,
                    # Include other relevant non-sensitive details if needed
                }
                await self._report_step_success(workflow_id, 'acquire_or_verify_account', result_data, workflow_state)
        else:
            # No valid credentials found, queue creation task
            await self.log_operation('warning', f"No valid credentials found for {service_name}. Queuing account creation.")
            # Create a new task dictionary for account creation
            creation_task = {
                'action': 'create_account',
                'service_url': task.get('service_url') or f"https://{service_name}", # Reconstruct URL if needed
                'service_name': service_name,
                'workflow_id': workflow_id,
                'workflow_state': workflow_state, # Pass workflow state along
                'priority': task.get('priority', 5) + 1, # Slightly higher priority than original task? Or same?
                'retries': 0, # Start retries for creation
                'excluded_strategy_ids': []
            }
            await self.task_queue.put(creation_task)
            # IMPORTANT: Do *not* report completion for the original 'acquire_or_verify_account' step here.
            # The workflow will proceed once the 'create_account' step (which reports as 'acquire_or_verify_account') completes.
            await self.log_operation('debug', f"Queued 'create_account' task for {service_name} as part of WF {workflow_id}.")


    async def _handle_generate_ugc(self, task: Dict):
        """Handles UGC video generation task (Simplified Implementation)."""
        service_name = task.get('target_service') or task.get('service_name') # Use target_service for UGC if available
        proxy = task.get('proxy')
        workflow_id = task.get('workflow_id')
        workflow_state = task.get('workflow_state', {})
        script = task.get('script', 'Default script text.') # Get script from task
        avatar_details = task.get('avatar_details', {}) # Get avatar details

        await self.log_operation('info', f"Starting UGC generation handler for {service_name}")
        browser, context, page = None, None, None
        success = False
        result_data = {}
        failure_reason = "Unknown failure during UGC generation."

        try:
            # 1. Get Credentials
            from utils.account_manager import get_valid_credentials
            async with self.session_maker() as session:
                credentials = await get_valid_credentials(session, service_name)
            if not credentials:
                raise ValueError(f"No valid credentials found for UGC service: {service_name}")
            await self.log_operation('debug', f"Using credentials for {credentials.get('email')} for UGC generation.")

            # 2. Launch Browser
            browser, context, page = await self._launch_browser_with_context(proxy)
            if not page: raise RuntimeError("Failed to launch browser page for UGC.")

            # 3. Login (Placeholder/Simplified)
            await self.log_operation('debug', f"Simulating login to {service_name}...")
            # await self._login_to_service(service_name, page, credentials) # << Replace with real call later
            await page.wait_for_timeout(random.uniform(1000, 3000)) # Simulate login time
            if random.random() < 0.05: raise RuntimeError("Simulated login failure.") # Simulate occasional login fail

            # 4. Generate Video (Placeholder/Simplified)
            await self.log_operation('debug', f"Simulating video generation on {service_name}...")
            # generated_video_url_or_id = await self._generate_avatar_video(service_name, page, script, avatar_details) # << Replace with real call later
            await page.wait_for_timeout(random.uniform(5000, 15000)) # Simulate generation time
            if random.random() < 0.1: raise RuntimeError("Simulated video generation failure.") # Simulate occasional generation fail
            generated_video_url_or_id = f"simulated_video_{random.randint(1000,9999)}"

            # 5. (Optional) Download - Placeholder
            # download_path = await self._download_asset(...) # << Replace with real call later

            success = True
            result_data = {'video_id': generated_video_url_or_id, 'service': service_name}
            await self.log_operation('info', f"Simulated UGC generation successful for {service_name}.")

        except (RuntimeError, ValueError, PlaywrightTimeoutError) as op_err:
            failure_reason = f"Operation Error: {op_err}"
            await self.log_operation('error', f"UGC generation failed for {service_name}. Reason: {failure_reason}")
            success = False
        except Exception as e:
            failure_reason = f"Unexpected Error: {type(e).__name__} - {e}"
            await self.log_operation('critical', f"Unexpected error during UGC generation for {service_name}: {failure_reason}")
            logger.exception(f"Unexpected error during UGC generation for {service_name}")
            success = False
        finally:
            # 6. Close Browser/Context
            if page:
                try: await page.close()
                except Exception as page_close_err: await self.log_operation('warning', f"Error closing page (UGC Gen): {page_close_err}")
            if context:
                try: await context.close()
                except Exception as context_close_err: await self.log_operation('warning', f"Error closing context (UGC Gen): {context_close_err}")
            if browser:
                try: await browser.close()
                except Exception as browser_close_err: await self.log_operation('warning', f"Error closing browser (UGC Gen): {browser_close_err}")
            await self.log_operation('debug', f"Browser resources closed for {service_name} UGC generation task.")

        # 7. Report Result
        if workflow_id:
            if success:
                await self._report_step_success(workflow_id, 'generate_ugc_video', result_data, workflow_state)
            else:
                await self._report_step_failure(workflow_id, 'generate_ugc_video', failure_reason, workflow_state)
        else:
             await self.log_operation('debug', f"No workflow ID for {service_name} UGC generation task. Skipping report.")


    async def _handle_edit_ugc(self, task: Dict):
        """Handles UGC video editing task (Simplified Implementation)."""
        service_name = task.get('target_service') or task.get('service_name')
        proxy = task.get('proxy')
        workflow_id = task.get('workflow_id')
        workflow_state = task.get('workflow_state', {})
        source_video_id = task.get('source_video_id') # ID/URL of video to edit
        edit_instructions = task.get('edit_instructions', {}) # Editing parameters

        await self.log_operation('info', f"Starting UGC editing handler for {service_name}")
        browser, context, page = None, None, None
        success = False
        result_data = {}
        failure_reason = "Unknown failure during UGC editing."

        if not source_video_id:
             failure_reason = "Missing 'source_video_id' in task for UGC editing."
             await self.log_operation('error', failure_reason)
             if workflow_id: await self._report_step_failure(workflow_id, 'edit_ugc_video', failure_reason, workflow_state)
             return

        try:
            # 1. Get Credentials
            from utils.account_manager import get_valid_credentials
            async with self.session_maker() as session:
                credentials = await get_valid_credentials(session, service_name)
            if not credentials:
                raise ValueError(f"No valid credentials found for UGC service: {service_name}")
            await self.log_operation('debug', f"Using credentials for {credentials.get('email')} for UGC editing.")

            # 2. Launch Browser
            browser, context, page = await self._launch_browser_with_context(proxy)
            if not page: raise RuntimeError("Failed to launch browser page for UGC.")

            # 3. Login (Placeholder/Simplified)
            await self.log_operation('debug', f"Simulating login to {service_name}...")
            # await self._login_to_service(service_name, page, credentials) # << Replace with real call later
            await page.wait_for_timeout(random.uniform(1000, 3000))
            if random.random() < 0.05: raise RuntimeError("Simulated login failure.")

            # 4. Edit Video (Placeholder/Simplified)
            await self.log_operation('debug', f"Simulating video editing on {service_name} for video {source_video_id}...")
            # edited_video_url_or_id = await self._edit_video(service_name, page, source_video_id, edit_instructions) # << Replace with real call later
            await page.wait_for_timeout(random.uniform(8000, 20000)) # Simulate editing time
            if random.random() < 0.1: raise RuntimeError("Simulated video editing failure.")
            edited_video_url_or_id = f"simulated_edited_video_{random.randint(1000,9999)}"

            # 5. (Optional) Download - Placeholder

            success = True
            result_data = {'edited_video_id': edited_video_url_or_id, 'service': service_name}
            await self.log_operation('info', f"Simulated UGC editing successful for {service_name}.")

        except (RuntimeError, ValueError, PlaywrightTimeoutError) as op_err:
            failure_reason = f"Operation Error: {op_err}"
            await self.log_operation('error', f"UGC editing failed for {service_name}. Reason: {failure_reason}")
            success = False
        except Exception as e:
            failure_reason = f"Unexpected Error: {type(e).__name__} - {e}"
            await self.log_operation('critical', f"Unexpected error during UGC editing for {service_name}: {failure_reason}")
            logger.exception(f"Unexpected error during UGC editing for {service_name}")
            success = False
        finally:
            # 6. Close Browser/Context
            if page:
                try: await page.close()
                except Exception as page_close_err: await self.log_operation('warning', f"Error closing page (UGC Edit): {page_close_err}")
            if context:
                try: await context.close()
                except Exception as context_close_err: await self.log_operation('warning', f"Error closing context (UGC Edit): {context_close_err}")
            if browser:
                try: await browser.close()
                except Exception as browser_close_err: await self.log_operation('warning', f"Error closing browser (UGC Edit): {browser_close_err}")
            await self.log_operation('debug', f"Browser resources closed for {service_name} UGC editing task.")

        # 7. Report Result
        if workflow_id:
            if success:
                await self._report_step_success(workflow_id, 'edit_ugc_video', result_data, workflow_state)
            else:
                await self._report_step_failure(workflow_id, 'edit_ugc_video', failure_reason, workflow_state)
        else:
             await self.log_operation('debug', f"No workflow ID for {service_name} UGC editing task. Skipping report.")


    async def _handle_discover_api(self, task: Dict):
        """Handles API discovery task (Simplified Implementation)."""
        service_url = task.get('service_url')
        proxy = task.get('proxy')
        workflow_id = task.get('workflow_id')
        workflow_state = task.get('workflow_state', {})
        service_name = task.get('service_name')

        await self.log_operation('info', f"Starting API discovery handler for {service_url}")
        browser, context, page = None, None, None
        success = False
        result_data = {}
        failure_reason = "Unknown failure during API discovery."
        captured_calls = [] # List to hold simulated captured calls

        try:
            # 1. Launch Browser
            browser, context, page = await self._launch_browser_with_context(proxy)
            if not page: raise RuntimeError("Failed to launch browser page for API discovery.")

            # 2. Setup Network Monitoring (Simplified)
            await self.log_operation('debug', "Simulating network monitoring setup...")
            # In real implementation: use page.on('request', ...) and page.on('response', ...)
            # def handle_request(request): captured_calls.append({'url': request.url, 'method': request.method, ...})
            # await page.on('request', handle_request)

            # 3. Navigate & Interact (Simplified)
            await self.log_operation('debug', f"Navigating to {service_url} for discovery...")
            await page.goto(service_url, wait_until='domcontentloaded')
            await page.wait_for_timeout(random.uniform(3000, 8000)) # Wait for dynamic content/requests

            # Simulate finding some API calls
            if random.random() < 0.7:
                 num_calls = random.randint(1, 5)
                 for _ in range(num_calls):
                     captured_calls.append({
                         'url': f"https://api.{service_name}/v{random.randint(1,3)}/resource{random.randint(10,99)}",
                         'method': random.choice(['GET', 'POST']),
                         'status': random.choice([200, 201, 404])
                     })
                 await self.log_operation('debug', f"Simulated capturing {len(captured_calls)} potential API calls.")

            # 4. Process & Store Results (Simplified)
            if captured_calls:
                 await self.log_operation('info', f"Processing {len(captured_calls)} captured calls for {service_name}...")
                 # In real implementation: Filter, analyze, and store calls (e.g., via ThinkTool KB)
                 result_data = {'discovered_calls': captured_calls, 'service': service_name}
                 success = True
            else:
                 await self.log_operation('warning', f"No potential API calls captured for {service_name}.")
                 # Consider this a success or failure? Let's say success (discovery ran, found nothing)
                 result_data = {'discovered_calls': [], 'service': service_name}
                 success = True


        except (RuntimeError, ValueError, PlaywrightTimeoutError) as op_err:
            failure_reason = f"Operation Error: {op_err}"
            await self.log_operation('error', f"API discovery failed for {service_name}. Reason: {failure_reason}")
            success = False
        except Exception as e:
            failure_reason = f"Unexpected Error: {type(e).__name__} - {e}"
            await self.log_operation('critical', f"Unexpected error during API discovery for {service_name}: {failure_reason}")
            logger.exception(f"Unexpected error during API discovery for {service_name}")
            success = False
        finally:
            # 5. Close Browser/Context
            if page:
                try: await page.close()
                except Exception as page_close_err: await self.log_operation('warning', f"Error closing page (API Disc): {page_close_err}")
            if context:
                try: await context.close()
                except Exception as context_close_err: await self.log_operation('warning', f"Error closing context (API Disc): {context_close_err}")
            if browser:
                try: await browser.close()
                except Exception as browser_close_err: await self.log_operation('warning', f"Error closing browser (API Disc): {browser_close_err}")
            await self.log_operation('debug', f"Browser resources closed for {service_name} API discovery task.")

        # 6. Report Result
        if workflow_id:
            if success:
                await self._report_step_success(workflow_id, 'discover_api', result_data, workflow_state)
            else:
                await self._report_step_failure(workflow_id, 'discover_api', failure_reason, workflow_state)
        else:
             await self.log_operation('debug', f"No workflow ID for {service_name} API discovery task. Skipping report.")


    # --- Workflow Reporting Helpers (Private methods within BrowsingAgent) ---

    async def _report_step_success(self, workflow_id: str, step_name: str, result_data: Dict, current_state: Dict):
        """Reports successful step completion to the orchestrator."""
        if not hasattr(self.orchestrator, 'report_ugc_step_complete'):
            await self.log_operation('warning', f"Orchestrator missing 'report_ugc_step_complete'. Cannot report success: WF={workflow_id}, Step={step_name}.")
            return
        try:
            result_data['success'] = True # Ensure success flag is set
            await self.orchestrator.report_ugc_step_complete(
                workflow_id=workflow_id, completed_step=step_name,
                result=result_data, current_state=current_state
            )
            await self.log_operation('debug', f"Reported step success: WF={workflow_id}, Step={step_name}")
        except Exception as e:
            await self.log_operation('error', f"Failed report step success: WF={workflow_id}, Step={step_name}: {e}")
            logger.error(f"Failed to report step success to orchestrator: {e}", exc_info=True)

    async def _report_step_failure(self, workflow_id: str, step_name: str, reason: str, current_state: Dict):
        """Reports failed step completion to the orchestrator."""
        if not hasattr(self.orchestrator, 'report_ugc_step_complete'):
            await self.log_operation('warning', f"Orchestrator missing 'report_ugc_step_complete'. Cannot report failure: WF={workflow_id}, Step={step_name}.")
            return
        try:
            await self.orchestrator.report_ugc_step_complete(
                workflow_id=workflow_id, completed_step=step_name,
                result={'success': False, 'reason': str(reason)}, # Ensure reason is string
                current_state=current_state
            )
            await self.log_operation('debug', f"Reported step failure: WF={workflow_id}, Step={step_name}, Reason: {str(reason)[:150]}...")
        except Exception as e:
            await self.log_operation('error', f"Failed report step failure: WF={workflow_id}, Step={step_name}: {e}")
            logger.error(f"Failed to report step failure to orchestrator: {e}", exc_info=True)


    # --- Account Creation Helpers ---

    async def _launch_browser_with_context(self, proxy: Optional[Dict] = None) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Launches Playwright browser, context, and page with proxy and fingerprinting."""
        await self.log_operation('debug', f"Launching browser with proxy: {proxy['server'] if proxy else 'None'}")
        browser, context, page = None, None, None
        playwright_instance = None

        try:
            playwright_instance = await async_playwright().start()
            browser_options = {
                'headless': True, # Use headless for production
                # Consider adding args for stealth if needed:
                # 'args': ['--disable-blink-features=AutomationControlled']
            }
            if proxy:
                browser_options['proxy'] = {
                    'server': f"http://{proxy['server']}", # Assuming HTTP proxy endpoint for Smartproxy
                    'username': proxy.get('username'),
                    'password': proxy.get('password')
                }
            browser = await playwright_instance.chromium.launch(**browser_options)

            # --- Fingerprint Configuration (Basic Random Selection) ---
            self.logger.debug("Selecting fingerprint configuration...")
            # TODO: Replace with a more sophisticated generator or larger pool
            fingerprint_profiles = [
                { # Profile 1: Win10 Chrome Latest
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    "viewport": {"width": 1920, "height": 1080},
                    "locale": "en-US",
                    "timezone_id": "America/New_York",
                    "extra_http_headers": {"Accept-Language": "en-US,en;q=0.9"}
                },
                { # Profile 2: Win11 Edge Latest
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
                    "viewport": {"width": 1600, "height": 900},
                    "locale": "en-GB",
                    "timezone_id": "Europe/London",
                     "extra_http_headers": {"Accept-Language": "en-GB,en;q=0.9"}
                },
                { # Profile 3: MacOS Safari Latest (Approx)
                     "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
                     "viewport": {"width": 1440, "height": 900},
                     "locale": "en-US",
                     "timezone_id": "America/Los_Angeles",
                     "extra_http_headers": {"Accept-Language": "en-US,en;q=0.9"}
                }
            ]
            fingerprint = random.choice(fingerprint_profiles)
            self.logger.info(f"Using selected fingerprint: User-Agent={fingerprint.get('user_agent')}, Viewport={fingerprint.get('viewport')}")
            # --- End Fingerprint Configuration ---

            # Prepare context options, including proxy if provided
            context_options = {
                # Proxy is handled by browser launch options, but keep it here if needed for context-specific proxy settings (unlikely with Playwright's launch options)
                # "proxy": proxy, # Use the 'proxy' dict passed into the method
                # --- Apply Fingerprint ---
                "user_agent": fingerprint.get("user_agent"),
                "viewport": fingerprint.get("viewport"),
                "locale": fingerprint.get("locale"),
                "timezone_id": fingerprint.get("timezone_id"),
                "extra_http_headers": fingerprint.get("extra_http_headers"),
                # Increase default navigation timeout
                "navigation_timeout": self.config.get("UGC_LOGIN_TIMEOUT", 60000),
                # TODO: Apply other fingerprint options via launch arguments or JS injection if needed
                # Other potential context options for fingerprinting:
                # device_scale_factor, has_touch, is_mobile, geolocation, permissions, etc.
                # --- End Apply Fingerprint ---
            }
            # Remove None values from options to avoid passing None to Playwright
            context_options = {k: v for k, v in context_options.items() if v is not None}

            # Create context using the prepared options
            context = await browser.new_context(**context_options)

            # Set default timeout for actions within the context
            context.set_default_timeout(self.config.get("UGC_ACTION_TIMEOUT", 30000))

            # --- Behavioral Simulation Hook ---
            self.logger.debug("Attempting to inject behavioral simulation script...")
            behavior_script_path = "scripts/behavior_simulation.js" # Define path
            try:
                # For now, assume it exists and just add the call (will fail if file missing)
                await context.add_init_script(path=behavior_script_path)
                self.logger.info(f"Attempted injection of behavioral simulation script: {behavior_script_path}")
                # Note: Playwright handles file reading internally for add_init_script(path=...)

            except Exception as script_err:
                 # Playwright might raise an error if the path is invalid/unreadable
                 self.logger.error(f"Failed to inject behavioral simulation script from {behavior_script_path}: {script_err}")
                 # Decide if this is critical - maybe context is still usable?
            # --- End Behavioral Simulation Hook ---

            page = await context.new_page()
            await self.log_operation('info', f"Browser context created successfully with proxy {proxy['server'] if proxy else 'None'} and fingerprint options.")
            return browser, context, page # Return all three on success

        except PlaywrightTimeoutError as pte:
            await self.log_operation('error', f"Playwright timeout during browser launch/setup: {pte}")
            if context: await context.close()
            if browser: await browser.close()
            # Don't stop playwright_instance here, might be needed by other concurrent tasks
            return None, None, None
        except Exception as e:
            await self.log_operation('error', f"Failed to launch browser/context: {type(e).__name__} - {e}")
            logger.exception("Browser launch failed") # Log full traceback
            # Ensure resources are cleaned up on failure
            if context:
                try: await context.close()
                except Exception: pass
            if browser:
                try: await browser.close()
                except Exception: pass
            # Don't stop playwright_instance here
            return None, None, None
        # Note: Playwright instance (`playwright_instance`) is not stopped here.
        # It's generally recommended to stop it globally when the application exits.

    async def _execute_signup_steps(self, page: Page, steps: List[Dict], strategy_id: Optional[int], service_name: str, context_data: Dict) -> Dict:
        """
        Executes a given list of signup steps on the page.
        Handles typing dynamic data, clicks, waits, and verification triggers.
        Returns dict with success status, error info, and potentially verification details.
        """
        await self.log_operation('debug', f"Executing {len(steps)} signup steps for strategy {strategy_id or 'inferred'} on {service_name}")
        result = {'success': False, 'error': None, 'step_index': -1, 'verification_needed': None, 'verification_selector': None}
        default_action_timeout = self.config.get("UGC_ACTION_TIMEOUT", 30000) # ms

        for i, step in enumerate(steps):
            action = step.get('action')
            selector = step.get('selector')
            value = step.get('value')
            field = step.get('field')
            url = step.get('url')
            timeout = step.get('timeout', default_action_timeout) # Use step-specific or default timeout
            verification_type = step.get('verification_type')
            verification_selector = step.get('verification_selector')

            result['step_index'] = i # Track current step index for error reporting

            try:
                await self.log_operation('debug', f"Step {i}: Action='{action}', Selector='{selector}', Field='{field}', Value='{value}', URL='{url}'")

                # --- Handle Verification Trigger ---
                if verification_type and verification_selector:
                    await self.log_operation('info', f"Verification step detected at index {i}. Type: {verification_type}, Selector: {verification_selector}")
                    result['verification_needed'] = verification_type
                    result['verification_selector'] = verification_selector
                    # Attempt verification immediately using the handler
                    verification_success = await self._handle_verification(page, verification_type, verification_selector, context_data.get('email'), context_data.get('phone_number'))
                    if not verification_success:
                        result['success'] = False
                        result['error'] = f"{verification_type.capitalize()} verification failed."
                        await self.log_operation('error', f"Step {i} failed: {result['error']}")
                        return result # Stop execution on verification failure
                    else:
                        await self.log_operation('info', f"Verification successful for step {i}. Continuing execution...")
                        # Continue to the next step after successful verification
                        continue

                # --- Handle Actions ---
                if action == 'navigate':
                    if not url: raise ValueError("Navigate action requires a 'url'.")
                    await page.goto(url, wait_until='domcontentloaded', timeout=timeout) # Use configured timeout
                    await page.wait_for_timeout(random.uniform(500, 1500)) # Small random wait after nav
                    # --- Behavioral Simulation Placeholder (Post-Navigate) ---
                    # Add human-like delays or actions after navigation.
                    # Examples:
                    # - await page.wait_for_timeout(random.uniform(1000, 3000)) # Longer pause after load
                    # - await page.mouse.wheel(0, random.randint(100, 500)) # Simulate scrolling down slightly
                    # --- End Behavioral Simulation Placeholder ---

                elif action == 'type':
                    if not selector: raise ValueError("Type action requires a 'selector'.")
                    if not field and not value: raise ValueError("Type action requires 'field' or 'value'.")

                    element = page.locator(selector)
                    await element.wait_for(state='visible', timeout=timeout) # Wait for element

                    text_to_type = ""
                    if field:
                        if field not in context_data: raise ValueError(f"Field '{field}' not found in context_data.")
                        text_to_type = context_data[field]
                    elif value:
                        text_to_type = value

                    await element.fill(text_to_type) # Use fill for input fields
                    await page.wait_for_timeout(random.uniform(100, 300)) # Small random wait after typing
                    # --- Behavioral Simulation Placeholder (Post-Type) ---
                    # Add human-like delays or mouse movements after typing.
                    # Examples:
                    # - await page.wait_for_timeout(random.uniform(200, 800)) # Pause after typing
                    # - await page.mouse.move(random.randint(10, 50), random.randint(10, 50)) # Slight random mouse move
                    # --- End Behavioral Simulation Placeholder ---

                elif action == 'click':
                    if not selector: raise ValueError("Click action requires a 'selector'.")
                    element = page.locator(selector)
                    await element.wait_for(state='visible', timeout=timeout)
                    # Consider different click methods if needed (force, modifiers)
                    # --- Behavioral Simulation Placeholder (Pre-Click) ---
                    # Simulate mouse movement towards the element before clicking.
                    # Examples:
                    # - target_box = await element.bounding_box()
                    # - if target_box:
                    # -     await page.mouse.move(target_box['x'] + target_box['width'] / 2 + random.uniform(-5, 5),
                    # -                           target_box['y'] + target_box['height'] / 2 + random.uniform(-5, 5),
                    # -                           steps=random.randint(5, 15)) # Simulate movement steps
                    # -     await page.wait_for_timeout(random.uniform(50, 250)) # Short pause before click
                    # --- End Behavioral Simulation Placeholder ---
                    await element.click(timeout=timeout)
                    await page.wait_for_timeout(random.uniform(300, 1000)) # Small random wait after click
                    # --- Behavioral Simulation Placeholder (Post-Click) ---
                    # Add delays or other actions after clicking.
                    # Examples:
                    # - await page.wait_for_timeout(random.uniform(500, 2000)) # Pause after click action completes
                    # - await page.evaluate(f"window.scrollBy(0, {random.randint(-100, 100)})") # Simulate minor scroll adjustment
                    # --- End Behavioral Simulation Placeholder ---

                elif action == 'wait': # Added wait action
                    if not selector: raise ValueError("Wait action requires a 'selector'.")
                    await page.wait_for_selector(selector, state='visible', timeout=timeout)
                    await self.log_operation('debug', f"Successfully waited for selector '{selector}'.")
                    # --- Behavioral Simulation Placeholder (Post-Wait) ---
                    # Add delays or actions after waiting for an element.
                    # Examples:
                    # - await page.wait_for_timeout(random.uniform(300, 1200)) # Pause after element appears
                    # - # Simulate reading/scanning the area
                    # - await page.mouse.move(random.randint(100, 300), random.randint(100, 300), steps=random.randint(3, 8))
                    # --- End Behavioral Simulation Placeholder ---

                else:
                    raise ValueError(f"Unknown action type: '{action}'")

                # Optional: Add screenshot capture here for debugging if needed
                # await page.screenshot(path=f"debug_step_{i}.png")

            except PlaywrightTimeoutError as pte:
                result['success'] = False
                result['error'] = f"TimeoutError on step {i} ({action}): {pte}"
                await self.log_operation('error', f"Step {i} failed: {result['error']}")
                # Capture page state on error?
                # result['page_url'] = page.url
                # result['html_snippet'] = await page.content() # Careful, can be large
                return result # Stop execution on timeout
            except ValueError as ve: # Catch config errors in steps
                 result['success'] = False
                 result['error'] = f"ConfigurationError on step {i} ({action}): {ve}"
                 await self.log_operation('error', f"Step {i} failed: {result['error']}")
                 return result
            except Exception as e:
                result['success'] = False
                result['error'] = f"Unexpected Error on step {i} ({action}): {type(e).__name__} - {e}"
                await self.log_operation('error', f"Step {i} failed: {result['error']}")
                logger.exception(f"Unexpected error during step {i} execution") # Log traceback
                # Capture page state on error?
                # result['page_url'] = page.url
                # result['html_snippet'] = await page.content()
                return result # Stop execution on unexpected error

        # If loop completes without errors
        result['success'] = True
        await self.log_operation('info', f"Successfully executed all {len(steps)} steps for strategy {strategy_id or 'inferred'}.")
        return result

    async def _handle_verification(self, page: Page, verification_type: str, verification_selector: str, email_alias: Optional[str], phone_number: Optional[str]) -> bool:
        """Handles email or SMS verification using _VerificationHandler."""
        await self.log_operation('info', f"Handling {verification_type} verification for {email_alias or phone_number}...")
        code = None
        retrieval_success = False

        # --- Retrieve Code ---
        try:
            if verification_type == 'email' and email_alias:
                await self.log_operation('debug', f"Attempting IMAP code retrieval for {email_alias}...")
                code = await self._verification_handler.get_verification_code_from_imap(email_alias)
                retrieval_success = bool(code)
            elif verification_type == 'phone' and phone_number:
                await self.log_operation('debug', f"Attempting Twilio SMS code retrieval for {phone_number}...")
                # TODO: Ensure phone number acquisition logic is in place before calling this
                # If using temporary numbers, acquire one here or pass it in context_data
                code = await self._verification_handler.get_sms_code_from_twilio(phone_number)
                retrieval_success = bool(code)
            else:
                await self.log_operation('error', f"Invalid verification type ({verification_type}) or missing data (email/phone).")
                return False # Cannot proceed

            if retrieval_success:
                await self.log_operation('info', f"Successfully retrieved {verification_type} code: {code}")
            else:
                await self.log_operation('error', f"Failed to retrieve {verification_type} code for {email_alias or phone_number}.")
                return False # Stop if code retrieval failed

        except Exception as retrieval_err:
            # Catch errors from the verification handler itself
            await self.log_operation('error', f"Error during {verification_type} code retrieval: {retrieval_err}")
            logger.error(f"Error retrieving {verification_type} code: {retrieval_err}", exc_info=True)
            return False

        # --- Fill Code into Page ---
        if code:
            try:
                await self.log_operation('debug', f"Attempting to fill code '{code}' into selector '{verification_selector}'.")
                element = page.locator(verification_selector)
                # Wait for the element to be visible before trying to fill
                await element.wait_for(state='visible', timeout=self.config.get("UGC_ACTION_TIMEOUT", 30000))
                await element.fill(code)
                await page.wait_for_timeout(random.uniform(500, 1000)) # Wait briefly after filling
                await self.log_operation('info', f"Successfully filled verification code '{code}' into selector '{verification_selector}'.")
                # Optional: Add a click action here if needed (e.g., click 'Verify' button specified in strategy)
                # verify_button_selector = step.get('verify_button_selector') # Assuming step context is available or passed
                # if verify_button_selector:
                #     await self.log_operation('debug', f"Clicking verification button: {verify_button_selector}")
                #     await page.locator(verify_button_selector).click()
                #     await page.wait_for_timeout(random.uniform(500, 1500)) # Wait after click
                return True
            except PlaywrightTimeoutError:
                await self.log_operation('error', f"Timeout waiting for or filling verification code into selector '{verification_selector}'.")
                return False
            except Exception as e:
                await self.log_operation('error', f"Error filling verification code into selector '{verification_selector}': {e}")
                logger.error(f"Error filling verification code: {e}", exc_info=True)
                return False
        else:
             # Should have returned earlier if retrieval failed, but double-check
             await self.log_operation('error', f"Code retrieval failed, cannot fill verification code.")
             return False


    async def _extract_api_key_after_signup(self, page: Page, service_name: str) -> Optional[str]:
        """Navigates dashboard/settings to find and extract API key using LLM analysis."""
        await self.log_operation('info', f"Attempting API key extraction for {service_name}...")
        api_key = None
        # --- Placeholder ---
        await asyncio.sleep(0.2) # Simulate navigation/analysis time
        if random.random() < 0.6: # Simulate 60% success rate for extraction
            api_key = f"sk-simulated-{fake.sha256()[:40]}"
            await self.log_operation('debug', f"Simulated API key extraction success: {api_key[:15]}...")
        else:
            await self.log_operation('warning', "Simulated API key extraction failure.")
        # --- End Placeholder ---
        return api_key

    async def _store_account_details(self, service_name: str, email: str, password: str, api_key: Optional[str], other_details: Optional[Dict] = None) -> bool:
        """Stores the newly created account details securely using account_manager."""
        await self.log_operation('info', f"Storing account details for {service_name} ({email}). API Key found: {'Yes' if api_key else 'No'}")

        # Import here to avoid circular dependency issues at module level if account_manager imports agents
        try:
            from utils.account_manager import store_credentials
        except ImportError:
            await self.log_operation('critical', "Failed to import utils.account_manager.store_credentials. Cannot store account.")
            logger.critical("Failed to import utils.account_manager.store_credentials.")
            return False

        try:
            secrets_to_store = {'password': password}
            if api_key:
                secrets_to_store['api_key'] = api_key
            if other_details:
                secrets_to_store.update(other_details)

            # Use the session_maker provided during initialization
            async with self.session_maker() as session:
                await store_credentials(
                    session=session,
                    service=service_name,
                    email=email,
                    secrets=secrets_to_store,
                    is_available=True # Mark as available initially
                )
                await session.commit() # Commit the transaction

            await self.log_operation('info', f"Successfully stored credentials for {service_name} ({email}) in database.")
            return True
        except Exception as e:
            await self.log_operation('error', f"Failed to store credentials for {service_name} ({email}): {e}")
            logger.error(f"Failed to store credentials for {service_name} ({email}): {e}", exc_info=True)
            # Attempt rollback if commit failed? Session context manager might handle this.
            return False

    async def _refine_and_retry_strategy(self, service_name: str, failed_strategy_id: int, failure_context: dict, task: Dict) -> bool:
        """Attempts to refine a strategy using _StrategyManager and requeue the task if successful."""
        await self.log_operation('info', f"Attempting refinement via StrategyManager for failed strategy {failed_strategy_id} ({service_name}).")

        # Call the StrategyManager's refine method
        new_strategy_id = await self._strategy_manager.refine_strategy(service_name, failed_strategy_id, failure_context)

        if new_strategy_id:
            await self.log_operation('info', f"Strategy refined by Manager (New ID: {new_strategy_id}). Requeuing task for {service_name}.")
            # Prepare the task for requeueing
            task['retries'] = task.get('retries', 0) + 1 # Increment retry count
            task['priority'] = task.get('priority', 5) + 1 # Adjust priority slightly
            # Add the failed strategy ID to the exclusion list for the next attempt
            task['excluded_strategy_ids'] = task.get('excluded_strategy_ids', []) + [failed_strategy_id]

            max_retries = self.config.get("ACCOUNT_CREATION_MAX_RETRIES")
            if task['retries'] <= max_retries:
                 await self.task_queue.put(task)
                 await self.log_operation('debug', f"Task for {service_name} requeued (Retry {task['retries']}/{max_retries}).")
                 return True # Indicate task was successfully requeued
            else:
                 await self.log_operation('warning', f"Max retries ({max_retries}) reached for {service_name}. Task not requeued after refinement.")
                 return False # Max retries exceeded
        else:
            # Refinement failed or was not possible according to StrategyManager
            await self.log_operation('warning', f"Strategy refinement failed or not possible for strategy {failed_strategy_id} ({service_name}).")
            return False # Indicate refinement/requeue failed

    # --- End Account Creation Helpers ---


    # --- Core Logic Implementation Methods (Private, implementations in next part) ---
    # These methods will contain the detailed logic migrated from the original script's large methods.

    # async def _execute_signup_steps(self, page: Page, steps: List[Dict], strategy_id: Optional[int], email_alias: str, password: str, name: str) -> Dict: ...
    # async def _check_and_adjust_daily_cap(self, service: str) -> Optional[int]: ...
    # async def _extract_api_key(self, page: Page, service: str) -> Optional[str]: ...
    # async def _login_to_service(self, service_name: str, page: Page, credentials: dict) -> bool: ...
    # async def _generate_avatar_video(self, service_name: str, page: Page, script: str, avatar_details: dict) -> Optional[str]: ...
    # async def _edit_video(self, service_name: str, page: Page, source_video_path: str, assets: dict) -> Optional[str]: ...
    # async def _download_asset(self, page: Page, download_trigger_selector: str, target_dir: str, timeout_ms: int) -> Optional[str]: ...
    # async def _setup_network_monitoring(self, page: Page, api_calls_list: List): ...
    # async def _get_today_account_count(self, service: str) -> int: ...
    # async def _store_account_wrapper(self, service: str, email: str, secrets: Dict) -> Optional[Any]: ...
    # async def _check_credit_status(self, service, account_sid=None, auth_token=None): ... # Needs careful placement/design
    # async def _store_credit_status(self, service, credits): ... # Needs careful placement/design
    # async def _store_gemini_results(self, type, data): ... # Needs careful placement/design


# --- End of BrowsingAgent Class Definition (Structure) ---

# Example Usage (Illustrative - Keep commented out)
# async def main():
#     # ... (Setup DB, Config, Mocks as before) ...
#     # Initialize Agent
#     agent = BrowsingAgent(...)
#     # Add tasks
#     await agent.task_queue.put(...)
#     # Run Agent
#     await agent.run()

# if __name__ == "__main__":
#     # ... (Setup logging, env vars) ...
#     # asyncio.run(main())
#     pass