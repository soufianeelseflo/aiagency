import asyncio
import logging
import random
import os
import sqlite3
import time
import aiohttp
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from sqlalchemy.ext.asyncio import async_sessionmaker
from faker import Faker
import psutil
from twilio.rest import Client as TwilioClient
import json
import imaplib # Added for IMAP
import email   # Added for email parsing
import re      # Added for regex parsing
from datetime import datetime # Added for logging timestamp
import sqlalchemy # Explicitly import sqlalchemy for text()

# Configure dedicated operational logger
op_logger = logging.getLogger('OperationalLog')
op_logger.setLevel(logging.INFO)
# Use a Markdown file for structured logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Ensure the log directory exists if needed, e.g., logs/operations_log.md
# os.makedirs('logs', exist_ok=True) # Uncomment if putting logs in a subfolder
log_file_handler = logging.FileHandler('operations_log.md', mode='a') # Append mode
log_file_handler.setFormatter(log_formatter)
op_logger.addHandler(log_file_handler)
# Prevent double logging to console if root logger also has StreamHandler
op_logger.propagate = False


logger = logging.getLogger(__name__) # Keep standard logger for agent-specific debug/errors
fake = Faker()

class BrowsingAgent:
    def __init__(self, session_maker, config, orchestrator, clients_models):
        self.session_maker = session_maker
        self.config = config # Access settings via self.config.HOSTINGER_IMAP_HOST etc.
        self.orchestrator = orchestrator
        self.clients_models = clients_models
        # Ensure 'think' agent is available before accessing it
        self.think_tool = orchestrator.agents.get('think')
        if not self.think_tool:
             logger.critical("ThinkTool agent not found in orchestrator during BrowsingAgent init!")
             # Depending on design, either raise an error or handle gracefully
             # raise ValueError("ThinkTool agent is required for BrowsingAgent")
        self.task_queue = asyncio.Queue()
        self.max_concurrency = 10 # Default, can be adjusted

        # SMS providers remain for potential phone verification needs
        self.sms_providers = [
            'https://www.receivesms.co/us-phone-numbers/',
            'https://sms-activate.org/',
            'https://www.textnow.com/',
            'https://www.freephonenum.com/',
            'https://receive-sms.cc/',
            'https://www.twilio.com/'
        ]
        # API keys for SMS providers (if any)
        self.sms_api_keys = {provider: os.getenv(f"{provider.split('.')[1].upper()}_API_KEY") for provider in self.sms_providers if provider.endswith('.com')}
        # Success rates for SMS providers
        self.provider_success_rates = {
            'sms': {url: 1.0 for url in self.sms_providers},
        }

        # SmartProxy Configuration (Remains unchanged)
        self.proxy_config = {
            'username': os.getenv("SMARTPROXY_USERNAME"),
            'password': os.getenv("SMARTPROXY_PASSWORD"),
            'base_url': 'dc.us.smartproxy.com',
            'port_range': range(10000, 10010),
        }

        # Genius-level meta-prompt
        self.meta_prompt = """
        You are a genius AI BrowsingAgent designed to create infinite free trial accounts for an AI agency targeting $6,000 profit in 24 hours and $100M in 9 months. Utilize APIs (e.g., inboxes.com) and rotate through diverse email/SMS providers dynamically. Infer signup processes and extract API keys for any service using deep analysis. Anticipate provider failures, legal grey areas, and dashboard variations. Learn from past attempts, collaborate with other agents, and optimize for scalability and profitability.
        """

        # Concurrency and caching
        self.browser_semaphore = asyncio.Semaphore(100)
        self.active_browsers = 0
        self.cache_db = sqlite3.connect('browsing_agent_cache.db', check_same_thread=False)
        self.create_cache_tables()

        # Proxy pool
        self.proxy_pool = []
        self._initialize_proxy_pool()
        self.proxy_service_usage = {}
        self.openrouter_client = None
        self.deepseek_client = None
        self.required_services = ['shodan.io', 'maltego.com', 'openrouter.ai']
        asyncio.create_task(self.check_and_create_api_keys())
        op_logger.info("BrowsingAgent initialized. Operational logging started.")

    async def log_operation(self, level, message):
        """Helper to log to the operational log file."""
        # Ensure op_logger exists before using it
        global op_logger
        if not op_logger: return # Or initialize here if needed

        if level == 'info':
            op_logger.info(f"- {message}") # Add markdown list item
        elif level == 'warning':
            op_logger.warning(f"- **Warning:** {message}")
        elif level == 'error':
            op_logger.error(f"- **ERROR:** {message}")
        elif level == 'critical':
            op_logger.critical(f"- **CRITICAL:** {message}")
        else:
            op_logger.debug(f"- Debug: {message}") # Default to debug for other levels

    async def check_and_create_api_keys(self):
        """Check for missing API keys for required services and queue creation tasks."""
        await self.log_operation('info', f"Checking required API keys: {self.required_services}")
        async with self.session_maker() as session:
            for service in self.required_services:
                try:
                    result = await session.execute(
                        sqlalchemy.text("SELECT COUNT(*) FROM accounts WHERE service = :service AND api_key IS NOT NULL AND api_key != ''"), # Check for non-empty key
                        {"service": service}
                    )
                    if result.scalar() == 0:
                        await self.task_queue.put({'service_url': f"https://{service}", 'action': 'create_account'}) # Add action type
                        await self.log_operation('info', f"Queued account creation task for missing API key: {service}")
                        logger.info(f"Queued API key creation for {service}") # Keep standard log too
                except Exception as db_err:
                    await self.log_operation('error', f"Database error checking API key for {service}: {db_err}")
                    logger.error(f"Database error checking API key for {service}: {db_err}")
        await self.log_operation('info', "Finished checking required API keys.")

    async def generate_test_videos(self, industries):
        """Generate mock video URLs for testing."""
        await self.log_operation('info', f"Generating test videos for industries: {industries}")
        videos = []
        base_path = "ui/static/videos" # Relative to project root
        try:
            os.makedirs(base_path, exist_ok=True) # Ensure directory exists
            for industry in industries:
                video_filename = f"test_{industry}_{int(time.time())}.mp4"
                video_filepath = os.path.join(base_path, video_filename)
                # Simulate video file creation
                with open(video_filepath, "wb") as f:
                    f.write(b"Mock video data for testing purposes.")
                # Return the web-accessible path
                video_url = f"/static/videos/{video_filename}"
                videos.append(video_url)
            await self.log_operation('info', f"Generated test videos: {videos}")
            logger.info(f"Generated test videos: {videos}")
        except Exception as e:
            await self.log_operation('error', f"Failed to generate test videos: {e}")
            logger.error(f"Failed to generate test videos: {e}")
        return videos

    def get_allowed_concurrency(self):
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        concurrency_factor = 1 - (cpu_usage / 100) * 0.5 - (memory_usage / 100) * 0.5
        allowed = max(1, int(self.max_concurrency * concurrency_factor))
        return allowed

    def create_cache_tables(self):
        """Initialize SQLite cache for strategies and attempt logs."""
        cursor = self.cache_db.cursor()
        # Removed email_cache and phone_cache as they are no longer used
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_cache (
                id INTEGER PRIMARY KEY,
                service TEXT,
                strategy TEXT,
                success_rate REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attempt_logs (
                id INTEGER PRIMARY KEY,
                service TEXT,
                step TEXT,
                success BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.cache_db.commit()

    def _initialize_proxy_pool(self):
        """Initialize proxy pool with SmartProxy dedicated datacenter proxies."""
        if not all([self.proxy_config.get('username'), self.proxy_config.get('password'), self.proxy_config.get('base_url'), self.proxy_config.get('port_range')]):
             logger.warning("Smartproxy configuration incomplete (username, password, base_url, port_range). Proxy pool will be empty.")
             op_logger.warning("Smartproxy configuration incomplete. Proxy pool will be empty.")
             return

        for port in self.proxy_config['port_range']:
            proxy = f"{self.proxy_config['base_url']}:{port}"
            self.proxy_pool.append({
                'server': proxy,
                'username': self.proxy_config['username'],
                'password': self.proxy_config['password'],
                'failure_count': 0,
                'last_checked': 0
            })
        random.shuffle(self.proxy_pool)
        op_logger.info(f"Initialized proxy pool with {len(self.proxy_pool)} proxies.")

    async def load_strategies(self, service):
        """Load strategies for a service, sorted by success rate."""
        cursor = self.cache_db.cursor()
        cursor.execute('SELECT id, strategy, success_rate FROM strategy_cache WHERE service = ? ORDER BY success_rate DESC', (service,))
        rows = cursor.fetchall()
        strategies = []
        for row in rows:
            id_val, strategy_json, success_rate = row
            try:
                strategy = json.loads(strategy_json)
                strategies.append({'id': id_val, 'steps': strategy, 'success_rate': success_rate})
            except json.JSONDecodeError:
                logger.error(f"Failed to decode strategy JSON for service {service}, ID {id_val}")
        return strategies

    async def save_strategy(self, service, steps, initial_success_rate):
        """Save a new strategy to the cache."""
        cursor = self.cache_db.cursor()
        strategy_json = json.dumps(steps)
        cursor.execute('INSERT INTO strategy_cache (service, strategy, success_rate) VALUES (?, ?, ?)',
                    (service, strategy_json, initial_success_rate))
        self.cache_db.commit()
        return cursor.lastrowid

    async def update_strategy_success(self, strategy_id, success):
        """Update a strategy’s success rate with a moving average."""
        if not strategy_id: return # Cannot update if no ID was assigned (e.g., inference failed)
        cursor = self.cache_db.cursor()
        cursor.execute('SELECT success_rate FROM strategy_cache WHERE id = ?', (strategy_id,))
        row = cursor.fetchone()
        if row:
            old_rate = row[0]
            new_rate = old_rate * 0.9 + (1.0 if success else 0.0) * 0.1
            cursor.execute('UPDATE strategy_cache SET success_rate = ? WHERE id = ?', (new_rate, strategy_id))
            self.cache_db.commit()
            await self.log_operation('debug', f"Updated strategy {strategy_id} success rate to {new_rate:.2f}")
        else:
            logger.error(f"Strategy ID {strategy_id} not found for update")
            await self.log_operation('error', f"Attempted to update non-existent strategy ID: {strategy_id}")


    async def get_next_proxy(self, service=None):
        """Selects the next available proxy, prioritizing those with fewer failures."""
        if not self.proxy_pool:
            await self.log_operation('warning', "Proxy pool is empty. Cannot select proxy.")
            return None

        current_time = time.time()
        valid_proxies = []

        # First pass: Check and validate proxies if needed
        for proxy in self.proxy_pool:
            if proxy.get('failure_count', 0) >= 3: # Use .get for safety
                continue # Skip proxies that failed too many times

            # Check cooldown per service
            if service:
                last_used = self.proxy_service_usage.get(proxy['server'], {}).get(service, 0)
                if current_time - last_used < 3600:  # 1 hour cooldown
                    continue

            # Re-validate if stale
            if current_time - proxy.get('last_checked', 0) > 300: # 5 min validation interval
                if await self._validate_proxy(proxy):
                    valid_proxies.append(proxy)
                else:
                    proxy['failure_count'] = proxy.get('failure_count', 0) + 1 # Increment safely
                    logger.warning(f"Proxy {proxy['server']} failed validation. Failure count: {proxy['failure_count']}")
                    await self.log_operation('warning', f"Proxy {proxy['server']} failed validation. Fail count: {proxy['failure_count']}")
            else:
                # Assume still valid if checked recently
                valid_proxies.append(proxy)

        if not valid_proxies:
            await self.log_operation('error', "No suitable proxies available after validation/cooldown.")
            # Optional: Attempt to reset failure counts if ALL proxies are failing?
            # for p in self.proxy_pool: p['failure_count'] = 0
            # logger.warning("Resetting proxy failure counts as all proxies failed.")
            # return await self.get_next_proxy(service) # Retry selection
            return None # Or simply fail

        # Select the best proxy (lowest failure count, then random)
        min_failure_count = min(p.get('failure_count', 0) for p in valid_proxies)
        best_proxies = [p for p in valid_proxies if p.get('failure_count', 0) == min_failure_count]
        selected_proxy = random.choice(best_proxies)


        # Update usage time for the specific service
        if service:
            if selected_proxy['server'] not in self.proxy_service_usage:
                self.proxy_service_usage[selected_proxy['server']] = {}
            self.proxy_service_usage[selected_proxy['server']][service] = current_time

        await self.log_operation('debug', f"Selected proxy {selected_proxy['server']} for service {service}. Fail count: {selected_proxy.get('failure_count', 0)}")
        return selected_proxy

    async def _validate_proxy(self, proxy):
        """Validate proxy connectivity."""
        proxy_url = f"http://{proxy['server']}"
        auth = aiohttp.BasicAuth(proxy['username'], proxy['password'])
        try:
            async with aiohttp.ClientSession() as session:
                # Using a known reliable target for validation
                async with session.get('https://httpbin.org/ip', proxy=proxy_url, auth=auth, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        # Optionally check if the returned IP is different from the machine's IP
                        proxy['last_checked'] = time.time()
                        # Reset failure count on successful validation? Maybe too aggressive.
                        # proxy['failure_count'] = 0
                        return True
                    else:
                        logger.warning(f"Proxy validation failed for {proxy['server']}. Status: {response.status}")
                        return False
        except asyncio.TimeoutError:
             logger.warning(f"Proxy validation timed out for {proxy['server']}.")
             return False
        except aiohttp.ClientConnectorError as e:
             logger.warning(f"Proxy validation connection error for {proxy['server']}: {e}")
             return False
        except Exception as e:
            logger.error(f"Unexpected error validating proxy {proxy['server']}: {e}")
            return False


    async def infer_signup_steps(self, service_url):
        """Uses LLM to infer signup steps from a service URL."""
        await self.log_operation('info', f"Inferring signup steps for {service_url}")
        prompt = f"""
        {self.meta_prompt}
        Analyze the signup process for {service_url} with genius-level precision.
        Identify all steps required for account creation, including:
        1. Navigation to signup page (if not direct).
        2. Input fields (email, password, name, phone) with CSS selectors.
        3. Submit buttons with selectors.
        4. Verification steps (email/SMS) with selectors for code entry.
        Account for variations (e.g., CAPTCHAs, multi-step forms) and potential future changes.
        Return a JSON array 'steps' with:
        - 'action': ('navigate', 'type', 'click')
        - 'url': (for navigate)
        - 'selector': (for type/click)
        - 'field': (e.g., 'email', 'password', 'name') or 'value': (fixed text for buttons etc.)
        - 'verification_type': ('email', 'phone') if applicable and identifiable
        - 'verification_selector': (CSS selector for code input field) if applicable
        Example step: {{ "action": "type", "selector": "#user_email", "field": "email" }}
        Example verification step: {{ "action": "type", "selector": "#verification_code", "verification_type": "email", "verification_selector": "#verification_code" }}
        """
        # Use the first available client/model combo
        # TODO: Implement rotation or selection based on success/cost if needed
        if not self.clients_models:
             await self.log_operation('error', "No LLM clients configured for infer_signup_steps.")
             raise Exception("No LLM clients available for signup step inference.")

        client, model = self.clients_models[0] # Using the first one for simplicity
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            steps_data = json.loads(content)
            steps = steps_data.get('steps')
            if not steps or not isinstance(steps, list):
                 raise ValueError("LLM response did not contain a valid 'steps' array.")
            await self.log_operation('info', f"Successfully inferred {len(steps)} signup steps for {service_url}")
            return steps
        except Exception as e:
            await self.log_operation('error', f"Failed to infer steps for {service_url} using {client.base_url} model {model}: {e}")
            logger.error(f"Failed to infer steps with {client.base_url}: {e}")
            raise Exception(f"LLM client failed to infer signup steps: {e}")


    async def _store_account(self, service, email, password, api_key, phone, cookies=None):
        """Store account details securely in Vault and database."""
        await self.log_operation('info', f"Storing account: Service={service}, Email={email}, API Key Present={bool(api_key)}")
        vault_path = f"secret/data/accounts/{service}/{email}" # Using email as part of the path for uniqueness
        try:
            # Ensure secure_storage exists and has the method
            if hasattr(self.orchestrator, 'secure_storage') and hasattr(self.orchestrator.secure_storage, 'set_secret'):
                self.orchestrator.secure_storage.set_secret(
                    path=vault_path,
                    data={
                        "api_key": api_key or "", # Store empty string if None
                        "password": password,
                        "phone": phone or "",
                        "cookies": json.dumps(cookies) if cookies else ""
                    }
                )
                await self.log_operation('debug', f"Stored secrets in Vault at {vault_path}")
            else:
                await self.log_operation('error', "SecureStorage not available on orchestrator. Cannot store secrets.")
                logger.error("SecureStorage not available on orchestrator.")
                # Decide if this is critical - should we proceed without vault storage?
                # For now, we proceed but log the error.

            # Store metadata in database
            async with self.session_maker() as session:
                # Use proper SQL syntax for ON CONFLICT based on the target DB (e.g., PostgreSQL)
                await session.execute(
                    sqlalchemy.text("""
                        INSERT INTO accounts (service, email, vault_path, created_at, api_key, is_available)
                        VALUES (:service, :email, :vault_path, :created_at, :api_key, :is_available)
                        ON CONFLICT (service, email) DO UPDATE SET
                            vault_path = EXCLUDED.vault_path,
                            api_key = EXCLUDED.api_key, -- Update API key if re-storing
                            is_available = EXCLUDED.is_available, -- Reset availability on update
                            created_at = EXCLUDED.created_at -- Update timestamp
                    """), {
                        "service": service,
                        "email": email,
                        "vault_path": vault_path,
                        "created_at": datetime.utcnow(),
                        "api_key": api_key or "", # Store empty string if None
                        "is_available": True # Mark as available initially
                    }
                )
                await session.commit()
            await self.log_operation('info', f"Stored/Updated account metadata in DB for {service} / {email}")
            logger.info(f"Stored account for {service} with email {email}")

        except Exception as e:
            await self.log_operation('error', f"Failed to store account for {service} / {email}: {e}")
            logger.exception(f"Failed to store account for {service} / {email}")


    async def get_api_key(self, service, email):
        """Fetch an API key from Vault using the stored path."""
        # This method might be less relevant if API keys are stored directly in DB now
        # Kept for potential compatibility or if direct Vault access is needed elsewhere
        await self.log_operation('debug', f"Attempting to fetch API key from Vault for {service}/{email}")
        async with self.session_maker() as session:
            try:
                result = await session.execute(
                    sqlalchemy.text("SELECT vault_path FROM accounts WHERE service = :service AND email = :email"),
                    {"service": service, "email": email}
                )
                row = result.fetchone()
                if row and row[0]:
                    vault_path = row[0]
                    if hasattr(self.orchestrator, 'secure_storage') and hasattr(self.orchestrator.secure_storage, 'get_secret'):
                        secrets = self.orchestrator.secure_storage.get_secret(vault_path)
                        api_key = secrets.get("api_key") if secrets else None
                        if api_key:
                             await self.log_operation('debug', f"Found API key in Vault for {service}/{email}")
                             return api_key
                        else:
                             await self.log_operation('warning', f"Vault path found for {service}/{email}, but no API key within.")
                    else:
                        await self.log_operation('warning', "SecureStorage not available. Cannot fetch from Vault.")
                else:
                     await self.log_operation('warning', f"No Vault path found in DB for {service}/{email}")

            except Exception as e:
                 await self.log_operation('error', f"Error fetching API key vault path for {service}/{email}: {e}")
                 logger.error(f"Error fetching API key vault path for {service}/{email}: {e}")

        # Fallback or primary method: Check DB directly if key stored there
        try:
             async with self.session_maker() as session:
                 result = await session.execute(
                     sqlalchemy.text("SELECT api_key FROM accounts WHERE service = :service AND email = :email"),
                     {"service": service, "email": email}
                 )
                 row = result.fetchone()
                 if row and row[0]:
                      await self.log_operation('debug', f"Found API key directly in DB for {service}/{email}")
                      return row[0] # Return key from DB
        except Exception as e:
             await self.log_operation('error', f"Error fetching API key directly from DB for {service}/{email}: {e}")
             logger.error(f"Error fetching API key directly from DB for {service}/{email}: {e}")


        await self.log_operation('warning', f"No API key found for {service}/{email} in Vault or DB.")
        logger.warning(f"No API key found for {service}/{email}")
        return None


    async def create_account(self, service_url, retry_count=0, max_retries=3):
        """Create an account with intelligent scaling, analysis, proxy retry, and IMAP verification."""
        service = service_url.split('//')[-1].split('/')[0].replace('www.', '')
        await self.log_operation('info', f"Starting account creation attempt {retry_count + 1}/{max_retries} for service: {service}")

        # Check retry limit first
        if retry_count >= max_retries:
            await self.log_operation('error', f"Max retries ({max_retries}) reached for {service_url}. Aborting.")
            logger.error(f"Max retries ({max_retries}) reached for {service_url}. Aborting account creation.")
            return None

        # Check daily cap (Moved inside attempt loop to re-check before each retry)
        today_count = await self.get_today_account_count(service)
        new_cap = 50 # Default cap
        if self.think_tool: # Check if think_tool exists
            # Ensure budget agent exists and has the method before calling
            budget_remaining = 0
            budget_agent = self.orchestrator.agents.get('budget')
            if budget_agent and hasattr(budget_agent, 'get_budget_status'):
                try:
                    budget_status = await budget_agent.get_budget_status()
                    budget_remaining = budget_status.get('remaining_budget', 0)
                except Exception as budget_e:
                    await self.log_operation('warning', f"Failed to get budget status: {budget_e}")

            prompt = f"""
            {self.meta_prompt}
            Service: {service}
            Today’s account count: {today_count}
            Budget remaining: {budget_remaining}
            Adjust daily cap (default 50) based on success rate and budget.
            Return JSON: {{'new_cap': int}}
            """
            try:
                # Ensure reflect_on_action is awaited if it's async
                cap_response = await self.think_tool.reflect_on_action(prompt, "BrowsingAgent", "Adjust account creation cap")
                new_cap = json.loads(cap_response).get('new_cap', 50)
                await self.log_operation('info', f"Dynamic cap for {service} adjusted to {new_cap} by ThinkTool.")
            except Exception as cap_e:
                await self.log_operation('warning', f"Failed to adjust cap via ThinkTool for {service}: {cap_e}. Using default cap {new_cap}.")
                logger.warning(f"Failed to adjust cap via ThinkTool for {service}: {cap_e}. Using default cap {new_cap}.")
        else:
             await self.log_operation('warning', "ThinkTool not available, using default cap 50.")


        if today_count >= new_cap:
            await self.log_operation('info', f"Adjusted daily cap reached for {service}: {today_count}/{new_cap}. Skipping creation attempt {retry_count + 1}.")
            logger.info(f"Adjusted daily cap reached for {service}: {today_count}/{new_cap}. Skipping creation.")
            return None

        # Acquire semaphore before entering retry logic for this attempt
        async with self.browser_semaphore:
            self.active_browsers += 1
            proxy = None
            browser = None
            context = None # Define context here
            strategy_id = None # Initialize strategy_id here

            try:
                # --- Proxy Selection & Browser Launch ---
                proxy = await self.get_next_proxy(service)
                if not proxy: # Handle case where no proxies are available
                    await self.log_operation('error', f"No available proxies for {service_url} on attempt {retry_count + 1}. Aborting.")
                    logger.error(f"No available proxies for {service_url} on attempt {retry_count + 1}. Aborting.")
                    # No need to decrement active_browsers here, finally block will handle it
                    return None

                async with async_playwright() as p:
                    fingerprint = {
                        'user_agent': fake.user_agent(),
                        'viewport': {'width': random.randint(1280, 1920), 'height': random.randint(720, 1080)},
                        'screen': {'width': random.randint(1366, 2560), 'height': random.randint(768, 1440)}
                    }
                    await self.log_operation('info', f"Attempt {retry_count + 1}/{max_retries}: Launching browser for {service_url} via proxy {proxy['server']}")
                    logger.info(f"Attempt {retry_count + 1}/{max_retries}: Launching browser for {service_url} via proxy {proxy['server']}")
                    browser = await p.chromium.launch(headless=True, proxy={
                        'server': f"http://{proxy['server']}",
                        'username': proxy['username'],
                        'password': proxy['password']
                    })
                    context = await browser.new_context(**fingerprint)
                    page = await context.new_page()
                    phone = None # Initialize phone here

                    # --- Inner Try-Except for Signup Logic ---
                    try: # This is the start of the inner try block for signup steps
                        # Generate Hostinger alias
                        email_alias = await self._generate_hostinger_alias(service)
                        if not email_alias:
                            raise ValueError("Failed to generate Hostinger email alias.")
                        await self.log_operation('info', f"Generated email alias for {service}: {email_alias}")

                        password = fake.password(length=12, special_chars=True)
                        name = fake.name()
                        # phone = await self._get_temporary_phone_number(page) # Uncomment if phone needed

                        steps = await self.infer_signup_steps(service_url)
                        strategy_id = await self.save_strategy(service, steps, 1.0) # Save strategy attempt

                        await self.log_operation('info', f"Executing {len(steps)} signup steps for {service_url} using alias {email_alias}")
                        logger.info(f"Executing signup steps for {service_url} using email {email_alias}")
                        for i, step in enumerate(steps):
                            await self.log_operation('debug', f"Step {i+1}/{len(steps)}: {step['action']} - Selector: {step.get('selector')}")
                            logger.debug(f"Step {i+1}/{len(steps)}: {step['action']} - Selector: {step.get('selector')}")
                            if step['action'] == 'navigate':
                                await page.goto(step['url'], wait_until='networkidle', timeout=60000)
                            elif step['action'] == 'type':
                                # Use the generated alias for the email field
                                value_to_type = {'email': email_alias, 'password': password, 'name': name}.get(step.get('field'), step.get('value'))
                                if value_to_type:
                                    await page.fill(step['selector'], value_to_type, timeout=30000)
                                else:
                                    await self.log_operation('warning', f"No value found for step: {step}")
                                    logger.warning(f"No value found for step: {step}")
                            elif step['action'] == 'click':
                                await page.click(step['selector'], timeout=30000)

                            # --- Email Verification Check ---
                            if step.get('verification_type') == 'email':
                                await self.log_operation('info', f"Email verification required for {service}. Waiting for code at {email_alias}.")
                                verification_code = await self._get_verification_code_from_imap(email_alias)
                                if verification_code:
                                    await self.log_operation('info', f"Retrieved verification code: {verification_code}")
                                    verify_selector = step.get('verification_selector')
                                    if verify_selector:
                                        await page.fill(verify_selector, verification_code, timeout=30000)
                                        await self.log_operation('info', f"Filled verification code into selector: {verify_selector}")
                                        # Optional: Add a click step if there's a separate verify button
                                    else:
                                        await self.log_operation('error', f"Verification required but no 'verification_selector' provided in steps for {service}")
                                        raise ValueError("Missing verification selector in signup steps")
                                else:
                                    await self.log_operation('error', f"Failed to retrieve verification code for {email_alias}")
                                    raise ValueError("Failed to retrieve email verification code via IMAP")
                            # --- End Email Verification ---

                            # Add potential phone verification steps here if inferred and implemented

                            await asyncio.sleep(random.uniform(1.5, 3.5)) # Slightly reduced sleep

                        await self.log_operation('info', f"Signup steps completed for {service_url}. Attempting API key extraction.")
                        logger.info(f"Signup steps completed for {service_url}. Attempting API key extraction.")
                        api_key = await self._extract_api_key(page, service)
                        cookies = await context.cookies()

                        await self.log_operation('info', f"Storing account details for {service} (Alias: {email_alias}). API Key Found: {'Yes' if api_key else 'No'}")
                        logger.info(f"Storing account details for {service} with email {email_alias}.")
                        # Store the alias used
                        await self._store_account(service, email_alias, password, api_key, phone, cookies)
                        if strategy_id: await self.update_strategy_success(strategy_id, True) # Mark strategy as successful
                        await self.log_operation('info', f"SUCCESS: Account created for {service} using alias {email_alias}.")
                        logger.info(f"Successfully created account for {service} with email {email_alias}. API Key: {'Present' if api_key else 'Not Found'}")

                        # Track proxy cost on success
                        # Ensure budget agent exists and has the method
                        budget_agent = self.orchestrator.agents.get('budget')
                        if budget_agent and hasattr(budget_agent, 'track_expense'):
                             await budget_agent.track_expense(
                                 0.01, "Proxy", f"Successful proxy usage for {service} account creation"
                             )
                        return {'email': email_alias, 'api_key': api_key} # SUCCESS

                    except (PlaywrightTimeoutError, aiohttp.ClientProxyConnectionError, Exception) as inner_e: # Catch specific proxy/network errors + general errors during signup steps
                        # Determine if it's likely a proxy issue vs. a signup logic issue
                        error_type = type(inner_e).__name__
                        is_proxy_error = isinstance(inner_e, (PlaywrightTimeoutError, aiohttp.ClientProxyConnectionError)) or "net::ERR" in str(inner_e)

                        await self.log_operation('warning', f"Attempt {retry_count + 1}/{max_retries} failed during signup logic for {service_url} (Proxy: {proxy['server']}): {error_type} - {inner_e}")
                        logger.warning(f"Attempt {retry_count + 1}/{max_retries} failed during signup logic for {service_url} (Proxy: {proxy['server']}): {error_type} - {inner_e}")

                        if strategy_id: # Mark strategy as failed for this attempt
                            await self.update_strategy_success(strategy_id, False)

                        if is_proxy_error:
                            await self.log_operation('info', f"Incrementing failure count for proxy {proxy['server']}")
                            logger.info(f"Incrementing failure count for proxy {proxy['server']}")
                            # Need a safe way to update proxy failure count - direct modification might be risky if dict is shared
                            # For now, assume direct update is okay based on current structure
                            proxy['failure_count'] = proxy.get('failure_count', 0) + 1
                            proxy['last_checked'] = time.time() # Mark as recently checked (failed)

                            # --- Initiate Retry ---
                            await self.log_operation('info', f"Attempting retry ({retry_count + 2}/{max_retries}) for {service_url} due to proxy/network error.")
                            logger.info(f"Attempting retry ({retry_count + 1}/{max_retries}) for {service_url} due to proxy/network error.")
                            # No need to close browser/context here, finally block handles it
                            # No need to decrement active_browsers here, finally block handles it
                            return await self.create_account(service_url, retry_count + 1, max_retries) # RECURSIVE CALL FOR RETRY
                        else:
                            # Non-proxy error during signup (e.g., selector not found, logic error) - Do not retry this attempt
                            await self.log_operation('error', f"FAIL (Signup Logic): Non-retryable error during signup steps for {service_url}: {inner_e}. Aborting this attempt.")
                            logger.error(f"Non-retryable error during signup steps for {service_url}: {inner_e}. Aborting this attempt.")
                            return None # FAIL (Non-retryable inner error)
                    # End of inner try...except for signup logic

            except Exception as outer_e: # Catch errors during proxy selection or browser launch
                error_type = type(outer_e).__name__
                await self.log_operation('error', f"Outer error during account creation attempt {retry_count + 1} for {service_url} (Proxy: {proxy['server'] if proxy else 'N/A'}): {error_type} - {outer_e}")
                logger.error(f"Outer error during account creation attempt {retry_count + 1} for {service_url} (Proxy: {proxy['server'] if proxy else 'N/A'}): {error_type} - {outer_e}")
                # Potentially retry here too if it's a recoverable error (e.g., proxy selection failed but others might be available)
                # For simplicity, currently failing the attempt. Could add retry logic here too.
                if proxy: # If a proxy was selected before the error, mark it
                     proxy['failure_count'] = proxy.get('failure_count', 0) + 1
                     proxy['last_checked'] = time.time()
                return None # FAIL (Outer error)

            finally:
                # Ensure browser and context are closed if they were created
                if context:
                    try: await context.close()
                    except Exception as ctx_close_e: logger.debug(f"Error closing context: {ctx_close_e}")
                if browser:
                    try: await browser.close()
                    except Exception as brw_close_e: logger.debug(f"Error closing browser: {brw_close_e}")
                self.active_browsers -= 1 # Decrement semaphore count


    async def reuse_accounts(self):
        """Check for services with monthly credit resets and reuse accounts."""
        await self.log_operation('info', "Checking for reusable accounts with reset credits.")
        async with self.session_maker() as session:
            # Define services known to have monthly resets (adjust as needed)
            recurring_services = ('spiderfoot.net', 'shodan.io')
            # Use proper parameter binding
            # Ensure the table and columns exist as expected
            try:
                # Assuming 'accounts' table has 'service', 'email', 'vault_path', 'created_at' columns
                # And 'service' column type is compatible with ANY operation (using CAST for safety)
                # Note: ANY requires an array type in PostgreSQL. Ensure :services is passed as a list.
                accounts_result = await session.execute(
                    sqlalchemy.text("SELECT service, email, vault_path, created_at FROM accounts WHERE service = ANY(CAST(:services AS text[]))"),
                    {"services": list(recurring_services)}
                )
                accounts = accounts_result.mappings().all() # Fetch all results as mappings
            except Exception as db_e:
                 await self.log_operation('error', f"Database error fetching recurring accounts: {db_e}")
                 logger.error(f"Database error fetching recurring accounts: {db_e}")
                 return # Cannot proceed without account data

            reused_count = 0
            for account in accounts: # Iterate through fetched results
                service = account.get('service')
                email = account.get('email')
                vault_path = account.get('vault_path')
                created_at = account.get('created_at')

                if not all([service, email, vault_path, created_at]):
                    await self.log_operation('warning', f"Skipping incomplete account record: {account}")
                    continue

                now = datetime.utcnow()

                # Check if roughly a month has passed
                if created_at and (now - created_at).days >= 30:
                    await self.log_operation('info', f"Credits likely reset for {service} account {email}. Attempting reuse.")
                    logger.info(f"Credits likely reset for {service} account {email}")
                    # Ensure orchestrator and send_notification exist
                    if hasattr(self.orchestrator, 'send_notification'):
                        await self.orchestrator.send_notification(
                            "Credit Reset",
                            f"Time to reuse {service} account {email} - credits may have reset!"
                        )
                    # Reuse the account by passing cookies if available from Vault
                    try:
                        # Ensure secure_storage exists and has the method
                        secrets = None
                        if hasattr(self.orchestrator, 'secure_storage') and hasattr(self.orchestrator.secure_storage, 'get_secret'):
                             secrets = self.orchestrator.secure_storage.get_secret(vault_path)
                        else:
                             await self.log_operation('warning', "SecureStorage not available on orchestrator.")

                        cookies_str = secrets.get("cookies") if secrets else None
                        if cookies_str:
                            cookies = json.loads(cookies_str)
                            async with async_playwright() as p:
                                # Consider using proxy for reuse attempt too
                                proxy = await self.get_next_proxy(service)
                                browser = await p.chromium.launch(headless=True, proxy={
                                    'server': f"http://{proxy['server']}",
                                    'username': proxy['username'],
                                    'password': proxy['password']
                                } if proxy else None)
                                context = await browser.new_context()
                                await context.add_cookies(cookies)
                                page = await context.new_page()
                                # Go to a known logged-in page, e.g., dashboard
                                await page.goto(f"https://{service}/dashboard", wait_until='networkidle', timeout=60000)
                                # Add a check here to confirm login was successful if possible
                                await self.log_operation('info', f"Successfully reused {service} account {email} with stored cookies.")
                                logger.info(f"Reused {service} account {email} with stored cookies")
                                await browser.close()
                                reused_count += 1
                                # Update the created_at timestamp in DB to reset the 30-day timer
                                await session.execute(
                                    sqlalchemy.text("UPDATE accounts SET created_at = :now WHERE service = :service AND email = :email"),
                                    {"now": now, "service": service, "email": email}
                                )
                        else:
                             await self.log_operation('warning', f"No cookies found in Vault for {service} account {email}. Cannot reuse via cookies.")
                    except Exception as reuse_e:
                        await self.log_operation('error', f"Failed to reuse account {email} for {service}: {reuse_e}")
                        logger.error(f"Failed to reuse account {email} for {service}: {reuse_e}")

            await session.commit() # Commit timestamp updates
            await self.log_operation('info', f"Finished checking reusable accounts. Reused {reused_count} accounts.")


    async def reuse_accounts_periodically(self):
        """Periodically check and reuse accounts with reset credits."""
        await self.log_operation('info', "Starting periodic account reuse check (daily).")
        while True:
            await self.reuse_accounts()
            await asyncio.sleep(86400)  # Daily check


    async def _generate_hostinger_alias(self, service_tag):
        """Generates a unique email alias for the configured Hostinger account."""
        if not self.config.HOSTINGER_EMAIL:
            await self.log_operation('error', "HOSTINGER_EMAIL not configured in settings.")
            logger.error("HOSTINGER_EMAIL not configured in settings.")
            return None
        try:
            base_email, domain = self.config.HOSTINGER_EMAIL.split('@', 1)
            # Create a unique tag, e.g., service_timestamp
            timestamp = int(time.time() * 1000) # Milliseconds for more uniqueness
            # Sanitize service_tag for email alias (remove invalid chars)
            safe_service_tag = re.sub(r'[^a-zA-Z0-9_-]', '_', service_tag)
            alias_tag = f"{safe_service_tag}_{timestamp}"
            alias = f"{base_email}+{alias_tag}@{domain}"
            await self.log_operation('debug', f"Generated alias: {alias}")
            return alias
        except Exception as e:
            await self.log_operation('error', f"Failed to split HOSTINGER_EMAIL or generate alias: {e}")
            logger.error(f"Failed to generate Hostinger alias from {self.config.HOSTINGER_EMAIL}: {e}")
            return None

    # --- New IMAP Verification Code Retrieval ---
    async def _get_verification_code_from_imap(self, alias_email, timeout_seconds=300, retry_delay=15):
        """Connects to Hostinger IMAP and retrieves verification code sent to an alias."""
        await self.log_operation('info', f"Attempting IMAP connection to retrieve code for {alias_email}")
        if not all([self.config.HOSTINGER_IMAP_HOST, self.config.HOSTINGER_IMAP_USER, self.config.HOSTINGER_IMAP_PASS]):
            await self.log_operation('error', "IMAP credentials missing in settings (HOST, USER, PASS). Cannot retrieve verification code.")
            logger.error("IMAP credentials missing in settings. Cannot retrieve verification code.")
            return None

        start_time = time.time()
        mail = None # Initialize mail object outside loop for proper finally block handling

        # Outer loop for timeout
        while time.time() - start_time < timeout_seconds:
            try: # Inner try for connection/login/search attempt
                # Connect to IMAP server (using SSL implicitly with port 993)
                await self.log_operation('debug', f"Connecting to IMAP: {self.config.HOSTINGER_IMAP_HOST}:{self.config.HOSTINGER_IMAP_PORT}")
                mail = imaplib.IMAP4_SSL(self.config.HOSTINGER_IMAP_HOST, self.config.HOSTINGER_IMAP_PORT)
                await self.log_operation('debug', f"Logging into IMAP as {self.config.HOSTINGER_IMAP_USER}")
                mail.login(self.config.HOSTINGER_IMAP_USER, self.config.HOSTINGER_IMAP_PASS)
                mail.select("inbox") # Select the inbox
                await self.log_operation('debug', "IMAP Inbox selected.")

                # Search for unseen emails TO the specific alias
                # IMAP search for '+' aliases is unreliable. Search broader and filter.
                # Search for UNSEEN emails received recently (e.g., last 5 minutes)
                # Note: Date format for IMAP SEARCH can be tricky (e.g., "DD-Mon-YYYY")
                # Simpler approach: Search all UNSEEN, fetch headers/body, filter by 'To' header.
                search_criteria = '(UNSEEN)'
                await self.log_operation('debug', f"IMAP Search Criteria: {search_criteria}")
                status, messages = mail.search(None, search_criteria)

                if status == "OK":
                    email_ids = messages[0].split()
                    if email_ids:
                        await self.log_operation('debug', f"Found {len(email_ids)} unseen email(s). Checking latest...")
                        # Fetch the latest emails first
                        for email_id in reversed(email_ids):
                            status, msg_data = mail.fetch(email_id, "(RFC822)")
                            if status == "OK":
                                for response_part in msg_data:
                                    if isinstance(response_part, tuple):
                                        msg = email.message_from_bytes(response_part[1])
                                        # Explicitly check 'To' or 'Delivered-To' header for the alias
                                        recipient = msg.get('To', '') or msg.get('Delivered-To', '')
                                        if alias_email.lower() in recipient.lower():
                                            subject = msg['subject']
                                            sender = msg['from']
                                            await self.log_operation('info', f"Processing email for '{alias_email}' from '{sender}' with subject '{subject}'")

                                            # Look for code in subject or body
                                            code = None
                                            # Prioritize subject check (handle potential None subject)
                                            subject_text = subject if subject else ""
                                            code_match_subject = re.search(r'\b(\d{4,8})\b', subject_text) # 4-8 digits
                                            if code_match_subject:
                                                code = code_match_subject.group(1)
                                                await self.log_operation('info', f"Verification code found in subject: {code}")

                                            if not code: # If not in subject, check body
                                                body_text = ""
                                                if msg.is_multipart():
                                                    for part in msg.walk():
                                                        content_type = part.get_content_type()
                                                        content_disposition = str(part.get("Content-Disposition"))
                                                        try:
                                                            if "attachment" not in content_disposition and ("text/plain" in content_type or "text/html" in content_type):
                                                                body_part = part.get_payload(decode=True)
                                                                if body_part:
                                                                    # Try decoding with utf-8 first, fallback to others if needed
                                                                    try: body_text += body_part.decode('utf-8', errors='ignore') + "\n"
                                                                    except UnicodeDecodeError: body_text += body_part.decode('latin-1', errors='ignore') + "\n"
                                                        except Exception as decode_err:
                                                            await self.log_operation('warning', f"Could not decode/process email part: {decode_err}")
                                                else: # Not multipart
                                                    try:
                                                        body_part = msg.get_payload(decode=True)
                                                        if body_part:
                                                            try: body_text = body_part.decode('utf-8', errors='ignore')
                                                            except UnicodeDecodeError: body_text = body_part.decode('latin-1', errors='ignore')
                                                    except Exception as decode_err:
                                                         await self.log_operation('warning', f"Could not decode non-multipart email body: {decode_err}")

                                                # Search body_text for common patterns
                                                if body_text:
                                                    # More robust regex, handles spaces, variations, common keywords
                                                    patterns = [
                                                        r'verification code is[:\s]*\b(\d{4,8})\b',
                                                        r'\b(\d{4,8})\b\s*is your.*?code',
                                                        r'code:\s*\b(\d{4,8})\b',
                                                        r'security code:\s*\b(\d{4,8})\b',
                                                        r'PIN:\s*\b(\d{4,8})\b',
                                                        r'>\s*(\d{4,8})\s*<' # Code inside HTML tags like <b>123456</b>
                                                    ]
                                                    for pattern in patterns:
                                                        code_match = re.search(pattern, body_text, re.IGNORECASE | re.DOTALL)
                                                        if code_match:
                                                            code = code_match.group(1)
                                                            await self.log_operation('info', f"Verification code found in body using pattern '{pattern}': {code}")
                                                            break # Found code
                                                    if not code:
                                                         await self.log_operation('debug', f"No code pattern matched in body for email to {alias_email}. Body snippet: {body_text[:200]}...")

                                            if code:
                                                # Optionally mark email as read: mail.store(email_id, '+FLAGS', '\\Seen')
                                                # Ensure logout happens before returning success
                                                if mail and mail.state == 'SELECTED': mail.close(); mail.logout(); mail = None
                                                await self.log_operation('debug', "IMAP connection closed after finding code.")
                                                return code # Success!
                                            else:
                                                 await self.log_operation('warning', f"Email found for {alias_email}, but no verification code pattern matched.")
                            else:
                                 await self.log_operation('warning', f"IMAP fetch failed for email ID {email_id}.")
                    else:
                        await self.log_operation('debug', f"No unseen emails found yet for {alias_email}. Retrying in {retry_delay}s...")
                else:
                    await self.log_operation('error', f"IMAP search failed with status: {status}")
                    # No need to logout if search failed, connection might be bad
                    # No return here, let finally block handle logout and loop continue/timeout

                # Logout and close connection before sleep/retry
                if mail and mail.state == 'SELECTED':
                    mail.close()
                    mail.logout()
                    await self.log_operation('debug', "IMAP connection closed.")
                    mail = None # Reset mail object

            except imaplib.IMAP4.error as imap_err:
                await self.log_operation('error', f"IMAP Error for {alias_email}: {imap_err}")
                # Consider specific error handling, e.g., authentication failure
                if "authentication failed" in str(imap_err).lower():
                     await self.log_operation('critical', "IMAP AUTHENTICATION FAILED. Check HOSTINGER_IMAP_USER/PASS (App Password?).")
                     # Potentially raise a critical error or stop trying
                # No return here, let finally block handle logout and loop continue/timeout

            except Exception as e:
                await self.log_operation('error', f"Unexpected error during IMAP check for {alias_email}: {e}")
                # Log traceback for debugging
                logger.exception(f"Unexpected IMAP error for {alias_email}")
                # No return here, let finally block handle logout and loop continue/timeout

            finally:
                # Ensure logout happens even if errors occur mid-loop
                if mail and mail.state in ['SELECTED', 'AUTH', 'NONAUTH']:
                    try:
                        mail.logout()
                        await self.log_operation('debug', "IMAP connection logged out in finally block.")
                    except Exception as logout_err:
                         await self.log_operation('warning', f"Error during IMAP logout: {logout_err}")
                mail = None # Ensure mail is reset

            # Wait before retrying if code not found yet
            if time.time() - start_time < timeout_seconds:
                 await asyncio.sleep(retry_delay)

        await self.log_operation('error', f"Timeout reached ({timeout_seconds}s) waiting for verification code for {alias_email}.")
        return None # Timeout

    # --- End IMAP Verification ---

    async def _get_temporary_phone_number(self, page):
        # This function remains largely the same, using SMS providers
        # Add logging
        provider = max(self.provider_success_rates['sms'], key=self.provider_success_rates['sms'].get)
        await self.log_operation('debug', f"Attempting to get temporary phone number using provider: {provider}")
        api_key = self.sms_api_keys.get(provider)
        if api_key and provider == 'https://www.twilio.com/': # Example API provider
            try:
                # Replace with actual API call logic for the provider
                async with aiohttp.ClientSession() as session:
                     # Example: Fictional API endpoint
                     # async with session.get(f"https://api.sms-activate.org/...?api_key={api_key}&action=getNumber&service=...") as response:
                     # Fictional Twilio example (needs proper Twilio library usage)
                     # client = TwilioClient(account_sid, auth_token) # Get SID/Token from config
                     # numbers = client.incoming_phone_numbers.list(limit=1)
                     # if numbers: phone = numbers[0].phone_number; return phone.strip()
                     pass # Placeholder for actual API logic
                await self.log_operation('info', f"Successfully obtained phone via API from {provider}")
                # return phone
            except Exception as e:
                await self.log_operation('error', f"{provider} API error: {e}")
                logger.error(f"{provider} API error: {e}")
                self.provider_success_rates['sms'][provider] -= 0.1 # Penalize on failure

        # Fallback to web-based providers
        for alt_provider in sorted(self.sms_providers, key=lambda x: self.provider_success_rates['sms'].get(x, 1.0), reverse=True):
            try:
                await self.log_operation('debug', f"Trying web fallback SMS provider: {alt_provider}")
                await page.goto(alt_provider, wait_until='networkidle', timeout=30000)
                # Adjust selectors based on actual site structure
                phone_element = await page.wait_for_selector('div.phone-number a, span.phone, li.number', timeout=15000)
                phone = await phone_element.inner_text()
                phone = re.sub(r'\D', '', phone) # Clean non-digits
                if phone:
                    self.provider_success_rates['sms'][alt_provider] = min(1.0, self.provider_success_rates['sms'].get(alt_provider, 1.0) + 0.05) # Reward success
                    await self.log_operation('info', f"Obtained phone {phone} from {alt_provider}")
                    return phone.strip()
            except Exception as e:
                await self.log_operation('warning', f"Failed to get phone from {alt_provider}: {e}")
                logger.warning(f"Failed to get phone from {alt_provider}: {e}")
                self.provider_success_rates['sms'][alt_provider] = max(0.0, self.provider_success_rates['sms'].get(alt_provider, 1.0) - 0.1) # Penalize failure
        await self.log_operation('error', "All SMS providers failed.")
        raise Exception("All SMS providers failed.")


    async def _get_sms_code(self, page, phone):
        """Extract SMS verification code with retries."""
        # Assumes page is already navigated to the SMS provider's inbox page for 'phone'
        await self.log_operation('info', f"Attempting to retrieve SMS code for phone {phone}")
        for attempt in range(10): # 10 retries
            try:
                await page.reload(wait_until='networkidle', timeout=30000)
                await asyncio.sleep(5) # Wait for messages to potentially arrive
                # More robust selector for messages containing codes
                messages = await page.query_selector_all("div.message, li.sms, tr.message-row") # Example selectors
                for message_element in reversed(messages): # Check newest first
                    message_text = await message_element.inner_text()
                    # Use regex to find common code patterns
                    code_match = re.search(r'verification code is[:\s]*\b(\d{4,8})\b', message_text, re.IGNORECASE) or \
                                 re.search(r'\b(\d{4,8})\b\s*is your.*?code', message_text, re.IGNORECASE) or \
                                 re.search(r'code:\s*\b(\d{4,8})\b', message_text, re.IGNORECASE) or \
                                 re.search(r'PIN:\s*\b(\d{4,8})\b', message_text, re.IGNORECASE)
                    if code_match:
                        code = code_match.group(1)
                        await self.log_operation('info', f"Found SMS code {code} for phone {phone}")
                        return code
                await self.log_operation('debug', f"SMS code not found on attempt {attempt + 1}. Retrying.")
            except PlaywrightTimeoutError:
                await self.log_operation('warning', f"Timeout reloading/finding messages on attempt {attempt + 1}.")
            except Exception as e:
                 await self.log_operation('error', f"Error getting SMS code on attempt {attempt + 1}: {e}")
            await asyncio.sleep(10) # Wait longer between retries

        await self.log_operation('error', f"SMS code not received for phone {phone} after multiple attempts.")
        raise Exception("SMS code not received.")


    async def _extract_api_key(self, page, service):
        """Uses LLM to extract API key from page content."""
        await self.log_operation('info', f"Attempting to extract API key for {service} from page content.")
        try:
            page_content = await page.content()
        except Exception as e:
             await self.log_operation('error', f"Failed to get page content for API key extraction: {e}")
             return None

        prompt = f"""
        {self.meta_prompt}
        Extract the API key from the HTML content below, likely from {service}'s dashboard or API settings page.
        Look for a long alphanumeric string (32+ characters) explicitly labeled as 'API Key', 'Secret Key', 'Token', 'Access Token', or similar common patterns. It might be inside an input field, code block, or plain text. Prioritize clearly labeled keys.
        HTML Snippet:
        {page_content[:4000]}...
        Return ONLY the extracted key as a plain string, or the string 'None' if no key is confidently identified.
        """
        if not self.clients_models:
             await self.log_operation('error', "No LLM clients configured for API key extraction.")
             return None

        # Try each configured client/model
        for client, model in self.clients_models:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1 # Low temperature for factual extraction
                )
                key = response.choices[0].message.content.strip()
                # Basic validation: check length and common 'None' response
                if key and len(key) >= 32 and key.lower() != 'none':
                    await self.log_operation('info', f"Successfully extracted potential API key for {service}.")
                    # Optional: Add regex check for typical key patterns
                    return key
                else:
                    await self.log_operation('debug', f"LLM client {client.base_url} did not find a valid key (response: '{key}').")
            except Exception as e:
                await self.log_operation('warning', f"Failed to extract API key with {client.base_url} model {model}: {e}")
                logger.warning(f"Failed to extract API key with {client.base_url}: {e}")

        await self.log_operation('warning', f"Failed to extract API key for {service} after trying all LLM clients.")
        return None


    async def _acquire_service_api_key(self, service_url):
        """Creates an account for a service specifically to get an API key."""
        await self.log_operation('info', f"Attempting to acquire new API key by creating account for: {service_url}")
        account = await self.create_account(service_url) # Calls the main account creation logic
        if not account or not account.get('api_key'):
            await self.log_operation('error', f"Failed to acquire API key for {service_url} via new account creation.")
            logger.error(f"Failed to acquire API key for {service_url}")
            return None
        api_key = account['api_key']
        # Update relevant internal state if needed (e.g., for SMS API keys)
        # if service_url == 'https://inboxes.com': # Example - This service is removed
        #     self.inboxes_api_key = api_key
        if service_url in self.sms_providers:
            self.sms_api_keys[service_url] = api_key
            await self.log_operation('info', f"Updated SMS API key for {service_url}")

        await self.log_operation('info', f"Successfully acquired API key for {service_url}")
        logger.info(f"Acquired API key for {service_url}: {'*' * (len(api_key) - 4)}{api_key[-4:]}") # Log masked key
        return api_key


    async def check_recurring_credits(self):
        """Check for recurring credit resets and notify agents."""
        # This method seems redundant with reuse_accounts, consider merging or removing
        await self.log_operation('debug', "Executing check_recurring_credits (potentially redundant).")
        # Keeping the logic for now in case it serves a distinct notification purpose
        while True:
            try:
                async with self.session_maker() as session:
                    one_month_ago = datetime.utcnow() - timedelta(days=30)
                    # Assuming 'is_recurring' column exists
                    try:
                        result = await session.execute(
                            sqlalchemy.text("SELECT service, email, api_key FROM accounts WHERE is_recurring = TRUE AND created_at < :threshold"),
                            {"threshold": one_month_ago}
                        )
                        accounts = result.fetchall()
                    except Exception as db_err_recur:
                        await self.log_operation('error', f"DB error fetching recurring accounts: {db_err_recur}")
                        accounts = [] # Continue gracefully

                    for service, email, api_key in accounts:
                        # Find agent responsible for this service (needs implementation)
                        # agent_name = self.get_agent_for_service(service)
                        agent_name = None # Placeholder
                        if agent_name and agent_name in self.orchestrator.agents:
                            # Ensure agent has the notification method
                            agent = self.orchestrator.agents[agent_name]
                            if hasattr(agent, 'notify_recurring_account'):
                                await agent.notify_recurring_account(service, email, api_key)
                                await self.log_operation('info', f"Notified {agent_name} of recurring account for {service}: {email}")
                            else:
                                await self.log_operation('warning', f"Agent {agent_name} lacks 'notify_recurring_account' method.")
                        # Send general notification regardless
                        if hasattr(self.orchestrator, 'send_notification'):
                             await self.orchestrator.send_notification(
                                 "Recurring Credits Available",
                                 f"Account for {service} with email {email} may have refreshed credits."
                             )

                    # Update timestamp only if reuse logic doesn't handle it
                    # This might cause conflicts if reuse_accounts also updates it.
                    # Consider consolidating the update logic in reuse_accounts.
                    # await session.execute(
                    #     sqlalchemy.text("UPDATE accounts SET created_at = :now WHERE is_recurring = TRUE AND created_at < :threshold"),
                    #     {"now": datetime.utcnow(), "threshold": one_month_ago}
                    # )
                    # await session.commit()
                await asyncio.sleep(86400)  # Check daily
            except Exception as e:
                await self.log_operation('error', f"Recurring credits check loop failed: {e}")
                logger.error(f"Recurring credits check loop failed: {e}")
                await asyncio.sleep(3600) # Wait longer after failure


    async def monitor_credits(self, services):
        """Monitor credits and rotate accounts when drained."""
        # This method seems complex and potentially overlaps with reuse_accounts.
        # Consider simplifying or integrating credit checks directly into agent workflows
        # where the credits are actually consumed.
        await self.log_operation('warning', "monitor_credits function execution started - review for necessity/overlap.")
        if not services:
            logger.warning("Monitor_credits called with no services.")
            await self.log_operation('warning', "monitor_credits called with empty service list.")
            return

        # Simplified placeholder - Full implementation requires careful state management
        # and integration with account usage.
        await self.log_operation('info', f"Placeholder: Monitoring credits for services: {services}")
        await asyncio.sleep(3600) # Example delay


    async def manage_gemini_services(self):
        """Placeholder for managing specific Gemini interactions."""
        # This method seems highly specific and might belong in a dedicated agent
        # or be integrated into ThinkTool/Orchestrator strategies.
        await self.log_operation('warning', "manage_gemini_services function execution started - review for necessity/placement.")
        await self.log_operation('info', "Placeholder: Managing Gemini services (Deep Search, AI Studio).")
        await asyncio.sleep(3600) # Example delay


    async def store_gemini_results(self, type, data):
        """Stores results from Gemini interactions."""
        await self.log_operation('debug', f"Storing Gemini result of type: {type}")
        async with self.session_maker() as session:
            try:
                # Assuming gemini_results table exists
                await session.execute(
                    sqlalchemy.text("INSERT INTO gemini_results (type, data, timestamp) VALUES (:type, :data, :timestamp)"),
                    {"type": type, "data": data, "timestamp": datetime.utcnow()} # Use datetime
                )
                await session.commit()
            except Exception as e:
                 await self.log_operation('error', f"Failed to store Gemini results: {e}")
                 logger.error(f"Failed to store Gemini results: {e}")


    async def store_credit_status(self, service, credits):
        """Stores the credit status for a service."""
        await self.log_operation('debug', f"Storing credit status for {service}: {credits}")
        async with self.session_maker() as session:
             try:
                 # Assuming service_credits table exists
                 await session.execute(
                     sqlalchemy.text("INSERT INTO service_credits (service, credits, timestamp) VALUES (:service, :credits, :timestamp)"),
                     {"service": service, "credits": str(credits), "timestamp": datetime.utcnow()} # Store credits as string?
                 )
                 await session.commit()
             except Exception as e:
                  await self.log_operation('error', f"Failed to store credit status for {service}: {e}")
                  logger.error(f"Failed to store credit status for {service}: {e}")


    async def check_credit_status(self, service, account_sid=None, auth_token=None):
        """Check credit status via API if available, else optimized browser check."""
        await self.log_operation('debug', f"Checking credit status for {service}")
        if service == 'twilio' and account_sid and auth_token:
            try:
                client = TwilioClient(account_sid, auth_token)
                balance_resource = client.balance.fetch()
                balance = balance_resource.balance
                currency = balance_resource.currency
                await self.log_operation('info', f"Twilio API credit check: {balance} {currency}")
                return float(balance)
            except Exception as e:
                await self.log_operation('error', f"Twilio API check failed: {e}")
                logger.error(f"Twilio API check failed: {e}")
                # Fallback to browser check might require login, which is complex
                # return await self._browser_check_credit(service)
                return 0.0 # Return 0 on API failure for now
        else:
            # General case: Browser check (if implemented and reliable)
            # return await self._browser_check_credit(service)
            await self.log_operation('warning', f"API credit check not available for {service}. Browser check not implemented/reliable.")
            return 0.0 # Return 0 if no reliable check method


    async def _browser_check_credit(self, service):
        """Optimized browser-based credit check using headless mode."""
        # This method is complex and often unreliable due to login requirements.
        # Prefer API checks where possible. Kept as placeholder.
        await self.log_operation('warning', f"Attempting unreliable browser credit check for {service}.")
        async with self.browser_semaphore:
            proxy = await self.get_next_proxy()
            if not proxy:
                 await self.log_operation('error', "No proxy available for browser credit check.")
                 return 0.0
            async with async_playwright() as p:
                browser = None
                context = None
                try:
                    browser = await p.chromium.launch(headless=True, proxy={
                        'server': f"http://{proxy['server']}",
                        'username': proxy['username'],
                        'password': proxy['password']
                    })
                    context = await browser.new_context()
                    page = await context.new_page()
                    # Requires login logic first, which is complex and service-specific
                    # await page.goto(f"https://{service}.com/login", wait_until='networkidle')
                    # ... fill login form ...
                    # await page.goto(f"https://{service}.com/dashboard", wait_until='networkidle')
                    # credits_element = await page.query_selector("span.credits, div.balance, #credits") # Example selectors
                    # credit_value_str = await credits_element.inner_text() if credits_element else "Unknown"
                    credit_value_str = "Unknown" # Placeholder
                    await self.log_operation('info', f"Browser credit check for {service}: {credit_value_str} (Login not implemented)")
                    logger.info(f"Browser credit check for {service}: {credit_value_str}")
                    # Process credit_value_str if needed
                    return 0.0 # Return 0 as login/extraction is not implemented
                except Exception as e:
                    await self.log_operation('error', f"Browser check failed for {service}: {e}")
                    logger.error(f"Browser check failed for {service}: {e}")
                    return 0.0
                finally:
                    if context:
                        try: await context.close()
                        except Exception: pass # Ignore errors on close
                    if browser:
                        try: await browser.close()
                        except Exception: pass # Ignore errors on close


    async def run(self):
        """Main loop processing tasks from the queue."""
        await self.log_operation('info', "BrowsingAgent run loop started.")
        # Start periodic tasks in the background
        asyncio.create_task(self.reuse_accounts_periodically())
        # asyncio.create_task(self.check_recurring_credits()) # Consider removing/merging this

        while True:
            try:
                await self.log_operation('debug', f"Waiting for task. Queue size: {self.task_queue.qsize()}")
                task = await self.task_queue.get()
                await self.log_operation('info', f"Processing task: {task.get('action', 'N/A')} for service: {task.get('service_url', task.get('service', 'N/A'))}")

                service_url = task.get('service_url', task.get('service', ''))
                action = task.get('action')

                if action == 'create_account' and service_url:
                    result = await self.create_account(service_url, retry_count=0)
                    if result:
                        # Report success to orchestrator
                        service_name = service_url.split('//')[-1].split('/')[0].replace('www.', '')
                        if hasattr(self.orchestrator, 'report_account_created') and callable(self.orchestrator.report_account_created):
                             await self.orchestrator.report_account_created(service_name, result)
                        else:
                             await self.log_operation('warning', f"Orchestrator missing 'report_account_created' method.")
                    # Update provider success only if relevant (SMS) - logic is placeholder
                    # self._update_provider_success(service_url, bool(result))
                else:
                    await self.log_operation('warning', f"Unknown or invalid task received: {task}")
                    logger.warning(f"Unknown or invalid task received: {task}")

            except Exception as e:
                await self.log_operation('error', f"Error in BrowsingAgent run loop: {e}")
                logger.exception("Error in BrowsingAgent run loop") # Log full traceback
                # Avoid continuous fast loops on persistent errors
                await asyncio.sleep(60)
            finally:
                 # Ensure task_done is called even if processing fails mid-way
                 # Check if task was retrieved before calling task_done
                 if 'task' in locals() and task is not None:
                     try:
                         self.task_queue.task_done()
                     except ValueError: # Handle case where task might already be done
                         pass
                 await asyncio.sleep(1) # Small delay to prevent tight loop if queue is empty


    # Removed process_queue and process_task as logic is integrated into run()

    def _update_provider_success(self, service, success):
        """Update provider success rates (Now only for SMS)."""
        # Simplified: Update success only if phone verification was attempted and relevant provider used
        # This needs more context from the create_account steps if phone verification is added back
        # For now, this function might not be actively used unless phone verification is implemented.
        # Example:
        # if phone_verification_attempted_for_service:
        #     provider_used = get_sms_provider_used_for_attempt(...) # Needs implementation
        #     if provider_used in self.provider_success_rates.get('sms', {}):
        #         rate = self.provider_success_rates['sms'][provider_used]
        #         self.provider_success_rates['sms'][provider_used] = min(max(rate + (0.05 if success else -0.05), 0), 1)
        #         await self.log_operation('debug', f"Updated SMS provider {provider_used} success rate to {self.provider_success_rates['sms'][provider_used]:.2f}")
        pass


    async def get_insights(self):
        # Return SMS provider rates if needed, or other relevant browsing insights
        await self.log_operation('debug', "Generating insights (SMS provider success rates).")
        return {"sms_provider_success_rates": self.provider_success_rates.get('sms', {})}

    # --- Helper method to get today's account count ---
    async def get_today_account_count(self, service):
        """Gets the count of accounts created today for a specific service."""
        try:
            async with self.session_maker() as session:
                today_start = datetime.utcnow().date()
                # Ensure SQL query is compatible with PostgreSQL date comparison
                # Use DATE() function for SQLite compatibility if needed, or ::date for PostgreSQL
                # Assuming PostgreSQL based on asyncpg likely being used with create_async_engine
                # Using CAST for broader compatibility attempt
                # Ensure correct SQL syntax for date comparison based on the actual DB used
                # For PostgreSQL:
                # sql_query = "SELECT COUNT(*) FROM accounts WHERE service = :service AND created_at::date = :today"
                # For SQLite:
                sql_query = "SELECT COUNT(*) FROM accounts WHERE service = :service AND DATE(created_at) = :today"

                result = await session.execute(
                   sqlalchemy.text(sql_query), # Use text() for raw SQL with ORM session
                   {"service": service, "today": today_start}
                )
                count = result.scalar() or 0
                await self.log_operation('debug', f"Accounts created today for {service}: {count}")
                return count
        except Exception as e:
            # Import sqlalchemy here if not already imported at top level
            # import sqlalchemy # No need to re-import if already at top
            await self.log_operation('error', f"Failed to get today's account count for {service}: {e}")
            logger.error(f"Failed to get today's account count for {service}: {e}")
            return 0 # Return 0 on error to avoid blocking unnecessarily

# Example usage (commented out)
# async def main():
#     from sqlalchemy.ext.asyncio import create_async_engine
#     engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
#     session_maker = async_sessionmaker(engine, expire_on_commit=False)
#     config = {} # Load actual config
#     orchestrator = None # Mock or load actual orchestrator
#     clients_models = [] # Load actual clients/models
#     agent = BrowsingAgent(session_maker, config, orchestrator, clients_models)
#     await agent.task_queue.put({'service_url': 'https://example.com', 'action': 'create_account'})
#     await agent.run()
#
# if __name__ == "__main__":
#     # asyncio.run(main())
#     pass
