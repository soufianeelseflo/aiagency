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
import json # Added import for json

logger = logging.getLogger(__name__)
fake = Faker()

class BrowsingAgent:
    def __init__(self, session_maker, config, orchestrator, clients_models):
        self.session_maker = session_maker
        self.config = config
        self.orchestrator = orchestrator
        self.clients_models = clients_models
        self.think_tool = orchestrator.agents['think']
        self.task_queue = asyncio.Queue()
        self.max_concurrency = 10

        # Diverse email providers for infinite trials
        self.email_providers = [
            'https://temp-mail.org/en/',
            'https://www.guerrillamail.com/',
            'https://10minutemail.com/',
            'https://mail.tm/en/',
            'https://inboxes.com'  # API-enabled
        ]
        # Diverse SMS providers for infinite trials
        self.sms_providers = [
            'https://www.receivesms.co/us-phone-numbers/',
            'https://sms-activate.org/',
            'https://www.textnow.com/',
            'https://www.freephonenum.com/',
            'https://receive-sms.cc/',
            'https://www.twilio.com/'
        ]
        self.inboxes_api_key = os.getenv("INBOXES_API_KEY")
        self.sms_api_keys = {provider: os.getenv(f"{provider.split('.')[1].upper()}_API_KEY") for provider in self.sms_providers if provider.endswith('.com')}

        # Success rates for dynamic rotation
        self.provider_success_rates = {
            'email': {url: 1.0 for url in self.email_providers},
            'sms': {url: 1.0 for url in self.sms_providers},
        }

        # SmartProxy Configuration
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

    async def generate_test_videos(self, industries):
        """Generate mock video URLs for testing."""
        videos = []
        for industry in industries:
            video_url = f"/static/videos/test_{industry}_{int(time.time())}.mp4"
            # Simulate video file creation (in production, integrate with a video service)
            # Ensure the 'ui/static/videos' directory exists
            os.makedirs(os.path.dirname(f"ui{video_url}"), exist_ok=True)
            with open(f"ui{video_url}", "wb") as f:
                f.write(b"Mock video data")
            videos.append(video_url)
        logger.info(f"Generated test videos: {videos}")
        return videos
    
    def get_allowed_concurrency(self):
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        concurrency_factor = 1 - (cpu_usage / 100) * 0.5 - (memory_usage / 100) * 0.5
        allowed = max(1, int(self.max_concurrency * concurrency_factor))
        return allowed

    def create_cache_tables(self):
        """Initialize SQLite cache for emails, phones, strategies, and attempt logs."""
        cursor = self.cache_db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_cache (
                id INTEGER PRIMARY KEY,
                email TEXT UNIQUE,
                provider TEXT,
                service TEXT,
                used BOOLEAN DEFAULT FALSE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS phone_cache (
                id INTEGER PRIMARY KEY,
                phone TEXT UNIQUE,
                provider TEXT,
                service TEXT,
                used BOOLEAN DEFAULT FALSE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
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

    async def load_strategies(self, service):
        """Load strategies for a service, sorted by success rate."""
        cursor = self.cache_db.cursor()
        cursor.execute('SELECT id, strategy, success_rate FROM strategy_cache WHERE service = ? ORDER BY success_rate DESC', (service,))
        rows = cursor.fetchall()
        strategies = []
        for row in rows:
            id = row[0]
            strategy = json.loads(row[1])
            success_rate = row[2]
            strategies.append({'id': id, 'steps': strategy, 'success_rate': success_rate})
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
        cursor = self.cache_db.cursor()
        cursor.execute('SELECT success_rate FROM strategy_cache WHERE id = ?', (strategy_id,))
        row = cursor.fetchone()
        if row:
            old_rate = row[0]
            new_rate = old_rate * 0.9 + (1.0 if success else 0.0) * 0.1
            cursor.execute('UPDATE strategy_cache SET success_rate = ? WHERE id = ?', (new_rate, strategy_id))
            self.cache_db.commit()
        else:
            logger.error(f"Strategy ID {strategy_id} not found for update")

    async def get_next_proxy(self, service=None):
        current_time = time.time()
        available_proxies = []
        for proxy in self.proxy_pool:
            if proxy['failure_count'] >= 3:
                continue
            if service:
                last_used = self.proxy_service_usage.get(proxy['server'], {}).get(service, 0)
                if current_time - last_used < 3600:  # 1 hour cooldown
                    continue
            if current_time - proxy.get('last_checked', 0) > 300:
                if await self._validate_proxy(proxy):
                    available_proxies.append(proxy)
                else:
                    proxy['failure_count'] += 1
                    logger.warning(f"Proxy {proxy['server']} failed validation.")
            else:
                available_proxies.append(proxy)
        if not available_proxies:
            raise Exception("No suitable proxies available.")
        selected_proxy = random.choice(available_proxies)
        if service:
            if selected_proxy['server'] not in self.proxy_service_usage:
                self.proxy_service_usage[selected_proxy['server']] = {}
            self.proxy_service_usage[selected_proxy['server']][service] = current_time
        return selected_proxy

    async def _validate_proxy(self, proxy):
        """Validate proxy connectivity."""
        try:
            async with aiohttp.ClientSession() as session:
                auth = aiohttp.BasicAuth(proxy['username'], proxy['password'])
                async with session.get('http://ipinfo.io/json', proxy=f"http://{proxy['server']}", auth=auth, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        proxy['last_checked'] = time.time()
                        proxy['failure_count'] = 0
                        return True
        except Exception:
            return False
        return False

    async def infer_signup_steps(self, service_url):
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
        - 'field': (e.g., 'email') or 'value': (fixed text)
        - 'verification_type': ('email', 'phone') if applicable
        - 'verification_selector': (for code entry)
        """
        for client, model in self.clients_models:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                steps = json.loads(response.choices[0].message.content)['steps']
                return steps
            except Exception as e:
                logger.warning(f"Failed to infer steps with {client.base_url}: {e}")
        raise Exception("All clients failed to infer signup steps.")

    async def create_account(self, service_url):
        """Create an account for any service with adaptive learning."""
        # Extract domain name to use as service identifier for caching/strategy
        try:
            service = service_url.split('//')[-1].split('/')[0].replace('www.', '')
        except Exception:
            logger.error(f"Could not extract domain from service_url: {service_url}")
            service = service_url # Fallback

        while self.active_browsers >= self.get_allowed_concurrency():
            await asyncio.sleep(1)

        async with self.browser_semaphore:
            self.active_browsers += 1
            strategy_id = None # Initialize strategy_id
            proxy = await self.get_next_proxy(service) # Pass service for cooldown
            # Track proxy cost
            proxy_cost = 0.01
            await self.orchestrator.agents['budget'].track_expense(
                proxy_cost, "Proxy", f"Proxy cost for creating account on {service}"
            )

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, proxy={
                    'server': f"http://{proxy['server']}",
                    'username': proxy['username'],
                    'password': proxy['password']
                })
                context = await browser.new_context(user_agent=fake.user_agent()) # Add user agent spoofing
                page = await context.new_page()
                phone = None # Initialize phone variable

                try:
                    # Generate credentials
                    email = await self._get_temporary_email(page, service) # Pass service for caching context
                    password = fake.password(length=12, special_chars=True)
                    name = fake.name()

                    # Load or infer strategy
                    loaded_strategies = await self.load_strategies(service)
                    if loaded_strategies and loaded_strategies[0]['success_rate'] > 0.5:
                         steps = loaded_strategies[0]['steps']
                         strategy_id = loaded_strategies[0]['id']
                         logger.info(f"Using loaded strategy (ID: {strategy_id}) for {service}")
                    else:
                        logger.info(f"Inferring new strategy for {service}")
                        steps = await self.infer_signup_steps(service_url)
                        if not steps: # Handle inference failure
                            raise Exception(f"Failed to infer signup steps for {service_url}")
                        strategy_id = await self.save_strategy(service, steps, 1.0) # Save newly inferred strategy
                        logger.info(f"Saved new strategy (ID: {strategy_id}) for {service}")

                    # Execute steps
                    for step in steps:
                        logger.info(f"Executing step for {service}: {step['action']}")
                        try:
                            if step['action'] == 'navigate':
                                await page.goto(step['url'], wait_until='networkidle', timeout=60000)
                            elif step['action'] == 'type':
                                value_to_type = {'email': email, 'password': password, 'name': name}.get(step.get('field'), step.get('value'))
                                if value_to_type is None:
                                     logger.warning(f"No value found for step: {step}")
                                     continue
                                await page.fill(step['selector'], value_to_type, timeout=30000)
                            elif step['action'] == 'click':
                                await page.click(step['selector'], timeout=30000)

                            await asyncio.sleep(random.uniform(1.5, 3.0)) # Slightly longer human-like delay
                            await self.log_attempt(service, json.dumps(step), True) # Log step success

                        except Exception as e:
                            await self.log_attempt(service, json.dumps(step), False) # Log step failure
                            logger.error(f"Step failed for {service}: {step} - {e}")
                            raise Exception(f"Step {step.get('action', 'unknown')} failed: {e}") # Re-raise to fail account creation

                    # Handle verification dynamically based on the last step's flags
                    last_step = steps[-1]
                    if last_step.get('requires_verification'):
                        verification_type = last_step.get('type')
                        verification_selector = last_step.get('verification_selector')
                        submit_selector = 'button[type="submit"], input[type="submit"], button:contains("Verify"), button:contains("Confirm")' # Common submit selectors

                        if not verification_selector:
                             logger.error(f"Verification required but no verification_selector defined for service {service}")
                             raise Exception("Missing verification selector in strategy")

                        if verification_type == 'email':
                            logger.info(f"Attempting email verification for {service}")
                            code = await self._get_email_verification_code(page, email) # Re-use page
                            await page.fill(verification_selector, code)
                            await page.click(submit_selector) # Click generic submit
                            logger.info(f"Email verification submitted for {service}")
                        elif verification_type == 'phone':
                            logger.info(f"Attempting phone verification for {service}")
                            phone = await self._get_temporary_phone_number(page) # Re-use page
                            # Assume phone number input field was handled in steps, just need code
                            sms_code = await self._get_sms_code(page, phone) # Re-use page
                            await page.fill(verification_selector, sms_code)
                            await page.click(submit_selector) # Click generic submit
                            logger.info(f"Phone verification submitted for {service}")
                        else:
                             logger.warning(f"Unknown verification type '{verification_type}' for service {service}")
                             # Proceed without verification if type is unknown/missing

                        await asyncio.sleep(random.uniform(2, 4)) # Wait after verification submit

                    # Extract API key (if applicable/possible) - Best effort
                    api_key = await self._extract_api_key(page, service) # Re-use page

                    # Store account details
                    await self._store_account(service, email, password, api_key, phone) # Pass phone if obtained
                    if strategy_id: await self.update_strategy_success(strategy_id, True)
                    logger.info(f"Successfully created account for {service} with email {email}")
                    return {'email': email, 'api_key': api_key}

                except Exception as e:
                    if strategy_id: await self.update_strategy_success(strategy_id, False)
                    logger.error(f"Account creation failed for {service_url}: {e}", exc_info=True)
                    # Don't re-raise here, let finally block handle cleanup
                    return None # Indicate failure
                finally:
                    self.active_browsers -= 1
                    await browser.close()

    async def _get_cached_email(self, service):
        """Retrieve a cached email if available."""
        async with self.session_maker() as session:
            result = await session.execute(
                "SELECT email FROM email_cache WHERE service = :service AND used = FALSE LIMIT 1",
                {"service": service}
            )
            row = result.fetchone()
            if row:
                await session.execute(
                    "UPDATE email_cache SET used = TRUE WHERE email = :email",
                    {"email": row[0]}
                )
                await session.commit()
                return row[0]
        return None

    async def _get_temporary_email(self, page, service):
        cached_email = await self._get_cached_email(service)
        if cached_email:
            return cached_email

        provider = max(self.provider_success_rates['email'], key=self.provider_success_rates['email'].get)
        if provider == 'https://inboxes.com' and self.inboxes_api_key:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"https://api.inboxes.com/v1/email?api_key={self.inboxes_api_key}") as response:
                        if response.status == 200:
                            data = await response.json()
                            email = data['email']
                            await self._cache_email(email, provider, service)
                            return email
                        else:
                            logger.warning(f"inboxes.com API failed: {response.status}")
                            self.provider_success_rates['email'][provider] -= 0.1
                            # Acquire new API key if rate-limited
                            if response.status == 429 or response.status == 403:
                                self.inboxes_api_key = await self._acquire_service_api_key('https://inboxes.com')
            except Exception as e:
                logger.error(f"inboxes.com API error: {e}")
                self.provider_success_rates['email'][provider] -= 0.1

        # Fallback to other providers
        for alt_provider in sorted(self.email_providers, key=lambda x: self.provider_success_rates['email'].get(x, 1.0), reverse=True):
            if alt_provider != provider:
                try:
                    await page.goto(alt_provider, wait_until='networkidle')
                    await page.wait_for_selector('input#mail', timeout=15000)
                    email = await page.input_value('input#mail')
                    await self._cache_email(email, alt_provider, service)
                    self.provider_success_rates['email'][alt_provider] += 0.05
                    return email
                except Exception as e:
                    logger.warning(f"Failed to get email from {alt_provider}: {e}")
                    self.provider_success_rates['email'][alt_provider] -= 0.1
        raise Exception("All email providers failed.")
    
    async def _cache_email(self, email, provider, service):
        """Cache email for reuse and tracking."""
        # Use a separate connection for cache operations to avoid thread issues if needed
        # For simplicity here, assuming check_same_thread=False handles it
        cursor = self.cache_db.cursor()
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO email_cache (email, provider, service, used) VALUES (?, ?, ?, FALSE)",
                (email, provider, service)
            )
            self.cache_db.commit()
            logger.debug(f"Cached email {email} from {provider} for {service}")
        except Exception as e:
            logger.error(f"Failed to cache email {email}: {e}")
        # No session needed as we are using sqlite3 directly here

    async def _get_email_verification_code(self, page, email):
        """Extract email verification code with retries."""
        provider = self.email_providers[0]
        for _ in range(10):
            await page.goto(f"{provider}/inbox", wait_until='domcontentloaded')
            await asyncio.sleep(5)
            try:
                code_element = await page.wait_for_selector('div:contains("verification code")', timeout=10000)
                code_text = await code_element.inner_text()
                code = ''.join(filter(str.isdigit, code_text))
                if len(code) >= 4:
                    return code
            except PlaywrightTimeoutError:
                continue
        raise Exception("Email verification code not received.")

    async def _get_temporary_phone_number(self, page):
        provider = max(self.provider_success_rates['sms'], key=self.provider_success_rates['sms'].get)
        api_key = self.sms_api_keys.get(provider)
        if api_key and provider == 'https://www.twilio.com/':
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"https://api.twilio.com/2010-04-01/Accounts/{api_key}/IncomingPhoneNumbers.json") as response:
                        if response.status == 200:
                            data = await response.json()
                            phone = data['incoming_phone_numbers'][0]['phone_number']
                            return phone.strip()
                        else:
                            logger.warning(f"Twilio API failed: {response.status}")
                            self.sms_api_keys[provider] = await self._acquire_service_api_key(provider)
            except Exception as e:
                logger.error(f"Twilio API error: {e}")
                self.provider_success_rates['sms'][provider] -= 0.1

        # Fallback to web-based providers
        for alt_provider in sorted(self.sms_providers, key=lambda x: self.provider_success_rates['sms'].get(x, 1.0), reverse=True):
            try:
                await page.goto(alt_provider, wait_until='networkidle')
                await page.wait_for_selector('div.phone-number', timeout=15000)
                phone_element = await page.query_selector('div.phone-number a')
                phone = await phone_element.inner_text()
                self.provider_success_rates['sms'][alt_provider] += 0.05
                return phone.strip()
            except Exception as e:
                logger.warning(f"Failed to get phone from {alt_provider}: {e}")
                self.provider_success_rates['sms'][alt_provider] -= 0.1
        raise Exception("All SMS providers failed.")

    async def _get_sms_code(self, page, phone):
        """Extract SMS verification code with retries."""
        for _ in range(10):
            await page.reload(wait_until='networkidle')
            await asyncio.sleep(5)
            try:
                code_element = await page.wait_for_selector('div.message', timeout=10000)
                code_text = await code_element.inner_text()
                code = ''.join(filter(str.isdigit, code_text))
                if len(code) >= 4:
                    return code
            except PlaywrightTimeoutError:
                continue
        raise Exception("SMS code not received.")

    async def _extract_api_key(self, page, service):
        page_content = await page.content()
        prompt = f"""
        {self.meta_prompt}
        Extract the API key from the HTML of {service}'s dashboard:
        {page_content[:2000]}...
        Identify a 32+ character alphanumeric string labeled 'API Key', 'Token', or similar.
        Handle various layouts and anticipate obfuscation or dynamic rendering.
        Return the key as a string or 'None' if not found.
        """
        for client, model in self.clients_models:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                key = response.choices[0].message.content.strip()
                if key and len(key) >= 32:
                    return key
            except Exception as e:
                logger.warning(f"Failed to extract API key with {client.base_url}: {e}")
        return None
    
    async def _acquire_service_api_key(self, service_url):
        account = await self.create_account(service_url)
        if not account or not account['api_key']:
            logger.error(f"Failed to acquire API key for {service_url}")
            return None
        api_key = account['api_key']
        if service_url == 'https://inboxes.com':
            self.inboxes_api_key = api_key
        elif service_url in self.sms_providers:
            self.sms_api_keys[service_url] = api_key
        logger.info(f"Acquired API key for {service_url}: {api_key}")
        return api_key

    async def _store_account(self, service, email, password, api_key, phone):
        """Store account details in PostgreSQL."""
        async with self.session_maker() as session:
            await session.execute("""
                INSERT INTO accounts (service, email, password, api_key, phone, created_at)
                VALUES (:service, :email, :password, :api_key, :phone, :created_at)
            """, {
                "service": service,
                "email": email,
                "password": password,
                "api_key": api_key,
                "phone": phone,
                "created_at": int(time.time())
            })
            await session.commit()

    async def monitor_credits(self, services):
        """Monitor credits and rotate accounts when drained."""
        if not services:
            logger.warning("Monitor_credits called with no services.")
            return

        async with async_playwright() as p:
            # Use a unique proxy per monitoring browser instance if possible, or rotate
            proxy = await self.get_next_proxy()
            browser = await p.chromium.launch(headless=True, proxy={
                'server': f"http://{proxy['server']}",
                'username': proxy['username'],
                'password': proxy['password']
            })
            context = await browser.new_context(user_agent=fake.user_agent())
            pages = {}
            # Initialize pages safely
            for service in services:
                try:
                    page = await context.new_page()
                    # Attempt initial navigation to dashboard
                    service_url = f"https://{service}" # Assuming dashboard is at root/dashboard path
                    dashboard_url = f"{service_url}/dashboard" # Common pattern
                    await page.goto(dashboard_url, wait_until='networkidle', timeout=60000)
                    pages[service] = page
                    logger.info(f"Initialized monitoring page for {service}")
                except Exception as e:
                    logger.error(f"Failed to initialize monitoring page for {service}: {e}. Skipping service.")
                    # Optionally close the failed page context?

            if not pages:
                 logger.error("Failed to initialize any monitoring pages. Exiting monitor_credits.")
                 await browser.close()
                 return

            while True: # Continuous monitoring loop
                await asyncio.sleep(random.uniform(280, 320)) # Check roughly every 5 minutes

                for service, page in list(pages.items()): # Use list to allow item removal
                    try:
                        # Refresh or re-navigate to ensure page is current
                        dashboard_url = f"https://{service}/dashboard"
                        await page.goto(dashboard_url, wait_until='networkidle', timeout=60000)

                        # Track proxy cost for each check
                        proxy_cost = 0.01
                        await self.orchestrator.agents['budget'].track_expense(
                            proxy_cost, "Proxy", f"Proxy cost for monitoring {service}"
                        )

                        # Check credits using a more robust selector strategy
                        credit_selectors = ["span.credits", "div.balance", "p[data-testid='credits']", "#credit-balance"]
                        credit_value_str = "Unknown"
                        for selector in credit_selectors:
                            credit_element = await page.query_selector(selector)
                            if credit_element:
                                credit_value_str = await credit_element.inner_text()
                                break # Found credits

                        # Process credit value
                        credit_value = 0.0
                        if credit_value_str != "Unknown":
                             try:
                                 # Extract numeric value, handling currency symbols etc.
                                 numeric_part = ''.join(filter(lambda x: x.isdigit() or x == '.', credit_value_str))
                                 if numeric_part:
                                     credit_value = float(numeric_part)
                                 else: credit_value = 0.0 # Treat non-numeric as zero
                             except ValueError:
                                 credit_value = 0.0 # Treat parsing errors as zero

                        logger.info(f"Credits check for {service}: Found '{credit_value_str}', Parsed as {credit_value}")

                        # If credits are drained, create a new account
                        if credit_value <= 0.01: # Use a small threshold instead of exact zero
                            logger.warning(f"Credits appear drained or very low for {service} ({credit_value}). Creating new account.")

                            # --- Provider Rotation Logic (Basic Example) ---
                            # If inboxes.com was likely used and failed recently, penalize it more
                            if self.provider_success_rates['email'].get('inboxes.com', 1.0) < 0.3:
                                logger.warning("inboxes.com success rate low, forcing fallback for next account creation.")
                                # Temporarily set success rate to 0 to force fallback in _get_temporary_email
                                original_rate = self.provider_success_rates['email']['inboxes.com']
                                self.provider_success_rates['email']['inboxes.com'] = 0
                                new_account = await self.create_account(f"https://{service}")
                                self.provider_success_rates['email']['inboxes.com'] = original_rate # Restore rate
                            else:
                                new_account = await self.create_account(f"https://{service}")
                            # --- End Rotation Logic ---

                            if new_account:
                                await self.orchestrator.report_account_created(service, new_account)
                                # Successfully created new account, re-navigate to dashboard
                                await page.goto(dashboard_url, wait_until='networkidle', timeout=60000)
                            else:
                                logger.error(f"Failed to create new account for {service} after credits drained.")
                                # Consider removing service from monitoring if creation fails repeatedly

                        # else: # Credits are fine
                            # logger.info(f"Credits OK for {service}: {credit_value}")

                    except PlaywrightTimeoutError:
                         logger.error(f"Timeout during credit check/navigation for {service}. Page might be stuck.")
                         # Consider closing and reopening the page/context for this service
                    except Exception as e:
                        logger.error(f"Unexpected error during credit check for {service}: {e}", exc_info=True)
                        # Consider removing the service from monitoring if errors persist
                        # del pages[service]
                        # await page.close()

            # Cleanup - This part might not be reached in continuous loop
            # await browser.close()

    async def manage_gemini_services(self):
        async with self.browser_semaphore:
            async with async_playwright() as p:
                browsers = []
                for _ in range(5):  # 5 browser instances
                    proxy = await self.get_next_proxy()
                    browser = await p.chromium.launch(headless=True, proxy={
                        'server': f"http://{proxy['server']}",
                        'username': proxy['username'],
                        'password': proxy['password']
                    })
                    context = await browser.new_context(
                        user_agent=fake.user_agent(),
                        viewport={'width': random.randint(1280, 1920), 'height': random.randint(720, 1080)}
                    )
                    browsers.append((browser, context))
                
                # Gemini deep search
                deep_search_page = await browsers[0][1].new_page()
                await deep_search_page.goto("https://gemini.google.com", wait_until='networkidle')
                await deep_search_page.click("button:contains('Deep Search')")
                await deep_search_page.fill("input#search", "AI agency profitable niches 2025")
                await deep_search_page.click("button[type='submit']")
                await deep_search_page.wait_for_selector("div.results", timeout=30000)
                results = await deep_search_page.content()
                await self.store_gemini_results("deep_search", results)
                
                # AI Studio image generation
                studio_page = await browsers[1][1].new_page()
                await studio_page.goto("https://aistudio.google.com", wait_until='networkidle')
                await studio_page.fill("input#prompt", "Unique UGC video thumbnail")
                await studio_page.click("button:contains('Generate')")
                await studio_page.wait_for_selector("img.generated", timeout=30000)
                image_data = await studio_page.screenshot(selector="img.generated")
                with open(f"/app/images/thumbnail_{int(time.time())}.png", "wb") as f:
                    f.write(image_data)
                
                # Monitor dashboards
                for i, service in enumerate(['deepgram', 'twilio', 'argil.ai', 'heygen', 'descript'][:3]):
                    page = await browsers[i+2][1].new_page()
                    await page.goto(f"https://{service}.com/dashboard", wait_until='networkidle')
                    while True:
                        credits = await page.query_selector("span.credits")
                        credit_value = await credits.inner_text() if credits else "Unknown"
                        await self.store_credit_status(service, credit_value)
                        await page.bring_to_front()
                        await asyncio.sleep(random.uniform(180, 300))  # Refresh every 3-5 minutes
                
                for browser, _ in browsers:
                    await browser.close()

    async def store_gemini_results(self, type, data):
        async with self.session_maker() as session:
            await session.execute(
                "INSERT INTO gemini_results (type, data, timestamp) VALUES (:type, :data, :timestamp)",
                {"type": type, "data": data, "timestamp": int(time.time())}
            )
            await session.commit()

    async def store_credit_status(self, service, credits):
        async with self.session_maker() as session:
            await session.execute(
                "INSERT INTO service_credits (service, credits, timestamp) VALUES (:service, :credits, :timestamp)",
                {"service": service, "credits": credits, "timestamp": int(time.time())}
            )
            await session.commit()

    async def check_credit_status(self, service, account_sid=None, auth_token=None):
        """Check credit status via API if available, else optimized browser check."""
        if service == 'twilio' and account_sid and auth_token:
            try:
                client = TwilioClient(account_sid, auth_token)
                balance = client.api.balance.fetch().balance
                logger.info(f"Twilio API credit check: ${balance}")
                return float(balance)
            except Exception as e:
                logger.error(f"Twilio API check failed: {e}")
                return await self._browser_check_credit(service)
        else:
            return await self._browser_check_credit(service)

    async def _browser_check_credit(self, service):
        """Optimized browser-based credit check using headless mode."""
        async with self.browser_semaphore:
            proxy = await self.get_next_proxy()
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, proxy={
                    'server': f"http://{proxy['server']}",
                    'username': proxy['username'],
                    'password': proxy['password']
                })
                context = await browser.new_context()
                page = await context.new_page()
                try:
                    await page.goto(f"https://{service}.com/dashboard", wait_until='networkidle')
                    credits = await page.query_selector("span.credits")
                    credit_value = await credits.inner_text() if credits else "Unknown"
                    logger.info(f"Browser credit check for {service}: {credit_value}")
                    return credit_value if credit_value != "Unknown" else 0.0
                except Exception as e:
                    logger.error(f"Browser check failed for {service}: {e}")
                    return 0.0
                finally:
                    await browser.close()

    async def run(self):
        """Run the BrowsingAgent to monitor credits and create accounts as needed."""
        services = ['deepgram', 'twilio', 'argil.ai', 'heygen', 'descript']  # Core services for your agency
        num_browsers = min(5, len(services))  # Cap at 5 browsers to respect VPS limits
        browser_tasks = []
        for i in range(num_browsers):
            browser_services = services[i::num_browsers]  # Evenly distribute services
            task = asyncio.create_task(self.monitor_credits(browser_services))
            browser_tasks.append(task)
        await asyncio.gather(*browser_tasks)

    async def process_task(self, task):
        """Process a task with error handling."""
        service = task['service']
        try:
            result = await self.create_account(service)
            await self.orchestrator.report_account_created(service, result)
            self._update_provider_success(service, True)
        except Exception as e:
            logger.error(f"Task failed for {service}: {str(e)}")
            self._update_provider_success(service, False)
            await self.orchestrator.report_error(service, str(e))
        finally:
            self.task_queue.task_done()

    def _update_provider_success(self, service, success):
        """Update provider success rates."""
        if 'email' in service.lower():
            provider = self.email_providers[0]
            rate = self.provider_success_rates['email'][provider]
            self.provider_success_rates['email'][provider] = min(max(rate + (0.05 if success else -0.05), 0), 1)
        elif 'sms' in service.lower():
            provider = self.sms_providers[0]
            rate = self.provider_success_rates['sms'][provider]
            self.provider_success_rates['sms'][provider] = min(max(rate + (0.05 if success else -0.05), 0), 1)

    async def get_insights(self):
        return {"provider_success_rates": self.provider_success_rates}  

# Example usage (not part of the agent itself):
# async def main():
#     from sqlalchemy.ext.asyncio import create_async_engine
#     engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
#     session_maker = async_sessionmaker(engine, expire_on_commit=False)
#     config = {}
#     orchestrator = Orchestrator()  # Assume this exists
#     agent = BrowsingAgent(session_maker, config, orchestrator)
#     await agent.task_queue.put({'service': 'deepgram'})
#     await agent.run()
#
# asyncio.run(main())