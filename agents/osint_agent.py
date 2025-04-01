
import asyncio
import logging
import random
import os
import json
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from utils.database import encrypt_data, decrypt_data
from models import OSINTData, Client, Lead
from openai import AsyncOpenAI as AsyncDeepSeekClient
import google.generativeai as genai
from web_ui import launch_browser, close_browser, navigate, find_element, click, get_text, wait_for_element
from bs4 import BeautifulSoup
import requests
from theHarvester import theHarvester
import exiftool
import photon
import sherlock
import maltego
import shodan
import reconng
import spiderfoot
import psutil
from fake_useragent import UserAgent
import aiohttp
from proxy_manager import ProxyManager
from fingerprint_generator import FingerprintGenerator
import subprocess
from ai_element_detector import AIElementDetector

# Configure advanced logging for genius-level debugging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class OSINTAgent:
    """A genius-level AI agent for Open-Source Intelligence (OSINT) tasks, designed to collect, analyze, and visualize publicly available data with unmatched precision and efficiency."""

    def __init__(self, session_maker, config, orchestrator, clients_models):
        self.session_maker = session_maker
        self.config = config
        self.orchestrator = orchestrator
        self.clients_models = clients_models
        self.think_tool = orchestrator.agents['think']
        self.task_queue = asyncio.Queue()
        self.max_concurrency = 10

        # OSINT tool configurations
        self.shodan_api_key = os.getenv("SHODAN_API_KEY")
        self.maltego_config = {"username": os.getenv("MALTEGO_USERNAME"), "password": os.getenv("MALTEGO_PASSWORD")}
        self.spiderfoot_config = {"api_key": os.getenv("SPIDERFOOT_API_KEY")}

        # Browser automation config
        self.proxy_config = {
            'username': os.getenv("SMARTPROXY_USERNAME"),
            'password': os.getenv("SMARTPROXY_PASSWORD"),
            'server': 'dc.us.smartproxy.com:10000'
        }

        # Genius-level meta-prompt
        self.meta_prompt = """
        You are a genius OSINTAgent tasked with collecting all available data for an AI agency targeting $6,000 in 24 hours and $100M in 9 months. Aggressively acquire data from all public sources, including grey areas, using advanced tools (Google Dorking, social scraping, dark pools). Respect legal boundaries, flag sensitive data, and anticipate future challenges. Learn from data patterns, collaborate with other agents, and optimize for actionable insights.
        """

        # Tool success rates with aggressive additions
        self.tool_success_rates = {
            'theHarvester': 1.0, 'ExifTool': 1.0, 'Photon': 1.0, 'Sherlock': 1.0,
            'Maltego': 1.0, 'Shodan': 1.0, 'Recon-ng': 1.0, 'SpiderFoot': 1.0,
            'GoogleDorking': 1.0, 'SocialMediaScraping': 1.0, 'DarkPoolSearch': 1.0
        }

        self.proxy_manager = ProxyManager(self.proxy_config)
        self.fingerprint_generator = FingerprintGenerator()
        self.ai_element_detector = AIElementDetector(clients_models)


    def get_allowed_concurrency(self):
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        concurrency_factor = 1 - (cpu_usage / 100) * 0.5 - (memory_usage / 100) * 0.5
        allowed = max(1, int(self.max_concurrency * concurrency_factor))
        return allowed  # Conservative capacity
        
    async def update_tool_success(self, tool, success):
        """Update tool success rate based on outcome."""
        rate = self.tool_success_rates.get(tool, 1.0)
        self.tool_success_rates[tool] = min(max(rate + (0.05 if success else -0.05), 0), 1)
        logger.info(f"Updated success rate for {tool}: {self.tool_success_rates[tool]}")

    async def select_best_tools(self, target):
        """Select top 3 tools based on historical success."""
        return sorted(self.tool_success_rates, key=self.tool_success_rates.get, reverse=True)[:3]
        
    async def send_to_research_agent(self, data_id):
        """Send analyzed OSINT data to ResearchAgent for UGC and business insights."""
        research_agent = self.orchestrator.agents.get('research')
        if research_agent:
            await research_agent.process_osint_data(data_id)
            logger.info(f"Sent data ID {data_id} to ResearchAgent")
        else:
            logger.warning("ResearchAgent not found in orchestrator")

    async def stealth_request(self, url):
        """Make a stealthy HTTP request with anti-detection measures."""
        ua = UserAgent()
        headers = {'User-Agent': ua.random, 'Accept-Language': 'en-US,en;q=0.9'}
        proxy = self.proxy_config
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, proxy=proxy['server'], auth=aiohttp.BasicAuth(proxy['username'], proxy['password'])) as response:
                await asyncio.sleep(random.uniform(0.5, 2))  # Human-like delay
                return await response.text()

    async def run(self):
        """Execute the OSINTAgent with relentless efficiency and brilliance."""
        while True:
            max_tasks = await self.get_available_concurrency()
            tasks = []
            for _ in range(min(max_tasks, self.task_queue.qsize())):
                task = await self.task_queue.get()
                tasks.append(self.process_task(task))
            await asyncio.gather(*tasks)

    async def process_task(self, task):
        """Process a single task with robust error handling."""
        task_type = task['type']
        try:
            if task_type == "collect_data":
                await self.collect_data(task['target'], task['tools'])
            elif task_type == "analyze_data":
                await self.analyze_data(task['data_id'])
            elif task_type == "visualize_data":
                await self.visualize_data(task['data_id'])
            else:
                logger.error(f"Unknown task type: {task_type} in task: {task}")
        except Exception as e:
            error_msg = f"Task {task_type} failed for {task.get('target', 'unknown')} (ID: {task.get('data_id', 'N/A')}): {e}"
            logger.error(error_msg)
            await self.orchestrator.report_error("OSINTAgent", error_msg)
        finally:
            self.task_queue.task_done()

    async def request_osint_task(self, task_type, **kwargs):
        """Queue an OSINT task with strategic intent."""
        await self.task_queue.put({'type': task_type, **kwargs})
        logger.info(f"OSINT task queued: {task_type} with args {kwargs}")

    async def collect_data(self, target, tools=None):
        selected_tools = await self.select_best_tools(target)
        aggressive_tools = ['GoogleDorking', 'SocialMediaScraping', 'DarkPoolSearch']
        tools_to_run = list(set(selected_tools + aggressive_tools))
        if tools:
            tools_to_run = list(set(tools + aggressive_tools))

        data = {}
        async def run_tool(tool):
            try:
                if tool == 'GoogleDorking':
                    data[tool] = await self.run_google_dorking(target)
                elif tool == 'SocialMediaScraping':
                    data[tool] = await self.run_social_scraping(target)
                elif tool == 'DarkPoolSearch':
                    data[tool] = await self.run_dark_pool_search(target)
                elif hasattr(self, f"run_{tool.lower()}"):
                    data[tool] = await self.run_tool_with_retry(getattr(self, f"run_{tool.lower()}"), target)
                await self.update_tool_success(tool, bool(data[tool]))
            except Exception as e:
                logger.error(f"Tool {tool} failed: {e}")
                data[tool] = {"error": str(e)}
                await self.update_tool_success(tool, False)

        await asyncio.gather(*(run_tool(tool) for tool in tools_to_run))

        async with self.session_maker() as session:
            osint_data = OSINTData(
                target=target,
                tools_used=tools_to_run,
                raw_data=json.dumps(data),
                timestamp=datetime.utcnow()
            )
            session.add(osint_data)
            await session.commit()
            await session.refresh(osint_data)
            logger.info(f"Collected data for {target} with ID {osint_data.id}")
            return osint_data.id

        async def run_single_tool(tool):
            # async with semaphore: # Uncomment if limiting concurrency per tool
                tool_data = {}
                try:
                    # Handle new aggressive tools first
                    if tool == 'GoogleDorking':
                        tool_data = await self.run_google_dorking(target)
                    elif tool == 'SocialMediaScraping':
                        tool_data = await self.run_social_scraping(target)
                    elif tool == 'DarkPoolSearch':
                        tool_data = await self.run_dark_pool_search(target)
                    # Handle existing tools via retry wrapper and dynamic getattr
                    elif hasattr(self, f"run_{tool.lower()}"):
                         tool_func = getattr(self, f"run_{tool.lower()}")
                         # Pass target or file path based on tool needs - requires more context
                         # Assuming most take 'target' for now
                         if tool == "ExifTool": # ExifTool needs a path, not target string
                              logger.warning(f"Skipping ExifTool in generic collect_data for target '{target}'. Needs file path.")
                              tool_data = {} # Cannot run on target string
                         else:
                              tool_data = await self.run_tool_with_retry(tool_func, target)
                    else:
                         logger.warning(f"No run method found for tool: {tool}")
                         tool_data = {}

                    data[tool] = tool_data # Store data even if empty
                    await self.update_tool_success(tool, bool(tool_data)) # Update success based on if data was returned
                except Exception as e:
                    logger.error(f"Tool {tool} threw exception for {target}: {e}", exc_info=True)
                    data[tool] = {"error": str(e)} # Store error
                    await self.update_tool_success(tool, False)

        tasks = [run_single_tool(tool) for tool in tools_to_run]
        await asyncio.gather(*tasks)

        # Store collected data in the database
        async with self.session_maker() as session:
            osint_data = OSINTData(
                target=target,
                tools_used=tools_to_run, # Log all tools attempted
                raw_data=json.dumps(data), # Store results/errors
                timestamp=datetime.utcnow()
            )
            session.add(osint_data)
            await session.commit()
            await session.refresh(osint_data)
            logger.info(f"Finished data collection for {target} with ID {osint_data.id}. Data keys: {list(data.keys())}")
            return osint_data.id

    async def run_tool_with_retry(self, tool_func, *args, max_retries=3, delay=5):
        """Run a tool with retry logic for resilience."""
        for attempt in range(max_retries):
            try:
                return await tool_func(*args)
            except Exception as e:
                logger.warning(f"Tool {tool_func.__name__} failed on attempt {attempt + 1}: {e}")
                if attempt + 1 == max_retries:
                    logger.error(f"Tool {tool_func.__name__} failed after {max_retries} attempts.")
                    return {}
                await asyncio.sleep(delay)

    async def run_theHarvester(self, target):
        """Run theHarvester to collect emails, IPs, and subdomains."""
        try:
            harvester = theHarvester.TheHarvester()
            results = harvester.search(target, ['bing', 'google', 'linkedin'])
            return results
        except Exception as e:
            logger.error(f"theHarvester failed for {target}: {e}")
            return {}

    async def run_exiftool(self, file_path):
        """Run ExifTool to extract metadata from a file."""
        try:
            with exiftool.ExifTool() as et:
                metadata = et.get_metadata(file_path)
            return metadata
        except Exception as e:
            logger.error(f"ExifTool failed for {file_path}: {e}")
            return {}

    async def run_photon(self, target_url):
        try:
            html = await self.stealth_request(target_url)
            photon_instance = photon.Photon(target_url)
            photon_instance.crawl(html=html)  # Pass pre-fetched HTML if supported
            return photon_instance.get_results()
        except Exception as e:
            logger.error(f"Photon failed for {target_url}: {e}")
            return {}

    async def run_sherlock(self, username):
        """Run Sherlock to track a username across platforms."""
        try:
            results = sherlock.sherlock(username)
            return results
        except Exception as e:
            logger.error(f"Sherlock failed for {username}: {e}")
            return {}

    async def run_maltego(self, target):
        """Run Maltego with real API integration."""
        try:
            from maltego_trx import MaltegoTransform  # Requires maltego-trx library
            transform = MaltegoTransform()
            transform.addEntity("maltego.Domain", target)
            response = transform.returnOutput()
            await self.update_tool_success('Maltego', True)
            return response
        except Exception as e:
            logger.error(f"Maltego failed for {target}: {e}")
            await self.update_tool_success('Maltego', False)
            return {}

    async def run_shodan(self, query):
        """Run Shodan to search for internet-connected devices."""
        try:
            shodan_instance = shodan.Shodan(self.shodan_api_key)
            results = shodan_instance.search(query)
            return results
        except Exception as e:
            logger.error(f"Shodan failed for query {query}: {e}")
            return {}

    async def run_reconng(self, target):
        """Run Recon-ng to perform reconnaissance on the target."""
        try:
            # Define the Recon-ng command to run the 'recon/domains-hosts/google_site_web' module
            command = [
                'recon-ng',
                '-m', 'recon/domains-hosts/google_site_web',
                '-o', f'TARGET={target}',
                '-x'
            ]
            # Execute the command asynchronously and capture output
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                results = stdout.decode('utf-8')
                await self.update_tool_success('Recon-ng', True)
                return {'reconng_output': results}
            else:
                logger.error(f"Recon-ng failed with error: {stderr.decode('utf-8')}")
                await self.update_tool_success('Recon-ng', False)
                return {}
        except Exception as e:
            logger.error(f"Recon-ng failed for {target}: {e}")
            await self.update_tool_success('Recon-ng', False)
            return {}

    async def run_spiderfoot(self, target):
        """Run SpiderFoot to perform an automated OSINT scan on the target."""
        try:
            api_key = self.spiderfoot_config['api_key']  # Set via env var SPIDERFOOT_API_KEY
            spiderfoot_url = "http://localhost:5001"  # Assumes SpiderFoot runs locally
            scan_name = f"OSINT_scan_{target}_{int(time.time())}"
            # Start a new scan
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{spiderfoot_url}/startscan",
                    json={
                        'apikey': api_key,
                        'scanname': scan_name,
                        'scantarget': target,
                        'modulelist': ['sfp_dns', 'sfp_portscan_tcp']  # Free-tier modules
                    }
                ) as response:
                    if response.status == 200:
                        scan_id = (await response.json())['id']
                    else:
                        logger.error(f"Failed to start SpiderFoot scan: {await response.text()}")
                        await self.update_tool_success('SpiderFoot', False)
                        return {}
                
                # Poll for scan completion
                while True:
                    async with session.get(
                        f"{spiderfoot_url}/scanstatus",
                        params={'scanid': scan_id, 'apikey': api_key}
                    ) as status_response:
                        status = await status_response.json()
                        if status['status'] == 'FINISHED':
                            break
                        await asyncio.sleep(10)  # Wait 10s between polls
                
                # Retrieve scan results
                async with session.get(
                    f"{spiderfoot_url}/scanresults",
                    params={'scanid': scan_id, 'apikey': api_key}
                ) as results_response:
                    results = await results_response.json()
                    await self.update_tool_success('SpiderFoot', True)
                    return results
        except Exception as e:
            logger.error(f"SpiderFoot failed for {target}: {e}")
            await self.update_tool_success('SpiderFoot', False)
            return {}
        
    async def run_google_dorking(self, target):
        # WARNING: Ensure compliance with Google's ToS and local laws
        dorks = [
            f"site:{target} filetype:pdf",
            f"site:{target} inurl:(login | admin)",
            f"site:{target} -inurl:(signup | login)"
        ]
        results = {}
        async with aiohttp.ClientSession() as session:
            for dork in dorks:
                try:
                    async with session.get(f"https://www.google.com/search?q={dork}", headers={'User-Agent': fake.user_agent()}) as response:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        links = [a['href'] for a in soup.find_all('a', href=True) if '/url?q=' in a['href']]
                        results[dork] = links
                except Exception as e:
                    logger.error(f"Google Dorking failed for {dork}: {e}")
        return results

    async def run_social_scraping(self, target):
        # WARNING: Adhere to platform ToS and privacy laws (e.g., GDPR, CCPA)
        platforms = ['twitter.com', 'linkedin.com', 'facebook.com']
        data = {}
        async with aiohttp.ClientSession() as session:
            for platform in platforms:
                try:
                    url = f"https://{platform}/search?q={target}"
                    async with session.get(url, headers={'User-Agent': fake.user_agent()}) as response:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        profiles = [a['href'] for a in soup.find_all('a', href=True) if '/profile' in a['href'] or '/user' in a['href']]
                        data[platform] = profiles
                except Exception as e:
                    logger.error(f"Social scraping failed for {platform}: {e}")
        return data

    async def run_dark_pool_search(self, target):
        # WARNING: Requires I2P or similar dark pool access; ensure legal compliance
        try:
            async with aiohttp.ClientSession() as session:
                # Simulated dark pool endpoint (replace with real I2P/Freenet integration)
                async with session.get(f"http://darkpool.example.com/search?q={target}", headers={'User-Agent': fake.user_agent()}) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"dark_web_data": data}
                    else:
                        logger.warning(f"Dark pool search failed: {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"Dark pool search failed for {target}: {e}")
            return {}
        
    async def run_gemini_deep_search(self, target):
        async with async_playwright() as p:
            proxy = self.proxy_config
            browser = await p.chromium.launch(headless=True, proxy={
                'server': proxy['server'],
                'username': proxy['username'],
                'password': proxy['password']
            })
            page = await browser.new_page()
            await page.goto("https://gemini.google.com", wait_until='networkidle')
            await page.click("button:contains('Deep Search')")
            await page.fill("input#search", f"OSINT {target}")
            await page.click("button[type='submit']")
            await page.wait_for_selector("div.results", timeout=30000)
            results = await page.content()
            await browser.close()
            return {"gemini_deep_search": results}

    async def minimal_browsing(self, url, actions):
        browser_config = await self.generate_browser_config()
        browser = await launch_browser(
            proxy=browser_config['proxy'],
            headless=True,
            user_agent=browser_config['user_agent'],
            viewport=browser_config['viewport']
        )
        try:
            await self.stealth_navigate(browser, url)
            collected_data = {'url': url, 'actions': []}

            # Use Gemini 2.0 Pro for navigation decisions
            prompt = f"""
            {self.meta_prompt}
            Given URL {url} and actions {json.dumps(actions)}, suggest optimal navigation steps.
            Return JSON with 'steps' (list of actions).
            """
            response = await self.gemini_pro.generate_content(prompt)
            gemini_steps = json.loads(response.text).get('steps', actions)

            for action in gemini_steps:
                if action['type'] == 'click':
                    element = await self.smart_find_element(browser, action['selector'])
                    await self.humanized_click(browser, element)
                    collected_data['actions'].append({'type': 'click', 'selector': action['selector']})
                elif action['type'] == 'get_text':
                    element = await self.smart_find_element(browser, action['selector'])
                    text = await self.extract_text_with_context(browser, element)
                    collected_data['actions'].append({'type': 'get_text', 'selector': action['selector'], 'text': text})
                    logger.info(f"Extracted contextual text from {url}: {text}")
                elif action['type'] == 'screenshot':
                    screenshot_path = await self.capture_intelligent_screenshot(browser, action.get('region'))
                    collected_data['actions'].append({'type': 'screenshot', 'path': screenshot_path})
                
                # Genius feature: Adaptive human-like delays based on page complexity
                page_complexity = await self.analyze_page_complexity(browser)
                await asyncio.sleep(random.uniform(1, 3) * page_complexity)

            logger.info(f"Genius-level minimal browsing completed for {url}")
            return collected_data
        except Exception as e:
            logger.error(f"Minimal browsing failed for {url}: {e}")
            await self.orchestrator.report_error("OSINTAgent", f"Browser failure at {url}: {e}")
            return None
        finally:
            await close_browser(browser)

    async def generate_browser_config(self):
        """Generate a dynamic browser configuration for stealth and evasion."""
        # Genius feature: Rotate proxies and spoof fingerprints dynamically
        proxy = await self.proxy_manager.get_fresh_proxy()
        user_agent = await self.fingerprint_generator.get_random_user_agent()
        viewport = random.choice([
            {'width': 1920, 'height': 1080},
            {'width': 1366, 'height': 768},
            {'width': 1440, 'height': 900}
        ])
        return {
            'proxy': proxy,
            'user_agent': user_agent,
            'viewport': viewport
        }

    async def stealth_navigate(self, browser, url):
        """Navigate to a URL with anti-bot evasion techniques."""
        # Genius feature: Mimic human navigation patterns
        headers = {
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': await self.generate_referer(url),
            'DNT': '1'  # Do Not Track for added realism
        }
        await navigate(browser, url, headers=headers)
        # Random scroll to simulate human behavior
        await browser.execute_script("window.scrollTo(0, document.body.scrollHeight * Math.random());")
        await asyncio.sleep(random.uniform(0.5, 1.5))

    async def smart_find_element(self, browser, selector):
        """Locate elements with resilience and intelligence."""
        try:
            element = await find_element(browser, selector, timeout=10)
            if not element:
                # Genius feature: Fallback to AI-driven element detection
                logger.warning(f"Selector {selector} not found, attempting AI detection")
                element = await self.ai_element_detector.find_element(browser, selector)
            return element
        except Exception as e:
            logger.error(f"Element detection failed for {selector}: {e}")
            raise

    async def humanized_click(self, browser, element):
        """Simulate a human-like click with cursor movement."""
        # Genius feature: Simulate mouse movement before clicking
        await browser.execute_script("""
            function simulateMouseMove(element) {
                const rect = element.getBoundingClientRect();
                const event = new MouseEvent('mousemove', {
                    bubbles: true,
                    clientX: rect.left + rect.width / 2,
                    clientY: rect.top + rect.height / 2
                });
                element.dispatchEvent(event);
            }
            simulateMouseMove(arguments[0]);
        """, element)
        await asyncio.sleep(random.uniform(0.1, 0.5))
        await click(browser, element)

    async def extract_text_with_context(self, browser, element):
        """Extract text with surrounding context for richer data."""
        text = await get_text(browser, element)
        # Genius feature: Include parent and sibling context
        context_script = """
            let el = arguments[0];
            let parent = el.parentElement ? el.parentElement.innerText : '';
            let siblings = Array.from(el.parentElement.children)
                .filter(child => child !== el)
                .map(child => child.innerText);
            return { main: el.innerText, parent: parent, siblings: siblings };
        """
        context = await browser.evaluate_script(context_script, element)
        return context

    async def capture_intelligent_screenshot(self, browser, region=None):
        """Capture a screenshot with intelligent region detection."""
        if region:
            screenshot = await browser.screenshot(region=region)
        else:
            # Genius feature: Auto-detect key content areas
            key_area = await self.detect_key_content_area(browser)
            screenshot = await browser.screenshot(region=key_area)
        path = f"screenshots/{uuid.uuid4()}.png"
        with open(path, 'wb') as f:
            f.write(screenshot)
        return path
    
    async def detect_key_content_area(self, browser):
        """Detect key content area using AIElementDetector."""
        try:
            key_element = await self.ai_element_detector.find_key_element(browser)
            if key_element:
                # Assuming find_key_element returns an element with a bounding_box method
                return await key_element.bounding_box()
            else:
                logger.warning("No key content area detected, using full page.")
                return await browser.viewport_size()
        except Exception as e:
            logger.error(f"Failed to detect key content area: {e}")
            return await browser.viewport_size()

    async def analyze_page_complexity(self, browser):
        """Analyze page complexity to adjust interaction timing."""
        complexity_script = """
            return {
                element_count: document.getElementsByTagName('*').length,
                script_count: document.getElementsByTagName('script').length,
                interactive_elements: document.querySelectorAll('button, a, input').length
            };
        """
        metrics = await browser.evaluate_script(complexity_script)
        # Normalize complexity to a factor between 1 and 2
        complexity = min(2, max(1, (metrics['element_count'] / 1000 + metrics['script_count'] / 50 + metrics['interactive_elements'] / 100)))
        return complexity
    
    async def analyze_page_complexity(self, browser):
        """Analyze page complexity to adjust interaction timing dynamically."""
        complexity_script = """
            return {
                element_count: document.getElementsByTagName('*').length,
                script_count: document.getElementsByTagName('script').length,
                interactive_elements: document.querySelectorAll('button, a, input').length
            };
        """
        metrics = await browser.evaluate_script(complexity_script)
        # Normalize complexity to a factor between 1 and 2
        complexity = min(2, max(1, (metrics['element_count'] / 1000 + metrics['script_count'] / 50 + metrics['interactive_elements'] / 100)))
        return complexity

    async def minimal_browsing(self, url, browser):
        """Perform minimal browsing with adaptive timing and error resilience."""
        try:
            complexity = await self.analyze_page_complexity(browser)
            await browser.goto(url, timeout=30000, wait_until="networkidle2")
            # Genius feature: Dynamic wait based on page complexity
            await asyncio.sleep(2 * complexity)

            # Validate browsing outcome with ThinkTool
            validation_prompt = f"""
            {self.meta_prompt}
            Validate the browsing outcome for URL: {url}.
            Current page metrics: {json.dumps({'complexity': complexity})}.
            Return a JSON response with:
            - 'success' (bool): Whether the page loaded correctly.
            - 'recommendations' (list): Suggestions for next steps.
            """
            validation_response = await self.think_tool.validate(validation_prompt)
            validation = json.loads(validation_response)
            if not validation.get('success', False):
                logger.warning(f"Browsing validation failed for {url}: {validation.get('recommendations', [])}")
                return False
            logger.info(f"Minimal browsing completed for {url} with complexity {complexity}")
            return True
        except Exception as e:
            logger.error(f"Browsing failed for {url}: {e}")
            return False

    async def analyze_data(self, data_id):
        """Analyze collected OSINT data with DeepSeek-R1 for genius-level insights, with robust error handling."""
        async with self.session_maker() as session:
            # Fetch the OSINT data from the database
            osint_data = await session.get(OSINTData, data_id)
            if not osint_data:
                logger.error(f"OSINT data ID {data_id} not found in database.")
                return

            # Parse raw data with error handling
            try:
                raw_data = json.loads(osint_data.raw_data)
            except json.JSONDecodeError as e:
                logger.error(f"Malformed raw_data for ID {data_id}: {e}")
                await self.orchestrator.report_error("OSINTAgent", f"JSON decode error in raw_data: {e}")
                return

            # Check if raw data is empty
            if not raw_data:
                logger.warning(f"No raw data available for analysis (ID: {data_id})")
                return

            # Initialize results dictionary
            analysis_results = {}

            # Analyze data from each tool with individual error handling
            for tool, data in raw_data.items():
                try:
                    if tool == "theHarvester":
                        analysis_results['emails'] = await self.analyze_emails(data.get('emails', []))
                    elif tool == "Sherlock":
                        analysis_results['social_presence'] = await self.analyze_social_presence(data)
                    elif tool == "Shodan":
                        analysis_results['device_vulnerabilities'] = await self.analyze_shodan_data(data)
                    else:
                        logger.warning(f"Unsupported tool {tool} in raw_data for ID {data_id}")
                except Exception as e:
                    logger.error(f"Analysis failed for tool {tool} (ID: {data_id}): {e}")
                    analysis_results[tool] = {"error": str(e)}

            # Use DeepSeek-R1 for advanced analysis
            prompt = f"""
            Analyze OSINT data: {json.dumps(raw_data, indent=2)}
            Segment data by type (emails, IPs, social, etc.), identify patterns, and predict trends.
            Return JSON with 'segments', 'patterns', 'predictions'.
            """
            try:
                response = await self.deepseek_client.chat.completions.create(
                    model="deepseek-r1",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    timeout=30  # Add timeout to handle network delays
                )
                analysis = json.loads(response.choices[0].message.content)
                if not isinstance(analysis, dict):
                    raise ValueError("Invalid analysis response: not a dictionary")
            except Exception as e:
                logger.error(f"DeepSeek-R1 analysis failed for ID {data_id}: {e}")
                analysis = {"error": str(e)}

            # Update learning model only if analysis is successful
            if 'error' not in analysis:
                await self.update_learning_model(analysis, tool, data_quality=1.0)
            else:
                await self.update_learning_model({}, tool, data_quality=0.0)

            # Store results and commit
            osint_data.analysis_results = json.dumps(analysis)
            await session.commit()
            await self.send_to_research_agent(data_id)
            logger.info(f"Data analysis completed for ID {data_id}")

    async def analyze_emails(self, emails):
        """Extract patterns and predict lead potential from email data."""
        patterns = {}
        lead_scores = {}
        for email in emails:
            domain = email.split('@')[-1]
            patterns[domain] = patterns.get(domain, 0) + 1
            # Genius feature: Predictive lead scoring
            lead_scores[email] = await self.predict_lead_potential(email)
        return {"patterns": patterns, "lead_scores": lead_scores}

    async def predict_lead_potential(self, email):
        """Predict lead potential using DeepSeek-R1."""
        prompt = f"Assess lead potential for email {email}. Return a score (0-100)."
        response = await self.deepseek_client.chat.completions.create(
            model="deepseek-r1",
            messages=[{"role": "user", "content": prompt}]
        )
        return float(response.choices[0].message.content)

    async def analyze_social_presence(self, sherlock_data):
        """Analyze social media for influence and engagement trends."""
        presence = {"platforms": [], "metrics": {}}
        for platform, status in sherlock_data.items():
            if status:
                presence['platforms'].append(platform)
                presence['metrics'][platform] = await self.fetch_platform_metrics(platform)
        return presence

    async def fetch_platform_metrics(self, platform):
        """Fetch real-time metrics for a social platform (placeholder)."""
        # Hypothetical API call to social media analytics
        return {"engagement": 0.75, "reach": 10000}

    async def analyze_shodan_data(self, shodan_data):
        """Identify vulnerabilities and assess risk levels."""
        vulnerabilities = []
        risk_score = 0
        for device in shodan_data.get('matches', []):
            if 'vulns' in device:
                vulnerabilities.extend(device['vulns'])
                risk_score += len(device['vulns']) * 10  # Weighted risk
        return {"vulnerabilities": list(set(vulnerabilities)), "risk_score": min(risk_score, 100)}
    
    async def run_reconng(self, target):
        """Run Recon-ng for free OSINT data collection."""
        try:
            import subprocess
            result = subprocess.run(
                ["recon-ng", "-m", "recon/domains-hosts/hackertarget", "-o", f"TARGET={target}", "-e"],
                capture_output=True, text=True, timeout=300
            )
            output = result.stdout
            logger.info(f"Recon-ng output for {target}: {output}")
            return {"reconng": output}
        except Exception as e:
            logger.error(f"Recon-ng failed for {target}: {e}")
            return {"error": str(e)}

    async def run_osint_workflow(self, target, tools):
        logger.info(f"Starting OSINT workflow for target: {target} with tools: {tools}")
        data_id = await self.collect_data(target, tools)
        if data_id:
            logger.info(f"Data collected for {target}, proceeding to analysis (ID: {data_id})")
            await self.analyze_data(data_id)
            if 'reconng' in tools:
                reconng_data = await self.run_reconng(target)
                async with self.session_maker() as session:
                    osint_data = await session.get(OSINTData, data_id)
                    raw_data = json.loads(osint_data.raw_data)
                    raw_data.update(reconng_data)
                    osint_data.raw_data = json.dumps(raw_data)
                    await session.commit()
            logger.info(f"Analysis completed for {target}, proceeding to visualization")
            await self.visualize_data(data_id)
            logger.info(f"Visualization completed for {target}, workflow finished")
        else:
            logger.warning(f"Data collection failed for {target}, workflow aborted")

    async def run_maltego_visualization(self, raw_data, analysis_results):
        """Generate a Maltego graph from raw OSINT data and analysis results."""
        try:
            from maltego_trx import MaltegoTransform
            transform = MaltegoTransform()

            # Add entities from raw_data
            for tool, data in raw_data.items():
                if tool == "theHarvester":
                    for email in data.get('emails', []):
                        entity = transform.addEntity("maltego.EmailAddress", email)
                        entity.addProperty("source", "Source", "theHarvester")
                    for ip in data.get('ips', []):
                        entity = transform.addEntity("maltego.IPv4Address", ip)
                        entity.addProperty("source", "Source", "theHarvester")
                elif tool == "Shodan":
                    for match in data.get('matches', []):
                        ip = match.get('ip_str', '')
                        if ip:
                            entity = transform.addEntity("maltego.IPv4Address", ip)
                            entity.addProperty("port", "Port", str(match.get('port', '')))
                elif tool == "Sherlock":
                    for platform, info in data.items():
                        if info.get('exists'):
                            entity = transform.addEntity("maltego.WebSite", platform)
                            entity.addProperty("url", "URL", info.get('url', ''))

            # Add insights from analysis_results
            for segment, details in analysis_results.get('segments', {}).items():
                if segment == "emails":
                    for email, score in details.get('lead_scores', {}).items():
                        entity = transform.addEntity("maltego.EmailAddress", email)
                        entity.addProperty("lead_score", "Lead Score", str(score))
                elif segment == "social_presence":
                    for platform in details.get('platforms', []):
                        entity = transform.addEntity("maltego.WebSite", platform)
                        entity.addProperty("metrics", "Metrics", str(details['metrics'].get(platform, {})))

            graph = await self.run_maltego_visualization(raw_data, analysis_results)
            logger.info("Maltego visualization generated successfully")
            return graph
        except ImportError as e:
            logger.error(f"Maltego library not installed: {e}")
            return {"error": "Maltego library missing"}
        except Exception as e:
            logger.error(f"Maltego visualization failed: {e}")
            return {"error": str(e)}

    async def run_maltego(self, target):
        """Run Maltego with real API integration."""
        try:
            from maltego_trx import MaltegoTransform  # Requires maltego-trx library
            transform = MaltegoTransform()
            transform.addEntity("maltego.Domain", target)
            response = transform.returnOutput()
            await self.update_tool_success('Maltego', True)
            return response
        except Exception as e:
            logger.error(f"Maltego failed for {target}: {e}")
            await self.update_tool_success('Maltego', False)
            return {}

    async def save_visualization(self, graph):
        """Save visualization with metadata for traceability."""        
        path = f"visualizations/{uuid.uuid4()}.graphml"
        with open(path, 'w') as f:
            f.write(json.dumps(graph))
        return path

    async def update_learning_model(self, analysis_results, tool, data_quality):
        """Update tool success rates based on data quality and relevance."""
        insights_present = 1.0 if 'insights' in analysis_results else 0.0
        success_score = (data_quality + insights_present) / 2
        current_rate = self.tool_success_rates.get(tool, 1.0)
        learning_rate = 0.1  # Smooths updates over time
        new_rate = current_rate + learning_rate * (success_score - current_rate)
        self.tool_success_rates[tool] = min(max(new_rate, 0), 1)
        logger.info(f"Updated success rate for {tool}: {self.tool_success_rates[tool]}")

    async def run_shodan_web(self, target):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(f"https://www.shodan.io/search?query={target}", wait_until="networkidle")
            results = await page.content()
            await browser.close()
            return {"html": results}

    @retry(stop=stop_after_attempt(50), wait=wait_exponential(multiplier=1, min=5, max=60))
    async def run_osint_workflow(self, target, tools):
        logger.info(f"Starting OSINT workflow for target: {target} with tools: {tools}")
        data_id = await self.collect_data(target, tools)
        if data_id:
            logger.info(f"Data collected for {target}, proceeding to analysis (ID: {data_id})")
            await self.analyze_data(data_id)
            logger.info(f"Analysis completed for {target}, proceeding to visualization")
            await self.visualize_data(data_id)
            logger.info(f"Visualization completed for {target}, workflow finished")
        else:
            logger.warning(f"Data collection failed for {target}, workflow aborted")

    async def trigger_integrations(self, data_id):
        """Integrate with other agents like EmailAgent for actionable outcomes."""
        async with self.session_maker() as session:
            osint_data = await session.get(OSINTData, data_id)
            analysis_results = json.loads(osint_data.analysis_results or '{}')
            if 'emails' in analysis_results:
                await self.email_agent.send_campaign(analysis_results['emails'])

    async def get_insights(self):
        async with self.session_maker() as session:
            insights = await session.execute("SELECT analysis_results FROM osint_data ORDER BY timestamp DESC LIMIT 1")
            row = insights.fetchone()
            return json.loads(row[0]) if row else {}
