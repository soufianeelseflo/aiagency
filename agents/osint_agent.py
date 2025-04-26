import asyncio
import logging
import random
import os
import json
import time # Architect-Zero: Added import
import uuid # Architect-Zero: Added import
from datetime import datetime, timedelta, timezone # Architect-Zero: Use timezone.utc
from sqlalchemy.ext.asyncio import AsyncSession
# from utils.database import encrypt_data, decrypt_data # Architect-Zero: Assume robust implementation
from models import OSINTData, Client, Lead # Architect-Zero: Ensure models match usage
from openai import AsyncOpenAI as AsyncLLMClient # Architect-Zero: Generic name for LLM client
# import google.generativeai as genai # Architect-Zero: Use only if directly needed, prefer unified client if possible

# --- Web Scraping / HTTP Requests ---
import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent # For basic scraping politeness

# --- OSINT Tool Libraries/Wrappers (Ensure installed) ---
# Note: Some tools might be command-line, others libraries.
# from theHarvester import ??? # Architect-Zero: Need specific import from theHarvester package if used as lib
# import exiftool # Requires python-exiftool library and exiftool binary
# import photon # Requires photon library
# import sherlock # Requires sherlock library
# import maltego_trx # Requires maltego-trx library
# import shodan # Requires shodan library
# import ??? # recon-ng is typically CLI
# import ??? # spiderfoot is typically web API or CLI

# --- System & Async ---
import subprocess # For running CLI tools
import psutil
import tenacity
from typing import Optional, Dict, Any, List, Tuple
import sqlalchemy # Architect-Zero: Added for SQLAlchemyError handling

# Import base class and prompt
from .base_agent import GeniusAgentBase, KBInterface
from prompts.agent_meta_prompts import OSINT_AGENT_META_PROMPT
import re # Import re for filename sanitization

# Configure advanced logging
logger = logging.getLogger(__name__)
# Logging should be configured globally

# Configure dedicated operational logger (assuming setup in main/orchestrator)
# Ensure op_logger is accessible or passed if needed, or use standard logger
# op_logger = logging.getLogger('OperationalLog')


class OSINTAgent(GeniusAgentBase): # Renamed and inherited
    """
    OSINT Agent (Genius Level): Collects, analyzes, and exploits public data using
    advanced techniques and tools, integrating findings into the central Knowledge Base.
    """
    AGENT_NAME = "OSINTAgent"

    def __init__(self, session_maker: AsyncSession, orchestrator: Any, kb_interface: KBInterface, shodan_api_key: Optional[str], spiderfoot_api_key: Optional[str]):
        """Initializes the OSINTAgent.

        Args:
            session_maker: SQLAlchemy async session maker.
            orchestrator: The main Orchestrator instance.
            kb_interface: The interface for interacting with the Knowledge Base.
            shodan_api_key: The Shodan API Key (fetched from Vault, can be None).
            spiderfoot_api_key: The SpiderFoot API Key (fetched from Vault, can be None).
        """
        super().__init__(agent_name=self.AGENT_NAME, kb_interface=kb_interface)
        # self.config is inherited
        # self.kb_interface is inherited
        self.session_maker = session_maker # Keep for saving OSINTData records
        self.orchestrator = orchestrator # Keep for reporting/notifications/LLM access proxy
        self.think_tool = orchestrator.agents.get('think') # Keep for analysis LLM calls initially

        # Store passed-in secrets
        self._shodan_api_key = shodan_api_key
        self._spiderfoot_api_key = spiderfoot_api_key

        # --- Internal State Initialization ---
        self.internal_state['task_queue'] = asyncio.Queue()
        self.internal_state['max_concurrency'] = int(self.config.get("OSINT_MAX_CONCURRENT_TOOLS", 5))
        self.internal_state['tool_semaphore'] = asyncio.Semaphore(self.internal_state['max_concurrency'])
        self.internal_state['memory_cache'] = {} # Simple cache for insights (consider moving to KB)
        self.internal_state['meta_prompt'] = OSINT_AGENT_META_PROMPT

        # Tool Configuration (API Keys passed in, URL from config)
        # self.internal_state['shodan_api_key'] = self.config.get("SHODAN_API_KEY") # Use self._shodan_api_key
        # self.internal_state['spiderfoot_api_key'] = self.config.get("SPIDERFOOT_API_KEY") # Use self._spiderfoot_api_key
        self.internal_state['spiderfoot_url'] = self.config.get("SPIDERFOOT_URL", "http://localhost:5001") # URL is config

        # Feature Flags for Aggressive Tools
        self.internal_state['enable_google_dorking'] = bool(self.config.get("OSINT_ENABLE_GOOGLE_DORKING", False))
        self.internal_state['enable_social_scraping'] = bool(self.config.get("OSINT_ENABLE_SOCIAL_SCRAPING", False))
        self.internal_state['enable_dark_pool'] = bool(self.config.get("OSINT_ENABLE_DARK_POOL", False))
        self.internal_state['enable_webcam_search'] = bool(self.config.get("OSINT_ENABLE_WEBCAM_SEARCH", False))

        # Tool Success Rates (Simplified - move to KB later for persistence/shared learning)
        self.internal_state['tool_success_rates'] = {
            'theHarvester': 0.8, 'ExifTool': 0.7, 'Photon': 0.6, 'Sherlock': 0.8,
            'Maltego': 0.7, 'Shodan': 0.9, 'Reconng': 0.7, 'SpiderFoot': 0.8,
            'GoogleDorking': 0.5, 'SocialMediaScraping': 0.5,
            'DarkPoolSearch': 0.0, 'WebcamSearch': 0.0
        }

        # --- User Agent ---
        try:
            # UserAgent might cause issues in some environments, handle failure
            from fake_useragent import UserAgent
            self.user_agent_generator = UserAgent()
        except Exception as ua_err:
             self.logger.warning(f"{self.AGENT_NAME}: Failed to initialize fake_useragent ({ua_err}). Using default UA string.")
             self.user_agent_generator = None # Flag that generator is unavailable

        self.logger.info(f"{self.AGENT_NAME} (Genius Level) initialized. Max concurrent tools: {self.internal_state['max_concurrency']}. "
                    f"Aggressive tools enabled: Dorking={self.internal_state['enable_google_dorking']}, Social={self.internal_state['enable_social_scraping']}, "
                    f"DarkPool={self.internal_state['enable_dark_pool']}, Webcam={self.internal_state['enable_webcam_search']}")
        if self.internal_state['enable_dark_pool'] or self.internal_state['enable_webcam_search']:
             self.logger.critical(f"{self.AGENT_NAME}: SECURITY/LEGAL WARNING: Dark Pool or Webcam Search enabled. Ensure full compliance and risk assessment.")

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


    async def update_memory_cache(self, new_insights: Dict):
        """Update the agent's memory cache with new insights."""
        # Architect-Zero: Implement more sophisticated caching/merging if needed
        self.memory_cache.update(new_insights)
        logger.debug(f"{self.AGENT_NAME}: Updated memory cache.")

    async def update_tool_success(self, tool: str, success: bool):
        """Rudimentary update of tool success rate."""
        # Architect-Zero: This is very basic. Real learning requires feedback on data quality.
        rate = self.tool_success_rates.get(tool, 0.5)
        adjustment = 0.05 if success else -0.05
        self.tool_success_rates[tool] = min(max(rate + adjustment, 0.1), 1.0) # Keep between 0.1 and 1.0
        logger.debug(f"{self.AGENT_NAME}: Updated success rate for {tool}: {self.tool_success_rates[tool]:.2f}")

    async def select_best_tools(self, target: str, num_tools: int = 3) -> List[str]:
        """Select top N tools based on perceived success rate."""
        # Architect-Zero: Consider target type (domain, email, username) to influence tool selection.
        # Filter out disabled tools
        enabled_tools = {tool: rate for tool, rate in self.tool_success_rates.items() if self._is_tool_enabled(tool)}
        sorted_tools = sorted(enabled_tools, key=enabled_tools.get, reverse=True)
        logger.debug(f"{self.AGENT_NAME}: Selected tools for '{target}': {sorted_tools[:num_tools]}")
        return sorted_tools[:num_tools]

    def _is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled via config flags."""
        if tool_name == 'GoogleDorking': return self.enable_google_dorking
        if tool_name == 'SocialMediaScraping': return self.enable_social_scraping
        if tool_name == 'DarkPoolSearch': return self.enable_dark_pool
        if tool_name == 'WebcamSearch': return self.enable_webcam_search
        # Assume other standard tools are always enabled if configured (e.g., have API key)
        if tool_name == 'Shodan': return bool(self._shodan_api_key) # Use stored key
        if tool_name == 'SpiderFoot': return bool(self._spiderfoot_api_key) and bool(self.internal_state.get('spiderfoot_url')) # Use stored key and URL from state
        # Add checks for other tools requiring keys/config if necessary
        return True # Default to enabled for tools without specific flags/keys

    async def run(self):
        """Main run loop processing tasks from the internal asyncio Queue."""
        logger.info(f"{self.AGENT_NAME} run loop started, processing tasks from internal queue.")
        while True:
            try:
                task = await self.task_queue.get()
                task_type = task.get('type', 'Unknown')
                target = task.get('target', task.get('data_id', 'N/A')) # Get identifier for logging
                logger.info(f"{self.AGENT_NAME}: Dequeued task: {task_type} for '{target}'")
                await self.process_task(task)
                # task_done is called within process_task's finally block
            except asyncio.CancelledError:
                 logger.info(f"{self.AGENT_NAME}: Run loop cancelled.")
                 break
            except Exception as e:
                 logger.critical(f"{self.AGENT_NAME}: CRITICAL error in run loop: {e}", exc_info=True)
                 if hasattr(self.orchestrator, 'report_error'):
                     await self.orchestrator.report_error(self.AGENT_NAME, f"Critical run loop error: {e}")
                 await asyncio.sleep(60) # Wait after critical error


    async def process_task(self, task: Dict):
        """Process a single OSINT task with error handling."""
        task_type = task.get('type', 'Unknown')
        target_info = task.get('target', task.get('data_id', 'N/A')) # For logging
        start_time = time.time()
        try:
            logger.info(f"{self.AGENT_NAME}: Starting task '{task_type}' for '{target_info}'")
            if task_type == "collect_data":
                target = task.get('target')
                if not target: raise ValueError("Missing 'target' for collect_data task")
                tools_requested = task.get('tools') # Optional list of specific tools
                await self.collect_data(target, tools_requested)
            elif task_type == "analyze_data":
                data_id = task.get('data_id')
                if not data_id: raise ValueError("Missing 'data_id' for analyze_data task")
                await self.analyze_data(data_id)
            elif task_type == "visualize_data":
                data_id = task.get('data_id')
                if not data_id: raise ValueError("Missing 'data_id' for visualize_data task")
                await self.visualize_data(data_id)
            elif task_type == "run_workflow": # Example combined task
                 target = task.get('target')
                 if not target: raise ValueError("Missing 'target' for run_workflow task")
                 tools_requested = task.get('tools')
                 await self.run_osint_workflow(target, tools_requested)
            else:
                logger.error(f"{self.AGENT_NAME}: Unknown task type '{task_type}' in task: {task}")

            elapsed = time.time() - start_time
            logger.info(f"{self.AGENT_NAME}: Finished task '{task_type}' for '{target_info}' in {elapsed:.2f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Task '{task_type}' for '{target_info}' failed after {elapsed:.2f}s: {e}"
            logger.error(f"{self.AGENT_NAME}: {error_msg}", exc_info=True)
            if hasattr(self.orchestrator, 'report_error'):
                 await self.orchestrator.report_error(self.AGENT_NAME, error_msg)
        finally:
            self.task_queue.task_done()

    async def request_osint_task(self, task_type: str, **kwargs):
        """Queue an OSINT task."""
        # Architect-Zero: Add validation or rate limiting for task requests if needed
        task = {'type': task_type, **kwargs}
        await self.task_queue.put(task)
        logger.info(f"{self.AGENT_NAME}: Queued task: {task_type} with args {kwargs}")

    async def collect_data(self, target: str, tools: Optional[List[str]] = None) -> Optional[int]:
        """
        Collect data using selected OSINT tools, respecting concurrency limits.
        Returns the database ID of the collected data.
        """
        logger.info(f"{self.AGENT_NAME}: Starting data collection for target: '{target}'")
        if tools:
            # Use requested tools if provided, ensuring they are enabled
            tools_to_run = [t for t in tools if self._is_tool_enabled(t)]
            logger.info(f"{self.AGENT_NAME}: Using requested tools (if enabled): {tools_to_run}")
        else:
            # Select best tools automatically + potentially add default enabled aggressive tools
            selected_tools = await self.select_best_tools(target, num_tools=int(self.config.get("OSINT_AUTO_SELECT_TOOLS", 3)))
            aggressive_tools_to_add = []
            if self.enable_google_dorking: aggressive_tools_to_add.append('GoogleDorking')
            if self.enable_social_scraping: aggressive_tools_to_add.append('SocialMediaScraping')
            # Combine, remove duplicates, ensure enabled
            tools_to_run = list(set(selected_tools + aggressive_tools_to_add))
            tools_to_run = [t for t in tools_to_run if self._is_tool_enabled(t)]
            logger.info(f"{self.AGENT_NAME}: Automatically selected tools: {tools_to_run}")

        if not tools_to_run:
            logger.warning(f"{self.AGENT_NAME}: No enabled tools selected or requested for target '{target}'. Skipping collection.")
            return None

        # --- Run tools concurrently using semaphore ---
        data_results: Dict[str, Any] = {}
        tasks = [self.run_single_tool_wrapper(tool, target, data_results) for tool in tools_to_run]
        await asyncio.gather(*tasks)
        # --- End concurrent execution ---

        if not data_results:
             logger.warning(f"{self.AGENT_NAME}: No data collected for target '{target}' from tools: {tools_to_run}")
             # Still save a record indicating an attempt was made? Optional.
             # For now, return None if nothing was collected.
             return None

        # Store collected data in the database
        try:
            async with self.session_maker() as session:
                # Architect-Zero: Encrypt raw_data if it contains sensitive PII
                # raw_data_encrypted = encrypt_data(json.dumps(data_results))
                raw_data_json = json.dumps(data_results)

                osint_data = OSINTData(
                    target=target,
                    tools_used=json.dumps(tools_to_run), # Store as JSON string
                    raw_data=raw_data_json, # Store results/errors
                    timestamp=datetime.now(timezone.utc),
                    analysis_results=None, # Analysis comes later
                    # visualization_path=None, # Add if model has this field
                    relevance='pending_analysis' # Initial relevance state
                )
                session.add(osint_data)
                await session.commit()
                await session.refresh(osint_data) # Get the ID
                data_id = osint_data.id
                logger.info(f"{self.AGENT_NAME}: Stored collected data for '{target}' with ID {data_id}. Tools attempted: {tools_to_run}")
                return data_id
        except sqlalchemy.exc.SQLAlchemyError as db_err:
             logger.error(f"{self.AGENT_NAME}: DB error saving collected data for '{target}': {db_err}", exc_info=True)
             return None
        except Exception as e:
             logger.error(f"{self.AGENT_NAME}: Unexpected error saving collected data for '{target}': {e}", exc_info=True)
             return None

    async def run_single_tool_wrapper(self, tool_name: str, target: str, results_dict: Dict):
        """Acquires semaphore, runs tool with retry, stores result/error, updates success."""
        async with self.tool_semaphore:
            logger.debug(f"{self.AGENT_NAME}: Running tool '{tool_name}' for target '{target}'")
            tool_data = {}
            success = False
            error_info = None
            try:
                # Find the corresponding run_ method (case-insensitive match)
                method_name = f"run_{tool_name}"
                tool_func = None
                for attr_name in dir(self):
                     if attr_name.lower() == method_name.lower():
                          potential_func = getattr(self, attr_name)
                          if callable(potential_func):
                               tool_func = potential_func
                               break

                if tool_func:
                     # Architect-Zero: Apply retry logic here
                     tool_data = await self.run_tool_with_retry(tool_func, target) # Pass target
                     success = bool(tool_data) and "error" not in tool_data # Success if data returned and no error key
                else:
                     error_info = f"No implementation found for tool: {tool_name}"
                     logger.warning(f"{self.AGENT_NAME}: {error_info}")

            except Exception as e:
                logger.error(f"{self.AGENT_NAME}: Tool '{tool_name}' failed for '{target}': {e}", exc_info=False) # Avoid excessive stack trace logging here
                error_info = str(e)
                success = False

            # Store result or error
            results_dict[tool_name] = tool_data if success else {"error": error_info or "Tool execution failed"}
            # Update success rate (basic)
            await self.update_tool_success(tool_name, success)
            logger.debug(f"{self.AGENT_NAME}: Finished tool '{tool_name}' for target '{target}'. Success: {success}")


    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=30), # Exponential backoff
        stop=tenacity.stop_after_attempt(3), # Retry 3 times
        retry=tenacity.retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, subprocess.TimeoutExpired, Exception)), # Retry on common I/O or tool errors
        reraise=True # Reraise after exhausting retries
    )
    async def run_tool_with_retry(self, tool_func: callable, *args, **kwargs) -> Dict:
        """Wrapper to run a specific tool function with tenacity retry logic."""
        # Note: tool_func should return a Dict. Empty dict {} signifies no results found.
        # Exceptions will trigger retries.
        logger.debug(f"Executing {tool_func.__name__} with args: {args}, kwargs: {kwargs}")
        return await tool_func(*args, **kwargs)

    # --- Helper Methods ---

    async def _scrape_simple_html(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetches and parses HTML from a URL using aiohttp and BeautifulSoup.
        Includes placeholder logic for proxy usage via Orchestrator.
        """
        self.logger.debug(f"Attempting simple scrape of URL: {url}")
        ua = self.user_agent_generator.random if self.user_agent_generator else 'Mozilla/5.0'
        headers = {'User-Agent': ua, 'Accept-Language': 'en-US,en;q=0.9', 'Accept': 'text/html'}
        proxy_url: Optional[str] = None

        # --- Proxy Integration Placeholder ---
        if hasattr(self.orchestrator, 'get_proxy'):
            try:
                # Assume get_proxy returns a valid proxy URL string or None
                proxy_url = await self.orchestrator.get_proxy(purpose="scraping", target_url=url)
                if proxy_url:
                    self.logger.debug(f"Using proxy for scraping {url}: {proxy_url.split('@')[-1]}") # Log proxy host, not creds
                else:
                    self.logger.debug(f"No proxy available or needed for {url}. Proceeding directly.")
            except Exception as proxy_err:
                self.logger.error(f"Failed to get proxy for scraping {url}: {proxy_err}")
                # Decide whether to proceed without proxy or fail
                # proxy_url = None # Ensure it's None if error occurs
        else:
            self.logger.warning("Orchestrator does not have 'get_proxy' method. Scraping without proxy.")
        # --- End Proxy Integration Placeholder ---

        try:
            # Create session inside try block after potentially getting proxy
            async with aiohttp.ClientSession(headers=headers) as session:
                request_kwargs = {"timeout": 30, "allow_redirects": True}
                if proxy_url:
                    request_kwargs["proxy"] = proxy_url

                async with session.get(url, **request_kwargs) as response:
                    self.logger.debug(f"Scrape request to {url} (via proxy: {bool(proxy_url)}) completed with status: {response.status}")
                    if response.status == 200:
                        # Check content type before reading potentially large non-HTML response
                        if 'text/html' in response.headers.get('Content-Type', '').lower():
                            soup = BeautifulSoup(html, 'html.parser')
                            self.logger.debug(f"Successfully scraped and parsed HTML from {url}")
                            return soup
                        else:
                            self.logger.warning(f"Content type for {url} is not text/html ({response.headers.get('Content-Type')}). Skipping parse.")
                            return None
                    else:
                        self.logger.warning(f"Failed to fetch {url} for scraping. Status: {response.status}")
                        return None
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout scraping URL: {url}")
            return None
        except aiohttp.ClientError as e:
             self.logger.error(f"Client error scraping {url}: {e}")
             return None
        except Exception as e:
            self.logger.error(f"Unexpected error scraping {url}: {e}", exc_info=True)
            return None

    # --- Tool Implementations ---
    # Architect-Zero: Add/Refine implementations below. Ensure they return dict.

    async def run_theHarvester(self, target: str) -> Dict:
        """Run theHarvester CLI tool."""
        # Requires theHarvester installed and in PATH
        logger.info(f"Running theHarvester for: {target}")
        # Example: Run against google, bing. Limit results.
        # Adjust sources and limit as needed. Check theHarvester --help.
        # Architect-Zero: Ensure target is sanitized if used directly in filename
        safe_target_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', target)
        output_file = f"/tmp/harvester_{safe_target_filename}_{uuid.uuid4()}.json" # Output to file in /tmp
        command = [
            "theHarvester",
            "-d", target,
            "-l", "100", # Limit results
            "-b", "google,bing", # Sources
            "-f", output_file
        ]
        results = {}
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"theHarvester failed for {target}. Return code: {process.returncode}. Stderr: {stderr.decode(errors='ignore')}")
                return {"error": f"theHarvester exited with code {process.returncode}", "stderr": stderr.decode(errors='ignore')}

            # Read results from the JSON file
            if os.path.exists(output_file):
                 try:
                     with open(output_file, 'r', encoding='utf-8') as f: # Specify encoding
                          results = json.load(f)
                 except json.JSONDecodeError as json_err:
                      logger.error(f"Failed to decode theHarvester JSON output {output_file}: {json_err}")
                      # Try reading stdout as fallback
                      results = {"error": "Failed to parse JSON output", "stdout": stdout.decode(errors='ignore')}
                 finally:
                      try:
                          os.remove(output_file) # Clean up temp file
                      except OSError as rm_err:
                           logger.warning(f"Failed to remove temp file {output_file}: {rm_err}")
            else:
                 logger.warning(f"theHarvester output file not found: {output_file}")
                 results = {"stdout": stdout.decode(errors='ignore')} # Fallback to stdout

            logger.info(f"theHarvester completed for {target}. Found emails: {len(results.get('emails',[]))}, hosts: {len(results.get('hosts',[]))}")
            return results # Return parsed JSON content
        except FileNotFoundError:
             logger.error("theHarvester command not found. Ensure it is installed and in PATH.")
             return {"error": "theHarvester command not found"}
        except Exception as e:
            logger.error(f"theHarvester execution failed for {target}: {e}", exc_info=True)
            # Clean up temp file on error if it exists
            if 'output_file' in locals() and os.path.exists(output_file):
                 try: os.remove(output_file)
                 except OSError: pass
            return {"error": str(e)}


    async def run_ExifTool(self, target: str) -> Dict: # Changed signature to accept target
        """Run ExifTool on a *local file path* (if target is a downloadable URL) or return error."""
        # Requires python-exiftool library and exiftool binary installed
        logger.info(f"Attempting ExifTool analysis for target: {target}")
        # Architect-Zero: Check if target looks like a URL pointing to a potential file
        if not (target.startswith("http://") or target.startswith("https://")):
             logger.warning(f"ExifTool requires a file path or downloadable URL, received target '{target}'. Cannot process.")
             return {"error": "ExifTool requires a file path or downloadable URL."}

        # Attempt to download the file temporarily
        temp_file_path = f"/tmp/exiftool_{uuid.uuid4()}"
        try:
            async with aiohttp.ClientSession() as session:
                 async with session.get(target, timeout=30) as response:
                      if response.status == 200:
                           with open(temp_file_path, 'wb') as f:
                                while True:
                                     chunk = await response.content.read(1024)
                                     if not chunk: break
                                     f.write(chunk)
                           logger.debug(f"Downloaded potential file from {target} to {temp_file_path}")
                      else:
                           logger.warning(f"Failed to download file from {target} for ExifTool. Status: {response.status}")
                           return {"error": f"Failed to download file (Status: {response.status})"}

            # Now run ExifTool on the downloaded file
            try:
                import exiftool
                with exiftool.ExifToolHelper() as et:
                    metadata_list = et.get_metadata(temp_file_path)
                logger.info(f"ExifTool completed for downloaded file from {target}")
                # Clean metadata keys (e.g., remove 'EXIF:', 'File:') prefixes if desired
                # Return metadata from the first file in the list (should only be one)
                return {k.split(':')[-1]: v for k, v in metadata_list[0].items()} if metadata_list else {}
            except FileNotFoundError:
                 logger.error("exiftool command not found. Ensure it is installed and in PATH.")
                 return {"error": "exiftool command not found"}
            except Exception as e:
                logger.error(f"ExifTool failed for downloaded file {temp_file_path}: {e}", exc_info=True)
                return {"error": str(e)}
            finally:
                 # Clean up downloaded file
                 if os.path.exists(temp_file_path):
                      try: os.remove(temp_file_path)
                      except OSError: pass

        except Exception as download_err:
             logger.error(f"Failed to download or process file from {target} for ExifTool: {download_err}", exc_info=True)
             if os.path.exists(temp_file_path): # Ensure cleanup even on download error
                  try: os.remove(temp_file_path)
                  except OSError: pass
             return {"error": f"Failed to download/process file: {download_err}"}


    async def run_Photon(self, target_url: str) -> Dict:
        """Run Photon web crawler (basic usage)."""
        # Requires photon library installed
        logger.info(f"Running Photon crawler on URL: {target_url}")
        # Architect-Zero: Photon can be noisy and resource-intensive. Use with caution.
        # Consider limiting depth or scope. Needs error handling.
        # This is a placeholder as direct library usage might be complex.
        # Running as CLI might be simpler if installed.
        # Example CLI approach:
        safe_target_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', target_url)
        output_dir = f"/tmp/photon_{safe_target_filename}_{uuid.uuid4()}"
        command = ["photon", "-u", target_url, "--keys", "--json", "-o", output_dir, "-l", "2", "-t", "10"] # Limit level, threads
        results = {}
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            # Set a timeout for the process
            timeout_seconds = 300 # 5 minutes
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)

            if process.returncode != 0:
                 logger.error(f"Photon CLI failed for {target_url}. Stderr: {stderr.decode(errors='ignore')}")
                 # Try to read partial results if dir exists?
            else:
                 logger.info(f"Photon completed for {target_url}. Checking output in {output_dir}")

            # Read results from expected output files in output_dir
            # Example: Read links.json if it exists (adjust path based on Photon's actual output structure)
            photon_target_dir = target_url.split('//')[-1].replace('/', '_') # Photon often creates dir based on target
            results_file = os.path.join(output_dir, photon_target_dir, "links.json") # Check Photon's output structure
            if os.path.exists(results_file):
                 try:
                     with open(results_file, 'r', encoding='utf-8') as f:
                          results['links'] = json.load(f)
                 except Exception as read_err:
                      logger.error(f"Error reading Photon results file {results_file}: {read_err}")
                      results['error'] = "Error reading results file"
            else:
                 logger.warning(f"Photon output file not found: {results_file}")
                 results['stdout'] = stdout.decode(errors='ignore')[:1000] # Store snippet

            # Clean up output directory
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

            return results
        except asyncio.TimeoutError:
             logger.error(f"Photon timed out after {timeout_seconds}s for {target_url}.")
             if process.returncode is None: process.kill()
             import shutil; shutil.rmtree(output_dir, ignore_errors=True) # Cleanup on timeout
             return {"error": f"Photon timed out after {timeout_seconds}s"}
        except FileNotFoundError:
             logger.error("photon command not found. Ensure it is installed.")
             return {"error": "photon command not found"}
        except Exception as e:
            logger.error(f"Photon execution failed for {target_url}: {e}", exc_info=True)
            import shutil; shutil.rmtree(output_dir, ignore_errors=True) # Cleanup on error
            return {"error": str(e)}


    async def run_Sherlock(self, username: str) -> Dict:
        """Run Sherlock username reconnaissance CLI tool."""
        # Requires Sherlock installed
        logger.info(f"Running Sherlock for username: {username}")
        # Example: Timeout after 5 minutes, save JSON output
        # Architect-Zero: Sanitize username for filename
        safe_username = re.sub(r'[^a-zA-Z0-9_-]', '_', username)
        output_file = f"/tmp/sherlock_{safe_username}_{uuid.uuid4()}.json"
        command = [
            "sherlock",
            username,
            "--json", output_file,
            "--timeout", "5", # Timeout per site request
            "--print-found" # Only print found accounts to stdout (reduces noise)
        ]
        results = {}
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            # Sherlock can run for a long time, set a global timeout
            timeout_seconds = 300 # 5 minutes total
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)

            if process.returncode != 0:
                logger.error(f"Sherlock failed for {username}. Stderr: {stderr.decode(errors='ignore')}")
                # Try reading partial JSON if it exists

            if os.path.exists(output_file):
                 try:
                     with open(output_file, 'r', encoding='utf-8') as f:
                          # Sherlock JSON output is one JSON object per line for found sites
                          found_sites = {}
                          for line in f:
                               try:
                                    site_data = json.loads(line)
                                    # Structure might vary, adapt parsing as needed
                                    # Example: { "username": "...", "name": "...", "url_main": "...", ... }
                                    site_name = site_data.get("name")
                                    if site_name:
                                         found_sites[site_name] = site_data # Use site name as key
                               except json.JSONDecodeError: continue
                          results = found_sites # Use the dictionary of found sites
                 except Exception as read_err:
                      logger.error(f"Error reading Sherlock results file {output_file}: {read_err}")
                      results = {"error": "Error reading results file"}
                 finally:
                      try: os.remove(output_file)
                      except OSError: pass
            else:
                 logger.warning(f"Sherlock output file not found: {output_file}")
                 # Parse stdout if --print-found was used
                 stdout_lines = stdout.decode(errors='ignore').splitlines()
                 found_urls = [line.split(' ')[-1] for line in stdout_lines if "http" in line]
                 results = {"found_urls_stdout": found_urls} if found_urls else {"error": "Sherlock output file not found, no URLs in stdout"}


            logger.info(f"Sherlock completed for {username}. Found {len(results)} potential profiles.")
            return results
        except asyncio.TimeoutError:
             logger.error(f"Sherlock timed out after {timeout_seconds}s for username {username}.")
             # Kill process if possible
             if 'process' in locals() and process.returncode is None:
                  try: process.kill()
                  except ProcessLookupError: pass
             # Check for partial results
             if 'output_file' in locals() and os.path.exists(output_file):
                  # ... read partial results as above ...
                  try: os.remove(output_file)
                  except OSError: pass
             return {"error": f"Sherlock timed out after {timeout_seconds}s"}
        except FileNotFoundError:
             logger.error("sherlock command not found. Ensure it is installed.")
             return {"error": "sherlock command not found"}
        except Exception as e:
            logger.error(f"Sherlock execution failed for {username}: {e}", exc_info=True)
            if 'output_file' in locals() and os.path.exists(output_file):
                 try: os.remove(output_file)
                 except OSError: pass
            return {"error": str(e)}


    async def run_Maltego(self, target: str) -> Dict:
        """Run Maltego transform (requires maltego-trx library and config)."""
        # Requires maltego-trx library
        logger.info(f"Running Maltego transform simulation for: {target}")
        # Architect-Zero: Direct Maltego execution is complex. This simulates adding an entity.
        # Real integration would involve setting up a local transform server (TDS)
        # or interacting with a commercial Maltego server API if available.
        # This implementation is a placeholder/simulation.
        try:
            from maltego_trx.maltego import MaltegoTransform, MaltegoMsg
            from maltego_trx.entities import Domain, EmailAddress, Person, PhoneNumber, Phrase, URL, IPv4Address # Add more as needed

            # request_msg = MaltegoMsg() # Simulate request message if needed
            response = MaltegoTransform() # Simulate response transform

            # Add the target as an entity
            target_entity = None
            if '@' in target and '.' in target: target_entity = response.addEntity(EmailAddress, target)
            elif '.' in target: target_entity = response.addEntity(Domain, target)
            else: target_entity = response.addEntity(Phrase, target)
            if target_entity: target_entity.setBookmark(1) # Bookmark primary target

            # Add example related entities (replace with actual transform logic)
            if isinstance(target_entity, Domain):
                 response.addEntity("maltego.NSRecord", f"ns1.{target}")
                 response.addEntity("maltego.MXRecord", f"mail.{target}")

            # Return the simulated entities added
            # The actual return format depends on how you consume this.
            # Returning a list of entities for simplicity.
            maltego_output = response.returnOutput()
            logger.info(f"Maltego simulation completed for {target}")
            # Parse maltego_output (XML) if needed, or return a simplified dict
            return {"maltego_entities_xml": maltego_output} # Placeholder return

        except ImportError:
            logger.error("maltego-trx library not installed. Cannot run Maltego transform.")
            return {"error": "maltego-trx library not installed"}
        except Exception as e:
            logger.error(f"Maltego simulation failed for {target}: {e}", exc_info=True)
            return {"error": str(e)}


    async def run_Shodan(self, query: str) -> Dict:
        """Run Shodan search using the API."""
        # Requires shodan library and API key
        logger.info(f"Running Shodan search for query: {query}")
        if not self._shodan_api_key: # Check stored key
            logger.warning("Shodan API key not available. Skipping Shodan search.")
            return {"error": "Shodan API key not available"}
        try:
            import shodan
            api = shodan.Shodan(self._shodan_api_key) # Use stored key
            # Use search_cursor for potentially large results, or simple search for smaller ones
            # results = api.search(query) # Simple search
            # Cursor example: Get first N results
            results_cursor = api.search_cursor(query)
            results = {'matches': [], 'total': 0} # Initialize structure
            count = 0
            limit = int(self.config.get("SHODAN_RESULT_LIMIT", 100))
            for banner in results_cursor:
                 results['matches'].append(banner)
                 count += 1
                 if count >= limit:
                      break
            # Get total from Shodan API if possible (might require separate call or exists in cursor)
            # Attempt to get total count (may vary by library version/API)
            try:
                 # Re-run search to get total (inefficient but common pattern if cursor doesn't provide it)
                 count_result = api.count(query)
                 results['total'] = count_result.get('total', 0)
            except Exception as count_err:
                 logger.warning(f"Could not get total Shodan result count for '{query}': {count_err}")


            logger.info(f"Shodan search completed for '{query}'. Found {results['total']} total, retrieved {len(results['matches'])} (limited to {limit}).")
            return results
        except ImportError:
            logger.error("shodan library not installed. Cannot run Shodan search.")
            return {"error": "shodan library not installed"}
        except shodan.APIError as e:
            logger.error(f"Shodan API error for query '{query}': {e}")
            return {"error": f"Shodan API error: {e}"}
        except Exception as e:
            logger.error(f"Shodan search failed for query '{query}': {e}", exc_info=True)
            return {"error": str(e)}


    async def run_Reconng(self, target: str) -> Dict: # Renamed method to match key used elsewhere
        """Run Recon-ng CLI tool with a specific module."""
        # Requires recon-ng installed
        logger.info(f"Running Recon-ng for target: {target}")
        # Architect-Zero: Choose module carefully. `hackertarget` is often free/less restricted.
        # Make module and options configurable. Sanitize target.
        module = self.config.get("RECONNG_MODULE", "recon/domains-hosts/hackertarget")
        # Basic sanitization: remove potentially harmful characters for CLI context
        safe_target = ''.join(c for c in target if c.isalnum() or c in ['.', '-', '_'])
        if not safe_target: return {"error": "Invalid target for Recon-ng"}

        # Use workspace to isolate results
        workspace_name = f"osint_{safe_target}_{uuid.uuid4()}"
        commands = f"""
        workspaces create {workspace_name}
        modules load {module}
        options set SOURCE {safe_target}
        run
        show hosts
        workspaces delete {workspace_name}
        exit
        """
        results = {}
        try:
            process = await asyncio.create_subprocess_exec(
                "recon-ng", "-r", "/dev/stdin", # Read commands from stdin
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate(input=commands.encode())

            if process.returncode != 0:
                logger.error(f"Recon-ng failed for {target}. Stderr: {stderr.decode(errors='ignore')}")
                return {"error": f"Recon-ng exited with code {process.returncode}", "stderr": stderr.decode(errors='ignore')}

            # Parse stdout for relevant information (e.g., hosts found)
            output_lines = stdout.decode(errors='ignore').splitlines()
            # Example parsing for 'show hosts' output (adjust based on actual output format)
            # Look for lines starting with IP or domain in the table format
            hosts = []
            in_table = False
            for line in output_lines:
                 if line.strip().startswith('----'): # Detect table separator
                      in_table = not in_table # Toggle state
                      continue
                 if in_table and line.strip():
                      parts = line.split()
                      if len(parts) > 0:
                           hosts.append(parts[0]) # Assume first column is host/IP

            results['reconng_output'] = "\n".join(output_lines) # Store raw output
            results['found_hosts'] = list(set(hosts)) # Store parsed hosts

            logger.info(f"Recon-ng ({module}) completed for {target}. Found {len(results['found_hosts'])} potential hosts.")
            return results
        except FileNotFoundError:
             logger.error("recon-ng command not found. Ensure it is installed and in PATH.")
             return {"error": "recon-ng command not found"}
        except Exception as e:
            logger.error(f"Recon-ng execution failed for {target}: {e}", exc_info=True)
            return {"error": str(e)}


    async def run_SpiderFoot(self, target: str) -> Dict:
        """Interact with a running SpiderFoot instance API."""
        # Requires a running SpiderFoot HX or embedded instance accessible
        spiderfoot_url = self.internal_state.get('spiderfoot_url')
        logger.info(f"Initiating SpiderFoot scan for target: {target} via {spiderfoot_url}")
        if not self._spiderfoot_api_key or not spiderfoot_url: # Check stored key and URL
            logger.warning("SpiderFoot URL or API key not available. Skipping scan.")
            return {"error": "SpiderFoot URL or API key not available"}

        scan_name = f"OSINTAgent_{target}_{int(time.time())}"
        # Select modules - use a safe subset by default, make configurable
        modules_to_run = self.config.get("SPIDERFOOT_MODULES", ["sfp_dnsresolve", "sfp_geoip", "sfp_ripe"]) # Example safe modules

        try:
            async with aiohttp.ClientSession() as session:
                # 1. Start Scan
                start_payload = {
                    'scanname': scan_name,
                    'scantarget': target,
                    'modulelist': modules_to_run,
                    'usecase': 'all' # Or specify 'footprint', 'investigate', 'passive'
                }
                headers = {'X-API-KEY': self._spiderfoot_api_key} # Use stored key
                scan_id = None
                logger.debug(f"Starting SpiderFoot scan: URL={spiderfoot_url}/startscan, Payload={start_payload}")
                async with session.post(f"{spiderfoot_url}/startscan", json=start_payload, headers=headers) as response:
                    if response.status == 200:
                        scan_id = await response.text() # API returns scan ID directly
                        scan_id = scan_id.strip('"') # Remove quotes if present
                        logger.info(f"SpiderFoot scan started for '{target}'. Scan ID: {scan_id}")
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to start SpiderFoot scan for '{target}'. Status: {response.status}. Response: {error_text}")
                        return {"error": f"Failed to start SpiderFoot scan (Status: {response.status})", "details": error_text}

                if not scan_id: return {"error": "Failed to obtain Scan ID from SpiderFoot"}

                # 2. Poll for Scan Completion
                poll_interval = int(self.config.get("SPIDERFOOT_POLL_INTERVAL_S", 30))
                max_poll_time = int(self.config.get("SPIDERFOOT_MAX_POLL_TIME_S", 600)) # 10 minutes max wait
                start_poll_time = time.time()
                while True:
                    if time.time() - start_poll_time > max_poll_time:
                         logger.error(f"SpiderFoot scan {scan_id} for '{target}' timed out after {max_poll_time}s.")
                         # Optionally try to retrieve partial results
                         return {"error": f"SpiderFoot scan timed out after {max_poll_time}s"}

                    await asyncio.sleep(poll_interval)
                    logger.debug(f"Polling SpiderFoot scan status for ID: {scan_id}")
                    async with session.get(f"{spiderfoot_url}/scanstatus?id={scan_id}", headers=headers) as status_response:
                         if status_response.status == 200:
                              status_text = await status_response.text()
                              status_text = status_text.strip('"')
                              logger.debug(f"SpiderFoot scan {scan_id} status: {status_text}")
                              if status_text == 'FINISHED':
                                   logger.info(f"SpiderFoot scan {scan_id} for '{target}' finished.")
                                   break
                              elif status_text in ['ERROR', 'ABORTED', 'ABORTING', 'UNKNOWN']: # Handle more terminal states
                                   logger.error(f"SpiderFoot scan {scan_id} for '{target}' ended with status: {status_text}")
                                   return {"error": f"SpiderFoot scan ended with status: {status_text}"}
                              # Else: continue polling for statuses like 'RUNNING', 'STARTING'
                         else:
                              # Handle polling error - maybe retry polling?
                              logger.warning(f"Failed to get SpiderFoot scan status for {scan_id}. Status: {status_response.status}")
                              # Continue polling for now

                # 3. Retrieve Scan Results (Summary or specific types)
                # Example: Get summary
                logger.debug(f"Retrieving SpiderFoot summary for scan ID: {scan_id}")
                async with session.get(f"{spiderfoot_url}/scansummary?id={scan_id}", headers=headers) as summary_response:
                     if summary_response.status == 200:
                          results = await summary_response.json()
                          logger.info(f"SpiderFoot scan {scan_id} for '{target}' completed. Retrieved summary.")
                          # Optionally retrieve detailed results by type if needed
                          # e.g., /scaneventresults?id={scan_id}&type=DNS_NAME
                          return {"summary": results}
                     else:
                          error_text = await summary_response.text()
                          logger.error(f"Failed to retrieve SpiderFoot results for scan {scan_id}. Status: {summary_response.status}. Response: {error_text}")
                          return {"error": f"Failed to retrieve SpiderFoot results (Status: {summary_response.status})", "details": error_text}

        except aiohttp.ClientConnectionError as e:
             logger.error(f"Could not connect to SpiderFoot at {spiderfoot_url}: {e}")
             return {"error": f"Could not connect to SpiderFoot at {spiderfoot_url}"}
        except Exception as e:
            logger.error(f"SpiderFoot interaction failed for {target}: {e}", exc_info=True)
            return {"error": str(e)}

    # --- Aggressive / Grey Area Tools (Use with Extreme Caution) ---

    async def run_GoogleDorking(self, target: str) -> Dict:
        """Perform Google Dorking using basic scraping. WARNING: High risk of IP block/CAPTCHA. Respect ToS."""
        # Import necessary library here or at top level
        import urllib.parse

        if not self.internal_state.get('enable_google_dorking'):
            self.logger.info("Google Dorking disabled by configuration.")
            return {"status": "disabled"}

        self.logger.warning(f"Executing Google Dorking for {target}. RISKY - may violate ToS / get blocked.")
        # Basic dorks - TODO: Make these configurable or dynamically generated
        dorks = [
            f"site:{target}",
            f'site:{target} intitle:"index of"',
            f'site:{target} filetype:pdf',
            f'site:{target} inurl:login',
            f'site:{target} "contact"',
            f'site:{target} "about us"',
        ]
        results = {}
        # Use the generated UA if available, otherwise default
        ua = self.user_agent_generator.random if self.user_agent_generator else 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        headers = {'User-Agent': ua, 'Accept-Language': 'en-US,en;q=0.9'}
        base_url = "https://www.google.com/search"

        try:
            # Use a single session for all dorks for potential cookie persistence (though Google might not rely on it)
            async with aiohttp.ClientSession(headers=headers) as session:
                for dork in dorks:
                    params = {'q': dork, 'num': 20} # Request more results
                    self.logger.debug(f"Executing Google Dork: {dork}")
                    try:
                        async with session.get(base_url, params=params, timeout=20) as response: # Add timeout
                            # Implement stricter rate limiting
                            await asyncio.sleep(random.uniform(5, 10)) # Longer, more random delay

                            if response.status == 200:
                                html = await response.text()
                                # Check for CAPTCHA page before parsing
                                if "CAPTCHA" in html or "unusual traffic" in html:
                                     self.logger.error(f"Google Dorking CAPTCHA encountered for {target} on dork '{dork}'. Aborting dorks.")
                                     results[dork] = {"error": "CAPTCHA / Blocked"}
                                     break # Stop dorking if blocked

                                soup = BeautifulSoup(html, 'html.parser')
                                links = []
                                # Refined selector for main result links
                                for link_tag in soup.select('div.g a'): # Select links within general result blocks
                                    href = link_tag.get('href')
                                    if href:
                                        if href.startswith('/url?q='): # Google redirect URL
                                            try:
                                                actual_link = href.split('/url?q=')[1].split('&sa=')[0]
                                                links.append(urllib.parse.unquote(actual_link))
                                            except Exception: pass # Ignore parsing errors
                                        elif href.startswith('http') and not href.startswith('https://webcache.googleusercontent.com') and not href.startswith('https://policies.google.com'):
                                            # Direct link, ignore cache/policy links
                                            links.append(href)

                                results[dork] = list(set(links)) # Unique links
                                self.logger.debug(f"Google Dork '{dork}' found {len(results[dork])} potential links.")
                            elif response.status == 429:
                                 self.logger.error(f"Google Dorking blocked (429 Too Many Requests) for {target} on dork '{dork}'. Aborting dorks.")
                                 results[dork] = {"error": "Blocked (429)"}
                                 break # Stop dorking if blocked
                            else:
                                logger.warning(f"Google Dork '{dork}' failed with status {response.status}")
                                results[dork] = {"error": f"HTTP Status {response.status}"}
                    except asyncio.TimeoutError:
                         logger.error(f"Google Dork '{dork}' timed out.")
                         results[dork] = {"error": "Timeout"}
                    except Exception as req_err:
                         logger.error(f"Error during Google Dork request '{dork}': {req_err}")
                         results[dork] = {"error": str(req_err)}
                         # Decide whether to break or continue on individual dork errors

            return {"google_dorks": results}
        except Exception as e:
            self.logger.error(f"Google Dorking failed for {target}: {e}", exc_info=True)
            return {"error": f"Google Dorking failed: {str(e)}"}


    async def run_SocialMediaScraping(self, target: str) -> Dict:
        """Basic social media profile searching. WARNING: Violates ToS of most platforms."""
        if not self.enable_social_scraping:
            return {"status": "disabled"}
        logger.warning(f"Executing Social Media Scraping for {target}. HIGHLY RISKY - likely violates ToS.")
        # Architect-Zero: This is extremely basic and likely to fail or be inaccurate.
        # Proper social media OSINT requires dedicated tools or APIs (if available).
        platforms = {
            "LinkedIn": f"https://www.linkedin.com/search/results/people/?keywords={target}",
            "Facebook": f"https://www.facebook.com/search/people/?q={target}",
            # Twitter/X search requires login or different endpoints usually
        }
        results = {}
        headers = {'User-Agent': self.user_agent.random if hasattr(self, 'user_agent') else 'Mozilla/5.0', 'Accept-Language': 'en-US,en;q=0.9'}
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                for name, url in platforms.items():
                    logger.debug(f"Scraping {name} for '{target}'")
                    async with session.get(url) as response:
                        await asyncio.sleep(random.uniform(3, 6)) # Rate limit
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            # Parsing is platform-specific and highly fragile
                            # Example: Look for profile links (very naive)
                            profile_links = []
                            for a in soup.find_all('a', href=True):
                                href = a['href']
                                # Add platform-specific checks
                                if name == "LinkedIn" and ('/in/' in href or '/pub/' in href):
                                     profile_links.append(href)
                                elif name == "Facebook" and ('profile.php?id=' in href or '/people/' in href):
                                     profile_links.append(href)
                            results[name] = {"potential_profiles": list(set(profile_links[:10]))} # Limit results
                        else:
                            logger.warning(f"Scraping {name} failed for '{target}' with status {response.status}")
                            results[name] = {"error": f"HTTP Status {response.status}"}
            return {"social_media_scan": results}
        except Exception as e:
            logger.error(f"Social Media Scraping failed for {target}: {e}", exc_info=True)
            return {"error": str(e)}


    async def run_DarkPoolSearch(self, target: str) -> Dict:
        """Placeholder for Dark Pool Search. WARNING: Requires specialized access (Tor/I2P) and carries extreme legal/ethical risks."""
        if not self.enable_dark_pool:
            return {"status": "disabled"}
        logger.critical(f"Executing Dark Pool Search for {target}. EXTREME RISK. Ensure legality and use appropriate anonymization (Tor/I2P).")
        # Architect-Zero: Implementation requires libraries like stem (for Tor) or i2p.socket
        # and access to relevant dark web search engines or forums. This is a non-functional placeholder.
        # Replace 'http://darkpool.example.com' with actual Tor/I2P accessible endpoint.
        # Ensure requests go through the Tor/I2P proxy.
        # proxy_url = "socks5h://localhost:9050" # Example Tor proxy
        # async with aiohttp.ClientSession(connector=aiohttp.ProxyConnector.from_url(proxy_url)) as session:
        #     async with session.get(f"http://darkpool.example.onion/search?q={target}") as response: ...
        return {"error": "Dark Pool Search not implemented. Requires Tor/I2P integration and valid endpoints."}


    async def run_WebcamSearch(self, target: str) -> Dict:
        """Placeholder for searching public webcams. WARNING: Privacy and legal implications."""
        if not self.enable_webcam_search:
            return {"status": "disabled"}
        logger.critical(f"Executing Webcam Search for {target}. PRIVACY/LEGAL RISK. Only search ethical, publicly intended streams.")
        # Architect-Zero: Requires specific webcam search engines (e.g., Shodan webcam search, Insecam - use ethically!)
        # or specific APIs. This is a non-functional placeholder.
        # Example using hypothetical API:
        # async with aiohttp.ClientSession() as session:
        #     async with session.get(f"http://webcamsearch.example.com/api?query={target}&key=API_KEY") as response: ...
        return {"error": "Webcam Search not implemented. Requires specific search engine integration."}

    # --- Analysis ---

    async def analyze_data(self, data_id: int):
        """Analyze collected OSINT data using LLMs and specific parsers."""
        logger.info(f"{self.AGENT_NAME}: Starting analysis for OSINT data ID: {data_id}")
        raw_data = {}
        analysis = {"error": "Analysis failed to complete"} # Default error state
        try:
            async with self.session_maker() as session:
                osint_data = await session.get(OSINTData, data_id)
                if not osint_data:
                    logger.error(f"{self.AGENT_NAME}: OSINT data ID {data_id} not found for analysis.")
                    return
                if not osint_data.raw_data:
                     logger.warning(f"{self.AGENT_NAME}: No raw data found for OSINT ID {data_id}. Skipping analysis.")
                     osint_data.relevance = 'no_data'
                     await session.commit()
                     return

                try:
                    raw_data = json.loads(osint_data.raw_data)
                except json.JSONDecodeError:
                    logger.error(f"{self.AGENT_NAME}: Failed to decode raw_data JSON for ID {data_id}. Cannot analyze.")
                    osint_data.analysis_results = json.dumps({"error": "Failed to decode raw_data JSON"})
                    osint_data.relevance = 'error'
                    await session.commit()
                    return

                # --- LLM-based Analysis ---
                # Architect-Zero: Use a powerful model for analysis. Structure prompt carefully.
                # Truncate raw_data if too large for prompt
                raw_data_str_for_prompt = json.dumps(raw_data, indent=2, ensure_ascii=False)
                max_prompt_chars = 10000 # Example limit
                if len(raw_data_str_for_prompt) > max_prompt_chars:
                     raw_data_str_for_prompt = raw_data_str_for_prompt[:max_prompt_chars] + "\n... (data truncated)"
                     logger.warning(f"{self.AGENT_NAME}: Truncated raw_data for LLM analysis prompt (ID: {data_id})")

                prompt = f"""
                Analyze the following OSINT data collected for target '{osint_data.target}'.
                Data was collected using tools: {osint_data.tools_used}
                Collected Data (JSON, potentially truncated):
                ```json
                {raw_data_str_for_prompt}
                ```

                Your Task:
                1. Summarize the key findings (e.g., primary email domains, key social profiles, potential vulnerabilities, company info).
                2. Identify potential leads or actionable insights relevant to a UGC (User-Generated Content) agency. Look for contact points (emails, specific LinkedIn profiles), marketing roles, company size indicators, technology used (e.g., marketing automation, CMS).
                3. Assess the overall data quality and potential reliability (Low, Medium, High) based on sources and consistency.
                4. Provide a concise overall summary.

                Return JSON: {{
                    "summary": "Overall summary of findings.",
                    "key_findings": {{ "emails": ["list of emails"], "social": ["list of profile URLs"], "vulnerabilities": ["list of CVEs or descriptions"], "company_info": ["size, industry, tech stack hints"], "other": ["other notable findings"] }},
                    "ugc_leads": {{ "contacts": ["list of potential contact emails/profiles"], "insights": ["list of relevant insights for UGC pitch"] }},
                    "data_quality_assessment": "Low/Medium/High",
                    "actionable": true/false
                }}
                """

                clients = await self.orchestrator.get_available_openrouter_clients() # Or specific LLM client getter
                if not clients:
                     logger.error(f"{self.AGENT_NAME}: No LLM clients available for analysis of ID {data_id}.")
                     analysis = {"error": "No LLM clients available"}
                else:
                     last_exception = None
                     content = None # Initialize content
                     for client in clients:
                         try:
                             llm_response = await client.chat.completions.create(
                                 model=self.config.get("OPENROUTER_ANALYSIS_MODEL", "google/gemini-pro-1.5"), # Use capable model
                                 messages=[{"role": "user", "content": prompt}],
                                 temperature=0.3,
                                 max_tokens=2000, # Allow longer analysis
                                 response_format={"type": "json_object"},
                                 timeout=float(self.config.get("OPENROUTER_ANALYSIS_TIMEOUT_S", 120.0)) # Longer timeout for analysis
                             )
                             content = llm_response.choices[0].message.content
                             analysis = json.loads(content)
                             logger.info(f"{self.AGENT_NAME}: LLM analysis successful for ID {data_id} using client {client.api_key[:5]}...")
                             break # Success
                         except (json.JSONDecodeError, KeyError, ValueError) as parse_err:
                              logger.warning(f"{self.AGENT_NAME}: Failed to parse LLM analysis response ({parse_err}) for ID {data_id}. Response: {content}")
                              last_exception = parse_err
                         except Exception as e:
                              logger.warning(f"{self.AGENT_NAME}: LLM client {client.api_key[:5]}... failed for analysis of ID {data_id}: {e}")
                              last_exception = e
                              if "rate limit" in str(e).lower() or "authentication" in str(e).lower():
                                   if hasattr(self.orchestrator, 'mark_client_unavailable'):
                                        await self.orchestrator.mark_client_unavailable(client.api_key)
                     else: # Loop finished without break
                          logger.error(f"{self.AGENT_NAME}: All LLM clients failed analysis for ID {data_id}. Last error: {last_exception}")
                          analysis = {"error": f"All LLM clients failed. Last error: {last_exception}"}


                # --- Store Analysis and Determine Relevance ---
                osint_data.analysis_results = json.dumps(analysis)
                # Determine relevance based on analysis results
                is_actionable = analysis.get('actionable', False)
                has_leads = bool(analysis.get('ugc_leads', {}).get('contacts')) or bool(analysis.get('ugc_leads', {}).get('insights'))
                if "error" in analysis:
                     osint_data.relevance = 'error'
                elif is_actionable or has_leads:
                     osint_data.relevance = 'high'
                elif analysis.get('key_findings'):
                     osint_data.relevance = 'medium'
                else:
                     osint_data.relevance = 'low'

                await session.commit()
                logger.info(f"{self.AGENT_NAME}: Analysis completed for ID {data_id}. Relevance assessed as '{osint_data.relevance}'.")

                # --- Trigger Integrations (e.g., send leads to sales agents) ---
                if osint_data.relevance in ['high', 'medium']:
                     await self.trigger_integrations(data_id, analysis)

        except sqlalchemy.exc.SQLAlchemyError as db_err:
             logger.error(f"{self.AGENT_NAME}: DB error during analysis for ID {data_id}: {db_err}", exc_info=True)
        except Exception as e:
            logger.error(f"{self.AGENT_NAME}: Unexpected error during analysis for ID {data_id}: {e}", exc_info=True)
            # Attempt to save error state to DB if possible
            try:
                 async with self.session_maker() as error_session:
                      osint_data = await error_session.get(OSINTData, data_id)
                      if osint_data:
                           osint_data.analysis_results = json.dumps({"error": f"Analysis failed: {e}"})
                           osint_data.relevance = 'error'
                           await error_session.commit()
            except Exception as final_err:
                 logger.error(f"{self.AGENT_NAME}: Failed to even save error state for ID {data_id}: {final_err}")


    async def trigger_integrations(self, data_id: int, analysis: Dict):
        """Send actionable insights/leads to other agents."""
        logger.info(f"{self.AGENT_NAME}: Triggering integrations for analyzed data ID: {data_id}")
        leads = analysis.get('ugc_leads', {})
        contacts = leads.get('contacts', [])
        insights = leads.get('insights', [])

        if not contacts and not insights:
             logger.info(f"{self.AGENT_NAME}: No specific contacts or insights found in analysis for ID {data_id} to trigger integrations.")
             return

        # Prepare payload for other agents
        # Extract relevant info like target name/domain if available
        target_name = "Unknown Target" # Default
        try:
             async with self.session_maker() as session:
                  osint_record = await session.get(OSINTData, data_id)
                  if osint_record: target_name = osint_record.target
        except Exception as e:
             logger.warning(f"Could not fetch target name for data_id {data_id}: {e}")

        # Example: Send structured leads to EmailAgent
        email_agent = self.orchestrator.agents.get('email') # Assumes 'email' is the key for PuppeteerAgent
        if email_agent and hasattr(email_agent, 'queue_osint_leads'):
            # Format leads specifically for EmailAgent if needed
            email_leads = []
            for contact in contacts:
                 # Basic check if contact is likely an email
                 if isinstance(contact, str) and '@' in contact:
                      email_leads.append({"email": contact, "name": "OSINT Lead", "source": f"OSINT_{data_id}"})
                 # Add logic to extract names/roles if analysis provides them
            if email_leads:
                try:
                    # Pass list of lead dicts
                    await email_agent.queue_osint_leads(email_leads)
                    logger.info(f"{self.AGENT_NAME}: Sent {len(email_leads)} email leads for ID {data_id} to EmailAgent.")
                except Exception as e:
                    logger.error(f"{self.AGENT_NAME}: Failed to queue leads for EmailAgent (ID: {data_id}): {e}", exc_info=True)

        # Example: Send contacts/insights to VoiceSalesAgent
        voice_agent = self.orchestrator.agents.get('voice_sales')
        if voice_agent and hasattr(voice_agent, 'queue_osint_leads'): # Assuming method exists
             try:
                 # Voice agent might need phone numbers primarily, or context for calls
                 voice_payload = {"osint_data_id": data_id, "target": target_name, "contacts": contacts, "insights": insights}
                 await voice_agent.queue_osint_leads(voice_payload) # Adapt payload as needed
                 logger.info(f"{self.AGENT_NAME}: Sent leads/insights for ID {data_id} to VoiceSalesAgent.")
             except Exception as e:
                 logger.error(f"{self.AGENT_NAME}: Failed to queue leads for VoiceSalesAgent (ID: {data_id}): {e}", exc_info=True)

        # Add other integrations as needed (e.g., CRM update, notification to ThinkTool)


    # --- Visualization ---

    async def visualize_data(self, data_id: int):
        """Generate and save a visualization of the OSINT data (e.g., Maltego graph)."""
        logger.info(f"{self.AGENT_NAME}: Starting visualization for OSINT data ID: {data_id}")
        try:
            async with self.session_maker() as session:
                osint_data = await session.get(OSINTData, data_id)
                if not osint_data:
                    logger.error(f"{self.AGENT_NAME}: OSINT data ID {data_id} not found for visualization.")
                    return
                if not osint_data.raw_data or not osint_data.analysis_results:
                     logger.warning(f"{self.AGENT_NAME}: Missing raw data or analysis for ID {data_id}. Cannot visualize.")
                     return

                raw_data = json.loads(osint_data.raw_data)
                analysis = json.loads(osint_data.analysis_results)

                # --- Maltego Visualization ---
                # Check if maltego-trx is available and enabled
                if self.config.get("OSINT_ENABLE_MALTEGO_VIZ", False):
                     graph_data = await self.run_maltego_visualization(raw_data, analysis)
                     if graph_data and "error" not in graph_data:
                          # Ensure filename is safe
                          safe_target_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', osint_data.target)
                          filename = f"maltego_{safe_target_filename}_{data_id}.graphml"
                          viz_path = await self.save_visualization(graph_data, filename) # Save as graphml
                          if viz_path:
                               # Update DB record with path
                               stmt = sql_text("UPDATE osint_data SET visualization_path = :path WHERE id = :id")
                               await session.execute(stmt, {"path": viz_path, "id": data_id})
                               await session.commit()
                               logger.info(f"{self.AGENT_NAME}: Maltego visualization saved for ID {data_id} at {viz_path}")
                          else:
                               logger.error(f"{self.AGENT_NAME}: Failed to save Maltego visualization for ID {data_id}")
                     elif graph_data and "error" in graph_data:
                          logger.error(f"{self.AGENT_NAME}: Maltego visualization failed for ID {data_id}: {graph_data['error']}")
                else:
                     logger.info(f"{self.AGENT_NAME}: Maltego visualization disabled. Skipping.")

                # --- Add other visualization types here (e.g., network graph, timeline) ---

        except sqlalchemy.exc.SQLAlchemyError as db_err:
             logger.error(f"{self.AGENT_NAME}: DB error during visualization for ID {data_id}: {db_err}", exc_info=True)
        except Exception as e:
            logger.error(f"{self.AGENT_NAME}: Unexpected error during visualization for ID {data_id}: {e}", exc_info=True)


    async def run_maltego_visualization(self, raw_data: Dict, analysis_results: Dict) -> Optional[Dict]:
        """Generate a Maltego graph structure (as dict/XML) from data."""
        logger.debug(f"Generating Maltego visualization data...")
        try:
            from maltego_trx.maltego import MaltegoTransform, MaltegoMsg
            from maltego_trx.entities import Domain, EmailAddress, Person, PhoneNumber, Phrase, URL, IPv4Address # Add more as needed

            response = MaltegoTransform() # No request needed for generation

            # Add entities from raw data (example parsers)
            target_entity = None # Track primary target if possible
            target_str = raw_data.get('target', 'Unknown Target') # Assuming target is stored
            # Determine entity type based on target format (basic)
            if '@' in target_str and '.' in target_str: target_entity = response.addEntity(EmailAddress, target_str)
            elif '.' in target_str: target_entity = response.addEntity(Domain, target_str)
            else: target_entity = response.addEntity(Phrase, target_str)
            if target_entity: target_entity.setBookmark(1) # Bookmark primary target

            for tool, data in raw_data.items():
                 if isinstance(data, dict) and "error" not in data:
                      if tool == 'theHarvester':
                           for email in data.get('emails', []): response.addEntity(EmailAddress, email)
                           for host in data.get('hosts', []): response.addEntity(Domain, host) # Or Website?
                           for ip in data.get('ips', []): response.addEntity(IPv4Address, ip)
                      elif tool == 'Sherlock':
                           for site_name, site_info in data.items(): # Iterate through found sites
                                if isinstance(site_info, dict) and site_info.get('url'):
                                     site_entity = response.addEntity(URL, site_info['url'])
                                     site_entity.addProperty('platform', 'Platform', 'strict', site_name)
                      # Add parsers for other tools (Shodan, Recon-ng, etc.)
                      elif tool == 'Shodan':
                           for match in data.get('matches', []):
                                ip = match.get('ip_str', '')
                                if ip:
                                     entity = response.addEntity(IPv4Address, ip)
                                     entity.addProperty('port', 'Port', 'strict', str(match.get('port', '')))
                                     entity.addProperty('hostname', 'Hostname', 'strict', ", ".join(match.get('hostnames', [])))
                                     entity.addProperty('org', 'Organization', 'strict', match.get('org', ''))
                      elif tool == 'Reconng': # Match the key used in run_single_tool_wrapper
                           for host in data.get('found_hosts', []):
                                if host: response.addEntity(Domain, host) # Assume domain/host

            # Add entities/links from analysis (example)
            if 'ugc_leads' in analysis_results:
                 for contact in analysis_results['ugc_leads'].get('contacts', []):
                      # Try to determine contact type
                      contact_entity = None
                      if isinstance(contact, str):
                           if '@' in contact: contact_entity = response.addEntity(EmailAddress, contact)
                           elif contact.replace('+','').isdigit(): contact_entity = response.addEntity(PhoneNumber, contact)
                           else: contact_entity = response.addEntity(Person, contact) # Assume name
                      elif isinstance(contact, dict): # Handle structured contacts if analysis provides them
                           name = contact.get('name')
                           email = contact.get('email')
                           phone = contact.get('phone')
                           if name: contact_entity = response.addEntity(Person, name)
                           if email:
                                email_entity = response.addEntity(EmailAddress, email)
                                if contact_entity: email_entity.addLink(targetEntity=contact_entity, transformName="Contact Email")
                           if phone:
                                phone_entity = response.addEntity(PhoneNumber, phone)
                                if contact_entity: phone_entity.addLink(targetEntity=contact_entity, transformName="Contact Phone")

                      if contact_entity: contact_entity.setLinkColor("0x00FF00") # Green link


            # Return the graph data (e.g., as XML string or structured dict)
            maltego_xml = response.returnOutput()
            logger.info("Maltego visualization generated successfully")
            # Parse maltego_output (XML) if needed, or return a simplified dict
            return {"maltego_xml_output": maltego_xml} # Return raw XML for saving

        except ImportError:
            logger.error("maltego-trx library not installed. Cannot generate Maltego visualization.")
            return {"error": "maltego-trx library not installed"}
        except Exception as e:
            logger.error(f"Maltego visualization generation failed: {e}", exc_info=True)
            return {"error": str(e)}


    async def save_visualization(self, graph_data: Dict, filename: str) -> Optional[str]:
        """Save visualization data to a file."""
        viz_dir = self.config.get("OSINT_VIZ_SAVE_DIR", "visualizations")
        try:
            os.makedirs(viz_dir, exist_ok=True)
            # Sanitize filename further
            safe_filename = re.sub(r'[^\w\.-]', '_', filename)
            save_path = os.path.join(viz_dir, safe_filename)

            # Assuming graph_data contains the XML string
            xml_output = graph_data.get("maltego_xml_output")
            if not xml_output:
                 logger.error("No Maltego XML output found in graph_data to save.")
                 return None

            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(xml_output) # Write the XML string

            logger.info(f"Visualization saved to: {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save visualization to {filename}: {e}", exc_info=True)
            return None

    # --- Workflow & Insights ---

    async def run_osint_workflow(self, target: str, tools: Optional[List[str]] = None):
        """Run the full collect -> analyze -> visualize workflow."""
        await self.log_operation('info', f"Starting full OSINT workflow for target: '{target}'")
        data_id = await self.collect_data(target, tools)
        if data_id:
            await self.log_operation('info', f"Data collection complete (ID: {data_id}). Proceeding to analysis.")
            await self.analyze_data(data_id)
            await self.log_operation('info', f"Analysis complete (ID: {data_id}). Proceeding to visualization.")
            await self.visualize_data(data_id)
            await self.log_operation('info', f"Visualization complete (ID: {data_id}). Workflow finished for '{target}'.")
        else:
            await self.log_operation('warning', f"Data collection failed for '{target}'. Workflow aborted.")


    async def get_insights(self) -> Dict[str, Any]:
        """Provide high-level insights about OSINT operations."""
        # Example: Return recent activity summary
        recent_tasks_count = 0
        recent_high_relevance_count = 0
        try:
            async with self.session_maker() as session:
                threshold = datetime.now(timezone.utc) - timedelta(hours=24)
                # Count recent tasks
                stmt_total = sql_text("SELECT COUNT(*) FROM osint_data WHERE timestamp >= :threshold")
                result_total = await session.execute(stmt_total, {"threshold": threshold})
                recent_tasks_count = result_total.scalar_one_or_none() or 0
                # Count recent high relevance tasks
                stmt_high = sql_text("SELECT COUNT(*) FROM osint_data WHERE timestamp >= :threshold AND relevance = 'high'")
                result_high = await session.execute(stmt_high, {"threshold": threshold})
                recent_high_relevance_count = result_high.scalar_one_or_none() or 0
        except Exception as e:
             logger.error(f"{self.AGENT_NAME}: Failed to get DB insights: {e}", exc_info=True)

        return {
            "agent_name": self.AGENT_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tasks_in_queue": self.internal_state['task_queue'].qsize(), # Access via internal_state
            "recent_tasks_24h": recent_tasks_count,
            "recent_high_relevance_24h": recent_high_relevance_count,
            "tool_success_rates": {k: round(v, 2) for k, v in self.internal_state['tool_success_rates'].items()}, # Access via internal_state
            "aggressive_tools_status": {
                 "GoogleDorking": self.internal_state['enable_google_dorking'], # Access via internal_state
                 "SocialMediaScraping": self.internal_state['enable_social_scraping'], # Access via internal_state
                 "DarkPoolSearch": self.internal_state['enable_dark_pool'], # Access via internal_state
                 "WebcamSearch": self.internal_state['enable_webcam_search'] # Access via internal_state
            }
        } # Correctly close the dictionary

    # --- Abstract Method Implementations (Placeholders) ---

    async def execute_task(self, task_details: Dict[str, Any]) -> Any:
        """Core method to execute the agent's primary function for a given task."""
        self.logger.info(f"execute_task received task: {task_details}")
        # Map task details to existing methods or queue them internally
        task_type = task_details.get('type')
        target = task_details.get('target')
        tools = task_details.get('tools')
        data_id = task_details.get('data_id')

        if task_type == "run_workflow" and target:
            # Directly run the workflow for simplicity in this example
            # Consider running as background task if long-running
            asyncio.create_task(self.run_osint_workflow(target, tools))
            return {"status": "workflow initiated"} # Workflow runs async
        elif task_type == "collect_data" and target:
            # Run collection as background task
            asyncio.create_task(self.collect_data(target, tools))
            return {"status": "collection initiated"}
        elif task_type == "analyze_data" and data_id:
            # Run analysis as background task
            asyncio.create_task(self.analyze_data(data_id))
            return {"status": "analysis initiated"} # Analysis runs async
        elif task_type == "visualize_data" and data_id:
             # Run visualization as background task
            asyncio.create_task(self.visualize_data(data_id))
            return {"status": "visualization initiated"} # Visualization runs async
        else:
            # Alternatively, queue tasks internally if agent manages its own queue
            # await self.request_osint_task(task_type, **task_details)
            # return {"status": "queued"}
            self.logger.warning(f"Unsupported task type or missing args for OSINTAgent execute_task: {task_details}")
            return {"status": "failed", "reason": f"Unsupported task or missing args: {task_type}"}


    async def learning_loop(self):
        """Periodic loop to update tool effectiveness or analyze patterns."""
        self.logger.info("learning_loop: Placeholder - Not yet implemented.")
        # TODO: Implement logic to:
        # 1. Analyze success/failure rates of tools based on collected data quality/relevance from KB.
        # 2. Update self.internal_state['tool_success_rates'] more intelligently.
        # 3. Potentially query KB for broad patterns across OSINT data using self.kb_interface.
        await asyncio.sleep(3600 * 4) # Example: Run every 4 hours

    async def self_critique(self) -> Dict[str, Any]:
        """Method for the agent to evaluate its own performance and strategy."""
        self.logger.info("self_critique: Placeholder - Not yet implemented.")
        # TODO: Implement logic to analyze:
        # - Overall data yield vs. cost/time (query ExpenseLog via KB/DB).
        # - Relevance scores of collected data (query OSINTData via KB/DB).
        # - Effectiveness of tool selection strategy (compare success rates vs outcomes).
        # - Success rate of aggressive techniques vs. blocks encountered.
        # Use self.kb_interface to fetch necessary data.
        return {"status": "ok", "feedback": "Self-critique not implemented."}

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Constructs context-rich prompts for LLM calls (e.g., data analysis)."""
        self.logger.debug(f"Generating dynamic prompt for OSINT task: {task_context.get('task')}")
        # Start with the base meta-prompt
        prompt_parts = [self.internal_state.get('meta_prompt', "Analyze OSINT data.")]

        # Add relevant context from the task
        prompt_parts.append("\n--- Current Task Context ---")
        for key, value in task_context.items():
             if key == 'raw_data_str_for_prompt': # Handle potentially large data
                  prompt_parts.append(f"\n**Collected Data (JSON, potentially truncated):**")
                  prompt_parts.append("```json")
                  prompt_parts.append(value) # Already truncated if needed
                  prompt_parts.append("```")
             elif isinstance(value, (str, int, float, bool)):
                  prompt_parts.append(f"{key.replace('_', ' ').title()}: {value}")
             # Add more specific handling for other expected context types if needed

        # Add relevant context from KB (Simulated)
        # prompt_parts.append("\n--- Relevant Knowledge (Simulated KB Retrieval) ---")
        # Example: Fetch known info about the target before analysis
        # if self.kb_interface and task_context.get('target'):
        #     known_data = await self.kb_interface.get_knowledge(query=f"Info on {task_context['target']}", limit=5)
        #     if known_data:
        #         prompt_parts.append("Previously known info:")
        #         for item in known_data: prompt_parts.append(f"- {item['content'][:100]}...")

        # Add Specific Instructions based on task
        prompt_parts.append("\n--- Instructions ---")
        if task_context.get('task') == 'Analyze OSINT Data':
             prompt_parts.append("1. Summarize the key findings from the 'Collected Data'.")
             prompt_parts.append("2. Identify potential leads or actionable insights relevant to a UGC agency (contacts, roles, company info, tech stack).")
             prompt_parts.append("3. Assess the overall data quality and reliability (Low, Medium, High).")
             prompt_parts.append("4. Provide a concise overall summary.")
             prompt_parts.append(f"5. **Output Format:** {task_context.get('desired_output_format', 'JSON object as specified previously.')}")
        else:
             prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")

        prompt_parts.append("```json") # Hint for the LLM if JSON is expected

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for OSINTAgent (length: {len(final_prompt)} chars)")
        return final_prompt
