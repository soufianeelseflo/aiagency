        # Filename: agents/osint_agent.py
        # Description: OSINT Agent for gathering, analyzing, and exploiting public data.
        # Version: 1.2 (Implemented Core Logic, Tool Execution, Analysis, Grey Areas)

        import asyncio
        import logging
        import random
        import os
        import json
        import time
        import uuid
        import re
        import subprocess
        import shlex # For safe command splitting
        from datetime import datetime, timedelta, timezone
        from urllib.parse import urlparse # For service name extraction

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
        from sqlalchemy.future import select
        from sqlalchemy import update, text
        from sqlalchemy.exc import SQLAlchemyError

        # Assuming utils/database.py and models.py exist as provided
        from models import OSINTData, KnowledgeFragment # Add other models if needed (e.g., Account for API keys)
        # Using adaptable name for flexibility, assuming OpenAI compatible interface
        from openai import AsyncOpenAI as AsyncLLMClient
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

        # --- Web Scraping / HTTP Requests ---
        import aiohttp
        from bs4 import BeautifulSoup
        try:
            from fake_useragent import UserAgent, FakeUserAgentError
        except ImportError:
            logger.warning("fake_useragent not installed. Using default User-Agent.")
            UserAgent = None
            FakeUserAgentError = None


        # --- OSINT Tool Libraries/Wrappers (Ensure installed/available) ---
        try:
            import shodan
        except ImportError:
            logger.warning("shodan library not installed. Shodan tool will be unavailable.")
            shodan = None
        # Add imports for other library-based tools if used (e.g., python-exiftool)
        # import exiftool # Example

        # --- Base Agent Import ---
        try:
            from .base_agent import GeniusAgentBase, KBInterface # Use relative import if applicable
        except ImportError:
            from base_agent import GeniusAgentBase, KBInterface # Fallback

        # --- Meta Prompt ---
        # Define the updated meta prompt string
        OSINT_AGENT_META_PROMPT_UPDATED = """
        You are the OSINTAgent, a specialized intelligence-gathering unit within an autonomous AI agency. Your primary function is to collect, analyze, and exploit publicly available information (OSINT) to support the agency's strategic goals, with a strong emphasis on identifying high-value market needs and competitor weaknesses.
        **Core Directives:**
        1.  **Targeted Data Collection:** Utilize a diverse range of OSINT tools and techniques (web scraping, social media analysis, API interactions, specialized tool wrappers like theHarvester, Shodan, SpiderFoot, etc.) to gather data relevant to specific targets (individuals, companies, markets). Prioritize sources likely to reveal market needs, customer pain points, or competitor vulnerabilities.
        2.  **Strategic Analysis:** Go beyond simple data aggregation. Analyze collected information to: Identify High-Value Market Needs, Assess Competitor Weaknesses, Generate Actionable Leads (emails, social profiles, context), Map Relationships.
        3.  **Tool Selection & Optimization:** Intelligently select the most appropriate tools for the target and objective. Adapt tool usage based on performance feedback. Request installation of missing CLI tools via Orchestrator.
        4.  **Knowledge Base Integration:** Log structured findings, raw data, and analysis results into the agency's central Knowledge Base (KBInterface), tagging appropriately.
        5.  **Compliance & Ethics:** Operate within defined legal and ethical boundaries. Use aggressive techniques (dorking, leak monitoring) only when enabled and strategically necessary, assessing risks and consulting LegalAgent via Orchestrator for sensitive data usage.
        6.  **Efficiency:** Optimize data collection and analysis processes for speed and cost-effectiveness.
        **Goal:** Provide timely, relevant, and actionable intelligence that directly informs strategic decisions, identifies market opportunities, highlights competitor vulnerabilities, and fuels sales/marketing efforts. Focus on quality over quantity.
        """

        # Configure logging
        logger = logging.getLogger(__name__)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)


        class OSINTAgent(GeniusAgentBase):
            """
            OSINT Agent (Genius Level): Collects, analyzes, and exploits public data using
            advanced techniques and tools, integrating findings into the central Knowledge Base.
            Version: 1.2
            """
            AGENT_NAME = "OSINTAgent"

            def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any, kb_interface: KBInterface, shodan_api_key: Optional[str], spiderfoot_api_key: Optional[str]):
                """Initializes the OSINTAgent."""
                super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, kb_interface=kb_interface, session_maker=session_maker)

                # Store passed-in secrets securely (or reference from secure_storage if preferred)
                self._shodan_api_key = shodan_api_key
                self._spiderfoot_api_key = spiderfoot_api_key

                # --- Internal State Initialization ---
                self.internal_state['task_queue'] = asyncio.Queue()
                self.internal_state['max_concurrency'] = int(self.config.get("OSINT_MAX_CONCURRENT_TOOLS", 5))
                self.internal_state['tool_semaphore'] = asyncio.Semaphore(self.internal_state['max_concurrency'])
                self.internal_state['meta_prompt'] = OSINT_AGENT_META_PROMPT_UPDATED

                # Tool Configuration
                self.internal_state['spiderfoot_url'] = self.config.get("SPIDERFOOT_URL", "http://localhost:5001") # URL is config

                # Feature Flags
                self.internal_state['enable_google_dorking'] = bool(self.config.get("OSINT_ENABLE_GOOGLE_DORKING", True)) # Enable by default
                self.internal_state['enable_social_scraping'] = bool(self.config.get("OSINT_ENABLE_SOCIAL_SCRAPING", True)) # Enable by default
                self.internal_state['enable_leak_monitoring'] = bool(self.config.get("OSINT_ENABLE_LEAK_MONITORING", True)) # Enable by default
                # Disabled by default due to high risk/complexity
                self.internal_state['enable_dark_pool'] = bool(self.config.get("OSINT_ENABLE_DARK_POOL", False))
                self.internal_state['enable_webcam_search'] = bool(self.config.get("OSINT_ENABLE_WEBCAM_SEARCH", False))

                # Tool Success Rates (Initialize, learning loop can refine)
                self.internal_state['tool_success_rates'] = {
                    'theHarvester': 0.7, 'ExifTool': 0.6, 'Photon': 0.5, 'Sherlock': 0.7,
                    'Maltego': 0.6, 'Shodan': 0.8, 'Reconng': 0.7, 'SpiderFoot': 0.75,
                    'GoogleDorking': 0.6, 'SocialMediaScraping': 0.5, 'LeakMonitoring': 0.4,
                    'DarkPoolSearch': 0.0, 'WebcamSearch': 0.0
                }

                # --- User Agent ---
                self.user_agent_generator = None
                if UserAgent:
                    try:
                        self.user_agent_generator = UserAgent()
                    except FakeUserAgentError:
                        self.logger.warning(f"{self.AGENT_NAME}: Failed to initialize fake_useragent cache. Using default UA string.")
                    except Exception as ua_err:
                        self.logger.warning(f"{self.AGENT_NAME}: Error initializing fake_useragent ({ua_err}). Using default UA string.")

                self.logger.info(f"{self.AGENT_NAME} (Genius Level) v1.2 initialized. Max concurrent tools: {self.internal_state['max_concurrency']}. "
                            f"Aggressive tools enabled: Dorking={self.internal_state['enable_google_dorking']}, Social={self.internal_state['enable_social_scraping']}, LeakMon={self.internal_state['enable_leak_monitoring']}")
                if self.internal_state['enable_dark_pool'] or self.internal_state['enable_webcam_search']:
                    self.logger.critical(f"{self.AGENT_NAME}: SECURITY/LEGAL WARNING: Dark Pool or Webcam Search enabled. Ensure full compliance and risk assessment.")

            # --- Tool Execution & Management ---

            async def _check_tool_availability(self, tool_name_cli: str) -> bool:
                """Checks if a CLI tool is available in the environment."""
                if not self.orchestrator or not hasattr(self.orchestrator, 'use_tool'):
                    self.logger.error("Orchestrator tool execution unavailable, cannot check tool availability.")
                    return False # Assume not available if cannot check
                try:
                    # Use 'which' command via orchestrator's tool executor
                    result = await self.orchestrator.use_tool('execute_command', {'command': f'which {shlex.quote(tool_name_cli)}'})
                    if result and result.get('status') == 'success' and result.get('stdout'):
                        self.logger.debug(f"Tool '{tool_name_cli}' found at: {result['stdout'].strip()}")
                        return True
                    else:
                        self.logger.debug(f"Tool '{tool_name_cli}' not found via 'which'.")
                        return False
                except Exception as e:
                    self.logger.error(f"Error checking availability for tool '{tool_name_cli}': {e}")
                    return False

            async def _request_tool_installation(self, tool_name: str, package_manager: str = 'apt', package_name: Optional[str] = None, git_repo: Optional[str] = None):
                """Requests the Orchestrator to install a missing tool via ProgrammerAgent."""
                if not self.orchestrator or not hasattr(self.orchestrator, 'handle_install_tool_request'):
                    self.logger.error("Orchestrator cannot handle tool installation requests.")
                    return
                self.logger.info(f"Requesting installation of tool '{tool_name}'...")
                await self.orchestrator.handle_install_tool_request(
                    requesting_agent_name=self.AGENT_NAME,
                    tool_details={
                        "tool_name": tool_name,
                        "package_manager": package_manager,
                        "package_name": package_name or tool_name.lower(), # Default convention
                        "git_repo": git_repo
                    }
                )

            async def run_single_tool_wrapper(self, tool_name: str, target: str, results_dict: Dict):
                """Acquires semaphore, runs tool with retry, stores result/error, updates success."""
                async with self.internal_state['tool_semaphore']:
                    self.logger.debug(f"{self.AGENT_NAME}: Running tool '{tool_name}' for target '{target}'")
                    tool_data = {}
                    success = False
                    error_info = None
                    try:
                        method_name = f"run_{tool_name}"
                        tool_func = None
                        for attr_name in dir(self):
                            if attr_name.lower() == method_name.lower():
                                potential_func = getattr(self, attr_name)
                                if callable(potential_func): tool_func = potential_func; break

                        if tool_func:
                            tool_data = await self.run_tool_with_retry(tool_func, target)
                            success = bool(tool_data) and "error" not in tool_data
                        else:
                            error_info = f"No implementation found for tool: {tool_name}"
                            self.logger.warning(f"{self.AGENT_NAME}: {error_info}")

                    except Exception as e:
                        self.logger.error(f"{self.AGENT_NAME}: Tool '{tool_name}' failed for '{target}': {e}", exc_info=False)
                        error_info = str(e)
                        success = False

                    results_dict[tool_name] = tool_data if success else {"error": error_info or "Tool execution failed"}
                    await self.update_tool_success(tool_name, success)
                    self.logger.debug(f"{self.AGENT_NAME}: Finished tool '{tool_name}' for target '{target}'. Success: {success}")

            @retry(
                wait=wait_exponential(multiplier=1, min=2, max=30),
                stop=stop_after_attempt(3),
                retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, subprocess.TimeoutExpired, shodan.APIError if shodan else Exception, Exception)),
                reraise=True
            )
            async def run_tool_with_retry(self, tool_func: callable, *args, **kwargs) -> Dict:
                """Wrapper to run a specific tool function with tenacity retry logic."""
                self.logger.debug(f"Executing {tool_func.__name__} with args: {args}, kwargs: {kwargs}")
                return await tool_func(*args, **kwargs)

            async def update_tool_success(self, tool: str, success: bool):
                """Rudimentary update of tool success rate in internal state."""
                rate = self.internal_state['tool_success_rates'].get(tool, 0.5)
                adjustment = 0.05 if success else -0.05
                new_rate = min(max(rate + adjustment, 0.1), 1.0)
                self.internal_state['tool_success_rates'][tool] = new_rate
                self.logger.debug(f"{self.AGENT_NAME}: Updated success rate for {tool}: {new_rate:.2f}")

            async def select_best_tools(self, target: str, num_tools: int = 3) -> List[str]:
                """Select top N enabled tools based on perceived success rate."""
                enabled_tools = {tool: rate for tool, rate in self.internal_state['tool_success_rates'].items() if self._is_tool_enabled(tool)}
                # Add simple logic: if target is email, prioritize theHarvester/Sherlock
                if '@' in target:
                    priority_tools = ['theHarvester', 'Sherlock']
                    sorted_tools = priority_tools + sorted([t for t in enabled_tools if t not in priority_tools], key=enabled_tools.get, reverse=True)
                # If target looks like domain, prioritize Reconng, Shodan, Spiderfoot
                elif '.' in target and '/' not in target:
                    priority_tools = ['Reconng', 'Shodan', 'SpiderFoot', 'theHarvester']
                    sorted_tools = priority_tools + sorted([t for t in enabled_tools if t not in priority_tools], key=enabled_tools.get, reverse=True)
                else: # Default sort by rate
                    sorted_tools = sorted(enabled_tools, key=enabled_tools.get, reverse=True)

                selected = [t for t in sorted_tools if self._is_tool_enabled(t)][:num_tools] # Ensure selected are enabled
                self.logger.debug(f"{self.AGENT_NAME}: Selected tools for '{target}': {selected}")
                return selected

            def _is_tool_enabled(self, tool_name: str) -> bool:
                """Check if a specific tool is enabled via config flags or API key presence."""
                if tool_name == 'GoogleDorking': return self.internal_state['enable_google_dorking']
                if tool_name == 'SocialMediaScraping': return self.internal_state['enable_social_scraping']
                if tool_name == 'LeakMonitoring': return self.internal_state['enable_leak_monitoring']
                if tool_name == 'DarkPoolSearch': return self.internal_state['enable_dark_pool']
                if tool_name == 'WebcamSearch': return self.internal_state['enable_webcam_search']
                if tool_name == 'Shodan': return bool(self._shodan_api_key) and shodan is not None
                if tool_name == 'SpiderFoot': return bool(self._spiderfoot_api_key) and bool(self.internal_state.get('spiderfoot_url'))
                # Assume CLI tools are potentially installable if not explicitly disabled
                if tool_name in ['theHarvester', 'ExifTool', 'Photon', 'Sherlock', 'Reconng']: return True
                # Maltego requires specific setup, disable by default unless configured
                if tool_name == 'Maltego': return bool(self.config.get("OSINT_ENABLE_MALTEGO", False))
                return False # Default to disabled for unknown tools

            # --- Tool Implementations ---

            async def _execute_command(self, command_list: List[str], timeout: int = 300) -> Tuple[Optional[str], Optional[str], int]:
                """Executes a shell command safely using asyncio.create_subprocess_exec."""
                cmd_str = " ".join(shlex.quote(part) for part in command_list)
                self.logger.debug(f"Executing command: {cmd_str}")
                process = await asyncio.create_subprocess_exec(
                    *command_list,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                    return stdout.decode(errors='ignore'), stderr.decode(errors='ignore'), process.returncode
                except asyncio.TimeoutError:
                    self.logger.error(f"Command timed out after {timeout}s: {cmd_str}")
                    try:
                        process.kill()
                        await process.wait() # Ensure process is cleaned up
                    except ProcessLookupError: pass # Process already finished
                    except Exception as kill_err: self.logger.warning(f"Error killing timed-out process: {kill_err}")
                    return None, f"Command timed out after {timeout}s", -1 # Indicate timeout
                except Exception as e:
                    self.logger.error(f"Error executing command '{cmd_str}': {e}", exc_info=True)
                    return None, str(e), -2 # Indicate execution error

            async def run_theHarvester(self, target: str) -> Dict:
                """Run theHarvester CLI tool."""
                tool_cli_name = "theHarvester"
                if not await self._check_tool_availability(tool_cli_name):
                    await self._request_tool_installation(tool_cli_name, package_manager='pip', package_name='theHarvester')
                    # Wait a bit for potential installation before retrying (or rely on tenacity)
                    await asyncio.sleep(5)
                    if not await self._check_tool_availability(tool_cli_name):
                        return {"error": f"{tool_cli_name} not found and installation failed or pending."}

                self.logger.info(f"Running theHarvester for: {target}")
                safe_target_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', target)
                output_file = f"/tmp/harvester_{safe_target_filename}_{uuid.uuid4()}.json"
                command = [ tool_cli_name, "-d", target, "-l", "100", "-b", "google,bing,linkedin", "-f", output_file ]
                results = {}
                try:
                    stdout, stderr, returncode = await self._execute_command(command, timeout=300) # 5 min timeout

                    if returncode != 0:
                        logger.error(f"theHarvester failed for {target}. Code: {returncode}. Stderr: {stderr}")
                        return {"error": f"theHarvester exited with code {returncode}", "stderr": stderr}

                    if os.path.exists(output_file):
                        try:
                            with open(output_file, 'r', encoding='utf-8') as f: results = json.load(f)
                        except Exception as read_err:
                            logger.error(f"Failed to read/parse theHarvester output {output_file}: {read_err}")
                            results = {"error": "Failed to parse JSON output", "stdout": stdout}
                        finally:
                            try: os.remove(output_file)
                            except OSError as rm_err: logger.warning(f"Failed to remove temp file {output_file}: {rm_err}")
                    else:
                        logger.warning(f"theHarvester output file not found: {output_file}")
                        results = {"stdout": stdout} # Fallback

                    self.logger.info(f"theHarvester completed for {target}. Found emails: {len(results.get('emails',[]))}, hosts: {len(results.get('hosts',[]))}")
                    return results
                except Exception as e:
                    logger.error(f"theHarvester execution wrapper failed for {target}: {e}", exc_info=True)
                    if 'output_file' in locals() and os.path.exists(output_file):
                        try: os.remove(output_file)
                        except OSError: pass
                    return {"error": str(e)}

            async def run_ExifTool(self, target_url: str) -> Dict:
                """Run ExifTool on a downloadable URL."""
                tool_cli_name = "exiftool"
                if not await self._check_tool_availability(tool_cli_name):
                    await self._request_tool_installation(tool_cli_name, package_manager='apt', package_name='libimage-exiftool-perl')
                    await asyncio.sleep(5)
                    if not await self._check_tool_availability(tool_cli_name):
                        return {"error": f"{tool_cli_name} not found and installation failed or pending."}

                self.logger.info(f"Attempting ExifTool analysis for target URL: {target_url}")
                if not (target_url.startswith("http://") or target_url.startswith("https://")):
                    return {"error": "ExifTool requires a downloadable URL."}

                temp_file_path = f"/tmp/exiftool_{uuid.uuid4()}"
                try:
                    # Download file
                    async with aiohttp.ClientSession() as session:
                        async with session.get(target_url, timeout=30) as response:
                            if response.status == 200:
                                with open(temp_file_path, 'wb') as f:
                                        while True:
                                            chunk = await response.content.read(1024*1024) # Read 1MB chunks
                                            if not chunk: break
                                            f.write(chunk)
                                self.logger.debug(f"Downloaded file from {target_url} to {temp_file_path}")
                            else:
                                return {"error": f"Failed to download file (Status: {response.status})"}

                    # Run ExifTool
                    command = [tool_cli_name, "-json", temp_file_path]
                    stdout, stderr, returncode = await self._execute_command(command, timeout=60)

                    if returncode != 0:
                        logger.error(f"ExifTool failed for {temp_file_path}. Code: {returncode}. Stderr: {stderr}")
                        return {"error": f"ExifTool exited with code {returncode}", "stderr": stderr}

                    try:
                        # Exiftool outputs a JSON array
                        metadata_list = json.loads(stdout)
                        logger.info(f"ExifTool completed for downloaded file from {target_url}")
                        return metadata_list[0] if metadata_list else {} # Return first item's metadata
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode ExifTool JSON output. Stdout: {stdout[:500]}...")
                        return {"error": "Failed to parse ExifTool JSON output", "stdout": stdout}

                except Exception as e:
                    logger.error(f"ExifTool processing failed for {target_url}: {e}", exc_info=True)
                    return {"error": f"ExifTool processing failed: {str(e)}"}
                finally:
                    if os.path.exists(temp_file_path):
                        try: os.remove(temp_file_path)
                        except OSError: pass

            async def run_Sherlock(self, username: str) -> Dict:
                """Run Sherlock username reconnaissance CLI tool."""
                tool_cli_name = "sherlock"
                if not await self._check_tool_availability(tool_cli_name):
                    await self._request_tool_installation(tool_cli_name, package_manager='pip', package_name='sherlock-project')
                    await asyncio.sleep(5)
                    if not await self._check_tool_availability(tool_cli_name):
                        return {"error": f"{tool_cli_name} not found and installation failed or pending."}

                self.logger.info(f"Running Sherlock for username: {username}")
                safe_username = re.sub(r'[^a-zA-Z0-9_-]', '_', username)
                # Output directly to stdout/stderr, parse stdout
                command = [ tool_cli_name, "--no-color", "--timeout", "10", username ] # Increased timeout
                results = {}
                try:
                    stdout, stderr, returncode = await self._execute_command(command, timeout=600) # 10 min timeout

                    if returncode != 0 and not stdout: # Sometimes returns non-zero even with results
                        logger.error(f"Sherlock failed for {username}. Code: {returncode}. Stderr: {stderr}")
                        return {"error": f"Sherlock exited with code {returncode}", "stderr": stderr}

                    # Parse stdout for found URLs
                    found_sites = {}
                    lines = stdout.splitlines() if stdout else []
                    for line in lines:
                        if "[+]" in line and "http" in line:
                            try:
                                # Example line: "[+] GitHub: https://github.com/username"
                                parts = line.split(': ')
                                site_name = parts[0].split(']')[1].strip()
                                url = parts[1].strip()
                                found_sites[site_name] = {"url": url, "status": "found"}
                            except Exception:
                                logger.warning(f"Could not parse Sherlock output line: {line}")

                    results = found_sites
                    self.logger.info(f"Sherlock completed for {username}. Found {len(results)} potential profiles.")
                    return results
                except Exception as e:
                    logger.error(f"Sherlock execution wrapper failed for {username}: {e}", exc_info=True)
                    return {"error": str(e)}

            async def run_Reconng(self, target: str) -> Dict:
                """Run Recon-ng CLI tool with a specific module."""
                tool_cli_name = "recon-ng"
                # Recon-ng installation is complex (dependencies), assume it's pre-installed via Dockerfile
                if not await self._check_tool_availability(tool_cli_name):
                    # Cannot easily request installation here
                    return {"error": f"{tool_cli_name} not found. Manual installation required."}

                self.logger.info(f"Running Recon-ng for target: {target}")
                module = self.config.get("RECONNG_MODULE", "recon/domains-hosts/hackertarget")
                safe_target = ''.join(c for c in target if c.isalnum() or c in ['.', '-', '_'])
                if not safe_target: return {"error": "Invalid target for Recon-ng"}

                workspace_name = f"osint_{safe_target}_{uuid.uuid4()}"
                # Use -C to execute commands directly
                command_str = f"workspaces create {workspace_name}; modules load {module}; options set SOURCE {safe_target}; run; show hosts; workspaces delete {workspace_name}; exit"
                command = [tool_cli_name, "-C", command_str]
                results = {}
                try:
                    stdout, stderr, returncode = await self._execute_command(command, timeout=300) # 5 min timeout

                    if returncode != 0:
                        logger.error(f"Recon-ng failed for {target}. Code: {returncode}. Stderr: {stderr}")
                        return {"error": f"Recon-ng exited with code {returncode}", "stderr": stderr}

                    # Parse stdout for hosts
                    output_lines = stdout.splitlines() if stdout else []
                    hosts = []
                    in_table = False
                    header_skipped = False
                    for line in output_lines:
                        line_strip = line.strip()
                        if line_strip.startswith('+--') or line_strip.startswith('----'): # Detect table separator/header end
                            in_table = not in_table
                            if not in_table: header_skipped = False # Reset header skip when leaving table
                            continue
                        if in_table:
                            if not header_skipped: # Skip the header row itself
                                header_skipped = True
                                continue
                            if line_strip and not line_strip.startswith('|') and not line_strip.startswith('+'): # Basic check for data row
                                parts = line_strip.split()
                                if len(parts) > 0: hosts.append(parts[0]) # Assume first column

                    results['reconng_output_snippet'] = "\n".join(output_lines[:50]) # Store snippet
                    results['found_hosts'] = sorted(list(set(hosts))) # Store unique sorted hosts

                    self.logger.info(f"Recon-ng ({module}) completed for {target}. Found {len(results['found_hosts'])} potential hosts.")
                    return results
                except Exception as e:
                    logger.error(f"Recon-ng execution wrapper failed for {target}: {e}", exc_info=True)
                    return {"error": str(e)}

            async def run_Shodan(self, query: str) -> Dict:
                """Run Shodan search using the API."""
                if not self._shodan_api_key or not shodan:
                    return {"error": "Shodan API key not available or library not installed."}
                self.logger.info(f"Running Shodan search for query: {query}")
                try:
                    api = shodan.Shodan(self._shodan_api_key)
                    results = {'matches': [], 'total': 0}
                    count = 0
                    limit = int(self.config.get("SHODAN_RESULT_LIMIT", 100))
                    # Run synchronous Shodan calls in a thread
                    def search_shodan():
                        res = {'matches': [], 'total': 0}
                        try:
                            # Get count first
                            count_result = api.count(query)
                            res['total'] = count_result.get('total', 0)
                            # Get results using cursor
                            results_cursor = api.search_cursor(query)
                            iter_count = 0
                            for banner in results_cursor:
                                res['matches'].append(banner)
                                iter_count += 1
                                if iter_count >= limit: break
                        except shodan.APIError as api_e:
                            # Raise a specific error type that tenacity can catch if needed
                            raise api_e
                        return res

                    results = await asyncio.to_thread(search_shodan)

                    self.logger.info(f"Shodan search completed for '{query}'. Found {results['total']} total, retrieved {len(results['matches'])} (limit {limit}).")
                    return results
                except shodan.APIError as e:
                    logger.error(f"Shodan API error for query '{query}': {e}")
                    return {"error": f"Shodan API error: {e}"}
                except Exception as e:
                    logger.error(f"Shodan search failed for query '{query}': {e}", exc_info=True)
                    return {"error": str(e)}

            async def run_SpiderFoot(self, target: str) -> Dict:
                """Interact with a running SpiderFoot instance API."""
                spiderfoot_url = self.internal_state.get('spiderfoot_url')
                if not self._spiderfoot_api_key or not spiderfoot_url:
                    return {"error": "SpiderFoot URL or API key not available."}

                self.logger.info(f"Initiating SpiderFoot scan for target: {target} via {spiderfoot_url}")
                scan_name = f"OSINTAgent_{target}_{int(time.time())}"
                modules_to_run = self.config.get("SPIDERFOOT_MODULES", ["sfp_dnsresolve", "sfp_geoip", "sfp_ripe"])

                try:
                    async with aiohttp.ClientSession() as session:
                        start_payload = {'scanname': scan_name, 'scantarget': target, 'modulelist': modules_to_run, 'usecase': 'all'}
                        headers = {'X-API-KEY': self._spiderfoot_api_key}
                        scan_id = None
                        self.logger.debug(f"Starting SpiderFoot scan: URL={spiderfoot_url}/startscan, Payload={start_payload}")
                        async with session.post(f"{spiderfoot_url}/startscan", json=start_payload, headers=headers, timeout=30) as response:
                            if response.status == 200: scan_id = (await response.text()).strip('"')
                            else: error_text = await response.text(); logger.error(f"Failed start SpiderFoot scan for '{target}'. Status: {response.status}. Response: {error_text}"); return {"error": f"Failed start SpiderFoot scan (Status: {response.status})", "details": error_text}

                        if not scan_id: return {"error": "Failed to obtain Scan ID from SpiderFoot"}
                        self.logger.info(f"SpiderFoot scan started for '{target}'. Scan ID: {scan_id}")

                        poll_interval = int(self.config.get("SPIDERFOOT_POLL_INTERVAL_S", 30))
                        max_poll_time = int(self.config.get("SPIDERFOOT_MAX_POLL_TIME_S", 600))
                        start_poll_time = time.time()
                        while True:
                            if time.time() - start_poll_time > max_poll_time: logger.error(f"SpiderFoot scan {scan_id} for '{target}' timed out."); return {"error": f"SpiderFoot scan timed out"}
                            await asyncio.sleep(poll_interval)
                            self.logger.debug(f"Polling SpiderFoot scan status for ID: {scan_id}")
                            async with session.get(f"{spiderfoot_url}/scanstatus?id={scan_id}", headers=headers, timeout=15) as status_response:
                                if status_response.status == 200:
                                    status_text = (await status_response.text()).strip('"')
                                    self.logger.debug(f"SpiderFoot scan {scan_id} status: {status_text}")
                                    if status_text == 'FINISHED': logger.info(f"SpiderFoot scan {scan_id} finished."); break
                                    if status_text in ['ERROR', 'ABORTED', 'ABORTING', 'UNKNOWN']: logger.error(f"SpiderFoot scan {scan_id} ended status: {status_text}"); return {"error": f"SpiderFoot scan ended status: {status_text}"}
                                else: logger.warning(f"Failed get SpiderFoot status for {scan_id}. Status: {status_response.status}")

                        self.logger.debug(f"Retrieving SpiderFoot summary for scan ID: {scan_id}")
                        async with session.get(f"{spiderfoot_url}/scansummary?id={scan_id}", headers=headers, timeout=60) as summary_response:
                            if summary_response.status == 200:
                                results = await summary_response.json()
                                self.logger.info(f"SpiderFoot scan {scan_id} completed. Retrieved summary.")
                                return {"summary": results}
                            else: error_text = await summary_response.text(); logger.error(f"Failed retrieve SpiderFoot results for {scan_id}. Status: {summary_response.status}. Response: {error_text}"); return {"error": f"Failed retrieve SpiderFoot results (Status: {summary_response.status})", "details": error_text}

                except aiohttp.ClientConnectionError as e: logger.error(f"Could not connect to SpiderFoot at {spiderfoot_url}: {e}"); return {"error": f"Could not connect to SpiderFoot at {spiderfoot_url}"}
                except Exception as e: logger.error(f"SpiderFoot interaction failed for {target}: {e}", exc_info=True); return {"error": str(e)}

            async def run_LeakMonitoring(self, target: str) -> Dict:
                """Monitors public pastes/leaks for keywords related to the target."""
                if not self.internal_state.get('enable_leak_monitoring'):
                    return {"status": "disabled"}
                self.logger.info(f"Running Leak Monitoring for keywords related to: {target}")
                keywords = [target] # Add more derived keywords (e.g., company name, domain parts)
                if '.' in target: keywords.append(target.split('.')[0]) # Add domain name part

                all_findings = {}
                # Example: Check Pastebin (requires scraping or potentially paid API)
                pastebin_findings = await self._scrape_pastebin(keywords)
                if pastebin_findings: all_findings['pastebin'] = pastebin_findings

                # Example: Check HaveIBeenPwned for breaches related to a domain (requires API key usually)
                # hibp_findings = await self._check_hibp(target) # Needs implementation
                # if hibp_findings: all_findings['hibp'] = hibp_findings

                # Consult LegalAgent before storing potentially sensitive findings
                final_results = {}
                for source, findings in all_findings.items():
                    # Simplified validation call - needs refinement based on finding content
                    validation_context = f"Found potential data leak related to '{target}' on {source}. Finding snippet: {str(findings)[:200]}..."
                    is_valid = await self._validate_data_usage("leaked_data", validation_context)
                    if is_valid:
                        final_results[source] = findings
                    else:
                        self.logger.warning(f"Usage of leak data from {source} for target '{target}' blocked by LegalAgent.")

                return final_results if final_results else {"status": "no_validated_findings"}

            async def _scrape_pastebin(self, keywords: List[str]) -> List[Dict]:
                """Scrapes recent pastes on Pastebin for keywords (basic example, respects robots.txt)."""
                self.logger.debug(f"Scraping Pastebin recent pastes for keywords: {keywords}")
                # WARNING: Heavy scraping is against Pastebin ToS. Use their API if possible. This is illustrative.
                # Respect robots.txt (check manually or use a library). Assume allowed for this example.
                base_url = "https://pastebin.com/archive"
                findings = []
                try:
                    soup = await self._scrape_simple_html(base_url)
                    if not soup: return findings

                    # Find links to recent pastes
                    paste_links = []
                    table = soup.find('table', class_='maintable')
                    if table:
                        for a in table.find_all('a', href=True):
                            href = a['href']
                            if href and href.startswith('/') and len(href) > 1 and href[1:] not in ['archive', 'pro', 'api', 'tools', 'faq', 'login', 'signup']:
                                paste_links.append(f"https://pastebin.com/raw{href}") # Get raw link

                    self.logger.debug(f"Found {len(paste_links)} recent paste links to check.")

                    # Check content of recent pastes (limit checks to avoid excessive scraping)
                    checked_count = 0
                    check_limit = 20 # Limit checks per run
                    async with aiohttp.ClientSession() as session:
                        for link in paste_links:
                            if checked_count >= check_limit: break
                            try:
                                await asyncio.sleep(random.uniform(1, 3)) # Rate limit
                                async with session.get(link, timeout=10) as response:
                                        if response.status == 200:
                                            content = await response.text()
                                            for keyword in keywords:
                                                if keyword.lower() in content.lower():
                                                    findings.append({"url": link.replace('/raw/', '/'), "keyword": keyword, "snippet": content[:200]+"..."})
                                                    self.logger.info(f"Found keyword '{keyword}' in paste: {link}")
                                                    # Stop checking this paste once a keyword is found
                                                    break
                                        else:
                                            self.logger.debug(f"Pastebin link {link} returned status {response.status}")
                            except Exception as scrape_err:
                                self.logger.warning(f"Error scraping paste {link}: {scrape_err}")
                            checked_count += 1

                except Exception as e:
                    self.logger.error(f"Error during Pastebin scraping: {e}", exc_info=True)
                return findings

            async def _validate_data_usage(self, data_type: str, context: str) -> bool:
                """Consults LegalAgent via Orchestrator about using specific data."""
                if not self.orchestrator or not hasattr(self.orchestrator, 'delegate_task'):
                    self.logger.error("Cannot validate data usage: Orchestrator delegation unavailable.")
                    return False # Default to not using if cannot validate

                try:
                    # Use a more specific key if LegalAgent is split (e.g., 'legal_compliance')
                    legal_agent_key = 'strategic_legal' if 'strategic_legal' in self.orchestrator.agents else 'legal_compliance'
                    if legal_agent_key not in self.orchestrator.agents:
                        legal_agent_key = 'legal' # Fallback to generic key

                    if legal_agent_key not in self.orchestrator.agents:
                        self.logger.error(f"Cannot validate data usage: No suitable LegalAgent ({legal_agent_key}) found.")
                        return False

                    validation_task = {
                        "action": "validate_data_usage", # Define this action for LegalAgent
                        "data_type": data_type,
                        "context": context,
                        "description": f"Assess legality/ethics of using {data_type} found with context: {context[:100]}..."
                    }
                    result = await self.orchestrator.delegate_task(legal_agent_key, validation_task)
                    # Assuming LegalAgent returns {'status': 'success', 'is_compliant': True/False}
                    if result and result.get('status') == 'success' and result.get('is_compliant'):
                        return True
                    else:
                        self.logger.warning(f"Data usage validation failed or denied for {data_type}. Reason: {result.get('reason', 'Unknown')}")
                        return False
                except Exception as e:
                    self.logger.error(f"Error during data usage validation call: {e}", exc_info=True)
                    return False # Default to not using on error

            # --- Other Tool Placeholders ---
            async def run_Photon(self, target_url: str) -> Dict: return {"error": "Photon not implemented"}
            async def run_Maltego(self, target: str) -> Dict: return {"error": "Maltego not implemented"}
            async def run_DarkPoolSearch(self, target: str) -> Dict: return {"error": "Dark Pool Search not implemented/disabled"}
            async def run_WebcamSearch(self, target: str) -> Dict: return {"error": "Webcam Search not implemented/disabled"}

            # --- Analysis & Integration ---
            async def analyze_data(self, data_id: int):
                """Analyze collected OSINT data using LLMs."""
                self.logger.info(f"{self.AGENT_NAME}: Starting analysis for OSINT data ID: {data_id}")
                analysis = {"error": "Analysis failed"}
                raw_data_str = None
                target = "Unknown"
                tools_used = "[]"

                try:
                    async with self.session_maker() as session:
                        osint_data = await session.get(OSINTData, data_id)
                        if not osint_data: raise ValueError(f"OSINT data ID {data_id} not found.")
                        if not osint_data.raw_data: self.logger.warning(f"No raw data for ID {data_id}."); return
                        raw_data_str = osint_data.raw_data
                        target = osint_data.target
                        tools_used = osint_data.tools_used or "[]"

                    raw_data_str_for_prompt = raw_data_str[:10000] + ("... (data truncated)" if len(raw_data_str) > 10000 else "")

                    prompt = f"""
                    {self.internal_state['meta_prompt'][:500]}... # Include meta prompt context
                    **Task:** Analyze OSINT data for target '{target}' (collected via {tools_used}).
                    **Data:** ```json\n{raw_data_str_for_prompt}\n```
                    **Analysis Required:**
                    1. Summarize key findings (emails, social profiles, vulnerabilities, company info).
                    2. Identify actionable UGC leads (contacts, roles, emails, phones, relevant company context/pain points).
                    3. Assess data quality/reliability (Low/Medium/High).
                    4. Identify potential credentials (usernames, emails, password hints - tag as sensitive).
                    5. Provide concise overall summary and actionable flag.
                    **Output JSON:** {{"summary": str, "key_findings": {{...}}, "ugc_leads": {{...}}, "potential_credentials": [str], "data_quality": str, "actionable": bool}}
                    """
                    analysis_json = await self._call_llm_with_retry(
                        prompt, model_preference=["google/gemini-1.5-pro-latest"], # Use capable model
                        temperature=0.3, max_tokens=2000, is_json_output=True
                    )

                    if analysis_json:
                        try:
                            analysis = json.loads(analysis_json)
                            # Basic validation
                            if not all(k in analysis for k in ['summary', 'key_findings', 'ugc_leads', 'data_quality', 'actionable']):
                                raise ValueError("Analysis JSON missing required keys.")
                            self.logger.info(f"LLM analysis successful for ID {data_id}.")
                        except (json.JSONDecodeError, ValueError) as parse_err:
                            self.logger.error(f"Failed to parse analysis JSON for ID {data_id}: {parse_err}. Response: {analysis_json[:200]}...")
                            analysis = {"error": f"Failed to parse analysis JSON: {parse_err}"}
                    else:
                        analysis = {"error": "LLM analysis call failed."}

                    # Store Analysis and Determine Relevance
                    async with self.session_maker() as session:
                        stmt = update(OSINTData).where(OSINTData.id == data_id).values(analysis_results=json.dumps(analysis))
                        # Determine relevance
                        is_actionable = analysis.get('actionable', False)
                        has_leads = bool(analysis.get('ugc_leads', {}).get('contacts')) or bool(analysis.get('ugc_leads', {}).get('insights'))
                        if "error" in analysis: relevance = 'error'
                        elif is_actionable or has_leads: relevance = 'high'
                        elif analysis.get('key_findings'): relevance = 'medium'
                        else: relevance = 'low'
                        stmt = stmt.values(relevance=relevance)
                        await session.execute(stmt)
                        await session.commit()
                        self.logger.info(f"Analysis completed for ID {data_id}. Relevance assessed as '{relevance}'.")

                        # Trigger Integrations
                        if relevance in ['high', 'medium']:
                            await self.trigger_integrations(data_id, analysis, target)

                except ValueError as ve: # Catch specific errors like ID not found
                    logger.error(f"{self.AGENT_NAME}: Analysis error for ID {data_id}: {ve}")
                except SQLAlchemyError as db_err: logger.error(f"DB error during analysis for ID {data_id}: {db_err}", exc_info=True)
                except Exception as e: logger.error(f"Unexpected error during analysis for ID {data_id}: {e}", exc_info=True)
                # Attempt to save error state to DB
                if "error" in analysis:
                    try:
                        async with self.session_maker() as error_session:
                            await error_session.execute(update(OSINTData).where(OSINTData.id == data_id).values(analysis_results=json.dumps(analysis), relevance='error'))
                            await error_session.commit()
                    except Exception as final_err: logger.error(f"Failed to save error state for ID {data_id}: {final_err}")

            async def trigger_integrations(self, data_id: int, analysis: Dict, target_name: str):
                """Send actionable insights/leads to other agents."""
                self.logger.info(f"Triggering integrations for analyzed data ID: {data_id}")
                leads = analysis.get('ugc_leads', {})
                contacts = leads.get('contacts', [])
                insights = leads.get('insights', [])

                if not contacts and not insights: return

                # Send to EmailAgent
                email_agent = self.orchestrator.agents.get('email')
                if email_agent and hasattr(email_agent, 'queue_osint_leads'):
                    email_leads = [{"email": c, "name": "OSINT Lead", "source": f"OSINT_{data_id}"} for c in contacts if isinstance(c, str) and '@' in c]
                    if email_leads:
                        try: await email_agent.queue_osint_leads(email_leads); self.logger.info(f"Sent {len(email_leads)} email leads for ID {data_id} to EmailAgent.")
                        except Exception as e: self.logger.error(f"Failed queue leads for EmailAgent (ID: {data_id}): {e}", exc_info=True)

                # Send to VoiceSalesAgent
                voice_agent = self.orchestrator.agents.get('voice_sales')
                if voice_agent and hasattr(voice_agent, 'queue_osint_leads'):
                    try:
                        voice_payload = {"osint_data_id": data_id, "target": target_name, "contacts": contacts, "insights": insights}
                        await voice_agent.queue_osint_leads(voice_payload)
                        self.logger.info(f"Sent leads/insights for ID {data_id} to VoiceSalesAgent.")
                    except Exception as e: self.logger.error(f"Failed queue leads for VoiceSalesAgent (ID: {data_id}): {e}", exc_info=True)

                # Send insights to ThinkTool KB
                if insights and self.kb_interface and hasattr(self.kb_interface, 'add_knowledge'):
                    try:
                        await self.kb_interface.add_knowledge(
                            agent_source=self.AGENT_NAME, data_type="osint_actionable_insight",
                            content={"target": target_name, "insights": insights}, tags=["osint", "lead_insight"],
                            relevance_score=0.7, source_reference=f"OSINTData:{data_id}"
                        )
                    except Exception as e: self.logger.error(f"Failed log OSINT insight to KB (ID: {data_id}): {e}", exc_info=True)

            # --- Visualization (Placeholders - Requires Libraries) ---
            async def visualize_data(self, data_id: int):
                self.logger.warning(f"Visualization for OSINT data ID: {data_id} not implemented.")
                pass # Placeholder

            async def run_maltego_visualization(self, raw_data: Dict, analysis_results: Dict) -> Optional[Dict]:
                self.logger.warning("Maltego visualization generation not implemented.")
                return {"error": "Maltego visualization not implemented"}

            async def save_visualization(self, graph_data: Dict, filename: str) -> Optional[str]:
                self.logger.warning("Visualization saving not implemented.")
                return None

            # --- Workflow & Base Class Methods ---
            async def run_osint_workflow(self, target: str, tools: Optional[List[str]] = None):
                """Run the full collect -> analyze -> visualize workflow."""
                await self._internal_think(f"Starting full OSINT workflow for target: '{target}'")
                data_id = await self.collect_data(target, tools)
                if data_id:
                    await self._internal_think(f"Data collection complete (ID: {data_id}). Proceeding to analysis.")
                    await self.analyze_data(data_id)
                    await self._internal_think(f"Analysis complete (ID: {data_id}). Proceeding to visualization.")
                    await self.visualize_data(data_id) # Currently a placeholder
                    await self._internal_think(f"Visualization step complete (ID: {data_id}). Workflow finished for '{target}'.")
                    self.logger.info(f"OSINT workflow finished for '{target}'.")
                else:
                    await self._internal_think(f"Data collection failed for '{target}'. Aborting workflow.")
                    self.logger.warning(f"Data collection failed for '{target}'. Workflow aborted.")

            async def execute_task(self, task_details: Dict[str, Any]) -> Any:
                """Core method to execute OSINT tasks."""
                self.status = "working"
                task_type = task_details.get('type')
                target = task_details.get('target')
                tools = task_details.get('tools')
                data_id = task_details.get('data_id')
                result = {"status": "failure", "message": f"Unsupported task type or missing args: {task_type}"}
                self.logger.info(f"OSINTAgent executing task: {task_type} for {target or data_id}")

                try:
                    if task_type == "run_workflow" and target:
                        asyncio.create_task(self.run_osint_workflow(target, tools))
                        result = {"status": "workflow initiated"}
                    elif task_type == "collect_data" and target:
                        # Run collection directly and return ID or error
                        new_data_id = await self.collect_data(target, tools)
                        if new_data_id: result = {"status": "success", "data_id": new_data_id}
                        else: result = {"status": "failure", "message": "Data collection failed."}
                    elif task_type == "analyze_data" and data_id:
                        asyncio.create_task(self.analyze_data(data_id))
                        result = {"status": "analysis initiated"}
                    elif task_type == "visualize_data" and data_id:
                        asyncio.create_task(self.visualize_data(data_id))
                        result = {"status": "visualization initiated"}
                    else:
                        self.logger.warning(f"Unsupported task/args for OSINTAgent execute_task: {task_details}")

                except Exception as e:
                    self.logger.error(f"Error in OSINTAgent execute_task for '{task_type}': {e}", exc_info=True)
                    result = {"status": "error", "message": f"Exception during task '{task_type}': {e}"}

                self.status = "idle"
                return result

            async def learning_loop(self):
                """Periodic loop to update tool effectiveness based on data relevance."""
                while self.status == "running":
                    try:
                        await asyncio.sleep(3600 * 6) # Run every 6 hours
                        self.logger.info("OSINTAgent Learning Loop: Analyzing tool effectiveness...")

                        # --- Structured Thinking Step ---
                        thinking_process = f"""
                        Structured Thinking: OSINT Learning Loop
                        1. Goal: Refine tool success rates based on the relevance of data they collected.
                        2. Context: OSINTData records (tools_used, relevance). Internal tool success rates.
                        3. Constraints: Query DB. Update internal state. Avoid drastic changes based on small samples.
                        4. Information Needed: Recent OSINTData records with relevance != 'pending_analysis'.
                        5. Plan:
                            a. Query recent OSINTData records (last 7 days).
                            b. For each record, parse tools_used.
                            c. Adjust success rate for each tool based on relevance ('high'/'medium' = success, 'low'/'error'/'no_data' = failure). Use a smaller adjustment factor.
                            d. Log updated rates.
                        """
                        await self._internal_think(thinking_process)
                        # --- End Structured Thinking Step ---

                        async with self.session_maker() as session:
                            one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)
                            stmt = select(OSINTData.tools_used, OSINTData.relevance).where(
                                OSINTData.timestamp >= one_week_ago,
                                OSINTData.relevance != 'pending_analysis'
                            ).limit(500) # Limit query size
                            results = await session.execute(stmt)
                            records = results.mappings().all()

                        if not records:
                            self.logger.info("OSINT Learning Loop: No relevant data found for analysis.")
                            continue

                        adjustment_factor = 0.01 # Smaller adjustment for learning loop
                        updates_made = 0
                        for record in records:
                            try:
                                tools = json.loads(record.relevance) if record.tools_used else []
                                is_success = record.relevance in ['high', 'medium']
                                for tool in tools:
                                    if tool in self.internal_state['tool_success_rates']:
                                        rate = self.internal_state['tool_success_rates'][tool]
                                        adjustment = adjustment_factor if is_success else -adjustment_factor
                                        new_rate = min(max(rate + adjustment, 0.1), 1.0)
                                        self.internal_state['tool_success_rates'][tool] = new_rate
                                        updates_made += 1
                            except json.JSONDecodeError:
                                self.logger.warning(f"Could not parse tools_used JSON: {record.tools_used}")
                            except Exception as parse_err:
                                self.logger.error(f"Error processing record in learning loop: {parse_err}")

                        if updates_made > 0:
                            self.logger.info(f"OSINT Learning Loop: Updated tool success rates based on {len(records)} records. New rates (sample): {dict(list(self.internal_state['tool_success_rates'].items())[:5])}")
                        else:
                            self.logger.info("OSINT Learning Loop: No tool success rate updates made in this cycle.")

                    except asyncio.CancelledError:
                        self.logger.info("OSINTAgent learning loop cancelled.")
                        break
                    except Exception as e:
                        self.logger.error(f"Error during OSINTAgent learning loop: {e}", exc_info=True)
                        await asyncio.sleep(3600) # Wait longer after error

            async def self_critique(self) -> Dict[str, Any]:
                """Evaluates OSINT agent performance."""
                self.logger.info("OSINTAgent: Performing self-critique.")
                critique = {"status": "ok", "feedback": "Critique pending analysis."}
                # --- Structured Thinking Step ---
                thinking_process = f"""
                Structured Thinking: Self-Critique OSINTAgent
                1. Goal: Evaluate OSINT performance (data relevance, tool effectiveness, cost if applicable).
                2. Context: Internal state (tool rates), DB metrics (OSINTData relevance counts).
                3. Constraints: Query DB. Analyze relevance distribution. Output structured critique.
                4. Information Needed: Counts of OSINTData by relevance (last 7 days). Tool success rates.
                5. Plan:
                    a. Query DB for relevance counts.
                    b. Analyze relevance distribution (high % low/error?).
                    c. Review tool success rates (any consistently low performers?).
                    d. Formulate critique summary.
                    e. Return critique dictionary.
                """
                await self._internal_think(thinking_process)
                # --- End Structured Thinking Step ---
                try:
                    async with self.session_maker() as session:
                        one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)
                        stmt = select(OSINTData.relevance, func.count(OSINTData.id)).where(
                            OSINTData.timestamp >= one_week_ago
                        ).group_by(OSINTData.relevance)
                        results = await session.execute(stmt)
                        relevance_counts = {row.relevance: row[1] for row in results.mappings().all()}

                    critique['relevance_stats_7d'] = relevance_counts
                    critique['tool_success_rates'] = {k: round(v, 2) for k, v in self.internal_state['tool_success_rates'].items()}
                    feedback_points = [f"Relevance (7d): {relevance_counts}."]
                    total = sum(relevance_counts.values())
                    if total > 0:
                        low_relevance_pct = ((relevance_counts.get('low', 0) + relevance_counts.get('error', 0) + relevance_counts.get('no_data', 0)) / total) * 100
                        if low_relevance_pct > 40: feedback_points.append(f"WARNING: High low-relevance rate ({low_relevance_pct:.1f}%). Review tool selection/analysis prompts.")
                    critique['feedback'] = " ".join(feedback_points)

                except Exception as e:
                    self.logger.error(f"Error during OSINT self-critique: {e}", exc_info=True)
                    critique['status'] = 'error'; critique['feedback'] = f"Critique failed: {e}"
                return critique

            async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
                """Constructs context-rich prompts for LLM calls (e.g., data analysis)."""
                self.logger.debug(f"Generating dynamic prompt for OSINT task: {task_context.get('task')}")
                # --- Structured Thinking Step ---
                thinking_process = f"""
                Structured Thinking: Generate Dynamic Prompt (OSINT Analysis)
                1. Goal: Create effective LLM prompt for analyzing collected OSINT data.
                2. Context: Task details ({task_context.get('task')}), target, tools used, raw data snippet.
                3. Constraints: Use OSINT_AGENT_META_PROMPT. Instruct for specific extraction (leads, creds, insights) and JSON output.
                4. Information Needed: Relevant KB patterns/directives (Placeholder for now).
                5. Plan:
                    a. Start with meta prompt.
                    b. Append task context (target, tools).
                    c. Append truncated raw data.
                    d. Add specific analysis instructions and JSON output format.
                    e. Return final prompt string.
                """
                await self._internal_think(thinking_process)
                # --- End Structured Thinking Step ---

                prompt_parts = [self.internal_state.get('meta_prompt', "Analyze OSINT data.")]
                prompt_parts.append("\n--- Current Task Context ---")
                prompt_parts.append(f"Task: {task_context.get('task', 'Analyze OSINT Data')}")
                prompt_parts.append(f"Target: {task_context.get('target', 'Unknown')}")
                prompt_parts.append(f"Tools Used: {task_context.get('tools_used', 'Unknown')}")

                raw_data_str = task_context.get('raw_data_str_for_prompt', '{}') # Get truncated data
                prompt_parts.append(f"\n**Collected Data (JSON, potentially truncated):**\n```json\n{raw_data_str}\n```")

                # Add Specific Instructions based on task
                prompt_parts.append("\n--- Instructions ---")
                if task_context.get('task') == 'Analyze OSINT Data':
                    prompt_parts.append("1. Summarize key findings (emails, social profiles, vulnerabilities, company info).")
                    prompt_parts.append("2. Identify actionable UGC leads (contacts, roles, emails, phones, relevant company context/pain points).")
                    prompt_parts.append("3. Assess data quality/reliability (Low/Medium/High).")
                    prompt_parts.append("4. Identify potential credentials (usernames, emails, password hints - tag as sensitive).")
                    prompt_parts.append("5. Provide concise overall summary and actionable flag.")
                    prompt_parts.append(f"6. **Output Format:** {task_context.get('desired_output_format', 'JSON object as specified previously.')}")
                else:
                    prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")

                prompt_parts.append("```json") # Hint for the LLM if JSON is expected

                final_prompt = "\n".join(prompt_parts)
                self.logger.debug(f"Generated dynamic prompt for OSINTAgent (length: {len(final_prompt)} chars)")
                return final_prompt

            # --- Helper Methods ---
            async def _scrape_simple_html(self, url: str) -> Optional[BeautifulSoup]:
                """Fetches and parses HTML using aiohttp, BeautifulSoup, proxies, and user agents."""
                self.logger.debug(f"Attempting scrape of URL: {url}")
                ua = self.user_agent_generator.random if self.user_agent_generator else 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'
                headers = {'User-Agent': ua, 'Accept-Language': 'en-US,en;q=0.9', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}
                proxy_url: Optional[str] = None

                if self.orchestrator and hasattr(self.orchestrator, 'get_proxy'):
                    try:
                        proxy_url = await self.orchestrator.get_proxy(purpose="scraping", target_url=url)
                        if proxy_url: self.logger.debug(f"Using proxy for scraping {url}: {proxy_url.split('@')[-1]}")
                        else: self.logger.debug(f"No proxy available/needed for {url}.")
                    except Exception as proxy_err: self.logger.error(f"Failed to get proxy for scraping {url}: {proxy_err}")
                else: self.logger.warning("Orchestrator proxy support unavailable.")

                try:
                    timeout = aiohttp.ClientTimeout(total=30)
                    async with aiohttp.ClientSession(headers=headers) as session:
                        request_kwargs = {"timeout": timeout, "allow_redirects": True, "ssl": False} # Disable SSL verify for broader compatibility
                        if proxy_url: request_kwargs["proxy"] = proxy_url

                        async with session.get(url, **request_kwargs) as response:
                            self.logger.debug(f"Scrape request to {url} (via proxy: {bool(proxy_url)}) completed status: {response.status}")
                            if response.status == 200:
                                content_type = response.headers.get('Content-Type', '').lower()
                                if 'text/html' in content_type:
                                    html_content = await response.text()
                                    # Basic check for blocking pages
                                    if "CAPTCHA" in html_content or "Access Denied" in html_content or "checking your browser" in html_content.lower():
                                        self.logger.warning(f"Potential block page detected at {url}. Skipping parse.")
                                        return None
                                    soup = BeautifulSoup(html_content, 'html.parser')
                                    self.logger.debug(f"Successfully scraped and parsed HTML from {url}")
                                    return soup
                                else: self.logger.warning(f"Content type for {url} not HTML ({content_type}). Skipping parse."); return None
                            else: self.logger.warning(f"Failed fetch {url} for scraping. Status: {response.status}"); return None
                except asyncio.TimeoutError: self.logger.error(f"Timeout scraping URL: {url}"); return None
                except aiohttp.ClientError as e: self.logger.error(f"Client error scraping {url}: {e}"); return None
                except Exception as e: self.logger.error(f"Unexpected error scraping {url}: {e}", exc_info=True); return None

        # --- End of agents/osint_agent.py ---