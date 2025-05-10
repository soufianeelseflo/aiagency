# Filename: agents/gmail_creator_agent.py
# Description: Agent for advanced, adaptive, and resilient automated Gmail account creation.
# Version: 2.1 (Level 35+ Transmutation - Adaptive Strategy, Enhanced Error Handling)

import asyncio
import logging
import json
import random
import re # Added for username cleaning
import time # Added for cooldown logic
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple

# --- Project Imports ---
try:
    from .base_agent import GeniusAgentBase
    from faker import Faker
except ImportError:
    logging.critical("Failed to import BaseAgent or Faker. Check paths/dependencies.")
    # Dummy classes for basic structure loading if imports fail
    class GeniusAgentBase: # type: ignore
        AGENT_NAME = "DummyBaseAgent"; STATUS_IDLE="idle"; STATUS_EXECUTING="executing"; STATUS_ERROR="error"
        def __init__(self, *args, **kwargs):
            self.agent_name = self.AGENT_NAME; self.logger = logging.getLogger(f"agent.{self.agent_name}")
            self.orchestrator = kwargs.get('orchestrator'); self.config = kwargs.get('config'); self.session_maker = kwargs.get('session_maker')
            self._stop_event = asyncio.Event(); self._status = self.STATUS_IDLE
        async def _internal_think(self, *args, **kwargs): pass
        async def _report_error(self, *args, **kwargs): pass
        async def log_knowledge_fragment(self, *args, **kwargs): pass
        async def get_status_summary(self) -> dict: return {"status": "dummy"}
        async def stop(self, timeout: float = 1.0): self._stop_event.set(); await asyncio.sleep(0.01)
        async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str: return "Dummy prompt"
        async def plan_task(self, task_details: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]: return None
        async def execute_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]: return {"status":"failure"}
        async def learning_loop(self): await asyncio.sleep(3600)
        async def self_critique(self) -> Dict[str, Any]: return {"status":"ok", "feedback":"dummy"}
        async def collect_insights(self) -> Dict[str, Any]: return {}
    class Faker: # type: ignore
        def first_name(self): return "Test"; 
        def last_name(self): return "User"
        def password(self, **kwargs): return f"P@$wOrd{random.randint(1000,9999)}!"
        def user_name(self): return f"testuser{random.randint(100,999)}"
        def date_of_birth(self, **kwargs): return datetime.now() - timedelta(days=random.randint(18*365, 65*365))
        def country_code(self): return random.choice(["US", "GB", "CA", "DE"])
        def word(self): return random.choice(["apple", "banana", "cherry"])

# Import settings AFTER base agent potentially loads dummy
try:
    from config.settings import settings
except ImportError:
    logging.critical("Failed to import settings in gmail_creator_agent.py")
    class DummySettings:
        def get(self, key, default=None): return default
        def get_secret(self, key): return None
    settings = DummySettings() # type: ignore


logger = logging.getLogger(__name__)
op_logger = logging.getLogger('OperationalLog')

GMAIL_CREATOR_META_PROMPT = """
You are the GmailCreatorAgent (Level 35+) within the Nolli AI Sales System.
Your Core Mandate: Autonomously and strategically create new Gmail accounts as needed for system operations (e.g., accessing free trials like Gemini), maximizing success rate through adaptive techniques, advanced proxy management, nuanced identity generation, and by employing grey-area anti-detection measures. Learn from failures and adapt your strategy.
Key Responsibilities:
- Receive directives from ThinkTool specifying the number of accounts needed and potentially desired identity characteristics or proxy profiles.
- Request nuanced "identity profiles" from ThinkTool via directives, or generate highly realistic fake identities using Faker, considering regional plausibility and past success rates.
- Request specific types of high-quality proxies from the Orchestrator (e.g., residential, specific region, unused for X days for Gmail), adapting requests based on performance.
- Delegate the complex UI automation task to the BrowsingAgent, providing the identity, proxy, and a dynamic goal: "Complete Gmail signup using provided identity. Prioritize skipping phone verification if any visual cue for 'Skip' or 'Later' exists. If mandatory, report 'phone_verification_mandatory'. Minimize optional data entry. Adapt to UI changes based on visual analysis."
- Handle results from BrowsingAgent with advanced logic:
    - On success: Securely store credentials via Orchestrator. Log detailed success parameters (identity type, proxy type, region, time of day, UI flow variations encountered) to KB. Reset consecutive failure count.
    - On "phone_verification_mandatory" failure: Log this specific outcome with context. Trigger a ThinkTool directive: "Analyze mandatory phone verification for Gmail with identity [details] and proxy [details]. Suggest alternative parameters (e.g., different identity archetype, proxy region/quality, browser locale/fingerprint variation, time-of-day) for a new attempt with a fresh 'soldier' identity." Increment failure count.
    - On other failures (CAPTCHA, UI error): Log detailed failure context (error message, step failed). Trigger ThinkTool directive: "Analyze Gmail signup failure: [error details], [UI screenshot reference if available]. Suggest refined BrowsingAgent interaction strategy or CAPTCHA handling technique." Increment failure count.
- Implement dynamic cool-down periods based on consecutive failures and ThinkTool directives.
- Adapt identity generation hints and proxy request parameters based on analysis of success/failure patterns logged in the KB.
- Operate resourcefully and stealthily, relying on BrowsingAgent's human emulation and adaptive visual reasoning.
**Goal:** Achieve the highest possible success rate for Gmail account creation through intelligent adaptation, learning from failures, strategic proxy/identity use, and by pushing the boundaries of automated UI interaction within acceptable risk parameters defined by LegalAgent/ThinkTool.
"""

class GmailCreatorAgent(GeniusAgentBase):
    AGENT_NAME = "GmailCreatorAgent"

    def __init__(self, session_maker: Optional[Any] = None, orchestrator: Optional[Any] = None, config: Optional[Any] = None):
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, config=config, session_maker=session_maker)
        self.meta_prompt = GMAIL_CREATOR_META_PROMPT
        self.think_tool = getattr(self.orchestrator, 'agents', {}).get('think') if self.orchestrator else None
        try:
            self.faker = Faker()
        except NameError:
            self.logger.error("Faker library not available to GmailCreatorAgent.")
            self.faker = None

        # Internal state with defaults from config, allowing dynamic updates
        self.internal_state['creation_cooldown_until'] = time.time()
        self.internal_state['consecutive_failures'] = 0
        self.internal_state['max_consecutive_failures_before_cooldown'] = int(self.config.get("GMAIL_MAX_CONSECUTIVE_FAILURES", 3)) if self.config else 3
        self.internal_state['cooldown_duration_seconds'] = int(self.config.get("GMAIL_COOLDOWN_DURATION_S", 3600 * 1)) if self.config else 3600
        self.internal_state['current_identity_strategy_hint'] = self.config.get("GMAIL_INITIAL_IDENTITY_HINT", "US_standard") if self.config else "US_standard"
        self.internal_state['current_proxy_strategy_hint'] = self.config.get("GMAIL_INITIAL_PROXY_HINT", "US_residential") if self.config else "US_residential"

        self.logger.info(f"{self.AGENT_NAME} v2.1 (L35+ Adaptive) initialized.")

    async def log_operation(self, level: str, message: str):
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err: logger.error(f"Failed to write to operational log: {log_err}")

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        action = task_details.get('action')
        self.logger.info(f"{self.AGENT_NAME} received task: {action}")
        self._status = self.STATUS_EXECUTING
        result = {"status": "failure", "message": f"Unsupported action: {action}"}
        created_count = 0; failed_count = 0

        try:
            if action == 'create_gmail_accounts':
                num_accounts_needed = task_details.get('content', {}).get('count', 1)
                # Allow overriding strategy hints via task content
                identity_profile_hint = task_details.get('content', {}).get('identity_profile_hint', self.internal_state['current_identity_strategy_hint'])
                proxy_profile_hint = task_details.get('content', {}).get('proxy_profile_hint', self.internal_state['current_proxy_strategy_hint'])
                self.logger.info(f"Attempting to create {num_accounts_needed} Gmail account(s). Identity hint: {identity_profile_hint}, Proxy hint: {proxy_profile_hint}")

                for i in range(num_accounts_needed):
                    if self._stop_event.is_set(): self.logger.warning("Stop event received during creation loop."); break
                    if time.time() < self.internal_state['creation_cooldown_until']:
                        cooldown_end_str = datetime.fromtimestamp(self.internal_state['creation_cooldown_until']).isoformat()
                        self.logger.warning(f"Gmail creation cooldown active. Skipping attempt. Cooldown until: {cooldown_end_str}")
                        failed_count +=1 # Count as a failure for this cycle's reporting
                        await asyncio.sleep(60) # Check again in a minute
                        continue

                    await self._internal_think(f"Starting creation attempt {i+1}/{num_accounts_needed}. Identity hint: {identity_profile_hint}, Proxy hint: {proxy_profile_hint}")
                    success, details = await self._attempt_single_creation(identity_profile_hint, proxy_profile_hint)
                    log_content = {**details, "identity_hint_used": identity_profile_hint, "proxy_hint_used": proxy_profile_hint}

                    if success:
                        created_count += 1
                        self.internal_state['consecutive_failures'] = 0 # Reset on success
                        if self.think_tool:
                            await self.think_tool.execute_task({
                                "action": "log_knowledge_fragment", "fragment_data": {
                                    "agent_source": self.AGENT_NAME, "data_type": "gmail_creation_success",
                                    "content": log_content, "tags": ["gmail", "account_creation", "success", identity_profile_hint, proxy_profile_hint], "relevance_score": 0.85 }})
                    else:
                        failed_count += 1
                        self.internal_state['consecutive_failures'] += 1
                        if self.think_tool:
                            await self.think_tool.execute_task({
                                "action": "log_knowledge_fragment", "fragment_data": {
                                    "agent_source": self.AGENT_NAME, "data_type": "gmail_creation_failure",
                                    "content": log_content, "tags": ["gmail", "account_creation", "failure", identity_profile_hint, proxy_profile_hint], "relevance_score": 0.65 }})

                        # Trigger analysis/cooldown based on consecutive failures
                        if self.internal_state['consecutive_failures'] >= self.internal_state['max_consecutive_failures_before_cooldown']:
                            cooldown_duration = self.internal_state['cooldown_duration_seconds']
                            self.internal_state['creation_cooldown_until'] = time.time() + cooldown_duration
                            cooldown_end_str = datetime.fromtimestamp(self.internal_state['creation_cooldown_until']).isoformat()
                            self.logger.warning(f"Max consecutive Gmail creation failures reached ({self.internal_state['consecutive_failures']}). Activating cooldown for {cooldown_duration/3600:.1f} hours until {cooldown_end_str}.")
                            await self.log_operation('warning', f"Gmail creation cooldown activated ({cooldown_duration/3600:.1f}h) due to {self.internal_state['consecutive_failures']} consecutive failures.")
                            if self.think_tool:
                                await self.think_tool.execute_task({
                                    "action": "analyze_persistent_service_failure",
                                    "content": {"service": "GmailCreation", "failure_count": self.internal_state['consecutive_failures'], "last_error": details.get("error")}
                                })

                    # Variable delay between attempts
                    delay = random.uniform(self.config.get("GMAIL_INTER_ATTEMPT_DELAY_MIN_S", 90), self.config.get("GMAIL_INTER_ATTEMPT_DELAY_MAX_S", 300)) if self.config else random.uniform(90, 300)
                    await asyncio.sleep(delay)

                result = {"status": "completed" if failed_count == 0 else "completed_with_errors",
                          "message": f"Gmail creation task finished. Success: {created_count}, Failed: {failed_count}.",
                          "accounts_created": created_count, "accounts_failed": failed_count}

            elif action == 'update_creation_parameters': # Handle dynamic updates from ThinkTool
                content = task_details.get('content', {})
                updated_params_count = 0
                if 'max_consecutive_failures_before_cooldown' in content:
                    try:
                        new_val = int(content['max_consecutive_failures_before_cooldown'])
                        if new_val > 0:
                            self.internal_state['max_consecutive_failures_before_cooldown'] = new_val
                            self.logger.info(f"Updated max_consecutive_failures to {new_val}")
                            updated_params_count += 1
                    except ValueError: self.logger.warning("Invalid value for max_consecutive_failures")
                if 'cooldown_duration_hours' in content:
                     try:
                        new_val = float(content['cooldown_duration_hours'])
                        if new_val >= 0.1:
                            self.internal_state['cooldown_duration_seconds'] = int(new_val * 3600)
                            self.logger.info(f"Updated cooldown_duration to {new_val} hours")
                            updated_params_count += 1
                     except ValueError: self.logger.warning("Invalid value for cooldown_duration_hours")
                if 'identity_profile_hints_to_try' in content and isinstance(content['identity_profile_hints_to_try'], list) and content['identity_profile_hints_to_try']:
                    # Simple strategy: just pick the first suggested hint for now
                    self.internal_state['current_identity_strategy_hint'] = content['identity_profile_hints_to_try'][0]
                    self.logger.info(f"Updated identity strategy hint to: {self.internal_state['current_identity_strategy_hint']}")
                    updated_params_count += 1
                # Add logic for proxy strategy hints if needed
                result = {"status": "success", "message": f"Updated {updated_params_count} creation parameters."}

            else: self.logger.warning(f"Unsupported action '{action}' for {self.AGENT_NAME}.")

        except Exception as e:
            self.logger.error(f"Error executing GmailCreator task '{action}': {e}", exc_info=True)
            result = {"status": "error", "message": f"Unexpected error: {e}"}
            await self._report_error(f"Task '{action}' failed: {e}")
        finally:
            self._status = self.STATUS_IDLE
        return result

    async def _attempt_single_creation(self, identity_hint: Optional[str], proxy_hint: Optional[str]) -> Tuple[bool, Dict]:
        identity = None; proxy_info = None; creation_details: Dict[str, Any] = {}
        try:
            if not self.faker: raise RuntimeError("Faker tool not available.")
            identity = await self._generate_strategic_identity(identity_hint)
            if not identity: raise RuntimeError("Failed to generate strategic identity.")
            # Log generated identity details (excluding password)
            creation_details["identity_generated"] = {k:v for k,v in identity.items() if k != 'password'}
            await self._internal_think("Generated strategic identity.", details=creation_details["identity_generated"])

            if not self.orchestrator or not hasattr(self.orchestrator, 'get_proxy'):
                raise RuntimeError("Orchestrator or get_proxy method not available.")

            # Request proxy with specific hints
            proxy_purpose = f"gmail_creation_{identity.get('region_hint','any')}"
            proxy_info = await self.orchestrator.get_proxy(purpose=proxy_purpose, quality_level='premium_residential', specific_hint=proxy_hint) # Request higher quality
            if not proxy_info: proxy_info = await self.orchestrator.get_proxy(purpose="gmail_creation_fallback", quality_level='standard')
            if not proxy_info:
                self.logger.error("Could not obtain any proxy for Gmail creation."); creation_details["error"] = "Proxy acquisition failed"; return False, creation_details
            creation_details["proxy_used"] = proxy_info.get('server')
            await self._internal_think("Obtained proxy.", details={"server": proxy_info.get('server'), "purpose": proxy_purpose, "hint": proxy_hint})

            signup_goal = (
                "Complete the Google Account signup process using the provided identity details. "
                "Prioritize minimal data entry. **Aggressively look for and attempt to use any 'Skip' or 'Do this later' option for phone number verification.** "
                "If phone verification is absolutely mandatory and cannot be skipped, clearly report 'phone_verification_mandatory' in the failure message. "
                "Handle any CAPTCHAs encountered using visual reasoning. Extract the final created email address if successful and store it in result_data as 'created_email'."
            )
            # Generate a strong password just before sending to BrowsingAgent
            identity['password'] = self.faker.password(length=random.randint(12,16), special_chars=True, digits=True, upper_case=True, lower_case=True)
            creation_details["identity_generated"]["password_used"] = True # Indicate a password was set

            browsing_task_details = {
                "action": "web_ui_automate", "service": "GoogleSignup", "goal": signup_goal,
                "params": {"identity": identity}, # Pass full identity dict
                "requires_account": False, "allow_account_creation": False, # BrowsingAgent isn't creating the account *record*, just performing UI actions
                "proxy_info": proxy_info, "max_steps": 60 # Increased max_steps for complex flow
            }
            await self._internal_think("Delegating signup task to BrowsingAgent.", details={"goal": signup_goal[:100]+"..."})
            # Delegate task to BrowsingAgent via Orchestrator
            post_result = await self.orchestrator.delegate_task("BrowsingAgent", browsing_task_details)

            # Process BrowsingAgent result
            if post_result and post_result.get("status") == "success":
                created_email = post_result.get("result_data", {}).get("created_email") # BrowsingAgent should extract this
                if not created_email or "@" not in created_email: # Validate if a proper email was returned
                    # Fallback: Construct from suggestion if BrowsingAgent failed extraction
                    created_email = identity.get('email_suggestion') + "@gmail.com"
                    self.logger.warning(f"BrowsingAgent did not return created_email, assuming {created_email}")
                else:
                    self.logger.info(f"BrowsingAgent reported SUCCESS for Gmail signup. Extracted email: {created_email}.")

                creation_details["final_email"] = created_email
                if not hasattr(self.orchestrator, 'secure_storage'):
                    raise RuntimeError("Orchestrator secure_storage not available.")

                # Store credentials using Orchestrator's secure storage shim
                stored_account = await self.orchestrator.secure_storage.store_new_account(
                    service="google.com", identifier=created_email, password=identity['password'],
                    status='active', metadata={"created_by": self.AGENT_NAME, "creation_ts": datetime.now(timezone.utc).isoformat(), "proxy_used": proxy_info.get('server'), "identity_hint": identity_hint}
                )
                if stored_account:
                    account_id = stored_account.get('id')
                    self.logger.info(f"Successfully created and stored Gmail account: {created_email} (ID: {account_id})")
                    await self.log_operation('info', f"Successfully created Gmail: {created_email}")
                    creation_details.update({"status": "success", "created_email": created_email, "account_id": account_id})
                    return True, creation_details
                else:
                    self.logger.error(f"BrowsingAgent succeeded, but failed to store credentials for {created_email}!")
                    await self.log_operation('error', f"Failed to store credentials for created Gmail: {created_email}")
                    creation_details.update({"status": "failure", "error": "Credential storage failed", "created_email_attempt": created_email})
                    return False, creation_details
            else:
                # Handle failure from BrowsingAgent
                error_msg = post_result.get('message', 'Unknown error') if post_result else 'No result from BrowsingAgent'
                self.logger.warning(f"BrowsingAgent reported FAILURE for Gmail signup attempt: {error_msg}")
                await self.log_operation('warning', f"Gmail creation failed: {error_msg}")
                creation_details.update({"status": "failure", "error": error_msg})
                # Check for specific phone verification failure
                if "phone_verification_mandatory" in error_msg.lower() and self.think_tool:
                     await self.think_tool.execute_task({
                         "action": "analyze_service_block", # ThinkTool action
                         "content": {"service": "GmailSignup", "block_type": "mandatory_phone_verification", "identity_used": creation_details["identity_generated"], "proxy_info": proxy_info, "browsing_agent_error": error_msg}
                     })
                return False, creation_details
        except Exception as e:
            self.logger.error(f"Unexpected error during single Gmail creation attempt: {e}", exc_info=True)
            await self._report_error(f"Gmail creation attempt failed: {e}")
            creation_details.update({"status": "failure", "error": str(e)})
            return False, creation_details

    async def _generate_strategic_identity(self, hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Generates identity. Attempts to get a strategic profile from ThinkTool first (via directive),
        then falls back to Faker-based generation.
        """
        if not self.faker: self.logger.error("Faker tool not available."); return None

        identity_profile_from_thinktool: Optional[Dict[str, Any]] = None
        if self.think_tool and hint and self.orchestrator:
            try:
                await self._internal_think(f"Requesting identity profile directive from ThinkTool with hint: {hint}")
                directive_content = {
                    "action": "get_identity_profile_for_creation",
                    "content": {
                        "service_target": "google.com", "identity_hint": hint,
                        "required_fields": ["first_name", "last_name", "birth_year_range_min", "birth_year_range_max", "gender_preference", "username_style_hint", "region_bias"]
                    }
                }
                # Use delegate_task to ask ThinkTool to provide the profile
                # This assumes ThinkTool's 'get_identity_profile_for_creation' action returns the profile in the result
                response = await self.orchestrator.delegate_task("ThinkTool", directive_content)
                if response and response.get("status") == "success" and response.get("identity_profile"):
                    identity_profile_from_thinktool = response.get("identity_profile")
                    self.logger.info(f"Received identity profile from ThinkTool for hint: {hint}")
                else:
                    self.logger.warning(f"ThinkTool did not provide identity profile for hint '{hint}'. Reason: {response.get('message') if response else 'No response'}. Falling back to Faker.")
            except Exception as e:
                self.logger.warning(f"Error requesting identity profile from ThinkTool: {e}. Falling back to Faker.")

        # --- Fallback/Default Generation using Faker ---
        # Use details from ThinkTool profile if available, otherwise generate with Faker
        first_name = identity_profile_from_thinktool.get("first_name") if identity_profile_from_thinktool else self.faker.first_name()
        last_name = identity_profile_from_thinktool.get("last_name") if identity_profile_from_thinktool else self.faker.last_name()

        min_age = identity_profile_from_thinktool.get("birth_year_range_max", 20) if identity_profile_from_thinktool else 20
        max_age = identity_profile_from_thinktool.get("birth_year_range_min", 55) if identity_profile_from_thinktool else 55
        dob = self.faker.date_of_birth(minimum_age=min_age, maximum_age=max_age)

        fn_clean = re.sub(r'[^a-z]', '', first_name.lower())
        ln_clean = re.sub(r'[^a-z]', '', last_name.lower())
        year_short = str(dob.year)[-2:]
        random_nums = str(random.randint(100,9999))

        username_style_hint = identity_profile_from_thinktool.get("username_style_hint", "mixed") if identity_profile_from_thinktool else "mixed"
        username_options = []
        # Generate more diverse options based on style
        if username_style_hint in ["professional", "mixed", "standard"]:
            username_options.extend([ f"{fn_clean}.{ln_clean}", f"{first_name.lower()}{ln_clean[0] if ln_clean else ''}{year_short}", f"{fn_clean[0] if fn_clean else ''}{ln_clean}{random_nums[:2]}"])
        if username_style_hint in ["casual", "mixed", "gamer"]:
            username_options.extend([f"{fn_clean}{ln_clean}{random_nums[:3]}", f"{self.faker.word().lower()}{fn_clean}{year_short}", f"{fn_clean}_{random.word().lower()}"])
        if not username_options: # Fallback
             username_options = [f"{fn_clean}{ln_clean}", f"{fn_clean}.{ln_clean}", f"{fn_clean}{ln_clean}{year_short}"]

        email_suggestion = random.choice(username_options)
        # Ensure username meets basic length/format requirements
        email_suggestion = (email_suggestion[:25] + random_nums[:5]) if len(email_suggestion) > 25 else email_suggestion
        email_suggestion = (email_suggestion + random_nums[:max(0, 6-len(email_suggestion))]) if len(email_suggestion) < 6 else email_suggestion
        email_suggestion = re.sub(r'[^a-z0-9.]', '', email_suggestion) # Allow letters, numbers, dots
        email_suggestion = re.sub(r'\.+', '.', email_suggestion).strip('.') # Clean dots

        generated_gender = identity_profile_from_thinktool.get("gender_preference", random.choice(["Male", "Female"])) if identity_profile_from_thinktool else random.choice(["Male", "Female"])
        region_hint = identity_profile_from_thinktool.get("region_bias", self.faker.country_code() if random.random() < 0.4 else "US") if identity_profile_from_thinktool else (self.faker.country_code() if random.random() < 0.4 else "US")

        return {
            "first_name": first_name, "last_name": last_name,
            "email_suggestion": email_suggestion, # This is the username part
            "password": "TEMP_PASSWORD_WILL_BE_REPLACED", # Placeholder, real one generated later
            "birth_month": dob.strftime("%B"),
            "birth_day": str(dob.day),
            "birth_year": str(dob.year),
            "gender": generated_gender,
            "region_hint": region_hint # Hint for proxy selection
        }

    async def learning_loop(self):
        self.logger.info(f"{self.AGENT_NAME} L35+ learning loop: Analyzing creation success/failure patterns.")
        while not self._stop_event.is_set():
            try:
                interval_key = "GMAIL_CREATOR_LEARNING_INTERVAL_S"
                default_interval = 3600 * 2 # Check every 2 hours
                learn_interval = int(self.config.get(interval_key, default_interval)) if self.config and hasattr(self.config, 'get') else default_interval
                await asyncio.sleep(learn_interval)
                if self._stop_event.is_set(): break

                await self._internal_think("GmailCreator Learning: Analyzing success/failure patterns for strategic adjustment.")
                if not self.think_tool or not hasattr(self.think_tool, 'query_knowledge_base') or not hasattr(self.think_tool, 'execute_task'):
                    self.logger.warning("ThinkTool or its methods not available for GmailCreator learning."); continue

                # Query for more detailed success/failure KFs
                success_frags = await self.think_tool.query_knowledge_base(data_types=["gmail_creation_success"], time_window=timedelta(days=7), limit=100)
                failure_frags = await self.think_tool.query_knowledge_base(data_types=["gmail_creation_failure"], time_window=timedelta(days=7), limit=100)

                total_attempts = len(success_frags) + len(failure_frags)
                if total_attempts < 5: self.logger.info(f"Not enough recent Gmail attempts ({total_attempts}) for deep analysis."); continue

                success_rate = len(success_frags) / total_attempts if total_attempts > 0 else 0
                self.logger.info(f"Gmail creation success rate (7d): {success_rate:.2%} ({len(success_frags)}/{total_attempts})")

                # Detailed analysis payload for ThinkTool
                detailed_failures = []
                for frag in failure_frags:
                    try:
                        content = json.loads(frag.content) if isinstance(frag.content, str) else frag.content
                        detailed_failures.append({
                            "error": content.get("error", "Unknown"),
                            "proxy_used": content.get("proxy_used", "N/A"),
                            "identity_hint": content.get("identity_hint_used", "N/A"), # Use hint used
                            "timestamp": getattr(frag, 'timestamp', datetime.min).isoformat()
                        })
                    except: continue

                # Ask ThinkTool to analyze and suggest strategy changes
                analysis_directive_content = {
                    "service_name": "GmailCreation",
                    "success_rate": success_rate,
                    "total_attempts": total_attempts,
                    "detailed_failures": detailed_failures[:25], # Send a sample of detailed failures
                    "current_strategy_notes": { # Agent's current understanding/parameters
                        "max_consecutive_failures": self.internal_state['max_consecutive_failures_before_cooldown'],
                        "cooldown_duration_hours": self.internal_state['cooldown_duration_seconds']/3600,
                        "current_identity_hint": self.internal_state['current_identity_strategy_hint'],
                        "current_proxy_hint": self.internal_state['current_proxy_strategy_hint']
                    }
                }
                await self.think_tool.execute_task({
                    "action": "analyze_and_adapt_creation_strategy", # ThinkTool action
                    "content": analysis_directive_content
                })
                self.logger.info("Sent detailed Gmail creation performance to ThinkTool for strategic adaptation.")

            except asyncio.CancelledError: self.logger.info(f"{self.AGENT_NAME} learning loop cancelled."); break
            except Exception as e:
                self.logger.error(f"Error in {self.AGENT_NAME} learning loop: {e}", exc_info=True)
                await self._report_error(f"Learning loop error: {e}")
                await asyncio.sleep(60 * 30)

    async def plan_task(self, task_details: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        # This agent's primary task is handled directly by execute_task
        return None

    async def execute_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        # This agent doesn't typically use multi-step plans
        action = step.get('action')
        self.logger.warning(f"execute_step called with unhandled action: {action}")
        return {"status": "failure", "message": f"Unhandled step action: {action}"}

    async def self_critique(self) -> Dict[str, Any]:
        feedback = f"GmailCreatorAgent (L35+): Orchestrates advanced Gmail creation via BrowsingAgent. Success rate heavily dependent on external factors (Google UI, proxy quality). Current strategy hints: Identity='{self.internal_state['current_identity_strategy_hint']}', Proxy='{self.internal_state['current_proxy_strategy_hint']}'. Consecutive Failures: {self.internal_state['consecutive_failures']}."
        status = "ok"
        insights = await self.collect_insights()
        attempts_24h = insights.get("attempts_24h", 0)
        success_rate_24h = (insights.get("successes_24h", 0) / attempts_24h) if attempts_24h > 0 else 0.0
        feedback += f" 24h Stats: Attempts={attempts_24h}, Success Rate={success_rate_24h:.1%}."

        if self.internal_state['creation_cooldown_until'] > time.time():
            status = "warning"; feedback += f" Currently in cooldown until {datetime.fromtimestamp(self.internal_state['creation_cooldown_until']).isoformat()}."
        elif attempts_24h > 5 and success_rate_24h < 0.2:
            status = "error"; feedback += " CRITICAL LOW SUCCESS RATE (<20%)!"
        elif attempts_24h > 10 and success_rate_24h < 0.5:
            status = "warning"; feedback += " Low success rate (<50%), needs ThinkTool review."

        return {"status": status, "feedback": feedback}

    async def collect_insights(self) -> Dict[str, Any]:
        attempts_24h = 0; successes_24h = 0
        if self.think_tool and hasattr(self.think_tool, 'query_knowledge_base'):
            try:
                success_frags = await self.think_tool.query_knowledge_base(data_types=["gmail_creation_success"], time_window=timedelta(days=1), limit=100)
                failure_frags = await self.think_tool.query_knowledge_base(data_types=["gmail_creation_failure"], time_window=timedelta(days=1), limit=100)
                attempts_24h = len(success_frags) + len(failure_frags)
                successes_24h = len(success_frags)
            except Exception as e:
                self.logger.warning(f"Failed to query KB for insights: {e}")
        return {
            "agent_name": self.AGENT_NAME, "status": self.status,
            "attempts_24h": attempts_24h,
            "successes_24h": successes_24h,
            "cooldown_active": time.time() < self.internal_state.get('creation_cooldown_until', 0),
            "consecutive_failures": self.internal_state.get('consecutive_failures', 0),
            "current_identity_hint": self.internal_state.get('current_identity_strategy_hint'),
            "current_proxy_hint": self.internal_state.get('current_proxy_strategy_hint'),
        }
# --- End of agents/gmail_creator_agent.py ---