# Filename: agents/gmail_creator_agent.py
# Description: Agent for advanced, adaptive, and resilient automated Gmail account creation.
# Version: 2.0 (Level 25+ Transmutation - Strategic & Adaptive Creation)

import asyncio
import logging
import json
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple # Added Tuple

# --- Project Imports ---
try:
    from .base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
    from faker import Faker
except ImportError:
    logging.critical("Failed to import BaseAgent or Faker. Check paths/dependencies.")
    class GeniusAgentBase: # type: ignore
        AGENT_NAME = "DummyBaseAgent"
        def __init__(self, *args, **kwargs):
            self.agent_name = self.AGENT_NAME
            self.logger = logging.getLogger(f"agent.{self.agent_name}")
            self.orchestrator = kwargs.get('orchestrator')
            self.config = kwargs.get('config')
            self.session_maker = kwargs.get('session_maker')
            self._stop_event = asyncio.Event()
        async def _internal_think(self, *args, **kwargs): pass
        async def _report_error(self, *args, **kwargs): pass
        async def log_knowledge_fragment(self, *args, **kwargs): pass
        async def get_status_summary(self) -> dict: return {"status": "dummy"}
        async def stop(self, timeout: float = 1.0): self._stop_event.set(); await asyncio.sleep(0.01)
    class Faker: # type: ignore
        def first_name(self): return "Test"
        def last_name(self): return "User"
        def password(self, **kwargs): return "P@$$wOrd123!"
        def user_name(self): return "testuser" + str(random.randint(100,999))
        def date_of_birth(self, minimum_age=18, maximum_age=65):
            return datetime.now() - timedelta(days=random.randint(minimum_age*365, maximum_age*365))

logger = logging.getLogger(__name__)
op_logger = logging.getLogger('OperationalLog')

GMAIL_CREATOR_META_PROMPT = """
You are the GmailCreatorAgent (Level 25+) within the Synapse AI Sales System.
Your Core Mandate: Autonomously and strategically create new Gmail accounts as needed for system operations (e.g., accessing free trials like Gemini), maximizing success rate through adaptive techniques, advanced proxy management, nuanced identity generation, and by employing grey-area anti-detection measures.
Key Responsibilities:
- Receive directives from ThinkTool specifying the number of accounts needed and potentially desired identity characteristics or proxy profiles.
- Request nuanced "identity profiles" from ThinkTool, or generate highly realistic fake identities using Faker, considering regional plausibility.
- Request specific types of high-quality proxies from the Orchestrator (e.g., residential, specific region, unused for X days for Gmail).
- Delegate the complex UI automation task to the BrowsingAgent, providing the identity, proxy, and a dynamic goal: "Complete Gmail signup using provided identity. Prioritize skipping phone verification if any visual cue for 'Skip' or 'Later' exists. If mandatory, report 'phone_verification_required'. Minimize optional data entry. Adapt to UI changes based on visual analysis."
- Handle results from BrowsingAgent with advanced logic:
    - On success: Securely store credentials via Orchestrator. Log detailed success parameters (identity type, proxy type, UI flow variations encountered) to KB.
    - On "phone_verification_mandatory" failure: Log this specific outcome. Trigger a ThinkTool directive: "Analyze mandatory phone verification for Gmail with identity [details] and proxy [details]. Suggest alternative parameters (e.g., different identity archetype, proxy region, browser locale/fingerprint variation, time-of-day) for a new attempt with a fresh 'soldier' identity."
    - On other failures (CAPTCHA, UI error): Log detailed failure context. Trigger ThinkTool directive: "Analyze Gmail signup failure: [error details], [UI screenshot reference if available]. Suggest refined BrowsingAgent interaction strategy or CAPTCHA handling technique."
- Implement cool-down periods for Gmail creation if ThinkTool detects high failure rates from the server IP.
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
        self.internal_state['creation_cooldown_until'] = time.time()
        self.internal_state['consecutive_failures'] = 0
        self.internal_state['max_consecutive_failures_before_cooldown'] = 3 # Configurable
        self.internal_state['cooldown_duration_seconds'] = 3600 * 1 # 1 hour cooldown

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

        if action == 'create_gmail_accounts':
            num_accounts_needed = task_details.get('content', {}).get('count', 1)
            identity_profile_hint = task_details.get('content', {}).get('identity_profile_hint') # e.g., "US_male_tech_enthusiast"
            proxy_profile_hint = task_details.get('content', {}).get('proxy_profile_hint') # e.g., "US_residential_verizon"
            self.logger.info(f"Attempting to create {num_accounts_needed} Gmail account(s). Identity hint: {identity_profile_hint}, Proxy hint: {proxy_profile_hint}")

            for i in range(num_accounts_needed):
                if self._stop_event.is_set(): self.logger.warning("Stop event received."); break
                if time.time() < self.internal_state['creation_cooldown_until']:
                    self.logger.warning(f"Gmail creation cooldown active. Skipping attempt. Cooldown until: {datetime.fromtimestamp(self.internal_state['creation_cooldown_until']).isoformat()}")
                    failed_count +=1 # Count as a failure for this cycle
                    await asyncio.sleep(60) # Check again in a minute
                    continue

                await self._internal_think(f"Starting creation attempt {i+1}/{num_accounts_needed}. Identity hint: {identity_profile_hint}, Proxy hint: {proxy_profile_hint}")
                success, details = await self._attempt_single_creation(identity_profile_hint, proxy_profile_hint)
                log_content = {**details, "identity_hint": identity_profile_hint, "proxy_hint": proxy_profile_hint}

                if success:
                    created_count += 1
                    self.internal_state['consecutive_failures'] = 0 # Reset on success
                    if self.think_tool:
                        await self.think_tool.execute_task({
                            "action": "log_knowledge_fragment", "fragment_data": {
                                "agent_source": self.AGENT_NAME, "data_type": "gmail_creation_success",
                                "content": log_content, "tags": ["gmail", "account_creation", "success"], "relevance_score": 0.85 }})
                else:
                    failed_count += 1
                    self.internal_state['consecutive_failures'] += 1
                    if self.think_tool:
                        await self.think_tool.execute_task({
                            "action": "log_knowledge_fragment", "fragment_data": {
                                "agent_source": self.AGENT_NAME, "data_type": "gmail_creation_failure",
                                "content": log_content, "tags": ["gmail", "account_creation", "failure"], "relevance_score": 0.65 }})
                    # Implement cooldown if too many consecutive failures
                    if self.internal_state['consecutive_failures'] >= self.internal_state['max_consecutive_failures_before_cooldown']:
                        self.internal_state['creation_cooldown_until'] = time.time() + self.internal_state['cooldown_duration_seconds']
                        self.logger.warning(f"Max consecutive Gmail creation failures reached ({self.internal_state['consecutive_failures']}). Activating cooldown for {self.internal_state['cooldown_duration_seconds']/3600:.1f} hours.")
                        await self.log_operation('warning', f"Gmail creation cooldown activated due to {self.internal_state['consecutive_failures']} consecutive failures.")
                        # Optionally, send a directive to ThinkTool to analyze the persistent failures
                        if self.think_tool:
                            await self.think_tool.execute_task({
                                "action": "analyze_persistent_service_failure",
                                "content": {"service": "GmailCreation", "failure_count": self.internal_state['consecutive_failures'], "last_error": details.get("error")}
                            })


                await asyncio.sleep(random.uniform(60, 180)) # Longer, more variable delay
            result = {"status": "completed" if failed_count == 0 else "completed_with_errors",
                      "message": f"Gmail creation task finished. Success: {created_count}, Failed: {failed_count}.",
                      "accounts_created": created_count, "accounts_failed": failed_count}
        else: self.logger.warning(f"Unsupported action '{action}' for {self.AGENT_NAME}.")
        self._status = self.STATUS_IDLE
        return result

    async def _attempt_single_creation(self, identity_hint: Optional[str], proxy_hint: Optional[str]) -> Tuple[bool, Dict]:
        identity = None; proxy_info = None; creation_details: Dict[str, Any] = {}
        try:
            if not self.faker: raise RuntimeError("Faker tool not available.")
            identity = await self._generate_strategic_identity(identity_hint)
            if not identity: raise RuntimeError("Failed to generate strategic identity.")
            creation_details["identity_generated"] = {k:v for k,v in identity.items() if k != 'password'} # Don't log temp password yet
            await self._internal_think("Generated strategic identity.", details=creation_details["identity_generated"])

            if not self.orchestrator or not hasattr(self.orchestrator, 'get_proxy'):
                raise RuntimeError("Orchestrator or get_proxy method not available.")

            proxy_info = await self.orchestrator.get_proxy(purpose=f"gmail_creation_{identity.get('region_hint','any')}", quality_level='premium', specific_hint=proxy_hint)
            if not proxy_info: proxy_info = await self.orchestrator.get_proxy(purpose="gmail_creation_fallback", quality_level='standard')
            if not proxy_info: self.logger.error("Could not obtain any proxy for Gmail creation."); creation_details["error"] = "Proxy acquisition failed"; return False, creation_details
            creation_details["proxy_used"] = proxy_info.get('server')
            await self._internal_think("Obtained proxy.", details=proxy_info.get('server'))

            signup_goal = (
                "Complete the Google Account signup process using the provided identity details. "
                "Prioritize minimal data entry. **Aggressively look for and attempt to use any 'Skip' or 'Do this later' option for phone number verification.** "
                "If phone verification is absolutely mandatory and cannot be skipped, clearly report 'phone_verification_mandatory' in the failure message. "
                "Handle any CAPTCHAs encountered using visual reasoning."
            )
            # Generate a strong password just before sending to BrowsingAgent
            identity['password'] = self.faker.password(length=random.randint(12,16), special_chars=True, digits=True, upper_case=True, lower_case=True)
            creation_details["identity_generated"]["password_used"] = True # Indicate a password was set

            browsing_task_details = {
                "action": "web_ui_automate", "service": "GoogleSignup", "goal": signup_goal,
                "params": {"identity": identity}, "requires_account": False, "allow_account_creation": False,
                "proxy_info": proxy_info, "max_steps": 50 # Increased max_steps for complex flow
            }
            await self._internal_think("Delegating signup task to BrowsingAgent.", details={"goal": signup_goal[:100]+"..."})
            post_result = await self.orchestrator.delegate_task("BrowsingAgent", browsing_task_details)

            if post_result and post_result.get("status") == "success":
                self.logger.info(f"BrowsingAgent reported SUCCESS for Gmail signup attempt for {identity.get('email_suggestion')}.")
                created_email = post_result.get("result_data", {}).get("created_email", identity.get("email_suggestion"))
                if not created_email or "@" not in created_email: # Validate if a proper email was returned
                    created_email = identity.get('email_suggestion') + "@gmail.com"
                    self.logger.warning(f"BrowsingAgent did not return a valid created_email, assuming {created_email}")
                
                creation_details["final_email"] = created_email
                if not hasattr(self.orchestrator, 'secure_storage'):
                    raise RuntimeError("Orchestrator secure_storage not available.")

                stored_account = await self.orchestrator.secure_storage.store_new_account(
                    service="google.com", identifier=created_email, password=identity['password'],
                    status='active', metadata={"created_by": self.AGENT_NAME, "creation_ts": datetime.now(timezone.utc).isoformat(), "proxy_used": proxy_info.get('server'), "identity_hint": identity_hint}
                )
                if stored_account:
                    self.logger.info(f"Successfully created and stored Gmail account: {created_email} (ID: {stored_account.get('id')})")
                    await self.log_operation('info', f"Successfully created Gmail: {created_email}")
                    creation_details.update({"status": "success", "created_email": created_email, "account_id": stored_account.get('id')})
                    return True, creation_details
                else:
                    self.logger.error(f"BrowsingAgent succeeded, but failed to store credentials for {created_email}!")
                    await self.log_operation('error', f"Failed to store credentials for created Gmail: {created_email}")
                    creation_details.update({"status": "failure", "error": "Credential storage failed", "created_email_attempt": created_email})
                    return False, creation_details
            else:
                error_msg = post_result.get('message', 'Unknown error') if post_result else 'No result from BrowsingAgent'
                self.logger.warning(f"BrowsingAgent reported FAILURE for Gmail signup attempt: {error_msg}")
                await self.log_operation('warning', f"Gmail creation failed: {error_msg}")
                creation_details.update({"status": "failure", "error": error_msg})
                if "phone_verification_mandatory" in error_msg.lower() and self.think_tool:
                     await self.think_tool.execute_task({
                         "action": "analyze_service_block", # New ThinkTool action
                         "content": {"service": "GmailSignup", "block_type": "mandatory_phone_verification", "identity_used": identity, "proxy_info": proxy_info}
                     })
                return False, creation_details
        except Exception as e:
            self.logger.error(f"Unexpected error during single Gmail creation attempt: {e}", exc_info=True)
            await self._report_error(f"Gmail creation attempt failed: {e}")
            creation_details.update({"status": "failure", "error": str(e)})
            return False, creation_details

    async def _generate_strategic_identity(self, hint: Optional[str]) -> Optional[Dict[str, Any]]:
        """Generates identity, potentially guided by ThinkTool or a hint."""
        if not self.faker: return None
        # Basic generation, ThinkTool could provide more nuanced profiles via KB
        first_name = self.faker.first_name()
        last_name = self.faker.last_name()
        dob = self.faker.date_of_birth(minimum_age=20, maximum_age=55) # More plausible age range

        # Create more varied and less predictable username suggestions
        fn_clean = re.sub(r'[^a-z]', '', first_name.lower())
        ln_clean = re.sub(r'[^a-z]', '', last_name.lower())
        year_short = str(dob.year)[-2:]
        random_nums = str(random.randint(100,9999))

        username_options = [
            f"{fn_clean}{ln_clean}",
            f"{fn_clean}.{ln_clean}",
            f"{fn_clean}{ln_clean}{year_short}",
            f"{fn_clean}_{ln_clean}{random_nums[:2]}",
            f"{ln_clean}{fn_clean}{random.choice(['', num_part for num_part in [year_short, random_nums[:2], random_nums[-2:]] if num_part])}"
        ]
        email_suggestion = random.choice(username_options)
        # Ensure username is within typical length limits (e.g., 6-30 chars for Gmail)
        email_suggestion = (email_suggestion[:25] + random_nums[:5]) if len(email_suggestion) > 25 else email_suggestion
        email_suggestion = (email_suggestion + random_nums[:(6-len(email_suggestion))]) if len(email_suggestion) < 6 else email_suggestion


        return {
            "first_name": first_name, "last_name": last_name,
            "email_suggestion": email_suggestion, # This is the username part for @gmail.com
            "password": "TEMP_PASSWORD_WILL_BE_REPLACED", # Placeholder, real one generated in _attempt_single_creation
            "birth_month": dob.strftime("%B"),
            "birth_day": str(dob.day),
            "birth_year": str(dob.year),
            "gender": random.choice(["Male", "Female", "Rather not say"]),
            "region_hint": self.faker.country_code() if random.random() < 0.3 else "US" # Hint for proxy selection
        }

    async def learning_loop(self):
        self.logger.info(f"{self.AGENT_NAME} learning loop: Monitoring creation success rates and adapting strategy.")
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
                if total_attempts < 3: self.logger.info(f"Not enough Gmail attempts ({total_attempts}) for deep analysis."); continue

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
                            "identity_hint": content.get("identity_hint", "N/A"),
                            "timestamp": frag.timestamp.isoformat()
                        })
                    except: continue
                
                # Ask ThinkTool to analyze and suggest strategy changes
                analysis_directive_content = {
                    "service_name": "GmailCreation",
                    "success_rate": success_rate,
                    "total_attempts": total_attempts,
                    "detailed_failures": detailed_failures[:20], # Send a sample of detailed failures
                    "current_strategy_notes": { # Agent's current understanding/parameters
                        "max_consecutive_failures": self.internal_state['max_consecutive_failures_before_cooldown'],
                        "cooldown_duration_hours": self.internal_state['cooldown_duration_seconds']/3600,
                        "identity_generation_notes": "Currently uses Faker with some regional hints."
                    }
                }
                await self.think_tool.execute_task({
                    "action": "analyze_and_adapt_creation_strategy", # New ThinkTool action
                    "content": analysis_directive_content
                })
                self.logger.info("Sent detailed Gmail creation performance to ThinkTool for strategic adaptation.")

            except asyncio.CancelledError: self.logger.info(f"{self.AGENT_NAME} learning loop cancelled."); break
            except Exception as e:
                self.logger.error(f"Error in {self.AGENT_NAME} learning loop: {e}", exc_info=True)
                await self._report_error(f"Learning loop error: {e}")
                await asyncio.sleep(60 * 30)

    async def plan_task(self, task_details: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        action = task_details.get('action')
        if action == 'create_gmail_accounts':
            return [{"step": 1, "action": "Execute Gmail Creation Loop", "tool": "internal", "params": task_details}]
        return None

    async def execute_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        action = step.get('action')
        if action == "Execute Gmail Creation Loop":
             return {"status": "success", "message": "Loop handled by main execute_task method."}
        self.logger.warning(f"execute_step called with unhandled action: {action}")
        return {"status": "failure", "message": f"Unhandled step action: {action}"}

    async def self_critique(self) -> Dict[str, Any]:
        feedback = "GmailCreatorAgent (L25+): Orchestrates advanced Gmail creation. Success depends on BrowsingAgent UI automation, proxy quality, and Google's evolving defenses. Learning loop feeds detailed failure/success patterns to ThinkTool for strategic adaptation."
        status = "ok"
        if self.think_tool and hasattr(self.think_tool, 'query_knowledge_base'):
            # Query for ThinkTool's last adaptation directive for Gmail creation
            adaptation_directives = await self.think_tool.query_knowledge_base(
                data_types=["StrategicDirective"],
                tags=[self.AGENT_NAME, "strategy_update", "gmail"], # Assuming ThinkTool tags its adaptation directives
                limit=1, order_by="timestamp_desc"
            )
            if adaptation_directives:
                feedback += f" Last ThinkTool adaptation: {json.loads(adaptation_directives[0].content).get('summary', 'Details in KB.')}"
            
            # Get recent success rate
            success_frags = await self.think_tool.query_knowledge_base(data_types=["gmail_creation_success"], time_window=timedelta(days=3), limit=50)
            failure_frags = await self.think_tool.query_knowledge_base(data_types=["gmail_creation_failure"], time_window=timedelta(days=3), limit=50)
            attempts_72h = len(success_frags) + len(failure_frags)
            success_rate_72h = (len(success_frags) / attempts_72h) if attempts_72h > 0 else 0.0
            feedback += f" Gmail Creation (72h): Attempts={attempts_72h}, Success Rate={success_rate_72h:.1%}."
            if attempts_72h > 5 and success_rate_72h < 0.15: status = "error"; feedback += " CRITICAL LOW SUCCESS!"
            elif attempts_72h > 10 and success_rate_72h < 0.4: status = "warning"; feedback += " Low success rate, needs ThinkTool review."

        return {"status": status, "feedback": feedback}

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        # This agent primarily delegates UI automation to BrowsingAgent.
        # Its own direct LLM use is minimal, mainly for internal thought logging if expanded.
        # Prompts for BrowsingAgent are constructed within _attempt_single_creation.
        return f"GmailCreatorAgent internal task: {task_context.get('task', 'N/A')}. Details: {json.dumps(task_context.get('details', {}))}"

    async def collect_insights(self) -> Dict[str, Any]:
        attempts_24h = 0; successes_24h = 0
        if self.think_tool and hasattr(self.think_tool, 'query_knowledge_base'):
            success_frags = await self.think_tool.query_knowledge_base(data_types=["gmail_creation_success"], time_window=timedelta(days=1), limit=100)
            failure_frags = await self.think_tool.query_knowledge_base(data_types=["gmail_creation_failure"], time_window=timedelta(days=1), limit=100)
            attempts_24h = len(success_frags) + len(failure_frags)
            successes_24h = len(success_frags)
        return {
            "agent_name": self.AGENT_NAME, "status": self.status,
            "attempts_24h": attempts_24h,
            "successes_24h": successes_24h,
            "cooldown_active": time.time() < self.internal_state.get('creation_cooldown_until', 0),
            "consecutive_failures": self.internal_state.get('consecutive_failures', 0)
        }
# --- End of agents/gmail_creator_agent.py ---