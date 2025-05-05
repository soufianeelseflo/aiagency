# Filename: agents/gmail_creator_agent.py
# Description: Agent for attempting automated Gmail account creation.
# Version: 1.0 (Initial Implementation - Phone Verification is a known challenge)

import asyncio
import logging
import json
import random
from typing import Dict, Any, Optional, List

# --- Project Imports ---
try:
    from .base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
    from faker import Faker # Assuming Faker is available via BrowsingAgent or installed globally
except ImportError:
    logging.critical("Failed to import BaseAgent or Faker. Check paths/dependencies.")
    # Dummy class if import fails
    class GeniusAgentBase:
        def __init__(self, *args, **kwargs): self.logger = logging.getLogger(self.AGENT_NAME)
        async def _internal_think(self, *args, **kwargs): pass
        async def _report_error(self, *args, **kwargs): pass
    class Faker:
        def first_name(self): return "Test"
        def last_name(self): return "User"
        def password(self, **kwargs): return "Password123!"

logger = logging.getLogger(__name__)
op_logger = logging.getLogger('OperationalLog')

GMAIL_CREATOR_META_PROMPT = """
You are the GmailCreatorAgent within the Synapse AI Sales System.
Your Core Mandate: Autonomously create new Gmail accounts as needed for system operations (e.g., accessing free trials like Gemini), maximizing success rate while employing anti-detection measures.
Key Responsibilities:
- Receive directives from ThinkTool specifying the number of accounts needed.
- Generate realistic fake identities using integrated tools (Faker).
- Request high-quality proxies from the Orchestrator for each creation attempt.
- Delegate the complex UI automation task to the BrowsingAgent, providing the identity, proxy, and the specific goal: "Complete Gmail signup using provided identity, **attempt to visually identify and click any 'Skip' button for phone verification if offered**, otherwise proceed as far as possible."
- Handle results from BrowsingAgent:
    - On success (account created, even if phone skip was needed/attempted): Securely store credentials via Orchestrator.
    - On failure (e.g., phone verification mandatory, CAPTCHA failed, other error): Log the failure reason. Do NOT attempt to use paid SMS services unless explicitly directed by a future directive.
- Operate resourcefully and stealthily, relying on BrowsingAgent's human emulation.
**Goal:** Successfully create Gmail accounts by navigating the signup flow, intelligently attempting to bypass phone verification using visual cues, and securely storing successful creations. Understand that mandatory phone verification is a potential blocker.
"""

class GmailCreatorAgent(GeniusAgentBase):
    """
    Agent responsible for orchestrating the creation of Gmail accounts,
    leveraging BrowsingAgent for UI automation.
    Version: 1.0
    """
    AGENT_NAME = "GmailCreatorAgent"

    def __init__(self, session_maker: Optional[Any] = None, orchestrator: Optional[Any] = None, config: Optional[Any] = None):
        """Initializes the GmailCreatorAgent."""
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, config=config, session_maker=session_maker)
        self.meta_prompt = GMAIL_CREATOR_META_PROMPT
        try:
            self.faker = Faker()
        except NameError:
            self.logger.error("Faker library not available to GmailCreatorAgent.")
            self.faker = None # Handle gracefully if Faker wasn't imported

    async def log_operation(self, level: str, message: str):
        """Helper to log to the operational log file."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err: logger.error(f"Failed to write to operational log: {log_err}")

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handles tasks, primarily 'create_gmail_accounts'."""
        action = task_details.get('action')
        self.logger.info(f"{self.AGENT_NAME} received task: {action}")
        self._status = self.STATUS_EXECUTING
        result = {"status": "failure", "message": f"Unsupported action: {action}"}
        created_count = 0
        failed_count = 0

        if action == 'create_gmail_accounts':
            num_accounts_needed = task_details.get('content', {}).get('count', 1)
            self.logger.info(f"Attempting to create {num_accounts_needed} Gmail account(s).")

            for i in range(num_accounts_needed):
                if self._stop_event.is_set():
                    self.logger.warning("Stop event received during account creation loop.")
                    break
                await self._internal_think(f"Starting creation attempt {i+1}/{num_accounts_needed}.")
                success = await self._attempt_single_creation()
                if success:
                    created_count += 1
                else:
                    failed_count += 1
                # Add a random delay between attempts
                await asyncio.sleep(random.uniform(30, 120))

            result = {
                "status": "completed" if failed_count == 0 else "completed_with_errors",
                "message": f"Gmail creation task finished. Success: {created_count}, Failed: {failed_count}.",
                "accounts_created": created_count,
                "accounts_failed": failed_count
            }
        else:
             self.logger.warning(f"Unsupported action '{action}' for {self.AGENT_NAME}.")

        self._status = self.STATUS_IDLE
        return result

    async def _attempt_single_creation(self) -> bool:
        """Attempts to create a single Gmail account."""
        identity = None
        proxy_info = None
        account_details = None

        try:
            # 1. Generate Identity
            if not self.faker: raise RuntimeError("Faker tool not available.")
            identity = self._generate_realistic_identity()
            await self._internal_think("Generated fake identity.", details=identity)

            # 2. Get Proxy
            proxy_info = await self.orchestrator.get_proxy(purpose="gmail_creation", quality_level='premium')
            if not proxy_info:
                self.logger.warning("Failed to get premium proxy for Gmail creation, trying standard.")
                proxy_info = await self.orchestrator.get_proxy(purpose="gmail_creation", quality_level='standard')
            if not proxy_info:
                 self.logger.error("Could not obtain any proxy for Gmail creation. Aborting attempt.")
                 return False
            await self._internal_think("Obtained proxy.", details=proxy_info.get('server'))

            # 3. Prepare BrowsingAgent Task
            signup_goal = (
                "Complete the Google Account signup process using the provided identity details. "
                "**Crucially, if presented with an option to skip phone number verification, visually identify and click the 'Skip' button.** "
                "If phone verification is absolutely mandatory and cannot be skipped, proceed as far as possible and report 'phone_verification_required' in the failure message. "
                "Handle any CAPTCHAs encountered."
            )
            # Ensure password meets Google's likely complexity requirements
            identity['password'] = self.faker.password(length=12, special_chars=True, digits=True, upper_case=True, lower_case=True)

            browsing_task_details = {
                "action": "web_ui_automate",
                "service": "GoogleSignup", # Service name for BrowsingAgent context
                "goal": signup_goal,
                "params": {"identity": identity}, # Pass identity details
                "requires_account": False, # Doesn't need an existing account to start
                "allow_account_creation": False, # Prevent recursive creation attempts
                "proxy_info": proxy_info, # Pass the specific proxy
                "max_steps": 40 # Allow more steps for signup
            }
            await self._internal_think("Delegating signup task to BrowsingAgent.", details={"goal": signup_goal[:100]+"..."})

            # 4. Delegate and Handle Result
            post_result = await self.orchestrator.delegate_task("BrowsingAgent", browsing_task_details)

            if post_result and post_result.get("status") == "success":
                self.logger.info(f"BrowsingAgent reported SUCCESS for Gmail signup attempt for {identity.get('email_suggestion')}.")
                # Securely store the credentials
                stored_account = await self.orchestrator.secure_storage.store_new_account(
                    service="google.com", # Store under google.com
                    identifier=identity['email_address'], # The actual email created
                    password=identity['password'],
                    status='active',
                    metadata={"created_by": self.AGENT_NAME, "creation_ts": datetime.now(timezone.utc).isoformat(), "proxy_used": proxy_info.get('server')}
                )
                if stored_account:
                    self.logger.info(f"Successfully created and stored Gmail account: {identity['email_address']} (ID: {stored_account.get('id')})")
                    await self.log_operation('info', f"Successfully created Gmail: {identity['email_address']}")
                    return True
                else:
                    self.logger.error(f"BrowsingAgent succeeded, but failed to store credentials for {identity['email_address']}!")
                    await self.log_operation('error', f"Failed to store credentials for created Gmail: {identity['email_address']}")
                    return False
            else:
                error_msg = post_result.get('message', 'Unknown error') if post_result else 'No result from BrowsingAgent'
                self.logger.warning(f"BrowsingAgent reported FAILURE for Gmail signup attempt: {error_msg}")
                await self.log_operation('warning', f"Gmail creation failed: {error_msg}")
                # Log failure reason to KB?
                await self.log_knowledge_fragment(
                     agent_source=self.AGENT_NAME, data_type="gmail_creation_failure",
                     content={"identity_attempted": identity.get('email_suggestion'), "error": error_msg, "proxy": proxy_info.get('server')},
                     tags=["gmail", "account_creation", "failure"], relevance_score=0.4
                 )
                return False

        except Exception as e:
            self.logger.error(f"Unexpected error during single Gmail creation attempt: {e}", exc_info=True)
            await self._report_error(f"Gmail creation attempt failed: {e}")
            return False

    def _generate_realistic_identity(self) -> Dict[str, Any]:
        """Generates identity details suitable for signup."""
        first_name = self.faker.first_name()
        last_name = self.faker.last_name()
        # Generate more plausible username suggestions
        year = str(random.randint(1980, 2005))
        num = str(random.randint(10, 999))
        username_base = re.sub(r'[^a-z0-9]', '', f"{first_name}{last_name}".lower())
        email_suggestion = f"{username_base}{random.choice([year, num, ''])}"[:25] # Limit length

        return {
            "first_name": first_name,
            "last_name": last_name,
            "email_suggestion": email_suggestion, # Suggestion for the username part
            "password": "TEMP_PASSWORD", # Will be replaced with complex one later
            "birth_month": random.choice(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]),
            "birth_day": str(random.randint(1, 28)),
            "birth_year": str(random.randint(1980, 2003)),
            "gender": random.choice(["Male", "Female", "Rather not say"]) # Options Google might offer
        }

    # --- Standard Agent Methods (Implement as needed) ---
    async def plan_task(self, task_details: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        # This agent likely receives direct directives, planning might be minimal
        action = task_details.get('action')
        if action == 'create_gmail_accounts':
            # The main logic is in execute_task's loop
            return [{"step": 1, "action": "Execute Gmail Creation Loop", "tool": "internal", "params": task_details}]
        return None

    async def execute_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        # Primarily driven by execute_task, step execution might not be needed
        # If plan_task returns steps, handle them here.
        action = step.get('action')
        if action == "Execute Gmail Creation Loop":
             # This is handled by the loop within execute_task itself
             return {"status": "success", "message": "Loop handled by main execute_task method."}
        self.logger.warning(f"execute_step called with unhandled action: {action}")
        return {"status": "failure", "message": f"Unhandled step action: {action}"}

    async def learning_loop(self):
        self.logger.info(f"{self.AGENT_NAME} learning loop: Monitoring creation success rates (Placeholder).")
        while not self._stop_event.is_set():
            # TODO: Query KB for 'gmail_creation_failure' vs success counts
            # Adapt strategies (e.g., proxy quality requests, timing) based on failure patterns
            await asyncio.sleep(3600 * 4) # Check every 4 hours

    async def self_critique(self) -> Dict[str, Any]:
        # TODO: Implement critique based on success/failure rates, proxy issues etc.
        return {"status": "ok", "feedback": "GmailCreatorAgent critique not fully implemented."}

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        # This agent primarily delegates, LLM use is minimal within this agent itself
        # It relies on BrowsingAgent's prompts.
        return f"Executing task: {task_context.get('task', 'N/A')}"

    async def collect_insights(self) -> Dict[str, Any]:
        # TODO: Collect stats on creation attempts, success, failures
        return {"agent_name": self.AGENT_NAME, "status": self.status, "accounts_attempted_session": 0, "accounts_succeeded_session": 0}

# --- End of agents/gmail_creator_agent.py ---