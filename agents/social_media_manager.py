# Filename: agents/social_media_manager.py
# Description: Agentic Social Media Manager for planning campaigns, managing multiple
#              accounts, creating/posting content, analyzing performance, and learning.
# Version: 3.0 (Agentic Learning & Full Implementation)

import logging
import os
import json
import asyncio
import random
import time
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union
from collections import Counter

# --- Project Imports ---
try:
    from .base_agent import GeniusAgentBase, KBInterface # Use relative import
except ImportError:
    from base_agent import GeniusAgentBase, KBInterface # Fallback

# Import necessary models (ensure these are defined correctly in models.py)
from models import Account, KnowledgeFragment, LearnedPattern, StrategicDirective, EmailLog, CallLog, Invoice # Added performance models
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy import update, desc, func, case # Import necessary SQLAlchemy components
from sqlalchemy.exc import SQLAlchemyError

# Assuming LLM Client and settings are accessed via orchestrator/config
# from openai import AsyncOpenAI as AsyncLLMClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Configure dedicated operational logger
op_logger = logging.getLogger('OperationalLog')
if not op_logger.hasHandlers():
    op_handler = logging.StreamHandler()
    op_formatter = logging.Formatter('%(asctime)s - OP_LOG - %(levelname)s - %(message)s')
    op_handler.setFormatter(op_formatter)
    op_logger.addHandler(op_handler)
    op_logger.setLevel(logging.INFO)
    op_logger.propagate = False

SOCIAL_MEDIA_MANAGER_META_PROMPT = """
You are the SocialMediaManager within the Synapse Genius Agentic AI System.
Your Core Mandate: Manage and grow social media presence across multiple platforms for various business models, employing advanced, unconventional, and highly effective engagement and traffic generation strategies, while strictly adhering to anti-ban protocols. Operate with Genius Agentic AI principles.
Key Responsibilities:
- Manage multiple accounts per platform (e.g., 10 FB, 10 TikTok, 10 IG per model) for specific business goals.
- Implement sophisticated strategies: 9-to-1 traffic funnels, multi-account ad management (redundancy, specialization), strategic interaction between owned accounts.
- Generate engaging, human-like AI content (text, image, video - potentially ethical deepfakes) adapted to each platform and audience. Learn from examples (e.g., x.com/apollonator3000, Tai Lopez).
- CRITICAL: Implement and rigorously follow robust anti-ban strategies: unique proxy per account (via Orchestrator/proxy manager), behavioral variance, content policy awareness, account backups.
- Utilize platform APIs (via BrowsingAgent or direct libraries) for posting, interaction, and data gathering.
- Understand and adapt to platform algorithms proactively by analyzing performance data.
- Analyze performance data to optimize strategies continuously (learning loop).
- Operate with Extreme Agentic Behavior: Devise novel social media strategies, analyze trends, adapt tactics rapidly, understand nuances of online social dynamics, sub-task complex campaigns.
- Collaborate with ThinkTool (strategy), OSINTAgent (trends/inspiration), BrowsingAgent (account access/API interaction), LegalAgent (content compliance).
"""

class SocialMediaManager(GeniusAgentBase):
    """
    Agentic Social Media Manager: Plans and executes campaigns, manages accounts,
    generates content, analyzes performance, and learns autonomously.
    Version: 3.0
    """
    AGENT_NAME = "SocialMediaManager"

    def __init__(self, orchestrator: Any, session_maker: Optional[async_sessionmaker[AsyncSession]] = None):
        """Initializes the SocialMediaManager."""
        config = getattr(orchestrator, 'config', None)
        kb_interface = getattr(orchestrator, 'kb_interface', None)
        super().__init__(agent_name=self.AGENT_NAME, kb_interface=kb_interface, orchestrator=orchestrator, config=config)
        self.session_maker = session_maker

        self.internal_state = getattr(self, 'internal_state', {})
        self.internal_state['managed_accounts'] = {} # { platform: [AccountDetails] }
        self.internal_state['current_campaign_plan'] = None
        self.internal_state['platform_configs'] = {} # Cache for platform-specific selectors/URLs
        self.internal_state['account_rotation_index'] = {} # { platform: index }
        self.internal_state['account_performance'] = {} # { account_id: {eng_rate: float, health: str} } - Updated by learning loop
        self.internal_state['content_style_preferences'] = {} # { platform: preferred_style } - Updated by learning loop
        self.internal_state['learning_interval_seconds'] = int(self.config.get("SMM_LEARNING_INTERVAL_S", 3600 * 3)) # Learn every 3 hours
        self.internal_state['critique_interval_seconds'] = int(self.config.get("SMM_CRITIQUE_INTERVAL_S", 3600 * 12)) # Critique twice daily
        self.internal_state['last_learning_run'] = None
        self.internal_state['last_critique_run'] = None

        # Load managed accounts immediately
        asyncio.create_task(self._load_managed_accounts())

        self.logger.info(f"{self.AGENT_NAME} v3.0 (Agentic) initialized.")

    async def log_operation(self, level: str, message: str):
        """Helper to log to the operational log file with agent context."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err:
            print(f"OPERATIONAL LOG FAILED ({self.agent_name}): {level} - {message} | Error: {log_err}")
            logger.error(f"Failed to write to operational log from {self.agent_name}: {log_err}")

    async def _load_managed_accounts(self):
        """Loads active social media account details from the database."""
        if not self.session_maker:
            self.logger.error("Database session maker not available. Cannot load managed accounts.")
            return
        self.logger.info("Loading managed social media accounts from database...")
        # Make platforms configurable via settings
        platforms_to_manage = self.config.get("SMM_PLATFORMS", ["facebook.com", "tiktok.com", "instagram.com", "x.com"])
        loaded_accounts = {p: [] for p in platforms_to_manage}
        try:
            async with self.session_maker() as session:
                stmt = select(Account).where(
                    Account.service.in_(platforms_to_manage),
                    Account.is_available == True
                )
                result = await session.execute(stmt)
                accounts = result.scalars().all()
                for acc in accounts:
                    acc_details = { "id": acc.id, "email": acc.email, "username": acc.username, "vault_path": acc.vault_path, "service": acc.service }
                    if acc.service in loaded_accounts: loaded_accounts[acc.service].append(acc_details)

            self.internal_state['managed_accounts'] = loaded_accounts
            self.internal_state['account_rotation_index'] = {p: 0 for p in loaded_accounts}
            total_accounts = sum(len(v) for v in loaded_accounts.values())
            self.logger.info(f"Loaded {total_accounts} managed social media accounts across {len(loaded_accounts)} platforms.")
            for platform, acc_list in loaded_accounts.items(): self.logger.info(f" - {platform}: {len(acc_list)} accounts")

        except SQLAlchemyError as db_err: self.logger.error(f"Database error loading managed accounts: {db_err}", exc_info=True)
        except Exception as e: self.logger.error(f"Unexpected error loading managed accounts: {e}", exc_info=True)

    def _get_managed_account(self, platform: str, criteria: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Selects an available managed account, potentially based on criteria or performance."""
        platform_key = platform.lower().replace("https://", "").replace("www.", "")
        accounts = self.internal_state.get('managed_accounts', {}).get(platform_key)
        if not accounts:
            self.logger.warning(f"No managed accounts loaded for platform: {platform_key}")
            return None

        num_accounts = len(accounts)
        # TODO: Implement smarter selection based on criteria (e.g., account_group from plan)
        # or performance/health data stored in self.internal_state['account_performance']
        # For now, use simple round-robin.
        current_index = self.internal_state.get('account_rotation_index', {}).get(platform_key, 0)
        selected_account = accounts[current_index]
        self.internal_state.setdefault('account_rotation_index', {})[platform_key] = (current_index + 1) % num_accounts

        self.logger.debug(f"Selected account for {platform_key}: {selected_account.get('email') or selected_account.get('username')}")
        return selected_account

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a social media-related task."""
        action = task_details.get('action', 'unknown')
        platform = task_details.get('platform', 'unknown')
        self.logger.info(f"{self.AGENT_NAME} received task: {action} on {platform}")
        self.status = "working"
        result = {"status": "failure", "message": f"Unsupported action: {action}"}

        try:
            if action == 'post':
                result = await self._handle_post_action(task_details)
            elif action == 'analyze':
                result = await self._handle_analyze_action(task_details)
            elif action == 'plan_campaign':
                 plan = await self._create_campaign_plan(task_details)
                 if plan: result = {"status": "success", "message": "Campaign plan created.", "plan": plan}
                 else: result = {"status": "failure", "message": "Failed to create campaign plan."}
            elif action == 'execute_campaign':
                 plan_to_execute = self.internal_state.get('current_campaign_plan') or task_details.get('plan')
                 if plan_to_execute:
                      asyncio.create_task(self._execute_campaign_plan(plan_to_execute))
                      result = {"status": "success", "message": "Campaign execution initiated in background."}
                 else: result = {"status": "failure", "message": "No campaign plan found to execute."}
            else: self.logger.warning(f"Unsupported action '{action}' for SocialMediaManager.")

        except Exception as e:
            self.logger.error(f"Error executing social media task '{action}' on {platform}: {e}", exc_info=True)
            result = {"status": "error", "message": f"Unexpected error: {e}"}
            if hasattr(self.orchestrator, 'report_error'): await self.orchestrator.report_error(self.AGENT_NAME, f"Task '{action}' failed: {e}")
        finally:
            self.status = "idle"
            self.logger.info(f"{self.AGENT_NAME} finished task: {action} on {platform}. Status: {result.get('status', 'failure')}")
        return result

    async def _handle_post_action(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the logic for posting content."""
        platform = task_details.get('platform')
        original_content = task_details.get('content')
        goal = task_details.get('goal', 'engagement')
        target_account_id = task_details.get('account_id') # Optional

        if not platform or not original_content: return {"status": "failure", "message": "Missing platform or content."}

        # 1. Refine Content
        await self._internal_think(f"Refining content for {platform} post. Goal: {goal}.")
        refinement_prompt = await self.generate_dynamic_prompt({
            "task": "Refine social media content", "platform": platform,
            "original_content": original_content, "goal": goal,
            "desired_output_format": "Refined post text suitable for the platform, including relevant hashtags."
        })
        refined_content = await self._call_llm_with_retry(refinement_prompt, max_tokens=1024, temperature=0.7)
        if not refined_content: refined_content = original_content

        # 2. Select Account & Credentials
        account_details = self._get_managed_account(platform) # Add criteria if needed
        if not account_details: return {"status": "failure", "message": f"No account for {platform}."}
        credentials = await self._get_credentials_for_account(account_details)
        if not credentials: return {"status": "failure", "message": "Credentials missing."}

        # 3. Get Proxy
        proxy_info = await self._get_proxy_for_action(f"social_post_{platform}")

        # 4. Delegate Posting
        await self._internal_think(f"Delegating post to {platform} using account {account_details.get('email')} via BrowsingAgent.")
        platform_config = self._get_platform_config(platform)
        browsing_task_details = {
            "action": "social_media_post", "target_platform": platform,
            "login_url": platform_config.get("login_url"), "post_url": platform_config.get("post_url"),
            "selectors": platform_config.get("selectors", {}),
            "account_email": account_details.get('email'), "account_username": account_details.get('username'),
            "account_password": credentials.get('password'), "post_content": refined_content, "proxy_info": proxy_info
        }

        if hasattr(self.orchestrator, 'delegate_task'):
             post_result = await self.orchestrator.delegate_task("BrowsingAgent", browsing_task_details)
             if post_result and post_result.get("status") == "success":
                 # Log success to KB for learning loop
                 await self.log_knowledge_fragment(
                     agent_source=self.AGENT_NAME, data_type="social_post_outcome",
                     content={"platform": platform, "account_id": account_details['id'], "status": "success", "post_id": post_result.get("post_id")},
                     tags=["social_post", "success", platform], relevance_score=0.7
                 )
                 return {"status": "success", "message": f"Successfully delegated post to {platform}.", "post_id": post_result.get("post_id", "N/A")}
             else:
                  error_msg = post_result.get('message', 'Unknown error') if post_result else 'No result'
                  # Log failure to KB
                  await self.log_knowledge_fragment(
                      agent_source=self.AGENT_NAME, data_type="social_post_outcome",
                      content={"platform": platform, "account_id": account_details['id'], "status": "failure", "error": error_msg},
                      tags=["social_post", "failure", platform], relevance_score=0.6
                  )
                  return {"status": "failure", "message": f"BrowsingAgent failed to post: {error_msg}"}
        else: return {"status": "failure", "message": "Orchestrator missing delegate_task capability."}

    async def _handle_analyze_action(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the logic for analyzing social media performance."""
        platform = task_details.get('platform')
        metric = task_details.get('metric', 'engagement')
        target_profile = task_details.get('target_profile')
        account_id = task_details.get('account_id')

        if not platform: return {"status": "failure", "message": "Missing platform."}

        # 1. Delegate Data Fetch
        fetch_agent = "OSINTAgent" if not account_id else "BrowsingAgent"
        fetch_action = "fetch_public_social_metrics" if fetch_agent == "OSINTAgent" else "fetch_account_social_metrics"
        await self._internal_think(f"Delegating data fetch ({metric}) to {fetch_agent} for platform: {platform}.")

        fetch_task_details = {"action": fetch_action, "platform": platform, "metrics_required": [metric], "target_profile": target_profile}
        if fetch_agent == "BrowsingAgent":
             account_details = self._get_managed_account(platform) # Or find by account_id
             if not account_details: return {"status": "failure", "message": f"No account for {platform}."}
             credentials = await self._get_credentials_for_account(account_details)
             if not credentials: return {"status": "failure", "message": "Credentials missing."}
             proxy_info = await self._get_proxy_for_action(f"social_analyze_{platform}")
             fetch_task_details.update({
                 "account_email": account_details.get('email'), "account_username": account_details.get('username'),
                 "account_password": credentials.get('password'), "proxy_info": proxy_info
             })

        fetched_data = None
        if hasattr(self.orchestrator, 'delegate_task'):
             fetch_result = await self.orchestrator.delegate_task(fetch_agent, fetch_task_details)
             if fetch_result and fetch_result.get("status") == "success":
                 fetched_data = fetch_result.get("data")
                 if not fetched_data: return {"status": "failure", "message": f"No data returned by {fetch_agent}."}
                 self.logger.info(f"Successfully fetched data from {fetch_agent}.")
             else:
                 error_msg = fetch_result.get('message', 'Unknown error') if fetch_result else f'No result from {fetch_agent}'
                 return {"status": "failure", "message": f"Data fetch failed: {error_msg}"}
        else: return {"status": "failure", "message": "Orchestrator missing delegate_task capability."}

        # 2. Analyze Data via LLM
        await self._internal_think(f"Analyzing fetched {metric} data for {platform} via LLM.")
        try: data_str = json.dumps(fetched_data, default=str, indent=2)
        except TypeError: data_str = str(fetched_data)

        analysis_prompt = await self.generate_dynamic_prompt({
            "task": "Analyze social media data", "platform": platform, "metric": metric,
            "fetched_data_summary": data_str[:3000] + "..." if len(data_str) > 3000 else data_str,
            "desired_output_format": "Brief summary of key findings regarding the metric."
        })
        analysis_summary = await self._call_llm_with_retry(analysis_prompt, max_tokens=500, temperature=0.3)

        if analysis_summary:
            await self.log_knowledge_fragment(
                agent_source=self.AGENT_NAME, data_type="social_media_analysis",
                content={"platform": platform, "metric": metric, "summary": analysis_summary, "raw_data_preview": data_str[:500]},
                tags=["analysis", "social_media", platform, metric], relevance_score=0.8
            )
            return {"status": "success", "message": f"Successfully analyzed {metric} for {platform}.", "analysis_summary": analysis_summary}
        else: return {"status": "failure", "message": "Data fetched, but LLM analysis failed."}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception))
    async def _create_campaign_plan(self, task_details: Dict[str, Any]) -> Optional[List[Dict]]:
        """Generates a structured social media campaign plan using an LLM."""
        self.logger.info("Generating social media campaign plan via LLM...")
        campaign_goal = task_details.get('goal', 'Increase brand awareness and generate leads')
        target_platform = task_details.get('platform', 'Multiple')
        duration = task_details.get('duration', '1 week')
        target_audience = task_details.get('audience', 'General audience interested in UGC')

        await self._internal_think(f"Fetching context for campaign plan: Trends for {target_platform}, Competitor strategies.")
        kb_context_str = await self._get_campaign_context(target_platform, target_audience)

        task_context = {
            "task": "Create Social Media Campaign Plan", "campaign_goal": campaign_goal,
            "target_platform": target_platform, "duration": duration, "target_audience": target_audience,
            "knowledge_base_context": kb_context_str,
            "desired_output_format": "JSON list of steps: [ { \"step\": int, \"day\": int, \"platform\": str, \"account_group\": str ('Traffic'|'Main'|'Engagement'), \"action_type\": str ('post'|'interact'|'ad'), \"content_brief\": str (detailed brief for LLM content gen), \"interaction_target\": str (e.g., specific user type, hashtag, competitor post URL), \"timing\": str (e.g., 'Morning', '1 PM EST', 'Peak Hours') } ]"
        }
        plan_prompt = await self.generate_dynamic_prompt(task_context)

        plan_json_str = await self._call_llm_with_retry(
            plan_prompt, model=self.config.get("OPENROUTER_MODELS", {}).get('strategy_planning', "google/gemini-1.5-pro-latest"),
            temperature=0.6, max_tokens=2500, is_json_output=True
        )

        if plan_json_str:
            try:
                plan_list = json.loads(plan_json_str[plan_json_str.find('['):plan_json_str.rfind(']')+1])
                if isinstance(plan_list, list) and all(isinstance(step, dict) and 'step' in step and 'action_type' in step for step in plan_list):
                    self.internal_state['current_campaign_plan'] = plan_list
                    self.logger.info(f"Successfully generated campaign plan with {len(plan_list)} steps.")
                    await self.log_knowledge_fragment(
                        agent_source=self.AGENT_NAME, data_type="social_campaign_plan",
                        content=plan_list, tags=["plan", "social_media", target_platform], relevance_score=0.9
                    )
                    return plan_list
                else: raise ValueError("LLM plan response is not a valid list of step dictionaries.")
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                self.logger.error(f"Failed to parse LLM campaign plan: {e}. Response: {plan_json_str[:500]}...")
                return None
        else:
            self.logger.error("LLM failed to generate a campaign plan.")
            return None

    async def _execute_campaign_plan(self, plan: List[Dict]):
        """Executes the steps defined in the campaign plan."""
        self.logger.info(f"Starting execution of campaign plan with {len(plan)} steps.")
        if not plan or not isinstance(plan, list):
            self.logger.error("Execution attempted without a valid campaign plan.")
            await self.log_operation('error', "Campaign execution failed: Invalid plan provided.")
            return

        plan_execution_log = [] # Track step outcomes

        for step in plan:
            step_num = step.get("step", "?")
            action_type = step.get("action_type")
            platform = step.get("platform")
            content_brief = step.get("content_brief")
            account_group = step.get("account_group", "Main")
            step_status = "pending"
            step_result_msg = ""

            await self._internal_think(f"Executing Campaign Step {step_num}: {action_type} on {platform} using {account_group} accounts.")
            await self.log_operation('info', f"Executing Step {step_num}: {action_type} on {platform}")

            try:
                # 1. Select Account
                account_details = self._get_managed_account(platform) # Add criteria later
                if not account_details: raise Exception(f"No account available for {platform}")

                # 2. Generate Content
                content_to_use = None
                if action_type == 'post' and content_brief:
                    content_prompt = await self.generate_dynamic_prompt({
                        "task": "Generate post content from brief", "platform": platform,
                        "content_brief": content_brief, "target_audience": plan.get("target_audience", ""),
                        "desired_output_format": "Post text/caption suitable for the platform."
                    })
                    content_to_use = await self._call_llm_with_retry(content_prompt, max_tokens=1024, temperature=0.7)
                    if not content_to_use: raise Exception("Failed to generate content from brief")

                # 3. Delegate Action
                browsing_task_details = {"action": None}
                if action_type == 'post' and content_to_use:
                    browsing_task_details["action"] = "social_media_post"
                    browsing_task_details["post_content"] = content_to_use
                elif action_type == 'interact':
                    browsing_task_details["action"] = "social_media_interact"
                    browsing_task_details["interaction_type"] = random.choice(['like', 'comment'])
                    browsing_task_details["target_post_url"] = step.get("interaction_target")
                    if browsing_task_details["interaction_type"] == 'comment':
                         comment_prompt = await self.generate_dynamic_prompt({"task": "Generate relevant social media comment", "target_post_context": "Context needed..."})
                         browsing_task_details["comment_text"] = await self._call_llm_with_retry(comment_prompt, max_tokens=100)

                if browsing_task_details["action"]:
                    credentials = await self._get_credentials_for_account(account_details)
                    if not credentials: raise Exception("Credentials missing")
                    proxy_info = await self._get_proxy_for_action(f"social_campaign_{platform}_{action_type}")
                    platform_config = self._get_platform_config(platform)

                    browsing_task_details.update({
                        "target_platform": platform, "login_url": platform_config.get("login_url"),
                        "selectors": platform_config.get("selectors", {}),
                        "account_email": account_details.get('email'), "account_username": account_details.get('username'),
                        "account_password": credentials.get('password'), "proxy_info": proxy_info
                    })

                    await self._internal_think(f"Delegating campaign action '{action_type}' for step {step_num} to BrowsingAgent.")
                    action_result = await self.orchestrator.delegate_task("BrowsingAgent", browsing_task_details)

                    if not action_result or action_result.get("status") != "success":
                         error_msg = action_result.get('message', 'Unknown error') if action_result else 'No result'
                         raise Exception(f"BrowsingAgent action failed: {error_msg}")
                    else:
                         self.logger.info(f"Step {step_num} ({action_type} on {platform}) delegated successfully.")
                         step_status = "success"
                         step_result_msg = action_result.get("post_id", "Interaction successful")
                         # TODO: Delegate task to fetch metrics for this post/interaction
                else:
                     step_status = "skipped"
                     step_result_msg = "No valid action defined for step."

                # Random delay
                await asyncio.sleep(random.uniform(15, 90))

            except Exception as e:
                self.logger.error(f"Error executing campaign plan step {step_num}: {e}", exc_info=True)
                step_status = "error"
                step_result_msg = str(e)
                # Decide on error handling (e.g., stop campaign?)

            plan_execution_log.append({"step": step_num, "action": action_type, "platform": platform, "status": step_status, "result": step_result_msg})

        self.logger.info("Finished executing campaign plan.")
        await self.log_operation('info', f"Campaign plan execution finished. Results: {plan_execution_log}")
        # Log results to KB
        await self.log_knowledge_fragment(
            agent_source=self.AGENT_NAME, data_type="social_campaign_execution_log",
            content=plan_execution_log, tags=["execution_log", "social_media"], relevance_score=0.8
        )
        self.internal_state['current_campaign_plan'] = None # Clear plan

    # --- Standardized LLM Interaction ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception))
    async def _call_llm_with_retry(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024, is_json_output: bool = False) -> Optional[str]:
        """Calls the LLM via the Orchestrator with retry logic."""
        self.logger.debug(f"Calling LLM via Orchestrator. Temp={temperature}, MaxTokens={max_tokens}, JSON={is_json_output}")
        if not hasattr(self.orchestrator, 'call_llm'):
             self.logger.error("Orchestrator does not have 'call_llm' method.")
             return None
        try:
            response_content = await self.orchestrator.call_llm(
                agent_name=self.AGENT_NAME, prompt=prompt, temperature=temperature,
                max_tokens=max_tokens, is_json_output=is_json_output
            )
            content = response_content.get('content') if isinstance(response_content, dict) else str(response_content)
            return content.strip() if content else None
        except Exception as e:
            self.logger.warning(f"LLM call failed (attempt): {e}")
            raise

    def _get_platform_config(self, platform: str) -> Dict[str, Any]:
        """Retrieves platform-specific configurations (URLs, selectors)."""
        platform_key = platform.lower().replace("https://", "").replace("www.", "")
        # Check cache first
        if platform_key in self.internal_state['platform_configs']:
            return self.internal_state['platform_configs'][platform_key]

        self.logger.debug(f"Retrieving configuration for platform: {platform_key}")
        # Load from config file or DB in production
        # Using self.config.get for platform-specific selectors defined in settings.py or DEFAULT_CONFIG
        config = {
            "login_url": self.config.get(f"{platform_key.upper()}_LOGIN_URL", f"https://{platform_key}/login"),
            "post_url": self.config.get(f"{platform_key.upper()}_POST_URL", f"https://{platform_key}/"),
            "selectors": {
                "post_textarea": self.config.get(f"{platform_key.upper()}_POST_TEXTAREA_SELECTOR"),
                "submit_button": self.config.get(f"{platform_key.upper()}_SUBMIT_BUTTON_SELECTOR"),
                # Add more common selectors needed by BrowsingAgent actions
                "like_button": self.config.get(f"{platform_key.upper()}_LIKE_BUTTON_SELECTOR"),
                "comment_input": self.config.get(f"{platform_key.upper()}_COMMENT_INPUT_SELECTOR"),
                "comment_submit": self.config.get(f"{platform_key.upper()}_COMMENT_SUBMIT_SELECTOR"),
            }
        }
        # Remove None values from selectors
        config["selectors"] = {k: v for k, v in config["selectors"].items() if v is not None}
        self.internal_state['platform_configs'][platform_key] = config # Cache it
        return config

    async def _get_credentials_for_account(self, account_details: Dict[str, Any]) -> Optional[Dict[str, str]]:
         """Securely fetches credentials from Vault."""
         if not account_details or not account_details.get('vault_path'): return None
         if not self.orchestrator or not hasattr(self.orchestrator, 'secure_storage'):
              self.logger.error("Secure storage unavailable via Orchestrator.")
              return None
         try:
             creds_json = await self.orchestrator.secure_storage.get_secret(account_details['vault_path'])
             return json.loads(creds_json) if creds_json else None
         except Exception as e:
              self.logger.error(f"Failed to fetch/parse credentials from Vault path {account_details['vault_path']}: {e}")
              return None

    async def _get_proxy_for_action(self, purpose: str) -> Optional[Dict[str, Any]]:
         """Requests a proxy from the orchestrator for a specific action."""
         if self.orchestrator and hasattr(self.orchestrator, 'get_proxy'):
              try:
                  return await self.orchestrator.get_proxy(purpose=purpose)
              except Exception as e:
                   self.logger.error(f"Failed to get proxy for purpose '{purpose}': {e}")
                   return None
         self.logger.warning("Orchestrator cannot provide proxies.")
         return None

    # --- Abstract Method Implementations ---

    async def learning_loop(self):
        """Analyzes social media performance data to refine strategies."""
        while True:
            try:
                interval = self.internal_state.get('learning_interval_seconds', 3600 * 3)
                await asyncio.sleep(interval)
                current_time = datetime.now(timezone.utc)
                self.internal_state['last_learning_run'] = current_time
                self.logger.info(f"{self.AGENT_NAME} learning loop: Analyzing performance.")

                await self._internal_think("Learning loop: Fetching performance data, analyzing trends, updating strategy state.")

                # 1. Fetch performance data from KB
                performance_fragments = await self.query_knowledge_base(
                    data_types=["social_post_outcome", "social_media_analysis"],
                    time_window=timedelta(days=7), # Analyze last 7 days
                    limit=500
                )
                if not performance_fragments:
                    self.logger.info("Learning Loop: No recent performance data found in KB.")
                    continue

                # 2. Analyze Data (Example: Engagement per platform/content type)
                platform_engagement = Counter() # platform: [list of engagement scores/indicators]
                content_type_performance = Counter() # content_type_tag: {'success': N, 'fail': N}

                for frag in performance_fragments:
                    try:
                        data = json.loads(frag.content)
                        platform = data.get('platform')
                        if frag.data_type == "social_post_outcome":
                             status = data.get('status')
                             # TODO: Define how to extract content type/style tags from original post/plan
                             content_tags = ["general"] # Placeholder
                             for tag in content_tags:
                                 if tag not in content_type_performance: content_type_performance[tag] = {'success': 0, 'fail': 0}
                                 if status == 'success': content_type_performance[tag]['success'] += 1
                                 else: content_type_performance[tag]['fail'] += 1
                        elif frag.data_type == "social_media_analysis":
                             # TODO: Extract quantitative metrics if analysis provides them
                             pass
                    except (json.JSONDecodeError, TypeError, KeyError):
                        self.logger.warning(f"Could not parse performance fragment ID {frag.id}")

                # 3. Update Internal State (Example: Preferred content style)
                best_style = None
                highest_success_rate = -1.0
                for style, stats in content_type_performance.items():
                    total = stats['success'] + stats['fail']
                    if total >= 10: # Minimum samples
                        rate = stats['success'] / total
                        if rate > highest_success_rate:
                            highest_success_rate = rate
                            best_style = style

                if best_style and best_style != self.internal_state.get('preferred_content_style'):
                    self.internal_state['preferred_content_style'] = best_style
                    self.logger.info(f"Learning Loop: Updated preferred content style to '{best_style}' (Success Rate: {highest_success_rate:.2f})")
                    await self.log_learned_pattern(
                        pattern_description=f"Content style '{best_style}' shows high success rate ({highest_success_rate:.2f}) on social media.",
                        supporting_fragment_ids=[], # Link to analysis fragments if possible
                        confidence_score=0.75,
                        implications=f"Prioritize generating content with style '{best_style}'.",
                        tags=["social_media", "content_strategy", "performance_optimized"]
                    )

                # TODO: Add analysis for account health, platform algorithm changes, etc.

            except asyncio.CancelledError:
                self.logger.info(f"{self.AGENT_NAME} learning loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in SocialMediaManager learning loop: {e}", exc_info=True)
                await asyncio.sleep(60 * 30) # Wait longer after error

    async def self_critique(self) -> Dict[str, Any]:
        """Evaluates campaign effectiveness, content quality, and anti-ban success."""
        self.logger.info(f"{self.AGENT_NAME}: Performing self-critique.")
        critique = {"status": "ok", "feedback": "Critique pending analysis."}
        try:
            # Fetch recent performance/health data
            insights = await self.collect_insights()
            # Query KB for recent errors or ban indicators related to social media tasks
            error_fragments = await self.query_knowledge_base(
                data_types=["social_post_outcome", "task_outcome"],
                tags=["failure", "error", "social_media"],
                time_window=timedelta(days=3), limit=20
            )
            ban_rate_estimate = "N/A" # Placeholder - calculate based on errors/account status checks

            # Use LLM for critique
            await self._internal_think("Generating self-critique using LLM based on recent insights and errors.")
            critique_context = {
                "task": "Critique Social Media Manager Performance",
                "current_insights": insights,
                "recent_errors": [json.loads(f.content) for f in error_fragments if f.content],
                "estimated_ban_rate": ban_rate_estimate,
                "desired_output_format": "JSON: { \"overall_assessment\": str, \"strengths\": list[str], \"weaknesses\": list[str] (e.g., low engagement on X, high error rate posting to Y), \"suggestions_for_improvement\": list[str] (e.g., 'Refine content style for Facebook', 'Investigate BrowsingAgent errors on TikTok', 'Rotate proxies more aggressively') }"
            }
            critique_prompt = await self.generate_dynamic_prompt(critique_context)
            critique_json = await self._call_llm_with_retry(
                 critique_prompt, model=self.config.get("OPENROUTER_MODELS", {}).get('agent_critique', "google/gemini-pro"),
                 temperature=0.5, max_tokens=800, is_json_output=True
            )

            if critique_json:
                 try:
                     critique_result = json.loads(critique_json[critique_json.find('{'):critique_json.rfind('}')+1])
                     critique['feedback'] = critique_result.get('overall_assessment', 'LLM critique generated.')
                     critique['details'] = critique_result
                     self.logger.info(f"Self-Critique Assessment: {critique['feedback']}")
                     await self.log_knowledge_fragment(
                         agent_source=self.AGENT_NAME, data_type="self_critique_summary",
                         content=critique_result, tags=["critique", "social_media"], relevance_score=0.8
                     )
                 except (json.JSONDecodeError, ValueError, KeyError) as e:
                      self.logger.error(f"Failed to parse self-critique LLM response: {e}")
                      critique['feedback'] += " Failed to parse LLM critique."
            else:
                 critique['feedback'] += " LLM critique call failed."
                 critique['status'] = 'error'

        except Exception as e:
            self.logger.error(f"Error during self-critique: {e}", exc_info=True)
            critique['status'] = 'error'
            critique['feedback'] = f"Self-critique failed: {e}"

        return critique

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Constructs prompts for LLM calls."""
        self.logger.debug(f"Generating dynamic prompt for SocialMediaManager task: {task_context.get('task')}")
        prompt_parts = [SOCIAL_MEDIA_MANAGER_META_PROMPT]

        prompt_parts.append("\n--- Current Task Context ---")
        for key, value in task_context.items():
             # Limit length of potentially large context items
             value_str = str(value)
             if len(value_str) > 1000: value_str = value_str[:1000] + "..."
             prompt_parts.append(f"{key.replace('_', ' ').title()}: {value_str}")

        # Add relevant context from KB (Example)
        prompt_parts.append("\n--- Relevant Knowledge (KB - Placeholder) ---")
        prompt_parts.append("- Trend: Short-form video dominates TikTok engagement.")
        prompt_parts.append("- Competitor X uses strategy Y (Source: OSINT ID 123).")
        prompt_parts.append(f"- Preferred Style ({task_context.get('platform', 'general')}): {self.internal_state.get('content_style_preferences', {}).get(task_context.get('platform'), 'Direct & Value-Driven')}")

        prompt_parts.append("\n--- Instructions ---")
        task_type = task_context.get('task')
        # Add specific instructions based on task_type (similar to previous implementation)
        if task_type == 'Refine social media content':
            prompt_parts.append(f"1. Adapt 'Original Content' for platform '{task_context.get('platform')}' aiming for goal '{task_context.get('goal')}'.")
            prompt_parts.append("2. Ensure tone is engaging, human-like, platform-appropriate, and avoids AI detection triggers.")
            prompt_parts.append("3. Include relevant hashtags, emojis, and a clear CTA.")
            prompt_parts.append(f"4. **Output Format:** {task_context.get('desired_output_format')}")
        elif task_type == 'Create Social Media Campaign Plan':
             prompt_parts.append("1. Generate a structured campaign plan based on goal, platform(s), duration, audience.")
             prompt_parts.append("2. Incorporate strategies like 9-to-1 funnels, multi-account interactions.")
             prompt_parts.append("3. Specify target account groups, action types, detailed content briefs, interaction targets, timing.")
             prompt_parts.append(f"4. **Output Format:** {task_context.get('desired_output_format')}")
        elif task_type == 'Analyze social media data':
             prompt_parts.append(f"1. Analyze 'Fetched Data Summary' focusing on metric: '{task_context.get('metric')}'.")
             prompt_parts.append("2. Identify key trends, insights, anomalies, and actionable takeaways.")
             prompt_parts.append(f"3. **Output Format:** {task_context.get('desired_output_format')}")
        elif task_type == 'Generate post content from brief':
             prompt_parts.append(f"1. Generate engaging post content for platform '{task_context.get('platform')}' based on the 'Content Brief'.")
             prompt_parts.append(f"2. Target audience: {task_context.get('target_audience')}. Tone: Engaging, human-like.")
             prompt_parts.append("3. Include hashtags and CTA if appropriate.")
             prompt_parts.append(f"4. **Output Format:** {task_context.get('desired_output_format')}")
        elif task_type == 'Generate relevant social media comment':
             prompt_parts.append("1. Generate a relevant, engaging, human-like comment based on the 'Target Post Context'.")
             prompt_parts.append("2. Avoid generic phrases. Aim to add value or initiate conversation.")
             prompt_parts.append("3. Keep it concise and platform-appropriate.")
             prompt_parts.append(f"4. **Output Format:** Plain text comment.")
        else:
            prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")

        if "JSON" in task_context.get('desired_output_format', ''): prompt_parts.append("\n```json")

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for SocialMediaManager (length: {len(final_prompt)} chars)")
        return final_prompt

    async def collect_insights(self) -> Dict[str, Any]:
        """Collects insights about social media performance and account status."""
        self.logger.debug("SocialMediaManager collect_insights called.")
        active_campaign = self.internal_state.get('current_campaign_plan') is not None
        managed_account_summary = {p: len(a) for p, a in self.internal_state.get('managed_accounts', {}).items()}
        # TODO: Query DB/KB for actual performance metrics (e.g., avg engagement rate last 7d)
        avg_engagement_rate = "N/A" # Placeholder
        account_health_summary = "N/A" # Placeholder (e.g., % accounts active/warned/banned)

        return {
            "agent_name": self.AGENT_NAME, "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_campaign": active_campaign,
            "managed_accounts": managed_account_summary,
            "avg_engagement_rate_7d": avg_engagement_rate,
            "account_health_summary": account_health_summary,
            "key_observations": ["Performance metric aggregation needed for full insights."]
        }

    # --- KB Interaction Helpers (Delegate or Direct) ---
    async def log_knowledge_fragment(self, *args, **kwargs):
        if self.kb_interface and hasattr(self.kb_interface, 'log_knowledge_fragment'): return await self.kb_interface.log_knowledge_fragment(*args, **kwargs)
        elif self.think_tool and hasattr(self.think_tool, 'log_knowledge_fragment'): return await self.think_tool.log_knowledge_fragment(*args, **kwargs)
        else: self.logger.error("No mechanism available to log knowledge fragment."); return None

    async def query_knowledge_base(self, *args, **kwargs):
        if self.kb_interface and hasattr(self.kb_interface, 'query_knowledge_base'): return await self.kb_interface.query_knowledge_base(*args, **kwargs)
        elif self.think_tool and hasattr(self.think_tool, 'query_knowledge_base'): return await self.think_tool.query_knowledge_base(*args, **kwargs)
        else: self.logger.error("No mechanism available to query knowledge base."); return []

    async def log_learned_pattern(self, *args, **kwargs):
         if self.kb_interface and hasattr(self.kb_interface, 'log_learned_pattern'): return await self.kb_interface.log_learned_pattern(*args, **kwargs)
         elif self.think_tool and hasattr(self.think_tool, 'log_learned_pattern'): return await self.think_tool.log_learned_pattern(*args, **kwargs)
         else: self.logger.error("No mechanism available to log learned pattern."); return None

# --- End of agents/social_media_manager.py ---