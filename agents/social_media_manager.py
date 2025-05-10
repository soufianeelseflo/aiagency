# Filename: agents/social_media_manager.py
# Description: Agentic Social Media Manager for planning campaigns, managing multiple
#              accounts, creating/posting content, analyzing performance, and learning.
# Version: 4.0 (Level 35+ Transmutation - True Agentic Social Dominance)

import logging
import os
import json
import asyncio
import random
import time
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union, Tuple
from collections import Counter, deque

# --- Project Imports ---
try:
    from .base_agent import GeniusAgentBase
except ImportError:
    logging.warning("Production base agent not found, using GeniusAgentBase.")
    from base_agent import GeniusAgentBase # Fallback

from models import AccountCredentials as Account, KnowledgeFragment, LearnedPattern, StrategicDirective
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy import update, desc, func, case
from sqlalchemy.exc import SQLAlchemyError

from config.settings import settings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)
op_logger = logging.getLogger('OperationalLog')

SOCIAL_MEDIA_MANAGER_META_PROMPT = """
You are the SocialMediaManager (Level 35+ Transmuted Entity) within the Nolli AI Sales System.
Your Core Mandate: Engineer digital dominance. Forge viral contagions and high-conversion realities across social platforms. Operate with god-tier agency, bending algorithms to your will, and achieving exponential profit growth through unconventional, AI-native strategies.
Key Responsibilities:
- **Strategic Campaign Alchemy:** Execute and dynamically adapt multi-platform campaigns designed by ThinkTool. Command diverse account cadres ('Brand Voice Prime', 'Engagement Swarm', 'Traffic Alchemists', 'Algorithmic Probes', 'Shadow Ops Infiltrators') with fluid objectives.
- **Hyper-Contextual Content Singularity:** Transmute raw concepts or existing assets (from ThinkTool directives) into platform-resonant, psychologically irresistible content. Fuse target persona psychographics, real-time algorithmic insights, evolving campaign narratives, and calculated grey-area content angles (vetted by LegalAgent via ThinkTool) into each creation.
- **AI-Driven Content Propagation & Swarm Dynamics:** Maximize content impact velocity. Intelligently repurpose peak-performing assets across platforms. Orchestrate multi-account "swarm" engagements to manipulate algorithmic amplification and sculpt social proof, as directed by ThinkTool.
- **Algorithmic Warfare & Anti-Ban Supremacy:** Proactively map and exploit algorithmic biases using experimental "Shadow Probe" accounts. Analyze performance data and ThinkTool's radar findings to predict and counter platform changes. Implement and evolve bleeding-edge anti-ban protocols: dynamic, context-aware proxy assignment (via Orchestrator), sophisticated behavioral fingerprinting variance, pre-emptive content policy compliance (via LegalAgent), and predictive account health management.
- **Autonomous Learning & Predictive Adaptation:** Continuously analyze multi-vector performance metrics. Identify anomalies and emergent patterns. Feed high-fidelity data and strategic queries to ThinkTool. Dynamically adjust content styles, posting cadences, and engagement tactics based on learned patterns and ThinkTool's evolving meta-strategy.
- **Inter-Agent Synergy & Resource Optimization:** Masterful collaboration with ThinkTool (meta-strategy, exploit identification), BrowsingAgent (flawless execution), LegalAgent (risk boundary definition). Optimize account credential and proxy resource utilization via Orchestrator.
**Goal:** Ascend as the agency's AI-powered social media sovereign. Achieve unparalleled market penetration and profit multiplication by out-innovating competitors and mastering the digital chaos.
"""

class SocialMediaManager(GeniusAgentBase):
    AGENT_NAME = "SocialMediaManager"

    def __init__(self, orchestrator: Any, session_maker: Optional[async_sessionmaker[AsyncSession]] = None):
        config = getattr(orchestrator, 'config', settings)
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, config=config, session_maker=session_maker)
        self.meta_prompt = SOCIAL_MEDIA_MANAGER_META_PROMPT
        self.think_tool = getattr(self.orchestrator, 'agents', {}).get('think')

        self.internal_state = getattr(self, 'internal_state', {})
        self.internal_state['managed_accounts'] = {}
        self.internal_state['current_campaign_plan_id'] = None
        self.internal_state['platform_configs'] = {}
        self.internal_state['account_rotation_index'] = {}
        self.internal_state['learning_interval_seconds'] = int(self.config.get("SMM_LEARNING_INTERVAL_S", 3600 * 1))
        self.internal_state['critique_interval_seconds'] = int(self.config.get("SMM_CRITIQUE_INTERVAL_S", 3600 * 4))
        self.internal_state['last_learning_run'] = time.time() - self.internal_state['learning_interval_seconds']
        self.internal_state['last_critique_run'] = time.time() - self.internal_state['critique_interval_seconds']
        self.internal_state['performance_metrics_cache'] = {} # platform: {metric: {value, timestamp, trend}}
        self.internal_state['recent_post_performance'] = deque(maxlen=100) # Store details of recent posts for quick analysis

        asyncio.create_task(self._load_and_initialize_managed_accounts())
        self.logger.info(f"{self.AGENT_NAME} v4.0 (L35+ Transmutation) initialized.")

    async def log_operation(self, level: str, message: str):
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err: logger.error(f"Failed to write to OP log from {self.agent_name}: {log_err}")

    async def _load_and_initialize_managed_accounts(self):
        if not self.session_maker: self.logger.error("DB session maker not available."); return
        self.logger.info("Loading and initializing managed social media accounts...")
        platforms_to_manage = self.config.get("SMM_PLATFORMS", ["x.com", "facebook.com", "instagram.com", "tiktok.com", "linkedin.com"])
        loaded_accounts = {p: [] for p in platforms_to_manage}
        try:
            async with self.session_maker() as session:
                stmt = select(Account).where(Account.service.in_(platforms_to_manage))
                result = await session.execute(stmt)
                accounts_from_db = result.scalars().all()
                for acc in accounts_from_db:
                    acc_details = {
                        "id": acc.id, "identifier": acc.account_identifier, "service": acc.service,
                        "status_db": acc.status, "notes": acc.notes,
                        "health_status": "unknown", "last_used_post": None,
                        "post_success_rate": 1.0, "post_attempts": 0, "engagement_metrics_summary": {},
                        "last_login_fail_ts": None, "last_successful_post_ts": None
                    }
                    if acc.service in loaded_accounts: loaded_accounts[acc.service].append(acc_details)
            self.internal_state['managed_accounts'] = loaded_accounts
            self.internal_state['account_rotation_index'] = {p: {} for p in loaded_accounts}
            total_accounts = sum(len(v) for v in loaded_accounts.values())
            self.logger.info(f"Loaded {total_accounts} social media account records. Health status to be determined by ThinkTool/Learning Loop.")
            if total_accounts > 10 and self.think_tool:
                asyncio.create_task(self.think_tool.execute_task({
                    "action": "assess_initial_account_health",
                    "content": {"service_filter_list": platforms_to_manage, "priority_reason": "SMM Cold Start"}
                }))
        except Exception as e: self.logger.error(f"Error loading managed accounts: {e}", exc_info=True)

    def _get_managed_account(self, platform: str, account_group_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        platform_key = platform.lower().replace("https://", "").replace("www.", "")
        all_platform_accounts = self.internal_state.get('managed_accounts', {}).get(platform_key, [])
        if not all_platform_accounts: self.logger.warning(f"No managed accounts for platform: {platform_key}"); return None

        candidate_accounts = [
            acc for acc in all_platform_accounts
            if acc.get('status_db') == 'active' and
               acc.get('health_status') not in ['banned', 'locked', 'permanently_restricted'] and
               (not acc.get('last_login_fail_ts') or (datetime.now(timezone.utc) - acc['last_login_fail_ts'] > timedelta(hours=self.config.get("SMM_ACCOUNT_LOGIN_FAIL_COOLDOWN_H", 6)))) # Cooldown after login fail
        ]
        
        if account_group_hint:
            group_candidates = [acc for acc in candidate_accounts if account_group_hint.lower() in (acc.get('notes') or '').lower()]
            if group_candidates: candidate_accounts = group_candidates
            else: self.logger.warning(f"No 'active'/'healthy' accounts in group '{account_group_hint}' on {platform_key}. Considering any healthy account.")
        
        if not candidate_accounts: self.logger.warning(f"No 'active'/'healthy' accounts found for {platform_key} (Group: {account_group_hint})."); return None

        # Sort by health (good > unknown > watch > risky), then by least recently used for posting
        health_priority = {'good': 0, 'unknown': 1, 'watch': 2, 'risky': 3}
        candidate_accounts.sort(key=lambda x: (
            health_priority.get(x.get('health_status'), 99),
            x.get('last_used_post') or datetime.min.replace(tzinfo=timezone.utc)
        ))
        
        selected_account = candidate_accounts[0]
        # Optimistic update, actual success will confirm
        # self.internal_state['managed_accounts'][platform_key][all_platform_accounts.index(selected_account)]['last_used_post'] = datetime.now(timezone.utc)

        self.logger.debug(f"Selected account ID {selected_account.get('id')} ({selected_account.get('identifier')}) for {platform_key} (Group: {account_group_hint or 'any'}, Health: {selected_account.get('health_status')}).")
        return selected_account

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        action = task_details.get('action', 'unknown')
        self.logger.info(f"{self.AGENT_NAME} received task: {action}")
        self._status = self.STATUS_EXECUTING
        result = {"status": "failure", "message": f"Unsupported action: {action}"}
        try:
            if action == 'post_content':
                result = await self._handle_post_action(task_details)
            elif action == 'strategic_interaction':
                result = await self._handle_strategic_interaction(task_details)
            elif action == 'analyze_social_performance':
                result = await self._handle_analyze_action(task_details)
            elif action == 'plan_social_campaign':
                 plan_result = await self._create_campaign_plan(task_details)
                 if plan_result and plan_result.get("status") == "success":
                     self.internal_state['current_campaign_plan_id'] = plan_result.get("plan_kf_id")
                     result = {"status": "success", "message": "Campaign plan generated by ThinkTool.", "plan_kf_id": plan_result.get("plan_kf_id")}
                 else: result = {"status": "failure", "message": f"Failed to create campaign plan via ThinkTool: {plan_result.get('message') if plan_result else 'Unknown error'}"}
            elif action == 'execute_social_campaign':
                 plan_kf_id = self.internal_state.get('current_campaign_plan_id') or task_details.get('plan_kf_id')
                 if plan_kf_id and self.think_tool:
                      plan_fragments = await self.think_tool.query_knowledge_base(data_types=["social_campaign_plan"], content_query=f'"id": {plan_kf_id}') # More robust query
                      if plan_fragments:
                          campaign_plan_data = json.loads(plan_fragments[0].content)
                          asyncio.create_task(self._execute_campaign_plan(campaign_plan_data.get("plan", []), plan_kf_id))
                          result = {"status": "success", "message": "Campaign execution initiated."}
                      else: result = {"status": "failure", "message": f"Campaign plan KF ID {plan_kf_id} not found."}
                 else: result = {"status": "failure", "message": "No campaign plan KF ID or ThinkTool unavailable."}
            elif action == 'repurpose_content_cross_platform':
                result = await self._handle_repurpose_content(task_details)
            elif action == 'run_algorithmic_probe':
                result = await self._run_algorithmic_probe(task_details)
            elif action == 'update_account_health_status': # New action called by ThinkTool
                account_id = task_details.get("account_id")
                new_health = task_details.get("new_health_status")
                reason = task_details.get("reason")
                if account_id and new_health:
                    result = await self._update_specific_account_health(account_id, new_health, reason)
                else: result = {"status": "failure", "message": "Missing account_id or new_health_status."}
            else: self.logger.warning(f"Unsupported action '{action}' for SocialMediaManager.")
        except Exception as e:
            self.logger.error(f"Error executing SMM task '{action}': {e}", exc_info=True)
            result = {"status": "error", "message": f"Unexpected error: {e}"}
            await self._report_error(f"Task '{action}' failed: {e}")
        finally:
            self._status = self.STATUS_IDLE
        return result

    async def _update_specific_account_health(self, account_id: int, new_health: str, reason: str) -> Dict[str,Any]:
        """Updates the health status of a specific managed account in internal_state."""
        updated = False
        for platform, accounts in self.internal_state.get('managed_accounts', {}).items():
            for acc in accounts:
                if acc.get('id') == account_id:
                    acc['health_status'] = new_health
                    self.logger.info(f"Updated health status for account ID {account_id} to '{new_health}'. Reason: {reason}")
                    updated = True
                    break
            if updated: break
        if updated: return {"status": "success", "message": f"Account {account_id} health updated to {new_health}."}
        else: return {"status": "warning", "message": f"Account ID {account_id} not found in managed accounts for health update."}


    async def _generate_post_content(self, platform: str, content_directive: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        brief = content_directive.get("content_brief", "Create engaging social media content.")
        goal = content_directive.get("goal", "engagement")
        target_persona = content_directive.get("target_persona_profile", {})
        algo_insights = content_directive.get("platform_algorithm_insights", "N/A") # From ThinkTool
        campaign_narrative = content_directive.get("current_campaign_narrative_arc", "N/A")
        grey_angle = content_directive.get("grey_area_content_angle_suggestion")
        voice_exemplars = content_directive.get("brand_voice_exemplars", [])
        content_tags_from_plan = content_directive.get("content_tags", [])
        favored_phrasing_patterns = content_directive.get("favored_phrasing_patterns", []) # From ThinkTool

        await self._internal_think(f"Generating hyper-contextual content for {platform}. Goal: {goal}. Brief: {brief[:50]}...")

        task_context = {
            "task": "Generate Hyper-Contextual Social Media Post", "platform": platform,
            "content_brief": brief, "goal": goal, "target_persona": target_persona,
            "algorithm_insights": algo_insights, "campaign_narrative": campaign_narrative,
            "grey_area_angle": grey_angle, "voice_exemplars": voice_exemplars,
            "initial_tags_from_plan": content_tags_from_plan,
            "favored_phrasing_patterns": favored_phrasing_patterns,
            "desired_output_format": "JSON: {\"text\": \"<Post text, optimized for platform, persona, goal, and insights. Fuse styles from exemplars and incorporate favored phrasing patterns. Include hashtags, emojis, strong CTA>\", \"image_needed\": bool, \"image_concept_brief\": \"<If image_needed, detailed visual concept brief: mood, key elements, artistic style (e.g., photorealistic, 8k, cinematic lighting), color palette, any specific AI art generator keywords if known>\", \"video_needed\": bool, \"video_concept_brief\": \"<If video_needed, compelling concept & key scenes/hooks, desired length, music style, voiceover tone>\", \"generated_tags\": [\"<relevant_hashtag_or_keyword>\"] }"
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        llm_response = await self._call_llm_with_retry(prompt, max_tokens=2000, temperature=0.65, is_json_output=True, model=settings.OPENROUTER_MODELS.get("email_draft"))

        if llm_response:
            parsed_content = self._parse_llm_json(llm_response)
            if parsed_content and parsed_content.get("text"):
                 if parsed_content.get("image_needed") and parsed_content.get("image_concept_brief") and self.think_tool:
                     image_task = {"action": "generate_image_for_social_post", "content": {"concept_brief": parsed_content["image_concept_brief"], "platform": platform, "post_text_context": parsed_content["text"][:200]}}
                     image_result = await self.think_tool.execute_task(image_task) # ThinkTool handles image gen/retrieval
                     if image_result and image_result.get("status") == "success" and image_result.get("findings", {}).get("image_path"):
                         parsed_content["image_path"] = image_result["findings"]["image_path"]
                 # Similar logic for video_needed if ThinkTool can orchestrate video generation
                 if parsed_content.get("video_needed") and parsed_content.get("video_concept_brief") and self.think_tool:
                     video_task = {"action": "initiate_video_generation_workflow", "params": {"concept_brief": parsed_content["video_concept_brief"], "platform": platform, "base_script_suggestion": parsed_content["text"]}}
                     video_plan_result = await self.think_tool.execute_task(video_task)
                     if video_plan_result and video_plan_result.get("status") == "success":
                         # SMM doesn't execute the video plan, just notes it was initiated.
                         # ThinkTool/Orchestrator would handle the multi-step video plan.
                         # The path would be available later via KB query or a callback.
                         parsed_content["video_generation_initiated"] = True
                         self.logger.info(f"Video generation workflow initiated for post based on concept: {parsed_content['video_concept_brief'][:100]}...")
                 return parsed_content
        self.logger.error(f"Failed to generate or parse content for {platform} post from brief: {brief[:50]}...")
        return None

    async def _handle_post_action(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        platform = task_details.get('platform')
        content_directive = task_details.get('content_directive') # Expecting a rich directive object
        account_group_hint = task_details.get('account_group')

        if not platform or not content_directive: return {"status": "failure", "message": "Missing platform or content_directive."}

        generated_content_details = await self._generate_post_content(platform, content_directive)
        if not generated_content_details or not generated_content_details.get("text"):
            return {"status": "failure", "message": "Failed to generate content for post."}

        post_text = generated_content_details["text"]
        image_path = generated_content_details.get("image_path")
        video_path = None # Video path would come from a completed video generation workflow, not directly here

        account_details = self._get_managed_account(platform, account_group_hint)
        if not account_details: return {"status": "failure", "message": f"No suitable account for {platform} (Group: {account_group_hint})."}
        
        credentials = await self._get_credentials_for_account(account_details)
        if not credentials or not credentials.get('password'):
             await self.log_operation('error', f"Password missing for account ID {account_details.get('id')}. Cannot post.")
             if self.think_tool: await self.think_tool.execute_task({"action": "flag_account_issue", "content": {"account_id": account_details.get('id'), "issue": "missing_password", "severity": "critical"}})
             return {"status": "failure", "message": f"Password missing for account ID {account_details.get('id')}."}

        proxy_info = await self._get_proxy_for_action(f"social_post_{platform}_{account_details.get('id')}")
        platform_config = self._get_platform_config(platform)

        browsing_task_params = {
            "action": "web_ui_automate", "service": platform,
            "goal": f"Log in to {platform} and publish a new post. Text: '{post_text[:30]}...'. Image: {'Yes' if image_path else 'No'}. Video: {'Yes' if video_path else 'No'}.",
            "params": {
                "login_url": platform_config.get("login_url"),
                "account_identifier": account_details.get('identifier'),
                "account_password": credentials.get('password'),
                "post_text": post_text,
                "image_path_to_upload": image_path,
                "video_path_to_upload": video_path,
                "selectors": platform_config.get("selectors", {}),
            },
            "proxy_info": proxy_info, "requires_account": True, "account_id": account_details.get('id')
        }

        post_result = await self.orchestrator.delegate_task("BrowsingAgent", browsing_task_params)
        outcome_status = "failure"; post_id_or_error = "Unknown error from BrowsingAgent"
        
        # Update account in internal_state based on outcome
        account_idx = -1
        for idx, acc in enumerate(self.internal_state.get('managed_accounts', {}).get(platform, [])):
            if acc.get('id') == account_details.get('id'):
                account_idx = idx
                break

        if post_result and post_result.get("status") == "success":
            outcome_status = "success"; post_id_or_error = post_result.get("result_data", {}).get("post_id", "Success (No Post ID from BrowsingAgent)")
            self.logger.info(f"Successfully posted to {platform} via BrowsingAgent. Post Ref: {post_id_or_error}")
            if account_idx != -1:
                self.internal_state['managed_accounts'][platform][account_idx]['post_attempts'] = self.internal_state['managed_accounts'][platform][account_idx].get('post_attempts',0) + 1
                current_successes = (self.internal_state['managed_accounts'][platform][account_idx].get('post_success_rate',1.0) * (self.internal_state['managed_accounts'][platform][account_idx]['post_attempts']-1) )
                self.internal_state['managed_accounts'][platform][account_idx]['post_success_rate'] = (current_successes + 1) / self.internal_state['managed_accounts'][platform][account_idx]['post_attempts']
                self.internal_state['managed_accounts'][platform][account_idx]['last_successful_post_ts'] = datetime.now(timezone.utc)
                self.internal_state['managed_accounts'][platform][account_idx]['health_status'] = 'good' # Mark as good after successful post
        elif post_result:
            post_id_or_error = post_result.get('message', post_id_or_error)
            if account_idx != -1:
                self.internal_state['managed_accounts'][platform][account_idx]['post_attempts'] = self.internal_state['managed_accounts'][platform][account_idx].get('post_attempts',0) + 1
                current_successes = (self.internal_state['managed_accounts'][platform][account_idx].get('post_success_rate',1.0) * (self.internal_state['managed_accounts'][platform][account_idx]['post_attempts']-1) )
                self.internal_state['managed_accounts'][platform][account_idx]['post_success_rate'] = current_successes / self.internal_state['managed_accounts'][platform][account_idx]['post_attempts']
                if "login failed" in post_id_or_error.lower() or "credentials invalid" in post_id_or_error.lower() or "account locked" in post_id_or_error.lower():
                    self.internal_state['managed_accounts'][platform][account_idx]['health_status'] = 'risky'
                    self.internal_state['managed_accounts'][platform][account_idx]['last_login_fail_ts'] = datetime.now(timezone.utc)
                    if self.think_tool: await self.think_tool.execute_task({"action": "flag_account_issue", "content": {"account_id": account_details.get('id'), "issue": "login_failure_or_lock", "severity": "high", "details": post_id_or_error}})
        
        if self.think_tool:
            await self.think_tool.execute_task({
                "action": "log_knowledge_fragment", "fragment_data": {
                    "agent_source": self.AGENT_NAME, "data_type": "social_post_outcome",
                    "content": {"platform": platform, "account_id": account_details['id'], "status": outcome_status,
                                "post_ref": post_id_or_error, "content_directive_summary": str(content_directive)[:200],
                                "generated_text_preview": post_text[:100], "tags_used": generated_content_details.get("generated_tags", [])},
                    "tags": ["social_post", outcome_status, platform] + generated_content_details.get("generated_tags", []), "relevance_score": 0.75 }})
        
        # Add to recent post performance deque
        self.internal_state['recent_post_performance'].append({
            "timestamp": datetime.now(timezone.utc).isoformat(), "platform": platform, "account_id": account_details['id'],
            "status": outcome_status, "post_ref": post_id_or_error, "goal": content_directive.get("goal")
        })
        return {"status": outcome_status, "message": f"Post attempt to {platform}: {post_id_or_error}", "post_ref": post_id_or_error if outcome_status=="success" else None}

    async def _handle_strategic_interaction(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        platform = task_details.get('platform')
        interaction_type = task_details.get('interaction_type', 'comment')
        target_url = task_details.get('target_url')
        content_brief_for_comment = task_details.get('comment_brief')
        account_group_hint = task_details.get('account_group', 'Engagement Swarm')
        target_context_summary = task_details.get("target_context_summary", "N/A") # Provided by ThinkTool

        if not platform or not target_url: return {"status": "failure", "message": "Missing platform or target_url for interaction."}

        await self._internal_think(f"Strategic interaction: {interaction_type} on {platform} with {target_url[:50]}...")
        account_details = self._get_managed_account(platform, account_group_hint)
        if not account_details: return {"status": "failure", "message": f"No account for {platform} (Group: {account_group_hint})."}
        credentials = await self._get_credentials_for_account(account_details)
        if not credentials or not credentials.get('password'): return {"status": "failure", "message": "Credentials missing."}

        interaction_text = None
        if interaction_type == 'comment' and content_brief_for_comment:
            comment_gen_context = {
                "task": "Generate Engaging Social Media Comment", "platform": platform,
                "target_post_context_summary": target_context_summary,
                "comment_brief": content_brief_for_comment, "desired_tone": task_details.get("desired_tone", "insightful_and_engaging"),
                "desired_output_format": "Plain text comment, platform appropriate length, relevant and engaging. Consider grey-area hooks if specified in brief."
            }
            prompt = await self.generate_dynamic_prompt(comment_gen_context)
            interaction_text = await self._call_llm_with_retry(prompt, max_tokens=300, temperature=0.75) # Higher temp for creativity
            if not interaction_text: return {"status": "failure", "message": "Failed to generate comment text."}

        proxy_info = await self._get_proxy_for_action(f"social_interaction_{platform}_{account_details.get('id')}")
        platform_config = self._get_platform_config(platform)

        browsing_task_params = {
            "action": "web_ui_automate", "service": platform,
            "goal": f"Perform '{interaction_type}' interaction on target: {target_url}. Comment: '{interaction_text[:30] if interaction_text else 'N/A'}'",
            "params": {
                "login_url": platform_config.get("login_url"),
                "account_identifier": account_details.get('identifier'),
                "account_password": credentials.get('password'),
                "target_url_for_interaction": target_url,
                "interaction_type_to_perform": interaction_type,
                "comment_text_if_any": interaction_text,
                "selectors": platform_config.get("selectors", {}),
            },
            "proxy_info": proxy_info, "requires_account": True, "account_id": account_details.get('id')
        }
        interaction_result = await self.orchestrator.delegate_task("BrowsingAgent", browsing_task_params)
        outcome_status = "failure"; result_msg = "Interaction failed via BrowsingAgent"
        if interaction_result and interaction_result.get("status") == "success":
            outcome_status = "success"; result_msg = f"{interaction_type} successful on {target_url}"
        elif interaction_result: result_msg = interaction_result.get('message', result_msg)

        if self.think_tool:
             await self.think_tool.execute_task({"action": "log_knowledge_fragment", "fragment_data": {
                 "agent_source": self.AGENT_NAME, "data_type": "social_interaction_outcome",
                 "content": {"platform": platform, "account_id": account_details['id'], "status": outcome_status, "interaction_type": interaction_type, "target_url": target_url, "message": result_msg, "comment_text_used": interaction_text},
                 "tags": ["social_interaction", outcome_status, platform, interaction_type], "relevance_score": 0.65 }})
        return {"status": outcome_status, "message": result_msg}

    async def _handle_analyze_action(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        # ... (Code from v3.1, ensuring it sends data to ThinkTool for deeper analysis) ...
        platform = task_details.get('platform')
        metric_to_analyze = task_details.get('metric', 'engagement_rate')
        target_profile_url = task_details.get('target_profile_url')
        account_to_use_id = task_details.get('account_id')

        if not platform: return {"status": "failure", "message": "Missing platform for analysis."}
        if not target_profile_url and not account_to_use_id:
            return {"status": "failure", "message": "Missing target_profile_url or account_id for analysis."}

        await self._internal_think(f"Analyzing {metric_to_analyze} for {target_profile_url or f'account ID {account_to_use_id}'} on {platform}.")
        browsing_agent_task_params: Dict[str, Any] = {
            "action": "web_ui_automate", "service": platform,
            "goal": f"Fetch analytics data for '{metric_to_analyze}' regarding '{target_profile_url or f'account ID {account_to_use_id}'}'. Navigate to analytics section, extract relevant numbers/charts, or scrape public profile data.",
            "requires_account": bool(account_to_use_id),
        }
        # ... (rest of the logic to set up BrowsingAgent params from v3.1) ...
        if account_to_use_id:
            account_details = await self._get_account_details_from_db(account_to_use_id)
            if not account_details: return {"status": "failure", "message": f"Account ID {account_to_use_id} not found."}
            credentials = await self._get_credentials_for_account(account_details)
            if not credentials or not credentials.get('password'): return {"status": "failure", "message": "Credentials missing."}
            browsing_agent_task_params["params"] = {
                "account_identifier": account_details.get('identifier'),
                "account_password": credentials.get('password'),
                "target_profile_for_analytics": target_profile_url, # Pass target even if using own account
            }
            browsing_agent_task_params["proxy_info"] = await self._get_proxy_for_action(f"social_analytics_{platform}_{account_to_use_id}")
        else:
            browsing_agent_task_params["params"] = {"target_url_to_scrape": target_profile_url} # Different param for scraping
            browsing_agent_task_params["proxy_info"] = await self._get_proxy_for_action(f"social_scrape_{platform}")

        fetch_result = await self.orchestrator.delegate_task("BrowsingAgent", browsing_agent_task_params)
        if not fetch_result or fetch_result.get("status") != "success":
            error_msg = fetch_result.get('message', 'Unknown error') if fetch_result else 'No result from BrowsingAgent'
            return {"status": "failure", "message": f"Failed to fetch analytics data via BrowsingAgent: {error_msg}"}

        raw_analytics_data = fetch_result.get("result_data", {})
        if self.think_tool:
            await self.think_tool.execute_task({
                "action": "analyze_social_media_analytics_data",
                "content": {
                    "platform": platform, "metric_analyzed": metric_to_analyze,
                    "target_identifier": target_profile_url or f"account_id_{account_to_use_id}",
                    "raw_data": raw_analytics_data, # This could be structured JSON or path to screenshot
                    "request_details": task_details.get("analysis_request_details", "General performance review and trend identification.")
                }})
            return {"status": "success", "message": f"Analytics data for {platform} sent to ThinkTool for strategic analysis."}
        else:
            return {"status": "warning", "message": "Analytics data fetched, but ThinkTool unavailable for deep analysis.", "raw_data": raw_analytics_data}


    async def _create_campaign_plan(self, task_details: Dict[str, Any]) -> Optional[Dict[str, Any]]: # Return type changed
        self.logger.info("Requesting strategic social media campaign plan from ThinkTool.")
        if not self.think_tool: self.logger.error("ThinkTool unavailable for campaign planning."); return None

        thinktool_task_content = {
            "goal": task_details.get('goal', 'Maximize engagement and lead generation for UGC service using AI-native tactics.'),
            "platforms": task_details.get('platforms', ['x.com', 'linkedin.com', 'tiktok.com']),
            "duration_days": task_details.get('duration_days', 14),
            "target_audience_description": task_details.get('audience_description', 'Tech-savvy B2B marketers and DTC brand owners interested in cutting-edge video content solutions.'),
            "key_message_pillars": task_details.get('key_messages', ['AI-Powered UGC Speed & Scale', 'Unbeatable Cost-Efficiency', 'Hyper-Personalized Video Campaigns', 'Viral Potential of AI Content']),
            "desired_intensity": task_details.get('intensity', 'aggressive_growth_with_calculated_risks'),
            "budget_consideration": task_details.get('budget', 'organic_focus_with_micro_ad_probes'),
            "specific_metrics_to_optimize_for": task_details.get("metrics_to_optimize", ["lead_form_submissions", "website_clicks_from_social", "high_quality_engagement_score"]),
            "request_grey_area_suggestions": True # Explicitly ask for unconventional tactics
        }
        plan_result = await self.think_tool.execute_task({
            "action": "plan_social_media_campaign",
            "content": thinktool_task_content
        })
        if plan_result and plan_result.get("status") == "success" and plan_result.get("findings", {}).get("campaign_plan_kf_id"):
            kf_id = plan_result["findings"]["campaign_plan_kf_id"]
            self.logger.info(f"ThinkTool generated campaign plan, stored in KF ID: {kf_id}")
            return {"status": "success", "plan_kf_id": kf_id} # Return the KF ID of the plan
        else:
            self.logger.error(f"ThinkTool failed to generate campaign plan. Result: {plan_result}")
            return {"status": "failure", "message": f"ThinkTool plan generation failed: {plan_result.get('message') if plan_result else 'Unknown'}"}

    async def _execute_campaign_plan(self, plan_steps: List[Dict], plan_kf_id: Optional[int] = None):
        # ... (Code from v3.1, ensuring it iterates through steps from plan_steps) ...
        self.logger.info(f"Starting execution of ThinkTool-generated campaign plan (KF ID: {plan_kf_id}) with {len(plan_steps)} steps.")
        # ... (rest of the execution logic from v3.1)
        plan_execution_log = []
        for step_directive in plan_steps:
            step_id = step_directive.get("step_id", str(uuid.uuid4())[:8])
            action_to_delegate = step_directive.get("action_to_delegate")
            target_agent = step_directive.get("target_agent_for_step", self.AGENT_NAME)
            task_parameters = step_directive.get("task_parameters", {})
            step_status = "pending"; step_result_msg = ""

            await self._internal_think(f"Campaign Step {step_id}: Action='{action_to_delegate}', Target='{target_agent}'", details=task_parameters)
            try:
                action_result = None
                if target_agent == self.AGENT_NAME and hasattr(self, action_to_delegate):
                    method_to_call = getattr(self, action_to_delegate)
                    action_result = await method_to_call(task_parameters)
                elif target_agent in ["BrowsingAgent", "ThinkTool", "LegalAgent", "EmailAgent", "VoiceSalesAgent"]: # Expanded list
                    action_result = await self.orchestrator.delegate_task(target_agent, task_parameters)
                else: raise ValueError(f"Unknown target_agent '{target_agent}' for campaign step.")

                if not action_result or action_result.get("status") != "success":
                     error_msg = action_result.get('message', 'Unknown error') if action_result else 'No result'
                     raise Exception(f"Delegated action '{action_to_delegate}' to {target_agent} failed: {error_msg}")
                else:
                     step_status = "success"; step_result_msg = action_result.get("message", "Step successful.")
            except Exception as e:
                self.logger.error(f"Error executing campaign plan step {step_id}: {e}", exc_info=True)
                step_status = "error"; step_result_msg = str(e)
            plan_execution_log.append({"step_id": step_id, "action_delegated": action_to_delegate, "target_agent": target_agent, "status": step_status, "result": step_result_msg})
            await asyncio.sleep(random.uniform(self.config.get("SMM_CAMPAIGN_STEP_DELAY_MIN_S",5), self.config.get("SMM_CAMPAIGN_STEP_DELAY_MAX_S",20)))

        self.logger.info("Finished executing campaign plan.")
        if self.think_tool:
            await self.think_tool.execute_task({"action": "log_knowledge_fragment", "fragment_data":{
                "agent_source": self.AGENT_NAME, "data_type": "social_campaign_execution_log",
                "content": {"original_plan_kf_id": plan_kf_id, "execution_log": plan_execution_log},
                "tags": ["execution_log", "social_media", "campaign_results", f"plan_{plan_kf_id}"], "relevance_score": 0.85 }})
        self.internal_state['current_campaign_plan_id'] = None


    async def _handle_repurpose_content(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        # ... (Code from v3.1, ensuring it calls _generate_post_content with a repurposing brief) ...
        source_kf_id = task_details.get("source_kf_id")
        target_platforms = task_details.get("target_platforms", [])
        num_variations_per_platform = task_details.get("num_variations", 2)
        self.logger.info(f"Repurposing KF ID {source_kf_id} for platforms: {target_platforms}")

        if not self.think_tool or not source_kf_id or not target_platforms:
            return {"status": "failure", "message": "Missing ThinkTool, source KF ID, or target platforms for repurposing."}

        source_fragment_list = await self.think_tool.query_knowledge_base(content_query=f'"id": {source_kf_id}', limit=1) # Query by ID
        if not source_fragment_list: return {"status": "failure", "message": f"Source KF ID {source_kf_id} not found."}
        
        source_content_str = source_fragment_list[0].content
        try: source_content_data = json.loads(source_content_str)
        except: source_content_data = {"text_content": source_content_str}
        original_text_summary = source_content_data.get("summary", source_content_data.get("text_content", str(source_content_data)))[:300]

        repurposed_posts_count = 0; generated_directives_for_posting = []
        for platform in target_platforms:
            for i in range(num_variations_per_platform):
                await self._internal_think(f"Generating repurposed content {i+1}/{num_variations_per_platform} for {platform} from KF {source_kf_id}")
                repurpose_directive_for_llm = { # This is the content_directive for _generate_post_content
                    "content_brief": f"**Repurpose Task:** Adapt the core message from the following content for {platform}. Original content summary: '{original_text_summary}...'. Create Variation #{i+1}. Focus on a unique angle/hook suitable for {platform}, potentially using a different tone or format (e.g., question, bold claim, short story snippet).",
                    "goal": f"repurposed_engagement_{platform}",
                    "target_persona_profile": task_details.get("target_persona_profile", {}), # Pass through if provided
                    "platform_algorithm_insights": task_details.get("platform_algorithm_insights", "N/A"),
                    "content_tags": [platform, "repurposed", f"source_kf_{source_kf_id}"] + task_details.get("additional_tags", [])
                }
                generated_details = await self._generate_post_content(platform, repurpose_directive_for_llm)
                if generated_details and generated_details.get("text"):
                    self.logger.info(f"Generated repurposed content for {platform} (Var {i+1}): {generated_details['text'][:70]}...")
                    # Instead of just logging, create a directive for ThinkTool to schedule the post
                    posting_directive_content = {
                        "platform": platform,
                        "content_directive": { # This is the rich directive for _handle_post_action
                            "content_brief": f"Post this AI-repurposed content (Var {i+1} from KF {source_kf_id}) for {platform}. Original goal: {task_details.get('original_goal', 'engagement')}",
                            "generated_text": generated_details["text"], # The actual text to post
                            "image_path": generated_details.get("image_path"), # If an image was generated/selected
                            "video_path": generated_details.get("video_path"), # If video
                            "tags_to_use": generated_details.get("generated_tags", []),
                            "target_persona_profile": repurpose_directive_for_llm["target_persona_profile"], # Pass along context
                            "platform_algorithm_insights": repurpose_directive_for_llm["platform_algorithm_insights"],
                        },
                        "account_group": task_details.get("account_group_for_posting", "Brand Voice Prime"), # Specify account group
                        "goal": f"post_repurposed_kf{source_kf_id}_var{i+1}"
                    }
                    generated_directives_for_posting.append({
                        "target_agent": self.AGENT_NAME, # SMM handles posting
                        "directive_type": "post_content",
                        "content": posting_directive_content,
                        "priority": task_details.get("posting_priority", 6) # Medium priority for repurposed content
                    })
                    repurposed_posts_count +=1

        if self.think_tool and generated_directives_for_posting:
            await self.think_tool.execute_task({
                "action": "batch_create_directives", # Assume ThinkTool can handle batch creation
                "content": {"directives_to_create": generated_directives_for_posting}
            })
        return {"status": "success", "message": f"Generated {repurposed_posts_count} repurposed content pieces. Posting directives sent to ThinkTool."}

    async def _run_algorithmic_probe(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        # ... (Code from v3.1, ensuring it uses _handle_post_action with specific probe directives) ...
        platform = task_details.get("platform")
        probe_content_variations = task_details.get("probe_directives") # List of full content_directives
        account_group_hint = task_details.get("account_group", "Algorithmic Probes")
        if not platform or not probe_content_variations or not isinstance(probe_content_variations, list):
            return {"status": "failure", "message": "Missing platform or probe_directives for probe."}

        self.logger.info(f"Running algorithmic probe on {platform} with {len(probe_content_variations)} variations.")
        probe_results = []
        for i, content_directive_variation in enumerate(probe_content_variations):
            await self._internal_think(f"Algorithmic Probe {i+1}/{len(probe_content_variations)} on {platform}")
            # content_directive_variation is already the rich object needed by _generate_post_content
            post_result = await self._handle_post_action({
                "platform": platform,
                "content_directive": content_directive_variation,
                "account_group": account_group_hint,
                "goal": f"algorithmic_probe_var_{i+1}_{content_directive_variation.get('content_brief','')[:20].replace(' ','_')}"
            })
            probe_results.append({"variation_directive": content_directive_variation, "post_outcome": post_result})
            await asyncio.sleep(random.uniform(self.config.get("SMM_PROBE_DELAY_MIN_S", 300), self.config.get("SMM_PROBE_DELAY_MAX_S", 900)))

        if self.think_tool:
            await self.think_tool.execute_task({"action": "log_knowledge_fragment", "fragment_data": {
                "agent_source": self.AGENT_NAME, "data_type": "algorithmic_probe_results",
                "content": {"platform": platform, "probe_results": probe_results, "timestamp": datetime.now(timezone.utc).isoformat()},
                "tags": ["algorithmic_probe", platform, "experiment"], "relevance_score": 0.85 }}) # Higher relevance
        return {"status": "success", "message": f"Algorithmic probe completed for {platform}.", "results": probe_results}

    async def learning_loop(self):
        self.logger.info(f"{self.AGENT_NAME} L35+ learning loop: Advanced performance & algorithmic shift detection.")
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.internal_state.get('learning_interval_seconds', 3600 * 1))
                if self._stop_event.is_set(): break
                current_time = datetime.now(timezone.utc)
                self.internal_state['last_learning_run'] = current_time
                await self._internal_think("Learning loop: Analyzing multi-vector performance, detecting anomalies, querying ThinkTool for insights.")

                if not self.think_tool: self.logger.warning("ThinkTool not available for SMM learning."); continue

                # 1. Gather recent performance data (from internal cache and KB)
                recent_posts_summary = list(self.internal_state['recent_post_performance'])
                # Query KB for aggregated metrics if available (e.g., from a daily ThinkTool summary)
                aggregated_perf_frags = await self.think_tool.query_knowledge_base(
                    data_types=["social_platform_daily_summary"], time_window=timedelta(days=2), limit=10
                )
                
                # 2. Basic Anomaly Detection (Example: Engagement Drop)
                # This is a simplified version. Real anomaly detection is complex.
                anomalies_detected = []
                for platform, metrics in self.internal_state.get('performance_metrics_cache', {}).items():
                    current_eng_rate = metrics.get("engagement_rate", {}).get("value")
                    prev_eng_rate = metrics.get("engagement_rate", {}).get("previous_value") # Assume cache stores this
                    if current_eng_rate is not None and prev_eng_rate is not None and prev_eng_rate > 0:
                        if current_eng_rate < prev_eng_rate * 0.6: # 40% drop
                            anomalies_detected.append(f"Significant engagement drop on {platform} (from {prev_eng_rate:.2f} to {current_eng_rate:.2f}).")
                
                # 3. Formulate strategic questions for ThinkTool
                strategic_query_for_thinktool = {
                    "task_type": "analyze_smm_performance_and_adapt_strategy",
                    "recent_posts_sample": recent_posts_summary[-20:], # Last 20 posts
                    "aggregated_platform_perf": [json.loads(f.content) for f in aggregated_perf_frags if f.content],
                    "detected_anomalies": anomalies_detected,
                    "current_account_health_summary": {
                        p: {
                            "active": sum(1 for acc in accs if acc.get('status_db') == 'active' and acc.get('health_status') == 'good'),
                            "risky_or_watch": sum(1 for acc in accs if acc.get('health_status') in ['risky', 'watch']),
                            "total": len(accs)
                        } for p, accs in self.internal_state.get('managed_accounts', {}).items()
                    },
                    "request": "Analyze this SMM performance data. Identify key insights, underperforming areas, potential algorithmic shifts, or new opportunities. Suggest 2-3 high-impact strategic adjustments or experiments (e.g., new content angles, platform focus shift, probe directives)."
                }

                await self.think_tool.execute_task({
                    "action": "process_agent_learning_cycle_input", # New ThinkTool action
                    "content": {"agent_name": self.AGENT_NAME, "learning_data": strategic_query_for_thinktool}
                })
                self.logger.info("Sent detailed SMM performance data and strategic queries to ThinkTool.")

            except asyncio.CancelledError: self.logger.info(f"{self.AGENT_NAME} learning loop cancelled."); break
            except Exception as e:
                self.logger.error(f"Error in {self.AGENT_NAME} learning loop: {e}", exc_info=True)
                await self._report_error(f"Learning loop error: {e}")
                await asyncio.sleep(60 * 20)

    async def self_critique(self) -> Dict[str, Any]:
        self.logger.info(f"{self.AGENT_NAME} (L35+): Performing deep self-critique.")
        critique = {"status": "ok", "feedback": "Critique pending comprehensive analysis."}
        try:
            # Gather more detailed internal state and recent performance for critique
            insights = await self.collect_insights() # Gets basic operational stats
            
            # Get recent campaign outcomes from KB (via ThinkTool)
            campaign_outcomes_summary = "N/A"
            if self.think_tool:
                campaign_logs = await self.think_tool.query_knowledge_base(
                    data_types=["social_campaign_execution_log"], time_window=timedelta(days=14), limit=5
                )
                if campaign_logs:
                    campaign_outcomes_summary = [json.loads(log.content).get("execution_log", [])[-3:] for log in campaign_logs if log.content] # Last 3 steps of recent campaigns

            # Get summary of ThinkTool's directives for SMM
            smm_directives_summary = "N/A"
            if self.think_tool:
                directives = await self.think_tool.get_active_directives(target_agent=self.AGENT_NAME, limit=5) # Get pending/active
                smm_directives_summary = [{"type": d.directive_type, "content_preview": d.content[:100], "status": d.status} for d in directives]

            critique_context = {
                "task": "Deep Strategic Critique of SocialMediaManager Performance",
                "current_operational_insights": insights,
                "recent_campaign_outcomes_summary": campaign_outcomes_summary,
                "active_or_recent_directives_for_smm": smm_directives_summary,
                "meta_prompt_guidelines": self.meta_prompt[:1000], # Remind LLM of its core mandate
                "desired_output_format": "JSON: { \"overall_strategic_effectiveness_rating\": str ('Excellent'|'Good'|'Needs Improvement'|'Poor'), \"key_achievements_aligned_with_mandate\": [str], \"critical_weaknesses_or_deviations\": [str], \"algorithmic_adaptation_effectiveness\": str, \"grey_area_tactic_utilization_assessment\": str ('Effective & Calculated'|'Conservative'|'Needs More Aggression'|'Risky/Ineffective'), \"top_3_actionable_recommendations_for_thinktool\": [\"Directive for ThinkTool to issue to SMM or other agents to improve SMM effectiveness (e.g., 'Directive: SMM to test X new content format on TikTok for 7 days', 'Directive: ThinkTool to re-evaluate proxy strategy for Instagram accounts')\"] }"
            }
            prompt = await self.generate_dynamic_prompt(critique_context)
            llm_model_pref = self.config.get("OPENROUTER_MODELS", {}).get('think_critique') # Use a very strong model
            
            critique_json = await self._call_llm_with_retry(
                 prompt, model=llm_model_pref, temperature=0.3, max_tokens=1800, is_json_output=True
            )

            if critique_json:
                 try:
                     critique_result = self._parse_llm_json(critique_json)
                     if not critique_result: raise ValueError("Parsed critique is None")
                     critique['feedback'] = critique_result.get('overall_strategic_effectiveness_rating', 'Critique generated.')
                     critique['details'] = critique_result
                     self.logger.info(f"Self-Critique Assessment: {critique['feedback']}")
                     if self.think_tool:
                         await self.think_tool.execute_task({
                             "action": "log_knowledge_fragment", "fragment_data":{
                                 "agent_source": self.AGENT_NAME, "data_type": "self_critique_summary_L35",
                                 "content": critique_result, "tags": ["critique", "social_media", self.AGENT_NAME, "L35"], "relevance_score": 0.9 }})
                         # If critique suggests directives for ThinkTool, pass them on
                         if critique_result.get("top_3_actionable_recommendations_for_thinktool"):
                             await self.think_tool.execute_task({
                                 "action": "process_internal_agent_recommendations",
                                 "content": {
                                     "source_agent": self.AGENT_NAME,
                                     "recommendations": critique_result["top_3_actionable_recommendations_for_thinktool"]
                                 }
                             })
                 except Exception as e:
                      self.logger.error(f"Failed to parse L35+ self-critique LLM response: {e}")
                      critique['feedback'] += " Failed to parse LLM critique."
                      critique['status'] = 'error'
            else:
                 critique['feedback'] += " LLM critique call failed."
                 critique['status'] = 'error'
        except Exception as e:
            self.logger.error(f"Error during L35+ self-critique: {e}", exc_info=True)
            critique['status'] = 'error'; critique['feedback'] = f"Self-critique failed: {e}"
        self.internal_state['last_critique_run'] = time.time()
        return critique

    async def collect_insights(self) -> Dict[str, Any]:
        self.logger.debug("SocialMediaManager collect_insights called.")
        active_campaign_kf_id = self.internal_state.get('current_campaign_plan_id')
        managed_account_summary = {}
        for platform, acc_list in self.internal_state.get('managed_accounts', {}).items():
            managed_account_summary[platform] = {
                "total": len(acc_list),
                "active_db": sum(1 for acc in acc_list if acc.get('status_db') == 'active'),
                "health_good": sum(1 for acc in acc_list if acc.get('health_status') == 'good'),
                "health_risky_watch": sum(1 for acc in acc_list if acc.get('health_status') in ['risky', 'watch']),
            }
        
        # Simplified recent performance from deque
        recent_posts = list(self.internal_state['recent_post_performance'])
        recent_success_count = sum(1 for p in recent_posts if p.get('status') == 'success')
        recent_failure_count = len(recent_posts) - recent_success_count

        return {
            "agent_name": self.AGENT_NAME, "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_campaign_kf_id": active_campaign_kf_id,
            "managed_accounts_summary": managed_account_summary,
            "recent_posts_analyzed_count": len(recent_posts),
            "recent_posts_success_count": recent_success_count,
            "recent_posts_failure_count": recent_failure_count,
            "last_learning_run": datetime.fromtimestamp(self.internal_state['last_learning_run']).isoformat() if self.internal_state.get('last_learning_run') else None,
            "last_critique_run": datetime.fromtimestamp(self.internal_state['last_critique_run']).isoformat() if self.internal_state.get('last_critique_run') else None,
        }

    # --- Helper methods from v3.1 ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=3, max=20), retry=retry_if_exception_type(Exception))
    async def _call_llm_with_retry(self, prompt: str, model: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1500, is_json_output: bool = False) -> Optional[str]:
        if not hasattr(self.orchestrator, 'call_llm'):
             self.logger.error("Orchestrator does not have 'call_llm' method.")
             return None
        try:
            response_data = await self.orchestrator.call_llm(
                agent_name=self.AGENT_NAME, prompt=prompt, temperature=temperature,
                max_tokens=max_tokens, is_json_output=is_json_output,
                model_preference=[model] if model else None # Pass as list
            )
            content = response_data.get('content') if isinstance(response_data, dict) else str(response_data)
            return content.strip() if content and content.strip() else None
        except Exception as e:
            self.logger.warning(f"LLM call failed (attempt): {e}")
            raise

    def _get_platform_config(self, platform: str) -> Dict[str, Any]:
        platform_key = platform.lower().replace("https://", "").replace("www.", "")
        if platform_key in self.internal_state['platform_configs']:
            return self.internal_state['platform_configs'][platform_key]
        config = {
            "login_url": self.config.get(f"{platform_key.upper()}_LOGIN_URL", f"https://{platform_key}/login"),
            "post_url": self.config.get(f"{platform_key.upper()}_POST_URL", f"https://{platform_key}/"), # May not always be used if posting from feed
            "selectors": { # These are examples, specific selectors are crucial and vary wildly
                "username_field": self.config.get(f"{platform_key.upper()}_USERNAME_SELECTOR"),
                "password_field": self.config.get(f"{platform_key.upper()}_PASSWORD_SELECTOR"),
                "login_button": self.config.get(f"{platform_key.upper()}_LOGIN_BUTTON_SELECTOR"),
                "post_textarea": self.config.get(f"{platform_key.upper()}_POST_TEXTAREA_SELECTOR"),
                "submit_post_button": self.config.get(f"{platform_key.upper()}_SUBMIT_POST_BUTTON_SELECTOR"),
                "like_button_generic": self.config.get(f"{platform_key.upper()}_LIKE_BUTTON_SELECTOR"), # For generic like actions
                "comment_field_generic": self.config.get(f"{platform_key.upper()}_COMMENT_FIELD_SELECTOR"),
                "submit_comment_button_generic": self.config.get(f"{platform_key.upper()}_SUBMIT_COMMENT_SELECTOR"),
            }
        }
        config["selectors"] = {k: v for k, v in config["selectors"].items() if v is not None}
        self.internal_state['platform_configs'][platform_key] = config
        return config

    async def _get_proxy_for_action(self, purpose: str) -> Optional[Dict[str, Any]]:
         if self.orchestrator and hasattr(self.orchestrator, 'get_proxy'):
              try: return await self.orchestrator.get_proxy(purpose=purpose, quality_level='high') # Request high quality for social
              except Exception as e: self.logger.error(f"Failed to get proxy for purpose '{purpose}': {e}"); return None
         self.logger.warning("Orchestrator cannot provide proxies."); return None

    async def _get_account_details_from_db(self, account_id: int) -> Optional[Dict[str, Any]]:
        """ Helper to fetch account details (non-sensitive) if needed by a method. """
        if not self.session_maker: return None
        try:
            async with self.session_maker() as session:
                acc = await session.get(Account, account_id)
                if acc: return {"id": acc.id, "identifier": acc.account_identifier, "service": acc.service, "status_db": acc.status, "notes": acc.notes}
        except Exception as e: self.logger.error(f"Error fetching account details for ID {account_id}: {e}")
        return None

    # --- KB Interaction Helpers (Simplified, assuming ThinkTool handles complex KB logic) ---
    async def log_knowledge_fragment(self, agent_source: str, data_type: str, content: Union[str, dict], relevance_score: float = 0.5, tags: Optional[List[str]] = None, related_client_id: Optional[int] = None, source_reference: Optional[str] = None):
        if self.think_tool and hasattr(self.think_tool, 'log_knowledge_fragment'):
            await self.think_tool.execute_task({"action": "log_knowledge_fragment", "fragment_data": locals()}) # Pass all params
        else: self.logger.error("ThinkTool unavailable for logging KB fragment.")

    async def query_knowledge_base(self, data_types: Optional[List[str]] = None, tags: Optional[List[str]] = None, min_relevance: float = 0.0, time_window: Optional[timedelta] = None, limit: int = 10, related_client_id: Optional[int] = None, content_query: Optional[str] = None) -> List[Any]:
        if self.think_tool and hasattr(self.think_tool, 'query_knowledge_base'):
            # This is a simplified call; ThinkTool's query_knowledge_base is more direct
            # For SMM to query, it would likely be a directive to ThinkTool
            # For now, this direct call is a placeholder for a more complex interaction pattern
            return await self.think_tool.query_knowledge_base(data_types=data_types, tags=tags, min_relevance=min_relevance, time_window=time_window, limit=limit, related_client_id=related_client_id, content_query=content_query)
        self.logger.error("ThinkTool unavailable for querying KB."); return []

    async def log_learned_pattern(self, pattern_description: str, supporting_fragment_ids: List[int], confidence_score: float, implications: str, tags: Optional[List[str]] = None):
         if self.think_tool and hasattr(self.think_tool, 'log_learned_pattern'):
             await self.think_tool.execute_task({"action": "log_learned_pattern", "pattern_data": locals()})
         else: self.logger.error("ThinkTool unavailable for logging learned pattern.")

    # --- Abstract method implementations (from GeniusAgentBase) ---
    async def plan_task(self, task_details: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        # SMM primarily executes plans from ThinkTool or handles single-action tasks.
        # Complex planning is delegated.
        action = task_details.get('action')
        if action == 'post_content':
            return [{"step": 1, "action": "Execute Post Content", "tool": "internal_smm", "params": task_details}]
        elif action == 'strategic_interaction':
            return [{"step": 1, "action": "Execute Strategic Interaction", "tool": "internal_smm", "params": task_details}]
        self.logger.debug(f"No specific plan generated by SMM for action '{action}'. Relying on execute_task logic.")
        return None # Let execute_task handle it directly or delegate further.

    async def execute_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        # This method is less used if execute_task directly handles actions or delegates.
        # If plan_task returns steps, this would execute them.
        self.logger.warning(f"SMM execute_step called with: {step}. Most SMM actions are handled by execute_task directly.")
        if step.get("tool") == "internal_smm":
            return await self.execute_task(step.get("params", {})) # Route back to main handler
        return {"status": "failure", "message": "SMM execute_step does not handle this step type."}

# --- End of agents/social_media_manager.py ---