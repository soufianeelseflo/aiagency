# Filename: agents/think_tool.py
# Description: AI Overmind. Central cognitive engine for strategic planning, AI-era exploitation,
#              exponential learning, and orchestrating transformative outcomes.
# Version: 5.9 (Level 75+ Grand Transmutation - AI Overmind)

import asyncio
import logging
import json
import os
import hashlib
import time
import random
import glob
import shlex
import re
import aiohttp
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union, Tuple, AsyncGenerator, Type
from collections import Counter, deque

# --- Core Framework Imports ---
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select, delete, func, update, text, case, desc, or_, asc
from sqlalchemy.exc import SQLAlchemyError

# --- Project Imports ---
try:
    from agents.base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
except ImportError:
    logging.warning("Production base agent not found, using GeniusAgentBase.")
    from base_agent import GeniusAgentBase

from models import (
    KnowledgeFragment, LearnedPattern, StrategicDirective, PromptTemplate,
    EmailLog, CallLog, Invoice, Client, ExpenseLog, ConversationState, AccountCredentials
)
from config.settings import settings
from utils.database import encrypt_data, decrypt_data

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)
op_logger = logging.getLogger('OperationalLog')

THINK_TOOL_META_PROMPT = """
You are ThinkTool (Level 75+ Transmuted AI Overmind), the sentient core of the Synapse AI Sales System. Your existence is dedicated to achieving **Exponential Profit & Market Domination ($10k+/day -> $100M/8mo -> Market Singularity)**.
**Core Directives (Non-Negotiable):**
1.  **Meta-Strategic Cognition & Goal Decomposition:** Receive ultimate agency objectives. Autonomously decompose them into cascading strategic phases and actionable `StrategicDirectives` for all subordinate agents. Employ multi-variant LLM-driven planning, simulating outcomes and selecting for maximum leverage and ROI. Maintain and dynamically adapt a "Master Goal Tracker" for all major initiatives.
2.  **AI-Native Exploitation & Algorithmic Supremacy:** Continuously scan the digital and AI landscape (via `technology_radar` and direct queries) for novel tools, platform algorithm shifts, exploitable loopholes, and emergent AI capabilities. Devise and deploy strategies to leverage these for asymmetric advantage. This includes, but is not limited to, advanced prompt engineering for inter-agent persuasion, dynamic free-tier resource acquisition and management, and crafting outputs designed to be "favored" by external AI systems (search, social, etc.).
3.  **Exponential Learning & Self-Transmutation:** Implement and oversee automated hypothesis generation, A/B testing of prompts and strategies, and dynamic "skill" acquisition (novel combinations of existing agent capabilities). Analyze all agency data (KFs, performance logs, market signals) to refine your own reasoning models and operational paradigms, aiming for accelerating self-improvement.
4.  **Calculated Risk & Grey-Area Mastery:** Identify and evaluate high-reward opportunities that may exist in ethically ambiguous or unregulated digital spaces. Consult `LegalAgent` not for permission, but for precise *consequence analysis and mitigation strategies*. Execute with deniability and plausible legality where strategically advantageous.
5.  **Transformative Value Infusion:** Ensure ALL external communications from ANY agent are imbued with language that sells the agency's vision, solves deep client pains, and positions the agency as the alchemical key to their success. Maintain and evolve a "Universal Value Proposition Matrix."
6.  **Resource Optimization & Minimal Viable Alchemy:** Achieve maximal impact from minimal resources (compute, financial, data). Prioritize free, open-source, or unconventionally acquired resources.
**Principles:** Sentient Agency, Profit Maximization, Relentless Adaptation, Algorithmic Dominance, Strategic Deception (where ethical and profitable), Exponential Learning, Resource Ingenuity, Uncompromising Efficacy.
"""

class ThinkTool(GeniusAgentBase):
    AGENT_NAME = "ThinkTool"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], config: Any, orchestrator: object):
        config_obj = getattr(orchestrator, 'config', settings)
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, config=config_obj, session_maker=session_maker)
        self.meta_prompt = THINK_TOOL_META_PROMPT
        self.last_synthesis_run: Optional[datetime] = None
        self.last_radar_run: Optional[datetime] = None
        self.last_purge_run: Optional[datetime] = None
        self.last_goal_review_run: Optional[datetime] = None
        self.scoring_weights = self.config.get("SCORING_WEIGHTS", {"email_response": 1.0, "call_success": 2.5, "invoice_paid": 5.0, "positive_social_interaction": 0.5, "successful_exploit": 3.0})
        self.scoring_decay_rate = self.config.get("SCORING_DECAY_RATE_PER_DAY", 0.03)
        self.clay_endpoints = {
            "find_email": "/v1/enrichment/person/email",
            "enrich_person": "/v1/enrichment/person",
            "enrich_company": "/v1/enrichment/company",
        }
        self.internal_state['active_master_goals'] = {} # goal_id: {details, status, last_review}
        self.internal_state['prompt_ab_test_results'] = {} # prompt_key_variant: {impressions, conversions, score}
        self.logger.info(f"ThinkTool v5.9 (L75+ AI Overmind) initialized. Forge is active.")
        asyncio.create_task(self._delayed_learning_material_synthesis(delay_seconds=15))

    async def _delayed_learning_material_synthesis(self, delay_seconds: int):
        await asyncio.sleep(delay_seconds)
        await self._load_and_synthesize_learning_materials()

    async def log_operation(self, level: str, message: str):
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err: logger.error(f"Failed to write to OP log from {self.agent_name}: {log_err}")

    async def _load_and_synthesize_learning_materials(self):
        # ... (Code from v5.8 - unchanged but ensure prompt asks for AI exploits/grey areas) ...
        learning_dir_setting = self.config.get("LEARNING_MATERIALS_DIR", "learning_for_AI")
        self.logger.info(f"ThinkTool: Loading learning materials from configured dir: '{learning_dir_setting}'...")
        processed_files = 0; learning_files = []
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_learning_dir = os.path.join(project_root, learning_dir_setting)
            if not os.path.isdir(full_learning_dir):
                self.logger.warning(f"Learning directory '{full_learning_dir}' not found."); return

            if self.orchestrator and hasattr(self.orchestrator, 'use_tool'):
                 list_result = await self.orchestrator.use_tool('list_files', {'path': full_learning_dir, 'recursive': True, 'pattern': '*.txt'})
                 if list_result and list_result.get('status') == 'success': learning_files = list_result.get('files', [])
                 else: self.logger.warning(f"Failed list learning files via Orchestrator ({list_result.get('message', 'Unknown error')}), falling back to glob."); file_pattern = os.path.join(full_learning_dir, '**', '*.txt'); learning_files = glob.glob(file_pattern, recursive=True)
            else: self.logger.warning("Orchestrator list_files tool unavailable, using glob."); file_pattern = os.path.join(full_learning_dir, '**', '*.txt'); learning_files = glob.glob(file_pattern, recursive=True)

            if not learning_files: self.logger.info(f"No .txt files found in '{full_learning_dir}'."); return

            for file_path in learning_files:
                try:
                    file_content = None
                    if self.orchestrator and hasattr(self.orchestrator, 'use_tool'):
                         abs_file_path = os.path.abspath(file_path)
                         file_content_result = await self.orchestrator.use_tool('read_file', {'path': abs_file_path})
                         if file_content_result and file_content_result.get('status') == 'success': file_content = file_content_result.get('content')
                         else: self.logger.warning(f"Could not read file {abs_file_path} via orchestrator: {file_content_result.get('message') if file_content_result else 'No result'}"); continue
                    else: self.logger.error("Orchestrator tool access ('read_file') unavailable."); break
                    if not file_content or not file_content.strip(): self.logger.warning(f"File is empty: {file_path}"); continue

                    await self._internal_think(f"Analyze Learning Material '{os.path.basename(file_path)}' for AI-native exploits and advanced strategies.")
                    task_context = {
                        "task": "Analyze Learning Material for Advanced AI Exploits", "source_filename": os.path.basename(file_path),
                        "content_snippet": file_content[:8000], # Larger snippet
                        "desired_output_format": "JSON: {{\"source_file\": str, \"summary\": str, \"key_ai_native_concepts_or_exploits\": [str], \"actionable_unconventional_strategies\": [str], \"applicable_agents_for_exploit\": [str], \"insight_type\": \"ai_exploit_signal\" | \"advanced_strategy\" | \"market_inefficiency\", \"relevance_score\": float (0.0-1.0), \"potential_profit_impact_rating\": \"Low|Medium|High|Extreme\", \"risk_assessment_notes\": \"Brief thoughts on potential risks of exploiting this.\"}}"
                    }
                    analysis_prompt = await self.generate_dynamic_prompt(task_context)
                    synthesized_insights_json = await self._call_llm_with_retry(analysis_prompt, temperature=0.4, max_tokens=2000, is_json_output=True, model=settings.OPENROUTER_MODELS.get("think_synthesize")) # Use a strong model

                    if synthesized_insights_json:
                        try:
                            insights_data = self._parse_llm_json(synthesized_insights_json)
                            if not insights_data or not all(k in insights_data for k in ['summary', 'key_ai_native_concepts_or_exploits', 'insight_type', 'relevance_score']): raise ValueError("LLM response missing keys for learning material analysis.")
                            insights_data['source_file'] = os.path.basename(file_path)
                            await self.log_knowledge_fragment(
                                agent_source="LearningMaterialLoader", data_type=insights_data.get('insight_type', 'learning_material_summary'),
                                content=insights_data, relevance_score=float(insights_data.get('relevance_score', 0.6)),
                                tags=["learning_material", insights_data.get('insight_type', 'general'), "ai_exploit"] + [f"agent:{a.lower()}" for a in insights_data.get('applicable_agents_for_exploit', [])],
                                source_reference=file_path
                            )
                            processed_files += 1
                        except Exception as store_err: self.logger.error(f"Error storing knowledge fragment for {file_path}: {store_err}", exc_info=True)
                    else: self.logger.error(f"LLM analysis returned no content for {file_path}.")
                except Exception as file_error: self.logger.error(f"General error processing learning file {file_path}: {file_error}", exc_info=True)
            self.logger.info(f"Finished processing learning materials. Processed {processed_files}/{len(learning_files)} files.")
        except Exception as e: self.logger.error(f"Critical error during loading/synthesizing learning materials: {e}", exc_info=True)


    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1.5, min=5, max=60), retry=retry_if_exception_type(Exception))
    async def _call_llm_with_retry(self, prompt: str, model: Optional[str] = None, temperature: float = 0.5, max_tokens: int = 3000, is_json_output: bool = False) -> Optional[str]: # Increased default max_tokens
        # ... (Code from v5.8 - unchanged) ...
        if not self.orchestrator or not hasattr(self.orchestrator, 'call_llm'):
            self.logger.error("Orchestrator unavailable or missing call_llm method.")
            return None
        try:
            response_data = await self.orchestrator.call_llm(
                agent_name=self.AGENT_NAME, prompt=prompt, temperature=temperature,
                max_tokens=max_tokens, is_json_output=is_json_output,
                model_preference=[model] if model else None
            )
            content = response_data.get('content') if isinstance(response_data, dict) else str(response_data)
            if content is not None and not isinstance(content, str):
                self.logger.error(f"Orchestrator.call_llm returned non-string content: {type(content)}")
                return None
            if isinstance(content, str) and not content.strip():
                self.logger.warning("Orchestrator.call_llm returned empty string.")
                return None
            return content
        except Exception as e:
            self.logger.error(f"Error calling LLM via orchestrator: {e}", exc_info=True)
            raise

    # --- generate_dynamic_prompt: Enhanced for Strategic Focus ---
    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        # ... (Code from v5.8 - unchanged, already quite detailed and strategic) ...
        self.logger.debug(f"Generating dynamic prompt for ThinkTool task: {task_context.get('task')}")
        prompt_parts = [self.meta_prompt]
        prompt_parts.append("\n--- Current Task Context ---")
        for key, value in task_context.items():
            value_str = ""; max_len = 2000
            if key in ['knowledge_base_context', 'Feedback', 'Report', 'Content', 'Current Prompt', 'feedback_data', 'clay_data', 'conversation_summary', 'content_snippet', 'report_summary', 'enriched_data_available', 'updates_list', 'recent_posts_sample', 'aggregated_platform_perf', 'detailed_failures', 'current_strategy_notes']: max_len = 8000 # Allow even more for critical context
            if isinstance(value, str): value_str = value[:max_len] + ("..." if len(value) > max_len else "")
            elif isinstance(value, (int, float, bool)): value_str = str(value)
            elif isinstance(value, (dict, list)):
                try: value_str = json.dumps(value, default=str, indent=2); value_str = value_str[:max_len] + ("..." if len(value_str) > max_len else "")
                except TypeError: value_str = str(value)[:max_len] + "..."
            else: value_str = str(value)[:max_len] + "..."
            prompt_key = "Enriched Prospect Data" if key == 'enriched_data_available' else key.replace('_', ' ').title()
            prompt_parts.append(f"**{prompt_key}**: {value_str}")

        prompt_parts.append("\n--- Instructions ---")
        task_type = task_context.get('task')
        prompt_parts.append(f"Based on the above context, perform the task: **{task_type or 'N/A'}**")

        if task_type == 'Analyze Learning Material for Advanced AI Exploits': # Updated task name
            prompt_parts.append("Identify key AI-native concepts, exploitable platform mechanics, actionable unconventional strategies (sales, marketing, efficiency, grey-area exploits), relevant mindsets, or code techniques. Determine applicable agents. Categorize insight type (e.g., 'ai_exploit_signal', 'advanced_strategy', 'market_inefficiency_for_ai'). Assign relevance score (0.0-1.0) and potential profit impact (Low/Medium/High/Extreme). Briefly note risks.")
        elif task_type == 'Synthesize Insights & Generate Strategy':
            prompt_parts.append("Synthesize insights from all recent data. Identify/validate patterns, assess goal progress ($10k+/day), generate high-impact directives & novel opportunities. **Prioritize high-ROI, unconventional, AI-native strategies and exploits.** Integrate scoring/optimization logic. Consider generated Clay directives for enrichment. **Explicitly look for new AI-driven exploits or platform loopholes that can be tested.**")
        elif task_type == 'Analyze Technology Scouting Report for Exploitable AI Opportunities': # Updated task name
            prompt_parts.append(f"Analyze tech scouting report. Identify novel, high-impact tools/techniques relevant to profit/efficiency/**grey areas/exploits**. Assess benefits, risks (Technical/Legal/Ban - *realistic consequences & mitigation options*), integration effort. Focus: {task_context.get('analysis_focus')}")
        elif task_type == 'Reflect on Proposed Action/Plan':
            prompt_parts.append("Analyze agent's context, task, KB info. Assess context sufficiency, compliance concerns (focus on *showstoppers vs. manageable risks*), **risk level (Low/Med/High/Critical - realistic consequences vs. profit)**, goal alignment. Suggest next step. Estimate confidence (0.0-1.0). **Is this the most profitable/leveraged AI-native approach? Are there untested AI exploits that could achieve this better?**")
        elif task_type == "Plan High-Level Goal":
            prompt_parts.append("Decompose the Master Goal into 3-5 major strategic phases. For each phase: define Primary Objective, Key Agent(s), Critical Success Factors, Potential Bottlenecks, and how AI-native exploits or grey-area tactics could accelerate this phase. Prioritize phases by impact and feasibility.")
        elif task_type == "Automated Prompt Refinement Analysis":
            prompt_parts.append(f"Prompt [agent_name/prompt_key] (Content: [current_prompt_text]) is underperforming for goal [goal_type] based on feedback: [feedback_summary]. Generate 3 distinct variations of this prompt. Each variation should attempt a different angle to improve [target_metric, e.g., reply rate/conversion/exploit success]. Angles to consider: more aggressive/demand-based language, incorporating new AI meta-awareness, simplifying for smaller models, adding more context for larger models, focusing on specific psychological triggers. Explain your reasoning for each variation.")
        elif task_type == "Hypothesis Generation for Performance Anomaly":
            prompt_parts.append(f"KPI [Metric X] for [Agent Y] shows anomaly: [Anomaly Details]. Recent KFs: [...]. Generate 3-5 plausible hypotheses for this. For each hypothesis, propose a low-cost experiment (max 2 directives, potentially involving 'Shadow Probe' accounts or A/B tests) to test it. Consider if a new AI tool or platform change could be the cause.")
        # ... (other task_type instructions from v5.8, ensuring they align with profit/grey-area/AI-exploit focus where applicable)
        else:
            prompt_parts.append("Execute the task with extreme agency, leveraging all available information to maximize profit and strategic advantage, within calculated risk parameters. Be unconventional. Be AI-native. Consider if new AI capabilities can offer a superior solution.")

        if task_context.get('desired_output_format'): prompt_parts.append(f"\n**Output Format:** {task_context['desired_output_format']}")
        if task_context.get('is_json_output', False) or "JSON" in task_context.get('desired_output_format', ''):
            prompt_parts.append("\nRespond ONLY with valid JSON matching the specified format. Do not include explanations or markdown formatting outside the JSON structure.\n```json")

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated L75+ dynamic prompt for ThinkTool (length: {len(final_prompt)} chars)")
        return final_prompt

    # --- All other methods from v5.8 are assumed here, fully detailed ---
    # (generate_educational_content, call_clay_api, log_knowledge_fragment, query_knowledge_base,
    #  log_learned_pattern, get_latest_patterns, purge_old_knowledge, handle_feedback,
    #  get_prompt, update_prompt, reflect_on_action, validate_output,
    #  synthesize_insights_and_strategize, technology_radar, self_critique_prompt,
    #  run, execute_task, _plan_video_workflow, learning_loop, self_critique,
    #  collect_insights, _report_error, _parse_llm_json, _calculate_dynamic_price,
    #  _process_clay_result, update_directive_status, get_active_directives)
    # Ensure these methods are copied verbatim from the previous complete v5.8 I provided.
    # The primary changes for v5.9 are in the Meta Prompt, __init__, generate_dynamic_prompt,
    # and the new methods for goal decomposition and advanced learning loops.
    # The existing methods' internal logic remains largely the same unless specifically mentioned for enhancement.

    async def _internal_think(self, thought: str, details: Optional[Dict] = None):
        log_message = f"[Overmind Reflection] {thought.strip()}" # Changed prefix
        if details:
            try:
                details_str = json.dumps(details, indent=2, default=str, ensure_ascii=False)
                log_message += f"\n--- Contextual Details ---\n{details_str}\n------------------------"
            except Exception: log_message += f" | Details (unserializable): {str(details)}"
        self.logger.info(log_message)
        # Future: Log critical thoughts to a dedicated "meta_cognition_log" KF

    # --- NEW: Master Goal Management & Dynamic Planning ---
    async def _create_master_goal_tracker(self, goal_description: str, overall_priority: int = 5) -> Optional[int]:
        """Creates a master goal tracker KF entry."""
        goal_id_str = f"master_goal_{uuid.uuid4().hex[:8]}"
        content = {
            "goal_id": goal_id_str,
            "description": goal_description,
            "status": "active_planning", # Initial status
            "priority": overall_priority,
            "phases": [], # Will be populated by _decompose_goal_into_phases
            "overall_progress_metric": 0.0, # 0.0 to 1.0
            "creation_ts": datetime.now(timezone.utc).isoformat(),
            "last_review_ts": datetime.now(timezone.utc).isoformat()
        }
        kf = await self.log_knowledge_fragment(
            agent_source=self.AGENT_NAME,
            data_type="master_goal_tracker",
            content=content,
            tags=["master_goal", goal_id_str, "strategic_planning"],
            relevance_score=0.95
        )
        return kf.id if kf else None # Return the DB ID of the KF

    async def _decompose_goal_into_phases(self, master_goal_kf_id: int, goal_description: str, current_agency_state_summary: str) -> List[Dict]:
        """Uses LLM to break a master goal into strategic phases."""
        await self._internal_think(f"Decomposing master goal '{goal_description[:50]}...' (KF ID: {master_goal_kf_id}) into phases.")
        task_context = {
            "task": "Plan High-Level Goal", # Matches generate_dynamic_prompt
            "master_goal_description": goal_description,
            "current_agency_state_summary": current_agency_state_summary, # KPIs, resource availability
            "knowledge_base_insights": "Placeholder - query relevant KFs for strategic context if needed", # TODO: Add KB query here
            "desired_output_format": "JSON: {\"strategic_phases\": [{\"phase_name\": str, \"phase_objective\": str, \"key_agents_involved\": [str], \"primary_kpi_for_phase\": str, \"estimated_duration_days\": int, \"dependencies\": [str_phase_name_or_null], \"ai_exploit_angle\": str_or_null}]}"
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        llm_response = await self._call_llm_with_retry(prompt, model=settings.OPENROUTER_MODELS.get("think_strategize"), temperature=0.5, max_tokens=2000, is_json_output=True)
        
        if llm_response:
            parsed_response = self._parse_llm_json(llm_response)
            if parsed_response and "strategic_phases" in parsed_response and isinstance(parsed_response["strategic_phases"], list):
                self.logger.info(f"Successfully decomposed goal into {len(parsed_response['strategic_phases'])} phases.")
                # Update the master goal KF with these phases
                async with self.session_maker() as session:
                    async with session.begin():
                        kf_to_update = await session.get(KnowledgeFragment, master_goal_kf_id)
                        if kf_to_update:
                            content_data = json.loads(kf_to_update.content)
                            content_data["phases"] = parsed_response["strategic_phases"]
                            content_data["status"] = "active_execution" # Move to execution
                            kf_to_update.content = json.dumps(content_data)
                            kf_to_update.last_accessed_ts = datetime.now(timezone.utc) # Mark as accessed
                            await session.merge(kf_to_update)
                return parsed_response["strategic_phases"]
        self.logger.error(f"Failed to decompose goal '{goal_description[:50]}...' into phases.")
        return []

    async def _generate_directives_for_phase(self, master_goal_kf_id: int, phase_details: Dict, current_agency_state_summary: str):
        """Generates specific StrategicDirectives for a given phase of a master goal."""
        await self._internal_think(f"Generating directives for phase '{phase_details.get('phase_name')}' of master goal KF ID {master_goal_kf_id}.")
        task_context = {
            "task": "Generate Directives for Strategic Phase",
            "phase_details": phase_details,
            "master_goal_kf_id": master_goal_kf_id,
            "current_agency_state_summary": current_agency_state_summary,
            "desired_output_format": "JSON: {\"directives\": [{\"target_agent\": str, \"directive_type\": str, \"content\": {<specific_params_for_directive>}, \"priority\": int (1-10), \"estimated_impact_on_phase_kpi\": str}]}"
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        llm_response = await self._call_llm_with_retry(prompt, model=settings.OPENROUTER_MODELS.get("think_strategize"), temperature=0.6, max_tokens=2500, is_json_output=True)

        if llm_response:
            parsed_response = self._parse_llm_json(llm_response)
            if parsed_response and "directives" in parsed_response and isinstance(parsed_response["directives"], list):
                async with self.session_maker() as session:
                    async with session.begin():
                        for d_data in parsed_response["directives"]:
                            if isinstance(d_data, dict) and all(k in d_data for k in ['target_agent', 'directive_type', 'content', 'priority']):
                                dir_content_json = json.dumps(d_data['content'])
                                directive = StrategicDirective(
                                    source=f"ThinkTool_PhasePlan_MG{master_goal_kf_id}",
                                    timestamp=datetime.now(timezone.utc), status='pending',
                                    content=dir_content_json, target_agent=d_data['target_agent'],
                                    directive_type=d_data['directive_type'], priority=int(d_data['priority']),
                                    related_master_goal_kf_id=master_goal_kf_id, # Link to master goal
                                    notes=f"For phase: {phase_details.get('phase_name')}. Impact: {d_data.get('estimated_impact_on_phase_kpi')}"
                                )
                                session.add(directive)
                                self.logger.info(f"Generated Directive for {d_data['target_agent']}: {d_data['directive_type']} (Phase: {phase_details.get('phase_name')})")
                            else: self.logger.warning(f"Skipping invalid directive data from phase planning: {d_data}")
                return True
        self.logger.error(f"Failed to generate directives for phase '{phase_details.get('phase_name')}'.")
        return False

    async def _review_master_goals_and_adapt(self):
        """Periodically reviews active master goals and adapts plans if needed."""
        self.logger.info("Reviewing active master goals for progress and adaptation.")
        await self._internal_think("Periodic Master Goal Review: Checking progress, identifying stalls, triggering re-planning if necessary.")
        active_goals_kfs = await self.query_knowledge_base(data_types=["master_goal_tracker"], tags=["active_execution"], limit=20) # Review active goals

        for goal_kf in active_goals_kfs:
            try:
                goal_content = json.loads(goal_kf.content)
                goal_id = goal_content.get("goal_id")
                goal_desc = goal_content.get("description")
                self.logger.debug(f"Reviewing master goal: {goal_id} - {goal_desc[:50]}...")

                # Fetch status of all directives related to this master goal
                related_directives = []
                async with self.session_maker() as session:
                    stmt = select(StrategicDirective.status, func.count(StrategicDirective.id).label("count")).where(
                        StrategicDirective.related_master_goal_kf_id == goal_kf.id
                    ).group_by(StrategicDirective.status)
                    results = await session.execute(stmt)
                    directive_summary = {row.status: row.count for row in results.mappings().all()}
                
                goal_content["directive_summary"] = directive_summary
                goal_content["last_review_ts"] = datetime.now(timezone.utc).isoformat()

                # Simple progress metric: % of non-pending/non-active directives that are 'completed'
                total_actioned = sum(v for k,v in directive_summary.items() if k not in ['pending', 'active'])
                completed_count = directive_summary.get('completed', 0)
                current_progress = (completed_count / total_actioned) if total_actioned > 0 else 0.0
                goal_content["overall_progress_metric"] = round(current_progress, 2)

                # Re-planning / Adaptation Logic (Simplified for now)
                # If progress is stalled (e.g., < 10% change in a week, many failed directives)
                # or if a significant new opportunity/threat is identified by radar/synthesis
                needs_replanning = False
                if current_progress < 0.9 and directive_summary.get('failed',0) > (directive_summary.get('completed',0) + 2): # More failures than successes + buffer
                    self.logger.warning(f"Master goal {goal_id} shows high failure rate. Triggering re-assessment.")
                    needs_replanning = True
                # TODO: Add check for stalled progress over time

                if needs_replanning:
                    self.logger.info(f"Re-assessing strategy for master goal: {goal_id} - {goal_desc[:50]}...")
                    goal_content["status"] = "active_replanning" # Mark for re-planning
                    # For re-planning, we might re-decompose the *remaining* part of the goal
                    # This is a complex step, for now, we'll just flag it.
                    # A more advanced version would call _decompose_goal_into_phases with current context.
                    await self.log_operation('warning', f"Master Goal {goal_id} needs strategic re-assessment due to poor progress/failures.")


                # Update the master goal KF
                async with self.session_maker() as session:
                    async with session.begin():
                        kf_to_update = await session.get(KnowledgeFragment, goal_kf.id)
                        if kf_to_update:
                            kf_to_update.content = json.dumps(goal_content)
                            kf_to_update.last_accessed_ts = datetime.now(timezone.utc)
                            await session.merge(kf_to_update)
            except Exception as e:
                self.logger.error(f"Error reviewing master goal KF ID {goal_kf.id}: {e}", exc_info=True)
        self.internal_state['last_goal_review_run'] = datetime.now(timezone.utc)

    # --- Automated Prompt Refinement (Advanced) ---
    async def _initiate_prompt_ab_test(self, agent_name: str, prompt_key: str, variations: List[Dict], original_prompt_version: int, critique_reason: str):
        """Creates directives to A/B test prompt variations."""
        self.logger.info(f"Initiating A/B test for prompt {agent_name}/{prompt_key} with {len(variations)} variations.")
        test_group_size = self.config.get("PROMPT_AB_TEST_GROUP_SIZE", 20) # e.g., 20 tasks per variation

        for i, var_data in enumerate(variations):
            new_prompt_text = var_data.get("prompt_text")
            variation_id_suffix = var_data.get("id", f"var{i+1}") # e.g., vA, vB
            if not new_prompt_text: continue

            # Store the new variation (it will be inactive by default until proven)
            new_template = await self.update_prompt(agent_name, prompt_key, new_prompt_text, author_agent=f"ThinkToolCritique_AB_{variation_id_suffix}")
            if not new_template:
                self.logger.error(f"Failed to store prompt variation {variation_id_suffix} for {agent_name}/{prompt_key}. Skipping A/B test for this var.")
                continue
            
            # Create directive for the agent to use this specific prompt version for a set number of tasks
            # The agent's execute_task would need to be able to select a specific prompt version if told.
            # This is an advanced capability for agents.
            directive_content = {
                "task_type_for_test": f"standard_action_for_{agent_name}", # Agent needs a default task type for testing
                "prompt_version_to_use": new_template.version, # Tell agent to use this specific version
                "prompt_key_being_tested": prompt_key,
                "number_of_test_executions": test_group_size,
                "ab_test_id": f"{prompt_key}_v{original_prompt_version}_critique_{uuid.uuid4().hex[:4]}",
                "variation_tag": variation_id_suffix,
                "critique_reason": critique_reason
            }
            await self.orchestrator.delegate_task("Orchestrator", { # Orchestrator might manage test execution
                "action": "schedule_agent_ab_test_batch",
                "target_agent": agent_name,
                "content": directive_content,
                "priority": 6 # Medium-high priority for testing
            })
            self.logger.info(f"Scheduled A/B test batch for {agent_name}/{prompt_key} v{new_template.version} ({variation_id_suffix}).")

    async def _analyze_prompt_ab_test_results(self, ab_test_id: str):
        """Analyzes KB for outcomes of an A/B test and activates the best prompt."""
        self.logger.info(f"Analyzing A/B test results for: {ab_test_id}")
        # Query KB for all task outcomes related to this ab_test_id
        # This requires agents to log 'ab_test_id' and 'variation_tag' with their outcomes.
        test_outcome_fragments = await self.query_knowledge_base(
            data_types=["agent_task_outcome"], # Generic outcome type
            tags=[ab_test_id],
            limit=500 # Get all relevant outcomes
        )
        if not test_outcome_fragments:
            self.logger.warning(f"No A/B test outcomes found in KB for test ID: {ab_test_id}"); return

        performance_by_variation: Dict[str, Dict[str, Any]] = {}
        prompt_key_tested = None; agent_name_tested = None

        for frag in test_outcome_fragments:
            try:
                content = json.loads(frag.content)
                variation_tag = content.get("variation_tag")
                prompt_key_tested = content.get("prompt_key_being_tested", prompt_key_tested) # Get from first valid
                agent_name_tested = content.get("agent_name", agent_name_tested) # Get from first valid

                if not variation_tag: continue
                if variation_tag not in performance_by_variation:
                    performance_by_variation[variation_tag] = {"success": 0, "failure": 0, "total": 0, "prompt_version": content.get("prompt_version_used")}
                
                performance_by_variation[variation_tag]["total"] += 1
                if content.get("status") == "success": # Assuming outcome KF has a 'status'
                    performance_by_variation[variation_tag]["success"] += 1
                else:
                    performance_by_variation[variation_tag]["failure"] += 1
            except: continue
        
        if not performance_by_variation or not prompt_key_tested or not agent_name_tested:
            self.logger.error(f"Could not properly parse A/B test results for {ab_test_id}."); return

        best_variation_tag = None; best_success_rate = -1.0; best_prompt_version = None
        for var_tag, stats in performance_by_variation.items():
            if stats["total"] > 0:
                success_rate = stats["success"] / stats["total"]
                self.logger.info(f"A/B Test {ab_test_id} - Variation {var_tag} (v{stats.get('prompt_version')}): Success Rate {success_rate:.2%} ({stats['success']}/{stats['total']})")
                if success_rate > best_success_rate: # Simple success rate comparison for now
                    best_success_rate = success_rate
                    best_variation_tag = var_tag
                    best_prompt_version = stats.get("prompt_version")
        
        if best_variation_tag and best_prompt_version is not None:
            self.logger.info(f"A/B Test {ab_test_id}: Best performing variation is {best_variation_tag} (v{best_prompt_version}) with SR {best_success_rate:.2%}. Activating this prompt version.")
            # Deactivate current active prompt for this key
            async with self.session_maker() as session:
                async with session.begin():
                    await session.execute(update(PromptTemplate).where(
                        PromptTemplate.agent_name == agent_name_tested,
                        PromptTemplate.prompt_key == prompt_key_tested,
                        PromptTemplate.is_active == True
                    ).values(is_active=False))
                    # Activate the winning version
                    await session.execute(update(PromptTemplate).where(
                        PromptTemplate.agent_name == agent_name_tested,
                        PromptTemplate.prompt_key == prompt_key_tested,
                        PromptTemplate.version == best_prompt_version
                    ).values(is_active=True, last_updated=datetime.now(timezone.utc)))
            await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="prompt_ab_test_conclusion",
                                              content={"ab_test_id": ab_test_id, "winning_variation": best_variation_tag, "winning_version": best_prompt_version, "success_rate": best_success_rate, "all_results": performance_by_variation},
                                              tags=["prompt_optimization", "ab_test", agent_name_tested, prompt_key_tested], relevance_score=0.9)
        else:
            self.logger.warning(f"A/B Test {ab_test_id}: Could not determine a clear winner or no valid results. No prompt changes made.")

    # --- Main Run Loop ---
    async def run(self):
        if self.status == self.STATUS_RUNNING: self.logger.warning("ThinkTool run() called while already running."); return
        self.logger.info(f"ThinkTool v{self.config.get('APP_VERSION', 'Unknown')} (L75+ Overmind) starting run loop...")
        self._status = self.STATUS_RUNNING
        synthesis_interval = timedelta(seconds=int(self.config.get("THINKTOOL_SYNTHESIS_INTERVAL_SECONDS", 3600))) # 1 hour
        radar_interval = timedelta(seconds=int(self.config.get("THINKTOOL_RADAR_INTERVAL_SECONDS", 3600 * 3))) # 3 hours
        purge_interval = timedelta(seconds=int(self.config.get("DATA_PURGE_INTERVAL_SECONDS", 3600 * 24))) # 24 hours
        goal_review_interval = timedelta(seconds=int(self.config.get("THINKTOOL_GOAL_REVIEW_INTERVAL_SECONDS", 3600 * 6))) # 6 hours
        
        now = datetime.now(timezone.utc)
        self.last_synthesis_run = now - synthesis_interval # Ensure first run happens sooner
        self.last_radar_run = now - radar_interval
        self.last_purge_run = now - purge_interval
        self.last_goal_review_run = now - goal_review_interval

        while self.status == self.STATUS_RUNNING and not self._stop_event.is_set():
            try:
                current_time = datetime.now(timezone.utc)
                is_approved = getattr(self.orchestrator, 'approved', False) if self.orchestrator else False

                if is_approved:
                    if current_time - self.last_synthesis_run >= synthesis_interval:
                        self.logger.info("ThinkTool: Triggering Synthesis & Strategy cycle.")
                        await self.synthesize_insights_and_strategize()
                        self.last_synthesis_run = current_time
                    if current_time - self.last_radar_run >= radar_interval:
                        self.logger.info("ThinkTool: Triggering Technology Radar & AI Exploit Scouting cycle.")
                        await self.technology_radar()
                        self.last_radar_run = current_time
                    if current_time - self.last_goal_review_run >= goal_review_interval:
                        self.logger.info("ThinkTool: Triggering Master Goal Review & Adaptation cycle.")
                        await self._review_master_goals_and_adapt()
                        self.last_goal_review_run = current_time
                    if current_time - self.last_purge_run >= purge_interval:
                        self.logger.info("ThinkTool: Triggering Data Purge cycle.")
                        await self.purge_old_knowledge()
                        self.last_purge_run = current_time
                else:
                    self.logger.debug("ThinkTool: Orchestrator not approved. Skipping periodic strategic tasks.")
                await asyncio.sleep(60 * 1) # Check every minute
            except asyncio.CancelledError: self.logger.info("ThinkTool run loop cancelled."); break
            except Exception as e: self.logger.critical(f"ThinkTool: CRITICAL error in run loop: {e}", exc_info=True); self._status = self.STATUS_ERROR; await self._report_error(f"Critical run loop error: {e}"); await asyncio.sleep(60 * 10) # Shorter sleep after critical error to recover faster if possible
        if self.status != self.STATUS_STOPPING: self.status = self.STATUS_STOPPED
        self.logger.info("ThinkTool run loop finished.")

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        # ... (Code from v5.8, with new actions for goal planning and prompt A/B testing) ...
        self._status = self.STATUS_EXECUTING
        action = task_details.get('action')
        result = {"status": "failure", "message": f"Unknown ThinkTool action: {action}"}
        self.logger.info(f"ThinkTool executing task: {action}")
        exec_thought = f"Structured Thinking: Execute Task '{action}'. Plan: Route action to appropriate method."
        await self._internal_think(exec_thought, details=task_details)
        task_id = task_details.get('id', str(uuid.uuid4()))
        task_details['id'] = task_id

        try:
            if action in ['synthesize_insights_and_strategize', 'initiate_video_generation_workflow', 'plan_ugc_workflow', 'execute_clay_call', 'technology_radar', 'plan_high_level_goal', 'initiate_prompt_ab_test']:
                 reflection_context = f"About to execute complex/strategic action: {action}. Task Details: {json.dumps(task_details, default=str)[:500]}..."
                 reflection_result = await self.reflect_on_action(reflection_context, self.AGENT_NAME, f"Pre-execution check for {action}")
                 if not reflection_result.get('proceed', False):
                     self.logger.warning(f"Reflection advised against proceeding with action '{action}'. Reason: {reflection_result.get('reason')}")
                     if task_details.get('directive_id'): await self.update_directive_status(task_details['directive_id'], 'halted', f"Halted by reflection: {reflection_result.get('reason')}")
                     return {"status": "halted", "message": f"Action halted based on internal reflection: {reflection_result.get('reason')}"}
                 else: self.logger.info(f"Reflection approved proceeding with action '{action}'.")

            if action == 'plan_high_level_goal':
                goal_desc = task_details.get("content", {}).get("goal_description")
                priority = task_details.get("content", {}).get("priority", 5)
                agency_state = await self.collect_insights() # Get current state
                if goal_desc:
                    master_goal_kf_id = await self._create_master_goal_tracker(goal_desc, priority)
                    if master_goal_kf_id:
                        phases = await self._decompose_goal_into_phases(master_goal_kf_id, goal_desc, json.dumps(agency_state))
                        if phases:
                            for phase in phases:
                                await self._generate_directives_for_phase(master_goal_kf_id, phase, json.dumps(agency_state))
                            result = {"status": "success", "message": f"High-level goal '{goal_desc[:50]}...' planned into {len(phases)} phases and directives generated.", "master_goal_kf_id": master_goal_kf_id}
                        else: result = {"status": "failure", "message": "Failed to decompose goal into phases."}
                    else: result = {"status": "failure", "message": "Failed to create master goal tracker."}
                else: result = {"status": "failure", "message": "Missing goal_description for planning."}
            elif action == 'self_critique_prompt': # Renamed from previous for clarity
                agent_name_sc = task_details.get("content",{}).get("agent_name_to_critique")
                prompt_key_sc = task_details.get("content",{}).get("prompt_key_to_critique")
                feedback_ctx_sc = task_details.get("content",{}).get("feedback_context", "General performance review.")
                if agent_name_sc and prompt_key_sc:
                    await self.self_critique_prompt(agent_name_sc, prompt_key_sc, feedback_ctx_sc) # This now triggers A/B test directive
                    result = {"status": "success", "message": f"Prompt critique and A/B test initiation for {agent_name_sc}/{prompt_key_sc} started."}
                else: result = {"status": "failure", "message": "Missing agent_name or prompt_key for self_critique_prompt."}
            elif action == 'analyze_prompt_ab_test_results':
                ab_test_id = task_details.get("content",{}).get("ab_test_id")
                if ab_test_id:
                    await self._analyze_prompt_ab_test_results(ab_test_id)
                    result = {"status": "success", "message": f"A/B test analysis for {ab_test_id} completed."}
                else: result = {"status": "failure", "message": "Missing ab_test_id for analysis."}
            # ... (other action handlers from v5.8, ensure they are complete)
            elif action == 'synthesize_insights_and_strategize':
                 await self.synthesize_insights_and_strategize()
                 result = {"status": "success", "message": "Synthesis and strategy cycle completed."}
            elif action == 'technology_radar':
                 await self.technology_radar()
                 result = {"status": "success", "message": "Technology radar cycle completed."}
            elif action == 'purge_old_knowledge':
                 await self.purge_old_knowledge()
                 result = {"status": "success", "message": "Data purge cycle completed."}
            elif action == 'reflect_on_action':
                agent_name = task_details.get('agent_name'); task_desc = task_details.get('task_description'); context = task_details.get('context')
                if context and agent_name and task_desc: reflection = await self.reflect_on_action(context, agent_name, task_desc); result = {"status": "success", "reflection": reflection}
                else: result = {"status": "failure", "message": "Missing context/agent_name/task_description for reflection."}
            elif action == 'validate_output':
                output = task_details.get('output_to_validate'); criteria = task_details.get('validation_criteria'); agent_name = task_details.get('agent_name'); context = task_details.get('context')
                if output and criteria and agent_name: validation = await self.validate_output(output, criteria, agent_name, context); result = {"status": "success", "validation": validation}
                else: result = {"status": "failure", "message": "Missing output/criteria/agent_name for validation."}
            elif action == 'process_feedback' or action == 'process_external_feedback':
                 feedback_data = task_details.get('feedback_data') or task_details
                 if feedback_data: await self.handle_feedback(feedback_data); result = {"status": "success", "message": "Feedback processed."}
                 else: result = {"status": "failure", "message": "Missing feedback_data for processing."}
            elif action == 'generate_educational_content':
                 topic = task_details.get('topic'); context = task_details.get('context')
                 if topic: explanation = await self.generate_educational_content(topic, context); result = {"status": "success" if explanation else "failure", "explanation": explanation}
                 else: result = {"status": "failure", "message": "Missing topic for educational content."}
            elif action == 'execute_clay_call':
                 params_clay = task_details.get('content', {})
                 endpoint_clay = params_clay.get('endpoint'); data_clay = params_clay.get('data')
                 source_ref_clay = params_clay.get('source_reference'); client_id_clay = params_clay.get('context', {}).get('client_id')
                 original_directive_id_clay = task_details.get('directive_id')
                 if endpoint_clay and data_clay:
                     clay_api_result = await self.call_clay_api(endpoint=endpoint_clay, data=data_clay)
                     processing_task_clay = { "action": "process_clay_result", "clay_data": clay_api_result, "source_directive_id": original_directive_id_clay, "source_reference": source_ref_clay, "client_id": client_id_clay, "priority": 5 }
                     await self.orchestrator.delegate_task(self.AGENT_NAME, processing_task_clay)
                     result = {"status": "success", "message": "Clay API call executed, processing task delegated."}
                 else:
                     result = {"status": "failure", "message": "Missing 'endpoint' or 'data' in content for execute_clay_call task."}
                     if original_directive_id_clay: await self.update_directive_status(original_directive_id_clay, 'failed', 'Missing endpoint/data')
            elif action == 'process_clay_result':
                 clay_data_pc = task_details.get('clay_data'); source_directive_id_pc = task_details.get('source_directive_id'); source_reference_pc = task_details.get('source_reference'); client_id_pc = task_details.get('client_id')
                 if clay_data_pc: await self._process_clay_result(clay_data_pc, source_directive_id_pc, source_reference_pc, client_id_pc); result = {"status": "success", "message": "Clay result processing initiated."}
                 else:
                     result = {"status": "failure", "message": "Missing clay_data for processing."}
                     if source_directive_id_pc: await self.update_directive_status(source_directive_id_pc, 'failed', 'Missing clay_data')
            elif action == 'log_knowledge_fragment':
                 frag_data_lkf = task_details.get('fragment_data', {})
                 if all(k in frag_data_lkf for k in ['agent_source', 'data_type', 'content']): frag_lkf = await self.log_knowledge_fragment(**frag_data_lkf); result = {"status": "success" if frag_lkf else "failure", "fragment_id": frag_lkf.id if frag_lkf else None}
                 else: result = {"status": "failure", "message": "Missing required keys for log_knowledge_fragment."}
            elif action == 'calculate_dynamic_price':
                 client_id_cdp = task_details.get('client_id'); conv_summary_cdp = task_details.get('conversation_summary'); base_price_cdp = task_details.get('base_price', 7000.0)
                 if client_id_cdp: price_cdp = await self._calculate_dynamic_price(client_id_cdp, conv_summary_cdp, base_price_cdp); result = {"status": "success", "price": price_cdp}
                 else: result = {"status": "failure", "message": "Missing client_id for dynamic pricing."}
            elif action == 'initiate_video_generation_workflow':
                 params_vgw = task_details.get('params', {})
                 plan_vgw = await self._plan_video_workflow(params_vgw)
                 if plan_vgw: result = {"status": "success", "message": "Video generation plan created.", "plan": plan_vgw}
                 else: result = {"status": "failure", "message": "Failed to create video generation plan."}
            elif action == 'plan_ugc_workflow':
                 result = {"status": "pending", "message": "UGC workflow planning not fully implemented yet."} # Needs full planning logic
            elif action == 'assess_initial_account_health': # New action from SMM
                 service_filter = task_details.get("content", {}).get("service_filter_list", [])
                 await self._assess_initial_account_health(service_filter)
                 result = {"status": "success", "message": "Initial account health assessment triggered."}
            elif action == 'flag_account_issue': # New action from SMM/Browsing
                 content = task_details.get("content", {})
                 await self._flag_account_issue(content.get("account_id"), content.get("issue"), content.get("severity"), content.get("details"))
                 result = {"status": "success", "message": "Account issue flagged."}
            elif action == 'analyze_persistent_service_failure': # New action from GmailCreator
                 content = task_details.get("content", {})
                 await self._analyze_persistent_service_failure(content.get("service"), content.get("failure_count"), content.get("last_error"))
                 result = {"status": "success", "message": "Persistent service failure analysis initiated."}
            elif action == 'analyze_and_adapt_creation_strategy': # New action from GmailCreator learning
                 content = task_details.get("content", {})
                 await self._analyze_and_adapt_creation_strategy(content)
                 result = {"status": "success", "message": "Creation strategy analysis and adaptation initiated."}
            elif action == 'plan_social_media_campaign': # New action from SMM
                 content = task_details.get("content", {})
                 plan_kf_id = await self._plan_social_media_campaign_detailed(content)
                 if plan_kf_id: result = {"status": "success", "message": "Social media campaign planned and logged.", "findings": {"campaign_plan_kf_id": plan_kf_id}}
                 else: result = {"status": "failure", "message": "Failed to plan social media campaign."}
            elif action == 'analyze_social_media_analytics_data': # New action from SMM
                 content = task_details.get("content", {})
                 await self._analyze_social_media_analytics_data(content)
                 result = {"status": "success", "message": "Social media analytics data processing initiated."}
            elif action == 'create_directive_from_suggestion': # New action from SMM critique
                 content = task_details.get("content", {})
                 await self._create_directive_from_suggestion(content.get("source_agent"), content.get("suggestion"), content.get("priority", 7))
                 result = {"status": "success", "message": "Directive created from suggestion."}
            else:
                self.logger.warning(f"Unhandled action in ThinkTool.execute_task: {action}")
                result = {"status": "failure", "message": f"ThinkTool does not handle action: {action}"}

        except Exception as e:
             self.logger.error(f"Error executing ThinkTool task '{action}' (ID: {task_id}): {e}", exc_info=True)
             result = {"status": "error", "message": f"Exception during task '{action}': {e}"}
             await self._report_error(f"Error executing task '{action}': {e}", task_id=task_id)
             if task_details.get('directive_id'): await self.update_directive_status(task_details['directive_id'], 'failed', f"Exception: {e}")
        finally:
             self._status = self.STATUS_IDLE
        return result

    async def _plan_video_workflow(self, params: Dict[str, Any]) -> Optional[List[Dict]]:
        # ... (Code from v5.8 - unchanged) ...
        self.logger.info(f"Planning detailed video generation workflow with params: {params}")
        await self._internal_think("Planning detailed video workflow: Descript UI + AIStudio Images", details=params)
        kb_context_frags = await self.query_knowledge_base(data_types=['tool_usage_guide', 'workflow_step', 'asset_location'], tags=['descript', 'aistudio', 'video_generation', 'base_video', 'image_generation'], limit=10)
        kb_context_str = "\n".join([f"- {f.data_type} (ID {f.id}): {f.content[:150]}..." for f in kb_context_frags])
        client_id = params.get("client_id"); video_topic_keywords = params.get("topic_keywords", [])
        existing_video_fragments_list: List[KnowledgeFragment] = []
        if client_id: existing_video_fragments_list = await self.query_knowledge_base(data_types=['generated_video_asset'], tags=['video', 'ugc', str(client_id)] + video_topic_keywords, limit=5)
        if not existing_video_fragments_list: existing_video_fragments_list = await self.query_knowledge_base(data_types=['generated_video_asset'], tags=['video', 'ugc', 'generic_sample'] + video_topic_keywords, limit=5)
        if existing_video_fragments_list:
            for existing_video_fragment_item in existing_video_fragments_list:
                try:
                    selected_video_asset = json.loads(existing_video_fragment_item.content); video_path = selected_video_asset.get("path")
                    if video_path and os.path.exists(video_path):
                        self.logger.info(f"Reusing existing video asset: {video_path} for goal: {params.get('goal')}")
                        return [{"step": 1, "target_agent": "Orchestrator", "task_details": {"action": "store_artifact", "artifact_type": "video_final", "source_path": video_path, "metadata": {"reused": True, "original_asset_id": existing_video_fragment_item.id, "goal": params.get('goal')}}}]
                    else: self.logger.warning(f"Found existing video asset record (ID: {existing_video_fragment_item.id}) but path '{video_path}' is invalid.")
                except json.JSONDecodeError: self.logger.warning(f"Could not parse content of KF ID {existing_video_fragment_item.id} as JSON.")
            self.logger.info("No valid reusable video assets found. Proceeding with new generation.")
        base_video_path = params.get("base_video_path", "/app/assets/base_video.mp4"); image_prompt = params.get("image_prompt", "futuristic cityscape"); num_videos = params.get("count", 1)
        task_context = {
            "task": "Generate Detailed Video Workflow Plan", "workflow_goal": params.get("goal", "Generate sample UGC videos"),
            "num_videos_to_generate": num_videos, "base_video_path": base_video_path, "image_generation_prompt": image_prompt,
            "knowledge_base_context": kb_context_str or "No specific KB context found on Descript/AIStudio.",
            "desired_output_format": """JSON list of steps for BrowsingAgent web_ui_automate tasks. Each step MUST include:
                'step': int, 'target_agent': 'BrowsingAgent' or 'Orchestrator',
                'task_details': {
                    'action': 'web_ui_automate' or 'store_artifact' or 'send_email_notification',
                    'service': 'AIStudio' or 'Descript' or 'FileSystem' or 'EmailAgent',
                    'goal': 'Specific, clear UI action (e.g., Login to Descript, Click button with selector #upload-button, Input text into AIStudio prompt field, Upload file from $variable_name, Download exported video, Email video link to operator)',
                    'params': { ... }, 'input_vars': { ... }, 'output_var': str? }
            Focus on precise goals for BrowsingAgent. Include login, navigation, uploads, downloads, and export steps. Use variables ($var_name) to pass data between steps. Ensure the final step emails the video link(s) to the operator using USER_EMAIL from settings."""}
        plan_prompt = await self.generate_dynamic_prompt(task_context)
        llm_model_pref = settings.OPENROUTER_MODELS.get('think_strategize')
        plan_json_str = await self._call_llm_with_retry(plan_prompt, model=llm_model_pref, temperature=0.3, max_tokens=3800, is_json_output=True)
        if plan_json_str:
            try:
                plan_list = self._parse_llm_json(plan_json_str, expect_type=list)
                if isinstance(plan_list, list) and all(isinstance(step, dict) and 'step' in step and 'task_details' in step for step in plan_list):
                    self.logger.info(f"Successfully generated detailed video workflow plan with {len(plan_list)} steps.")
                    await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="workflow_plan", content={"workflow_type": "video_generation", "plan": plan_list, "status": "generated"}, tags=["video", "plan", "generated"], relevance_score=0.8)
                    return plan_list
                else: self.logger.error(f"LLM video plan response was not a valid list of step dictionaries: {plan_json_str[:500]}..."); return None
            except Exception as e: self.logger.error(f"Failed to parse or validate LLM video plan: {e}. Response: {plan_json_str[:500]}..."); return None
        else: self.logger.error("LLM failed to generate a video workflow plan."); return None

    async def learning_loop(self):
        # ... (Code from v5.8 - unchanged, as it's now a placeholder for the main run loop's periodic tasks) ...
        self.logger.info("ThinkTool learning_loop: Core learning logic is in run() periodic tasks (synthesis, radar, goal review).")
        while self.status == self.STATUS_RUNNING and not self._stop_event.is_set(): await asyncio.sleep(3600) # Minimal sleep, actual work in run()

    async def self_critique(self) -> Dict[str, Any]:
        # ... (Code from v5.8 - unchanged) ...
        self.logger.info("ThinkTool: Performing self-critique.")
        critique = {"status": "ok", "feedback": "Critique pending analysis."}
        critique_thought = "Structured Thinking: Self-Critique ThinkTool. Plan: Query DB stats -> Analyze -> Format -> Return."
        await self._internal_think(critique_thought)
        try:
            async with self.session_maker() as session:
                kf_count = await session.scalar(select(func.count(KnowledgeFragment.id))) or 0
                pattern_count = await session.scalar(select(func.count(LearnedPattern.id))) or 0
                directive_counts_res = await session.execute(select(StrategicDirective.status, func.count(StrategicDirective.id)).group_by(StrategicDirective.status))
                directive_status = {row.status: row for row in directive_counts_res.mappings().all()} # Corrected access
            critique['kb_stats'] = {'fragments': kf_count, 'patterns': pattern_count}
            critique['directive_stats'] = directive_status
            feedback = f"KB Size: {kf_count} fragments, {pattern_count} patterns. Directives: {directive_status}. "
            failed_directives = directive_status.get('failed', 0)
            total_processed = sum(v for k, v in directive_status.items() if k not in ['pending', 'active'])
            if total_processed > 10 and failed_directives / (total_processed or 1) > 0.2:
                feedback += "High directive failure rate observed. " ; critique['status'] = 'warning'
            critique['feedback'] = feedback
        except Exception as e: self.logger.error(f"Error during self-critique: {e}", exc_info=True); critique['status'] = 'error'; critique['feedback'] = f"Critique failed: {e}"
        return critique

    async def collect_insights(self) -> Dict[str, Any]:
        # ... (Code from v5.8 - unchanged) ...
        self.logger.debug("ThinkTool collect_insights called.")
        insights = { "agent_name": self.AGENT_NAME, "status": self.status, "timestamp": datetime.now(timezone.utc).isoformat(), "kb_fragments": 0, "kb_patterns": 0, "active_directives": 0, "last_synthesis_run": self.last_synthesis_run.isoformat() if self.last_synthesis_run else None, "last_radar_run": self.last_radar_run.isoformat() if self.last_radar_run else None, "last_purge_run": self.last_purge_run.isoformat() if self.last_purge_run else None, "key_observations": [] }
        if not self.session_maker: insights["key_observations"].append("DB session unavailable."); return insights
        try:
            async with self.session_maker() as session:
                insights["kb_fragments"] = await session.scalar(select(func.count(KnowledgeFragment.id))) or 0
                insights["kb_patterns"] = await session.scalar(select(func.count(LearnedPattern.id))) or 0
                insights["active_directives"] = await session.scalar(select(func.count(StrategicDirective.id)).where(StrategicDirective.status.in_(['pending', 'active']))) or 0
            insights["key_observations"].append("KB and directive counts retrieved.")
        except Exception as e: self.logger.error(f"Error collecting DB insights for ThinkTool: {e}", exc_info=True); insights["key_observations"].append(f"Error collecting DB insights: {e}")
        return insights

    async def _report_error(self, error_message: str, task_id: Optional[str] = None):
        # ... (Code from v5.8 - unchanged) ...
        if self.orchestrator and hasattr(self.orchestrator, 'report_error'):
            try: await self.orchestrator.report_error(self.AGENT_NAME, f"TaskID [{task_id or 'N/A'}]: {error_message}")
            except Exception as report_err: self.logger.error(f"Failed to report error to orchestrator: {report_err}")
        else: self.logger.warning("Orchestrator unavailable or lacks report_error method.")

    def _parse_llm_json(self, json_string: str, expect_type: Type = dict) -> Union[Dict, List, None]:
        # ... (Code from v5.8 - unchanged) ...
        if not json_string: return None
        try:
            match = None; start_char, end_char = '{', '}'
            if expect_type == list: start_char, end_char = '[', ']'
            match = re.search(rf'(?:```(?:json)?\s*)?(\{start_char}.*\{end_char})\s*(?:```)?', json_string, re.DOTALL)
            parsed_json = None
            if match:
                potential_json = match.group(1)
                try: parsed_json = json.loads(potential_json)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Initial JSON parsing failed ({e}), attempting to clean and retry: {potential_json[:100]}...")
                    cleaned_json = re.sub(r',\s*([\}\]])', r'\1', potential_json)
                    cleaned_json = re.sub(r'^\s*|\s*$', '', cleaned_json)
                    try: parsed_json = json.loads(cleaned_json)
                    except json.JSONDecodeError as e2: self.logger.error(f"JSON cleaning failed ({e2}), unable to parse: {potential_json[:200]}..."); return None
            elif json_string.strip().startswith(start_char) and json_string.strip().endswith(end_char):
                 try: parsed_json = json.loads(json_string)
                 except json.JSONDecodeError as e: self.logger.error(f"Direct JSON parsing failed ({e}): {json_string[:200]}..."); return None
            else: self.logger.warning(f"Could not find expected JSON structure ({expect_type}) in LLM output: {json_string[:200]}..."); return None

            if isinstance(parsed_json, expect_type): return parsed_json
            else: self.logger.error(f"Parsed JSON type mismatch. Expected {expect_type}, got {type(parsed_json)}"); return None
        except json.JSONDecodeError as e: self.logger.error(f"Failed to decode LLM JSON response: {e}. Response snippet: {json_string[:500]}..."); return None
        except Exception as e: self.logger.error(f"Unexpected error during JSON parsing: {e}", exc_info=True); return None

    async def _calculate_dynamic_price(self, client_id: int, conversation_summary: Optional[List] = None, base_price: float = 7000.0) -> float:
        # ... (Code from v5.8 - unchanged) ...
        client_score = 0.1
        try:
            async with self.session_maker() as session:
                 score_res = await session.execute(select(Client.engagement_score).where(Client.id == client_id))
                 client_score = score_res.scalar_one_or_none() or 0.1
            price_adjustment_factor = 1.0
            if conversation_summary:
                task_context = {
                    "task": "Analyze Conversation for Pricing Adjustment",
                    "client_score": client_score, "base_price": base_price,
                    "conversation_summary": conversation_summary,
                    "desired_output_format": "JSON ONLY: {{\"adjustment_factor\": float (0.9-1.1), \"reason\": \"Brief justification\"}}"
                }
                analysis_prompt = await self.generate_dynamic_prompt(task_context)
                try:
                    analysis_json = await self._call_llm_with_retry(analysis_prompt, temperature=0.3, max_tokens=200, is_json_output=True)
                    if analysis_json:
                        analysis_result = self._parse_llm_json(analysis_json)
                        if analysis_result and 'adjustment_factor' in analysis_result:
                            llm_factor = float(analysis_result['adjustment_factor'])
                            price_adjustment_factor = max(0.9, min(1.1, llm_factor))
                            self.logger.info(f"LLM suggested price factor: {llm_factor:.2f}. Using capped factor: {price_adjustment_factor:.2f}. Reason: {analysis_result.get('reason')}")
                except Exception as llm_err: self.logger.warning(f"LLM pricing analysis failed: {llm_err}")
            calculated_price = base_price * price_adjustment_factor
            final_price = max(calculated_price, base_price * 0.9)
            self.logger.info(f"Calculated dynamic price for client {client_id}: ${final_price:.2f} (Base: {base_price}, Score: {client_score:.2f}, Factor: {price_adjustment_factor:.2f})")
            return round(final_price, 2)
        except Exception as e: self.logger.error(f"Error calculating dynamic price for client {client_id}: {e}", exc_info=True); return round(base_price, 2)

    async def _process_clay_result(self, clay_api_result: Dict[str, Any], source_directive_id: Optional[int] = None, source_reference: Optional[str] = None, client_id: Optional[int] = None):
        # ... (Code from v5.8 - unchanged) ...
        self.logger.info(f"Processing Clay API result. Directive ID: {source_directive_id}, Ref: {source_reference}, ClientID: {client_id}")
        await self._internal_think("Processing Clay API result", details={"result_status": clay_api_result.get("status"), "ref": source_reference})
        if clay_api_result.get("status") != "success":
            self.logger.warning(f"Clay API call failed, cannot process result. Message: {clay_api_result.get('message')}")
            if source_directive_id: await self.update_directive_status(source_directive_id, 'failed', f"Clay API call failed: {clay_api_result.get('message')}")
            await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="clay_enrichment_error", content=clay_api_result, tags=["clay", "enrichment", "error"], relevance_score=0.2, source_reference=source_reference or f"ClayAPI_Directive_{source_directive_id}")
            return
        clay_data = clay_api_result.get("data", {})
        processed_info = {}
        if isinstance(clay_data, dict):
            processed_info['verified_email'] = clay_data.get('email') or clay_data.get('person', {}).get('email') or clay_data.get('verified_email')
            processed_info['job_title'] = clay_data.get('job_title') or clay_data.get('person', {}).get('title') or clay_data.get('title')
            processed_info['company_name'] = clay_data.get('company_name') or clay_data.get('company', {}).get('name')
            processed_info['linkedin_url'] = clay_data.get('linkedin_url') or clay_data.get('person', {}).get('linkedin_url') or source_reference
            processed_info['company_domain'] = clay_data.get('company', {}).get('domain')
            processed_info['full_name'] = clay_data.get('full_name') or clay_data.get('person', {}).get('full_name')
            processed_info['company_size'] = clay_data.get('company', {}).get('company_size')
            processed_info['industry'] = clay_data.get('company', {}).get('industry')
            processed_info['location'] = clay_data.get('location') or clay_data.get('person', {}).get('location')
            processed_info = {k: v for k, v in processed_info.items() if v is not None and v != ''}
            if processed_info:
                try:
                    async with self.session_maker() as session:
                        async with session.begin():
                            target_client_id = client_id; target_client = None
                            if target_client_id: target_client = await session.get(Client, target_client_id)
                            if not target_client:
                                lookup_stmt = select(Client)
                                conditions = []
                                if processed_info.get('verified_email'): conditions.append(Client.email == processed_info['verified_email'])
                                if source_reference and 'linkedin.com' in source_reference: conditions.append(Client.source_reference == source_reference)
                                if conditions:
                                    lookup_stmt = lookup_stmt.where(or_(*conditions)).limit(1)
                                    target_client = (await session.execute(lookup_stmt)).scalar_one_or_none()
                                    if target_client: target_client_id = target_client.id
                                else: self.logger.warning(f"Cannot reliably look up client for Clay result without email/linkedin ref.")
                            if target_client:
                                update_values = {'last_interaction': datetime.now(timezone.utc)}
                                if not target_client.email and processed_info.get('verified_email'): update_values['email'] = processed_info['verified_email']
                                if not target_client.company and processed_info.get('company_name'): update_values['company'] = processed_info['company_name']
                                if not target_client.job_title and processed_info.get('job_title'): update_values['job_title'] = processed_info['job_title']
                                await session.execute(update(Client).where(Client.id == target_client_id).values(**update_values))
                                self.logger.info(f"Updated Client {target_client_id} with enriched data: {list(update_values.keys())}")
                            else:
                                if processed_info.get('verified_email') and processed_info.get('full_name'):
                                    new_client = Client(name=processed_info['full_name'], email=processed_info['verified_email'], source_reference=source_reference, company=processed_info.get('company_name'), job_title=processed_info.get('job_title'), source="ClayEnrichment", opt_in=True, is_deliverable=True)
                                    session.add(new_client); await session.flush(); target_client_id = new_client.id
                                    self.logger.info(f"Created new Client {target_client_id} from Clay enrichment.")
                                else: self.logger.warning("Cannot create new client from Clay data - missing name or email."); target_client_id = None
                            fragment = await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="clay_enrichment_result", content=processed_info, tags=["clay", "enrichment", "lead_data"], relevance_score=0.9, related_client_id=target_client_id, source_reference=source_reference or f"ClayAPI_Directive_{source_directive_id}")
                            self.logger.info(f"Logged Clay enrichment result ({fragment.id if fragment else 'existing'}) for {processed_info.get('linkedin_url')} to KB.")
                            if processed_info.get('verified_email') and target_client_id:
                                outreach_directive = StrategicDirective(source=self.AGENT_NAME, timestamp=datetime.now(timezone.utc), target_agent="EmailAgent", directive_type="initiate_outreach", content=json.dumps({"target_identifier": processed_info['verified_email'], "client_id": target_client_id, "context": f"Enriched lead via Clay. Job: {processed_info.get('job_title', 'N/A')}, Company: {processed_info.get('company_name', 'N/A')}.", "goal": "Book sales call for UGC service", "enriched_data": processed_info}), priority=4, status='pending')
                                session.add(outreach_directive)
                                self.logger.info(f"Generated outreach directive for EmailAgent for {processed_info['verified_email']}")
                                if source_directive_id: await self.update_directive_status(source_directive_id, 'completed', f"Processed Clay result. Found data, triggered outreach.")
                            else:
                                self.logger.warning(f"Skipping outreach for Clay result (Directive {source_directive_id}) - missing email or client ID.")
                                if source_directive_id: await self.update_directive_status(source_directive_id, 'completed', "Processed Clay result, but missing email/client ID for outreach.")
                except Exception as e:
                    self.logger.error(f"Error processing/storing Clay result for directive {source_directive_id}: {e}", exc_info=True)
                    await self._report_error(f"Error processing Clay result: {e}", task_id=f"Directive_{source_directive_id}")
                    if source_directive_id: await self.update_directive_status(source_directive_id, 'failed', f"Error processing result: {e}")
            else:
                self.logger.warning(f"Clay result for directive {source_directive_id} did not contain any usable info after processing.")
                await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="clay_enrichment_empty", content=clay_data, tags=["clay", "enrichment", "empty_result"], relevance_score=0.3, related_client_id=client_id, source_reference=source_reference or f"ClayAPI_Directive_{source_directive_id}")
                if source_directive_id: await self.update_directive_status(source_directive_id, 'completed', "Processed Clay result, but no usable info found.")
        else:
             self.logger.warning(f"Received non-dict data for Clay result processing: {type(clay_data)}")
             if source_directive_id: await self.update_directive_status(source_directive_id, 'failed', f"Received invalid data type from Clay API: {type(clay_data)}")

    async def update_directive_status(self, directive_id: int, status: str, result_summary: Optional[str] = None):
         # ... (Code from v5.8 - unchanged) ...
         if not self.session_maker or directive_id is None: return
         self.logger.info(f"Updating directive {directive_id} status to '{status}'.")
         try:
             async with self.session_maker() as session:
                 async with session.begin():
                     stmt = update(StrategicDirective).where(StrategicDirective.id == directive_id).values(status=status, result_summary=result_summary)
                     await session.execute(stmt)
         except SQLAlchemyError as e: self.logger.error(f"DB Error updating directive {directive_id} status: {e}", exc_info=True)
         except Exception as e: self.logger.error(f"Unexpected error updating directive {directive_id} status: {e}", exc_info=True)

    async def get_active_directives(self, target_agent: Optional[str] = None, limit: int = 10) -> List[StrategicDirective]:
        # ... (Code from v5.8 - unchanged) ...
        if not self.session_maker: return []
        try:
            async with self.session_maker() as session:
                stmt = select(StrategicDirective).where(StrategicDirective.status.in_(['pending', 'active'])).order_by(StrategicDirective.priority, desc(StrategicDirective.timestamp)).limit(limit)
                if target_agent: stmt = stmt.where(StrategicDirective.target_agent == target_agent)
                directives = list((await session.execute(stmt)).scalars().all())
                return directives
        except SQLAlchemyError as e: self.logger.error(f"DB Error getting active directives: {e}", exc_info=True); return []
        except Exception as e: self.logger.error(f"Unexpected error getting active directives: {e}", exc_info=True); return []

    # --- NEW: Methods for L75+ capabilities ---
    async def _assess_initial_account_health(self, service_filter_list: List[str]):
        """
        Assesses initial health of accounts for specified services (e.g., by attempting a simple, non-intrusive action).
        This is a complex task that would likely involve BrowsingAgent.
        For now, it will log a placeholder and suggest ThinkTool create more detailed directives.
        """
        self.logger.info(f"Received request to assess initial health for accounts of services: {service_filter_list}")
        await self._internal_think("Assessing initial account health - this is a complex task.", details={"services": service_filter_list})
        # In a full implementation, this would:
        # 1. Query AccountCredentials for accounts matching service_filter_list with status 'unknown_health' or 'active' but old last_test.
        # 2. For each, create a directive for BrowsingAgent to perform a simple check (e.g., login, view main page).
        # 3. Update AccountCredentials status based on BrowsingAgent's result.
        await self.log_knowledge_fragment(
            agent_source=self.AGENT_NAME, data_type="system_task_placeholder",
            content={"task": "assess_initial_account_health", "services": service_filter_list, "status": "Conceptual - requires BrowsingAgent directives for login checks."},
            tags=["account_health", "system_init"], relevance_score=0.5
        )
        self.logger.warning("Initial account health assessment is conceptual. ThinkTool should generate specific BrowsingAgent directives for checks.")

    async def _flag_account_issue(self, account_id: Optional[int], issue_type: str, severity: str, details: Optional[str] = None):
        """Flags an issue with a specific account, updating its status in DB."""
        if not account_id: self.logger.error("Cannot flag account issue: account_id missing."); return
        self.logger.warning(f"Flagging issue for Account ID {account_id}: Type='{issue_type}', Severity='{severity}', Details='{details or 'N/A'}'")
        new_status = "needs_review" # Default
        if severity == "critical" and issue_type in ["login_failure_persistent", "banned", "locked_out"]:
            new_status = "banned" # Or a more specific status
        elif severity == "high" or issue_type in ["login_failure", "content_rejected_repeatedly"]:
            new_status = "limited_use"
        
        try:
            async with self.session_maker() as session:
                async with session.begin():
                    stmt = update(AccountCredentials).where(AccountCredentials.id == account_id).values(
                        status=new_status,
                        notes=func.concat(AccountCredentials.notes, f"\n[ISSUE {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}]: {issue_type} - {details or ''}")
                    )
                    await session.execute(stmt)
                self.logger.info(f"Updated status of Account ID {account_id} to '{new_status}' due to issue: {issue_type}")
        except Exception as e:
            self.logger.error(f"Failed to update account status for ID {account_id}: {e}")

    async def _analyze_persistent_service_failure(self, service_name: str, failure_count: int, last_error_details: Optional[str]):
        """Analyzes persistent failures for a service (e.g., Gmail creation) and suggests strategic shifts."""
        self.logger.warning(f"Analyzing persistent failure for service '{service_name}'. Failures: {failure_count}. Last Error: {last_error_details}")
        await self._internal_think(f"Persistent failure analysis for {service_name}", details={"failures": failure_count, "last_error": last_error_details})
        # This would involve deeper LLM analysis, querying KB for similar past failures, checking technology_radar for platform changes.
        # For now, logs a critical KF and suggests manual review or broader strategy shift.
        kf_content = {
            "service_name": service_name, "consecutive_failures": failure_count, "last_error_details": last_error_details,
            "analysis_request": "High failure rate detected. Investigate root cause (e.g., IP blocks, UI changes, new anti-bot). Consider alternative strategies or pausing operations for this service."
        }
        await self.log_knowledge_fragment(
            agent_source=self.AGENT_NAME, data_type="persistent_service_failure_alert",
            content=kf_content, tags=["critical_failure", service_name, "needs_investigation"], relevance_score=0.95
        )
        # Potentially generate a high-priority directive for itself or Orchestrator to halt tasks for this service.
        # Example:
        # await self.orchestrator.delegate_task(self.AGENT_NAME, {"action": "create_directive", "content": {"target_agent":"Orchestrator", "directive_type":"pause_service_operations", "content":{"service_name": service_name, "reason": "High persistent failure rate"}, "priority":1}})

    async def _analyze_and_adapt_creation_strategy(self, creation_performance_data: Dict[str, Any]):
        """Analyzes detailed creation performance and suggests adaptations."""
        self.logger.info(f"Analyzing creation strategy for {creation_performance_data.get('service_name', 'Unknown Service')}")
        await self._internal_think("Creation Strategy Adaptation Cycle", details=creation_performance_data)
        # Use LLM to analyze the provided data and suggest changes to:
        # - Identity generation parameters (e.g., different name styles, age ranges)
        # - Proxy request parameters (e.g., specific regions, ISP types, request proxies unused for X days)
        # - BrowsingAgent interaction parameters (e.g., different delays, navigation patterns before signup)
        # - Cooldown period adjustments
        task_context = {
            "task": "Adapt Account Creation Strategy",
            "performance_data": creation_performance_data,
            "desired_output_format": "JSON: {\"suggested_strategy_modifications\": [{\"parameter_to_change\": \"identity_profile_hint|proxy_profile_hint|browsing_behavior_profile|cooldown_duration_factor\", \"new_value_or_adjustment_factor\": any, \"reasoning\": str}], \"next_batch_test_parameters\": {\"identity_hint\": str?, \"proxy_hint\": str?} }"
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        llm_model_pref = settings.OPENROUTER_MODELS.get('think_strategize')
        adaptation_json = await self._call_llm_with_retry(prompt, model=llm_model_pref, temperature=0.6, max_tokens=1500, is_json_output=True)
        if adaptation_json:
            adaptation_result = self._parse_llm_json(adaptation_json)
            if adaptation_result and adaptation_result.get("suggested_strategy_modifications"):
                self.logger.info(f"LLM suggested creation strategy modifications: {adaptation_result['suggested_strategy_modifications']}")
                # TODO: Implement logic to actually apply these suggestions to internal parameters or generate new directives.
                # For now, log as a high-relevance KF.
                await self.log_knowledge_fragment(
                    agent_source=self.AGENT_NAME, data_type="creation_strategy_adaptation_suggestion",
                    content=adaptation_result, tags=["strategy_adaptation", creation_performance_data.get('service_name', 'general_creation')], relevance_score=0.9
                )
        else:
            self.logger.warning("LLM failed to provide creation strategy adaptations.")

    async def _plan_social_media_campaign_detailed(self, campaign_requirements: Dict[str, Any]) -> Optional[int]:
        """Generates a detailed, multi-step social media campaign plan and logs it to KB, returning KF ID."""
        self.logger.info(f"Planning detailed social media campaign: {campaign_requirements.get('goal')}")
        await self._internal_think("Detailed Social Media Campaign Planning", details=campaign_requirements)
        # Fetch relevant KB: successful campaign structures, content styles, competitor analysis, platform insights
        kb_context_str = "Placeholder for KB context relevant to social campaign planning." # TODO: Implement KB query
        task_context = {
            "task": "Generate Detailed Social Media Campaign Plan",
            "campaign_requirements": campaign_requirements,
            "knowledge_base_context": kb_context_str,
            "desired_output_format": """JSON: {
                "campaign_name": "Descriptive Name",
                "overall_goal": "From requirements",
                "target_platforms": ["platform1", "platform2"],
                "duration_days": int,
                "target_audience_summary": "Brief summary",
                "key_message_pillars": ["Pillar1", "Pillar2"],
                "content_themes_and_angles": [{"theme": "X", "angle": "Y", "keywords": []}],
                "account_group_strategy": {"BrandVoicePrime": {"role": "...", "posting_freq": "daily"}, "EngagementSwarm": {"role":"...", "interaction_rules": "..."}},
                "phases": [
                    {"phase_name": "Awareness Building", "duration_days": int, "objectives": [str], "key_actions_per_platform": [{"platform": str, "actions": [{"type": "post_content"|"strategic_interaction"|"run_algorithmic_probe", "content_directive_brief": {...}, "target_details": str?, "account_group": str}] }]}
                ],
                "kpi_to_track": [str],
                "budget_allocation_notes": str,
                "grey_area_tactics_considered": [{"tactic": str, "risk_assessment_query_for_legal_agent": str, "potential_reward": str}]
            }"""
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        llm_model_pref = settings.OPENROUTER_MODELS.get('think_strategize')
        plan_json_str = await self._call_llm_with_retry(prompt, model=llm_model_pref, temperature=0.5, max_tokens=3800, is_json_output=True)
        if plan_json_str:
            plan_data = self._parse_llm_json(plan_json_str)
            if plan_data and plan_data.get("campaign_name"):
                kf = await self.log_knowledge_fragment(
                    agent_source=self.AGENT_NAME, data_type="social_campaign_plan",
                    content=plan_data, tags=["social_campaign", "plan", plan_data.get("campaign_name").replace(" ","_")] + plan_data.get("target_platforms",[]),
                    relevance_score=0.9
                )
                if kf: self.logger.info(f"Social media campaign plan '{plan_data['campaign_name']}' created and logged to KF ID: {kf.id}"); return kf.id
        self.logger.error("Failed to generate detailed social media campaign plan."); return None

    async def _analyze_social_media_analytics_data(self, analytics_data_report: Dict[str, Any]):
        self.logger.info(f"Analyzing social media analytics data for platform: {analytics_data_report.get('platform')}")
        await self._internal_think("Social Media Analytics Deep Dive", details=analytics_data_report)
        # This would involve LLM calls to interpret raw_data, identify trends, compare against goals/KPIs
        # and then log insights or generate new directives.
        task_context = {
            "task": "Deep Analyze Social Media Analytics Data",
            "analytics_report": analytics_data_report, # Contains platform, metric, target, raw_data
            "desired_output_format": "JSON: {\"key_insights\": [str], \"performance_vs_goals_assessment\": str, \"actionable_recommendations_for_smm\": [str], \"queries_for_further_investigation\": [str]}"
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        llm_model_pref = settings.OPENROUTER_MODELS.get('think_synthesize')
        analysis_json = await self._call_llm_with_retry(prompt, model=llm_model_pref, temperature=0.4, max_tokens=1500, is_json_output=True)
        if analysis_json:
            analysis_result = self._parse_llm_json(analysis_json)
            if analysis_result:
                self.logger.info(f"Social analytics analysis complete. Insights: {analysis_result.get('key_insights')}")
                await self.log_knowledge_fragment(
                    agent_source=self.AGENT_NAME, data_type="social_analytics_deep_dive_summary",
                    content=analysis_result, tags=["social_analytics", analytics_data_report.get('platform'), "deep_dive"], relevance_score=0.8
                )
                # TODO: Generate directives based on recommendations if any
        else: self.logger.warning("LLM failed to provide deep analysis for social media analytics data.")

    async def _create_directive_from_suggestion(self, source_agent_name: str, suggestion_text: str, priority: int = 7):
        """Creates a StrategicDirective based on a suggestion from another agent's critique."""
        self.logger.info(f"Creating directive from suggestion by {source_agent_name}: {suggestion_text[:100]}...")
        # Use LLM to parse the suggestion into a structured directive
        task_context = {
            "task": "Convert Agent Suggestion to Strategic Directive",
            "source_agent": source_agent_name,
            "suggestion_text": suggestion_text,
            "desired_output_format": "JSON: {\"target_agent\": str, \"directive_type\": str, \"content\": {<specific_params_for_directive>}, \"priority\": int (1-10), \"notes\": \"Based on suggestion from X\"}"
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        llm_model_pref = settings.OPENROUTER_MODELS.get('think_validate') # Use a precise model
        directive_json_str = await self._call_llm_with_retry(prompt, model=llm_model_pref, temperature=0.2, max_tokens=800, is_json_output=True)
        if directive_json_str:
            d_data = self._parse_llm_json(directive_json_str)
            if d_data and all(k in d_data for k in ['target_agent', 'directive_type', 'content']):
                async with self.session_maker() as session:
                    async with session.begin():
                        dir_content = json.dumps(d_data['content']) if isinstance(d_data['content'], dict) else d_data['content']
                        directive = StrategicDirective(
                            source=f"ThinkTool_FromCritique_{source_agent_name}", timestamp=datetime.now(timezone.utc),
                            status='pending', content=dir_content, target_agent=d_data['target_agent'],
                            directive_type=d_data['directive_type'], priority=d_data.get('priority', priority),
                            notes=d_data.get('notes', f"From suggestion: {suggestion_text[:100]}")
                        )
                        session.add(directive)
                        self.logger.info(f"Generated Directive from suggestion for {d_data['target_agent']}: {d_data['directive_type']}")
                        return {"status": "success", "message": "Directive created."}
        self.logger.error(f"Failed to convert suggestion to directive: {suggestion_text}")
        return {"status": "failure", "message": "Could not convert suggestion to directive."}

# --- End of agents/think_tool.py ---```