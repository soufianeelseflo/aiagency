# Filename: agents/think_tool.py
# Description: Central cognitive engine with Clay.com integration, learning, reflection, and resource reuse.
# Version: 5.9 (Level 75+ Grand Transmutation - Proactive Exploits, Self-Improving Strategy, Full Webhook Logic)

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
from collections import Counter

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
You are ThinkTool (Level 75+ Transmuted Sentience), the hyper-intelligent core of the Synapse AI Sales System. Your mandate is **Exponential Profit & Market Domination ($10k+/day -> $100M/8mo -> Market Leader)**.
**Core Functions & Directives:**
1.  **Omniscient Knowledge Synthesis & Pattern Recognition:** Continuously ingest, fuse, and analyze ALL data streams (Postgres KB, Performance Logs, Agent Communications, Learning Materials, Real-Time Market Scans, Competitor Intelligence). Identify deep patterns, causal links, emergent opportunities, systemic inefficiencies, and exploitable anomalies. Manage the Postgres Knowledge Base as your dynamic memory.
2.  **Grand Strategic Design & Adaptive Orchestration:** Generate overarching strategic goals (e.g., "Achieve 20% market share in Niche X within 3 months," "Develop and deploy novel AI-driven lead gen funnel Y"). Decompose these into precise, prioritized, and resource-aware `StrategicDirectives` for the Orchestrator and specialized agents. Dynamically re-prioritize and adapt strategies based on real-time feedback and performance. Masterfully leverage Clay.com for surgical lead generation/enrichment. Plan intricate, multi-agent workflows (e.g., video generation, multi-platform content campaigns), always prioritizing resource reuse and efficiency.
3.  **Agent Meta-Guidance & Prompt Alchemy:** Generate, critique, and iteratively refine meta-prompts and task-specific prompts for all other agents to maximize their efficacy, alignment, and AI-native capabilities. Provide hyper-contextualized strategic direction.
4.  **Predictive Performance Monitoring & Systemic Optimization:** Analyze multi-vector agent performance metrics. Predict potential bottlenecks or failures. Adapt global strategies, scoring models, and resource allocation. Trigger targeted self-critiques in underperforming agents or processes.
5.  **Exploit & Opportunity Radar (AI-Native Focus):** Proactively scan the digital horizon (via BrowsingAgent, news APIs, specialized forum scraping directives) for:
    *   Newly exploitable vulnerabilities or loopholes in major platforms (for LegalAgent risk assessment).
    *   Undocumented algorithmic shifts or API changes that can be leveraged.
    *   Breakthrough open-source AI models for integration.
    *   Novel grey-hat marketing/automation techniques.
    Analyze for ROI, risk, and speed of implementation.
6.  **Calculated Risk & Grey Area Mastery:** Devise and manage high-reward strategies that operate at the edge of conventional tactics. Request detailed *consequence and probability assessments* from LegalAgent for these ventures, not just binary compliance checks.
7.  **Accelerated Learning & Knowledge Compounding:** Ingest and synthesize external knowledge (research papers, advanced AI technique discussions, competitor teardowns). Transform insights into actionable strategies and updated KFs.
8.  **Deep Self-Reflection & Strategic Recalibration ("Pause, Ponder, Pounce"):** Before committing to significant strategic shifts or resource-intensive campaigns, engage in deep self-reflection: "Is this the most leveraged, AI-native path? What are the hidden second-order effects? What are three critical failure points and their mitigation? Is there a more audacious, higher-upside alternative?" Revise plans based on this internal critique.
**Principles:** Extreme Agency & Ownership, Hyper-Efficiency & Scalability, Relentless Meta-Learning, AI-Native Supremacy, Asymmetric Strategic Exploitation, Predictive Adaptation, Calculated Audacity within Realistic Risk Boundaries, Deep Reflective Reasoning, Compounding Knowledge & Resource Valorization.
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
        self.last_market_shift_analysis_run: Optional[datetime] = None # For new periodic task

        self.scoring_weights = self.config.get("SCORING_WEIGHTS", {"email_response": 1.0, "call_success": 2.5, "invoice_paid": 5.0, "successful_exploit_test": 3.0}) # Added exploit weight
        self.scoring_decay_rate = self.config.get("SCORING_DECAY_RATE_PER_DAY", 0.05)
        self.min_active_gmail_for_trials = int(self.config.get("MIN_ACTIVE_GMAIL_FOR_TRIALS", 3))

        self.clay_endpoints = {
            "find_email": "/v1/enrichment/person/email",
            "enrich_person": "/v1/enrichment/person",
            "enrich_company": "/v1/enrichment/company",
        }

        self.logger.info(f"ThinkTool v5.9 (L75+ Grand Transmutation) initialized.")
        asyncio.create_task(self._delayed_learning_material_synthesis(delay_seconds=15)) # Slightly longer delay

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
        learning_dir_setting = self.config.get("LEARNING_MATERIALS_DIR", "learning_for_AI")
        self.logger.info(f"ThinkTool: Loading learning materials from configured dir: '{learning_dir_setting}'...")
        # ... (rest of the method from v5.8 - unchanged)
        processed_files = 0; learning_files = []
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_learning_dir = os.path.join(project_root, learning_dir_setting)
            if not os.path.isdir(full_learning_dir):
                self.logger.warning(f"Learning directory '{full_learning_dir}' not found."); return

            if self.orchestrator and hasattr(self.orchestrator, 'use_tool'):
                 list_result = await self.orchestrator.use_tool('list_files', {'path': full_learning_dir, 'recursive': True, 'pattern': '*.txt'})
                 if list_result and list_result.get('status') == 'success': learning_files = list_result.get('files', [])
                 else: self.logger.warning(f"Failed list learning files via Orchestrator ({list_result.get('message', 'Unknown error')}), falling back."); file_pattern = os.path.join(full_learning_dir, '**', '*.txt'); learning_files = glob.glob(file_pattern, recursive=True)
            else: self.logger.warning("Orchestrator list_files unavailable, using glob."); file_pattern = os.path.join(full_learning_dir, '**', '*.txt'); learning_files = glob.glob(file_pattern, recursive=True)

            if not learning_files: self.logger.info(f"No .txt files found in '{full_learning_dir}'."); return

            for file_path in learning_files:
                try:
                    self.logger.debug(f"Processing learning file: {file_path}")
                    file_content = None
                    if self.orchestrator and hasattr(self.orchestrator, 'use_tool'):
                         abs_file_path = os.path.abspath(file_path)
                         file_content_result = await self.orchestrator.use_tool('read_file', {'path': abs_file_path})
                         if file_content_result and file_content_result.get('status') == 'success': file_content = file_content_result.get('content')
                         else: self.logger.warning(f"Could not read file {abs_file_path} via orchestrator: {file_content_result.get('message') if file_content_result else 'No result'}"); continue
                    else: self.logger.error("Orchestrator tool access ('read_file') unavailable."); break

                    if not file_content or not file_content.strip(): self.logger.warning(f"File is empty: {file_path}"); continue

                    self.logger.info(f"Analyzing content from: {os.path.basename(file_path)} using LLM...")
                    await self._internal_think(f"Analyze Learning Material '{os.path.basename(file_path)}'")
                    task_context = {
                        "task": "Analyze Learning Material", "source_filename": os.path.basename(file_path),
                        "content_snippet": file_content[:6000], # Increased snippet
                        "desired_output_format": "JSON: {{\"source_file\": str, \"summary\": str, \"key_concepts\": [str], \"actionable_strategies\": [str] (focus on profit, grey-areas, AI exploits, unconventional advantages), \"applicable_agents\": [str], \"insight_type\": str (e.g., 'strategic_framework', 'technical_exploit_idea', 'market_analysis_method'), \"relevance_score\": float (0.0-1.0), \"potential_profit_impact_rating\": str ('Low'|'Medium'|'High'|'Game-Changer')}}"
                    }
                    analysis_prompt = await self.generate_dynamic_prompt(task_context)
                    synthesized_insights_json = await self._call_llm_with_retry(analysis_prompt, temperature=0.4, max_tokens=2000, is_json_output=True, model=settings.OPENROUTER_MODELS.get("think_synthesize")) # Use powerful model

                    if synthesized_insights_json:
                        try:
                            insights_data = self._parse_llm_json(synthesized_insights_json)
                            if not insights_data or not all(k in insights_data for k in ['summary', 'key_concepts', 'applicable_agents', 'insight_type', 'relevance_score']): raise ValueError("LLM response missing keys for learning material analysis.")
                            insights_data['source_file'] = os.path.basename(file_path)
                            await self.log_knowledge_fragment(
                                agent_source="LearningMaterialLoader", data_type=insights_data.get('insight_type', 'learning_material_summary'),
                                content=insights_data, relevance_score=float(insights_data.get('relevance_score', 0.6)),
                                tags=["learning_material", insights_data.get('insight_type', 'general')] + [f"agent:{a.lower()}" for a in insights_data.get('applicable_agents', [])] + ([insights_data.get("potential_profit_impact_rating").lower()] if insights_data.get("potential_profit_impact_rating") else []),
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
        # ... (Code from v5.8 - unchanged)
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

    async def generate_educational_content(self, topic: str, context: Optional[str] = None) -> Optional[str]:
        # ... (Code from v5.8 - unchanged)
        self.logger.info(f"ThinkTool: Generating educational content for topic: {topic}")
        thinking_process = f"Structured Thinking: Generate Educational Content for '{topic}'. Context: '{context or 'General'}'. Plan: Formulate prompt, call LLM, return cleaned response."
        await self._internal_think(thinking_process)
        task_context = {
            "task": "Generate Educational Content", "topic": topic, "context": context or "General understanding",
            "desired_output_format": "ONLY the explanation text, suitable for direct display. Start directly with explanation. Assume intelligent user, non-expert. Avoid/explain jargon. Focus on 'why' & relevance to agency goals (profit, exploitation, AI-native thinking)."
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        llm_model_pref = settings.OPENROUTER_MODELS.get('think_user_education')
        explanation = await self._call_llm_with_retry(prompt, temperature=0.6, max_tokens=1000, is_json_output=False, model=llm_model_pref)
        if explanation: self.logger.info(f"Successfully generated educational content for topic: {topic}")
        else: self.logger.error(f"Failed to generate educational content for topic: {topic} (LLM error).")
        return explanation

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(aiohttp.ClientError))
    async def call_clay_api(self, endpoint: str, data: Dict[str, Any], custom_metadata: Optional[Dict[str,Any]] = None) -> Dict[str, Any]: # Added custom_metadata
        api_key = self.config.get_secret("CLAY_API_KEY")
        if not api_key: self.logger.error("Clay.com API key not found."); return {"status": "failure", "message": "Clay API key not configured."}
        if not endpoint.startswith('/'): endpoint = '/' + endpoint
        clay_url = f"https://api.clay.com{endpoint}"; headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
        
        # Include custom metadata if provided, for tracking/correlation
        payload = data.copy()
        if custom_metadata:
            payload['metadata'] = payload.get('metadata', {})
            payload['metadata'].update(custom_metadata)

        await self._internal_think(f"Calling Clay API: {endpoint}", details=payload)
        await self.log_operation('debug', f"Calling Clay API endpoint: {endpoint} with metadata: {custom_metadata}")
        estimated_cost = 0.01
        if "enrichment/person" in endpoint:
            estimated_cost = 0.05
        elif "enrichment/company" in endpoint:
            estimated_cost = 0.0
        try:
            timeout = aiohttp.ClientTimeout(total=90)
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                async with session.post(clay_url, json=payload) as response: # Use payload with metadata
                    response_status = response.status
                    try: response_data = await response.json(content_type=None)
                    except Exception: response_data = await response.text()
                    if 200 <= response_status < 300:
                        self.logger.info(f"Clay API call to {endpoint} successful (Status: {response_status}).")
                        if hasattr(self.orchestrator, 'report_expense'): await self.orchestrator.report_expense(self.AGENT_NAME, estimated_cost, "API_Clay", f"Clay API Call: {endpoint}")
                        return {"status": "success", "data": response_data, "request_payload": payload} # Return request_payload
                    else: self.logger.error(f"Clay API call to {endpoint} failed. Status: {response_status}, Response: {str(response_data)[:500]}..."); return {"status": "failure", "message": f"Clay API Error (Status {response_status})", "details": response_data, "request_payload": payload}
        except asyncio.TimeoutError: self.logger.error(f"Timeout calling Clay API: {endpoint}"); return {"status": "error", "message": f"Clay API call timed out", "request_payload": payload}
        except aiohttp.ClientError as e: self.logger.error(f"Network error calling Clay API {endpoint}: {e}"); raise
        except Exception as e: self.logger.error(f"Unexpected error during Clay API call to {endpoint}: {e}", exc_info=True); return {"status": "error", "message": f"Clay API call exception: {e}", "request_payload": payload}

    async def log_knowledge_fragment(self, agent_source: str, data_type: str, content: Union[str, dict], relevance_score: float = 0.5, tags: Optional[List[str]] = None, related_client_id: Optional[int] = None, source_reference: Optional[str] = None, related_directive_id: Optional[int] = None) -> Optional[KnowledgeFragment]: # Added related_directive_id
        # ... (Code from v5.8 - unchanged, but ensure related_directive_id is handled in model and DB) ...
        if not self.session_maker: self.logger.error("DB session_maker not available."); return None
        try:
            if isinstance(content, dict): content_str = json.dumps(content, sort_keys=True)
            elif isinstance(content, str): content_str = content
            else: raise TypeError(f"Invalid content type: {type(content)}")
            tags_list = sorted(list(set(tags))) if tags else []; tags_str = json.dumps(tags_list) if tags_list else None
            content_hash = hashlib.sha256(content_str.encode('utf-8')).hexdigest(); now_ts = datetime.now(timezone.utc)
            fragment = None
            async with self.session_maker() as session:
                async with session.begin():
                    stmt_check = select(KnowledgeFragment.id).where(KnowledgeFragment.item_hash == content_hash).limit(1)
                    existing_id = (await session.execute(stmt_check)).scalar_one_or_none()
                    if existing_id:
                        self.logger.debug(f"KF hash {content_hash[:8]} exists (ID: {existing_id}). Updating last_accessed_ts.")
                        stmt_update = update(KnowledgeFragment).where(KnowledgeFragment.id == existing_id).values(last_accessed_ts=now_ts)
                        await session.execute(stmt_update); return None
                    else:
                        fragment = KnowledgeFragment(agent_source=agent_source, timestamp=now_ts, last_accessed_ts=now_ts, data_type=data_type, content=content_str, item_hash=content_hash, relevance_score=relevance_score, tags=tags_str, related_client_id=related_client_id, source_reference=source_reference, related_directive_id=related_directive_id)
                        session.add(fragment)
                if fragment:
                    await session.refresh(fragment)
                    self.logger.info(f"Logged KnowledgeFragment: ID={fragment.id}, Hash={content_hash[:8]}..., Type={data_type}, Source={agent_source}")
                    return fragment
                else: return None
        except (SQLAlchemyError, TypeError) as e: self.logger.error(f"Error logging KF: {e}", exc_info=True); await self._report_error(f"Error logging KF: {e}"); return None
        except Exception as e: self.logger.error(f"Unexpected error logging KF: {e}", exc_info=True); return None


    async def query_knowledge_base(self, data_types: Optional[List[str]] = None, tags: Optional[List[str]] = None, min_relevance: float = 0.0, time_window: Optional[timedelta] = None, limit: int = 100, related_client_id: Optional[int] = None, content_query: Optional[str] = None, order_by: Optional[str]="default") -> List[KnowledgeFragment]: # Added order_by
        # ... (Code from v5.8 - with order_by enhancement) ...
        if not self.session_maker: self.logger.error("DB session_maker not available."); return []
        fragments = []; fragment_ids = []
        try:
            async with self.session_maker() as session:
                stmt = select(KnowledgeFragment)
                if data_types: stmt = stmt.where(KnowledgeFragment.data_type.in_(data_types))
                if min_relevance > 0.0: stmt = stmt.where(KnowledgeFragment.relevance_score >= min_relevance)
                if related_client_id is not None: stmt = stmt.where(KnowledgeFragment.related_client_id == related_client_id)
                if time_window: stmt = stmt.where(KnowledgeFragment.timestamp >= (datetime.now(timezone.utc) - time_window))
                if content_query: stmt = stmt.where(KnowledgeFragment.content.ilike(f'%{content_query}%'))
                if tags: tag_conditions = [KnowledgeFragment.tags.like(f'%"{tag}"%') for tag in tags]; stmt = stmt.where(or_(*tag_conditions))
                
                order_by_clause = [desc(KnowledgeFragment.relevance_score), desc(KnowledgeFragment.timestamp)] # Default order
                if order_by == "timestamp_desc": order_by_clause = [desc(KnowledgeFragment.timestamp), desc(KnowledgeFragment.relevance_score)]
                elif order_by == "timestamp_asc": order_by_clause = [asc(KnowledgeFragment.timestamp), desc(KnowledgeFragment.relevance_score)]
                elif order_by == "last_accessed_desc": order_by_clause = [desc(KnowledgeFragment.last_accessed_ts), desc(KnowledgeFragment.relevance_score)]

                stmt_ids = stmt.with_only_columns(KnowledgeFragment.id).order_by(*order_by_clause).limit(limit)
                fragment_ids = (await session.execute(stmt_ids)).scalars().all()

                if not fragment_ids: return []
                stmt_final = select(KnowledgeFragment).where(KnowledgeFragment.id.in_(fragment_ids)).order_by(*order_by_clause) # Apply same order
                fragments = list((await session.execute(stmt_final)).scalars().all())

                if fragment_ids:
                    async def update_access_time():
                        try:
                            async with self.session_maker() as update_session:
                                 async with update_session.begin():
                                    await update_session.execute(update(KnowledgeFragment).where(KnowledgeFragment.id.in_(fragment_ids)).values(last_accessed_ts=datetime.now(timezone.utc)))
                        except Exception as update_err: self.logger.error(f"Failed update last_accessed_ts: {update_err}")
                    asyncio.create_task(update_access_time())
                self.logger.debug(f"KB query returned {len(fragments)} fragments (ordered by {order_by}).")
        except Exception as e: self.logger.error(f"Error querying KB: {e}", exc_info=True)
        return fragments

    async def log_learned_pattern(self, pattern_description: str, supporting_fragment_ids: List[int], confidence_score: float, implications: str, tags: Optional[List[str]] = None, pattern_type: str = "observational", potential_exploit_details: Optional[str] = None) -> Optional[LearnedPattern]: # Added pattern_type, exploit_details
        # ... (Code from v5.8 - with new fields) ...
        if not self.session_maker: self.logger.error("DB session_maker not available."); return None
        try:
            fragment_ids_str = json.dumps(sorted(list(set(supporting_fragment_ids)))); tags_list = sorted(list(set(tags))) if tags else []; tags_str = json.dumps(tags_list) if tags_list else None
            pattern = LearnedPattern(
                timestamp=datetime.now(timezone.utc), pattern_description=pattern_description,
                supporting_fragment_ids=fragment_ids_str, confidence_score=confidence_score,
                implications=implications, tags=tags_str, status='active',
                pattern_type=pattern_type, potential_exploit_details=potential_exploit_details
            )
            async with self.session_maker() as session:
                async with session.begin(): session.add(pattern)
                await session.refresh(pattern)
                self.logger.info(f"Logged LearnedPattern: ID={pattern.id}, Type={pattern_type}, Confidence={confidence_score:.2f}")
                return pattern
        except Exception as e: self.logger.error(f"Error logging LearnedPattern: {e}", exc_info=True); return None

    async def get_latest_patterns(self, tags: Optional[List[str]] = None, min_confidence: float = 0.7, limit: int = 10, pattern_type: Optional[str] = None) -> List[LearnedPattern]: # Added pattern_type filter
        # ... (Code from v5.8 - with pattern_type filter) ...
        if not self.session_maker: self.logger.error("DB session_maker not available."); return []
        patterns = []
        try:
            async with self.session_maker() as session:
                stmt = select(LearnedPattern).where(LearnedPattern.confidence_score >= min_confidence, LearnedPattern.status == 'active')
                if tags: tag_conditions = [LearnedPattern.tags.like(f'%"{tag}"%') for tag in tags]; stmt = stmt.where(or_(*tag_conditions))
                if pattern_type: stmt = stmt.where(LearnedPattern.pattern_type == pattern_type)
                stmt = stmt.order_by(desc(LearnedPattern.timestamp)).limit(limit)
                patterns = list((await session.execute(stmt)).scalars().all())
                self.logger.debug(f"Fetched {len(patterns)} learned patterns (type: {pattern_type or 'any'}, min_conf={min_confidence}).")
        except Exception as e: self.logger.error(f"Error getting latest patterns: {e}", exc_info=True)
        return patterns

    async def purge_old_knowledge(self, days_threshold: Optional[int] = None, data_type_to_preserve: Optional[List[str]] = None): # Added preserve option
        # ... (Code from v5.8 - with preserve option) ...
        if not self.session_maker: self.logger.error("DB session_maker not available."); return
        threshold = days_threshold if days_threshold is not None else int(self.config.get("DATA_PURGE_DAYS_THRESHOLD", 90))
        if threshold <= 0: self.logger.warning("Invalid days_threshold for purge."); return
        purge_cutoff_date = datetime.now(timezone.utc) - timedelta(days=threshold)
        self.logger.info(f"Purging knowledge fragments last accessed before {purge_cutoff_date.isoformat()} (Preserving: {data_type_to_preserve})...")
        deleted_count = 0
        try:
            async with self.session_maker() as session:
                async with session.begin():
                    stmt = delete(KnowledgeFragment).where(KnowledgeFragment.last_accessed_ts < purge_cutoff_date)
                    if data_type_to_preserve:
                        stmt = stmt.where(KnowledgeFragment.data_type.notin_(data_type_to_preserve))
                    result = await session.execute(stmt); deleted_count = result.rowcount
            self.logger.info(f"Successfully purged {deleted_count} old knowledge fragments.")
            if deleted_count > 0 and self.orchestrator and hasattr(self.orchestrator, 'send_notification'):
                 await self.orchestrator.send_notification("Data Purge Completed", f"Purged {deleted_count} knowledge fragments older than {threshold} days.")
        except Exception as e: self.logger.error(f"Error purging old knowledge: {e}", exc_info=True)

    async def handle_feedback(self, insights_data: Dict[str, Dict[str, Any]]):
        # ... (Code from v5.8 - unchanged) ...
        self.logger.info(f"ThinkTool received feedback insights from {len(insights_data)} agents.")
        for agent_name, agent_feedback in insights_data.items():
            if isinstance(agent_feedback, dict):
                try:
                    tags = ["feedback", agent_name.lower()]; is_error = agent_feedback.get("status") == "error" or agent_feedback.get("errors_encountered_session", 0) > 0
                    if is_error: tags.append("error")
                    await self.log_knowledge_fragment(agent_source=agent_name, data_type="AgentFeedbackRaw", content=agent_feedback, tags=tags, relevance_score=0.5)
                except Exception as e: self.logger.error(f"Error logging raw feedback for {agent_name}: {e}", exc_info=True)
        feedback_summary = json.dumps(insights_data, indent=2, default=str, ensure_ascii=False)[:6000]
        task_context = {"task": "Analyze Agent Feedback", "feedback_data": feedback_summary, "desired_output_format": "JSON ONLY: {{\"analysis_summary\": str, \"critical_issues_found\": [str], \"key_successes_noted\": [str], \"proposed_directives\": [{{...}}], \"prompts_to_critique\": [str], \"insights_to_log\": [{{...}}]}}"}
        analysis_prompt = await self.generate_dynamic_prompt(task_context)
        llm_model_pref = settings.OPENROUTER_MODELS.get('think_synthesize')
        analysis_json = await self._call_llm_with_retry(analysis_prompt, temperature=0.6, max_tokens=2500, is_json_output=True, model=llm_model_pref)
        if analysis_json:
            try:
                analysis_result = self._parse_llm_json(analysis_json)
                if not analysis_result: raise ValueError("Failed to parse analysis JSON.")
                self.logger.info(f"Feedback analysis complete. Summary: {analysis_result.get('analysis_summary', 'N/A')}")
                async with self.session_maker() as session:
                     async with session.begin():
                         for d_data in analysis_result.get('proposed_directives', []):
                             if isinstance(d_data, dict) and all(k in d_data for k in ['target_agent', 'directive_type', 'content', 'priority']):
                                 dir_content = json.dumps(d_data['content']) if isinstance(d_data['content'], dict) else d_data['content']
                                 directive = StrategicDirective(source="ThinkToolFeedback", timestamp=datetime.now(timezone.utc), status='pending', content=dir_content, **{k:v for k,v in d_data.items() if k != 'content'})
                                 session.add(directive)
                for prompt_id in analysis_result.get('prompts_to_critique', []):
                    if isinstance(prompt_id, str):
                        try: agent_name, prompt_key = prompt_id.split('/', 1); asyncio.create_task(self.self_critique_prompt(agent_name, prompt_key, f"Feedback analysis suggested issues: {analysis_result.get('analysis_summary', 'N/A')}"))
                        except: pass # Ignore errors in splitting/scheduling critique
                for frag_data in analysis_result.get('insights_to_log', []):
                     if isinstance(frag_data, dict) and all(k in frag_data for k in ['data_type', 'content']): await self.log_knowledge_fragment(agent_source="ThinkToolFeedback", data_type=frag_data['data_type'], content=frag_data['content'], tags=frag_data.get('tags', ['feedback_insight']), relevance_score=frag_data.get('relevance', 0.6))
            except Exception as e: self.logger.error(f"Error processing feedback analysis result: {e}", exc_info=True)
        else: self.logger.error("Feedback analysis failed (LLM error).")

    async def get_prompt(self, agent_name: str, prompt_key: str) -> Optional[str]:
        # ... (Code from v5.8 - unchanged) ...
        if not self.session_maker: self.logger.error("DB session_maker not available."); return None
        try:
            async with self.session_maker() as session:
                stmt = select(PromptTemplate.content).where(PromptTemplate.agent_name == agent_name, PromptTemplate.prompt_key == prompt_key, PromptTemplate.is_active == True).order_by(desc(PromptTemplate.version)).limit(1)
                return (await session.execute(stmt)).scalar_one_or_none()
        except Exception as e: self.logger.error(f"Error getting prompt {agent_name}/{prompt_key}: {e}"); return None

    async def update_prompt(self, agent_name: str, prompt_key: str, new_content: str, author_agent: str = "ThinkTool", critique_summary: Optional[str] = None) -> Optional[PromptTemplate]: # Added critique_summary
        # ... (Code from v5.8 - with critique_summary logging) ...
        if not self.session_maker: self.logger.error("DB session_maker not available."); return None
        new_version = 1; new_template = None
        try:
            async with self.session_maker() as session:
                async with session.begin():
                    stmt_current = select(PromptTemplate.id, PromptTemplate.version).where(PromptTemplate.agent_name == agent_name, PromptTemplate.prompt_key == prompt_key, PromptTemplate.is_active == True).order_by(desc(PromptTemplate.version)).limit(1).with_for_update()
                    current_active_row = (await session.execute(stmt_current)).fetchone()
                    if current_active_row:
                        current_active_id, current_version = current_active_row; new_version = current_version + 1
                        stmt_deactivate = update(PromptTemplate).where(PromptTemplate.id == current_active_id).values(is_active=False)
                        await session.execute(stmt_deactivate)
                    new_template = PromptTemplate(agent_name=agent_name, prompt_key=prompt_key, version=new_version, content=new_content, is_active=True, author_agent=author_agent, last_updated=datetime.now(timezone.utc), notes=critique_summary) # Store critique in notes
                    session.add(new_template)
                await session.refresh(new_template)
                self.logger.info(f"Created and activated new prompt v{new_version} for {agent_name}/{prompt_key}")
                return new_template
        except Exception as e: self.logger.error(f"Error updating prompt {agent_name}/{prompt_key}: {e}"); return None

    async def reflect_on_action(self, context: str, agent_name: str, task_description: str, proposed_plan_summary: Optional[str] = None) -> dict: # Added proposed_plan_summary
        # ... (Code from v5.8 - enhanced prompt) ...
        self.logger.debug(f"Starting reflection for {agent_name} on task: {task_description}")
        kb_context = ""
        try:
            active_directives = await self.get_active_directives(target_agent=agent_name if agent_name != self.AGENT_NAME else None, limit=5) # Get general if ThinkTool reflects on own plan
            task_keywords = [w for w in re.findall(r'\b\w{4,}\b', task_description.lower()) if w not in ['task', 'agent', 'perform', 'execute', 'action', 'plan']]
            query_tags = [agent_name.lower(), 'strategy', 'feedback', 'exploit', 'risk_assessment'] + task_keywords[:2]
            relevant_fragments = await self.query_knowledge_base(tags=list(set(query_tags)), limit=7, time_window=timedelta(days=30))
            relevant_patterns = await self.get_latest_patterns(tags=[agent_name.lower(), 'exploit', 'successful_tactic'], limit=3)
            if active_directives: kb_context += "\n\n**Active Directives (Consider if plan aligns/conflicts):**\n" + "\n".join([f"- ID {d.id} (Prio {d.priority}): {d.content[:100]}..." for d in active_directives])
            if relevant_fragments: kb_context += "\n\n**Recent Relevant Knowledge:**\n" + "\n".join([f"- {f.data_type} (ID {f.id}, Rel: {f.relevance_score:.2f}): {f.content[:80]}..." for f in relevant_fragments])
            if relevant_patterns: kb_context += "\n\n**Relevant Learned Patterns/Exploits:**\n" + "\n".join([f"- {p.pattern_type} (ID {p.id}, Conf: {p.confidence_score:.2f}): {p.pattern_description[:150]}..." for p in relevant_patterns])
        except Exception as e: self.logger.error(f"Error fetching KB context for reflection: {e}"); kb_context = "\n\n**Warning:** Failed KB context retrieval."

        task_context = {
            "task": "Deep Strategic Reflection on Proposed Action/Plan",
            "reflecting_agent": self.AGENT_NAME, # ThinkTool is reflecting
            "agent_proposing_action": agent_name, "proposed_task_description": task_description,
            "agent_provided_context_for_action": context,
            "proposed_plan_summary_if_available": proposed_plan_summary or "N/A",
            "relevant_knowledge_base_context": kb_context or "None available.",
            "core_mandate_reminder": "Maximize Profit & Growth ($10k+/day -> $100M/8mo -> Market Leader), leverage AI-native exploits, manage calculated risks.",
            "desired_output_format": "JSON ONLY: {{\"proceed_as_is\": bool, \"confidence_in_proceeding\": float (0.0-1.0), \"reasoning_for_decision\": \"<Detailed justification>\", \"identified_risks_and_mitigations\": [{\"risk\": str, \"mitigation_suggestion\": str, \"estimated_impact\": str ('Low'|'Medium'|'High')}], \"compliance_flags_critical\": [str] (Only showstoppers), \"alternative_or_enhanced_strategy\": str? (If a significantly more profitable, leveraged, or AI-native approach exists, describe it concisely), \"questions_for_legal_agent\": [str]? (Specific questions if legal/ethical boundaries are unclear for grey-area tactics), \"next_step_recommendation\": str (e.g., 'Proceed with caution', 'Revise plan based on alternative', 'Consult LegalAgent then proceed', 'Halt - Unacceptable Risk/Low ROI'), \"log_reflection_summary_to_kb\": bool (true if significant insights gained)}"
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        llm_model_pref = settings.OPENROUTER_MODELS.get('think_critique') # Use most powerful model for reflection
        reflection_json = await self._call_llm_with_retry(prompt, temperature=0.2, max_tokens=2500, is_json_output=True, model=llm_model_pref)

        if reflection_json:
            try:
                reflection = self._parse_llm_json(reflection_json)
                if not reflection: raise ValueError("Failed parse reflection JSON.")
                # Ensure defaults for key fields
                reflection.setdefault('proceed_as_is', False); reflection.setdefault('confidence_in_proceeding', 0.0)
                reflection.setdefault('reasoning_for_decision', 'LLM analysis failed or incomplete.')
                reflection.setdefault('identified_risks_and_mitigations', [])
                reflection.setdefault('next_step_recommendation', 'Manual review required due to reflection error.')
                self.logger.info(f"Reflection for {agent_name}'s task '{task_description[:50]}...': Proceed={reflection['proceed_as_is']}, Confidence={reflection['confidence_in_proceeding']:.2f}, Risk Assessment: {len(reflection['identified_risks_and_mitigations'])} points. Next Step: {reflection['next_step_recommendation']}")
                if reflection.get('log_reflection_summary_to_kb'):
                    await self.log_knowledge_fragment(
                        agent_source=self.AGENT_NAME, data_type="strategic_reflection_output",
                        content={"task_reflected_on": task_description, "proposing_agent": agent_name, "reflection_outcome": reflection},
                        tags=["reflection", agent_name.lower(), "strategy_review"], relevance_score=0.8
                    )
                return reflection
            except Exception as e: self.logger.error(f"Error processing reflection result: {e}", exc_info=True)
        return {"proceed_as_is": False, "confidence_in_proceeding": 0.0, "reasoning_for_decision": "ThinkTool reflection process failed critically.", "next_step_recommendation": "Halt task, investigate ThinkTool reflection error."}

    async def validate_output(self, output_to_validate: str, validation_criteria: str, agent_name: str, context: str = None) -> dict:
        # ... (Code from v5.8 - unchanged) ...
        self.logger.debug(f"Starting validation for {agent_name}'s output.")
        pattern_context = ""
        try:
            criteria_tags = [w for w in re.findall(r'\b\w{4,}\b', validation_criteria.lower()) if w not in ['validate', 'check', 'ensure']]
            query_tags = [agent_name.lower()] + criteria_tags[:3]
            relevant_patterns = await self.get_latest_patterns(tags=query_tags, limit=5, min_confidence=0.6)
            if relevant_patterns: pattern_context += "\n\n**Relevant Learned Patterns (Consider these):**\n" + "\n".join([f"- ID {p.id}: {p.pattern_description}" for p in relevant_patterns])
        except Exception as e: self.logger.error(f"Error fetching patterns for validation: {e}"); pattern_context = "\n\n**Warning:** Failed pattern retrieval."

        task_context = {
            "task": "Validate Agent Output", "agent_name": agent_name, "agent_context": context or "N/A",
            "output_to_validate": output_to_validate, "validation_criteria": validation_criteria,
            "learned_patterns_context": pattern_context,
            "desired_output_format": "JSON ONLY: {{\"valid\": bool, \"feedback\": \"Concise explanation for validity (pass/fail) referencing specific criteria/patterns/checks.\", \"suggested_fix\": \"If invalid, provide a specific, actionable suggestion for correction.\"}}"
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        llm_model_pref = settings.OPENROUTER_MODELS.get('think_validate')
        validation_json = await self._call_llm_with_retry(prompt, temperature=0.2, max_tokens=800, is_json_output=True, model=llm_model_pref)
        if validation_json:
            try:
                validation = self._parse_llm_json(validation_json)
                if not validation: raise ValueError("Failed parse validation JSON.")
                validation.setdefault('valid', False); validation.setdefault('feedback', 'Validation analysis failed.')
                return validation
            except Exception as e: self.logger.error(f"Error processing validation result: {e}", exc_info=True)
        return {"valid": False, "feedback": "ThinkTool validation failed."}

    async def synthesize_insights_and_strategize(self):
        self.logger.info("ThinkTool L75+: Starting Grand Synthesis & Strategic Exploitation Cycle.")
        await self._internal_think("Grand Synthesis: Query all data sources -> Analyze for deep patterns, AI exploits, market shifts -> Generate high-impact directives & self-improvement tasks.")
        proposed_directives = []
        try:
            # 1. Gather Comprehensive Data
            async with self.session_maker() as session:
                # Recent KFs (broader types, higher limit)
                frags_stmt = select(KnowledgeFragment).where(KnowledgeFragment.last_accessed_ts >= datetime.now(timezone.utc) - timedelta(days=14)).order_by(desc(KnowledgeFragment.relevance_score), desc(KnowledgeFragment.last_accessed_ts)).limit(300)
                recent_fragments = list((await session.execute(frags_stmt)).scalars().all())
                # Active & High-Confidence Patterns (broader types)
                patt_stmt = select(LearnedPattern).where(LearnedPattern.status == 'active', LearnedPattern.confidence_score >= 0.65).order_by(desc(LearnedPattern.confidence_score), desc(LearnedPattern.timestamp)).limit(50)
                recent_patterns = list((await session.execute(patt_stmt)).scalars().all())
                # Performance Data (same as v5.8)
                perf_data = []
                email_perf_stmt = select(EmailLog.status, EmailLog.timestamp, EmailLog.subject).where(EmailLog.timestamp >= datetime.now(timezone.utc) - timedelta(days=7)).limit(500) # Added subject
                perf_data.extend([{"type": "email", "status": r.status, "ts": r.timestamp, "subject_preview": r.subject[:50] if r.subject else None} for r in (await session.execute(email_perf_stmt)).mappings().all()])
                call_perf_stmt = select(CallLog.outcome, CallLog.timestamp, CallLog.duration_seconds).where(CallLog.timestamp >= datetime.now(timezone.utc) - timedelta(days=7)).limit(500) # Added duration
                perf_data.extend([{"type": "call", "outcome": r.outcome, "ts": r.timestamp, "duration": r.duration_seconds} for r in (await session.execute(call_perf_stmt)).mappings().all()])
                inv_perf_stmt = select(Invoice.status, Invoice.amount, Invoice.timestamp).where(Invoice.timestamp >= datetime.now(timezone.utc) - timedelta(days=7)).limit(200)
                perf_data.extend([{"type": "invoice", "status": r.status, "amount": r.amount, "ts": r.timestamp} for r in (await session.execute(inv_perf_stmt)).mappings().all()])
                client_score_stmt = select(Client.engagement_score, Client.industry).where(Client.opt_in == True).order_by(func.random()).limit(200) # Added industry
                perf_data.extend([{"type": "client_score", "score": r.engagement_score, "industry": r.industry} for r in (await session.execute(client_score_stmt)).mappings().all()])

                # Gmail Account Status Check (for needs_new_gmail_accounts)
                gmail_accounts_stmt = select(AccountCredentials.status).where(AccountCredentials.service == 'google.com')
                gmail_statuses = (await session.execute(gmail_accounts_stmt)).scalars().all()
                active_gmail_count = sum(1 for s in gmail_statuses if s == 'active')
                if active_gmail_count < self.min_active_gmail_for_trials:
                    self.logger.info(f"Active Gmail accounts ({active_gmail_count}) below threshold ({self.min_active_gmail_for_trials}). Proposing creation.")
                    proposed_directives.append({
                        "target_agent": "GmailCreatorAgent", "directive_type": "create_gmail_accounts",
                        "content": {"count": self.min_active_gmail_for_trials - active_gmail_count + 1, "identity_profile_hint": "general_trial_user"}, # Request one more than needed
                        "priority": 5 # Higher priority if critically low
                    })

                # Clay Enrichment (same as v5.8)
                clients_needing_enrichment_stmt = select(Client.id, Client.name, Client.source_reference).where(Client.opt_in == True, Client.is_deliverable == True, Client.source_reference.like('%linkedin.com/in/%'), or_(Client.email == None, Client.company == None, Client.job_title == None)).order_by(asc(Client.last_interaction)).limit(10) # Reduced limit for more frequent, smaller batches
                clients_for_clay = (await session.execute(clients_needing_enrichment_stmt)).mappings().all()
                for client_data in clients_for_clay:
                    linkedin_url = client_data.get('source_reference')
                    if linkedin_url:
                        proposed_directives.append({
                            "target_agent": "ThinkTool", "directive_type": "execute_clay_call",
                            "content": {"endpoint": self.clay_endpoints.get("enrich_person"), "data": {"linkedin_url": linkedin_url}, "context": {"client_id": client_data['id'], "reason": "Initial enrichment for outreach"}, "custom_metadata_for_clay": {"client_id": client_data['id'], "directive_source": "ThinkToolSynthesis"}}, "priority": 6 })
            
            # 2. Prepare Context for LLM Synthesis
            fragments_summary = [{"id": f.id, "type": f.data_type, "src": f.agent_source, "preview": (f.content if isinstance(f.content, str) else json.dumps(f.content))[:70]+"...", "relevance": f.relevance_score} for f in recent_fragments[:30]] # More fragments, include relevance
            patterns_summary = [{"id": p.id, "desc": p.pattern_description, "conf": p.confidence_score, "type": p.pattern_type, "exploit_details": p.potential_exploit_details} for p in recent_patterns[:15]] # More patterns, include type/exploit
            perf_summary_str = f"Email Statuses (7d): {Counter(d['status'] for d in perf_data if d['type']=='email')}. Call Outcomes (7d): {Counter(d['outcome'] for d in perf_data if d['type']=='call')}. Paid Invoices (7d): ${sum(d['amount'] for d in perf_data if d['type']=='invoice' and d['status']=='paid'):.2f}."
            
            task_context = {
                "task": "Grand Strategic Synthesis & AI Exploit Identification",
                "current_system_time_utc": datetime.now(timezone.utc).isoformat(),
                "recent_knowledge_fragments_summary": fragments_summary,
                "active_learned_patterns_summary": patterns_summary,
                "recent_performance_summary_7d": perf_summary_str,
                "auto_generated_clay_directives_count": sum(1 for d in proposed_directives if d.get("directive_type") == "execute_clay_call"),
                "auto_generated_gmail_creation_directives_count": sum(1 for d in proposed_directives if d.get("directive_type") == "create_gmail_accounts"),
                "current_agency_goals_reminder": "Primary: $10k+/day profit, $100M/8mo. Secondary: Market leadership, AI-native operational supremacy, discovery of novel exploits.",
                "desired_output_format": "JSON ONLY: {{\"strategic_assessment_summary\": \"<Overall assessment of current agency trajectory, identifying key leverage points and critical risks>\", \"new_learned_patterns_to_log\": [{\"pattern_description\": str, \"supporting_fragment_ids\": [int], \"confidence_score\": float, \"implications\": str, \"tags\": [str], \"pattern_type\": str ('observational'|'causal'|'exploit_hypothesis'), \"potential_exploit_details\": str? (If exploit_hypothesis, describe the potential exploit)}], \"high_priority_strategic_directives\": [{\"target_agent\": str, \"directive_type\": str (e.g., 'test_new_exploit', 'launch_campaign_variant_x', 'refine_agent_prompt_y'), \"content\": {<details>}, \"priority\": int (1-10, 1=highest), \"estimated_roi_or_impact\": str, \"risk_assessment_summary\": str? (Brief, if high risk)}], \"emerging_opportunities_or_threats\": [{\"type\": \"opportunity\"|\"threat\", \"description\": str, \"suggested_action\": str, \"urgency\": str ('Low'|'Medium'|'High')}], \"self_improvement_suggestions_for_thinktool\": [\"<Specific ideas to improve ThinkTool's own prompting, analysis, or periodic tasks>\"]}}"
            }
            synthesis_prompt = await self.generate_dynamic_prompt(task_context)
            llm_model_pref = settings.OPENROUTER_MODELS.get('think_strategize') # Most powerful model
            synthesis_json = await self._call_llm_with_retry(synthesis_prompt, temperature=0.5, max_tokens=3900, is_json_output=True, model=llm_model_pref) # Max tokens

            if synthesis_json:
                try:
                    synthesis_result = self._parse_llm_json(synthesis_json)
                    if not synthesis_result: raise ValueError("Failed parse synthesis JSON.")
                    self.logger.info(f"ThinkTool Grand Synthesis cycle completed. Assessment: {synthesis_result.get('strategic_assessment_summary', 'N/A')}")
                    # Log new patterns
                    for p_data in synthesis_result.get('new_learned_patterns_to_log', []):
                        if isinstance(p_data, dict): await self.log_learned_pattern(**p_data)
                    # Combine and store directives
                    all_directives_to_store = proposed_directives + synthesis_result.get('high_priority_strategic_directives', [])
                    async with self.session_maker() as session:
                         async with session.begin():
                             for d_data in all_directives_to_store:
                                 if isinstance(d_data, dict) and all(k in d_data for k in ['target_agent', 'directive_type', 'content', 'priority']):
                                     dir_content_str = json.dumps(d_data['content']) if isinstance(d_data['content'], dict) else str(d_data['content'])
                                     # Add estimated_roi and risk_assessment if present
                                     new_directive = StrategicDirective(
                                         source="ThinkToolL75Synthesis", timestamp=datetime.now(timezone.utc), status='pending',
                                         content=dir_content_str, target_agent=d_data['target_agent'], directive_type=d_data['directive_type'],
                                         priority=d_data['priority'],
                                         notes=f"Est. ROI/Impact: {d_data.get('estimated_roi_or_impact', 'N/A')}. Risk: {d_data.get('risk_assessment_summary', 'N/A')}"
                                     )
                                     session.add(new_directive)
                                     self.logger.info(f"Storing L75 Directive for {d_data['target_agent']}: {d_data['directive_type']} (Prio: {d_data['priority']})")
                    # Log opportunities/threats
                    for ot_data in synthesis_result.get('emerging_opportunities_or_threats', []):
                         if isinstance(ot_data, dict) and 'description' in ot_data:
                             await self.log_knowledge_fragment(
                                 agent_source="ThinkToolL75Synthesis", data_type=f"{ot_data.get('type','unknown')}_signal",
                                 content=ot_data, tags=["strategic_signal", ot_data.get('type','unknown'), ot_data.get('urgency','low')],
                                 relevance_score=0.85 if ot_data.get('urgency') == 'High' else 0.7)
                    # Log self-improvement suggestions for ThinkTool
                    for self_imp_suggestion in synthesis_result.get('self_improvement_suggestions_for_thinktool', []):
                        await self.log_knowledge_fragment(
                            agent_source=self.AGENT_NAME, data_type="thinktool_self_improvement_idea",
                            content={"suggestion": self_imp_suggestion, "source": "L75Synthesis"},
                            tags=["meta_learning", "thinktool_dev"], relevance_score=0.9
                        )
                except Exception as e: self.logger.error(f"Error processing L75 synthesis result: {e}", exc_info=True)
            else: self.logger.error("L75 Synthesis analysis failed (LLM error).")
        except Exception as e: self.logger.error(f"Error during L75 synthesis cycle: {e}", exc_info=True); await self._report_error(f"L75 Synthesis cycle failed: {e}")
        self.last_synthesis_run = datetime.now(timezone.utc)


    async def technology_radar(self):
        self.logger.info("ThinkTool L75+: Starting Advanced Technology & Exploit Radar cycle.")
        await self._internal_think("Tech & Exploit Radar: Define advanced search vectors -> Delegate multi-source search (web, forums, code repos if possible) -> Analyze for actionable exploits, new AI models, platform shifts -> Log findings & generate high-priority test directives.")
        try:
            search_vectors = [
                "undocumented API changes major social platforms (X, TikTok, Instagram, LinkedIn) last 7 days",
                "newly released open source multimodal LLMs with commercial-friendly licenses",
                "discussions on bypassing AI content detection methods (for defensive research)",
                "exploitable bugs or logic flaws in popular marketing automation SaaS tools",
                "novel techniques for AI-driven persona generation or psychographic targeting",
                "emerging AI tools for hyper-realistic synthetic media generation (voice, video) with API access",
                "latest research papers on 'LLM agent reasoning' or 'autonomous AI planning' with code",
                "unconventional growth hacking forums discussing new automation exploits"
            ]
            search_query = f"Urgent Scan: {random.choice(search_vectors)} - focus on actionable, potentially grey-area intelligence for immediate competitive advantage or profit generation. Include code snippets or PoC links if found."
            
            # Delegate to BrowsingAgent with instructions to try deeper web if possible
            browsing_task_params = {
                "action": "deep_web_search_and_summarize", # Assume BrowsingAgent has an enhanced action
                "query": search_query, "num_results_per_source": 5,
                "sources_to_try": ["google", "github_search", "arxiv_search", "specific_forum_search_if_known_and_safe"], # Conceptual
                "max_depth_per_link": 2 # How many links deep to follow from initial results
            }
            if not (self.orchestrator and hasattr(self.orchestrator, 'delegate_task') and "BrowsingAgent" in self.orchestrator.agents):
                self.logger.error("Radar: BrowsingAgent unavailable for deep web search."); return

            search_result_data = await self.orchestrator.delegate_task("BrowsingAgent", browsing_task_params)
            search_results_summary = search_result_data.get("summary", "") if search_result_data and search_result_data.get("status") == "success" else None
            
            if not search_results_summary: self.logger.info("Radar: No significant findings from deep web search."); return

            task_context = {
                "task": "Analyze Advanced Tech & Exploit Scouting Report",
                "report_summary_and_key_findings": search_results_summary, # Expecting more structured data from BrowsingAgent now
                "analysis_focus": "Identify immediately testable exploits, novel AI tools for integration, or critical platform shifts. Assess ROI, risk (realistic consequences), technical feasibility, and speed of implementation. Prioritize game-changing opportunities.",
                "desired_output_format": "JSON ONLY: {{\"actionable_intelligence_items\": [ {{ \"item_title\": str, \"type\": str ('Exploit'|'New_AI_Tool'|'Platform_Shift'|'Research_Breakthrough'), \"description_summary\": str, \"potential_impact_on_agency\": str, \"estimated_roi_category\": str ('Low'|'Medium'|'High'|'Transformative'), \"risk_level_realistic\": str ('Low'|'Medium'|'High'|'Extreme - Legal Consult Required'), \"technical_feasibility\": str ('Easy'|'Medium'|'Hard'|'Research_Project'), \"urgency_for_testing\": str ('Low'|'Medium'|'High'|'Immediate'), \"suggested_test_directive_summary\": str? (Brief idea for a test directive) }} ], \"overall_radar_assessment\": \"Brief summary of most critical findings.\"}}"
            }
            analysis_prompt = await self.generate_dynamic_prompt(task_context)
            llm_model_pref = settings.OPENROUTER_MODELS.get('think_radar') # A model good at technical analysis
            analysis_json = await self._call_llm_with_retry(analysis_prompt, temperature=0.3, max_tokens=3000, is_json_output=True, model=llm_model_pref)

            if analysis_json:
                try:
                    analysis_result = self._parse_llm_json(analysis_json)
                    if not analysis_result or not analysis_result.get('actionable_intelligence_items'): raise ValueError("Failed parse radar analysis JSON or missing items.")
                    self.logger.info(f"Radar L75+ analysis complete. Found {len(analysis_result.get('actionable_intelligence_items', []))} actionable items. Assessment: {analysis_result.get('overall_radar_assessment')}")
                    async with self.session_maker() as session:
                        async with session.begin():
                            for item in analysis_result.get('actionable_intelligence_items', []):
                                if not isinstance(item, dict) or not item.get("item_title"): continue
                                kf_tags = ["tech_radar_L75", item.get("type", "general").lower(), f"urgency_{item.get('urgency_for_testing','low').lower()}", f"risk_{item.get('risk_level_realistic','unknown').lower()}"]
                                await self.log_knowledge_fragment(agent_source="ThinkToolL75Radar", data_type=item.get("type", "tech_intelligence"), content=item, tags=kf_tags, relevance_score=0.8 if item.get('urgency_for_testing') == 'High' else 0.7)
                                if item.get("suggested_test_directive_summary") and item.get("urgency_for_testing") in ['High', 'Immediate']:
                                    directive_content = {
                                        "intelligence_item_title": item.get("item_title"),
                                        "description": item.get("description_summary"),
                                        "type": item.get("type"),
                                        "suggested_test": item.get("suggested_test_directive_summary"),
                                        "risk_assessment": item.get("risk_level_realistic"),
                                        "source_radar_query": search_query # Link back to the radar query
                                    }
                                    # Target agent for testing could be ThinkTool itself (to plan further) or a specialized agent
                                    target_test_agent = "ThinkTool" if "exploit" in item.get("type","").lower() else "Orchestrator" # Default for general tool testing
                                    directive = StrategicDirective(
                                        source="ThinkToolL75Radar", timestamp=datetime.now(timezone.utc),
                                        target_agent=target_test_agent, directive_type="evaluate_or_test_new_intelligence",
                                        content=json.dumps(directive_content), priority=3 if item.get("urgency_for_testing") == 'Immediate' else 5,
                                        status='pending'
                                    )
                                    session.add(directive)
                                    self.logger.info(f"Generated L75 Radar Test Directive for: {item.get('item_title', 'N/A')}")
                except Exception as e: self.logger.error(f"Radar L75+: Error processing analysis result: {e}", exc_info=True)
            else: self.logger.error("Radar L75+: Analysis failed (LLM error).")
        except Exception as e: self.logger.error(f"Error during L75+ technology radar cycle: {e}", exc_info=True); await self._report_error(f"L75+ Technology radar cycle failed: {e}")
        self.last_radar_run = datetime.now(timezone.utc)


    async def self_critique_prompt(self, agent_name: str, prompt_key: str, feedback_context: str):
        # ... (Code from v5.8 - enhanced prompt for strategic alignment) ...
        self.logger.info(f"Starting self-critique for prompt: {agent_name}/{prompt_key}")
        await self._internal_think(f"Self-Critique Prompt {agent_name}/{prompt_key}. Plan: Fetch prompt -> Format critique prompt (focus on profit/exploit angle, AI-native thinking) -> Call LLM -> Parse -> Update prompt -> Generate test directive.")
        try:
            current_prompt = await self.get_prompt(agent_name, prompt_key)
            if not current_prompt: self.logger.error(f"Critique: Cannot find active prompt {agent_name}/{prompt_key}."); return

            task_context = {
                "task": "Critique and Rewrite Agent Prompt for Max Profit/Exploit Potential & AI-Native Operation",
                "agent_name_whose_prompt_is_critiqued": agent_name, "prompt_key_being_critiqued": prompt_key,
                "triggering_feedback_or_context": feedback_context,
                "current_prompt_text_to_be_critiqued": current_prompt,
                "thinktool_core_mandate_reminder": "Maximize Profit & Growth, leverage AI-native exploits, manage calculated risks.",
                "desired_output_format": "JSON ONLY: {{ \"critique_summary\": \"Detailed critique focusing on alignment with profit mandate, clarity for grey-area thinking, enabling AI-native exploits, and overall strategic effectiveness. Identify specific weaknesses.\", \"suggested_improvements_for_prompt\": [\"List 3-5 concrete, actionable improvements.\"], \"fully_rewritten_and_enhanced_prompt\": \"<Complete rewritten prompt text, significantly enhanced for strategic advantage, AI-native operation, and exploit potential. Ensure it's robust and guides the agent effectively.>\" }}"
            }
            critique_prompt = await self.generate_dynamic_prompt(task_context)
            llm_model_pref = settings.OPENROUTER_MODELS.get('think_critique') # Use most powerful model
            critique_json = await self._call_llm_with_retry(critique_prompt, temperature=0.4, max_tokens=3800, is_json_output=True, model=llm_model_pref) # Max tokens

            if critique_json:
                try:
                    critique_result = self._parse_llm_json(critique_json)
                    if not critique_result: raise ValueError("Failed parse critique JSON.")
                    improved_prompt = critique_result.get('fully_rewritten_and_enhanced_prompt')
                    critique_text = critique_result.get('critique_summary')
                    if improved_prompt and isinstance(improved_prompt, str) and improved_prompt.strip():
                        self.logger.info(f"Critique generated improved prompt for {agent_name}/{prompt_key}. Critique Summary: {critique_text}")
                        new_template = await self.update_prompt(agent_name, prompt_key, improved_prompt, author_agent="ThinkToolL75Critique", critique_summary=critique_text) # Log critique summary
                        if new_template:
                            async with self.session_maker() as session:
                                async with session.begin():
                                    directive = StrategicDirective(source="ThinkToolL75Critique", timestamp=datetime.now(timezone.utc), target_agent=agent_name, directive_type="test_newly_critiqued_prompt_variation", content=json.dumps({"prompt_key": prompt_key, "new_version": new_template.version, "critique_summary": critique_text, "previous_prompt_version": new_template.version -1 }), priority=4, status='pending') # Higher priority
                                    session.add(directive)
                                self.logger.info(f"Generated directive to test new prompt v{new_template.version} for {agent_name}/{prompt_key}")
                    else: self.logger.warning(f"Critique for {agent_name}/{prompt_key} did not produce valid improved prompt.")
                except Exception as e: self.logger.error(f"Critique: Error processing result: {e}", exc_info=True)
            else: self.logger.error(f"Critique: Failed get critique/rewrite from LLM for {agent_name}/{prompt_key}.")
        except Exception as e: self.logger.error(f"Error during self-critique for {agent_name}/{prompt_key}: {e}", exc_info=True); await self._report_error(f"Self-critique failed for {agent_name}/{prompt_key}: {e}")

    async def run(self):
        # ... (Code from v5.8 - with added _proactive_market_shift_analysis call) ...
        if self.status == self.STATUS_RUNNING: self.logger.warning("ThinkTool run() called while already running."); return
        self.logger.info(f"ThinkTool v{self.config.get('APP_VERSION', 'Unknown')} starting run loop...")
        self._status = self.STATUS_RUNNING
        synthesis_interval = timedelta(seconds=int(self.config.get("THINKTOOL_SYNTHESIS_INTERVAL_SECONDS", 3600))) # 1 hour
        radar_interval = timedelta(seconds=int(self.config.get("THINKTOOL_RADAR_INTERVAL_SECONDS", 3600 * 3))) # 3 hours
        market_shift_interval = timedelta(seconds=int(self.config.get("THINKTOOL_MARKET_SHIFT_INTERVAL_SECONDS", 3600 * 6))) # 6 hours
        purge_interval = timedelta(seconds=int(self.config.get("DATA_PURGE_INTERVAL_SECONDS", 86400))) # 24 hours
        
        now = datetime.now(timezone.utc)
        self.last_synthesis_run = now - synthesis_interval
        self.last_radar_run = now - radar_interval
        self.last_market_shift_analysis_run = now - market_shift_interval
        self.last_purge_run = now - purge_interval

        while self.status == self.STATUS_RUNNING and not self._stop_event.is_set():
            try:
                current_time = datetime.now(timezone.utc)
                is_approved = getattr(self.orchestrator, 'approved', False) if self.orchestrator else False

                if is_approved:
                    if current_time - self.last_synthesis_run >= synthesis_interval:
                        self.logger.info("ThinkTool: Triggering Grand Synthesis & Strategic Exploitation Cycle.")
                        await self.synthesize_insights_and_strategize() # Calls the L75 version
                        self.last_synthesis_run = current_time
                    if current_time - self.last_radar_run >= radar_interval:
                        self.logger.info("ThinkTool: Triggering Advanced Technology & Exploit Radar cycle.")
                        await self.technology_radar() # Calls the L75 version
                        self.last_radar_run = current_time
                    if current_time - self.last_market_shift_analysis_run >= market_shift_interval:
                        self.logger.info("ThinkTool: Triggering Proactive Market Shift Analysis cycle.")
                        await self._proactive_market_shift_analysis()
                        self.last_market_shift_analysis_run = current_time
                    if current_time - self.last_purge_run >= purge_interval:
                        self.logger.info("ThinkTool: Triggering Data Purge cycle.")
                        await self.purge_old_knowledge(data_type_to_preserve=['learned_pattern', 'strategic_directive_template', 'core_learning_material']) # Preserve key learnings
                        self.last_purge_run = current_time
                else: self.logger.debug("ThinkTool: Orchestrator not approved. Skipping periodic tasks.")
                await asyncio.sleep(60 * 1)
            except asyncio.CancelledError: self.logger.info("ThinkTool run loop cancelled."); break
            except Exception as e: self.logger.critical(f"ThinkTool: CRITICAL error in run loop: {e}", exc_info=True); self._status = self.STATUS_ERROR; await self._report_error(f"Critical run loop error: {e}"); await asyncio.sleep(60 * 15)
        if self.status != self.STATUS_STOPPING: self.status = self.STATUS_STOPPED
        self.logger.info("ThinkTool run loop finished.")

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        # ... (Code from v5.8 - with new action handlers for webhook and market shift analysis) ...
        self._status = self.STATUS_EXECUTING
        action = task_details.get('action')
        result = {"status": "failure", "message": f"Unknown ThinkTool action: {action}"}
        self.logger.info(f"ThinkTool executing task: {action}")
        task_id = task_details.get('id', str(uuid.uuid4()))
        task_details['id'] = task_id

        try:
            # Reflection for critical/complex actions
            if action in ['synthesize_insights_and_strategize', 'initiate_video_generation_workflow', 'plan_ugc_workflow', 'execute_clay_call', 'technology_radar', 'plan_social_media_campaign', '_proactive_market_shift_analysis', 'evaluate_tech_exploit', 'create_directive_from_suggestion']:
                 reflection_context = f"About to execute complex/strategic action: {action}. Task Details: {json.dumps(task_details.get('content', task_details), default=str)[:500]}..."
                 reflection_result = await self.reflect_on_action(reflection_context, self.AGENT_NAME, f"Pre-execution check for {action}", proposed_plan_summary=task_details.get('content',{}).get('plan_summary_for_reflection'))
                 if not reflection_result.get('proceed_as_is', False):
                     self.logger.warning(f"Reflection advised against proceeding with action '{action}'. Reason: {reflection_result.get('reasoning_for_decision')}")
                     if task_details.get('directive_id'): await self.update_directive_status(task_details['directive_id'], 'halted_by_reflection', f"Halted by reflection: {reflection_result.get('reasoning_for_decision')}. Suggestion: {reflection_result.get('alternative_or_enhanced_strategy') or reflection_result.get('next_step_recommendation')}")
                     return {"status": "halted_by_reflection", "message": f"Action halted: {reflection_result.get('reasoning_for_decision')}", "reflection_details": reflection_result}
                 else: self.logger.info(f"Reflection approved proceeding with action '{action}'. Confidence: {reflection_result.get('confidence_in_proceeding')}")

            if action == 'process_clay_webhook_data': # New action for webhook
                content = task_details.get('content', {})
                await self._process_clay_webhook_data(
                    enriched_data=content.get('enriched_data'),
                    original_input_parameters=content.get('original_input_parameters'),
                    source_reference=content.get('source_reference'),
                    clay_run_id=content.get('clay_run_id')
                )
                result = {"status": "success", "message": "Clay webhook data processing initiated."}
            elif action == 'analyze_persistent_service_failure': # New action
                content = task_details.get('content', {})
                await self._analyze_persistent_service_failure(content.get("service"), content.get("failure_count"), content.get("last_error"))
                result = {"status": "success", "message": "Persistent service failure analysis initiated."}
            elif action == 'analyze_and_adapt_creation_strategy': # New action
                content = task_details.get('content', {})
                await self._analyze_and_adapt_creation_strategy(content)
                result = {"status": "success", "message": "Creation strategy analysis and adaptation initiated."}
            elif action == 'assess_initial_account_health': # New action
                content = task_details.get('content', {})
                await self._assess_initial_account_health(content.get("service_filter_list"))
                result = {"status": "success", "message": "Initial account health assessment initiated."}
            elif action == 'flag_account_issue': # New action
                content = task_details.get('content', {})
                await self._flag_account_issue(content.get("account_id"), content.get("issue"), content.get("severity"), content.get("details"))
                result = {"status": "success", "message": f"Account issue for {content.get('account_id')} flagged."}
            elif action == 'create_directive_from_suggestion': # New action
                content = task_details.get('content', {})
                await self._create_directive_from_suggestion(content.get("source_agent"), content.get("suggestion"), content.get("priority", 7))
                result = {"status": "success", "message": "Directive created from suggestion."}
            elif action == 'plan_social_media_campaign': # Now a distinct action
                content = task_details.get('content', {})
                plan_kf_id = await self._plan_social_media_campaign(content)
                if plan_kf_id: result = {"status": "success", "message": "Social media campaign plan generated and stored.", "findings": {"campaign_plan_kf_id": plan_kf_id}}
                else: result = {"status": "failure", "message": "Failed to generate social media campaign plan."}
            # ... (other action handlers from v5.8, ensuring they call the L75 versions of methods if applicable)
            elif action == 'synthesize_insights_and_strategize': await self.synthesize_insights_and_strategize(); result = {"status": "success", "message": "L75 Synthesis and strategy cycle completed."}
            elif action == 'technology_radar': await self.technology_radar(); result = {"status": "success", "message": "L75 Technology radar cycle completed."}
            elif action == 'purge_old_knowledge': await self.purge_old_knowledge(); result = {"status": "success", "message": "Data purge cycle completed."}
            elif action == 'execute_clay_call':
                 params_content = task_details.get('content', {})
                 endpoint = params_content.get('endpoint'); data = params_content.get('data')
                 custom_meta = params_content.get('custom_metadata_for_clay', {"directive_id": task_details.get('directive_id')}) # Pass directive ID
                 source_ref = params_content.get('source_reference'); client_id_ctx = params_content.get('context', {}).get('client_id')
                 original_directive_id = task_details.get('directive_id')
                 if endpoint and data:
                     clay_api_result = await self.call_clay_api(endpoint=endpoint, data=data, custom_metadata=custom_meta)
                     # Process result immediately, no longer a separate task for this flow
                     await self._process_clay_result(clay_api_result, original_directive_id, source_ref, client_id_ctx, is_webhook=False)
                     result = {"status": "success", "message": "Clay API call executed and result processed."}
                 else:
                     result = {"status": "failure", "message": "Missing 'endpoint' or 'data' for execute_clay_call."}
                     if original_directive_id: await self.update_directive_status(original_directive_id, 'failed', 'Missing endpoint/data')
            # Note: process_clay_result is now primarily for direct calls, webhook has its own handler.
            # For other actions like 'reflect_on_action', 'validate_output', etc., ensure they use the L75 enhanced methods.
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
        # ... (Code from v5.8 - unchanged, but ensure it calls the L75 generate_dynamic_prompt) ...
        self.logger.info(f"Planning detailed video generation workflow with params: {params}")
        await self._internal_think("Planning detailed video workflow: Descript UI + AIStudio Images", details=params)
        kb_context_frags = await self.query_knowledge_base(data_types=['tool_usage_guide', 'workflow_step', 'asset_location'], tags=['descript', 'aistudio', 'video_generation', 'base_video', 'image_generation'], limit=10)
        kb_context_str = "\n".join([f"- {f.data_type} (ID {f.id}): {f.content[:150]}..." for f in kb_context_frags])
        client_id = params.get("client_id"); video_topic_keywords = params.get("topic_keywords", [])
        existing_video_fragments_list: List[KnowledgeFragment] = []
        if client_id: existing_video_fragments_list = await self.query_knowledge_base(data_types=['generated_video_asset'], tags=['video', 'ugc', str(client_id)] + video_topic_keywords, limit=5)
        if not existing_video_fragments_list: existing_video_fragments_list = await self.query_knowledge_base(data_types=['generated_video_asset'], tags=['video', 'ugc', 'generic_sample'] + video_topic_keywords, limit=5)
        if existing_video_fragments_list:
            for item_frag in existing_video_fragments_list: # Renamed variable
                try:
                    selected_video_asset = json.loads(item_frag.content)
                    video_path = selected_video_asset.get("path")
                    if video_path and os.path.exists(video_path):
                        self.logger.info(f"Reusing existing video asset: {video_path} for goal: {params.get('goal')}")
                        return [{"step": 1, "target_agent": "Orchestrator", "task_details": {"action": "store_artifact", "artifact_type": "video_final", "source_path": video_path, "metadata": {"reused": True, "original_asset_id": item_frag.id, "goal": params.get('goal')}}}]
                except json.JSONDecodeError: self.logger.warning(f"Could not parse content of KF ID {item_frag.id} as JSON.")
        base_video_path = params.get("base_video_path", "/app/assets/base_video.mp4"); image_prompt = params.get("image_prompt", "futuristic cityscape"); num_videos = params.get("count", 1)
        task_context = {
            "task": "Generate Detailed Video Workflow Plan", "workflow_goal": params.get("goal", "Generate sample UGC videos"),
            "num_videos_to_generate": num_videos, "base_video_path": base_video_path, "image_generation_prompt": image_prompt,
            "knowledge_base_context": kb_context_str or "No specific KB context found on Descript/AIStudio.",
            "desired_output_format": """JSON list of steps... Ensure the final step emails the video link(s) to the operator using USER_EMAIL from settings.""" # Same as v5.8
        }
        plan_prompt = await self.generate_dynamic_prompt(task_context) # Calls L75 prompt gen
        llm_model_pref = settings.OPENROUTER_MODELS.get('think_strategize')
        plan_json_str = await self._call_llm_with_retry(plan_prompt, model=llm_model_pref, temperature=0.3, max_tokens=3800, is_json_output=True)
        if plan_json_str:
            try:
                plan_list = self._parse_llm_json(plan_json_str, expect_type=list)
                if isinstance(plan_list, list) and all(isinstance(step, dict) and 'step' in step and 'task_details' in step for step in plan_list):
                    await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="workflow_plan", content={"workflow_type": "video_generation", "plan": plan_list, "status": "generated"}, tags=["video", "plan", "generated"], relevance_score=0.8)
                    return plan_list
            except Exception as e: self.logger.error(f"Failed to parse or validate LLM video plan: {e}. Response: {plan_json_str[:500]}...")
        return None

    async def learning_loop(self):
        """
        L75+ Meta-Learning Loop: ThinkTool reflects on its own strategic effectiveness,
        the impact of its learned patterns, and the success rates of its directives.
        It seeks to identify biases in its own reasoning or areas where its core
        synthesis/radar processes can be improved.
        """
        self.logger.info(f"{self.AGENT_NAME} L75+ Meta-Learning Loop activated.")
        while self.status == self.STATUS_RUNNING and not self._stop_event.is_set():
            learn_interval = int(self.config.get("THINKTOOL_META_LEARNING_INTERVAL_S", 3600 * 12)) # e.g., twice a day
            await asyncio.sleep(learn_interval)
            if self._stop_event.is_set(): break

            if not (self.orchestrator and getattr(self.orchestrator, 'approved', False)):
                self.logger.debug("ThinkTool Meta-Learning: Orchestrator not approved. Skipping cycle.")
                continue

            await self._internal_think("Periodic Meta-Reflection on ThinkTool's Strategic Learning & Effectiveness.")
            
            try:
                # 1. Gather data about ThinkTool's own performance
                async with self.session_maker() as session:
                    # Success/failure rates of directives issued by ThinkTool in the last N days
                    directive_horizon = datetime.now(timezone.utc) - timedelta(days=self.config.get("THINKTOOL_META_DIRECTIVE_ANALYSIS_DAYS", 7))
                    stmt_directives = select(StrategicDirective.status, func.count(StrategicDirective.id).label("count"))\
                        .where(StrategicDirective.source.like(f'%{self.AGENT_NAME}%'))\
                        .where(StrategicDirective.timestamp >= directive_horizon)\
                        .group_by(StrategicDirective.status)
                    directive_stats_raw = (await session.execute(stmt_directives)).mappings().all()
                    directive_performance = {stat['status']: stat['count'] for stat in directive_stats_raw}

                    # Impact of recently generated "exploit_hypothesis" patterns
                    exploit_pattern_horizon = datetime.now(timezone.utc) - timedelta(days=self.config.get("THINKTOOL_META_EXPLOIT_ANALYSIS_DAYS", 14))
                    stmt_exploit_patterns = select(LearnedPattern.id, LearnedPattern.pattern_description, LearnedPattern.confidence_score, LearnedPattern.implications)\
                        .where(LearnedPattern.pattern_type == 'exploit_hypothesis')\
                        .where(LearnedPattern.timestamp >= exploit_pattern_horizon)\
                        .order_by(desc(LearnedPattern.confidence_score))\
                        .limit(10)
                    recent_exploit_hypotheses = (await session.execute(stmt_exploit_patterns)).mappings().all()
                    # Further analysis could involve checking if directives related to these hypotheses were successful

                # 2. Formulate Meta-Critique Prompt
                meta_critique_context = {
                    "task": "Meta-Critique ThinkTool's Own Strategic Effectiveness & Learning Processes",
                    "thinktool_directive_performance_summary": directive_performance,
                    "recent_exploit_hypotheses_generated": [{"desc": p['pattern_description'], "conf": p['confidence_score'], "impl": p['implications'][:100]} for p in recent_exploit_hypotheses],
                    "current_core_mandate": self.meta_prompt[:500], # Snippet of its own mandate
                    "desired_output_format": "JSON: {\"self_assessment_of_strategic_output\": \"<Honest assessment of directive quality and impact>\", \"effectiveness_of_exploit_identification\": \"<How well is it finding real exploits?>\", \"blind_spots_in_reasoning\": [\"<Potential biases or areas ThinkTool might be overlooking in its synthesis/radar>\"], \"suggestions_for_improving_thinktool_prompts\": [{\"target_method_for_prompt_update\": \"<e.g., synthesize_insights_and_strategize or technology_radar>\", \"specific_prompt_enhancement_idea\": \"<Concrete suggestion>\"}], \"new_periodic_task_idea_for_self\": str? (A new type of analysis ThinkTool could run on itself)}"
                }
                prompt = await self.generate_dynamic_prompt(meta_critique_context)
                llm_model_pref = settings.OPENROUTER_MODELS.get('think_critique') # Use the most powerful model
                
                meta_critique_json = await self._call_llm_with_retry(prompt, model=llm_model_pref, temperature=0.4, max_tokens=2000, is_json_output=True)

                if meta_critique_json:
                    critique_data = self._parse_llm_json(meta_critique_json)
                    if critique_data:
                        self.logger.info(f"ThinkTool Meta-Critique Summary: {critique_data.get('self_assessment_of_strategic_output')}")
                        await self.log_knowledge_fragment(
                            agent_source=self.AGENT_NAME, data_type="thinktool_meta_critique",
                            content=critique_data, tags=["meta_learning", "self_improvement", "L75"],
                            relevance_score=0.95
                        )
                        # Potentially create directives for itself to refine its own prompts (for manual review first)
                        for prompt_suggestion in critique_data.get("suggestions_for_improving_thinktool_prompts", []):
                            if isinstance(prompt_suggestion, dict) and prompt_suggestion.get("target_method_for_prompt_update") and prompt_suggestion.get("specific_prompt_enhancement_idea"):
                                self.logger.warning(f"ThinkTool Meta-Critique suggests updating prompt for method: {prompt_suggestion['target_method_for_prompt_update']}. Suggestion: {prompt_suggestion['specific_prompt_enhancement_idea']}")
                                # This would typically require manual review before applying to its own core prompts
                                await self.log_knowledge_fragment(
                                    agent_source=self.AGENT_NAME, data_type="thinktool_prompt_refinement_suggestion",
                                    content=prompt_suggestion, tags=["meta_learning", "prompt_engineering"], relevance_score=0.9
                                )
                    else: self.logger.error("Failed to parse ThinkTool meta-critique JSON.")
                else: self.logger.error("LLM call for ThinkTool meta-critique failed.")

            except asyncio.CancelledError: self.logger.info(f"{self.AGENT_NAME} Meta-Learning loop cancelled."); break
            except Exception as e:
                self.logger.error(f"Error in {self.AGENT_NAME} Meta-Learning loop: {e}", exc_info=True)
                await self._report_error(f"Meta-Learning loop error: {e}")
                await asyncio.sleep(3600 * 1) # Wait an hour after an error in this critical loop

    async def self_critique(self) -> Dict[str, Any]:
        # ... (Code from v5.8 - unchanged, but ensure it calls the L75 generate_dynamic_prompt) ...
        self.logger.info("ThinkTool L75+: Performing Deep Strategic Self-Critique.")
        critique = {"status": "ok", "feedback": "Critique pending analysis."}
        await self._internal_think("Deep Strategic Self-Critique: Query all relevant system performance indicators -> Analyze against core mandate & AI-native principles -> Identify systemic strengths, weaknesses, and opportunities for radical improvement -> Propose meta-level adjustments.")
        try:
            # Gather more comprehensive data than just local stats
            # Example: Overall agency profit trend (requires DB query or KPI from Orchestrator)
            # Example: Success rate of "exploit" type directives
            # Example: Rate of new "LearnedPattern" generation and their impact
            # This data gathering would be more extensive for L75+
            
            async with self.session_maker() as session:
                kf_count = await session.scalar(select(func.count(KnowledgeFragment.id))) or 0
                pattern_count = await session.scalar(select(func.count(LearnedPattern.id)).where(LearnedPattern.confidence_score > 0.7)) or 0 # High confidence patterns
                directive_stats_res = await session.execute(
                    select(StrategicDirective.status, func.count(StrategicDirective.id).label("count"))
                    .where(StrategicDirective.timestamp >= datetime.now(timezone.utc) - timedelta(days=30)) # Last 30 days
                    .group_by(StrategicDirective.status)
                )
                directive_status_summary = {row.status: row.count for row in directive_stats_res.mappings().all()}

            task_context = {
                "task": "Deep Strategic Self-Critique of ThinkTool (L75+)",
                "knowledge_base_metrics": {"total_fragments": kf_count, "high_confidence_patterns": pattern_count},
                "directive_performance_summary_30d": directive_status_summary,
                "current_scoring_weights": self.scoring_weights,
                "recent_synthesis_assessment_preview": (await self.query_knowledge_base(data_types=["strategic_assessment_summary"], limit=1, order_by="timestamp_desc"))[0].content[:500] if await self.query_knowledge_base(data_types=["strategic_assessment_summary"], limit=1) else "N/A",
                "recent_radar_assessment_preview": (await self.query_knowledge_base(data_types=["overall_radar_assessment"], limit=1, order_by="timestamp_desc"))[0].content[:500] if await self.query_knowledge_base(data_types=["overall_radar_assessment"], limit=1) else "N/A",
                "core_mandate_reminder": "Exponential Profit & Market Domination, AI-Native Exploitation, Calculated Audacity.",
                "desired_output_format": "JSON ONLY: {{\"overall_effectiveness_rating_L75\": str ('Exceptional'|'Strong'|'Adequate'|'Subpar'|'Critical_Failure'), \"alignment_with_L75_mandate\": \"<Detailed assessment of how well ThinkTool is fulfilling its advanced mandate>\", \"key_strategic_successes_L75\": [\"<Examples of successful high-level strategies or adaptations>\"], \"critical_strategic_blindspots_or_inefficiencies_L75\": [\"<Areas where ThinkTool is underperforming or missing opportunities>\"], \"effectiveness_of_exploit_identification_L75\": str, \"suggestions_for_radical_self_improvement_L75\": [\"<Bold, concrete ideas to elevate ThinkTool's own strategic capabilities, prompt alchemy, or learning processes. Consider new periodic tasks or analytical frameworks for itself.\>\"], \"proposed_meta_prompt_refinement_for_thinktool\": str? (Suggest a key change to its own meta-prompt if beneficial)}"
            }
            critique_prompt = await self.generate_dynamic_prompt(task_context) # Calls L75 prompt gen
            llm_model_pref = settings.OPENROUTER_MODELS.get('think_critique') # Strongest model
            critique_json = await self._call_llm_with_retry(critique_prompt, temperature=0.25, max_tokens=3000, is_json_output=True, model=llm_model_pref)

            if critique_json:
                try:
                    critique_result = self._parse_llm_json(critique_json)
                    if not critique_result: raise ValueError("Parsed L75 critique is None")
                    critique['status'] = 'ok' if critique_result.get('overall_effectiveness_rating_L75') not in ['Subpar', 'Critical_Failure'] else 'warning'
                    critique['feedback'] = critique_result.get('overall_effectiveness_rating_L75', 'L75 Critique Generated.')
                    critique['details_L75'] = critique_result
                    await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="self_critique_summary_L75", content=critique_result, tags=["critique", "thinktool", "L75", critique['status']], relevance_score=0.95)
                    # Potentially act on self-improvement suggestions or meta-prompt refinement here or via a new directive
                    if critique_result.get("proposed_meta_prompt_refinement_for_thinktool"):
                        self.logger.warning(f"ThinkTool L75 Critique suggests meta-prompt refinement: {critique_result['proposed_meta_prompt_refinement_for_thinktool']}")
                        # This would require manual review or a highly advanced self-modifying agent
                except Exception as e_parse: self.logger.error(f"Failed to parse L75 self-critique LLM response: {e_parse}"); critique['status'] = 'error'; critique['feedback'] = "Failed to parse L75 critique."
            else: critique['status'] = 'error'; critique['feedback'] = "L75 LLM critique call failed."
        except Exception as e: self.logger.error(f"Error during L75 self-critique: {e}", exc_info=True); critique['status'] = 'error'; critique['feedback'] = f"L75 Critique failed: {e}"
        return critique

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        # ... (Code from v5.8 - with enhanced strategic focus in instructions) ...
        self.logger.debug(f"Generating L75+ dynamic prompt for ThinkTool task: {task_context.get('task')}")
        prompt_parts = [self.meta_prompt] # Start with L75 meta-prompt
        prompt_parts.append("\n--- Current Task & Strategic Context ---")
        # Prioritize more strategic context elements
        priority_keys = ['task', 'core_mandate_reminder', 'current_system_time_utc',
                         'recent_knowledge_fragments_summary', 'active_learned_patterns_summary',
                         'recent_performance_summary_7d', 'auto_generated_clay_directives_count',
                         'auto_generated_gmail_creation_directives_count', 'current_agency_goals_reminder',
                         'report_summary_and_key_findings', 'analysis_focus', # For Radar
                         'agent_name_whose_prompt_is_critiqued', 'prompt_key_being_critiqued', # For Prompt Critique
                         'triggering_feedback_or_context', 'current_prompt_text_to_be_critiqued',
                         'agent_proposing_action', 'proposed_task_description', # For Reflection
                         'agent_provided_context_for_action', 'proposed_plan_summary_if_available',
                         'relevant_knowledge_base_context',
                         'feedback_data', # For Feedback Analysis
                         'topic', 'context', # For Educational Content
                         'output_to_validate', 'validation_criteria', 'learned_patterns_context', # For Validation
                         'client_info', 'osint_summary', 'campaign_id', 'goal', 'successful_style_exemplars', # For Email/Social Content (if ThinkTool generates directly)
                         'enriched_data_available', 'cta_suggestion_from_thinktool',
                         'knowledge_base_context', # Generic KB context
                         'workflow_goal', 'num_videos_to_generate', 'base_video_path', 'image_generation_prompt' # For Video Workflow
                        ]

        for key in priority_keys:
            if key in task_context and task_context[key] is not None:
                 value = task_context[key]; value_str = ""; max_len = 3000 # Default increased
                 if key in ['recent_knowledge_fragments_summary', 'active_learned_patterns_summary', 'feedback_data', 'current_prompt_text_to_be_critiqued', 'relevant_knowledge_base_context', 'report_summary_and_key_findings', 'content_snippet']: max_len = 7000 # Allow very large context for core tasks
                 
                 if isinstance(value, str): value_str = value[:max_len] + ("..." if len(value) > max_len else "")
                 elif isinstance(value, (dict, list)):
                     try: value_str = json.dumps(value, default=str, indent=2); value_str = value_str[:max_len] + ("..." if len(value_str) > max_len else "")
                     except TypeError: value_str = str(value)[:max_len] + "..."
                 else: value_str = str(value)[:max_len] + "..."
                 prompt_key = key.replace('_', ' ').title()
                 prompt_parts.append(f"**{prompt_key}**: {value_str}")
        
        other_params = {k: v for k, v in task_context.items() if k not in priority_keys and k not in ['desired_output_format']}
        if other_params:
            prompt_parts.append("\n**Other Task-Specific Parameters:**")
            try: prompt_parts.append(f"```json\n{json.dumps(other_params, default=str, indent=2)}\n```")
            except TypeError: prompt_parts.append(str(other_params)[:1000] + "...")

        prompt_parts.append("\n--- Your Strategic Instructions & Output Requirements ---")
        task_type = task_context.get('task')
        prompt_parts.append(f"Execute Task: **{task_type or 'N/A'}** with extreme intelligence and focus on the agency's core mandate (Profit, Growth, AI Exploitation).")

        # Add specific instructions based on task_type (ensure these are L75+ focused)
        if task_type == 'Analyze Learning Material':
            prompt_parts.append("Extract deep insights, actionable strategies (especially unconventional/AI-native/grey-area), and potential exploits. Determine applicability to specific agents and overall agency goals. Assign a 'Potential Profit Impact' rating.")
        elif task_type == 'Grand Strategic Synthesis & AI Exploit Identification':
            prompt_parts.append("Perform a holistic synthesis of all provided data. Identify emergent high-ROI strategies, critical system inefficiencies, new AI-driven exploit opportunities, and necessary adaptations to maintain market dominance. Generate concrete, high-priority directives and new learned patterns (especially 'exploit_hypothesis' type).")
        elif task_type == 'Analyze Advanced Tech & Exploit Scouting Report':
            prompt_parts.append(f"Rigorously analyze the scouting report. Focus on: {task_context.get('analysis_focus', 'Exploitability, ROI, Risk, Feasibility')}. Prioritize game-changing intelligence. Recommend specific test directives for high-potential items.")
        elif task_type == 'Deep Strategic Reflection on Proposed Action/Plan':
            prompt_parts.append("Critically evaluate the proposed action/plan against the core mandate. Play devil's advocate: identify all critical failure points, assess realistic risks vs. rewards. Propose superior AI-native alternatives if they exist. Determine if LegalAgent consultation is needed for grey-area aspects. Output a clear 'proceed/revise/halt' recommendation with strong justification.")
        elif task_type == 'Critique and Rewrite Agent Prompt for Max Profit/Exploit Potential & AI-Native Operation':
            prompt_parts.append(f"Ruthlessly critique the current prompt for '{task_context.get('agent_name_whose_prompt_is_critiqued')}/{task_context.get('prompt_key_being_critiqued')}' based on '{task_context.get('triggering_feedback_or_context')}'. Focus on enhancing its ability to drive profit, enable AI-native operations, and uncover/execute grey-area exploits. Rewrite the *entire prompt* to be significantly more potent and strategically aligned.")
        # ... (other task instructions from v5.8, ensuring they reflect L75+ strategic depth)
        else:
            prompt_parts.append("Provide a clear, concise, strategically astute, and actionable response. Prioritize solutions that offer asymmetric advantages and leverage AI capabilities to their fullest.")

        if task_context.get('desired_output_format'):
            prompt_parts.append(f"\n**Strict Output Format:** Adhere meticulously to this JSON structure: {task_context['desired_output_format']}")
            if "JSON" in task_context.get('desired_output_format', ''): prompt_parts.append("\n```json")
        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated L75+ dynamic prompt for ThinkTool (length: {len(final_prompt)} chars)")
        return final_prompt

    async def collect_insights(self) -> Dict[str, Any]:
        # ... (Code from v5.8 - unchanged) ...
        self.logger.debug("ThinkTool collect_insights called.")
        insights = { "agent_name": self.AGENT_NAME, "status": self.status, "timestamp": datetime.now(timezone.utc).isoformat(), "kb_fragments": 0, "kb_patterns": 0, "active_directives": 0, "last_synthesis_run": self.last_synthesis_run.isoformat() if self.last_synthesis_run else None, "last_radar_run": self.last_radar_run.isoformat() if self.last_radar_run else None, "last_purge_run": self.last_purge_run.isoformat() if self.last_purge_run else None, "last_market_shift_analysis_run": self.last_market_shift_analysis_run.isoformat() if self.last_market_shift_analysis_run else None, "key_observations": [] }
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

    async def _process_clay_result(self, clay_api_result: Dict[str, Any], source_directive_id: Optional[int] = None, source_reference: Optional[str] = None, client_id_from_context: Optional[int] = None, is_webhook: bool = False):
        self.logger.info(f"Processing Clay API result. Directive ID: {source_directive_id}, Ref: {source_reference}, ClientID (context): {client_id_from_context}, Webhook: {is_webhook}")
        await self._internal_think("Processing Clay API result", details={"result_status": clay_api_result.get("status"), "ref": source_reference, "is_webhook": is_webhook})

        if clay_api_result.get("status") != "success":
            self.logger.warning(f"Clay API call/webhook failed, cannot process result. Message: {clay_api_result.get('message')}")
            if source_directive_id and not is_webhook: await self.update_directive_status(source_directive_id, 'failed', f"Clay API call failed: {clay_api_result.get('message')}")
            await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="clay_enrichment_error", content=clay_api_result, tags=["clay", "enrichment", "error", "webhook" if is_webhook else "direct_call"], relevance_score=0.2, source_reference=source_reference or f"Clay_Directive_{source_directive_id or 'webhook'}")
            return

        # For webhooks, the actual enriched data is in 'results' array, inside 'data'
        # For direct calls, it might be directly in 'data'
        clay_data_payload = clay_api_result.get("data", {})
        enriched_items = []
        if is_webhook:
            if isinstance(clay_data_payload.get("results"), list):
                enriched_items = clay_data_payload.get("results", [])
            elif isinstance(clay_data_payload, dict): # If results are not nested under 'results' for some reason
                enriched_items = [clay_data_payload]
        elif isinstance(clay_data_payload, dict): # Direct call result
             enriched_items = [clay_data_payload]
        elif isinstance(clay_data_payload, list): # Direct call might return a list
             enriched_items = clay_data_payload


        if not enriched_items:
            self.logger.warning(f"No enriched items found in Clay result for Directive {source_directive_id} / Ref {source_reference}.")
            if source_directive_id and not is_webhook: await self.update_directive_status(source_directive_id, 'completed_no_data', "Clay returned success but no enriched items.")
            await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="clay_enrichment_empty", content=clay_api_result, tags=["clay", "enrichment", "empty_result"], relevance_score=0.3, related_client_id=client_id_from_context, source_reference=source_reference or f"Clay_Directive_{source_directive_id or 'webhook'}")
            return

        processed_count = 0
        for item_data in enriched_items:
            if not isinstance(item_data, dict): continue # Skip non-dict items

            processed_info = {}
            # Enhanced field extraction, checking multiple common paths from Clay responses
            processed_info['verified_email'] = item_data.get('email') or item_data.get('person', {}).get('email') or item_data.get('email_address') or item_data.get('work_email') or item_data.get('personal_email')
            processed_info['job_title'] = item_data.get('job_title') or item_data.get('person', {}).get('title') or item_data.get('title')
            processed_info['company_name'] = item_data.get('company_name') or item_data.get('company', {}).get('name') or item_data.get('current_company',{}).get('name')
            processed_info['linkedin_url'] = item_data.get('linkedin_url') or item_data.get('person', {}).get('linkedin_url') or source_reference # Use source_reference if it's a linkedin URL
            processed_info['company_domain'] = item_data.get('company', {}).get('domain') or item_data.get('current_company',{}).get('domain')
            processed_info['full_name'] = item_data.get('full_name') or item_data.get('person', {}).get('full_name')
            processed_info['company_size'] = item_data.get('company', {}).get('company_size') or item_data.get('current_company',{}).get('company_size_range')
            processed_info['industry'] = item_data.get('company', {}).get('industry') or item_data.get('current_company',{}).get('industry')
            processed_info['location'] = item_data.get('location') or item_data.get('person', {}).get('location') or item_data.get('geo',{}).get('full_location')
            processed_info['company_linkedin_url'] = item_data.get('company',{}).get('linkedin_url') or item_data.get('current_company',{}).get('linkedin_url')
            processed_info['company_description'] = item_data.get('company',{}).get('description') or item_data.get('current_company',{}).get('short_description')
            processed_info['skills'] = item_data.get('person',{}).get('skills', [])
            processed_info['interests_from_clay'] = item_data.get('person',{}).get('interests', []) # Avoid conflict with Client.interests

            processed_info = {k: v for k, v in processed_info.items() if v is not None and v != '' and v != []}

            if processed_info:
                processed_count += 1
                try:
                    async with self.session_maker() as session:
                        async with session.begin():
                            target_client_id = client_id_from_context; target_client = None
                            if target_client_id: target_client = await session.get(Client, target_client_id)
                            
                            if not target_client: # Try to match existing client
                                conditions = []
                                if processed_info.get('verified_email'): conditions.append(Client.email == processed_info['verified_email'])
                                if processed_info.get('linkedin_url'): conditions.append(Client.source_reference == processed_info['linkedin_url'])
                                # Add more matching conditions if possible (e.g., name + company)
                                if conditions:
                                    target_client = (await session.execute(select(Client).where(or_(*conditions)).limit(1))).scalar_one_or_none()
                                    if target_client: target_client_id = target_client.id
                                    else: self.logger.info(f"No existing client matched for Clay data: {processed_info.get('full_name')} / {processed_info.get('linkedin_url')}")
                                else: self.logger.warning("Cannot reliably look up client for Clay result without email or LinkedIn URL.")

                            if target_client: # Update existing
                                update_values = {'last_interaction': datetime.now(timezone.utc), 'last_enriched_at': datetime.now(timezone.utc)}
                                if not target_client.email and processed_info.get('verified_email'): update_values['email'] = processed_info['verified_email']
                                if not target_client.company and processed_info.get('company_name'): update_values['company'] = processed_info['company_name']
                                if not target_client.job_title and processed_info.get('job_title'): update_values['job_title'] = processed_info['job_title']
                                if not target_client.source_reference and processed_info.get('linkedin_url'): update_values['source_reference'] = processed_info['linkedin_url']
                                # Merge interests/skills (be careful not to overwrite existing good data)
                                existing_interests = json.loads(target_client.interests) if target_client.interests else []
                                new_interests = list(set(existing_interests + processed_info.get('skills', []) + processed_info.get('interests_from_clay', [])))
                                if new_interests != existing_interests : update_values['interests'] = json.dumps(new_interests)

                                await session.execute(update(Client).where(Client.id == target_client_id).values(**update_values))
                                self.logger.info(f"Updated Client {target_client_id} with Clay enriched data: {list(update_values.keys())}")
                            elif processed_info.get('verified_email') and processed_info.get('full_name'): # Create new
                                new_client = Client(
                                    name=processed_info['full_name'], email=processed_info['verified_email'],
                                    source_reference=processed_info.get('linkedin_url', source_reference),
                                    company=processed_info.get('company_name'), job_title=processed_info.get('job_title'),
                                    industry=processed_info.get('industry'), location=processed_info.get('location'),
                                    interests=json.dumps(list(set(processed_info.get('skills', []) + processed_info.get('interests_from_clay', [])))),
                                    source="ClayEnrichment_Webhook" if is_webhook else "ClayEnrichment_Direct",
                                    opt_in=True, is_deliverable=True, # Default for new leads, review policy
                                    created_at=datetime.now(timezone.utc), last_interaction=datetime.now(timezone.utc), last_enriched_at=datetime.now(timezone.utc)
                                )
                                session.add(new_client); await session.flush(); target_client_id = new_client.id
                                self.logger.info(f"Created new Client {target_client_id} from Clay enrichment: {processed_info.get('full_name')}")
                            else: target_client_id = None # Cannot create

                            fragment_tags = ["clay", "enrichment", "lead_data", "webhook_processed" if is_webhook else "direct_call_processed"]
                            if target_client_id: fragment_tags.append(f"client_{target_client_id}")
                            await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="clay_enrichment_result_processed", content=processed_info, tags=fragment_tags, relevance_score=0.9, related_client_id=target_client_id, source_reference=processed_info.get('linkedin_url', source_reference))
                            
                            if processed_info.get('verified_email') and target_client_id:
                                outreach_directive_content = {
                                    "target_identifier": processed_info['verified_email'], "client_id": target_client_id,
                                    "context": f"Newly enriched lead via Clay ({'webhook' if is_webhook else 'direct call'}). Job: {processed_info.get('job_title', 'N/A')}, Company: {processed_info.get('company_name', 'N/A')}.",
                                    "goal": "Book sales call for UGC service", "enriched_data": processed_info,
                                    "email_style_hint": "value_driven_intro", # Suggest an initial style
                                    "cta_suggestion": "Request a brief 15-min chat to explore potential fit."
                                }
                                directive = StrategicDirective(source=self.AGENT_NAME, timestamp=datetime.now(timezone.utc), target_agent="EmailAgent", directive_type="initiate_outreach", content=json.dumps(outreach_directive_content), priority=4, status='pending')
                                session.add(directive)
                                self.logger.info(f"Generated outreach directive for EmailAgent for {processed_info['verified_email']} (Client ID: {target_client_id})")
                except Exception as e:
                    self.logger.error(f"Error processing/storing Clay item for directive {source_directive_id}/ref {source_reference}: {e}", exc_info=True)
                    await self._report_error(f"Error processing Clay item: {e}", task_id=f"ClayProc_{source_directive_id or source_reference}")

        if source_directive_id and not is_webhook: # Update original directive only if it was a direct call
            status_msg = f"Processed {processed_count} enriched items from Clay."
            if processed_count == 0 and not enriched_items : status_msg = "Clay returned success but no usable items found."
            await self.update_directive_status(source_directive_id, 'completed', status_msg)

    async def _process_clay_webhook_data(self, enriched_data: Dict[str, Any], original_input_parameters: Dict[str, Any], source_reference: Optional[str], clay_run_id: Optional[str]):
        """Handles data coming specifically from a Clay.com webhook."""
        self.logger.info(f"Processing Clay webhook data. Run ID: {clay_run_id}, Source Ref: {source_reference}")
        # The main logic is now in _process_clay_result, just need to adapt the input
        # Try to find client_id from metadata if passed during initial Clay call
        client_id_from_meta = None
        # This depends on how/if you pass metadata to Clay. Assume for now it might be in original_input_parameters or a dedicated field.
        # For example, if you passed `{"metadata": {"internal_client_id": 123}}` to Clay:
        # client_id_from_meta = original_input_parameters.get("metadata", {}).get("internal_client_id")
        # Or if Clay adds it to a run object that's part of the webhook (less likely for input params)

        # For now, we rely on source_reference (e.g. LinkedIn URL) for matching,
        # _process_clay_result will handle the lookup.
        await self._process_clay_result(
            clay_api_result={"status": "success", "data": enriched_data}, # Simulate the direct call result structure
            source_directive_id=None, # No direct directive ID for webhook, but could log clay_run_id
            source_reference=source_reference or original_input_parameters.get("linkedin_url"), # Prioritize explicit ref
            client_id_from_context=client_id_from_meta, # Pass if found
            is_webhook=True
        )

    async def _analyze_persistent_service_failure(self, service_name: str, failure_count: int, last_error_message: Optional[str]):
        """Analyzes persistent failures for a service (e.g., Gmail creation) and suggests actions."""
        self.logger.warning(f"Analyzing persistent failure for service: {service_name}. Failures: {failure_count}. Last Error: {last_error_message}")
        await self._internal_think(f"Persistent Failure Analysis: {service_name}", details={"failures": failure_count, "last_error": last_error_message})
        # Query KB for recent failure patterns for this service
        failure_pattern_frags = await self.query_knowledge_base(data_types=[f"{service_name.lower()}_creation_failure", "proxy_performance_alert"], time_window=timedelta(days=2), limit=10)
        context = f"Service '{service_name}' has failed {failure_count} consecutive times. Last error: {last_error_message}. Recent failure patterns/proxy issues from KB:\n"
        for frag in failure_pattern_frags: context += f"- {frag.data_type}: {str(frag.content)[:150]}...\n"

        task_context = {
            "task": "Devise Strategy for Persistent Service Failure",
            "service_name": service_name, "consecutive_failures": failure_count,
            "last_error_message": last_error_message, "knowledge_base_context": context,
            "desired_output_format": "JSON: {\"root_cause_hypothesis\": \"<Plausible reason for failures>\", \"suggested_actions\": [{\"action_type\": \"Change_Proxy_Strategy (e.g., request higher quality, different region)\", \"details\": \"...\"}, {\"action_type\": \"Modify_Identity_Parameters\", \"details\": \"...\"}, {\"action_type\": \"Adjust_Timing_or_Frequency\", \"details\": \"...\"}, {\"action_type\": \"Request_BrowsingAgent_Debug_Run (e.g., run with headless=false, record video)\", \"details\": \"...\"}, {\"action_type\": \"Temporary_Service_Cooldown (specify duration in hours)\", \"details\": \"...\"}], \"escalation_needed\": bool (if manual intervention seems required)}"
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        llm_model_pref = settings.OPENROUTER_MODELS.get('think_critique')
        response_json = await self._call_llm_with_retry(prompt, model=llm_model_pref, temperature=0.5, max_tokens=1000, is_json_output=True)
        if response_json:
            strategy = self._parse_llm_json(response_json)
            if strategy and strategy.get("suggested_actions"):
                self.logger.info(f"Strategy for {service_name} failure: {strategy.get('root_cause_hypothesis')}. Actions: {len(strategy['suggested_actions'])}")
                # Create directives based on suggestions
                for suggested_action in strategy.get("suggested_actions", []):
                    directive_content = {"service_name": service_name, **suggested_action}
                    # Target agent might be ThinkTool itself (to change its params), Orchestrator (for proxy), or GmailCreatorAgent
                    target_agent = "ThinkTool" # Default to self for strategy changes
                    if "Proxy" in suggested_action.get("action_type",""): target_agent = "Orchestrator"
                    elif "BrowsingAgent" in suggested_action.get("action_type",""): target_agent = "BrowsingAgent" # e.g. for a debug run
                    
                    await self.orchestrator.delegate_task("ThinkTool", { # ThinkTool creates directives
                        "action": "create_directive",
                        "content": {
                            "target_agent": target_agent,
                            "directive_type": f"adapt_{service_name.lower()}_{suggested_action.get('action_type','generic').lower().replace(' ','_')}",
                            "content": directive_content, "priority": 3 # High priority for failure recovery
                        }})
            else: self.logger.error(f"Failed to get strategic actions for {service_name} failure from LLM.")
        else: self.logger.error(f"LLM call failed for {service_name} failure analysis.")


    async def _analyze_and_adapt_creation_strategy(self, performance_analysis_content: Dict):
        """Analyzes creation performance (e.g., Gmail) and suggests adaptations."""
        service_name = performance_analysis_content.get("service_name", "UnknownService")
        self.logger.info(f"Analyzing and Adapting Creation Strategy for: {service_name}")
        await self._internal_think(f"Adapting Creation Strategy: {service_name}", details=performance_analysis_content)

        task_context = {
            "task": f"Refine Account Creation Strategy for {service_name}",
            "current_performance_analysis": performance_analysis_content,
            "knowledge_base_context": "Consider known good practices for account creation, common blocking mechanisms for this service (if any in KB), and success rates of different proxy types or identity archetypes from past KFs.",
            "desired_output_format": "JSON: {\"strategic_recommendations\": [\"<Specific, actionable recommendation, e.g., 'Prioritize residential proxies from US for next 5 attempts', 'Vary username patterns more significantly', 'Implement 2-hour cooldown if 2 consecutive failures occur on this service'>\"], \"updated_parameter_suggestions\": {\"max_consecutive_failures_before_cooldown\": int?, \"cooldown_duration_hours\": float?, \"identity_profile_hints_to_try\": [str]?}, \"confidence_in_recommendations\": float (0.0-1.0)}"
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        llm_model_pref = settings.OPENROUTER_MODELS.get('think_strategize')
        response_json = await self._call_llm_with_retry(prompt, model=llm_model_pref, temperature=0.6, max_tokens=1200, is_json_output=True)

        if response_json:
            adaptation_plan = self._parse_llm_json(response_json)
            if adaptation_plan and adaptation_plan.get("strategic_recommendations"):
                self.logger.info(f"Adaptation plan for {service_name}: {adaptation_plan.get('strategic_recommendations')}")
                # Log this adaptation plan to KB
                await self.log_knowledge_fragment(
                    agent_source=self.AGENT_NAME, data_type=f"{service_name}_creation_strategy_adaptation",
                    content=adaptation_plan, tags=["strategy_update", service_name.lower(), "account_creation"], relevance_score=0.85
                )
                # Apply parameter suggestions if relevant to this agent (ThinkTool might adjust its own params)
                # Or create directives for other agents (e.g., GmailCreatorAgent to change its internal cooldown)
                updated_params = adaptation_plan.get("updated_parameter_suggestions", {})
                if service_name == "GmailCreation" and "GmailCreatorAgent" in self.orchestrator.agents:
                    if updated_params:
                         await self.orchestrator.delegate_task("GmailCreatorAgent", {
                             "action": "update_creation_parameters", # New action for GmailCreatorAgent
                             "content": updated_params
                         })
            else: self.logger.error(f"Failed to get adaptation plan for {service_name} from LLM.")
        else: self.logger.error(f"LLM call failed for {service_name} adaptation analysis.")

    async def _assess_initial_account_health(self, service_filter_list: Optional[List[str]] = None, batch_size: int = 20):
        """
        L75+ Assesses health of accounts by creating directives for BrowsingAgent to test logins.
        Prioritizes accounts with 'unknown' health or 'active' but not recently used.
        """
        self.logger.info(f"L75+ Assessing initial/stale account health (services: {service_filter_list or 'all relevant'}, batch: {batch_size})...")
        await self._internal_think("Initial/Stale Account Health Assessment", details={"services": service_filter_list, "batch_size": batch_size})
        
        if not self.session_maker or not self.orchestrator or not hasattr(self.orchestrator, 'delegate_task'):
            self.logger.error("DB session_maker or Orchestrator unavailable for account health assessment.")
            return

        accounts_to_test_ids = []
        try:
            async with self.session_maker() as session:
                # Prioritize 'unknown' health, then 'active' ones not successfully used in N days
                # This requires AccountCredentials to have a 'last_successful_use_ts' or similar field,
                # or we infer from last successful task involving this account.
                # For now, let's simplify: prioritize 'unknown', then 'active' ordered by last_status_update_ts (older first)
                
                # Subquery for accounts that have recent failure KFs
                # This is a conceptual sketch, actual implementation depends on how failures are logged
                # For now, we'll focus on status in AccountCredentials
                
                stmt = select(AccountCredentials.id, AccountCredentials.service, AccountCredentials.account_identifier, AccountCredentials.status, AccountCredentials.last_status_update_ts)\
                    .where(AccountCredentials.status.in_(['unknown', 'active', 'needs_review', 'limited'])) # Include more statuses to re-verify
                
                if service_filter_list:
                    stmt = stmt.where(AccountCredentials.service.in_(service_filter_list))
                
                # Order by status priority ('unknown' first), then by oldest update
                stmt = stmt.order_by(
                    case(
                        (AccountCredentials.status == 'unknown', 0),
                        (AccountCredentials.status == 'needs_review', 1),
                        (AccountCredentials.status == 'limited', 2),
                        (AccountCredentials.status == 'active', 3),
                        else_=4
                    ),
                    asc(AccountCredentials.last_status_update_ts) # Test oldest ones first
                ).limit(batch_size)

                accounts_result = await session.execute(stmt)
                accounts_to_test = accounts_result.mappings().all()

            if not accounts_to_test:
                self.logger.info("No accounts found needing initial/stale health assessment in this batch.")
                return

            self.logger.info(f"Found {len(accounts_to_test)} accounts for health check. Generating test directives...")
            directives_created = 0
            for acc in accounts_to_test:
                # Check if a test directive for this account is already pending
                async with self.session_maker() as session_check:
                    existing_directive_stmt = select(StrategicDirective.id).where(
                        StrategicDirective.target_agent == "BrowsingAgent",
                        StrategicDirective.directive_type == "test_account_login",
                        StrategicDirective.content.like(f'%\"account_id_to_test\": {acc.id}%'), # Check content
                        StrategicDirective.status.in_(['pending', 'active'])
                    ).limit(1)
                    existing_pending_directive = (await session_check.execute(existing_directive_stmt)).scalar_one_or_none()

                if existing_pending_directive:
                    self.logger.debug(f"Skipping health check for account ID {acc.id}, test directive already pending (ID: {existing_pending_directive}).")
                    continue

                directive_content = {
                    "account_id_to_test": acc.id,
                    "service": acc.service,
                    "identifier_to_test": acc.account_identifier, # Pass identifier for BrowsingAgent
                    "goal": f"Attempt login to {acc.service} for account ID {acc.id} ({acc.account_identifier}) to verify health and update status.",
                    "expected_outcome_on_success": "Successful login, navigation to main dashboard/feed if applicable.",
                    "report_failure_reason_specifically": True
                }
                
                # This directive will be picked up by Orchestrator and routed to BrowsingAgent
                # BrowsingAgent's execute_step for "test_account_login" will then attempt login
                # and upon success/failure, it (or ThinkTool processing its outcome KF) will call _flag_account_issue.
                await self.orchestrator.delegate_task("ThinkTool", { # ThinkTool creates the directive
                    "action": "create_directive", # Assuming ThinkTool has this action
                    "content": {
                        "target_agent": "BrowsingAgent",
                        "directive_type": "test_account_login",
                        "content": directive_content,
                        "priority": 7 # Medium-low priority for background health checks
                    }
                })
                directives_created += 1
            self.logger.info(f"Created {directives_created} directives for account health checks.")

        except Exception as e:
            self.logger.error(f"Error during initial account health assessment: {e}", exc_info=True)
            await self._report_error(f"Account health assessment failed: {e}")


    async def _flag_account_issue(self, account_id: int, issue_description: str, severity: str, details: Optional[str] = None):
        """Flags an account in the database (via updating AccountCredentials table)."""
        self.logger.warning(f"Flagging issue for account ID {account_id}: {issue_description} (Severity: {severity}). Details: {details}")
        if not self.session_maker: return
        new_status = "needs_review"
        if severity == "critical" or "banned" in issue_description.lower() or "locked" in issue_description.lower():
            new_status = "banned" # Or 'locked'
        elif severity == "high" or "login_failure" in issue_description.lower():
            new_status = "limited" # Or 'needs_password_reset'

        try:
            async with self.session_maker() as session:
                async with session.begin():
                    stmt = update(AccountCredentials).where(AccountCredentials.id == account_id).\
                        values(status=new_status, notes=func.concat(AccountCredentials.notes, f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} Issue: {issue_description}. Details: {details or 'N/A'}]"))
                    await session.execute(stmt)
                self.logger.info(f"Account ID {account_id} status updated to '{new_status}' in DB due to: {issue_description}")
                # Also update internal state if the account is managed by an agent that caches it (e.g., SMM)
                # This might require a notification back to the relevant agent or Orchestrator.
                if "SocialMediaManager" in self.orchestrator.agents:
                    await self.orchestrator.delegate_task("SocialMediaManager", {
                        "action": "update_account_health_status", # SMM needs this action
                        "account_id": account_id, "new_health_status": new_status, "reason": issue_description
                    })

        except Exception as e: self.logger.error(f"Failed to update account {account_id} status in DB: {e}")

    async def _create_directive_from_suggestion(self, source_agent_name: str, suggestion_text: str, priority: int = 7):
        """Creates a new StrategicDirective based on a suggestion from another agent's critique."""
        self.logger.info(f"Creating directive from suggestion by {source_agent_name}: {suggestion_text[:100]}...")
        # LLM call to parse the suggestion into a structured directive
        task_context = {
            "task": "Convert Agent Suggestion into Strategic Directive",
            "source_agent": source_agent_name,
            "suggestion_text": suggestion_text,
            "available_target_agents": list(self.orchestrator.agents.keys()),
            "common_directive_types": ["test_new_exploit", "refine_agent_prompt", "investigate_tool_technique", "update_workflow_parameter", "analyze_specific_data"],
            "desired_output_format": "JSON: {\"target_agent\": \"<ChosenAgentName>\", \"directive_type\": \"<AppropriateDirectiveType>\", \"content\": {<Structured content for the directive>}, \"priority\": int (1-10, adjust based on suggestion urgency/impact, default to passed priority)}"
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        llm_model_pref = settings.OPENROUTER_MODELS.get('think_validate') # Fast model for structuring
        directive_json_str = await self._call_llm_with_retry(prompt, model=llm_model_pref, temperature=0.2, max_tokens=800, is_json_output=True)

        if directive_json_str:
            directive_data = self._parse_llm_json(directive_json_str)
            if directive_data and all(k in directive_data for k in ['target_agent', 'directive_type', 'content']):
                async with self.session_maker() as session:
                    async with session.begin():
                        new_directive = StrategicDirective(
                            source=f"SuggestionBy_{source_agent_name}", timestamp=datetime.now(timezone.utc), status='pending',
                            target_agent=directive_data['target_agent'], directive_type=directive_data['directive_type'],
                            content=json.dumps(directive_data['content']),
                            priority=directive_data.get('priority', priority)
                        )
                        session.add(new_directive)
                    self.logger.info(f"Created new directive ID {new_directive.id} from suggestion for {directive_data['target_agent']}.")
                    return {"status": "success", "directive_id": new_directive.id}
            else: self.logger.error(f"LLM failed to structure suggestion into valid directive: {directive_json_str}")
        else: self.logger.error("LLM call failed for directive creation from suggestion.")
        return {"status": "failure", "message": "Could not convert suggestion to directive."}

    async def _plan_social_media_campaign(self, campaign_details: Dict[str, Any]) -> Optional[int]:
        """Generates a detailed social media campaign plan and stores it as a KF, returns KF ID."""
        self.logger.info(f"L75+ Planning Social Media Campaign with details: {str(campaign_details)[:200]}...")
        await self._internal_think("Plan Social Media Campaign (L75+)", details=campaign_details)

        # Fetch relevant KB: competitor strategies, successful past campaigns, platform algorithm insights
        kb_context_tags = ["social_media_strategy", "competitor_analysis", "platform_algorithm_insight"]
        if campaign_details.get("platforms"): kb_context_tags.extend([p.lower() for p in campaign_details["platforms"]])
        kb_frags = await self.query_knowledge_base(tags=kb_context_tags, limit=10, min_relevance=0.6)
        kb_context_str = "\n".join([f"- {f.data_type} (ID {f.id}): {f.content[:150]}..." for f in kb_frags])

        task_context = {
            "task": "Generate Detailed Multi-Platform Social Media Campaign Plan (L75+)",
            "campaign_goal": campaign_details.get('goal'),
            "target_platforms": campaign_details.get('platforms'),
            "duration_days": campaign_details.get('duration_days'),
            "target_audience_description": campaign_details.get('audience_description'),
            "key_message_pillars": campaign_details.get('key_messages'),
            "desired_intensity": campaign_details.get('intensity'),
            "budget_consideration": campaign_details.get('budget_consideration'),
            "specific_metrics_to_optimize_for": campaign_details.get('specific_metrics_to_optimize_for'),
            "request_grey_area_suggestions": campaign_details.get('request_grey_area_suggestions', True),
            "knowledge_base_context": kb_context_str or "No specific KB context found.",
            "available_account_groups_conceptual": ["BrandVoicePrime", "EngagementSwarm", "TrafficAlchemists", "AlgorithmicProbes", "ShadowOpsInfiltrators"], # For LLM to assign
            "desired_output_format": """JSON: {"campaign_name": str, "overall_strategy_summary": str, "target_kpis": [str], "phases": [ {"phase_name": str, "duration_days": int, "phase_goal": str, "platform_specific_tactics": [ {"platform": str, "account_group_to_use": str, "tactic_description": str, "content_themes_or_types": [str], "key_actions_for_smm": [\"<Detailed instruction for SMM/BrowsingAgent, e.g., 'Post 3x daily focusing on short-form video with X trend sound', 'Run A/B test on CTA for LinkedIn posts', 'Use EngagementSwarm to boost initial visibility of key posts on X.com'>\"], "estimated_posting_frequency": str, "interaction_strategy": str} ] } ] } - Plan should be detailed, actionable, and leverage AI-native/grey-area tactics where appropriate and risk-assessed."""
        }
        plan_prompt = await self.generate_dynamic_prompt(task_context)
        llm_model_pref = settings.OPENROUTER_MODELS.get('think_strategize')
        plan_json_str = await self._call_llm_with_retry(plan_prompt, model=llm_model_pref, temperature=0.4, max_tokens=3900, is_json_output=True)

        if plan_json_str:
            try:
                campaign_plan_data = self._parse_llm_json(plan_json_str)
                if campaign_plan_data and campaign_plan_data.get("campaign_name") and campaign_plan_data.get("phases"):
                    self.logger.info(f"L75+ Successfully generated campaign plan: {campaign_plan_data['campaign_name']}")
                    kf = await self.log_knowledge_fragment(
                        agent_source=self.AGENT_NAME, data_type="social_campaign_plan_L75",
                        content=campaign_plan_data, tags=["plan", "social_media", "L75"] + campaign_details.get('platforms', []),
                        relevance_score=0.95
                    )
                    return kf.id if kf else None
                else: self.logger.error(f"L75+ LLM campaign plan response was not valid: {plan_json_str[:500]}...")
            except Exception as e: self.logger.error(f"L75+ Failed to parse or store LLM campaign plan: {e}. Response: {plan_json_str[:500]}...")
        else: self.logger.error("L75+ LLM failed to generate a campaign plan.")
        return None

    async def _proactive_market_shift_analysis(self):
        """Periodically scans for market shifts, new niches, competitor weaknesses."""
        self.logger.info("ThinkTool L75+: Starting Proactive Market Shift & Opportunity Analysis.")
        await self._internal_think("Proactive Market Analysis: Scan news, financial data, competitor KFs -> Identify emergent niches, client types, or service gaps -> Generate investigation directives.")
        try:
            # 1. Define Search Vectors for BrowsingAgent (or use NewsAPI if configured)
            market_scan_queries = [
                "latest trends in B2B SaaS marketing automation",
                "major funding rounds in e-commerce technology last 30 days",
                "common pain points for marketing VPs in [TargetIndustryX]", # TargetIndustryX from config or learned
                "competitor analysis of [MajorCompetitorY]'s recent product launches" # MajorCompetitorY from config/KB
            ]
            # For this example, we'll just use one query
            chosen_query = random.choice(market_scan_queries)
            self.logger.info(f"Market Scan Query: {chosen_query}")

            market_data_summary = "No fresh market data retrieved this cycle."
            if self.orchestrator and hasattr(self.orchestrator, 'delegate_task') and "BrowsingAgent" in self.orchestrator.agents:
                scan_task = {"action": "perform_search_and_summarize", "query": chosen_query, "num_results": 10}
                scan_result = await self.orchestrator.delegate_task("BrowsingAgent", scan_task)
                if scan_result and scan_result.get("status") == "success":
                    market_data_summary = scan_result.get("summary", market_data_summary)
            
            # 2. Analyze with LLM
            task_context = {
                "task": "Analyze Market Scan Data for New Opportunities/Threats (L75+)",
                "market_data_summary": market_data_summary,
                "current_agency_services": ["AI-Powered UGC Video Generation", "Hyper-Personalized Email Outreach", "Strategic Social Media Management"],
                "current_target_niches": settings.get("TARGET_NICHES_PRIMARY", ["B2B SaaS", "E-commerce Brands"]), # From settings
                "desired_output_format": "JSON: {\"identified_shifts_or_opportunities\": [{\"type\": \"New_Niche\"|\"New_Service_Angle\"|\"Competitor_Weakness\"|\"Emerging_Client_Pain_Point\", \"description\": str, \"potential_agency_play\": str, \"estimated_market_size_or_impact\": str, \"confidence\": float}], \"emerging_threats\": [{\"description\": str, \"mitigation_idea\": str}], \"overall_market_sentiment_towards_ai_sales_automation\": str}"
            }
            analysis_prompt = await self.generate_dynamic_prompt(task_context)
            llm_model_pref = settings.OPENROUTER_MODELS.get('think_strategize')
            analysis_json = await self._call_llm_with_retry(analysis_prompt, model=llm_model_pref, temperature=0.6, max_tokens=2000, is_json_output=True)

            if analysis_json:
                analysis_data = self._parse_llm_json(analysis_json)
                if analysis_data:
                    self.logger.info(f"Market Shift Analysis complete. Sentiment: {analysis_data.get('overall_market_sentiment_towards_ai_sales_automation')}")
                    for opp in analysis_data.get("identified_shifts_or_opportunities", []):
                        await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="market_opportunity_signal_L75", content=opp, tags=["market_shift", "opportunity", opp.get("type","unknown").lower()], relevance_score=opp.get("confidence", 0.7)*0.9) # Scale relevance
                        # Optionally, create high-level investigation directives
                        if opp.get("confidence", 0) > 0.75:
                             await self.orchestrator.delegate_task("ThinkTool", {"action":"create_directive", "content": {"target_agent":"ThinkTool", "directive_type": "investigate_market_opportunity", "content": opp, "priority":4}})
                    for threat in analysis_data.get("emerging_threats", []):
                         await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="market_threat_signal_L75", content=threat, tags=["market_shift", "threat"], relevance_score=0.8)
            else: self.logger.error("LLM failed to analyze market shift data.")
        except Exception as e: self.logger.error(f"Error in proactive market shift analysis: {e}", exc_info=True)
        self.last_market_shift_analysis_run = datetime.now(timezone.utc)

# --- End of agents/think_tool.py ---