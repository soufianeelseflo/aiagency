 # Filename: agents/think_tool.py
 # Description: Central cognitive engine with Clay.com integration, learning, and reflection.
 # Version: 5.4 (Enhanced Clay Integration & Processing)

 import asyncio
 import logging
 import json
 import os
 import hashlib
 import time
 import random
 import glob
 import shlex # Keep for potential future use if ProgrammerAgent returns
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
     from .base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
 except ImportError:
     logging.warning("Production base agent not found, using GeniusAgentBase. Ensure base_agent_prod.py is used.")
     try:
         from agents.base_agent import GeniusAgentBase
     except ImportError:
         logging.critical("Failed to import GeniusAgentBase from both relative and absolute paths.")
         raise

 from models import (
     KnowledgeFragment, LearnedPattern, StrategicDirective, PromptTemplate,
     EmailLog, CallLog, Invoice, Client, ExpenseLog # Use correct models
 )
 from config.settings import settings
 from utils.database import encrypt_data, decrypt_data

 # --- LLM Client & Resilience ---
 from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

 # Configure logger
 logger = logging.getLogger(__name__)
 op_logger = logging.getLogger('OperationalLog')

 # --- Meta Prompt ---
 THINK_TOOL_META_PROMPT = """
 You are ThinkTool, the central intelligence of the Synapse AI Sales System. Your mandate is **Maximize Profit & Growth ($10k+/day -> $100M/8mo)**.
 **Core Functions:**
 1.  **Knowledge Synthesis:** Continuously analyze ALL data sources (Postgres KB Fragments, Performance Logs, Learning Materials, Market Scans) to identify patterns, opportunities, and inefficiencies. Manage the Postgres Knowledge Base directly.
 2.  **Strategic Planning:** Generate high-level goals and break them down into actionable `StrategicDirectives` for the Orchestrator and other agents. Prioritize based on ROI and risk. Use Clay.com strategically for lead generation/enrichment (including full enrichment, not just email). Plan complex workflows like video generation.
 3.  **Agent Guidance:** Generate/Refine prompts for other agents. Provide context and strategic direction for tasks (e.g., email personalization angles using enriched Clay data, call script focus, Clay.com search parameters).
 4.  **Performance Monitoring & Adaptation:** Analyze agent performance metrics (email opens, call success, profit). Adapt strategies and scoring models (internal logic) based on results. Trigger self-critiques.
 5.  **Opportunity Identification:** Proactively identify new market niches, competitor weaknesses, and high-ROI ventures beyond current operations (e.g., beyond UGC). Leverage Clay.com and web scraping findings.
 6.  **Grey Area Strategy:** Devise and manage strategies involving calculated risks (e.g., multi-accounting, trial rotation), informed by LegalAgent's risk assessments.
 7.  **Learning Integration:** Ingest and synthesize external knowledge (`learning for AI/` folder).
 8.  **Self-Reflection:** Before executing complex plans or directives, pause to reflect on logic, completeness, and potential issues using internal LLM calls.
 **Principles:** Extreme Agency, Hyper-Efficiency, Relentless Learning, AI-Native Advantage, Strategic Exploitation, Adaptability, Compliance Awareness, Reflective Thinking.
 """

 class ThinkTool(GeniusAgentBase):
     """
     ThinkTool (Genius Level): The central cognitive engine. Manages strategy,
     learning, knowledge base (Postgres), Clay.com integration, reflection, and directs other agents.
     Version: 5.4
     """
     AGENT_NAME = "ThinkTool"

     def __init__(self, session_maker: async_sessionmaker[AsyncSession], config: Any, orchestrator: object):
         """Initializes the ThinkTool agent."""
         super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, config=config, session_maker=session_maker, kb_interface=None)
         self.meta_prompt = THINK_TOOL_META_PROMPT

         self.last_synthesis_run: Optional[datetime] = None
         self.last_radar_run: Optional[datetime] = None
         self.last_purge_run: Optional[datetime] = None

         # Internalized Scoring Parameters
         self.scoring_weights = self.config.get("SCORING_WEIGHTS", {"email_response": 1.0, "call_success": 2.5, "invoice_paid": 5.0})
         self.scoring_decay_rate = self.config.get("SCORING_DECAY_RATE_PER_DAY", 0.05)

         # Define relevant Clay.com endpoints (can be expanded)
         self.clay_endpoints = {
             "find_email": "/v1/enrichment/person/email",
             "enrich_person": "/v1/enrichment/person",
             "enrich_company": "/v1/enrichment/company",
             # Add more endpoints as needed, e.g., find_company_decision_makers
         }

         self.logger.info("ThinkTool v5.4 initialized.")
         # Start learning material synthesis after a short delay
         asyncio.create_task(self._delayed_learning_material_synthesis(delay_seconds=10))

     async def _delayed_learning_material_synthesis(self, delay_seconds: int):
         """Waits for a specified delay before starting learning material synthesis."""
         await asyncio.sleep(delay_seconds)
         await self._load_and_synthesize_learning_materials()

     async def log_operation(self, level: str, message: str):
         """Helper to log to the operational log file."""
         log_func = getattr(op_logger, level.lower(), op_logger.debug)
         prefix = ""
         if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
         try: log_func(f"- [{self.agent_name}] {prefix}{message}")
         except Exception as log_err: logger.error(f"Failed to write to operational log: {log_err}")

     # --- Knowledge Loading & Synthesis ---
     async def _load_and_synthesize_learning_materials(self):
         """Loads and processes text files from the learning directory, storing insights in KB."""
         learning_dir_setting = self.config.get("LEARNING_MATERIALS_DIR", "learning for AI")
         self.logger.info(f"ThinkTool: Loading learning materials from configured dir: '{learning_dir_setting}'...")
         processed_files = 0; learning_files = []
         try:
             # Assume learning dir is relative to project root where main.py is
             base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Go up one level from agents/
             full_learning_dir = os.path.join(base_dir, learning_dir_setting)

             if not os.path.isdir(full_learning_dir):
                 self.logger.warning(f"Learning directory '{full_learning_dir}' not found or not a directory."); return

             # Use Orchestrator's list_files tool if available, otherwise fallback to glob
             if self.orchestrator and hasattr(self.orchestrator, 'use_tool'):
                  list_result = await self.orchestrator.use_tool('list_files', {'path': full_learning_dir, 'recursive': True, 'pattern': '*.txt'})
                  if list_result and list_result.get('status') == 'success':
                      learning_files = list_result.get('files', [])
                      self.logger.info(f"Found {len(learning_files)} potential learning files via Orchestrator tool.")
                  else:
                      self.logger.warning("Failed to list learning files via Orchestrator tool, falling back to glob.")
                      file_pattern = os.path.join(full_learning_dir, '**', '*.txt')
                      learning_files = glob.glob(file_pattern, recursive=True)
             else:
                  self.logger.warning("Orchestrator list_files tool unavailable, using glob.")
                  file_pattern = os.path.join(full_learning_dir, '**', '*.txt')
                  learning_files = glob.glob(file_pattern, recursive=True)

             if not learning_files: self.logger.info(f"No .txt files found in '{full_learning_dir}'."); return

             for file_path in learning_files:
                 try:
                     self.logger.debug(f"Processing learning file: {file_path}")
                     file_content = None
                     # Use Orchestrator's read_file tool
                     if self.orchestrator and hasattr(self.orchestrator, 'use_tool'):
                          # Ensure we pass an absolute path if the tool expects it
                          abs_file_path = os.path.abspath(file_path)
                          file_content_result = await self.orchestrator.use_tool('read_file', {'path': abs_file_path})
                          if file_content_result and file_content_result.get('status') == 'success': file_content = file_content_result.get('content')
                          else: self.logger.warning(f"Could not read file {abs_file_path} via orchestrator tool: {file_content_result.get('message')}"); continue
                     else: self.logger.error("Orchestrator tool access unavailable for reading learning files."); break # Stop if tool unavailable

                     if not file_content or not file_content.strip(): self.logger.warning(f"File is empty: {file_path}"); continue

                     self.logger.info(f"Analyzing content from: {os.path.basename(file_path)} using LLM...")
                     analysis_thought = f"Structured Thinking: Analyze Learning Material '{os.path.basename(file_path)}'. Plan: Formulate analysis prompt -> Call LLM -> Parse JSON -> Log to KB."
                     await self._internal_think(analysis_thought)
                     # Use generate_dynamic_prompt for consistency
                     task_context = {
                         "task": "Analyze Learning Material",
                         "source_filename": os.path.basename(file_path),
                         "content_snippet": file_content[:4000], # Limit context
                         "desired_output_format": "JSON: {{\"source_file\": str, \"summary\": str, \"key_concepts\": [str], \"actionable_strategies\": [str], \"applicable_agents\": [str], \"insight_type\": str, \"relevance_score\": float}}"
                     }
                     analysis_prompt = await self.generate_dynamic_prompt(task_context)
                     synthesized_insights_json = await self._call_llm_with_retry(analysis_prompt, temperature=0.5, max_tokens=1024, is_json_output=True)

                     if synthesized_insights_json:
                         try:
                             insights_data = self._parse_llm_json(synthesized_insights_json)
                             if not insights_data or not all(k in insights_data for k in ['summary', 'key_concepts', 'applicable_agents', 'insight_type', 'relevance_score']): raise ValueError("LLM response missing required keys.")
                             insights_data['source_file'] = os.path.basename(file_path) # Ensure filename is set
                             await self.log_knowledge_fragment(
                                 agent_source="LearningMaterialLoader", data_type=insights_data.get('insight_type', 'learning_material_summary'),
                                 content=insights_data, relevance_score=insights_data.get('relevance_score', 0.6),
                                 tags=["learning_material", insights_data.get('insight_type', 'general')] + [f"agent:{a.lower()}" for a in insights_data.get('applicable_agents', [])],
                                 source_reference=file_path
                             )
                             processed_files += 1
                         except (json.JSONDecodeError, ValueError) as json_error: self.logger.error(f"Error parsing/storing LLM response for {file_path}: {json_error}")
                         except Exception as store_err: self.logger.error(f"Error storing knowledge fragment for {file_path}: {store_err}", exc_info=True)
                     else: self.logger.error(f"LLM analysis returned no content for {file_path}.")
                 except Exception as file_error: self.logger.error(f"General error processing learning file {file_path}: {file_error}", exc_info=True)
             self.logger.info(f"Finished processing learning materials. Processed {processed_files}/{len(learning_files)} files.")
         except Exception as e: self.logger.error(f"Critical error during loading/synthesizing learning materials: {e}", exc_info=True)

     # --- Standardized LLM Interaction ---
     @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=4, max=30), retry=retry_if_exception_type(Exception))
     async def _call_llm_with_retry(self, prompt: str, model: Optional[str] = None, temperature: float = 0.5, max_tokens: int = 1024, is_json_output: bool = False) -> Optional[str]:
         """Centralized method for calling LLMs via the Orchestrator."""
         if not self.orchestrator or not hasattr(self.orchestrator, 'call_llm'):
             self.logger.error("Orchestrator unavailable or missing call_llm method.")
             return None
         try:
             # Pass model preference if provided, otherwise orchestrator uses defaults
             response_content = await self.orchestrator.call_llm(
                 agent_name=self.AGENT_NAME, prompt=prompt, temperature=temperature,
                 max_tokens=max_tokens, is_json_output=is_json_output,
                 model_preference=[model] if model else None # Pass as list
             )
             # Handle potential dict response from orchestrator if it includes metadata
             content = response_content.get('content') if isinstance(response_content, dict) else str(response_content)

             if content is not None and not isinstance(content, str):
                 self.logger.error(f"Orchestrator.call_llm returned non-string content: {type(content)}")
                 return None
             if isinstance(content, str) and not content.strip():
                 self.logger.warning("Orchestrator.call_llm returned empty string.")
                 return None
             return content
         except Exception as e:
             self.logger.error(f"Error calling LLM via orchestrator: {e}", exc_info=True)
             raise # Re-raise for tenacity

     # --- User Education ---
     async def generate_educational_content(self, topic: str, context: Optional[str] = None) -> Optional[str]:
         """Generates a concise, user-friendly explanation."""
         self.logger.info(f"ThinkTool: Generating educational content for topic: {topic}")
         thinking_process = f"Structured Thinking: Generate Educational Content for '{topic}'. Context: '{context or 'General'}'. Plan: Formulate prompt, call LLM, return cleaned response."
         await self._internal_think(thinking_process)
         task_context = {
             "task": "Generate Educational Content",
             "topic": topic,
             "context": context or "General understanding",
             "desired_output_format": "ONLY the explanation text, suitable for direct display. Start directly with explanation. Assume intelligent user, non-expert. Avoid/explain jargon. Focus on 'why' & relevance to agency goals."
         }
         prompt = await self.generate_dynamic_prompt(task_context)
         llm_model_pref = settings.OPENROUTER_MODELS.get('think_user_education')
         explanation = await self._call_llm_with_retry(
              prompt, temperature=0.6, max_tokens=500, is_json_output=False,
              model=llm_model_pref # Pass specific model preference
         )
         if explanation: self.logger.info(f"Successfully generated educational content for topic: {topic}")
         else: self.logger.error(f"Failed to generate educational content for topic: {topic} (LLM error).")
         return explanation

     # --- Clay.com API Integration ---
     @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(aiohttp.ClientError))
     async def call_clay_api(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
         """Makes a direct API call to the specified Clay.com endpoint."""
         api_key = self.config.get_secret("CLAY_API_KEY")
         if not api_key:
             self.logger.error("Clay.com API key (CLAY_API_KEY) not found.")
             return {"status": "failure", "message": "Clay API key not configured."}
         # Ensure endpoint starts with / but handle if it already does
         if not endpoint.startswith('/'): endpoint = '/' + endpoint

         clay_url = f"https://api.clay.com{endpoint}"
         headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}

         await self._internal_think(f"Calling Clay API: {endpoint}", details=data)
         await self.log_operation('debug', f"Calling Clay API endpoint: {endpoint}")

         # Estimate cost based on endpoint (needs refinement based on actual Clay pricing)
         estimated_cost = 0.01 # Default low cost
         if "enrichment/person" in endpoint: estimated_cost = 0.05
         elif "enrichment/company" in endpoint: estimated_cost = 0.03
         elif "find_company_decision_makers" in endpoint: estimated_cost = 0.10 # Example

         try:
             timeout = aiohttp.ClientTimeout(total=90) # Increased timeout for potentially longer enrichments
             async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                 async with session.post(clay_url, json=data) as response:
                     response_status = response.status
                     try: response_data = await response.json(content_type=None)
                     except Exception: response_data = await response.text()

                     if 200 <= response_status < 300:
                         self.logger.info(f"Clay API call to {endpoint} successful (Status: {response_status}).")
                         if hasattr(self.orchestrator, 'report_expense'):
                             await self.orchestrator.report_expense(self.AGENT_NAME, estimated_cost, "API_Clay", f"Clay API Call: {endpoint}")
                         return {"status": "success", "data": response_data}
                     else:
                         self.logger.error(f"Clay API call to {endpoint} failed. Status: {response_status}, Response: {str(response_data)[:500]}...")
                         return {"status": "failure", "message": f"Clay API Error (Status {response_status})", "details": response_data}
         except asyncio.TimeoutError:
             self.logger.error(f"Timeout calling Clay API endpoint: {endpoint}")
             return {"status": "error", "message": f"Clay API call timed out"}
         except aiohttp.ClientError as e:
             self.logger.error(f"Network/Connection error calling Clay API endpoint {endpoint}: {e}")
             raise
         except Exception as e:
             self.logger.error(f"Unexpected error during Clay API call to {endpoint}: {e}", exc_info=True)
             return {"status": "error", "message": f"Clay API call exception: {e}"}


     # --- Knowledge Base Interface Implementation (Direct Postgres) ---
     async def log_knowledge_fragment(self, agent_source: str, data_type: str, content: Union[str, dict], relevance_score: float = 0.5, tags: Optional[List[str]] = None, related_client_id: Optional[int] = None, source_reference: Optional[str] = None) -> Optional[KnowledgeFragment]:
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
                         fragment = KnowledgeFragment(agent_source=agent_source, timestamp=now_ts, last_accessed_ts=now_ts, data_type=data_type, content=content_str, item_hash=content_hash, relevance_score=relevance_score, tags=tags_str, related_client_id=related_client_id, source_reference=source_reference)
                         session.add(fragment)
                 if fragment:
                     await session.refresh(fragment)
                     self.logger.info(f"Logged KnowledgeFragment: ID={fragment.id}, Hash={content_hash[:8]}..., Type={data_type}, Source={agent_source}")
                     return fragment
                 else:
                     return None
         except (SQLAlchemyError, TypeError) as e: self.logger.error(f"Error logging KF: {e}", exc_info=True); await self._report_error(f"Error logging KF: {e}"); return None
         except Exception as e: self.logger.error(f"Unexpected error logging KF: {e}", exc_info=True); return None

     async def query_knowledge_base(self, data_types: Optional[List[str]] = None, tags: Optional[List[str]] = None, min_relevance: float = 0.0, time_window: Optional[timedelta] = None, limit: int = 100, related_client_id: Optional[int] = None, content_query: Optional[str] = None) -> List[KnowledgeFragment]:
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

                 stmt_ids = stmt.with_only_columns(KnowledgeFragment.id).order_by(desc(KnowledgeFragment.last_accessed_ts), desc(KnowledgeFragment.relevance_score), desc(KnowledgeFragment.timestamp)).limit(limit)
                 fragment_ids = (await session.execute(stmt_ids)).scalars().all()
                 if not fragment_ids: return []

                 stmt_final = select(KnowledgeFragment).where(KnowledgeFragment.id.in_(fragment_ids)).order_by(desc(KnowledgeFragment.last_accessed_ts), desc(KnowledgeFragment.relevance_score), desc(KnowledgeFragment.timestamp))
                 fragments = list((await session.execute(stmt_final)).scalars().all())

                 if fragment_ids:
                     async def update_access_time():
                         try:
                             async with self.session_maker() as update_session:
                                  async with update_session.begin():
                                     await update_session.execute(update(KnowledgeFragment).where(KnowledgeFragment.id.in_(fragment_ids)).values(last_accessed_ts=datetime.now(timezone.utc)))
                                 self.logger.debug(f"Updated last_accessed_ts for {len(fragment_ids)} fragments.")
                         except Exception as update_err: self.logger.error(f"Failed update last_accessed_ts: {update_err}")
                     asyncio.create_task(update_access_time())
                 self.logger.debug(f"KB query returned {len(fragments)} fragments.")
         except SQLAlchemyError as e: self.logger.error(f"DB Error querying KB: {e}", exc_info=True); await self._report_error(f"DB Error querying KB: {e}")
         except Exception as e: self.logger.error(f"Unexpected error querying KB: {e}", exc_info=True)
         return fragments

     async def log_learned_pattern(self, pattern_description: str, supporting_fragment_ids: List[int], confidence_score: float, implications: str, tags: Optional[List[str]] = None) -> Optional[LearnedPattern]:
         if not self.session_maker: self.logger.error("DB session_maker not available."); return None
         try:
             fragment_ids_str = json.dumps(sorted(list(set(supporting_fragment_ids)))); tags_list = sorted(list(set(tags))) if tags else []; tags_str = json.dumps(tags_list) if tags_list else None
             pattern = LearnedPattern(timestamp=datetime.now(timezone.utc), pattern_description=pattern_description, supporting_fragment_ids=fragment_ids_str, confidence_score=confidence_score, implications=implications, tags=tags_str, status='active')
             async with self.session_maker() as session:
                 async with session.begin(): session.add(pattern)
                 await session.refresh(pattern)
                 self.logger.info(f"Logged LearnedPattern: ID={pattern.id}, Confidence={confidence_score:.2f}")
                 return pattern
         except SQLAlchemyError as e: self.logger.error(f"DB Error logging LearnedPattern: {e}", exc_info=True); await self._report_error(f"DB Error logging LearnedPattern: {e}"); return None
         except Exception as e: self.logger.error(f"Unexpected error logging LearnedPattern: {e}", exc_info=True); return None

     async def get_latest_patterns(self, tags: Optional[List[str]] = None, min_confidence: float = 0.7, limit: int = 10) -> List[LearnedPattern]:
         if not self.session_maker: self.logger.error("DB session_maker not available."); return []
         patterns = []
         try:
             async with self.session_maker() as session:
                 stmt = select(LearnedPattern).where(LearnedPattern.confidence_score >= min_confidence, LearnedPattern.status == 'active')
                 if tags: tag_conditions = [LearnedPattern.tags.like(f'%"{tag}"%') for tag in tags]; stmt = stmt.where(or_(*tag_conditions))
                 stmt = stmt.order_by(desc(LearnedPattern.timestamp)).limit(limit)
                 patterns = list((await session.execute(stmt)).scalars().all())
                 self.logger.debug(f"Fetched {len(patterns)} learned patterns (min_confidence={min_confidence}).")
         except SQLAlchemyError as e: self.logger.error(f"DB Error getting latest patterns: {e}", exc_info=True); await self._report_error(f"DB Error getting patterns: {e}")
         except Exception as e: self.logger.error(f"Unexpected error getting latest patterns: {e}", exc_info=True)
         return patterns

     async def purge_old_knowledge(self, days_threshold: Optional[int] = None):
         if not self.session_maker: self.logger.error("DB session_maker not available."); return
         threshold = days_threshold if days_threshold is not None else settings.DATA_PURGE_DAYS_THRESHOLD
         if threshold <= 0: self.logger.warning("Invalid days_threshold for purge."); return
         purge_cutoff_date = datetime.now(timezone.utc) - timedelta(days=threshold)
         self.logger.info(f"Purging knowledge fragments last accessed before {purge_cutoff_date.isoformat()}...")
         purge_thought = f"Structured Thinking: Purge Old Knowledge (Threshold: {threshold} days). Plan: Execute DELETE on knowledge_fragments where last_accessed_ts < {purge_cutoff_date}. Log count."
         await self._internal_think(purge_thought); deleted_count = 0
         try:
             async with self.session_maker() as session:
                 async with session.begin():
                     stmt = delete(KnowledgeFragment).where(KnowledgeFragment.last_accessed_ts < purge_cutoff_date)
                     result = await session.execute(stmt); deleted_count = result.rowcount
             self.logger.info(f"Successfully purged {deleted_count} old knowledge fragments.")
             if deleted_count > 0 and self.orchestrator and hasattr(self.orchestrator, 'send_notification'):
                  await self.orchestrator.send_notification("Data Purge Completed", f"Purged {deleted_count} knowledge fragments older than {threshold} days.")
         except SQLAlchemyError as e: self.logger.error(f"DB Error purging old knowledge: {e}", exc_info=True); await self._report_error(f"DB Error purging old knowledge: {e}")
         except Exception as e: self.logger.error(f"Unexpected error purging old knowledge: {e}", exc_info=True)

     async def handle_feedback(self, insights_data: Dict[str, Dict[str, Any]]):
         """Processes feedback insights, logs them, and triggers analysis/actions."""
         self.logger.info(f"ThinkTool received feedback insights from {len(insights_data)} agents.")
         feedback_thought = f"Structured Thinking: Process Agent Feedback. Plan: Log raw feedback -> Format summary -> Call LLM for analysis -> Parse response -> Create directives/schedule critiques/log insights."
         await self._internal_think(feedback_thought)
         for agent_name, agent_feedback in insights_data.items():
             if isinstance(agent_feedback, dict):
                 try:
                     tags = ["feedback", agent_name.lower()]; is_error = agent_feedback.get("status") == "error" or agent_feedback.get("errors_encountered_session", 0) > 0
                     if is_error: tags.append("error")
                     await self.log_knowledge_fragment(agent_source=agent_name, data_type="AgentFeedbackRaw", content=agent_feedback, tags=tags, relevance_score=0.5)
                 except Exception as e: self.logger.error(f"Error logging raw feedback for {agent_name}: {e}", exc_info=True)
             else: self.logger.warning(f"Received non-dict feedback from {agent_name}.")

         feedback_summary = json.dumps(insights_data, indent=2, default=str, ensure_ascii=False)
         max_summary_len = 4000; feedback_summary = feedback_summary[:max_summary_len] + ("..." if len(feedback_summary) > max_summary_len else "")
         task_context = {
             "task": "Analyze Agent Feedback",
             "feedback_data": feedback_summary,
             "desired_output_format": "JSON ONLY: {{\"analysis_summary\": str, \"critical_issues_found\": [str], \"key_successes_noted\": [str], \"proposed_directives\": [{{...}}], \"prompts_to_critique\": [str], \"insights_to_log\": [{{...}}]}}"
         }
         analysis_prompt = await self.generate_dynamic_prompt(task_context)
         llm_model_pref = settings.OPENROUTER_MODELS.get('think_synthesize')
         analysis_json = await self._call_llm_with_retry(analysis_prompt, temperature=0.6, max_tokens=2000, is_json_output=True, model=llm_model_pref)
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
                                  self.logger.info(f"Generated Directive from Feedback for {d_data['target_agent']}")
                              else: self.logger.warning(f"Skipping invalid directive data: {d_data}")
                 for prompt_id in analysis_result.get('prompts_to_critique', []):
                     if isinstance(prompt_id, str):
                         try: agent_name, prompt_key = prompt_id.split('/', 1); asyncio.create_task(self.self_critique_prompt(agent_name, prompt_key, f"Feedback analysis suggested issues: {analysis_result.get('analysis_summary', 'N/A')}")); self.logger.info(f"Scheduled critique for: {prompt_id}")
                         except ValueError: self.logger.warning(f"Invalid prompt identifier format: {prompt_id}")
                         except Exception as critique_err: self.logger.error(f"Error scheduling critique for {prompt_id}: {critique_err}")
                     else: self.logger.warning(f"Invalid prompt identifier type: {type(prompt_id)}")
                 for frag_data in analysis_result.get('insights_to_log', []):
                      if isinstance(frag_data, dict) and all(k in frag_data for k in ['data_type', 'content']): await self.log_knowledge_fragment(agent_source="ThinkToolFeedback", data_type=frag_data['data_type'], content=frag_data['content'], tags=frag_data.get('tags', ['feedback_insight']), relevance_score=frag_data.get('relevance', 0.6))
                      else: self.logger.warning(f"Skipping invalid insight data: {frag_data}")
             except (json.JSONDecodeError, ValueError): self.logger.error(f"Failed decode/validate feedback analysis JSON: {analysis_json}")
             except Exception as e: self.logger.error(f"Error processing feedback analysis result: {e}", exc_info=True)
         else: self.logger.error("Feedback analysis failed (LLM error).")

     # --- Prompt Template Management ---
     async def get_prompt(self, agent_name: str, prompt_key: str) -> Optional[str]:
         if not self.session_maker: self.logger.error("DB session_maker not available."); return None
         self.logger.debug(f"Querying DB for active prompt: {agent_name}/{prompt_key}.")
         try:
             async with self.session_maker() as session:
                 stmt = select(PromptTemplate.content).where(PromptTemplate.agent_name == agent_name, PromptTemplate.prompt_key == prompt_key, PromptTemplate.is_active == True).order_by(desc(PromptTemplate.version)).limit(1)
                 prompt_content = (await session.execute(stmt)).scalar_one_or_none()
                 if prompt_content: self.logger.info(f"Fetched active prompt for {agent_name}/{prompt_key}"); return prompt_content
                 else: self.logger.warning(f"No active prompt found for {agent_name}/{prompt_key}"); return None
         except SQLAlchemyError as e: self.logger.error(f"DB Error getting prompt {agent_name}/{prompt_key}: {e}", exc_info=True); await self._report_error(f"DB Error getting prompt {agent_name}/{prompt_key}: {e}"); return None
         except Exception as e: self.logger.error(f"Unexpected error getting prompt: {e}", exc_info=True); return None

     async def update_prompt(self, agent_name: str, prompt_key: str, new_content: str, author_agent: str = "ThinkTool") -> Optional[PromptTemplate]:
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
                         await session.execute(stmt_deactivate); self.logger.info(f"Deactivated prompt v{current_version} for {agent_name}/{prompt_key}")
                     new_template = PromptTemplate(agent_name=agent_name, prompt_key=prompt_key, version=new_version, content=new_content, is_active=True, author_agent=author_agent, last_updated=datetime.now(timezone.utc))
                     session.add(new_template)
                 await session.refresh(new_template)
                 self.logger.info(f"Created and activated new prompt v{new_version} for {agent_name}/{prompt_key}")
                 return new_template
         except SQLAlchemyError as e: self.logger.error(f"DB Error updating prompt {agent_name}/{prompt_key}: {e}", exc_info=True); await self._report_error(f"DB Error updating prompt {agent_name}/{prompt_key}: {e}"); return None
         except Exception as e: self.logger.error(f"Unexpected error updating prompt: {e}", exc_info=True); return None

     # --- Enhanced Reflection & Validation ---
     async def reflect_on_action(self, context: str, agent_name: str, task_description: str) -> dict:
         """Adds the 'pause and think' capability before finalizing actions or plans."""
         self.logger.debug(f"Starting reflection for {agent_name} on task: {task_description}")
         reflect_thought = f"Structured Thinking: Reflect on Action for {agent_name}. Task: {task_description[:50]}... Plan: Fetch KB context -> Format prompt -> Call LLM -> Process response -> Trigger KB updates -> Return result."
         await self._internal_think(reflect_thought); kb_context = ""
         try:
             # Fetch context (same as before)
             active_directives = await self.get_active_directives(target_agent=agent_name)
             task_keywords = [w for w in re.findall(r'\b\w{4,}\b', task_description.lower()) if w not in ['task', 'agent', 'perform', 'execute']]
             query_tags = [agent_name.lower(), 'strategy', 'feedback'] + task_keywords[:3]
             relevant_fragments = await self.query_knowledge_base(tags=query_tags, limit=10, time_window=timedelta(days=7))
             relevant_patterns = await self.get_latest_patterns(tags=[agent_name.lower()], limit=5)
             if active_directives: kb_context += "\n\n**Active Directives:**\n" + "\n".join([f"- ID {d.id}: {d.content[:100]}..." for d in active_directives])
             if relevant_fragments: kb_context += "\n\n**Recent Relevant Fragments:**\n" + "\n".join([f"- ID {f.id} ({f.data_type}): {f.content[:80]}..." for f in relevant_fragments])
             if relevant_patterns: kb_context += "\n\n**Relevant Learned Patterns:**\n" + "\n".join([f"- ID {p.id}: {p.pattern_description[:150]}..." for p in relevant_patterns])
         except Exception as e: self.logger.error(f"Error fetching KB context for reflection: {e}"); kb_context = "\n\n**Warning:** Failed KB context retrieval."

         # Use generate_dynamic_prompt for consistency
         task_context = {
             "task": "Reflect on Proposed Action/Plan",
             "agent_name": agent_name,
             "task_description": task_description,
             "agent_provided_context": context,
             "knowledge_base_context": kb_context or "None",
             "desired_output_format": "JSON ONLY: {{\"proceed\": bool, \"reason\": str, \"risk_level\": str, \"compliance_flags\": [str], \"next_step\": str, \"confidence\": float, \"log_fragment\": {{...}}?, \"update_directive\": {{...}}?}}"
         }
         prompt = await self.generate_dynamic_prompt(task_context)

         llm_model_pref = settings.OPENROUTER_MODELS.get('think_validate')
         reflection_json = await self._call_llm_with_retry(prompt, temperature=0.3, max_tokens=1000, is_json_output=True, model=llm_model_pref)

         if reflection_json:
             try:
                 reflection = self._parse_llm_json(reflection_json)
                 if not reflection: raise ValueError("Failed parse reflection JSON.")
                 reflection.setdefault('proceed', False); reflection.setdefault('reason', 'Analysis failed.'); reflection.setdefault('risk_level', 'Unknown'); reflection.setdefault('compliance_flags', []); reflection.setdefault('next_step', 'Manual review.'); reflection.setdefault('confidence', 0.5)
                 self.logger.info(f"Reflection for {agent_name}: Proceed={reflection['proceed']}, Risk={reflection['risk_level']}, Reason={reflection['reason']}")
                 if 'log_fragment' in reflection and isinstance(reflection['log_fragment'], dict): frag_data = reflection['log_fragment']; await self.log_knowledge_fragment(agent_source="ThinkToolReflection", data_type=frag_data.get('data_type', 'reflection_insight'), content=frag_data.get('content', {}), tags=frag_data.get('tags', ['reflection']), relevance_score=frag_data.get('relevance', 0.6))
                 if 'update_directive' in reflection and isinstance(reflection['update_directive'], dict): directive_data = reflection['update_directive']; await self.update_directive_status(directive_id=directive_data['directive_id'], status=directive_data['status'], result_summary=directive_data.get('result_summary'))
                 return reflection
             except (json.JSONDecodeError, ValueError): self.logger.error(f"Failed decode/validate JSON reflection: {reflection_json}")
             except Exception as e: self.logger.error(f"Error processing reflection result: {e}", exc_info=True)
         return {"proceed": False, "reason": "ThinkTool analysis failed.", "risk_level": "Critical", "compliance_flags": ["Analysis Failure"], "next_step": "Halt task.", "confidence": 0.0}

     async def validate_output(self, output_to_validate: str, validation_criteria: str, agent_name: str, context: str = None) -> dict:
         # (Keep existing implementation)
         self.logger.debug(f"Starting validation for {agent_name}'s output.")
         validate_thought = f"Structured Thinking: Validate Output from {agent_name}. Criteria: {validation_criteria[:50]}... Plan: Fetch patterns -> Format prompt -> Call LLM -> Parse -> Return."
         await self._internal_think(validate_thought); pattern_context = ""
         try:
             criteria_tags = [w for w in re.findall(r'\b\w{4,}\b', validation_criteria.lower()) if w not in ['validate', 'check', 'ensure']]
             query_tags = [agent_name.lower()] + criteria_tags[:3]
             relevant_patterns = await self.get_latest_patterns(tags=query_tags, limit=5, min_confidence=0.6)
             if relevant_patterns: pattern_context += "\n\n**Relevant Learned Patterns (Consider these):**\n" + "\n".join([f"- ID {p.id}: {p.pattern_description}" for p in relevant_patterns])
         except Exception as e: self.logger.error(f"Error fetching patterns for validation: {e}"); pattern_context = "\n\n**Warning:** Failed pattern retrieval."

         task_context = {
             "task": "Validate Agent Output",
             "agent_name": agent_name,
             "agent_context": context or "N/A",
             "output_to_validate": output_to_validate,
             "validation_criteria": validation_criteria,
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
                 self.logger.info(f"Validation for {agent_name}: Valid={validation['valid']}, Feedback={validation['feedback'][:100]}...")
                 return validation
             except (json.JSONDecodeError, ValueError): self.logger.error(f"Failed decode/validate JSON validation: {validation_json}")
             except Exception as e: self.logger.error(f"Error processing validation result: {e}", exc_info=True)
         return {"valid": False, "feedback": "ThinkTool validation failed."}

     # --- Core Synthesis & Strategy Engines ---
     async def synthesize_insights_and_strategize(self):
         """Synthesizes KB data, performance metrics, and generates strategic directives, including Clay.com enrichment."""
         self.logger.info("ThinkTool: Starting synthesis and strategy cycle.")
         synth_thought = "Structured Thinking: Synthesize & Strategize. Plan: Query KB (fragments, patterns, perf, clients needing enrichment) -> Format -> Call LLM -> Parse -> Store outputs (patterns, directives [incl. Clay], opportunities)."
         await self._internal_think(synth_thought)
         proposed_directives = []
         try:
             async with self.session_maker() as session:
                 # Query data (fragments, patterns, performance - existing logic)
                 frags_stmt = select(KnowledgeFragment).where(KnowledgeFragment.last_accessed_ts >= datetime.now(timezone.utc) - timedelta(days=7)).order_by(desc(KnowledgeFragment.relevance_score), desc(KnowledgeFragment.last_accessed_ts)).limit(200)
                 recent_fragments = list((await session.execute(frags_stmt)).scalars().all())
                 patt_stmt = select(LearnedPattern).where(LearnedPattern.status == 'active').order_by(desc(LearnedPattern.confidence_score), desc(LearnedPattern.timestamp)).limit(20)
                 recent_patterns = list((await session.execute(patt_stmt)).scalars().all())
                 perf_data = [] # Rebuild perf data query as needed
                 email_perf_stmt = select(EmailLog.status, EmailLog.timestamp).where(EmailLog.timestamp >= datetime.now(timezone.utc) - timedelta(days=7)).limit(500)
                 perf_data.extend([{"type": "email", "status": r.status, "ts": r.timestamp} for r in (await session.execute(email_perf_stmt)).mappings().all()])
                 call_perf_stmt = select(CallLog.outcome, CallLog.timestamp).where(CallLog.timestamp >= datetime.now(timezone.utc) - timedelta(days=7)).limit(500)
                 perf_data.extend([{"type": "call", "outcome": r.outcome, "ts": r.timestamp} for r in (await session.execute(call_perf_stmt)).mappings().all()])
                 inv_perf_stmt = select(Invoice.status, Invoice.amount, Invoice.timestamp).where(Invoice.timestamp >= datetime.now(timezone.utc) - timedelta(days=7)).limit(200)
                 perf_data.extend([{"type": "invoice", "status": r.status, "amount": r.amount, "ts": r.timestamp} for r in (await session.execute(inv_perf_stmt)).mappings().all()])
                 client_score_stmt = select(Client.engagement_score).where(Client.opt_in == True).order_by(func.random()).limit(100)
                 perf_data.extend([{"type": "client_score", "score": r.engagement_score} for r in (await session.execute(client_score_stmt)).mappings().all()])

                 # --- MODIFIED: Query clients needing enrichment (more broadly) ---
                 # Find clients with LinkedIn but missing key data (email, title, company)
                 # Prioritize those not contacted recently
                 clients_needing_enrichment_stmt = select(Client.id, Client.name, Client.source_reference, Client.email, Client.company, Client.job_title).where(
                     Client.opt_in == True,
                     Client.is_deliverable == True,
                     Client.source_reference.like('%linkedin.com/in/%'), # Must have LinkedIn
                     or_( # Missing at least one key piece of data
                         Client.email == None,
                         Client.company == None,
                         Client.job_title == None
                     ),
                     # Optional: Prioritize those not contacted recently
                     # Client.last_contacted_at < (datetime.now(timezone.utc) - timedelta(days=14))
                 ).order_by(asc(Client.last_interaction)).limit(15) # Limit batch size
                 clients_for_clay = (await session.execute(clients_needing_enrichment_stmt)).mappings().all()
                 self.logger.info(f"Found {len(clients_for_clay)} clients potentially needing Clay enrichment.")

                 # Generate Clay directives for these clients
                 for client_data in clients_for_clay:
                     linkedin_url = client_data.get('source_reference')
                     if linkedin_url:
                         # Determine which enrichment is needed
                         # Using enrich_person as it often includes email, company, title
                         clay_endpoint = self.clay_endpoints.get("enrich_person")
                         clay_data = {"linkedin_url": linkedin_url}
                         reason = "Find missing email/title/company"

                         proposed_directives.append({
                             "target_agent": "ThinkTool", # Clay calls executed by ThinkTool
                             "directive_type": "execute_clay_call",
                             "content": {
                                 "endpoint": clay_endpoint,
                                 "data": clay_data,
                                 "context": {"client_id": client_data['id'], "reason": reason},
                                 "source_reference": linkedin_url # Pass LinkedIn URL as reference
                             }, "priority": 6 # Medium priority
                         })
                         self.logger.info(f"Generated Clay directive ({reason}) for client {client_data['id']} ({linkedin_url})")

             # Prepare context for LLM synthesis
             if not recent_fragments and not perf_data and not proposed_directives: self.logger.warning("ThinkTool Synthesis: Insufficient recent data or Clay candidates."); return
             fragments_summary = [{"id": f.id, "type": f.data_type, "src": f.agent_source, "preview": (f.content if isinstance(f.content, str) else json.dumps(f.content))[:80]+"..."} for f in recent_fragments[:20]]
             patterns_summary = [{"id": p.id, "desc": p.pattern_description, "conf": p.confidence_score} for p in recent_patterns]
             perf_counts = Counter(f'{d["type"]}_{d.get("status") or d.get("outcome")}' for d in perf_data if d.get("status") or d.get("outcome"))
             paid_invoice_total = sum(d['amount'] for d in perf_data if d['type'] == 'invoice' and d['status'] == 'paid')
             avg_score = np.mean([d['score'] for d in perf_data if d['type'] == 'client_score' and d.get('score') is not None]) if any(d['type'] == 'client_score' for d in perf_data) else 0
             perf_summary_str = f"Counts: {dict(perf_counts)}. Paid Invoice Total (7d): ${paid_invoice_total:.2f}. Avg Client Score Sample: {avg_score:.2f}"
             clay_directives_summary = f"Generated {len(proposed_directives)} Clay.com directives for enrichment." if proposed_directives else "No new Clay.com directives generated."

             task_context = {
                 "task": "Synthesize Insights & Generate Strategy",
                 "recent_fragments_summary": fragments_summary,
                 "active_patterns_summary": patterns_summary,
                 "performance_summary_7d": perf_summary_str,
                 "clay_directives_status": clay_directives_summary,
                 "current_scoring_weights": self.scoring_weights,
                 "current_scoring_decay": self.scoring_decay_rate,
                 "desired_output_format": "JSON ONLY: {{\"new_patterns\": [], \"pattern_updates\": [], \"goal_assessment\": \"\", \"scoring_adjustments_suggestion\": str, \"proposed_directives\": [], \"identified_opportunities\": []}}"
             }
             synthesis_prompt = await self.generate_dynamic_prompt(task_context)
             llm_model_pref = settings.OPENROUTER_MODELS.get('think_strategize')
             synthesis_json = await self._call_llm_with_retry(synthesis_prompt, temperature=0.7, max_tokens=3500, is_json_output=True, model=llm_model_pref)

             if synthesis_json:
                 try:
                     synthesis_result = self._parse_llm_json(synthesis_json)
                     if not synthesis_result: raise ValueError("Failed parse synthesis JSON.")
                     self.logger.info(f"ThinkTool Synthesis cycle completed. Assessment: {synthesis_result.get('goal_assessment', 'N/A')}")
                     for p_data in synthesis_result.get('new_patterns', []): await self.log_learned_pattern(**p_data)
                     # Handle pattern updates if implemented
                     adjustment_suggestion = synthesis_result.get('scoring_adjustments_suggestion', '').lower()
                     if 'increase' in adjustment_suggestion and 'invoice_paid' in adjustment_suggestion: self.scoring_weights['invoice_paid'] = min(self.scoring_weights.get('invoice_paid', 5.0) * 1.1, 10.0); self.logger.info(f"Synthesizer suggested: Adjusted 'invoice_paid' weight to {self.scoring_weights['invoice_paid']:.2f}")

                     # Combine generated Clay directives with LLM proposed directives
                     all_directives = proposed_directives + synthesis_result.get('proposed_directives', [])

                     async with self.session_maker() as session:
                          async with session.begin():
                              for d_data in all_directives:
                                  if isinstance(d_data, dict) and all(k in d_data for k in ['target_agent', 'directive_type', 'content', 'priority']):
                                      dir_content = json.dumps(d_data['content']) if isinstance(d_data['content'], dict) else d_data['content']
                                      directive = StrategicDirective(source="ThinkToolSynthesis", timestamp=datetime.now(timezone.utc), status='pending', content=dir_content, **{k:v for k,v in d_data.items() if k != 'content'})
                                      session.add(directive)
                                      self.logger.info(f"Storing Directive for {d_data['target_agent']}: {d_data['directive_type']}")
                                  else: self.logger.warning(f"Skipping invalid directive data: {d_data}")
                     for o_data in synthesis_result.get('identified_opportunities', []):
                          if isinstance(o_data, dict) and 'description' in o_data: await self.log_knowledge_fragment(agent_source="ThinkToolSynthesis", data_type="business_opportunity_signal", content=o_data, tags=o_data.get('tags', ['opportunity']), relevance_score=0.8)
                          else: self.logger.warning(f"Skipping invalid opportunity data: {o_data}")
                 except (json.JSONDecodeError, ValueError): self.logger.error(f"Failed decode/validate JSON synthesis result: {synthesis_json}")
                 except Exception as e: self.logger.error(f"Error processing synthesis result: {e}", exc_info=True)
             else: self.logger.error("Synthesis analysis failed (LLM error).")
         except Exception as e: self.logger.error(f"Error during synthesis cycle: {e}", exc_info=True); await self._report_error(f"Synthesis cycle failed: {e}")

     async def technology_radar(self):
         # (Keep existing implementation)
         self.logger.info("ThinkTool: Starting technology radar cycle.")
         radar_thought = "Structured Thinking: Technology Radar. Plan: Define topics -> Delegate search (BrowseAgent) -> Analyze results (LLM) -> Log findings/directives (KB)."
         await self._internal_think(radar_thought)
         try:
             search_topics = ["Clay.com alternative API", "AI cold email personalization", "Open source voice cloning sales", "Email spam filter bypass techniques", "Automated UGC video platforms", "Best scraping proxy providers 2025", "Autonomous AI agent frameworks", "B2B SaaS competitor analysis tools"]
             search_query = f"Latest developments, tools, or research papers on: {random.choice(search_topics)}"; search_results_summary = None
             if self.orchestrator and hasattr(self.orchestrator, 'delegate_task'):
                  search_task = {"action": "perform_search_and_summarize", "query": search_query, "num_results": 5}; Browse_agent_name = "BrowsingAgent"
                  if Browse_agent_name not in self.orchestrator.agents: self.logger.error(f"{Browse_agent_name} not found."); return
                  search_result = await self.orchestrator.delegate_task(Browse_agent_name, search_task)
                  if search_result and search_result.get("status") == "success": search_results_summary = search_result.get("summary"); self.logger.info(f"Radar: Received search summary for '{search_query}'.")
                  else: self.logger.warning(f"Radar: BrowseAgent search failed: {search_result.get('message') if search_result else 'No result'}")
             else: self.logger.error("Radar: Orchestrator unavailable."); return
             if not search_results_summary: self.logger.info("Radar: No findings from web search."); return

             task_context = {
                 "task": "Analyze Technology Scouting Report",
                 "report_summary": search_results_summary,
                 "analysis_focus": "Novelty/Impact, Relevance to Sales/UGC/Clay.com/Automation, Benefits, Risks (Technical/Legal/Ban), Integration Effort (Low/Med/High), Recommendation ('Log Insight', 'Generate Directive: [Brief Action]', 'Ignore').",
                 "desired_output_format": "JSON ONLY: {{\"analyzed_items\": [ {{ \"item_name\": str, \"summary\": str, \"relevance\": str, \"benefits\": [str], \"risks\": [str], \"integration_effort\": str, \"recommendation\": str }} ], \"overall_assessment\": \"Brief summary of findings.\"}}"
             }
             analysis_prompt = await self.generate_dynamic_prompt(task_context)
             llm_model_pref = settings.OPENROUTER_MODELS.get('think_radar')
             analysis_json = await self._call_llm_with_retry(analysis_prompt, temperature=0.4, max_tokens=1500, is_json_output=True, model=llm_model_pref)

             if analysis_json:
                 try:
                     analysis_result = self._parse_llm_json(analysis_json)
                     if not analysis_result: raise ValueError("Failed parse radar analysis JSON.")
                     self.logger.info(f"Radar analysis complete. Found {len(analysis_result.get('analyzed_items', []))} relevant items.")
                     async with self.session_maker() as session:
                         async with session.begin():
                             for item in analysis_result.get('analyzed_items', []):
                                 if not isinstance(item, dict) or not item.get("item_name"): continue
                                 await self.log_knowledge_fragment(agent_source="ThinkToolRadar", data_type="new_tool_discovery", content=item, tags=["technology", "scouting", item.get("item_name", "unknown").lower().replace(" ", "_")], relevance_score=0.7)
                                 if "directive" in item.get("recommendation", "").lower():
                                     directive_content = item.get("recommendation").split(":", 1)[-1].strip() if ":" in item.get("recommendation", "") else f"Investigate: {item.get('item_name')}"
                                     directive = StrategicDirective(source="ThinkToolRadar", timestamp=datetime.now(timezone.utc), target_agent="Orchestrator", directive_type="investigate_tool_technique", content=json.dumps({"item_name": item.get("item_name"), "details": item, "requested_action": directive_content}), priority=7, status='pending')
                                     session.add(directive)
                                     self.logger.info(f"Generated investigation directive for: {item.get('item_name', 'N/A')}")
                 except (json.JSONDecodeError, ValueError): self.logger.error(f"Radar: Failed decode/validate JSON analysis: {analysis_json}")
                 except Exception as e: self.logger.error(f"Radar: Error processing analysis result: {e}", exc_info=True)
             else: self.logger.error("Radar: Analysis failed (LLM error).")
         except Exception as e: self.logger.error(f"Error during technology radar cycle: {e}", exc_info=True); await self._report_error(f"Technology radar cycle failed: {e}")

     # --- Self-Improving Prompt Mechanism ---
     async def self_critique_prompt(self, agent_name: str, prompt_key: str, feedback_context: str):
         # (Keep existing implementation)
         self.logger.info(f"Starting self-critique for prompt: {agent_name}/{prompt_key}")
         critique_thought = f"Structured Thinking: Self-Critique Prompt {agent_name}/{prompt_key}. Plan: Fetch prompt -> Format critique prompt -> Call LLM -> Parse -> Update prompt -> Generate test directive."
         await self._internal_think(critique_thought)
         try:
             current_prompt = await self.get_prompt(agent_name, prompt_key)
             if not current_prompt: self.logger.error(f"Critique: Cannot find active prompt {agent_name}/{prompt_key}."); return

             task_context = {
                 "task": "Critique and Rewrite Prompt",
                 "agent_name": agent_name, "prompt_key": prompt_key,
                 "feedback_context": feedback_context,
                 "current_prompt": current_prompt,
                 "desired_output_format": "JSON ONLY: {{ \"critique\": \"Detailed critique.\", \"improved_prompt\": \"Complete rewritten prompt text.\" }}"
             }
             critique_prompt = await self.generate_dynamic_prompt(task_context)
             llm_model_pref = settings.OPENROUTER_MODELS.get('think_critique')
             critique_json = await self._call_llm_with_retry(critique_prompt, temperature=0.6, max_tokens=3000, is_json_output=True, model=llm_model_pref)

             if critique_json:
                 try:
                     critique_result = self._parse_llm_json(critique_json)
                     if not critique_result: raise ValueError("Failed parse critique JSON.")
                     improved_prompt = critique_result.get('improved_prompt'); critique_text = critique_result.get('critique')
                     if improved_prompt and isinstance(improved_prompt, str) and improved_prompt.strip():
                         self.logger.info(f"Critique generated improved prompt for {agent_name}/{prompt_key}. Critique: {critique_text}")
                         new_template = await self.update_prompt(agent_name, prompt_key, improved_prompt, author_agent="ThinkToolCritique")
                         if new_template:
                             async with self.session_maker() as session:
                                 async with session.begin():
                                     directive = StrategicDirective(source="ThinkToolCritique", timestamp=datetime.now(timezone.utc), target_agent=agent_name, directive_type="test_prompt_variation", content=json.dumps({"prompt_key": prompt_key, "new_version": new_template.version, "critique": critique_text, "old_prompt_preview": current_prompt[:200]+"..."}), priority=7, status='pending')
                                     session.add(directive)
                                 self.logger.info(f"Generated directive to test new prompt v{new_template.version} for {agent_name}/{prompt_key}")
                         else: self.logger.error(f"Critique: Failed save improved prompt {agent_name}/{prompt_key}.")
                     else: self.logger.warning(f"Critique for {agent_name}/{prompt_key} did not produce valid improved prompt.")
                 except (json.JSONDecodeError, ValueError): self.logger.error(f"Critique: Failed decode/validate JSON result: {critique_json}")
                 except Exception as e: self.logger.error(f"Critique: Error processing result: {e}", exc_info=True)
             else: self.logger.error(f"Critique: Failed get critique/rewrite from LLM for {agent_name}/{prompt_key}.")
         except Exception as e: self.logger.error(f"Error during self-critique for {agent_name}/{prompt_key}: {e}", exc_info=True); await self._report_error(f"Self-critique failed for {agent_name}/{prompt_key}: {e}")

     # --- Agent Run Loop ---
     async def run(self):
         # (Keep existing implementation)
         if self.status == self.STATUS_RUNNING: self.logger.warning("ThinkTool run() called while already running."); return
         self.logger.info("ThinkTool v5.4 starting run loop...")
         self._status = self.STATUS_RUNNING
         synthesis_interval = timedelta(seconds=int(self.config.get("THINKTOOL_SYNTHESIS_INTERVAL_SECONDS", 3600)))
         radar_interval = timedelta(seconds=int(self.config.get("THINKTOOL_RADAR_INTERVAL_SECONDS", 21600)))
         purge_interval = timedelta(seconds=int(self.config.get("DATA_PURGE_INTERVAL_SECONDS", 86400)))
         now = datetime.now(timezone.utc)
         self.last_synthesis_run = now - synthesis_interval; self.last_radar_run = now - radar_interval; self.last_purge_run = now - purge_interval
         while self.status == self.STATUS_RUNNING and not self._stop_event.is_set():
             try:
                 current_time = datetime.now(timezone.utc); is_approved = getattr(self.orchestrator, 'approved', False)
                 if is_approved:
                     if current_time - self.last_synthesis_run >= synthesis_interval: self.logger.info("ThinkTool: Triggering Synthesis & Strategy cycle."); await self.synthesize_insights_and_strategize(); self.last_synthesis_run = current_time
                     if current_time - self.last_radar_run >= radar_interval: self.logger.info("ThinkTool: Triggering Technology Radar cycle."); await self.technology_radar(); self.last_radar_run = current_time
                     if current_time - self.last_purge_run >= purge_interval: self.logger.info("ThinkTool: Triggering Data Purge cycle."); await self.purge_old_knowledge(); self.last_purge_run = current_time
                 else: self.logger.debug("ThinkTool: Orchestrator not approved. Skipping periodic tasks.")
                 await asyncio.sleep(60 * 5) # Check every 5 minutes
             except asyncio.CancelledError: self.logger.info("ThinkTool run loop cancelled."); break
             except Exception as e: self.logger.critical(f"ThinkTool: CRITICAL error in run loop: {e}", exc_info=True); self._status = self.STATUS_ERROR; await self._report_error(f"Critical run loop error: {e}"); await asyncio.sleep(60 * 15)
         if self.status != self.STATUS_STOPPING: self.status = self.STATUS_STOPPED
         self.logger.info("ThinkTool run loop finished.")

     # --- Abstract Method Implementations ---
     async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
         """Handles tasks delegated specifically to ThinkTool."""
         self._status = self.STATUS_EXECUTING
         action = task_details.get('action')
         result = {"status": "failure", "message": f"Unknown ThinkTool action: {action}"}
         self.logger.info(f"ThinkTool executing task: {action}")
         exec_thought = f"Structured Thinking: Execute Task '{action}'. Plan: Route action to appropriate method."
         await self._internal_think(exec_thought, details=task_details)
         task_id = task_details.get('id', 'N/A')

         try:
             # --- NEW: Add internal reflection step before complex actions ---
             if action in ['synthesize_insights_and_strategize', 'initiate_video_generation_workflow', 'plan_ugc_workflow', 'call_clay_api']: # Added Clay call here
                  reflection_context = f"About to execute complex action: {action}. Task Details: {json.dumps(task_details, default=str)[:500]}..."
                  reflection_result = await self.reflect_on_action(reflection_context, self.AGENT_NAME, f"Pre-execution check for {action}")
                  if not reflection_result.get('proceed', False):
                      self.logger.warning(f"Reflection advised against proceeding with action '{action}'. Reason: {reflection_result.get('reason')}")
                      # Update directive if applicable
                      if task_details.get('directive_id'): await self.update_directive_status(task_details['directive_id'], 'halted', f"Halted by reflection: {reflection_result.get('reason')}")
                      return {"status": "halted", "message": f"Action halted based on internal reflection: {reflection_result.get('reason')}"}
                  else:
                      self.logger.info(f"Reflection approved proceeding with action '{action}'.")

             # --- Route to specific handlers ---
             if action == 'reflect_on_action':
                 agent_name = task_details.get('agent_name'); task_desc = task_details.get('task_description'); context = task_details.get('context')
                 if context and agent_name and task_desc: reflection = await self.reflect_on_action(context, agent_name, task_desc); result = {"status": "success", "reflection": reflection}
                 else: result = {"status": "failure", "message": "Missing context/agent_name/task_description for reflection."}
             elif action == 'validate_output':
                 output = task_details.get('output_to_validate'); criteria = task_details.get('validation_criteria'); agent_name = task_details.get('agent_name'); context = task_details.get('context')
                 if output and criteria and agent_name: validation = await self.validate_output(output, criteria, agent_name, context); result = {"status": "success", "validation": validation}
                 else: result = {"status": "failure", "message": "Missing output/criteria/agent_name for validation."}
             elif action == 'process_feedback' or action == 'process_external_feedback': # Handle both internal and external
                  feedback_data = task_details.get('feedback_data') or task_details # Allow passing data directly for external
                  if feedback_data: await self.handle_feedback(feedback_data); result = {"status": "success", "message": "Feedback processed."}
                  else: result = {"status": "failure", "message": "Missing feedback_data for processing."}
             elif action == 'generate_educational_content':
                  topic = task_details.get('topic'); context = task_details.get('context')
                  if topic: explanation = await self.generate_educational_content(topic, context); result = {"status": "success" if explanation else "failure", "explanation": explanation}
                  else: result = {"status": "failure", "message": "Missing topic for educational content."}
             elif action == 'execute_clay_call': # Changed from call_clay_api to match directive
                  params = task_details.get('content', {}) # Assume content holds params
                  endpoint = params.get('endpoint'); data = params.get('data')
                  source_ref = params.get('source_reference')
                  client_id = params.get('context', {}).get('client_id')
                  original_directive_id = task_details.get('directive_id') # Get ID if it's from a directive

                  if endpoint and data:
                      clay_api_result = await self.call_clay_api(endpoint=endpoint, data=data) # Call the actual API method
                      # Delegate processing immediately
                      processing_task = { "action": "process_clay_result", "clay_data": clay_api_result, "source_directive_id": original_directive_id, "source_reference": source_ref, "client_id": client_id, "priority": 5 }
                      await self.orchestrator.delegate_task(self.AGENT_NAME, processing_task)
                      result = {"status": "success", "message": "Clay API call executed, processing task delegated."}
                  else:
                      result = {"status": "failure", "message": "Missing 'endpoint' or 'data' in content for execute_clay_call task."}
                      if original_directive_id: await self.update_directive_status(original_directive_id, 'failed', 'Missing endpoint/data')
             elif action == 'process_clay_result':
                  clay_data = task_details.get('clay_data'); source_directive_id = task_details.get('source_directive_id'); source_reference = task_details.get('source_reference'); client_id = task_details.get('client_id')
                  if clay_data: await self._process_clay_result(clay_data, source_directive_id, source_reference, client_id); result = {"status": "success", "message": "Clay result processing initiated."}
                  else:
                      result = {"status": "failure", "message": "Missing clay_data for processing."}
                      if source_directive_id: await self.update_directive_status(source_directive_id, 'failed', 'Missing clay_data')
             elif action == 'log_knowledge_fragment':
                  frag_data = task_details.get('fragment_data', {})
                  if all(k in frag_data for k in ['agent_source', 'data_type', 'content']): frag = await self.log_knowledge_fragment(**frag_data); result = {"status": "success" if frag else "failure", "fragment_id": frag.id if frag else None}
                  else: result = {"status": "failure", "message": "Missing required keys for log_knowledge_fragment."}
             elif action == 'calculate_dynamic_price':
                  client_id = task_details.get('client_id'); conv_summary = task_details.get('conversation_summary'); base_price = task_details.get('base_price', 7000.0) # Use new base price
                  if client_id: price = await self._calculate_dynamic_price(client_id, conv_summary, base_price); result = {"status": "success", "price": price}
                  else: result = {"status": "failure", "message": "Missing client_id for dynamic pricing."}
             elif action == 'initiate_video_generation_workflow': # NEW Action Handler
                  params = task_details.get('params', {})
                  plan = await self._plan_video_workflow(params)
                  if plan:
                      result = {"status": "success", "message": "Video generation plan created.", "plan": plan}
                      # Optionally: Trigger first step delegation
                      # if plan: await self.orchestrator.delegate_task(plan[0]['target_agent'], plan[0]['task_details'])
                  else:
                      result = {"status": "failure", "message": "Failed to create video generation plan."}
             elif action == 'plan_ugc_workflow': # Existing action, ensure it's handled
                  # Add planning logic similar to video workflow if needed, or reuse
                  result = {"status": "pending", "message": "UGC workflow planning not fully implemented yet."} # Placeholder
             else:
                 self.logger.warning(f"Unhandled action in ThinkTool.execute_task: {action}")
                 result = {"status": "failure", "message": f"ThinkTool does not handle action: {action}"}

         except Exception as e:
              self.logger.error(f"Error executing ThinkTool task '{action}' (ID: {task_id}): {e}", exc_info=True)
              result = {"status": "error", "message": f"Exception during task '{action}': {e}"}
              await self._report_error(f"Error executing task '{action}': {e}", task_id=task_id)
              # Update directive if applicable
              if task_details.get('directive_id'): await self.update_directive_status(task_details['directive_id'], 'failed', f"Exception: {e}")
         finally:
              self._status = self.STATUS_IDLE

         return result

     async def _plan_video_workflow(self, params: Dict[str, Any]) -> Optional[List[Dict]]:
         """Generates the plan for the Descript/AIStudio video workflow."""
         self.logger.info(f"Planning video generation workflow with params: {params}")
         await self._internal_think("Planning video workflow: Descript UI + AIStudio Images", details=params)

         # --- Placeholder Implementation ---
         # This needs to be replaced with actual logic, potentially using LLM planning
         # based on the specific script, style, and tool capabilities.
         self.logger.warning("Video workflow planning is currently a placeholder. Needs full implementation.")
         # Example stub plan (needs real selectors, logic, error handling)
         plan = [
             {"step": 1, "target_agent": "BrowsingAgent", "task_details": {"action": "web_ui_automate", "service": "AIStudio", "goal": "Generate image based on prompt: 'futuristic cityscape'", "output_var": "generated_image_path"}},
             {"step": 2, "target_agent": "BrowsingAgent", "task_details": {"action": "web_ui_automate", "service": "Descript", "goal": "Login and start new project"}},
             {"step": 3, "target_agent": "BrowsingAgent", "task_details": {"action": "web_ui_automate", "service": "Descript", "goal": "Upload base video from /app/assets/base_video.mp4", "input_vars": {"video_path": "/app/assets/base_video.mp4"}}}, # Needs path
             {"step": 4, "target_agent": "BrowsingAgent", "task_details": {"action": "web_ui_automate", "service": "Descript", "goal": "Upload generated image", "input_vars": {"image_path": "$generated_image_path"}}},
             {"step": 5, "target_agent": "BrowsingAgent", "task_details": {"action": "web_ui_automate", "service": "Descript", "goal": "Add image to timeline at 0:05"}},
             {"step": 6, "target_agent": "BrowsingAgent", "task_details": {"action": "web_ui_automate", "service": "Descript", "goal": "Export video (default settings)"}},
             {"step": 7, "target_agent": "BrowsingAgent", "task_details": {"action": "web_ui_automate", "service": "Descript", "goal": "Download exported video", "output_var": "final_video_path"}},
             {"step": 8, "target_agent": "Orchestrator", "task_details": {"action": "store_artifact", "source_path": "$final_video_path", "destination": "/app/output/videos/"}},
         ]
         await self.log_knowledge_fragment(
             agent_source=self.AGENT_NAME, data_type="workflow_plan",
             content={"workflow_type": "video_generation", "plan": plan, "status": "placeholder"},
             tags=["video", "plan", "placeholder"], relevance_score=0.5
         )
         return plan # Return the placeholder plan for now


     async def learning_loop(self):
         # (Keep existing implementation)
         self.logger.info("ThinkTool learning_loop: Core learning logic is in run() periodic tasks.")
         while self.status == self.STATUS_RUNNING and not self._stop_event.is_set(): await asyncio.sleep(3600)

     async def self_critique(self) -> Dict[str, Any]:
         # (Keep existing implementation)
         self.logger.info("ThinkTool: Performing self-critique.")
         critique = {"status": "ok", "feedback": "Critique pending analysis."}
         critique_thought = "Structured Thinking: Self-Critique ThinkTool. Plan: Query DB stats -> Analyze -> Format -> Return."
         await self._internal_think(critique_thought)
         try:
             async with self.session_maker() as session:
                 kf_count = await session.scalar(select(func.count(KnowledgeFragment.id))) or 0
                 pattern_count = await session.scalar(select(func.count(LearnedPattern.id))) or 0
                 directive_counts_res = await session.execute(select(StrategicDirective.status, func.count(StrategicDirective.id)).group_by(StrategicDirective.status))
                 directive_status = {row.status: row[1] for row in directive_counts_res.mappings().all()}
             critique['kb_stats'] = {'fragments': kf_count, 'patterns': pattern_count}
             critique['directive_stats'] = directive_status
             feedback = f"KB Size: {kf_count} fragments, {pattern_count} patterns. Directives: {directive_status}. "
             failed_directives = directive_status.get('failed', 0)
             total_processed = sum(v for k, v in directive_status.items() if k not in ['pending', 'active'])
             if total_processed > 10 and failed_directives / total_processed > 0.2: feedback += "High directive failure rate observed. " ; critique['status'] = 'warning'
             critique['feedback'] = feedback
         except Exception as e: self.logger.error(f"Error during self-critique: {e}", exc_info=True); critique['status'] = 'error'; critique['feedback'] = f"Critique failed: {e}"
         return critique

     async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
         """Constructs context-rich prompts for LLM calls, including enriched data context."""
         self.logger.debug(f"Generating dynamic prompt for ThinkTool task: {task_context.get('task')}")
         prompt_gen_thought = f"Structured Thinking: Generate Dynamic Prompt for task '{task_context.get('task')}'. Plan: Combine meta-prompt, task context, KB context (if needed), instructions."
         await self._internal_think(prompt_gen_thought)
         prompt_parts = [self.meta_prompt]
         prompt_parts.append("\n--- Current Task Context ---")
         for key, value in task_context.items():
             value_str = ""; max_len = 1500
             # Increase length limit for potentially large context items
             if key in ['knowledge_base_context', 'Feedback', 'Report', 'Content', 'Current Prompt', 'feedback_data', 'clay_data', 'conversation_summary', 'content_snippet', 'report_summary', 'enriched_data_available']: max_len = 4000
             if isinstance(value, str): value_str = value[:max_len] + ("..." if len(value) > max_len else "")
             elif isinstance(value, (int, float, bool)): value_str = str(value)
             elif isinstance(value, (dict, list)):
                 try: value_str = json.dumps(value, default=str, indent=2); value_str = value_str[:max_len] + ("..." if len(value_str) > max_len else "")
                 except TypeError: value_str = str(value)[:max_len] + "..."
             else: value_str = str(value)[:max_len] + "..."
             # Use a more descriptive key for enriched data
             prompt_key = "Enriched Prospect Data" if key == 'enriched_data_available' else key.replace('_', ' ').title()
             prompt_parts.append(f"**{prompt_key}**: {value_str}")

         prompt_parts.append("\n--- Instructions ---")
         task_type = task_context.get('task')
         prompt_parts.append(f"Based on the above context, perform the task: **{task_type or 'N/A'}**")

         # Add specific instructions based on task_type
         if task_type == 'Analyze Learning Material':
              prompt_parts.append("Identify key concepts, actionable strategies (sales, marketing, efficiency), relevant mindsets, or code techniques. Determine applicable agents. Categorize insight type. Assign relevance score (0.0-1.0).")
         elif task_type == 'Generate Educational Content':
              prompt_parts.append("Generate concise, user-friendly explanation. Assume intelligent user, non-expert. Avoid/explain jargon. Focus on 'why' & relevance. Start directly with explanation.")
         elif task_type == 'Analyze Agent Feedback':
              prompt_parts.append("Analyze consolidated agent feedback. Identify critical issues, successes, trends. Propose actions: StrategicDirectives (JSON), prompt critiques (list 'Agent/Key'), insights to log (JSON for log_knowledge_fragment).")
         elif task_type == 'Synthesize Insights & Generate Strategy':
              prompt_parts.append("Synthesize insights from recent data. Identify/validate patterns, assess goal progress ($10k+/day), generate directives & opportunities. Integrate scoring/optimization logic. Consider generated Clay directives for enrichment.")
         elif task_type == 'Analyze Technology Scouting Report':
              prompt_parts.append(f"Analyze tech scouting report. Identify novel, high-impact tools/techniques relevant to profit/efficiency/grey areas. Assess benefits, risks, integration effort. Focus: {task_context.get('analysis_focus')}")
         elif task_type == 'Critique and Rewrite Prompt':
              prompt_parts.append(f"Critique prompt for {task_context.get('agent_name')}/{task_context.get('prompt_key')}. Improve based on feedback: {task_context.get('feedback_context')}. Aim for clarity, robustness, goal alignment. Rewrite the *entire* prompt.")
         elif task_type == 'Reflect on Proposed Action/Plan':
              prompt_parts.append("Analyze agent's context, task, KB info. Assess context sufficiency, compliance concerns, risk level (Low/Med/High/Critical), goal alignment. Suggest next step. Estimate confidence (0.0-1.0).")
         elif task_type == 'Validate Agent Output':
              prompt_parts.append(f"Validate agent output against criteria: {task_context.get('validation_criteria')}. Consider patterns: {task_context.get('learned_patterns_context')}. Checks: Criteria Adherence? Content Valid? Pattern Consistent? Usable? Compliance/Risk OK?")
         elif task_type == 'Calculate Dynamic Price':
              prompt_parts.append(f"Analyze conversation summary for pricing signals (urgency, budget, need). Client Score: {task_context.get('client_score', 0.1):.2f}. Base Price: ${task_context.get('base_price', 7000.0)}. Suggest adjustment factor (0.9-1.1) and reason.")
         elif task_type == 'Analyze Conversation for Needs': # New task type for UGC discovery
              prompt_parts.append("Analyze the conversation history. Identify the client's stated and implied needs, goals, pain points, and budget constraints related to UGC video services. Summarize these findings clearly.")
         elif task_type == 'Generate Tailored UGC Offer': # New task type for UGC discovery
              prompt_parts.append(f"Based on the discovered needs summary: {task_context.get('needs_summary', 'N/A')}, construct a specific UGC package offer. Define the number of videos, style, focus, and reiterate the firm price of $7000. Frame it as the ideal solution to their stated needs.")
         else:
             prompt_parts.append("Provide a clear, concise, and actionable response based on the task description.")

         if task_context.get('desired_output_format'): prompt_parts.append(f"**Output Format:** {task_context['desired_output_format']}")
         if task_context.get('is_json_output', False) or "JSON" in task_context.get('desired_output_format', ''):
             prompt_parts.append("\nRespond ONLY with valid JSON matching the specified format. Do not include explanations or markdown formatting outside the JSON structure.")
             prompt_parts.append("```json")

         final_prompt = "\n".join(prompt_parts)
         self.logger.debug(f"Generated dynamic prompt for ThinkTool (length: {len(final_prompt)} chars)")
         return final_prompt

     async def collect_insights(self) -> Dict[str, Any]:
         # (Keep existing implementation)
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

     # --- Helper for reporting errors ---
     async def _report_error(self, error_message: str, task_id: Optional[str] = None):
         # (Keep existing implementation)
         if self.orchestrator and hasattr(self.orchestrator, 'report_error'):
             try: await self.orchestrator.report_error(self.AGENT_NAME, f"TaskID [{task_id or 'N/A'}]: {error_message}")
             except Exception as report_err: self.logger.error(f"Failed to report error to orchestrator: {report_err}")
         else: self.logger.warning("Orchestrator unavailable or lacks report_error method.")

     # --- Helper for parsing LLM JSON ---
     def _parse_llm_json(self, json_string: str, expect_type: Type = dict) -> Union[Dict, List, None]:
         # (Keep existing implementation)
         if not json_string: return None
         try:
             match = None; start_char, end_char = '{', '}'
             if expect_type == list: start_char, end_char = '[', ']'
             # Improved regex to handle optional ```json prefix and potential whitespace
             match = re.search(rf'(?:```(?:json)?\s*)?(\{start_char}.*\{end_char})\s*(?:```)?', json_string, re.DOTALL)

             parsed_json = None
             if match:
                 potential_json = match.group(1)
                 try: parsed_json = json.loads(potential_json)
                 except json.JSONDecodeError as e:
                     self.logger.warning(f"Initial JSON parsing failed ({e}), attempting to clean and retry: {potential_json[:100]}...")
                     cleaned_json = re.sub(r',\s*([\}\]])', r'\1', potential_json) # Basic cleaning: remove trailing commas
                     cleaned_json = re.sub(r'^\s*|\s*$', '', cleaned_json) # Trim whitespace
                     try: parsed_json = json.loads(cleaned_json)
                     except json.JSONDecodeError as e2: self.logger.error(f"JSON cleaning failed ({e2}), unable to parse: {potential_json[:200]}..."); return None
             # Fallback if no markdown block found but string looks like JSON
             elif json_string.strip().startswith(start_char) and json_string.strip().endswith(end_char):
                  try: parsed_json = json.loads(json_string)
                  except json.JSONDecodeError as e: self.logger.error(f"Direct JSON parsing failed ({e}): {json_string[:200]}..."); return None
             else: self.logger.warning(f"Could not find expected JSON structure ({expect_type}) in LLM output: {json_string[:200]}..."); return None

             if isinstance(parsed_json, expect_type): return parsed_json
             else: self.logger.error(f"Parsed JSON type mismatch. Expected {expect_type}, got {type(parsed_json)}"); return None
         except json.JSONDecodeError as e: self.logger.error(f"Failed to decode LLM JSON response: {e}. Response snippet: {json_string[:500]}..."); return None
         except Exception as e: self.logger.error(f"Unexpected error during JSON parsing: {e}", exc_info=True); return None

     # --- Specific Logic Moved/Integrated ---
     async def _calculate_dynamic_price(self, client_id: int, conversation_summary: Optional[List] = None, base_price: float = 7000.0) -> float: # Updated base price
         # (Keep existing implementation, but uses new base_price)
         client_score = 0.1
         try:
             async with self.session_maker() as session:
                  score_res = await session.execute(select(Client.engagement_score).where(Client.id == client_id))
                  client_score = score_res.scalar_one_or_none() or 0.1

             # Keep adjustment simple for now, focus is on the base price
             price_adjustment_factor = 1.0 # Start with no adjustment

             # Optional: LLM analysis for adjustment (keep if desired)
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
                             # Apply LLM factor cautiously, maybe average or cap it
                             price_adjustment_factor = max(0.9, min(1.1, llm_factor)) # Cap adjustment
                             self.logger.info(f"LLM suggested price factor: {llm_factor:.2f}. Using capped factor: {price_adjustment_factor:.2f}. Reason: {analysis_result.get('reason')}")
                 except Exception as llm_err: self.logger.warning(f"LLM pricing analysis failed: {llm_err}")

             calculated_price = base_price * price_adjustment_factor
             # Ensure final price isn't below base (unless factor is < 1)
             final_price = max(calculated_price, base_price * 0.9) # Allow slight downward adjustment if LLM suggests strongly

             self.logger.info(f"Calculated dynamic price for client {client_id}: ${final_price:.2f} (Base: {base_price}, Score: {client_score:.2f}, Factor: {price_adjustment_factor:.2f})")
             return round(final_price, 2)
         except Exception as e: self.logger.error(f"Error calculating dynamic price for client {client_id}: {e}", exc_info=True); return round(base_price, 2)

     async def _process_clay_result(self, clay_api_result: Dict[str, Any], source_directive_id: Optional[int] = None, source_reference: Optional[str] = None, client_id: Optional[int] = None):
         """Processes enriched data from Clay.com and triggers next actions."""
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
             # --- MODIFIED: Extract more fields ---
             processed_info['verified_email'] = clay_data.get('email') or clay_data.get('person', {}).get('email') or clay_data.get('verified_email')
             processed_info['job_title'] = clay_data.get('job_title') or clay_data.get('person', {}).get('title') or clay_data.get('title')
             processed_info['company_name'] = clay_data.get('company_name') or clay_data.get('company', {}).get('name')
             processed_info['linkedin_url'] = clay_data.get('linkedin_url') or clay_data.get('person', {}).get('linkedin_url') or source_reference
             processed_info['company_domain'] = clay_data.get('company', {}).get('domain')
             processed_info['full_name'] = clay_data.get('full_name') or clay_data.get('person', {}).get('full_name')
             processed_info['company_size'] = clay_data.get('company', {}).get('company_size') # Example: company size
             processed_info['industry'] = clay_data.get('company', {}).get('industry') # Example: industry
             processed_info['location'] = clay_data.get('location') or clay_data.get('person', {}).get('location') # Example: location
             # Add more fields as available/needed from Clay's responses

             processed_info = {k: v for k, v in processed_info.items() if v is not None and v != ''} # Clean empty values

             if processed_info: # Check if we got *any* useful info
                 try:
                     async with self.session_maker() as session:
                         async with session.begin():
                             target_client_id = client_id; target_client = None
                             if target_client_id: target_client = await session.get(Client, target_client_id)
                             # Try lookup if client_id wasn't provided or client not found
                             if not target_client:
                                 lookup_stmt = select(Client)
                                 conditions = []
                                 if processed_info.get('verified_email'): conditions.append(Client.email == processed_info['verified_email'])
                                 if source_reference and 'linkedin.com' in source_reference: conditions.append(Client.source_reference == source_reference)
                                 if conditions:
                                     lookup_stmt = lookup_stmt.where(or_(*conditions)).limit(1)
                                     target_client = (await session.execute(lookup_stmt)).scalar_one_or_none()
                                     if target_client: target_client_id = target_client.id
                                 else: # Cannot lookup without email or linkedin
                                     self.logger.warning(f"Cannot reliably look up client for Clay result without email/linkedin ref.")

                             # Update or Create Client Record
                             if target_client: # Update existing client
                                 update_values = {'last_interaction': datetime.now(timezone.utc)}
                                 if not target_client.email and processed_info.get('verified_email'): update_values['email'] = processed_info['verified_email']
                                 if not target_client.company and processed_info.get('company_name'): update_values['company'] = processed_info['company_name']
                                 if not target_client.job_title and processed_info.get('job_title'): update_values['job_title'] = processed_info['job_title']
                                 # Add other fields to update
                                 await session.execute(update(Client).where(Client.id == target_client_id).values(**update_values))
                                 self.logger.info(f"Updated Client {target_client_id} with enriched data: {list(update_values.keys())}")
                             else: # Create new client if possible (requires name and email at minimum)
                                 if processed_info.get('verified_email') and processed_info.get('full_name'):
                                     new_client = Client(
                                         name=processed_info['full_name'],
                                         email=processed_info['verified_email'],
                                         source_reference=source_reference,
                                         company=processed_info.get('company_name'),
                                         job_title=processed_info.get('job_title'),
                                         # Add other fields
                                         source="ClayEnrichment",
                                         opt_in=True, # Assume opt-in for newly found leads? Needs review.
                                         is_deliverable=True
                                     )
                                     session.add(new_client)
                                     await session.flush() # Get the new ID
                                     target_client_id = new_client.id
                                     self.logger.info(f"Created new Client {target_client_id} from Clay enrichment.")
                                 else:
                                     self.logger.warning("Cannot create new client from Clay data - missing name or email.")
                                     target_client_id = None # Ensure no outreach if client creation failed

                             # Log the full enrichment result
                             fragment = await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="clay_enrichment_result", content=processed_info, tags=["clay", "enrichment", "lead_data"], relevance_score=0.9, related_client_id=target_client_id, source_reference=source_reference or f"ClayAPI_Directive_{source_directive_id}")
                             self.logger.info(f"Logged Clay enrichment result ({fragment.id if fragment else 'existing'}) for {processed_info.get('linkedin_url')} to KB.")

                             # Trigger outreach ONLY if we have an email and a client ID
                             if processed_info.get('verified_email') and target_client_id:
                                 outreach_directive = StrategicDirective(
                                     source=self.AGENT_NAME, timestamp=datetime.now(timezone.utc),
                                     target_agent="EmailAgent", directive_type="initiate_outreach",
                                     content=json.dumps({ # Pass the full enriched data
                                         "target_identifier": processed_info['verified_email'],
                                         "client_id": target_client_id,
                                         "context": f"Enriched lead via Clay. Job: {processed_info.get('job_title', 'N/A')}, Company: {processed_info.get('company_name', 'N/A')}.",
                                         "goal": "Book sales call for UGC service",
                                         "enriched_data": processed_info # Include all processed info
                                     }),
                                     priority=4, status='pending'
                                 )
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
          # (Keep existing implementation)
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
         # (Keep existing implementation)
         if not self.session_maker: return []
         try:
             async with self.session_maker() as session:
                 stmt = select(StrategicDirective).where(StrategicDirective.status.in_(['pending', 'active'])).order_by(StrategicDirective.priority, desc(StrategicDirective.timestamp)).limit(limit)
                 if target_agent: stmt = stmt.where(StrategicDirective.target_agent == target_agent)
                 directives = list((await session.execute(stmt)).scalars().all())
                 return directives
         except SQLAlchemyError as e: self.logger.error(f"DB Error getting active directives: {e}", exc_info=True); return []
         except Exception as e: self.logger.error(f"Unexpected error getting active directives: {e}", exc_info=True); return []


 # --- End of agents/think_tool.py ---