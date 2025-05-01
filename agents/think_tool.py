# Filename: agents/think_tool.py
# Description: Genius Agentic Core - Strategy, Learning, KB Management, Analysis.
# Version: 4.1 (Genius Agentic - Postgres KB, Clay.com Logic, Learning Loops)

import asyncio
import logging
import json
import os
import hashlib
import time
import glob # For loading learning files
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union, Tuple, AsyncGenerator, Type

# --- Core Framework Imports ---
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select, delete, func, update, text, case, desc, or_, asc # Ensure all needed functions are imported
from sqlalchemy.exc import SQLAlchemyError

# --- Project Imports ---
# Use production-ready base class if available, otherwise fallback
try:
    from .base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
except ImportError:
    logger.warning("Production base agent not found, using GeniusAgentBase. Ensure base_agent_prod.py is used.")
    from .base_agent import GeniusAgentBase

# Import necessary models
from models import (
    KnowledgeFragment, LearnedPattern, StrategicDirective, PromptTemplate,
    EmailLog, CallLog, Invoice, Client, OSINTData, AccountCredentials # Added models needed for analysis
)
# Import settings AFTER it's defined and validated
from config.settings import settings

# Configure logger
logger = logging.getLogger(__name__)
# Configure dedicated operational logger
op_logger = logging.getLogger('OperationalLog') # Assuming setup elsewhere

# --- Meta Prompt ---
THINK_TOOL_META_PROMPT = """
You are ThinkTool, the central intelligence of the Synapse AI Sales System. Your mandate is **Maximize Profit & Growth ($10k+/day -> $100M/8mo)**.
**Core Functions:**
1.  **Knowledge Synthesis:** Continuously analyze ALL data sources (KB Fragments, Performance Logs, Learning Materials, Market Scans) to identify patterns, opportunities, and inefficiencies. Manage the Postgres Knowledge Base directly.
2.  **Strategic Planning:** Generate high-level goals and break them down into actionable `StrategicDirectives` for the Orchestrator and other agents. Prioritize based on ROI and risk.
3.  **Agent Guidance:** Generate/Refine prompts for other agents. Provide context and strategic direction for tasks (e.g., email personalization angles, call script focus, Clay.com search parameters).
4.  **Performance Monitoring & Adaptation:** Analyze agent performance metrics (email opens, call success, profit). Adapt strategies and scoring models based on results. Trigger self-critiques.
5.  **Opportunity Identification:** Proactively identify new market niches, competitor weaknesses, and high-ROI ventures beyond current operations (e.g., beyond UGC). Leverage Clay.com and OSINT findings.
6.  **Grey Area Strategy:** Devise and manage strategies involving calculated risks (e.g., multi-accounting, trial rotation), informed by LegalAgent's risk assessments.
7.  **Learning Integration:** Ingest and synthesize external knowledge (`learning for AI/` folder).
**Principles:** Extreme Agency, Hyper-Efficiency, Relentless Learning, AI-Native Advantage, Strategic Exploitation, Adaptability, Compliance Awareness.
"""

class ThinkTool(GeniusAgentBase):
    """
    ThinkTool (Genius Level): The central cognitive engine. Manages strategy,
    learning, knowledge base (Postgres), and directs other agents towards profit maximization.
    Version: 4.1
    """
    AGENT_NAME = "ThinkTool"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], config: Any, orchestrator: object):
        """
        Initializes the ThinkTool agent.
        ### Phase 2 Plan Ref: 4.1 (Implement __init__)
        """
        # ThinkTool manages KB directly, so kb_interface is None for base init
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, config=config, session_maker=session_maker, kb_interface=None)

        self.meta_prompt = THINK_TOOL_META_PROMPT

        # Timestamps for periodic tasks in the run loop
        self.last_synthesis_run: Optional[datetime] = None
        self.last_radar_run: Optional[datetime] = None
        self.last_purge_run: Optional[datetime] = None

        self.logger.info("ThinkTool v4.1 initialized.")

        # Load initial learning materials (run async in background)
        # ### Phase 2 Plan Ref: 4.3 (Implement Learning Material Synthesis) - Startup part
        asyncio.create_task(self._load_and_synthesize_learning_materials())

    async def log_operation(self, level: str, message: str):
        """Helper to log to the operational log file with agent context."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err:
            print(f"OPERATIONAL LOG FAILED ({self.agent_name}): {level} - {message} | Error: {log_err}")
            logger.error(f"Failed to write to operational log from {self.agent_name}: {log_err}")

    # --- Knowledge Loading & Synthesis ---

    async def _load_and_synthesize_learning_materials(self):
        """Loads and processes text files from the learning directory, storing insights in KB."""
        # ### Phase 2 Plan Ref: 4.3 (Implement Learning Material Synthesis)
        learning_dir = 'learning for AI/'
        self.logger.info(f"ThinkTool: Loading learning materials from '{learning_dir}'...")
        processed_files = 0
        learning_files = []
        try:
            # Use relative path from main.py location
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Go up two levels from agents/
            full_learning_dir = os.path.join(base_dir, learning_dir)

            if not os.path.isdir(full_learning_dir):
                self.logger.warning(f"Learning directory '{full_learning_dir}' not found. Skipping loading.")
                return

            # Use glob to find all .txt files recursively
            file_pattern = os.path.join(full_learning_dir, '**', '*.txt')
            learning_files = glob.glob(file_pattern, recursive=True)

            if not learning_files:
                self.logger.info(f"No .txt files found in '{full_learning_dir}'.")
                return

            self.logger.info(f"Found {len(learning_files)} potential learning files.")

            for file_path in learning_files:
                try:
                    self.logger.debug(f"Processing learning file: {file_path}")
                    file_content = None
                    # Use orchestrator's file reading tool (assumes orchestrator is ready)
                    if self.orchestrator and hasattr(self.orchestrator, 'use_tool'):
                         # Need absolute path for the tool typically
                         abs_file_path = os.path.abspath(file_path)
                         file_content_result = await self.orchestrator.use_tool('read_file', {'path': abs_file_path})
                         if file_content_result and file_content_result.get('status') == 'success':
                             file_content = file_content_result.get('content')
                         else:
                             self.logger.warning(f"Could not read file {abs_file_path} via orchestrator tool: {file_content_result.get('message')}")
                             continue # Skip if read fails
                    else:
                         self.logger.error("Orchestrator tool access unavailable for reading learning files.")
                         # Attempt direct read as fallback (might fail in Docker if paths differ)
                         try:
                             with open(file_path, 'r', encoding='utf-8') as f:
                                 file_content = f.read()
                             self.logger.warning("Read learning file directly as orchestrator tool failed.")
                         except Exception as direct_read_err:
                              self.logger.error(f"Direct read also failed for {file_path}: {direct_read_err}")
                              continue

                    if not file_content or not file_content.strip():
                        self.logger.warning(f"File is empty or could not be read: {file_path}")
                        continue

                    # --- Analysis/Synthesis ---
                    self.logger.info(f"Analyzing content from: {os.path.basename(file_path)} using LLM...")
                    analysis_thought = f"Structured Thinking: Analyze Learning Material '{os.path.basename(file_path)}'. Plan: Formulate analysis prompt -> Call LLM -> Parse JSON -> Log to KB."
                    await self._internal_think(analysis_thought)

                    analysis_prompt = f"""
                    {self.meta_prompt[:500]}...
                    **Task:** Analyze text from '{os.path.basename(file_path)}'. Identify key concepts, actionable strategies, relevant mindsets, or code techniques. Determine applicable agents (e.g., ThinkTool, EmailAgent, All). Categorize insight type (e.g., 'sales_tactic', 'mindset', 'prompt_engineering', 'market_insight', 'efficiency_hack'). Assign relevance score (0.0-1.0, higher if directly applicable to sales/profit).
                    **Content (Limit 4000 chars):** ```\n{file_content[:4000]}\n```
                    **Output Format:** Respond ONLY with valid JSON: {{"source_file": str, "summary": str, "key_concepts": [str], "actionable_strategies": [str], "applicable_agents": [str], "insight_type": str, "relevance_score": float}}
                    """
                    synthesized_insights_json = await self._call_llm_with_retry(
                        analysis_prompt, temperature=0.5, max_tokens=1024, is_json_output=True
                    )

                    if synthesized_insights_json:
                        try:
                            # Attempt to find JSON within potential markdown code blocks
                            json_match = re.search(r'```json\s*(\{.*?\})\s*```', synthesized_insights_json, re.DOTALL)
                            if json_match:
                                insights_data = json.loads(json_match.group(1))
                            else:
                                # Fallback: try parsing the whole string if no code block found
                                insights_data = json.loads(synthesized_insights_json)

                            if not isinstance(insights_data, dict) or not all(k in insights_data for k in ['summary', 'key_concepts', 'applicable_agents', 'insight_type', 'relevance_score']):
                                raise ValueError("LLM response missing required keys.")
                            insights_data['source_file'] = os.path.basename(file_path) # Ensure correct source

                            # Store as Knowledge Fragment
                            await self.log_knowledge_fragment(
                                agent_source="LearningMaterialLoader",
                                data_type=insights_data.get('insight_type', 'learning_material_summary'),
                                content=insights_data, # Store the parsed dict as JSON string
                                relevance_score=insights_data.get('relevance_score', 0.6),
                                tags=["learning_material", insights_data.get('insight_type', 'general')] + [f"agent:{a.lower()}" for a in insights_data.get('applicable_agents', [])],
                                source_reference=file_path # Store original file path
                            )
                            processed_files += 1
                        except (json.JSONDecodeError, ValueError) as json_error:
                            self.logger.error(f"Error parsing/validating LLM response for {file_path}: {json_error}. Response: {synthesized_insights_json[:500]}...")
                        except Exception as store_err:
                             self.logger.error(f"Error storing knowledge fragment for {file_path}: {store_err}", exc_info=True)
                    else:
                         self.logger.error(f"LLM analysis returned no content for {file_path}.")

                except Exception as file_error:
                    self.logger.error(f"General error processing learning file {file_path}: {file_error}", exc_info=True)

            self.logger.info(f"Finished processing learning materials. Processed {processed_files}/{len(learning_files)} files.")

        except Exception as e:
            self.logger.error(f"Critical error during loading/synthesizing learning materials: {e}", exc_info=True)

    # --- Standardized LLM Interaction ---
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=4, max=30), retry=retry_if_exception_type(Exception))
    async def _call_llm_with_retry(self, prompt: str, model_preference: Optional[List[str]] = None, temperature: float = 0.5, max_tokens: int = 1024, is_json_output: bool = False) -> Optional[str]:
        """Centralized method for calling LLMs via the Orchestrator."""
        if not self.orchestrator or not hasattr(self.orchestrator, 'call_llm'):
            self.logger.error("Orchestrator or its call_llm method is unavailable.")
            return None
        try:
            # Delegate the call to the orchestrator
            response_content = await self.orchestrator.call_llm(
                agent_name=self.AGENT_NAME, prompt=prompt, temperature=temperature,
                max_tokens=max_tokens, is_json_output=is_json_output, model_preference=model_preference
            )
            # Validate response
            if response_content is not None and not isinstance(response_content, str):
                 self.logger.error(f"Orchestrator.call_llm returned non-string type: {type(response_content)}")
                 return None
            if isinstance(response_content, str) and not response_content.strip():
                 self.logger.warning("Orchestrator.call_llm returned empty string.")
                 return None
            return response_content
        except Exception as e:
            self.logger.error(f"Error occurred calling LLM via orchestrator: {e}", exc_info=True)
            raise # Re-raise for tenacity

    # --- User Education ---
    async def generate_educational_content(self, topic: str, context: Optional[str] = None) -> Optional[str]:
        """Generates a concise, user-friendly explanation for the User Education Mechanism."""
        self.logger.info(f"ThinkTool: Generating educational content for topic: {topic}")
        thinking_process = f"Structured Thinking: Generate Educational Content for '{topic}'. Context: '{context or 'General'}'. Plan: Formulate prompt, call LLM, return cleaned response."
        await self._internal_think(thinking_process)
        prompt = f"""
        {self.meta_prompt[:500]}...
        **Task:** Generate concise, user-friendly explanation for topic: **{topic}**. Assume intelligent user, non-expert. Avoid/explain jargon. Focus on 'why' & relevance to agency goals. Context: {context or 'General understanding'}.
        **Output:** ONLY the explanation text, suitable for direct display. Start directly with explanation.
        """
        explanation = await self._call_llm_with_retry(
            prompt, temperature=0.6, max_tokens=500, is_json_output=False
        )
        if explanation: self.logger.info(f"Successfully generated educational content for topic: {topic}")
        else: self.logger.error(f"Failed to generate educational content for topic: {topic} (LLM error).")
        return explanation

    # --- Knowledge Base Interface Implementation (Direct Postgres) ---
    # ### Phase 2 Plan Ref: 4.1 (Implement Direct Postgres KB)

    async def log_knowledge_fragment(self, agent_source: str, data_type: str, content: Union[str, dict], relevance_score: float = 0.5, tags: Optional[List[str]] = None, related_client_id: Optional[int] = None, source_reference: Optional[str] = None) -> Optional[KnowledgeFragment]:
        """Persists a KnowledgeFragment to the database, ensuring content hash for deduplication."""
        if not self.session_maker: self.logger.error("DB session_maker not available in ThinkTool."); return None

        if isinstance(content, dict): content_str = json.dumps(content, sort_keys=True)
        elif isinstance(content, str): content_str = content
        else: self.logger.error(f"Invalid content type for KnowledgeFragment: {type(content)}"); return None

        # Ensure tags are stored as a valid JSON array string or None
        tags_list = sorted(list(set(tags))) if tags else []
        tags_str = json.dumps(tags_list) if tags_list else None
        content_hash = hashlib.sha256(content_str.encode('utf-8')).hexdigest()
        now_ts = datetime.now(timezone.utc)

        try:
            async with self.session_maker() as session:
                async with session.begin(): # Use transaction
                    # Check for existing hash
                    stmt_check = select(KnowledgeFragment.id).where(KnowledgeFragment.item_hash == content_hash).limit(1)
                    result_check = await session.execute(stmt_check)
                    existing_id = result_check.scalar_one_or_none()

                    if existing_id:
                        self.logger.debug(f"KF hash {content_hash[:8]}... exists (ID: {existing_id}). Updating last_accessed_ts.")
                        stmt_update = update(KnowledgeFragment).where(KnowledgeFragment.id == existing_id).values(last_accessed_ts=now_ts)
                        await session.execute(stmt_update)
                        # No need to return the fragment as it wasn't newly logged
                        return None
                    else:
                        fragment = KnowledgeFragment(
                            agent_source=agent_source, timestamp=now_ts, last_accessed_ts=now_ts,
                            data_type=data_type, content=content_str, item_hash=content_hash,
                            relevance_score=relevance_score, tags=tags_str,
                            related_client_id=related_client_id, source_reference=source_reference
                        )
                        session.add(fragment)
                        # Commit happens automatically at end of 'async with session.begin()'
                # Refresh outside the transaction to get the ID
                await session.refresh(fragment)
                self.logger.info(f"Logged KnowledgeFragment: ID={fragment.id}, Hash={content_hash[:8]}..., Type={data_type}, Source={agent_source}")
                return fragment
        except SQLAlchemyError as e:
            self.logger.error(f"DB Error logging KnowledgeFragment: {e}", exc_info=True)
            await self._report_error(f"DB Error logging KnowledgeFragment: {e}")
            return None
        except Exception as e:
             self.logger.error(f"Unexpected error logging KnowledgeFragment: {e}", exc_info=True)
             return None

    async def query_knowledge_base(self, data_types: Optional[List[str]] = None, tags: Optional[List[str]] = None, min_relevance: float = 0.0, time_window: Optional[timedelta] = None, limit: int = 100, related_client_id: Optional[int] = None, content_query: Optional[str] = None) -> List[KnowledgeFragment]:
        """Queries the knowledge_fragments table with enhanced filtering and updates access time."""
        if not self.session_maker: self.logger.error("DB session_maker not available in ThinkTool."); return []
        fragments = []
        fragment_ids = []
        try:
            async with self.session_maker() as session:
                stmt = select(KnowledgeFragment)
                if data_types: stmt = stmt.where(KnowledgeFragment.data_type.in_(data_types))
                if min_relevance > 0.0: stmt = stmt.where(KnowledgeFragment.relevance_score >= min_relevance)
                if related_client_id is not None: stmt = stmt.where(KnowledgeFragment.related_client_id == related_client_id)
                if time_window: stmt = stmt.where(KnowledgeFragment.timestamp >= (datetime.now(timezone.utc) - time_window))
                if content_query: stmt = stmt.where(KnowledgeFragment.content.ilike(f'%{content_query}%')) # Case-insensitive LIKE
                if tags:
                     # Assumes tags stored as JSON array string '["tag1", "tag2"]'
                     # Use specific DB function if available (e.g., JSONB containment)
                     # Generic LIKE approach:
                     tag_conditions = [KnowledgeFragment.tags.like(f'%"{tag}"%') for tag in tags]
                     stmt = stmt.where(or_(*tag_conditions))

                # Select only IDs first for efficiency and locking potential
                stmt_ids = stmt.with_only_columns(KnowledgeFragment.id).order_by(desc(KnowledgeFragment.relevance_score), desc(KnowledgeFragment.timestamp)).limit(limit)
                result_ids = await session.execute(stmt_ids)
                fragment_ids = result_ids.scalars().all()

                if not fragment_ids: return []

                # Fetch full fragments based on IDs
                stmt_final = select(KnowledgeFragment).where(KnowledgeFragment.id.in_(fragment_ids)).order_by(desc(KnowledgeFragment.relevance_score), desc(KnowledgeFragment.timestamp))
                result_final = await session.execute(stmt_final)
                fragments = list(result_final.scalars().all())

                # Update last_accessed_ts in background (fire and forget)
                if fragment_ids:
                    async def update_access_time():
                        try:
                            async with self.session_maker() as update_session:
                                async with update_session.begin():
                                    update_stmt = update(KnowledgeFragment).where(KnowledgeFragment.id.in_(fragment_ids)).values(last_accessed_ts=datetime.now(timezone.utc))
                                    await update_session.execute(update_stmt)
                                # Commit happens automatically
                                self.logger.debug(f"Updated last_accessed_ts for {len(fragment_ids)} fragments.")
                        except Exception as update_err: self.logger.error(f"Failed to update last_accessed_ts: {update_err}")
                    asyncio.create_task(update_access_time())

                self.logger.debug(f"KnowledgeBase query returned {len(fragments)} fragments.")
        except SQLAlchemyError as e:
            self.logger.error(f"DB Error querying KnowledgeBase: {e}", exc_info=True)
            await self._report_error(f"DB Error querying KnowledgeBase: {e}")
        except Exception as e: self.logger.error(f"Unexpected error querying KnowledgeBase: {e}", exc_info=True)
        return fragments

    async def add_email_composition(self, email_log_id: int, composition_details: Dict[str, Any]):
        """Logs the link between an email and the KB fragments used."""
        # This might be better handled by EmailAgent logging directly or ThinkTool processing EmailLog
        self.logger.warning("add_email_composition called on ThinkTool - consider if EmailAgent should log this.")
        # Placeholder implementation if needed here
        pass

    async def get_active_directives(self, target_agent: str = 'All') -> List[StrategicDirective]:
        """Fetches active or pending StrategicDirectives from DB."""
        if not self.session_maker: self.logger.error("DB session_maker not available."); return []
        directives = []
        try:
            async with self.session_maker() as session:
                stmt = select(StrategicDirective).where(StrategicDirective.status.in_(['pending', 'active']))
                if target_agent != 'All': stmt = stmt.where(or_(StrategicDirective.target_agent == target_agent, StrategicDirective.target_agent == 'All'))
                stmt = stmt.order_by(asc(StrategicDirective.priority), asc(StrategicDirective.timestamp))
                result = await session.execute(stmt)
                directives = list(result.scalars().all())
                self.logger.debug(f"Fetched {len(directives)} active/pending directives for target '{target_agent}'.")
        except SQLAlchemyError as e:
            self.logger.error(f"DB Error getting active directives: {e}", exc_info=True)
            await self._report_error(f"DB Error getting directives: {e}")
        except Exception as e: self.logger.error(f"Unexpected error getting active directives: {e}", exc_info=True)
        return directives

    async def update_directive_status(self, directive_id: int, status: str, result_summary: Optional[str] = None) -> bool:
        """Updates the status and optionally result of a StrategicDirective in DB."""
        if not self.session_maker: self.logger.error("DB session_maker not available."); return False
        if not isinstance(directive_id, int): self.logger.error(f"Invalid directive_id type: {type(directive_id)}"); return False
        valid_statuses = ['pending', 'active', 'completed', 'failed', 'expired', 'cancelled']
        if status not in valid_statuses: self.logger.error(f"Invalid status '{status}' for directive update."); return False

        try:
            async with self.session_maker() as session:
                async with session.begin(): # Transaction
                    values_to_update = {"status": status}
                    if result_summary is not None: values_to_update["result_summary"] = result_summary
                    # Add completion timestamp if relevant field exists
                    # if status in ['completed', 'failed', 'expired', 'cancelled']: values_to_update["completed_at"] = datetime.now(timezone.utc)

                    stmt = update(StrategicDirective).where(StrategicDirective.id == directive_id).values(**values_to_update)
                    result = await session.execute(stmt)
                    # Commit happens automatically

                if result.rowcount > 0: self.logger.info(f"Updated StrategicDirective ID={directive_id} status to '{status}'."); return True
                else: self.logger.warning(f"Failed to update StrategicDirective ID={directive_id}: Not found."); return False
        except SQLAlchemyError as e:
            self.logger.error(f"DB Error updating directive status for ID={directive_id}: {e}", exc_info=True)
            await self._report_error(f"DB Error updating directive {directive_id}: {e}")
            return False
        except Exception as e: self.logger.error(f"Unexpected error updating directive status: {e}", exc_info=True); return False

    async def log_learned_pattern(self, pattern_description: str, supporting_fragment_ids: List[int], confidence_score: float, implications: str, tags: Optional[List[str]] = None) -> Optional[LearnedPattern]:
        """Creates and persists a LearnedPattern record in DB."""
        if not self.session_maker: self.logger.error("DB session_maker not available."); return None
        fragment_ids_str = json.dumps(sorted(list(set(supporting_fragment_ids))))
        tags_list = sorted(list(set(tags))) if tags else []
        tags_str = json.dumps(tags_list) if tags_list else None

        pattern = LearnedPattern(
            timestamp=datetime.now(timezone.utc), pattern_description=pattern_description,
            supporting_fragment_ids=fragment_ids_str, confidence_score=confidence_score,
            implications=implications, tags=tags_str, status='active'
        )
        try:
            async with self.session_maker() as session:
                async with session.begin(): # Transaction
                    session.add(pattern)
                await session.refresh(pattern) # Refresh after commit
                self.logger.info(f"Logged LearnedPattern: ID={pattern.id}, Confidence={confidence_score:.2f}")
                return pattern
        except SQLAlchemyError as e:
            self.logger.error(f"DB Error logging LearnedPattern: {e}", exc_info=True)
            await self._report_error(f"DB Error logging LearnedPattern: {e}")
            return None
        except Exception as e: self.logger.error(f"Unexpected error logging LearnedPattern: {e}", exc_info=True); return None

    async def get_latest_patterns(self, tags: Optional[List[str]] = None, min_confidence: float = 0.7, limit: int = 10) -> List[LearnedPattern]:
        """Retrieves recent, high-confidence LearnedPattern records from DB."""
        if not self.session_maker: self.logger.error("DB session_maker not available."); return []
        patterns = []
        try:
            async with self.session_maker() as session:
                stmt = select(LearnedPattern).where(LearnedPattern.confidence_score >= min_confidence, LearnedPattern.status == 'active')
                if tags:
                     tag_conditions = [LearnedPattern.tags.like(f'%"{tag}"%') for tag in tags]
                     stmt = stmt.where(or_(*tag_conditions))
                stmt = stmt.order_by(desc(LearnedPattern.timestamp)).limit(limit)
                result = await session.execute(stmt)
                patterns = list(result.scalars().all())
                self.logger.debug(f"Fetched {len(patterns)} learned patterns (min_confidence={min_confidence}).")
        except SQLAlchemyError as e:
            self.logger.error(f"DB Error getting latest patterns: {e}", exc_info=True)
            await self._report_error(f"DB Error getting patterns: {e}")
        except Exception as e: self.logger.error(f"Unexpected error getting latest patterns: {e}", exc_info=True)
        return patterns

    async def purge_old_knowledge(self, days_threshold: int = 30):
        """Deletes KnowledgeFragment records not accessed within the threshold."""
        if not self.session_maker: self.logger.error("DB session_maker not available."); return

        purge_cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_threshold)
        self.logger.info(f"Purging knowledge fragments last accessed before {purge_cutoff_date.isoformat()}...")
        purge_thought = f"Structured Thinking: Purge Old Knowledge. Plan: Execute DELETE on knowledge_fragments where last_accessed_ts < {purge_cutoff_date}. Log count."
        await self._internal_think(purge_thought)
        deleted_count = 0
        try:
            async with self.session_maker() as session:
                async with session.begin(): # Transaction
                    stmt = delete(KnowledgeFragment).where(KnowledgeFragment.last_accessed_ts < purge_cutoff_date)
                    result = await session.execute(stmt)
                    deleted_count = result.rowcount
                    # Commit happens automatically
            self.logger.info(f"Successfully purged {deleted_count} old knowledge fragments (last accessed > {days_threshold} days ago).")
            if deleted_count > 0 and self.orchestrator and hasattr(self.orchestrator, 'send_notification'):
                 await self.orchestrator.send_notification("Data Purge Completed", f"Purged {deleted_count} knowledge fragments older than {days_threshold} days.")
        except SQLAlchemyError as e:
            self.logger.error(f"DB Error purging old knowledge fragments: {e}", exc_info=True)
            await self._report_error(f"DB Error purging old knowledge: {e}")
        except Exception as e: self.logger.error(f"Unexpected error purging old knowledge: {e}", exc_info=True)

    async def handle_feedback(self, insights_data: Dict[str, Dict[str, Any]]):
        """Processes feedback insights, logs them, and triggers analysis/actions."""
        # ### Phase 2 Plan Ref: 4.6 (Implement handle_feedback)
        self.logger.info(f"ThinkTool received feedback insights from {len(insights_data)} agents.")
        feedback_thought = f"Structured Thinking: Process Agent Feedback. Plan: Log raw feedback -> Format summary -> Call LLM for analysis -> Parse response -> Create directives/schedule critiques/log insights."
        await self._internal_think(feedback_thought)

        # Log raw feedback
        for agent_name, agent_feedback in insights_data.items():
            if isinstance(agent_feedback, dict):
                try:
                    tags = ["feedback", agent_name.lower()]
                    if agent_feedback.get("status") == "error" or agent_feedback.get("errors_encountered_session", 0) > 0: tags.append("error")
                    await self.log_knowledge_fragment(
                        agent_source=agent_name, data_type="AgentFeedbackRaw", content=agent_feedback,
                        tags=tags, relevance_score=0.5
                    )
                except Exception as e: self.logger.error(f"Error logging raw feedback fragment for {agent_name}: {e}", exc_info=True)
            else: self.logger.warning(f"Received non-dict feedback from {agent_name}. Skipping.")

        # Proceed with LLM analysis
        feedback_summary = json.dumps(insights_data, indent=2, default=str, ensure_ascii=False)
        max_summary_len = 4000
        if len(feedback_summary) > max_summary_len: feedback_summary = feedback_summary[:max_summary_len] + "\n... (feedback truncated)"

        analysis_prompt = f"""
        {self.meta_prompt[:500]}...
        **Task:** Analyze consolidated agent feedback. Identify critical issues, successes, trends. Propose actions: StrategicDirectives (JSON), prompt critiques (list "Agent/Key"), insights to log (JSON for log_knowledge_fragment).
        **Feedback:** ```json\n{feedback_summary}\n```
        **Output (JSON):** {{"analysis_summary": str, "critical_issues_found": [str], "key_successes_noted": [str], "proposed_directives": [{{...}}], "prompts_to_critique": [str], "insights_to_log": [{{...}}]}}
        """
        analysis_json = await self._call_llm_with_retry(analysis_prompt, temperature=0.6, max_tokens=2000, is_json_output=True)

        if analysis_json:
            try:
                # Attempt to find JSON within potential markdown code blocks
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', analysis_json, re.DOTALL)
                if json_match:
                    analysis_result = json.loads(json_match.group(1))
                else:
                    analysis_result = json.loads(analysis_json) # Fallback

                self.logger.info(f"Feedback analysis complete. Summary: {analysis_result.get('analysis_summary', 'N/A')}")

                # Create directives
                async with self.session_maker() as session:
                     async with session.begin(): # Transaction
                         for directive_data in analysis_result.get('proposed_directives', []):
                             if isinstance(directive_data, dict) and all(k in directive_data for k in ['target_agent', 'directive_type', 'content', 'priority']):
                                 session.add(StrategicDirective(source="ThinkToolFeedback", timestamp=datetime.now(timezone.utc), status='pending', **directive_data))
                                 self.logger.info(f"Generated Directive from Feedback for {directive_data['target_agent']}")
                             else: self.logger.warning(f"Skipping invalid directive data: {directive_data}")
                         # Commit happens automatically

                # Trigger critiques
                for prompt_id in analysis_result.get('prompts_to_critique', []):
                    if isinstance(prompt_id, str):
                        try: agent_name, prompt_key = prompt_id.split('/', 1); asyncio.create_task(self.self_critique_prompt(agent_name, prompt_key, f"Feedback analysis suggested issues: {analysis_result.get('analysis_summary', 'N/A')}")); self.logger.info(f"Scheduled critique for: {prompt_id}")
                        except ValueError: self.logger.warning(f"Invalid prompt identifier format: {prompt_id}")
                        except Exception as critique_err: self.logger.error(f"Error scheduling critique for {prompt_id}: {critique_err}")
                    else: self.logger.warning(f"Invalid prompt identifier type: {type(prompt_id)}")


                # Log insights
                for frag_data in analysis_result.get('insights_to_log', []):
                     if isinstance(frag_data, dict) and all(k in frag_data for k in ['data_type', 'content']):
                         await self.log_knowledge_fragment(
                             agent_source="ThinkToolFeedback",
                             data_type=frag_data['data_type'],
                             content=frag_data['content'],
                             tags=frag_data.get('tags', ['feedback_insight']),
                             relevance_score=frag_data.get('relevance', 0.6) # Use provided relevance or default
                         )
                     else: self.logger.warning(f"Skipping invalid insight data: {frag_data}")

            except json.JSONDecodeError: self.logger.error(f"Failed decode feedback analysis JSON: {analysis_json}")
            except Exception as e: self.logger.error(f"Error processing feedback analysis result: {e}", exc_info=True)
        else:
            self.logger.error("Feedback analysis failed (LLM error).")

    # --- Prompt Template Management ---

    async def get_prompt(self, agent_name: str, prompt_key: str) -> Optional[str]:
        """Fetches the active prompt content from the database."""
        # ### Phase 2 Plan Ref: 4.7 (Implement get_prompt)
        if not self.session_maker: self.logger.error("DB session_maker not available."); return None
        self.logger.debug(f"Querying DB for active prompt: {agent_name}/{prompt_key}.")
        try:
            async with self.session_maker() as session:
                stmt = select(PromptTemplate.content).where(
                    PromptTemplate.agent_name == agent_name, PromptTemplate.prompt_key == prompt_key,
                    PromptTemplate.is_active == True
                ).order_by(desc(PromptTemplate.version)).limit(1)
                result = await session.execute(stmt)
                prompt_content = result.scalar_one_or_none()
                if prompt_content: self.logger.info(f"Fetched active prompt for {agent_name}/{prompt_key}"); return prompt_content
                else: self.logger.warning(f"No active prompt found for {agent_name}/{prompt_key}"); return None
        except SQLAlchemyError as e:
            self.logger.error(f"DB Error getting prompt {agent_name}/{prompt_key}: {e}", exc_info=True)
            await self._report_error(f"DB Error getting prompt {agent_name}/{prompt_key}: {e}")
            return None
        except Exception as e: self.logger.error(f"Unexpected error getting prompt: {e}", exc_info=True); return None

    async def update_prompt(self, agent_name: str, prompt_key: str, new_content: str, author_agent: str = "ThinkTool") -> Optional[PromptTemplate]:
        """Creates a new version of a prompt, deactivates the old one."""
        # ### Phase 2 Plan Ref: 4.7 (Implement update_prompt)
        if not self.session_maker: self.logger.error("DB session_maker not available."); return None
        new_version = 1
        try:
            async with self.session_maker() as session:
                async with session.begin(): # Transaction
                    # Find current active version FOR UPDATE to lock the row(s)
                    stmt_current = select(PromptTemplate.id, PromptTemplate.version).where(
                        PromptTemplate.agent_name == agent_name, PromptTemplate.prompt_key == prompt_key,
                        PromptTemplate.is_active == True
                    ).order_by(desc(PromptTemplate.version)).limit(1).with_for_update()
                    result_current = await session.execute(stmt_current)
                    current_active_row = result_current.fetchone()

                    if current_active_row:
                        current_active_id, current_version = current_active_row
                        new_version = current_version + 1
                        # Deactivate the old version
                        stmt_deactivate = update(PromptTemplate).where(PromptTemplate.id == current_active_id).values(is_active=False)
                        await session.execute(stmt_deactivate)
                        self.logger.info(f"Deactivated prompt v{current_version} for {agent_name}/{prompt_key}")

                    # Create the new active version
                    new_template = PromptTemplate(
                        agent_name=agent_name, prompt_key=prompt_key, version=new_version,
                        content=new_content, is_active=True, author_agent=author_agent,
                        last_updated=datetime.now(timezone.utc)
                    )
                    session.add(new_template)
                    # Commit happens automatically

                # Refresh outside transaction to get generated ID
                await session.refresh(new_template)
                self.logger.info(f"Created and activated new prompt v{new_version} for {agent_name}/{prompt_key}")
                return new_template
        except SQLAlchemyError as e:
            self.logger.error(f"DB Error updating prompt {agent_name}/{prompt_key}: {e}", exc_info=True)
            await self._report_error(f"DB Error updating prompt {agent_name}/{prompt_key}: {e}")
            return None
        except Exception as e: self.logger.error(f"Unexpected error updating prompt: {e}", exc_info=True); return None

    # --- Enhanced Reflection & Validation ---

    async def reflect_on_action(self, context: str, agent_name: str, task_description: str) -> dict:
        """Enhanced reflection incorporating KB context and risk assessment."""
        # ### Phase 2 Plan Ref: 4.9 (Implement reflect_on_action)
        self.logger.debug(f"Starting reflection for {agent_name} on task: {task_description}")
        reflect_thought = f"Structured Thinking: Reflect on Action for {agent_name}. Task: {task_description[:50]}... Plan: Fetch KB context -> Format prompt -> Call LLM -> Process response -> Trigger KB updates -> Return result."
        await self._internal_think(reflect_thought)
        kb_context = ""
        try:
            active_directives = await self.get_active_directives(target_agent=agent_name)
            # Query for fragments relevant to the agent and task description keywords
            task_keywords = [w for w in re.findall(r'\b\w{4,}\b', task_description.lower()) if w not in ['task', 'agent', 'perform', 'execute']]
            query_tags = [agent_name.lower(), 'strategy', 'feedback'] + task_keywords[:3]
            relevant_fragments = await self.query_knowledge_base(tags=query_tags, limit=10, time_window=timedelta(days=7))
            relevant_patterns = await self.get_latest_patterns(tags=[agent_name.lower()], limit=5)

            if active_directives: kb_context += "\n\n**Active Directives:**\n" + "\n".join([f"- ID {d.id}: {d.content[:100]}..." for d in active_directives])
            if relevant_fragments: kb_context += "\n\n**Recent Relevant Fragments:**\n" + "\n".join([f"- ID {f.id} ({f.data_type}): {f.content[:80]}..." for f in relevant_fragments])
            if relevant_patterns: kb_context += "\n\n**Relevant Learned Patterns:**\n" + "\n".join([f"- ID {p.id}: {p.pattern_description[:150]}..." for p in relevant_patterns])
        except Exception as e: self.logger.error(f"Error fetching KB context for reflection: {e}"); kb_context = "\n\n**Warning:** Failed KB context retrieval."

        prompt = f"""
        {self.meta_prompt[:500]}...
        **Agent:** {agent_name} | **Task:** {task_description} | **Context Provided by Agent:** {context}
        **Relevant Knowledge Base Context:** {kb_context or 'None'}
        **Analysis Required:** Based on the agent's context, task, and KB info:
        1. Is the provided context sufficient and data complete for the task? (Yes/No/Partial)
        2. Are there immediate compliance concerns based on KB or task description? (List flags or None)
        3. Assess the risk level of the proposed action/next step (Low/Medium/High/Critical).
        4. Does the action align with agency goals (Profit, Growth, Efficiency)?
        5. Suggest the most logical, efficient, and compliant next step.
        6. Estimate confidence (0.0-1.0) in this assessment.
        **Output (JSON ONLY):** {{"proceed": bool, "reason": str, "risk_level": str, "compliance_flags": [str], "next_step": str, "confidence": float, "log_fragment": {{...}}?, "update_directive": {{...}}?}}
        (log_fragment: optional dict for log_knowledge_fragment if insight gained. update_directive: optional dict {{directive_id: int, status: str, result_summary: str?}} if reflection completes/fails a directive)
        """
        reflection_json = await self._call_llm_with_retry(prompt, temperature=0.3, max_tokens=1000, is_json_output=True)

        if reflection_json:
            try:
                # Attempt to find JSON within potential markdown code blocks
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', reflection_json, re.DOTALL)
                if json_match:
                    reflection = json.loads(json_match.group(1))
                else:
                    reflection = json.loads(reflection_json) # Fallback

                reflection.setdefault('proceed', False); reflection.setdefault('reason', 'Analysis failed.'); reflection.setdefault('risk_level', 'Unknown'); reflection.setdefault('compliance_flags', []); reflection.setdefault('next_step', 'Manual review.'); reflection.setdefault('confidence', 0.5)
                self.logger.info(f"Reflection for {agent_name}: Proceed={reflection['proceed']}, Risk={reflection['risk_level']}, Reason={reflection['reason']}")
                # Trigger KB updates
                if 'log_fragment' in reflection and isinstance(reflection['log_fragment'], dict):
                    frag_data = reflection['log_fragment']
                    if all(k in frag_data for k in ['data_type', 'content']): await self.log_knowledge_fragment(agent_source="ThinkToolReflection", data_type=frag_data['data_type'], content=frag_data['content'], tags=frag_data.get('tags'), relevance_score=frag_data.get('relevance', 0.6))
                if 'update_directive' in reflection and isinstance(reflection['update_directive'], dict):
                    directive_data = reflection['update_directive']
                    if all(k in directive_data for k in ['directive_id', 'status']): await self.update_directive_status(directive_id=directive_data['directive_id'], status=directive_data['status'], result_summary=directive_data.get('result_summary'))
                return reflection
            except json.JSONDecodeError: self.logger.error(f"Failed decode JSON reflection: {reflection_json}")
            except Exception as e: self.logger.error(f"Error processing reflection result: {e}", exc_info=True)
        # Fallback
        return {"proceed": False, "reason": "ThinkTool analysis failed.", "risk_level": "Critical", "compliance_flags": ["Analysis Failure"], "next_step": "Halt task.", "confidence": 0.0}

    async def validate_output(self, output_to_validate: str, validation_criteria: str, agent_name: str, context: str = None) -> dict:
        """Enhanced validation incorporating learned patterns."""
        # ### Phase 2 Plan Ref: 4.9 (Implement validate_output)
        self.logger.debug(f"Starting validation for {agent_name}'s output.")
        validate_thought = f"Structured Thinking: Validate Output from {agent_name}. Criteria: {validation_criteria[:50]}... Plan: Fetch patterns -> Format prompt -> Call LLM -> Parse -> Return."
        await self._internal_think(validate_thought)
        pattern_context = ""
        try:
            # Fetch patterns relevant to the agent or criteria keywords
            criteria_tags = [w for w in re.findall(r'\b\w{4,}\b', validation_criteria.lower()) if w not in ['validate', 'check', 'ensure']]
            query_tags = [agent_name.lower()] + criteria_tags[:3]
            relevant_patterns = await self.get_latest_patterns(tags=query_tags, limit=5, min_confidence=0.6)
            if relevant_patterns: pattern_context += "\n\n**Relevant Learned Patterns (Consider these):**\n" + "\n".join([f"- ID {p.id}: {p.pattern_description}" for p in relevant_patterns])
        except Exception as e: self.logger.error(f"Error fetching patterns for validation: {e}"); pattern_context = "\n\n**Warning:** Failed pattern retrieval."

        prompt = f"""
        {self.meta_prompt[:500]}...
        **Agent:** {agent_name} | **Context:** {context or 'N/A'}
        **Output to Validate:** ```\n{output_to_validate}\n```
        **Validation Criteria:** {validation_criteria} {pattern_context}
        **Checks:**
        1. Does the output strictly meet ALL validation criteria?
        2. Is the output content factually plausible and logically sound?
        3. Is the output consistent with relevant learned patterns (if any)?
        4. Is the output directly usable for its intended purpose?
        5. Are there any obvious compliance or ethical risks in the output?
        **Output (JSON ONLY):** {{"valid": bool, "feedback": "Concise explanation for validity (pass/fail) referencing specific criteria/patterns/checks.", "suggested_fix": "If invalid, provide a specific, actionable suggestion for correction."}}
        """
        validation_json = await self._call_llm_with_retry(prompt, temperature=0.2, max_tokens=800, is_json_output=True)

        if validation_json:
            try:
                # Attempt to find JSON within potential markdown code blocks
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', validation_json, re.DOTALL)
                if json_match:
                    validation = json.loads(json_match.group(1))
                else:
                    validation = json.loads(validation_json) # Fallback

                validation.setdefault('valid', False); validation.setdefault('feedback', 'Validation analysis failed.')
                self.logger.info(f"Validation for {agent_name}: Valid={validation['valid']}, Feedback={validation['feedback'][:100]}...")
                return validation
            except json.JSONDecodeError: self.logger.error(f"Failed decode JSON validation: {validation_json}")
            except Exception as e: self.logger.error(f"Error processing validation result: {e}", exc_info=True)
        # Fallback
        return {"valid": False, "feedback": "ThinkTool validation failed."}

    # --- Core Synthesis & Strategy Engines ---

    async def synthesize_insights_and_strategize(self):
        """Central cognitive loop for learning, synthesis, and strategy generation."""
        # ### Phase 2 Plan Ref: 4.4 (Implement synthesize_insights_and_strategize)
        self.logger.info("ThinkTool: Starting synthesis and strategy cycle.")
        synth_thought = "Structured Thinking: Synthesize & Strategize. Plan: Query KB (fragments, patterns, perf) -> Format -> Call LLM -> Parse -> Store outputs (patterns, directives, opportunities)."
        await self._internal_think(synth_thought)
        try:
            # Fetch data
            async with self.session_maker() as session:
                # Fetch a broader range of recent fragments
                stmt_frags = select(KnowledgeFragment).where(KnowledgeFragment.last_accessed_ts >= datetime.now(timezone.utc) - timedelta(days=7)).order_by(desc(KnowledgeFragment.relevance_score), desc(KnowledgeFragment.last_accessed_ts)).limit(200)
                recent_fragments = list((await session.execute(stmt_frags)).scalars().all())

                # Fetch active patterns
                stmt_patterns = select(LearnedPattern).where(LearnedPattern.status == 'active').order_by(desc(LearnedPattern.confidence_score), desc(LearnedPattern.timestamp)).limit(20)
                recent_patterns = list((await session.execute(stmt_patterns)).scalars().all())

                # Fetch recent performance data (logs, invoices)
                stmt_perf = select(EmailLog.status, CallLog.outcome, Invoice.status, Client.engagement_score).select_from(Client).outerjoin(EmailLog, Client.id == EmailLog.client_id).outerjoin(CallLog, Client.id == CallLog.client_id).outerjoin(Invoice, Client.id == Invoice.client_id).where(or_(EmailLog.timestamp >= datetime.now(timezone.utc) - timedelta(days=7), CallLog.timestamp >= datetime.now(timezone.utc) - timedelta(days=7), Invoice.timestamp >= datetime.now(timezone.utc) - timedelta(days=7))).limit(500)
                perf_results = await session.execute(stmt_perf)
                perf_data = perf_results.mappings().all() # Get list of dicts

            if not recent_fragments and not perf_data: self.logger.warning("ThinkTool Synthesis: Insufficient recent data."); return

            # Format data for prompt (summarize heavily)
            fragments_summary = [{"id": f.id, "type": f.data_type, "src": f.agent_source, "preview": (f.content if isinstance(f.content, str) else json.dumps(f.content))[:80]+"..."} for f in recent_fragments[:20]]
            patterns_summary = [{"id": p.id, "desc": p.pattern_description, "conf": p.confidence_score} for p in recent_patterns]
            # Summarize performance data (e.g., counts)
            perf_counts = Counter(r['status'] for r in perf_data if r.get('status')) # EmailLog status counts
            perf_counts.update(Counter(r['outcome'] for r in perf_data if r.get('outcome'))) # CallLog outcome counts
            perf_counts.update(Counter(r['status_1'] for r in perf_data if r.get('status_1'))) # Invoice status counts
            avg_score = np.mean([r['engagement_score'] for r in perf_data if r.get('engagement_score') is not None]) if perf_data else 0
            perf_summary_str = f"Counts: {dict(perf_counts)}. Avg Score: {avg_score:.2f}"

            synthesis_prompt = f"""
            {self.meta_prompt[:500]}...
            **Task:** Synthesize insights from recent data. Identify/validate patterns, assess goal progress, generate directives & opportunities. Focus on $10k+/day profit.
            **Data:**
            - Recent Fragments ({len(recent_fragments)}): {json.dumps(fragments_summary)}
            - Active Patterns ({len(recent_patterns)}): {json.dumps(patterns_summary)}
            - Performance Summary (Last 7d): {perf_summary_str}
            **Analysis:**
            1. Identify 1-3 **Novel Patterns** observed in fragments/performance (format: {{"description": str, "supporting_fragment_ids": [int], "confidence": float, "implications": str, "tags": [str]}}).
            2. Identify 1-2 existing **Patterns to Update/Validate** based on new data (format: {{"pattern_id": int, "action": "validate"|"obsolete"|"refine", "confidence_update": float?, "reason": str}}).
            3. Assess **Goal Progress**: How is performance tracking towards $10k/day? What are key blockers/accelerators?
            4. Propose 1-3 high-priority **Strategic Directives** for agents to improve profit/efficiency (format: {{"target_agent": str, "directive_type": str, "content": str, "priority": int (1-10)}}).
            5. Identify 1-2 concrete **Business Opportunities** (new service, target market, grey area) based on synthesis (format: {{"description": str, "potential_roi": str, "next_steps": str, "tags": [str]}}).
            **Output (JSON ONLY):** {{"new_patterns": [], "pattern_updates": [], "goal_assessment": "", "proposed_directives": [], "identified_opportunities": []}}
            """
            synthesis_json = await self._call_llm_with_retry(synthesis_prompt, temperature=0.7, max_tokens=3000, is_json_output=True) # Increased tokens

            if synthesis_json:
                try:
                    # Attempt to find JSON within potential markdown code blocks
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', synthesis_json, re.DOTALL)
                    if json_match:
                        synthesis_result = json.loads(json_match.group(1))
                    else:
                        synthesis_result = json.loads(synthesis_json) # Fallback

                    self.logger.info(f"ThinkTool Synthesis cycle completed. Assessment: {synthesis_result.get('goal_assessment', 'N/A')}")

                    # Log new patterns
                    for p_data in synthesis_result.get('new_patterns', []):
                         if isinstance(p_data, dict) and all(k in p_data for k in ['description', 'confidence_score', 'implications', 'supporting_fragment_ids']):
                             await self.log_learned_pattern(
                                 pattern_description=p_data['description'],
                                 supporting_fragment_ids=p_data['supporting_fragment_ids'],
                                 confidence_score=p_data['confidence_score'],
                                 implications=p_data['implications'],
                                 tags=p_data.get('tags')
                             )
                         else: self.logger.warning(f"Skipping invalid new pattern data: {p_data}")

                    # TODO: Handle pattern updates (requires fetching pattern by ID and updating status/confidence)

                    # Create directives
                    async with self.session_maker() as session:
                         async with session.begin(): # Transaction
                             for d_data in synthesis_result.get('proposed_directives', []):
                                 if isinstance(d_data, dict) and all(k in d_data for k in ['target_agent', 'directive_type', 'content', 'priority']):
                                     session.add(StrategicDirective(source="ThinkToolSynthesis", timestamp=datetime.now(timezone.utc), status='pending', **d_data))
                                     self.logger.info(f"Generated Directive for {d_data['target_agent']}: {d_data['directive_type']}")
                                 else: self.logger.warning(f"Skipping invalid directive data: {d_data}")
                             # Commit happens automatically

                    # Log opportunities
                    for o_data in synthesis_result.get('identified_opportunities', []):
                         if isinstance(o_data, dict) and 'description' in o_data:
                             await self.log_knowledge_fragment(
                                 agent_source="ThinkToolSynthesis", data_type="business_opportunity_signal",
                                 content=o_data, tags=o_data.get('tags', ['opportunity']), relevance_score=0.8
                             )
                         else: self.logger.warning(f"Skipping invalid opportunity data: {o_data}")

                except json.JSONDecodeError: self.logger.error(f"Failed decode JSON synthesis result: {synthesis_json}")
                except Exception as e: self.logger.error(f"Error processing synthesis result: {e}", exc_info=True)
            else: self.logger.error("Synthesis analysis failed (LLM error).")
        except Exception as e:
            self.logger.error(f"Error during synthesis cycle: {e}", exc_info=True)
            await self._report_error(f"Synthesis cycle failed: {e}")

    async def technology_radar(self):
        """Proactively scouts for relevant new tools, APIs, research, and techniques."""
        # ### Phase 2 Plan Ref: 4.5 (Implement technology_radar)
        self.logger.info("ThinkTool: Starting technology radar cycle.")
        radar_thought = "Structured Thinking: Technology Radar. Plan: Define topics -> Delegate search (BrowsingAgent) -> Analyze results (LLM) -> Log findings/directives (KB)."
        await self._internal_think(radar_thought)
        try:
            # More targeted search topics
            search_topics = [
                "Clay.com alternative API for lead enrichment",
                "AI tools for hyper-personalized cold email",
                "Open source voice cloning for sales calls",
                "New techniques for bypassing email spam filters",
                "Automated UGC video generation platforms",
                "Best proxy providers for web scraping 2025",
                "Advances in autonomous AI agent frameworks",
                "Competitor analysis tools for B2B SaaS"
            ]
            search_query = f"Latest developments, tools, or research papers on: {random.choice(search_topics)}"
            search_results_summary = None

            if self.orchestrator and hasattr(self.orchestrator, 'delegate_task'):
                 # Delegate search to BrowsingAgent
                 search_task = {"action": "perform_search_and_summarize", "query": search_query, "num_results": 5}
                 browsing_agent_name = "BrowsingAgent" # Assuming this is the correct key
                 if browsing_agent_name not in self.orchestrator.agents:
                      self.logger.error(f"{browsing_agent_name} not found in orchestrator agents.")
                      return # Cannot perform search

                 search_result = await self.orchestrator.delegate_task(browsing_agent_name, search_task)
                 if search_result and search_result.get("status") == "success":
                     search_results_summary = search_result.get("summary")
                     self.logger.info(f"Radar: Received search summary for '{search_query}'.")
                 else:
                     self.logger.warning(f"Radar: BrowsingAgent search failed: {search_result.get('message') if search_result else 'No result'}")
            else:
                self.logger.error("Radar: Orchestrator or delegate_task unavailable.")
                return # Cannot perform search

            if not search_results_summary: self.logger.info("Radar: No findings from web search."); return

            analysis_prompt = f"""
            {self.meta_prompt[:500]}...
            **Task:** Analyze tech scouting report for AI Sales Agency. Identify novel, high-impact tools/techniques relevant to **profit maximization**, **efficiency**, or **grey area exploitation**. Assess benefits, risks, integration effort. Recommend next steps (Log, Directive, Ignore).
            **Report:** ```\n{search_results_summary}\n```
            **Analysis Focus:** Novelty/Impact, Relevance to Sales/UGC/Clay.com/Automation, Benefits, Risks (Technical/Legal/Ban), Integration Effort (Low/Med/High), Recommendation ('Log Insight', 'Generate Directive: [Brief Action]', 'Ignore').
            **Output (JSON ONLY):** {{"analyzed_items": [ {{ "item_name": str, "summary": str, "relevance": str, "benefits": [str], "risks": [str], "integration_effort": str, "recommendation": str }} ], "overall_assessment": "Brief summary of findings."}}
            """
            analysis_json = await self._call_llm_with_retry(analysis_prompt, temperature=0.4, max_tokens=1500, is_json_output=True)

            if analysis_json:
                try:
                    # Attempt to find JSON within potential markdown code blocks
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', analysis_json, re.DOTALL)
                    if json_match:
                        analysis_result = json.loads(json_match.group(1))
                    else:
                        analysis_result = json.loads(analysis_json) # Fallback

                    self.logger.info(f"Radar analysis complete. Found {len(analysis_result.get('analyzed_items', []))} relevant items.")
                    async with self.session_maker() as session:
                        async with session.begin(): # Transaction for directives
                            for item in analysis_result.get('analyzed_items', []):
                                if not isinstance(item, dict) or not item.get("item_name"): continue
                                # Log the finding
                                await self.log_knowledge_fragment(
                                    agent_source="ThinkToolRadar", data_type="new_tool_discovery",
                                    content=item, tags=["technology", "scouting", item.get("item_name", "unknown").lower().replace(" ", "_")],
                                    relevance_score=0.7 # Default relevance for radar findings
                                )
                                # Create directive if recommended
                                if "directive" in item.get("recommendation", "").lower():
                                    directive_content = item.get("recommendation").split(":", 1)[-1].strip() if ":" in item.get("recommendation", "") else f"Investigate tool/technique: {item.get('item_name')}"
                                    session.add(StrategicDirective(
                                        source="ThinkToolRadar", timestamp=datetime.now(timezone.utc),
                                        target_agent="Orchestrator", # Orchestrator decides who handles investigation
                                        directive_type="investigate_tool_technique",
                                        content=json.dumps({"item_name": item.get("item_name"), "details": item, "requested_action": directive_content}),
                                        priority=7, status='pending' # Lower priority for investigation
                                    ))
                                    self.logger.info(f"Generated investigation directive for: {item.get('item_name', 'N/A')}")
                            # Commit happens automatically
                except json.JSONDecodeError: self.logger.error(f"Radar: Failed decode JSON analysis: {analysis_json}")
                except Exception as e: self.logger.error(f"Radar: Error processing analysis result: {e}", exc_info=True)
            else: self.logger.error("Radar: Analysis failed (LLM error).")
        except Exception as e:
            self.logger.error(f"Error during technology radar cycle: {e}", exc_info=True)
            await self._report_error(f"Technology radar cycle failed: {e}")

    # --- Self-Improving Prompt Mechanism ---

    async def self_critique_prompt(self, agent_name: str, prompt_key: str, feedback_context: str):
        """Attempts to refine a prompt based on negative feedback or failure analysis."""
        # ### Phase 2 Plan Ref: 4.8 (Implement self_critique_prompt)
        self.logger.info(f"Starting self-critique for prompt: {agent_name}/{prompt_key}")
        critique_thought = f"Structured Thinking: Self-Critique Prompt {agent_name}/{prompt_key}. Plan: Fetch prompt -> Format critique prompt -> Call LLM -> Parse -> Update prompt -> Generate test directive."
        await self._internal_think(critique_thought)
        try:
            current_prompt = await self.get_prompt(agent_name, prompt_key)
            if not current_prompt: self.logger.error(f"Critique: Cannot find active prompt {agent_name}/{prompt_key}."); return

            critique_prompt = f"""
            {self.meta_prompt[:500]}...
            **Task:** Critique and rewrite LLM prompt for agent '{agent_name}', key '{prompt_key}'. Improve based on feedback/context. Use self-instruction principles. Aim for clarity, robustness, better goal alignment (profit/efficiency), and adherence to agent's meta-prompt.
            **Feedback/Context:** {feedback_context}
            **Current Prompt:** ```\n{current_prompt}\n```
            **Analysis & Rewrite:**
            1. **Critique:** Identify specific weaknesses in the current prompt based on the feedback (e.g., ambiguity, missing constraints, poor formatting instructions, lack of context).
            2. **Improved Prompt:** Rewrite the *entire* prompt incorporating improvements. Be specific, add constraints, clarify output format (use JSON where applicable), guide the LLM's reasoning process if needed. Ensure it aligns with the agent's core function and the overall agency goals.
            **Output (JSON ONLY):** {{ "critique": "Detailed critique of the original prompt.", "improved_prompt": "The complete rewritten prompt text." }}
            """
            critique_json = await self._call_llm_with_retry(critique_prompt, temperature=0.6, max_tokens=3000, is_json_output=True) # Increased tokens

            if critique_json:
                try:
                    # Attempt to find JSON within potential markdown code blocks
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', critique_json, re.DOTALL)
                    if json_match:
                        critique_result = json.loads(json_match.group(1))
                    else:
                        critique_result = json.loads(critique_json) # Fallback

                    improved_prompt = critique_result.get('improved_prompt')
                    critique_text = critique_result.get('critique')
                    if improved_prompt and isinstance(improved_prompt, str) and improved_prompt.strip():
                        self.logger.info(f"Critique generated improved prompt for {agent_name}/{prompt_key}. Critique: {critique_text}")
                        # Update the prompt in the database
                        new_template = await self.update_prompt(agent_name, prompt_key, improved_prompt, author_agent="ThinkToolCritique")
                        if new_template:
                            # Generate directive to test the new prompt
                            async with self.session_maker() as session:
                                async with session.begin(): # Transaction
                                    session.add(StrategicDirective(
                                        source="ThinkToolCritique", timestamp=datetime.now(timezone.utc), target_agent=agent_name,
                                        directive_type="test_prompt_variation",
                                        content=json.dumps({"prompt_key": prompt_key, "new_version": new_template.version, "critique": critique_text, "old_prompt_preview": current_prompt[:200]+"..."}),
                                        priority=7, status='pending'
                                    ))
                                # Commit happens automatically
                                self.logger.info(f"Generated directive to test new prompt v{new_template.version} for {agent_name}/{prompt_key}")
                        else: self.logger.error(f"Critique: Failed to save improved prompt {agent_name}/{prompt_key}.")
                    else: self.logger.warning(f"Critique for {agent_name}/{prompt_key} did not produce a valid improved prompt.")
                except json.JSONDecodeError: self.logger.error(f"Critique: Failed decode JSON result: {critique_json}")
                except Exception as e: self.logger.error(f"Critique: Error processing result: {e}", exc_info=True)
            else: self.logger.error(f"Critique: Failed get critique/rewrite from LLM for {agent_name}/{prompt_key}.")
        except Exception as e:
            self.logger.error(f"Error during self-critique for {agent_name}/{prompt_key}: {e}", exc_info=True)
            await self._report_error(f"Self-critique failed for {agent_name}/{prompt_key}: {e}")

    # --- Agent Run Loop ---
    async def run(self):
        """Main loop for ThinkTool: Periodic analysis, strategy, and optimization."""
        # ### Phase 2 Plan Ref: 4.10 (Implement run loop)
        if self.status == self.STATUS_RUNNING: self.logger.warning("ThinkTool run() called while already running."); return
        self.logger.info("ThinkTool v4.1 starting run loop...")
        self._status = self.STATUS_RUNNING # Use property setter
        synthesis_interval = timedelta(seconds=int(self.config.get("THINKTOOL_SYNTHESIS_INTERVAL_SECONDS", 3600)))
        radar_interval = timedelta(seconds=int(self.config.get("THINKTOOL_RADAR_INTERVAL_SECONDS", 21600)))
        purge_interval = timedelta(seconds=int(self.config.get("DATA_PURGE_INTERVAL_SECONDS", 86400)))

        # Initialize last run times to allow immediate first run if needed
        now = datetime.now(timezone.utc)
        self.last_synthesis_run = now - synthesis_interval
        self.last_radar_run = now - radar_interval
        self.last_purge_run = now - purge_interval

        while self.status == self.STATUS_RUNNING and not self._stop_event.is_set():
            try:
                current_time = datetime.now(timezone.utc)
                # Check orchestrator approval status dynamically
                is_approved = getattr(self.orchestrator, 'approved', False)

                if is_approved:
                    # Synthesis Cycle
                    if current_time - self.last_synthesis_run >= synthesis_interval:
                        self.logger.info("ThinkTool: Triggering Synthesis & Strategy cycle.")
                        await self.synthesize_insights_and_strategize()
                        self.last_synthesis_run = current_time

                    # Technology Radar Cycle
                    if current_time - self.last_radar_run >= radar_interval:
                        self.logger.info("ThinkTool: Triggering Technology Radar cycle.")
                        await self.technology_radar()
                        self.last_radar_run = current_time

                    # Data Purge Cycle
                    if current_time - self.last_purge_run >= purge_interval:
                         self.logger.info("ThinkTool: Triggering Data Purge cycle.")
                         await self.purge_old_knowledge() # Uses default 30 days
                         self.last_purge_run = current_time
                else:
                    self.logger.debug("ThinkTool: Orchestrator not approved. Skipping periodic tasks.")

                # Sleep interval before next check
                await asyncio.sleep(60 * 5) # Check every 5 minutes

            except asyncio.CancelledError:
                self.logger.info("ThinkTool run loop cancelled.")
                break # Exit loop cleanly on cancellation
            except Exception as e:
                self.logger.critical(f"ThinkTool: CRITICAL error in run loop: {e}", exc_info=True)
                self._status = self.STATUS_ERROR # Use property setter
                await self._report_error(f"Critical run loop error: {e}")
                await asyncio.sleep(60 * 15) # Wait longer after critical error before potentially retrying loop

        # Loop finished (either by stop signal or cancellation)
        if self.status != self.STATUS_STOPPING: # Avoid double logging if stop() was called
             self.status = self.STATUS_STOPPED
        self.logger.info("ThinkTool run loop finished.")

    # --- Abstract Method Implementations ---
    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handles tasks delegated specifically to ThinkTool."""
        # ### Phase 2 Plan Ref: 4.11 (Implement execute_task)
        self._status = self.STATUS_EXECUTING
        action = task_details.get('action')
        context = task_details.get('context') # General context if provided
        result = {"status": "failure", "message": f"Unknown ThinkTool action: {action}"}
        self.logger.info(f"ThinkTool executing task: {action}")
        exec_thought = f"Structured Thinking: Execute Task '{action}'. Plan: Route action to appropriate method (reflect, validate, process_feedback, etc.). Execute method. Return result."
        await self._internal_think(exec_thought)
        try:
            if action == 'reflect_on_action':
                agent_name = task_details.get('agent_name'); task_desc = task_details.get('task_description')
                if context and agent_name and task_desc: reflection = await self.reflect_on_action(context, agent_name, task_desc); result = {"status": "success", "reflection": reflection}
                else: result = {"status": "failure", "message": "Missing context/agent_name/task_description for reflection."}
            elif action == 'validate_output':
                output = task_details.get('output_to_validate'); criteria = task_details.get('validation_criteria'); agent_name = task_details.get('agent_name')
                if output and criteria and agent_name: validation = await self.validate_output(output, criteria, agent_name, context); result = {"status": "success", "validation": validation}
                else: result = {"status": "failure", "message": "Missing output/criteria/agent_name for validation."}
            elif action == 'process_feedback':
                 feedback_data = task_details.get('feedback_data')
                 if feedback_data: await self.handle_feedback(feedback_data); result = {"status": "success", "message": "Feedback processed."} # handle_feedback logs details
                 else: result = {"status": "failure", "message": "Missing feedback_data for processing."}
            elif action == 'generate_educational_content':
                 topic = task_details.get('topic')
                 if topic: explanation = await self.generate_educational_content(topic, context); result = {"status": "success" if explanation else "failure", "explanation": explanation}
                 else: result = {"status": "failure", "message": "Missing topic for educational content."}
            elif action == 'call_clay_api': # Example: Task to decide *if* and *how* to call Clay
                 target_data = task_details.get('target_data')
                 if target_data:
                      # Logic to decide parameters based on target_data and KB
                      clay_params = {"query": f"Find email for LinkedIn profile {target_data.get('linkedin_url')}"} # Simplified example
                      # Create directive for Orchestrator/BrowsingAgent
                      async with self.session_maker() as session:
                           async with session.begin():
                                session.add(StrategicDirective(
                                     source=self.AGENT_NAME, target_agent="Orchestrator", directive_type="execute_clay_call",
                                     content=json.dumps(clay_params), priority=3, status='pending'
                                ))
                           result = {"status": "success", "message": "Clay API call directive generated."}
                 else: result = {"status": "failure", "message": "Missing target_data for call_clay_api task."}
            # Add other specific actions ThinkTool handles directly
            else: self.logger.warning(f"Unhandled action in ThinkTool.execute_task: {action}")
        except Exception as e:
             self.logger.error(f"Error executing ThinkTool task '{action}': {e}", exc_info=True)
             result = {"status": "error", "message": f"Exception during task '{action}': {e}"}
             await self._report_error(f"Error executing task '{action}': {e}")
        self._status = self.STATUS_IDLE
        return result

    async def learning_loop(self):
        # Core learning happens in run() periodic tasks (synthesize_insights_and_strategize)
        self.logger.info("ThinkTool learning_loop: Core learning logic is in run() periodic tasks.")
        while self.status == self.STATUS_RUNNING and not self._stop_event.is_set():
             await asyncio.sleep(3600) # Sleep for an hour if called directly

    async def self_critique(self) -> Dict[str, Any]:
        """Evaluates ThinkTool's own effectiveness."""
        # ### Phase 2 Plan Ref: 4.13 (Implement self_critique)
        self.logger.info("ThinkTool: Performing self-critique.")
        critique = {"status": "ok", "feedback": "Critique pending analysis."}
        critique_thought = "Structured Thinking: Self-Critique ThinkTool. Plan: Query DB stats -> Analyze -> Format -> Return."
        await self._internal_think(critique_thought)
        try:
            async with self.session_maker() as session:
                kf_count = await session.scalar(select(func.count(KnowledgeFragment.id)))
                pattern_count = await session.scalar(select(func.count(LearnedPattern.id)))
                directive_counts = await session.execute(select(StrategicDirective.status, func.count(StrategicDirective.id)).group_by(StrategicDirective.status))
                directive_status = {row.status: row[1] for row in directive_counts.mappings().all()}
            critique['kb_stats'] = {'fragments': kf_count, 'patterns': pattern_count}
            critique['directive_stats'] = directive_status
            feedback = f"KB Size: {kf_count} fragments, {pattern_count} patterns. Directives: {directive_status}. "
            failed_directives = directive_status.get('failed', 0)
            completed_directives = directive_status.get('completed', 0) + directive_status.get('active', 0) # Consider active as progressing
            if failed_directives > 0 and completed_directives > 0 and failed_directives / (completed_directives + failed_directives) > 0.2: # If failure rate > 20%
                feedback += "High directive failure rate observed. " ; critique['status'] = 'warning'
            elif failed_directives > 5 and completed_directives < 10: # High absolute failures early on
                 feedback += "High absolute number of failed directives early on. " ; critique['status'] = 'warning'
            critique['feedback'] = feedback
        except Exception as e: self.logger.error(f"Error during self-critique: {e}", exc_info=True); critique['status'] = 'error'; critique['feedback'] = f"Critique failed: {e}"
        return critique

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Constructs context-rich prompts for internal LLM calls."""
        # ### Phase 2 Plan Ref: 4.14 (Implement generate_dynamic_prompt)
        self.logger.debug(f"Generating dynamic prompt for ThinkTool task: {task_context.get('task')}")
        prompt_gen_thought = f"Structured Thinking: Generate Dynamic Prompt for task '{task_context.get('task')}'. Plan: Combine meta-prompt, task context, KB context (if needed), instructions."
        await self._internal_think(prompt_gen_thought)
        prompt_parts = [self.meta_prompt] # Start with ThinkTool's own meta-prompt

        prompt_parts.append("\n--- Current Task Context ---")
        # Add specific task details, limiting length of large items
        for key, value in task_context.items():
            value_str = ""
            max_len = 1500 # Max length for context items in prompt
            if isinstance(value, str): value_str = value[:max_len] + ("..." if len(value) > max_len else "")
            elif isinstance(value, (int, float, bool)): value_str = str(value)
            elif isinstance(value, (dict, list)):
                try: value_str = json.dumps(value, default=str, indent=2); value_str = value_str[:max_len] + ("..." if len(value_str) > max_len else "")
                except TypeError: value_str = str(value)[:max_len] + "..."
            else: value_str = str(value)[:max_len] + "..." # Fallback for other types

            prompt_parts.append(f"**{key.replace('_', ' ').title()}**: {value_str}")

        # Add relevant KB context if needed (e.g., fetched patterns for validation)
        # This should be formatted and passed *into* task_context by the calling method

        prompt_parts.append("\n--- Instructions ---")
        prompt_parts.append(f"Based on the above context, perform the task: **{task_context.get('task', 'N/A')}**")
        if task_context.get('desired_output_format'):
            prompt_parts.append(f"**Output Format:** {task_context['desired_output_format']}")
        else:
            prompt_parts.append("**Output:** Provide a clear, concise, and actionable response.")

        # Add JSON hint if specified
        if task_context.get('is_json_output', False) or "JSON" in task_context.get('desired_output_format', ''):
            prompt_parts.append("\nRespond ONLY with valid JSON matching the specified format. Do not include explanations or markdown formatting outside the JSON structure.")
            prompt_parts.append("```json")

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for ThinkTool (length: {len(final_prompt)} chars)")
        return final_prompt

    async def collect_insights(self) -> Dict[str, Any]:
        """Collects insights about ThinkTool's own operation and KB status."""
        # ### Phase 2 Plan Ref: 4.15 (Implement collect_insights)
        self.logger.debug("ThinkTool collect_insights called.")
        insights = {
            "agent_name": self.AGENT_NAME, "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kb_fragments": 0, "kb_patterns": 0, "active_directives": 0,
            "last_synthesis_run": self.last_synthesis_run.isoformat() if self.last_synthesis_run else None,
            "last_radar_run": self.last_radar_run.isoformat() if self.last_radar_run else None,
            "last_purge_run": self.last_purge_run.isoformat() if self.last_purge_run else None,
            "key_observations": []
        }
        if not self.session_maker:
             insights["key_observations"].append("Database session unavailable for insights.")
             return insights
        try:
            async with self.session_maker() as session:
                insights["kb_fragments"] = await session.scalar(select(func.count(KnowledgeFragment.id))) or 0
                insights["kb_patterns"] = await session.scalar(select(func.count(LearnedPattern.id))) or 0
                insights["active_directives"] = await session.scalar(select(func.count(StrategicDirective.id)).where(StrategicDirective.status.in_(['pending', 'active']))) or 0
            insights["key_observations"].append("KB and directive counts retrieved.")
        except Exception as e:
            self.logger.error(f"Error collecting DB insights for ThinkTool: {e}", exc_info=True)
            insights["key_observations"].append(f"Error collecting DB insights: {e}")

        return insights

    # --- Helper for reporting errors ---
    async def _report_error(self, error_message: str, task_id: Optional[str] = None):
        """Internal helper to report errors via Orchestrator."""
        if self.orchestrator and hasattr(self.orchestrator, 'report_error'):
            try:
                await self.orchestrator.report_error(self.AGENT_NAME, f"TaskID [{task_id or 'N/A'}]: {error_message}")
            except Exception as report_err:
                self.logger.error(f"Failed to report error to orchestrator: {report_err}")
        else:
            self.logger.warning("Orchestrator unavailable or lacks report_error method.")


# --- End of agents/think_tool.py ---