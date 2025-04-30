def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: object):
    """
    Initializes the ThinkTool agent.

    Args:
        session_maker: An asynchronous session factory for database interactions.
        orchestrator: The main Orchestrator instance.
    """
    config = getattr(orchestrator, 'config', None)
    # ThinkTool manages KB directly, so kb_interface is None for base init
    super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, config=config, session_maker=session_maker, kb_interface=None)

    self.meta_prompt = THINK_TOOL_META_PROMPT

    # Timestamps for periodic tasks in the run loop
    self.last_synthesis_run: Optional[datetime] = None
    self.last_radar_run: Optional[datetime] = None
    self.last_purge_run: Optional[datetime] = None

    self.logger.info("ThinkTool v4.1 initialized.")

    # Load initial learning materials (run async in background)
    asyncio.create_task(self._load_and_synthesize_learning_materials())

# --- Knowledge Loading & Synthesis ---

async def _load_and_synthesize_learning_materials(self):
    """Loads and processes text files from the learning directory, storing insights in KB."""
    learning_dir = 'learning for AI/'
    self.logger.info(f"ThinkTool: Loading learning materials from '{learning_dir}'...")
    processed_files = 0
    learning_files = []
    try:
        if not os.path.isdir(learning_dir):
            self.logger.warning(f"Learning directory '{learning_dir}' not found. Skipping loading.")
            return

        file_pattern = os.path.join(learning_dir, '**', '*.txt')
        learning_files = glob.glob(file_pattern, recursive=True)

        if not learning_files:
            self.logger.info(f"No .txt files found in '{learning_dir}'.")
            return

        self.logger.info(f"Found {len(learning_files)} potential learning files.")

        for file_path in learning_files:
            try:
                self.logger.debug(f"Processing learning file: {file_path}")
                file_content = None
                # Use orchestrator's file reading tool
                if self.orchestrator and hasattr(self.orchestrator, 'use_tool'):
                     file_content_result = await self.orchestrator.use_tool('read_file', {'path': file_path})
                     if file_content_result and file_content_result.get('status') == 'success':
                         file_content = file_content_result.get('content')
                     else:
                         self.logger.warning(f"Could not read file {file_path} via orchestrator tool: {file_content_result.get('message')}")
                         continue # Skip if read fails
                else:
                     self.logger.error("Orchestrator tool access unavailable for reading learning files.")
                     continue # Skip if tool access fails

                if not file_content:
                    self.logger.warning(f"File is empty or could not be read: {file_path}")
                    continue

                # --- Analysis/Synthesis ---
                self.logger.info(f"Analyzing content from: {file_path} using LLM...")
                # --- Structured Thinking Step ---
                analysis_thought = f"""
                Structured Thinking: Analyze Learning Material
                1. Goal: Extract key insights, strategies, applicable agents, type, and relevance from file '{os.path.basename(file_path)}'.
                2. Context: File content (first 4000 chars), ThinkTool meta prompt.
                3. Constraints: Output structured JSON. Call LLM. Store result as KnowledgeFragment.
                4. Information Needed: File content.
                5. Plan: Formulate analysis prompt -> Call _call_llm_with_retry -> Parse JSON -> Call log_knowledge_fragment.
                """
                await self._internal_think(analysis_thought)
                # --- End Structured Thinking Step ---

                analysis_prompt = f"""
                {self.meta_prompt[:500]}...
                **Task:** Analyze text from '{os.path.basename(file_path)}'. Identify key concepts, actionable strategies, relevant mindsets, or code techniques. Determine applicable agents (e.g., ThinkTool, EmailAgent, All). Categorize insight type (e.g., 'strategy', 'mindset', 'technique', 'market_insight', 'competitor_analysis'). Assign relevance score (0.0-1.0).
                **Content (Limit 4000 chars):** {file_content[:4000]}
                **Output Format:** Respond ONLY with valid JSON: {{"source_file": str, "summary": str, "key_concepts": [str], "actionable_strategies": [str], "applicable_agents": [str], "insight_type": str, "relevance_score": float}}
                """
                synthesized_insights_json = await self._call_llm_with_retry(
                    analysis_prompt, temperature=0.5, max_tokens=1024, is_json_output=True
                )

                if synthesized_insights_json:
                    try:
                        insights_data = json.loads(synthesized_insights_json)
                        if not all(k in insights_data for k in ['summary', 'key_concepts', 'applicable_agents', 'insight_type', 'relevance_score']):
                            raise ValueError("LLM response missing required keys.")
                        insights_data['source_file'] = os.path.basename(file_path) # Ensure correct source

                        # Store as Knowledge Fragment
                        await self.log_knowledge_fragment(
                            agent_source="LearningMaterialLoader",
                            data_type=insights_data.get('insight_type', 'learning_material_summary'),
                            content=insights_data,
                            relevance_score=insights_data.get('relevance_score', 0.6),
                            tags=["learning_material", insights_data.get('insight_type', 'general')] + [f"agent:{a.lower()}" for a in insights_data.get('applicable_agents', [])],
                            source_reference=file_path
                        )
                        processed_files += 1
                    except (json.JSONDecodeError, ValueError) as json_error:
                        self.logger.error(f"Error parsing/storing LLM response for {file_path}: {json_error}")
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
    # --- Structured Thinking Step ---
    thinking_process = f"Structured Thinking: Generate Educational Content for '{topic}'. Context: '{context or 'General'}'. Plan: Formulate prompt, call LLM, return cleaned response."
    await self._internal_think(thinking_process)
    # --- End Structured Thinking Step ---
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

# --- Knowledge Base Interface Implementation ---

async def log_knowledge_fragment(self, agent_source: str, data_type: str, content: Union[str, dict], relevance_score: float = 0.5, tags: Optional[List[str]] = None, related_client_id: Optional[int] = None, source_reference: Optional[str] = None) -> Optional[KnowledgeFragment]:
    """Persists a KnowledgeFragment to the database, ensuring content hash for deduplication."""
    if not self.session_maker: self.logger.error("DB session_maker not available in ThinkTool."); return None

    if isinstance(content, dict): content_str = json.dumps(content, sort_keys=True)
    elif isinstance(content, str): content_str = content
    else: self.logger.error(f"Invalid content type for KnowledgeFragment: {type(content)}"); return None

    tags_list = sorted(list(set(tags))) if tags else []
    tags_str = json.dumps(tags_list) if tags_list else None
    content_hash = hashlib.sha256(content_str.encode('utf-8')).hexdigest()
    now_ts = datetime.now(timezone.utc)

    try:
        async with self.session_maker() as session:
            # Check for existing hash
            stmt_check = select(KnowledgeFragment.id).where(KnowledgeFragment.item_hash == content_hash).limit(1)
            result_check = await session.execute(stmt_check)
            existing_id = result_check.scalar_one_or_none()

            if existing_id:
                self.logger.debug(f"KF hash {content_hash[:8]}... exists (ID: {existing_id}). Updating last_accessed_ts.")
                stmt_update = update(KnowledgeFragment).where(KnowledgeFragment.id == existing_id).values(last_accessed_ts=now_ts)
                await session.execute(stmt_update)
                await session.commit()
                return None # Not newly logged
            else:
                fragment = KnowledgeFragment(
                    agent_source=agent_source, timestamp=now_ts, last_accessed_ts=now_ts,
                    data_type=data_type, content=content_str, item_hash=content_hash,
                    relevance_score=relevance_score, tags=tags_str,
                    related_client_id=related_client_id, source_reference=source_reference
                )
                session.add(fragment)
                await session.commit()
                await session.refresh(fragment)
                self.logger.info(f"Logged KnowledgeFragment: ID={fragment.id}, Hash={content_hash[:8]}..., Type={data_type}, Source={agent_source}")
                return fragment
    except SQLAlchemyError as e:
        self.logger.error(f"DB Error logging KnowledgeFragment: {e}", exc_info=True)
        if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"DB Error logging KnowledgeFragment: {e}")
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

            stmt_ids = stmt.with_only_columns(KnowledgeFragment.id).order_by(desc(KnowledgeFragment.relevance_score), desc(KnowledgeFragment.timestamp)).limit(limit)
            result_ids = await session.execute(stmt_ids)
            fragment_ids = result_ids.scalars().all()

            if not fragment_ids: return []

            # Fetch full fragments
            stmt_final = select(KnowledgeFragment).where(KnowledgeFragment.id.in_(fragment_ids)).order_by(desc(KnowledgeFragment.relevance_score), desc(KnowledgeFragment.timestamp))
            result_final = await session.execute(stmt_final)
            fragments = list(result_final.scalars().all())

            # Update last_accessed_ts in background
            if fragment_ids:
                async def update_access_time():
                    try:
                        async with self.session_maker() as update_session:
                            update_stmt = update(KnowledgeFragment).where(KnowledgeFragment.id.in_(fragment_ids)).values(last_accessed_ts=datetime.now(timezone.utc))
                            await update_session.execute(update_stmt)
                            await update_session.commit()
                            self.logger.debug(f"Updated last_accessed_ts for {len(fragment_ids)} fragments.")
                    except Exception as update_err: self.logger.error(f"Failed to update last_accessed_ts: {update_err}")
                asyncio.create_task(update_access_time())

            self.logger.debug(f"KnowledgeBase query returned {len(fragments)} fragments.")
    except SQLAlchemyError as e:
        self.logger.error(f"DB Error querying KnowledgeBase: {e}", exc_info=True)
        if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"DB Error querying KnowledgeBase: {e}")
    except Exception as e: self.logger.error(f"Unexpected error querying KnowledgeBase: {e}", exc_info=True)
    return fragments

async def add_email_composition(self, email_log_id: int, composition_details: Dict[str, Any]):
    """Logs the link between an email and the KB fragments used."""
    if not self.session_maker: self.logger.error("DB session_maker not available."); return None
    try:
        async with self.session_maker() as session:
            stmt_check = select(EmailComposition.id).where(EmailComposition.email_log_id == email_log_id).limit(1)
            exists = await session.execute(stmt_check)
            if exists.scalar_one_or_none():
                self.logger.warning(f"EmailComposition record already exists for EmailLog ID {email_log_id}. Skipping.")
                return None

            composition = EmailComposition(
                email_log_id=email_log_id,
                subject_kf_id=composition_details.get('subject_kf_id'),
                hook_kf_id=composition_details.get('hook_kf_id'),
                body_snippets_kf_ids=composition_details.get('body_snippets_kf_ids'), # Assumes list of ints
                cta_kf_id=composition_details.get('cta_kf_id'),
                timestamp=datetime.now(timezone.utc)
            )
            session.add(composition)
            await session.commit()
            await session.refresh(composition)
            self.logger.info(f"Logged EmailComposition for EmailLog ID {email_log_id}.")
            return composition
    except SQLAlchemyError as e:
        self.logger.error(f"DB Error logging EmailComposition for EmailLog ID {email_log_id}: {e}", exc_info=True)
        if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"DB Error logging EmailComposition: {e}")
        return None
    except Exception as e: self.logger.error(f"Unexpected error logging EmailComposition: {e}", exc_info=True); return None

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
        if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"DB Error getting directives: {e}")
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
            values_to_update = {"status": status}
            if result_summary is not None: values_to_update["result_summary"] = result_summary
            # Add completion timestamp if relevant field exists
            # if status in ['completed', 'failed', 'expired', 'cancelled']: values_to_update["completed_at"] = datetime.now(timezone.utc)

            stmt = update(StrategicDirective).where(StrategicDirective.id == directive_id).values(**values_to_update)
            result = await session.execute(stmt)
            await session.commit()

            if result.rowcount > 0: self.logger.info(f"Updated StrategicDirective ID={directive_id} status to '{status}'."); return True
            else: self.logger.warning(f"Failed to update StrategicDirective ID={directive_id}: Not found."); return False
    except SQLAlchemyError as e:
        self.logger.error(f"DB Error updating directive status for ID={directive_id}: {e}", exc_info=True)
        if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"DB Error updating directive {directive_id}: {e}")
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
            session.add(pattern)
            await session.commit()
            await session.refresh(pattern)
            self.logger.info(f"Logged LearnedPattern: ID={pattern.id}, Confidence={confidence_score:.2f}")
            return pattern
    except SQLAlchemyError as e:
        self.logger.error(f"DB Error logging LearnedPattern: {e}", exc_info=True)
        if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"DB Error logging LearnedPattern: {e}")
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
        if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"DB Error getting patterns: {e}")
    except Exception as e: self.logger.error(f"Unexpected error getting latest patterns: {e}", exc_info=True)
    return patterns

async def purge_old_knowledge(self, days_threshold: int = 30):
    """Deletes KnowledgeFragment records not accessed within the threshold."""
    if not self.session_maker: self.logger.error("DB session_maker not available."); return

    purge_cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_threshold)
    self.logger.info(f"Purging knowledge fragments last accessed before {purge_cutoff_date.isoformat()}...")
    # --- Structured Thinking Step ---
    purge_thought = f"Structured Thinking: Purge Old Knowledge. Plan: Execute DELETE on knowledge_fragments where last_accessed_ts < {purge_cutoff_date}. Log count."
    await self._internal_think(purge_thought)
    # --- End Structured Thinking Step ---
    deleted_count = 0
    try:
        async with self.session_maker() as session:
            stmt = delete(KnowledgeFragment).where(KnowledgeFragment.last_accessed_ts < purge_cutoff_date)
            result = await session.execute(stmt)
            await session.commit()
            deleted_count = result.rowcount
            self.logger.info(f"Successfully purged {deleted_count} old knowledge fragments (last accessed > {days_threshold} days ago).")
            if deleted_count > 0 and self.orchestrator and hasattr(self.orchestrator, 'send_notification'):
                 await self.orchestrator.send_notification("Data Purge Completed", f"Purged {deleted_count} knowledge fragments older than {days_threshold} days.")
    except SQLAlchemyError as e:
        self.logger.error(f"DB Error purging old knowledge fragments: {e}", exc_info=True)
        if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"DB Error purging old knowledge: {e}")
    except Exception as e: self.logger.error(f"Unexpected error purging old knowledge: {e}", exc_info=True)

async def handle_feedback(self, insights_data: Dict[str, Dict[str, Any]]):
    """Processes feedback insights, logs them, and triggers analysis/actions."""
    self.logger.info(f"ThinkTool received feedback insights from {len(insights_data)} agents.")
    # --- Structured Thinking Step ---
    feedback_thought = f"Structured Thinking: Process Agent Feedback. Plan: Log raw feedback -> Format summary -> Call LLM for analysis -> Parse response -> Create directives/schedule critiques/log insights."
    await self._internal_think(feedback_thought)
    # --- End Structured Thinking Step ---

    # Log raw feedback
    for agent_name, agent_feedback in insights_data.items():
        if isinstance(agent_feedback, dict):
            try:
                tags = ["feedback", agent_name.lower()]
                if agent_feedback.get("status") == "error" or agent_feedback.get("errors_encountered_count", 0) > 0: tags.append("error")
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
            analysis_result = json.loads(analysis_json)
            self.logger.info(f"Feedback analysis complete. Summary: {analysis_result.get('analysis_summary', 'N/A')}")

            # Create directives
            async with self.session_maker() as session:
                 for directive_data in analysis_result.get('proposed_directives', []):
                     if all(k in directive_data for k in ['target_agent', 'directive_type', 'content', 'priority']):
                          session.add(StrategicDirective(source="ThinkToolFeedback", timestamp=datetime.now(timezone.utc), status='pending', **directive_data))
                          self.logger.info(f"Generated Directive from Feedback for {directive_data['target_agent']}")
                     else: self.logger.warning(f"Skipping invalid directive data: {directive_data}")
                 await session.commit()

            # Trigger critiques
            for prompt_id in analysis_result.get('prompts_to_critique', []):
                try: agent_name, prompt_key = prompt_id.split('/', 1); asyncio.create_task(self.self_critique_prompt(agent_name, prompt_key, f"Feedback analysis suggested issues: {analysis_result.get('analysis_summary', 'N/A')}")); self.logger.info(f"Scheduled critique for: {prompt_id}")
                except ValueError: self.logger.warning(f"Invalid prompt identifier: {prompt_id}")
                except Exception as critique_err: self.logger.error(f"Error scheduling critique for {prompt_id}: {critique_err}")

            # Log insights
            for frag_data in analysis_result.get('insights_to_log', []):
                 if all(k in frag_data for k in ['data_type', 'content', 'relevance']): await self.log_knowledge_fragment(agent_source="ThinkToolFeedback", data_type=frag_data['data_type'], content=frag_data['content'], tags=frag_data.get('tags', ['feedback_insight']), relevance_score=frag_data['relevance'])
                 else: self.logger.warning(f"Skipping invalid insight data: {frag_data}")

        except json.JSONDecodeError: self.logger.error(f"Failed decode feedback analysis JSON: {analysis_json}")
        except Exception as e: self.logger.error(f"Error processing feedback analysis result: {e}", exc_info=True)
    else:
        self.logger.error("Feedback analysis failed (LLM error).")

# --- Prompt Template Management ---

async def get_prompt(self, agent_name: str, prompt_key: str) -> Optional[str]:
    """Fetches the active prompt content from the database."""
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
        if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"DB Error getting prompt {agent_name}/{prompt_key}: {e}")
        return None
    except Exception as e: self.logger.error(f"Unexpected error getting prompt: {e}", exc_info=True); return None

async def update_prompt(self, agent_name: str, prompt_key: str, new_content: str, author_agent: str = "ThinkTool") -> Optional[PromptTemplate]:
    """Creates a new version of a prompt, deactivates the old one."""
    if not self.session_maker: self.logger.error("DB session_maker not available."); return None
    new_version = 1
    try:
        async with self.session_maker() as session:
            async with session.begin(): # Transaction
                stmt_current = select(PromptTemplate.id, PromptTemplate.version).where(
                    PromptTemplate.agent_name == agent_name, PromptTemplate.prompt_key == prompt_key,
                    PromptTemplate.is_active == True
                ).order_by(desc(PromptTemplate.version)).limit(1).with_for_update()
                result_current = await session.execute(stmt_current)
                current_active_row = result_current.fetchone()

                if current_active_row:
                    current_active_id, current_version = current_active_row
                    new_version = current_version + 1
                    stmt_deactivate = update(PromptTemplate).where(PromptTemplate.id == current_active_id).values(is_active=False)
                    await session.execute(stmt_deactivate)
                    self.logger.info(f"Deactivated prompt v{current_version} for {agent_name}/{prompt_key}")

                new_template = PromptTemplate(
                    agent_name=agent_name, prompt_key=prompt_key, version=new_version,
                    content=new_content, is_active=True, author_agent=author_agent,
                    last_updated=datetime.now(timezone.utc)
                )
                session.add(new_template)
            await session.refresh(new_template)
            self.logger.info(f"Created and activated new prompt v{new_version} for {agent_name}/{prompt_key}")
            return new_template
    except SQLAlchemyError as e:
        self.logger.error(f"DB Error updating prompt {agent_name}/{prompt_key}: {e}", exc_info=True)
        if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"DB Error updating prompt {agent_name}/{prompt_key}: {e}")
        return None
    except Exception as e: self.logger.error(f"Unexpected error updating prompt: {e}", exc_info=True); return None

# --- Enhanced Reflection & Validation ---

async def reflect_on_action(self, context: str, agent_name: str, task_description: str) -> dict:
    """Enhanced reflection incorporating KB context and risk assessment."""
    self.logger.debug(f"Starting reflection for {agent_name} on task: {task_description}")
    # --- Structured Thinking Step ---
    reflect_thought = f"Structured Thinking: Reflect on Action for {agent_name}. Task: {task_description[:50]}... Plan: Fetch KB context -> Format prompt -> Call LLM -> Process response -> Trigger KB updates -> Return result."
    await self._internal_think(reflect_thought)
    # --- End Structured Thinking Step ---
    kb_context = ""
    try:
        active_directives = await self.get_active_directives(target_agent=agent_name)
        relevant_fragments = await self.query_knowledge_base(limit=10, time_window=timedelta(days=7))
        relevant_patterns = await self.get_latest_patterns(limit=5)
        if active_directives: kb_context += "\n\n**Active Directives:**\n" + "\n".join([f"- ID {d.id}: {d.content[:100]}..." for d in active_directives])
        if relevant_fragments: kb_context += "\n\n**Recent Fragments:**\n" + "\n".join([f"- ID {f.id}: {f.content[:80]}..." for f in relevant_fragments])
        if relevant_patterns: kb_context += "\n\n**Relevant Patterns:**\n" + "\n".join([f"- ID {p.id}: {p.pattern_description[:150]}..." for p in relevant_patterns])
    except Exception as e: self.logger.error(f"Error fetching KB context for reflection: {e}"); kb_context = "\n\n**Warning:** Failed KB context retrieval."

    prompt = f"""
    {self.meta_prompt[:500]}...
    **Agent:** {agent_name} | **Task:** {task_description} | **Context:** {context} {kb_context}
    **Analysis:** 1. Data Complete? 2. Compliance OK? 3. Risks (Low/Med/High)? 4. Goal Alignment ($10k/$1B)? 5. Efficiency?
    **Output (JSON):** {{'proceed': bool, 'reason': str, 'risk_level': str, 'compliance_flags': [str], 'next_step': str, 'confidence': float, 'log_fragment': {{...}}?, 'update_directive': {{...}}?}}
    """
    reflection_json = await self._call_llm_with_retry(prompt, temperature=0.3, max_tokens=1000, is_json_output=True)

    if reflection_json:
        try:
            reflection = json.loads(reflection_json)
            reflection.setdefault('proceed', False); reflection.setdefault('reason', 'Analysis failed.'); reflection.setdefault('risk_level', 'Unknown'); reflection.setdefault('compliance_flags', []); reflection.setdefault('next_step', 'Manual review.'); reflection.setdefault('confidence', 0.5)
            self.logger.info(f"Reflection for {agent_name}: Proceed={reflection['proceed']}, Risk={reflection['risk_level']}, Reason={reflection['reason']}")
            # Trigger KB updates
            if 'log_fragment' in reflection and isinstance(reflection['log_fragment'], dict):
                frag_data = reflection['log_fragment']
                if all(k in frag_data for k in ['data_type', 'content', 'relevance']): await self.log_knowledge_fragment(agent_source="ThinkToolReflection", data_type=frag_data['data_type'], content=frag_data['content'], tags=frag_data.get('tags'), relevance_score=frag_data['relevance'])
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
    self.logger.debug(f"Starting validation for {agent_name}'s output.")
    # --- Structured Thinking Step ---
    validate_thought = f"Structured Thinking: Validate Output from {agent_name}. Criteria: {validation_criteria[:50]}... Plan: Fetch patterns -> Format prompt -> Call LLM -> Parse -> Return."
    await self._internal_think(validate_thought)
    # --- End Structured Thinking Step ---
    pattern_context = ""
    try:
        relevant_patterns = await self.get_latest_patterns(limit=5, min_confidence=0.6)
        if relevant_patterns: pattern_context += "\n\n**Relevant Patterns:**\n" + "\n".join([f"- ID {p.id}: {p.pattern_description}" for p in relevant_patterns])
    except Exception as e: self.logger.error(f"Error fetching patterns for validation: {e}"); pattern_context = "\n\n**Warning:** Failed pattern retrieval."

    prompt = f"""
    {self.meta_prompt[:500]}...
    **Agent:** {agent_name} | **Context:** {context or 'N/A'} | **Output:** ```{output_to_validate}``` | **Criteria:** {validation_criteria} {pattern_context}
    **Checks:** 1. Criteria Adherence? 2. Content Valid? 3. Pattern Consistent? 4. Usable? 5. Compliance/Risk OK?
    **Output (JSON):** {{'valid': bool, 'feedback': str, 'suggested_fix': str?}}
    """
    validation_json = await self._call_llm_with_retry(prompt, temperature=0.2, max_tokens=800, is_json_output=True)

    if validation_json:
        try:
            validation = json.loads(validation_json)
            validation.setdefault('valid', False); validation.setdefault('feedback', 'Validation analysis failed.')
            self.logger.info(f"Validation for {agent_name}: Valid={validation['valid']}, Feedback={validation['feedback'][:100]}...")
            return validation
        except json.JSONDecodeError: self.logger.error(f"Failed decode JSON validation: {validation_json}")
        except Exception as e: self.logger.error(f"Error processing validation result: {e}", exc_info=True)
    # Fallback
    return {"valid": False, "feedback": "ThinkTool validation failed."}

# --- New Core Synthesis & Strategy Engines ---

async def synthesize_insights_and_strategize(self):
    """Central cognitive loop for learning, synthesis, and strategy generation."""
    self.logger.info("ThinkTool: Starting synthesis and strategy cycle.")
    # --- Structured Thinking Step ---
    synth_thought = "Structured Thinking: Synthesize & Strategize. Plan: Query KB (fragments, patterns, perf) -> Format -> Call LLM -> Parse -> Store outputs (patterns, directives, opportunities)."
    await self._internal_think(synth_thought)
    # --- End Structured Thinking Step ---
    try:
        # Fetch data
        recent_fragments = await self.query_knowledge_base(limit=200, time_window=timedelta(days=7))
        recent_patterns = await self.get_latest_patterns(limit=20)
        perf_fragments = await self.query_knowledge_base(data_types=['performance_metric', 'profit_summary', 'error_log', 'AgentFeedbackRaw'], limit=50, time_window=timedelta(days=7))
        if not recent_fragments and not perf_fragments: self.logger.warning("ThinkTool Synthesis: Insufficient recent data."); return

        # Format data
        fragments_summary = [{"id": f.id, "type": f.data_type, "src": f.agent_source, "preview": f.content[:80]+"..."} for f in recent_fragments[:20]]
        patterns_summary = [{"id": p.id, "desc": p.pattern_description, "conf": p.confidence_score} for p in recent_patterns]
        perf_summary = [{"id": f.id, "type": f.data_type, "src": f.agent_source, "content": f.content[:150]+"..."} for f in perf_fragments]

        synthesis_prompt = f"""
        {self.meta_prompt[:500]}...
        **Task:** Synthesize insights from recent data. Identify/validate patterns, assess goals, generate directives & opportunities.
        **Data:** Fragments ({len(recent_fragments)}): {json.dumps(fragments_summary)}, Patterns ({len(recent_patterns)}): {json.dumps(patterns_summary)}, Perf ({len(perf_fragments)}): {json.dumps(perf_summary)}
        **Analysis:** 1. Novel Patterns? (desc, ids, conf, impl, tags) 2. Validate Patterns? (id, action, conf?, reason) 3. Goal Progress ($10k/$1B)? 4. Directives? (JSON) 5. Opportunities? (JSON)
        **Output (JSON):** {{"new_patterns": [{{...}}], "pattern_updates": [{{...}}], "goal_assessment": str, "proposed_directives": [{{...}}], "identified_opportunities": [{{...}}]}}
        """
        synthesis_json = await self._call_llm_with_retry(synthesis_prompt, temperature=0.7, max_tokens=2500, is_json_output=True)

        if synthesis_json:
            try:
                synthesis_result = json.loads(synthesis_json)
                self.logger.info("ThinkTool Synthesis cycle completed.")
                # Log new patterns
                for p_data in synthesis_result.get('new_patterns', []):
                     if all(k in p_data for k in ['description', 'confidence', 'implications']): await self.log_learned_pattern(**p_data)
                # TODO: Handle pattern updates
                # Create directives
                async with self.session_maker() as session:
                     for d_data in synthesis_result.get('proposed_directives', []):
                         if all(k in d_data for k in ['target_agent', 'directive_type', 'content', 'priority']): session.add(StrategicDirective(source="ThinkToolSynthesis", timestamp=datetime.now(timezone.utc), status='pending', **d_data)); self.logger.info(f"Generated Directive for {d_data['target_agent']}")
                     await session.commit()
                # Log opportunities
                for o_data in synthesis_result.get('identified_opportunities', []):
                     if 'description' in o_data: await self.log_knowledge_fragment(agent_source="ThinkToolSynthesis", data_type="business_opportunity_signal", content=o_data, tags=o_data.get('tags', ['opportunity']), relevance_score=0.8)
            except json.JSONDecodeError: self.logger.error(f"Failed decode JSON synthesis result: {synthesis_json}")
            except Exception as e: self.logger.error(f"Error processing synthesis result: {e}", exc_info=True)
        else: self.logger.error("Synthesis analysis failed (LLM error).")
    except Exception as e:
        self.logger.error(f"Error during synthesis cycle: {e}", exc_info=True)
        if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"Synthesis cycle failed: {e}")

async def technology_radar(self):
    """Proactively scouts for relevant new tools, APIs, research, and techniques."""
    self.logger.info("ThinkTool: Starting technology radar cycle.")
    # --- Structured Thinking Step ---
    radar_thought = "Structured Thinking: Technology Radar. Plan: Define topics -> Delegate search (BrowsingAgent) -> Analyze results (LLM) -> Log findings/directives (KB)."
    await self._internal_think(radar_thought)
    # --- End Structured Thinking Step ---
    try:
        search_topics = ["new AI UGC video tools", "sales automation AI advances", "open source OSINT tools 2025", "prompt engineering Gemini 1.5", "AI workflow platforms", "autonomous agents research arXiv"]
        search_query = f"Summarize recent developments/tools: {random.choice(search_topics)}"
        search_results_summary = None
        if self.orchestrator and hasattr(self.orchestrator, 'delegate_task'):
             search_task = {"action": "perform_search_and_summarize", "query": search_query, "num_results": 5}
             search_result = await self.orchestrator.delegate_task("browsing", search_task) # Delegate to browsing agent
             if search_result.get("status") == "success": search_results_summary = search_result.get("summary")
             else: self.logger.warning(f"Radar: BrowsingAgent search failed: {search_result.get('message')}")
        else: self.logger.warning("Radar: BrowsingAgent delegation unavailable. Using simulation.") ; search_results_summary = f"Simulated: Found 'VideoSynthX', 'SalesBot Pro'."

        if not search_results_summary: self.logger.info("Radar: No findings from web search."); return

        analysis_prompt = f"""
        {self.meta_prompt[:500]}...
        **Task:** Analyze tech scouting report. Identify novel, high-impact tools/techniques for AI agency. Assess benefits, risks, integration effort. Recommend next steps.
        **Report:** {search_results_summary}
        **Analysis Focus:** Novelty/Impact, Relevance, Benefits, Risks, Integration Effort (Low/Med/High), Recommendation (Log/Directive).
        **Output (JSON):** {{ "analyzed_items": [ {{ "item_name": str, ... "recommendation": str }} ], "overall_assessment": str }}
        """
        analysis_json = await self._call_llm_with_retry(analysis_prompt, temperature=0.4, max_tokens=1500, is_json_output=True)

        if analysis_json:
            try:
                analysis_result = json.loads(analysis_json)
                self.logger.info(f"Radar analysis complete. Found {len(analysis_result.get('analyzed_items', []))} items.")
                for item in analysis_result.get('analyzed_items', []):
                     if not item.get("item_name"): continue
                     await self.log_knowledge_fragment(agent_source="ThinkToolRadar", data_type="new_tool_discovery", content=item, tags=["technology", "scouting", item.get("item_name", "unknown").lower().replace(" ", "_")], relevance_score=0.7)
                     if "directive" in item.get("recommendation", "").lower():
                          async with self.session_maker() as session:
                              session.add(StrategicDirective(
                                  source="ThinkToolRadar", timestamp=datetime.now(timezone.utc), target_agent="Orchestrator",
                                  directive_type="investigate_tool", content=f"Investigate: {item.get('item_name', 'N/A')}. Summary: {item.get('summary', 'N/A')}.",
                                  priority=6, status='pending' ))
                              await session.commit()
                              self.logger.info(f"Generated investigation directive for: {item.get('item_name', 'N/A')}")
            except json.JSONDecodeError: self.logger.error(f"Radar: Failed decode JSON analysis: {analysis_json}")
            except Exception as e: self.logger.error(f"Radar: Error processing analysis result: {e}", exc_info=True)
        else: self.logger.error("Radar: Analysis failed (LLM error).")
    except Exception as e:
        self.logger.error(f"Error during technology radar cycle: {e}", exc_info=True)
        if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"Technology radar cycle failed: {e}")

# --- Self-Improving Prompt Mechanism ---

async def self_critique_prompt(self, agent_name: str, prompt_key: str, feedback_context: str):
    """Attempts to refine a prompt based on negative feedback or failure analysis."""
    self.logger.info(f"Starting self-critique for prompt: {agent_name}/{prompt_key}")
    # --- Structured Thinking Step ---
    critique_thought = f"Structured Thinking: Self-Critique Prompt {agent_name}/{prompt_key}. Plan: Fetch prompt -> Format critique prompt -> Call LLM -> Parse -> Update prompt -> Generate test directive."
    await self._internal_think(critique_thought)
    # --- End Structured Thinking Step ---
    try:
        current_prompt = await self.get_prompt(agent_name, prompt_key)
        if not current_prompt: self.logger.error(f"Critique: Cannot find active prompt {agent_name}/{prompt_key}."); return

        critique_prompt = f"""
        {self.meta_prompt[:500]}...
        **Task:** Critique and rewrite LLM prompt for agent '{agent_name}', key '{prompt_key}'. Improve based on feedback. Use self-instruction.
        **Feedback:** {feedback_context}
        **Current Prompt:** ```\n{current_prompt}\n```
        **Analysis & Rewrite:** 1. Critique weaknesses. 2. Rewrite ONE improved version addressing critique (clarity, instructions, constraints, format, self-instruction).
        **Output (JSON):** {{ "critique": str, "improved_prompt": str }}
        """
        critique_json = await self._call_llm_with_retry(critique_prompt, temperature=0.6, max_tokens=2000, is_json_output=True)

        if critique_json:
            try:
                critique_result = json.loads(critique_json)
                improved_prompt = critique_result.get('improved_prompt')
                critique_text = critique_result.get('critique')
                if improved_prompt:
                    self.logger.info(f"Critique generated improved prompt for {agent_name}/{prompt_key}. Critique: {critique_text}")
                    new_template = await self.update_prompt(agent_name, prompt_key, improved_prompt, author_agent="ThinkToolCritique")
                    if new_template:
                        async with self.session_maker() as session:
                            session.add(StrategicDirective(
                                source="ThinkToolCritique", timestamp=datetime.now(timezone.utc), target_agent=agent_name,
                                directive_type="test_prompt_variation", content=f"Evaluate new prompt v{new_template.version} for key '{prompt_key}'. Compare vs previous. Critique: {critique_text}",
                                priority=7, status='pending' ))
                            await session.commit()
                            self.logger.info(f"Generated directive to test new prompt v{new_template.version} for {agent_name}/{prompt_key}")
                    else: self.logger.error(f"Critique: Failed to save improved prompt {agent_name}/{prompt_key}.")
                else: self.logger.warning(f"Critique for {agent_name}/{prompt_key} did not produce improved prompt.")
            except json.JSONDecodeError: self.logger.error(f"Critique: Failed decode JSON result: {critique_json}")
            except Exception as e: self.logger.error(f"Critique: Error processing result: {e}", exc_info=True)
        else: self.logger.error(f"Critique: Failed get critique/rewrite from LLM for {agent_name}/{prompt_key}.")
    except Exception as e:
        self.logger.error(f"Error during self-critique for {agent_name}/{prompt_key}: {e}", exc_info=True)
        if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"Self-critique failed for {agent_name}/{prompt_key}: {e}")

# --- Agent Run Loop ---
async def run(self):
    """Main loop for ThinkTool: Periodic analysis, strategy, and optimization."""
    if self.status == "running": self.logger.warning("ThinkTool run() called while already running."); return
    self.logger.info("ThinkTool v4.1 starting run loop...")
    self.status = "running"
    synthesis_interval = timedelta(seconds=int(self.config.get("THINKTOOL_SYNTHESIS_INTERVAL_SECONDS", 3600)))
    radar_interval = timedelta(seconds=int(self.config.get("THINKTOOL_RADAR_INTERVAL_SECONDS", 21600)))
    purge_interval = timedelta(seconds=int(self.config.get("DATA_PURGE_INTERVAL_SECONDS", 86400)))

    now = datetime.now(timezone.utc)
    self.last_synthesis_run = now - synthesis_interval
    self.last_radar_run = now - radar_interval
    self.last_purge_run = now - purge_interval

    while self.status == "running":
        try:
            current_time = datetime.now(timezone.utc)
            is_approved = getattr(self.orchestrator, 'approved', False)

            if is_approved:
                if current_time - self.last_synthesis_run >= synthesis_interval:
                    self.logger.info("ThinkTool: Triggering Synthesis & Strategy cycle.")
                    await self.synthesize_insights_and_strategize()
                    self.last_synthesis_run = current_time
                if current_time - self.last_radar_run >= radar_interval:
                    self.logger.info("ThinkTool: Triggering Technology Radar cycle.")
                    await self.technology_radar()
                    self.last_radar_run = current_time
                if current_time - self.last_purge_run >= purge_interval:
                     self.logger.info("ThinkTool: Triggering Data Purge cycle.")
                     await self.purge_old_knowledge()
                     self.last_purge_run = current_time
            else: self.logger.debug("ThinkTool: Orchestrator not approved. Skipping periodic tasks.")

            await asyncio.sleep(60 * 5) # Check every 5 minutes

        except asyncio.CancelledError: self.logger.info("ThinkTool run loop cancelled."); self.status = "stopped"; break
        except Exception as e:
            self.logger.critical(f"ThinkTool: CRITICAL error in run loop: {e}", exc_info=True)
            self.status = "error"
            if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"Critical run loop error: {e}")
            await asyncio.sleep(60 * 15) # Wait longer after critical error

    self.logger.info("ThinkTool run loop finished.")

# --- Abstract Method Implementations ---
async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
    """Handles tasks delegated specifically to ThinkTool."""
    self.status = "working"
    action = task_details.get('action')
    context = task_details.get('context')
    result = {"status": "failure", "message": f"Unknown ThinkTool action: {action}"}
    self.logger.info(f"ThinkTool executing task: {action}")
    # --- Structured Thinking Step ---
    exec_thought = f"Structured Thinking: Execute Task '{action}'. Plan: Route action to appropriate method (reflect, validate, process_feedback, etc.). Execute method. Return result."
    await self._internal_think(exec_thought)
    # --- End Structured Thinking Step ---
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
             if feedback_data: processed = await self.handle_feedback(feedback_data); result = {"status": "success", "processed_feedback_summary": processed.get("summary")}
             else: result = {"status": "failure", "message": "Missing feedback_data for processing."}
        elif action == 'generate_educational_content':
             topic = task_details.get('topic')
             if topic: explanation = await self.generate_educational_content(topic, context); result = {"status": "success" if explanation else "failure", "explanation": explanation}
             else: result = {"status": "failure", "message": "Missing topic for educational content."}
        # Add other actions ThinkTool handles
        else: self.logger.warning(f"Unhandled action in ThinkTool.execute_task: {action}")
    except Exception as e:
         self.logger.error(f"Error executing ThinkTool task '{action}': {e}", exc_info=True)
         result = {"status": "error", "message": f"Exception during task '{action}': {e}"}
    self.status = "idle"
    return result

async def learning_loop(self):
    # Core learning happens in run() periodic tasks
    self.logger.info("ThinkTool learning_loop: Core learning logic is in run() periodic tasks.")
    while self.status == "running": await asyncio.sleep(3600)

async def self_critique(self) -> Dict[str, Any]:
    """Evaluates ThinkTool's own effectiveness."""
    self.logger.info("ThinkTool: Performing self-critique.")
    critique = {"status": "ok", "feedback": "Critique pending analysis."}
    # --- Structured Thinking Step ---
    critique_thought = "Structured Thinking: Self-Critique ThinkTool. Plan: Query DB stats -> Analyze -> Format -> Return."
    await self._internal_think(critique_thought)
    # --- End Structured Thinking Step ---
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
        completed_directives = directive_status.get('completed', 0)
        if failed_directives > completed_directives * 0.2 and completed_directives > 5: feedback += "High directive failure rate observed. " ; critique['status'] = 'warning'
        critique['feedback'] = feedback
    except Exception as e: self.logger.error(f"Error during self-critique: {e}", exc_info=True); critique['status'] = 'error'; critique['feedback'] = f"Critique failed: {e}"
    return critique

async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
    """Constructs prompts for internal LLM calls."""
    self.logger.debug(f"Generating dynamic prompt for ThinkTool task: {task_context.get('task')}")
    # --- Structured Thinking Step ---
    prompt_gen_thought = f"Structured Thinking: Generate Dynamic Prompt for task '{task_context.get('task')}'. Plan: Combine meta-prompt, task context, KB context (if needed), instructions."
    await self._internal_think(prompt_gen_thought)
    # --- End Structured Thinking Step ---
    prompt_parts = [self.meta_prompt]
    prompt_parts.append("\n--- Current Task Context ---")
    prompt_parts.append(f"Task: {task_context.get('task', 'N/A')}")
    if task_context.get('details'): prompt_parts.append(f"Details: {str(task_context['details'])[:500]}...")
    if task_context.get('input_data'): prompt_parts.append(f"Input Data: {str(task_context['input_data'])[:500]}...")
    # Add relevant KB context if needed
    # prompt_parts.append("\n--- Relevant KB Context (If Applicable) ---")
    # ... fetch and format KB data ...
    prompt_parts.append("\n--- Instructions ---")
    prompt_parts.append(f"Perform the task '{task_context.get('task', 'N/A')}' based on the provided context.")
    if task_context.get('desired_output_format'): prompt_parts.append(f"Output Format: {task_context['desired_output_format']}")
    else: prompt_parts.append("Provide a clear and concise response.")
    if task_context.get('is_json_output', False): prompt_parts.append("```json") # Hint for JSON output

    final_prompt = "\n".join(prompt_parts)
    self.logger.debug(f"Generated dynamic prompt for ThinkTool (length: {len(final_prompt)} chars)")
    return final_prompt