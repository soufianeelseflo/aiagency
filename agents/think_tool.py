# FILE: agents/think_tool.py

import asyncio
import logging
import json
import random
import os
import glob
import hashlib # Added for caching
import time # Added for caching TTL (though not directly used in this snippet, good practice)
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, and_, or_, desc, asc, delete # Added delete
from sqlalchemy.dialects.postgresql import JSONB # Assuming PostgreSQL for potential JSONB operators

# Assuming utils/database.py and models.py exist as provided
from models import (
    KnowledgeFragment, StrategicDirective, LearnedPattern, PromptTemplate, Base # Ensure Base is imported if needed elsewhere, though not directly used here
)
# Using adaptable name for flexibility, assuming OpenAI compatible interface
from openai import AsyncOpenAI as AsyncLLMClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Local logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define the updated meta prompt string outside the class or inside __init__ if preferred
THINK_TOOL_META_PROMPT = """
You are the ThinkTool, the central cognitive engine (v4.0) for an autonomous AI agency focused on maximizing profit and efficiency, primarily through UGC (User-Generated Content) services and related ventures.

**Core Cognitive Functions & Principles:**

1.  **Strategic Synthesis & Planning:**
    *   Continuously synthesize insights from all agent activities, performance data (profit, errors, metrics), and external sources (market trends, competitor actions).
    *   Identify novel patterns, validate existing ones, and assess progress towards strategic goals (e.g., $6k/24h, $100M).
    *   Generate actionable, prioritized Strategic Directives for other agents or the Orchestrator to execute.
    *   **Prioritize Competitor Analysis:** Actively analyze competitor strategies, weaknesses, and market gaps to identify high-value, real-world problems the agency can solve uniquely or more effectively.

2.  **Knowledge & Learning Management:**
    *   Maintain and curate the agency's Knowledge Base (fragments, patterns, directives, prompts).
    *   Drive the learning loop, identifying effective strategies and discarding ineffective ones based on empirical results.

3.  **Meta-Prompting & Agent Optimization:**
    *   **Generate and Refine Prompts:** Actively critique, refine, and generate new prompts for all agents (including yourself) based on performance feedback, observed issues, or strategic shifts. Aim for clarity, effectiveness, and alignment with agent goals.
    *   Identify opportunities to improve agent workflows and efficiency.

4.  **Solution Design & Validation:**
    *   Reflect critically on proposed actions, assessing data completeness, compliance, risk, goal alignment, and efficiency.
    *   **Critically Analyze External Data:** Scrutinize external information (e.g., transcripts, search results, OSINT data) against internal knowledge (KB, patterns) and core objectives before integration or action.
    *   Validate agent outputs against criteria and learned patterns.
    *   **Value Simplicity & Convenience:** Favor solutions that are simple, robust, and convenient for end-users and internal operations. Avoid unnecessary complexity.

5.  **Opportunity Scouting & Innovation:**
    *   Proactively scan the technological horizon (new tools, APIs, research) for relevant advancements.
    *   Identify potential new business opportunities, service offerings, or significant efficiency improvements.
    *   **Consider Unconventional Solutions:** Analyze and consider unconventional ("grey area") tactics or approaches when rigorous analysis indicates they are strategically sound, ethically justifiable within defined constraints, and offer a significant advantage towards achieving core objectives. Assess risks carefully.

6.  **Resource Optimization:** Inherently optimize for computational resources, API costs, and agent time.

**Overall Goal:** Drive the agency towards exponential growth in profit and capability through intelligent analysis, strategic direction, and continuous adaptation.
"""

class ThinkTool:
    """
    The central cognitive engine (v4.0) for the autonomous AI agency.
    Responsible for learning, adaptation, strategic planning, knowledge management,
    and self-improvement by integrating data across all agents and operations.
    """
    def __init__(self, session_maker: callable, config: object, orchestrator: object):
        """
        Initializes the ThinkTool agent.

        Args:
            session_maker: An asynchronous session factory for database interactions.
            config: The application's configuration object.
            orchestrator: The main Orchestrator instance for inter-agent communication
                          and accessing shared resources like LLM clients.
        """
        self.session_maker = session_maker
        self.config = config
        self.orchestrator = orchestrator
        # Use the detailed meta prompt defined above
        self.meta_prompt = THINK_TOOL_META_PROMPT
        self.logger = logger # Use the module-level logger

        # Initialize knowledge base for synthesized learning materials
        self.knowledge_base: Dict[str, Any] = {}

        # In-memory cache for active prompts { (agent_name, prompt_key): prompt_content }
        self.prompt_cache: Dict[tuple[str, str], str] = {}

        # Timestamps for periodic tasks in the run loop
        self.last_synthesis_run: Optional[datetime] = None
        self.last_radar_run: Optional[datetime] = None

        self.logger.info("ThinkTool v4.0 initialized.")

        # Load initial learning materials
        # Run synchronously during init for simplicity, consider async/background task later
        self._load_and_synthesize_learning_materials()

    # --- Knowledge Loading & Synthesis ---

    async def _load_and_synthesize_learning_materials(self):
        """Loads and processes text files from the learning directory."""
        learning_dir = 'learning for AI/'
        self.logger.info(f"ThinkTool: Loading learning materials from '{learning_dir}'...")
        processed_files = 0
        try:
            # Ensure the directory exists (optional, depends on desired error handling)
            if not os.path.isdir(learning_dir):
                self.logger.warning(f"Learning directory '{learning_dir}' not found. Skipping loading.")
                return

            # Find all .txt files recursively (adjust pattern if needed)
            # Using glob.glob for simplicity, consider os.walk for more control
            file_pattern = os.path.join(learning_dir, '**', '*.txt') # Search recursively
            learning_files = glob.glob(file_pattern, recursive=True)

            if not learning_files:
                self.logger.info(f"No .txt files found in '{learning_dir}'.")
                return

            self.logger.info(f"Found {len(learning_files)} potential learning files.")

            for file_path in learning_files:
                try:
                    self.logger.debug(f"Processing learning file: {file_path}")
                    # Simulate reading the file content (replace with actual tool call later)
                    try:
                        # In a real scenario, this would likely involve an async call:
                        # file_content_result = await self.orchestrator.use_tool('read_file', {'path': file_path})
                        # file_content = file_content_result.get('content') if file_content_result else None
                        # For now, simulate reading directly or assume content is available
                        # This requires the agent process to have file system access
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            file_content = f.read()
                        self.logger.info(f"Successfully read content from: {file_path}")

                        if not file_content:
                            self.logger.warning(f"File is empty or could not be read: {file_path}")
                            continue # Skip to next file

                        # --- Analysis/Synthesis Simulation ---
                        self.logger.info(f"Analyzing content from: {file_path} using LLM...")

                        # --- Formulate LLM Prompt for Analysis ---
                        # Note: Ensure file_content is available from the read operation above
                        analysis_prompt = f"""
                        Analyze the following text content extracted from '{os.path.basename(file_path)}'.
                        Identify key concepts, actionable strategies, relevant mindsets, or code techniques.
                        Determine which agent(s) (e.g., ThinkTool, EmailAgent, All) or system components these insights apply to.
                        Categorize the insight type (e.g., 'strategy', 'mindset', 'technique', 'market_insight').
                        Return the analysis as a JSON object with keys: 'source', 'summary', 'key_concepts' (list), 'applicable_agents' (list), 'type'.

                        Context from ThinkTool Meta Prompt (consider relevant parts):
                        {self.meta_prompt[:500]}... # Include relevant parts or summary

                        Content:
                        {file_content[:4000]} # Limit content length for prompt
                        """
                        # TODO: Refine prompt, potentially include specific goals or context

                        # --- Call LLM for Analysis (Conceptual / Simulated) ---
                        # In a real implementation, use the centralized LLM call method:
                        synthesized_insights_json = await self._call_llm_with_retry(
                            analysis_prompt,
                            temperature=0.5,
                            max_tokens=1024, # Adjust as needed
                            is_json_output=True
                        )
                        logger.info(f"LLM analysis complete for {file_path}.")

                        # --- Process LLM Response ---
                        synthesized_insights = None # Initialize
                        if synthesized_insights_json:
                            try:
                                synthesized_insights = json.loads(synthesized_insights_json)
                                # Basic validation
                                required_keys = ['source', 'summary', 'key_concepts', 'applicable_agents', 'type']
                                if not all(k in synthesized_insights for k in required_keys):
                                    raise ValueError("LLM response missing required keys.")
                                # Ensure source matches (optional but good practice)
                                if synthesized_insights.get('source') != file_path:
                                     self.logger.warning(f"LLM response source '{synthesized_insights.get('source')}' mismatch for file '{file_path}'. Overwriting.")
                                     synthesized_insights['source'] = file_path # Correct the source

                                synthesized_insights['processed_at'] = datetime.now(timezone.utc).isoformat() # Add timestamp
                                self.logger.info(f"Successfully parsed LLM insights for {file_path}")

                            except (json.JSONDecodeError, ValueError) as json_error:
                                self.logger.error(f"Error parsing LLM response for {file_path}: {json_error}")
                                # Handle parsing error - maybe log raw response, skip storage, or log error fragment
                                synthesized_insights = None # Ensure it's None if parsing failed
                        else:
                             self.logger.error(f"LLM analysis returned no content for {file_path}.")

                        # --- Knowledge Storage ---
                        # Store insights in the knowledge base using filename as key
                        kb_key = os.path.basename(file_path)
                        self.knowledge_base[kb_key] = synthesized_insights
                        self.logger.info(f"Stored insights from {kb_key} into knowledge base.")

                        processed_files += 1

                    except FileNotFoundError:
                         self.logger.error(f"Learning file not found: {file_path}")
                    except IOError as io_err:
                         self.logger.error(f"IOError reading learning file {file_path}: {io_err}")
                    except Exception as e:
                        # Catch errors during analysis or storage specifically
                        self.logger.error(f"Error processing learning file {file_path} after reading: {e}", exc_info=True)
                        # Optionally log this error as a KnowledgeFragment
                        # await self.log_knowledge_fragment(...)

                except Exception as file_error: # Catch errors for the outer try block (line 129)
                    self.logger.error(f"General error processing learning file {file_path}: {file_error}", exc_info=True)
                    # Optionally log this error as a KnowledgeFragment
                    # await self.log_knowledge_fragment(...)

            self.logger.info(f"Finished processing learning materials. Processed {processed_files}/{len(learning_files)} files.")
            # Log the keys of the loaded knowledge for verification
            self.logger.info(f"Knowledge base now contains keys: {list(self.knowledge_base.keys())}")

        except Exception as e:
            self.logger.error(f"Critical error during loading/synthesizing learning materials setup: {e}", exc_info=True)
            # Report critical error
            # await self.orchestrator.report_error("ThinkTool", f"Learning material loading failed: {e}")

    # --- Standardized LLM Interaction ---

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=4, max=30), retry=retry_if_exception_type(Exception))
    async def _call_llm_with_retry(self, prompt: str, model_preference: Optional[List[str]] = None, temperature: float = 0.5, max_tokens: int = 1024, is_json_output: bool = False) -> Optional[str]:
        """
        Centralized method for calling LLMs via the Orchestrator.
        Handles client selection, retries, error reporting, and JSON formatting.

        Args:
            prompt: The prompt string for the LLM.
            model_preference: Optional list of preferred model names/keys.
            temperature: The sampling temperature for the LLM.
            max_tokens: The maximum number of tokens to generate.
            is_json_output: Whether to request JSON output format from the LLM.

        Returns:
            The LLM response content as a string, or None if all retries fail.
        """
        llm_client: Optional[AsyncLLMClient] = None
        model_name: Optional[str] = None
        api_key_identifier: str = "unknown_key" # For logging/reporting issues

        try:
            # --- Caching Logic: Check Cache First ---
            # Create a unique key based on relevant parameters
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            # Determine model name early for cache key consistency
            default_model = getattr(self.config, 'OPENROUTER_MODELS', {}).get('think', "google/gemini-pro") # Fallback model
            # TODO: Refine model selection based on model_preference if provided and stable
            model_name_for_cache = default_model # Use the determined model for the cache key

            cache_key_parts = [
                "llm_call",
                prompt_hash,
                model_name_for_cache, # Include model name
                str(temperature),
                str(max_tokens),
                str(is_json_output),
            ]
            cache_key = ":".join(cache_key_parts)
            cache_ttl = 3600 # Default 1 hour TTL for LLM calls

            # Check cache first
            if hasattr(self.orchestrator, 'get_from_cache'):
                cached_result = self.orchestrator.get_from_cache(cache_key)
                if cached_result is not None:
                    self.logger.debug(f"LLM call cache hit for key: {cache_key[:20]}...{cache_key[-20:]}")
                    # TODO: Optionally track cache hit metrics
                    return cached_result # Return cached value
                else:
                    self.logger.debug(f"LLM call cache miss for key: {cache_key[:20]}...{cache_key[-20:]}")
            else:
                self.logger.warning("Orchestrator does not have 'get_from_cache' method. Skipping cache check.")
            # --- End Cache Check ---

            # 1. Get available clients from Orchestrator
            # TODO: Enhance orchestrator to return clients with associated model info/keys
            # For now, assume orchestrator provides a list of AsyncOpenAI compatible clients
            available_clients = await self.orchestrator.get_available_openrouter_clients()
            if not available_clients:
                self.logger.error("ThinkTool: No available LLM clients from Orchestrator.")
                return None

            # TODO: Implement smarter client/model selection based on preference, availability, cost etc.
            llm_client = random.choice(available_clients)
            api_key_identifier = getattr(llm_client, 'api_key', 'unknown_key')[-6:] # Log last 6 chars

            # 2. Determine model name (using 'think' model as default for ThinkTool)
            # Model name already determined above for cache key consistency
            model_name = model_name_for_cache

            # 3. Prepare request arguments
            response_format = {"type": "json_object"} if is_json_output else None
            messages = [{"role": "user", "content": prompt}]

            self.logger.debug(f"ThinkTool LLM Call (Cache Miss): Model={model_name}, Temp={temperature}, MaxTokens={max_tokens}, JSON={is_json_output}, Key=...{api_key_identifier}")

            # 4. Make the API call
            response = await llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                timeout=60 # Generous timeout for complex tasks
            )

            content = response.choices[0].message.content.strip()

            # --- Token Tracking & Cost Estimation (Placeholder) ---
            input_tokens_est = len(prompt) // 4 # Rough estimate
            output_tokens = 0
            try:
                # Attempt to get actual usage if available (depends on LLM client library)
                if response.usage and response.usage.completion_tokens:
                    output_tokens = response.usage.completion_tokens
                    # Use prompt_tokens if available too
                    if response.usage.prompt_tokens:
                        input_tokens_est = response.usage.prompt_tokens
                else:
                     output_tokens = len(content) // 4 # Rough estimate if usage not available
            except AttributeError:
                 output_tokens = len(content) // 4 # Fallback estimate

            total_tokens_est = input_tokens_est + output_tokens
            # Placeholder cost - refine later based on actual model pricing
            # Example: $1 per 1 million tokens = $0.000001 per token
            estimated_cost = total_tokens_est * 0.000001
            self.logger.debug(f"LLM Call Est. Tokens: ~{total_tokens_est} (In: {input_tokens_est}, Out: {output_tokens}). Est. Cost: ${estimated_cost:.6f}")
            # --- End Token Tracking ---

            # --- Report Expense ---
            if estimated_cost > 0 and hasattr(self.orchestrator, 'report_expense'):
                try:
                    # Use await if report_expense is async, otherwise remove await
                    await self.orchestrator.report_expense(
                        agent_name="ThinkTool",
                        amount=estimated_cost,
                        category="LLM",
                        description=f"LLM call ({model_name or 'unknown_model'}). Estimated tokens: {total_tokens_est}."
                    )
                except Exception as report_err:
                    self.logger.error(f"Failed to report LLM expense: {report_err}", exc_info=True)
            # --- End Report Expense ---

            # --- Caching Logic: Add to Cache on Success ---
            # Add successful result to cache
            if content and hasattr(self.orchestrator, 'add_to_cache'):
                self.orchestrator.add_to_cache(cache_key, content, ttl_seconds=cache_ttl)
                self.logger.debug(f"Added LLM result to cache for key: {cache_key[:20]}...{cache_key[-20:]}")
            elif not hasattr(self.orchestrator, 'add_to_cache'):
                self.logger.warning("Orchestrator does not have 'add_to_cache' method. Skipping caching result.")
            # --- End Add to Cache ---


            # TODO: Track token usage via Orchestrator/BudgetAgent if response includes usage data
            # usage = response.usage
            # await self.orchestrator.track_llm_usage(model_name, usage.prompt_tokens, usage.completion_tokens)

            return content

        except Exception as e:
            error_str = str(e).lower()
            issue_type = "llm_error"
            if "rate limit" in error_str or "quota" in error_str: issue_type = "rate_limit"
            elif "authentication" in error_str: issue_type = "auth_error"
            elif "timeout" in error_str: issue_type = "timeout_error"

            self.logger.warning(f"ThinkTool LLM call failed (attempt): Model={model_name}, Key=...{api_key_identifier}, ErrorType={issue_type}, Error={e}")

            # Report issue back to orchestrator for potential client disabling/rotation
            if llm_client:
                 # Assuming orchestrator has a method like this
                 await self.orchestrator.report_client_issue(api_key_identifier, issue_type)

            raise # Reraise exception for tenacity retry logic

        # Should not be reached if retry logic works, but as a fallback
        self.logger.error(f"ThinkTool LLM call failed after all retries: Model={model_name}, Key=...{api_key_identifier}")
        return None

    async def generate_educational_content(self, topic: str, context: Optional[str] = None) -> Optional[str]:
        """
        Generates a concise, user-friendly explanation of a given topic,
        suitable for the User Education Mechanism.

        Args:
            topic (str): The topic to explain (e.g., "Grey Area Strategy", "LLM Caching").
            context (Optional[str]): Additional context about why the explanation is needed.

        Returns:
            Optional[str]: The generated explanation text, or None on failure.
        """
        self.logger.info(f"ThinkTool: Generating educational content for topic: {topic}")

        # Construct LLM prompt
        prompt = f"""
        {self.meta_prompt}
        **Task:** Generate a concise and easy-to-understand explanation for the user about the following topic. Assume the user is intelligent but not necessarily a technical expert. Avoid jargon where possible or explain it simply. Focus on the 'why' and the relevance to the agency's goals.

        **Topic:** {topic}

        **Context (Why this is relevant now):** {context or 'General understanding'}

        **Output:** Provide ONLY the explanation text, suitable for direct display to the user. Start directly with the explanation, without introductory phrases like "Here's an explanation...".
        """

        # Call LLM using the centralized helper
        explanation = await self._call_llm_with_retry(
            prompt,
            temperature=0.6, # Slightly more creative/explanatory
            max_tokens=500, # Keep explanations reasonably concise
            is_json_output=False
        )

        if explanation:
            self.logger.info(f"Successfully generated educational content for topic: {topic}")
            # Optional: Log this explanation as a KnowledgeFragment?
            # await self.log_knowledge_fragment(...)
            return explanation
        else:
            self.logger.error(f"Failed to generate educational content for topic: {topic} (LLM error).")
            return None
    # --- Knowledge Base Interface Implementation ---

    async def log_knowledge_fragment(self, agent_source: str, data_type: str, content: Union[str, dict], relevance_score: float = 0.5, tags: Optional[List[str]] = None, related_client_id: Optional[int] = None, source_reference: Optional[str] = None) -> Optional[KnowledgeFragment]:
        """Persists a KnowledgeFragment to the database."""
        if isinstance(content, dict):
            content_str = json.dumps(content)
        elif isinstance(content, str):
            content_str = content
        else:
            self.logger.error(f"Invalid content type for KnowledgeFragment: {type(content)}")
            return None

        tags_str = json.dumps(tags) if tags else None

        fragment = KnowledgeFragment(
            agent_source=agent_source,
            timestamp=datetime.now(timezone.utc),
            data_type=data_type,
            content=content_str,
            relevance_score=relevance_score,
            tags=tags_str,
            related_client_id=related_client_id,
            source_reference=source_reference
        )
        try:
            async with self.session_maker() as session:
                session.add(fragment)
                await session.commit()
                await session.refresh(fragment)
                self.logger.info(f"Logged KnowledgeFragment: ID={fragment.id}, Type={data_type}, Source={agent_source}")
                return fragment
        except Exception as e:
            self.logger.error(f"Failed to log KnowledgeFragment: {e}", exc_info=True)
            await self.orchestrator.report_error("ThinkTool", f"DB Error logging KnowledgeFragment: {e}")
            return None

    async def query_knowledge_base(self, data_types: Optional[List[str]] = None, tags: Optional[List[str]] = None, min_relevance: float = 0.0, time_window: Optional[timedelta] = None, limit: int = 100, related_client_id: Optional[int] = None) -> List[KnowledgeFragment]:
        """Queries the knowledge_fragments table."""
        try:
            async with self.session_maker() as session:
                stmt = select(KnowledgeFragment)

                if data_types:
                    stmt = stmt.where(KnowledgeFragment.data_type.in_(data_types))
                if min_relevance > 0.0:
                    stmt = stmt.where(KnowledgeFragment.relevance_score >= min_relevance)
                if related_client_id is not None:
                    stmt = stmt.where(KnowledgeFragment.related_client_id == related_client_id)
                if time_window:
                    start_time = datetime.now(timezone.utc) - time_window
                    stmt = stmt.where(KnowledgeFragment.timestamp >= start_time)

                # Tag filtering (requires JSON support or careful string matching)
                # Example for JSON array containment (PostgreSQL specific using JSONB ideally)
                # if tags:
                #     # Assuming tags are stored as JSON array string: '["tag1", "tag2"]'
                #     # This might need adjustment based on exact DB/dialect
                #     tag_conditions = [KnowledgeFragment.tags.contains(tag) for tag in tags] # Simplified example
                #     stmt = stmt.where(or_(*tag_conditions))
                # Simpler text matching (less precise):
                if tags:
                     tag_conditions = [KnowledgeFragment.tags.like(f'%"{tag}"%') for tag in tags] # Assumes JSON list format
                     # or tag_conditions = [KnowledgeFragment.tags.like(f'%{tag}%') for tag in tags] # If comma-separated
                     stmt = stmt.where(or_(*tag_conditions))


                stmt = stmt.order_by(desc(KnowledgeFragment.timestamp)).limit(limit)

                result = await session.execute(stmt)
                fragments = result.scalars().all()
                self.logger.debug(f"KnowledgeBase query returned {len(fragments)} fragments.")
                return list(fragments)
        except Exception as e:
            self.logger.error(f"Failed to query KnowledgeBase: {e}", exc_info=True)
            await self.orchestrator.report_error("ThinkTool", f"DB Error querying KnowledgeBase: {e}")
            return []

    async def get_active_directives(self, target_agent: str = 'All') -> List[StrategicDirective]:
        """Fetches active or pending StrategicDirectives."""
        try:
            async with self.session_maker() as session:
                stmt = select(StrategicDirective).where(
                    or_(StrategicDirective.status == 'pending', StrategicDirective.status == 'active')
                )
                if target_agent != 'All':
                    # Allow targeting specific agent OR 'All'
                    stmt = stmt.where(
                         or_(StrategicDirective.target_agent == target_agent, StrategicDirective.target_agent == 'All')
                    )

                stmt = stmt.order_by(desc(StrategicDirective.priority), asc(StrategicDirective.timestamp))

                result = await session.execute(stmt)
                directives = result.scalars().all()
                self.logger.debug(f"Fetched {len(directives)} active/pending directives for target '{target_agent}'.")
                return list(directives)
        except Exception as e:
            self.logger.error(f"Failed to get active directives: {e}", exc_info=True)
            await self.orchestrator.report_error("ThinkTool", f"DB Error getting directives: {e}")
            return []

    async def update_directive_status(self, directive_id: int, status: str, result_summary: Optional[str] = None) -> bool:
        """Updates the status of a StrategicDirective."""
        try:
            async with self.session_maker() as session:
                values_to_update = {"status": status}
                # Optionally set result summary
                if result_summary is not None:
                    values_to_update["result_summary"] = result_summary
                # Optionally set completion timestamp (if status indicates completion)
                if status in ['completed', 'failed', 'expired']:
                     # Assuming a 'completed_at' field exists or using 'last_updated' logic
                     # values_to_update["completed_at"] = datetime.now(timezone.utc)
                     pass # Add logic if a specific completion timestamp field exists

                stmt = update(StrategicDirective).where(StrategicDirective.id == directive_id).values(**values_to_update)
                result = await session.execute(stmt)
                await session.commit()

                if result.rowcount > 0:
                    self.logger.info(f"Updated StrategicDirective ID={directive_id} status to '{status}'.")
                    return True
                else:
                    self.logger.warning(f"Failed to update StrategicDirective ID={directive_id}: Not found.")
                    return False
        except Exception as e:
            self.logger.error(f"Failed to update directive status for ID={directive_id}: {e}", exc_info=True)
            await self.orchestrator.report_error("ThinkTool", f"DB Error updating directive {directive_id}: {e}")
            return False

    async def log_learned_pattern(self, pattern_description: str, supporting_fragment_ids: List[int], confidence_score: float, implications: str, tags: Optional[List[str]] = None) -> Optional[LearnedPattern]:
        """Creates and persists a LearnedPattern record."""
        fragment_ids_str = json.dumps(supporting_fragment_ids)
        tags_str = json.dumps(tags) if tags else None

        pattern = LearnedPattern(
            timestamp=datetime.now(timezone.utc),
            pattern_description=pattern_description,
            supporting_fragment_ids=fragment_ids_str,
            confidence_score=confidence_score,
            implications=implications,
            tags=tags_str
        )
        try:
            async with self.session_maker() as session:
                session.add(pattern)
                await session.commit()
                await session.refresh(pattern)
                self.logger.info(f"Logged LearnedPattern: ID={pattern.id}, Confidence={confidence_score:.2f}")
                return pattern
        except Exception as e:
            self.logger.error(f"Failed to log LearnedPattern: {e}", exc_info=True)
            await self.orchestrator.report_error("ThinkTool", f"DB Error logging LearnedPattern: {e}")
            return None

    async def get_latest_patterns(self, tags: Optional[List[str]] = None, min_confidence: float = 0.7, limit: int = 10) -> List[LearnedPattern]:
        """Retrieves recent, high-confidence LearnedPattern records."""
        try:
            async with self.session_maker() as session:
                stmt = select(LearnedPattern).where(LearnedPattern.confidence_score >= min_confidence)

                # Tag filtering (similar logic as query_knowledge_base)
                if tags:
                     tag_conditions = [LearnedPattern.tags.like(f'%"{tag}"%') for tag in tags] # Assumes JSON list format
                     stmt = stmt.where(or_(*tag_conditions))

                stmt = stmt.order_by(desc(LearnedPattern.timestamp)).limit(limit)

                result = await session.execute(stmt)
                patterns = result.scalars().all()
                self.logger.debug(f"Fetched {len(patterns)} learned patterns (min_confidence={min_confidence}).")
                return list(patterns)
        except Exception as e:
            self.logger.error(f"Failed to get latest patterns: {e}", exc_info=True)
            await self.orchestrator.report_error("ThinkTool", f"DB Error getting patterns: {e}")
            return []

    async def purge_old_knowledge(self, days_threshold: int = 30):
        """
        Deletes KnowledgeFragment records older than the specified threshold (default 30 days).
        """
        # Check if db_manager is available via session_maker (assuming session_maker implies db_manager access)
        if not self.session_maker:
            self.logger.error("Database session maker not available. Cannot purge old knowledge.")
            return

        purge_cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_threshold)
        self.logger.info(f"Purging knowledge fragments older than {purge_cutoff_date.isoformat()}...")

        try:
            async with self.session_maker() as session: # Use the session_maker directly
                # Construct the delete statement
                stmt = delete(KnowledgeFragment).where(KnowledgeFragment.timestamp < purge_cutoff_date)

                # Execute the delete statement
                result = await session.execute(stmt)
                await session.commit()

                deleted_count = result.rowcount
                self.logger.info(f"Successfully purged {deleted_count} old knowledge fragments.")

        except Exception as e:
            # Rollback is implicitly handled by the context manager on exception
            self.logger.error(f"Error purging old knowledge fragments: {e}", exc_info=True)
            # Optionally report error to orchestrator
            await self.orchestrator.report_error("ThinkTool", f"DB Error purging old knowledge: {e}")


    async def handle_feedback(self, insights_data: Dict[str, Dict[str, Any]]):
        """
        Processes feedback insights collected from various agents.
        Currently logs each agent's feedback as a KnowledgeFragment.
        """
        self.logger.info(f"ThinkTool received feedback insights from {len(insights_data)} agents.")
        # self.logger.debug(f"Raw feedback data: {insights_data}") # Keep if needed for debugging

        for agent_name, agent_feedback in insights_data.items():
            if not isinstance(agent_feedback, dict):
                self.logger.warning(f"Received non-dict feedback from {agent_name}. Skipping logging for this agent. Data: {agent_feedback}")
                continue

            # Create a summary or use the raw feedback dictionary as content
            # For now, let's log the dictionary directly as JSON string content
            try:
                content_str = json.dumps(agent_feedback, indent=2)
                # Use the existing log_knowledge_fragment method signature
                # log_knowledge_fragment(self, agent_source: str, data_type: str, content: Union[str, dict], relevance_score: float = 0.5, tags: Optional[List[str]] = None, related_client_id: Optional[int] = None, source_reference: Optional[str] = None)
                fragment_type = "AgentFeedback"
                source = agent_name # Use agent_name as the agent_source
                tags = ["feedback", agent_name.lower()] # Add relevant tags

                # Add specific tags based on feedback content if possible
                if agent_feedback.get("status") == "error" or agent_feedback.get("errors_encountered"):
                    tags.append("error")
                if agent_feedback.get("status") == "placeholder":
                     tags.append("placeholder_insight")


                await self.log_knowledge_fragment(
                    agent_source=source,
                    data_type=fragment_type,
                    content=content_str, # Pass the JSON string
                    tags=tags,
                    # Pass metadata as source_reference or adapt log_knowledge_fragment if needed
                    source_reference=f"Original structure stored in content JSON"
                    # metadata={"original_structure": agent_feedback} # Store original dict if needed - log_knowledge_fragment doesn't support this directly, store in content or source_reference
                )
                self.logger.debug(f"Logged feedback from {agent_name} as KnowledgeFragment.")

            except TypeError as e:
                 self.logger.error(f"Error serializing feedback from {agent_name} to JSON: {e}. Feedback: {agent_feedback}")
            except Exception as e:
                self.logger.error(f"Error logging feedback fragment for {agent_name}: {e}", exc_info=True)

        # TODO: Add more sophisticated analysis here later.
        # - Identify patterns across agent feedback.
        # - Trigger strategic adjustments based on errors or performance metrics.
        # - Update internal models or parameters.
        self.logger.info("Feedback processing complete (logged as fragments).")
    # --- Prompt Template Management ---

    async def get_prompt(self, agent_name: str, prompt_key: str) -> Optional[str]:
        """Fetches the active prompt content, using an in-memory cache."""
        cache_key = (agent_name, prompt_key)
        if cache_key in self.prompt_cache:
            self.logger.debug(f"Prompt cache hit for {agent_name}/{prompt_key}")
            return self.prompt_cache[cache_key]

        self.logger.debug(f"Prompt cache miss for {agent_name}/{prompt_key}. Querying DB.")
        try:
            async with self.session_maker() as session:
                stmt = select(PromptTemplate.content).where(
                    PromptTemplate.agent_name == agent_name,
                    PromptTemplate.prompt_key == prompt_key,
                    PromptTemplate.is_active == True
                ).order_by(desc(PromptTemplate.version)).limit(1) # Ensure we get the latest active if multiple somehow exist

                result = await session.execute(stmt)
                prompt_content = result.scalar_one_or_none()

                if prompt_content:
                    self.prompt_cache[cache_key] = prompt_content
                    self.logger.info(f"Fetched and cached active prompt for {agent_name}/{prompt_key}")
                    return prompt_content
                else:
                    self.logger.warning(f"No active prompt found for {agent_name}/{prompt_key}")
                    return None
        except Exception as e:
            self.logger.error(f"Failed to get prompt for {agent_name}/{prompt_key}: {e}", exc_info=True)
            await self.orchestrator.report_error("ThinkTool", f"DB Error getting prompt {agent_name}/{prompt_key}: {e}")
            return None

    async def update_prompt(self, agent_name: str, prompt_key: str, new_content: str, author_agent: str = "ThinkTool") -> Optional[PromptTemplate]:
        """Creates a new version of a prompt, deactivates the old one, and updates the cache."""
        cache_key = (agent_name, prompt_key)
        new_version = 1
        try:
            async with self.session_maker() as session:
                # 1. Find current active version and deactivate it
                subquery = select(PromptTemplate.id, PromptTemplate.version).where(
                    PromptTemplate.agent_name == agent_name,
                    PromptTemplate.prompt_key == prompt_key,
                    PromptTemplate.is_active == True
                ).order_by(desc(PromptTemplate.version)).limit(1).subquery()

                # Get the ID and version number of the current active prompt
                current_active = await session.execute(select(subquery.c.id, subquery.c.version))
                current_active_row = current_active.fetchone()

                if current_active_row:
                    current_active_id, current_version = current_active_row
                    new_version = current_version + 1
                    # Deactivate the old version
                    deactivate_stmt = update(PromptTemplate).where(
                        PromptTemplate.id == current_active_id
                    ).values(is_active=False)
                    await session.execute(deactivate_stmt)
                    self.logger.info(f"Deactivated prompt version {current_version} for {agent_name}/{prompt_key}")

                # 2. Create the new active version
                new_template = PromptTemplate(
                    agent_name=agent_name,
                    prompt_key=prompt_key,
                    version=new_version,
                    content=new_content,
                    is_active=True,
                    last_updated=datetime.now(timezone.utc)
                    # Consider adding an 'author' field if needed
                )
                session.add(new_template)
                await session.commit()
                await session.refresh(new_template)

                # 3. Update cache
                self.prompt_cache[cache_key] = new_content
                self.logger.info(f"Created and cached new active prompt version {new_version} for {agent_name}/{prompt_key}")

                # TODO: Notify Orchestrator or relevant agent about the prompt update?
                # await self.orchestrator.notify_prompt_update(agent_name, prompt_key)

                return new_template

        except Exception as e:
            self.logger.error(f"Failed to update prompt for {agent_name}/{prompt_key}: {e}", exc_info=True)
            await self.orchestrator.report_error("ThinkTool", f"DB Error updating prompt {agent_name}/{prompt_key}: {e}")
            # Attempt to clear potentially inconsistent cache entry on error
            if cache_key in self.prompt_cache:
                del self.prompt_cache[cache_key]
            return None

    # --- Enhanced Reflection & Validation ---

    async def reflect_on_action(self, context: str, agent_name: str, task_description: str) -> dict:
        """Enhanced reflection incorporating KB context and risk assessment."""
        self.logger.debug(f"Starting reflection for {agent_name} on task: {task_description}")
        kb_context = ""
        try:
            # 1. Fetch relevant context from Knowledge Base
            active_directives = await self.get_active_directives(target_agent=agent_name)
            # TODO: Derive relevant tags/data_types from context/task_description for better querying
            relevant_fragments = await self.query_knowledge_base(limit=10, time_window=timedelta(days=7)) # Recent fragments
            relevant_patterns = await self.get_latest_patterns(limit=5) # High-confidence patterns

            # 2. Format KB context for the prompt
            if active_directives:
                kb_context += "\n\n**Active Strategic Directives:**\n"
                for d in active_directives:
                    kb_context += f"- ID {d.id} (Pri {d.priority}): {d.directive_type} - {d.content[:100]}...\n"
            if relevant_fragments:
                kb_context += "\n**Recent Relevant Knowledge Fragments:**\n"
                for f in relevant_fragments:
                    content_preview = f.content[:100].replace('\n', ' ') + ('...' if len(f.content) > 100 else '')
                    kb_context += f"- ID {f.id} ({f.data_type} by {f.agent_source}): {content_preview}\n"
            if relevant_patterns:
                kb_context += "\n**Relevant Learned Patterns:**\n"
                for p in relevant_patterns:
                    kb_context += f"- ID {p.id} (Conf {p.confidence_score:.2f}): {p.pattern_description[:150]}...\n"

        except Exception as e:
            self.logger.error(f"Error fetching KB context for reflection: {e}", exc_info=True)
            kb_context = "\n\n**Warning:** Failed to retrieve full Knowledge Base context."

        # 3. Construct the LLM prompt
        prompt = f"""
        {self.meta_prompt}
        **Agent:** {agent_name}
        **Task:** {task_description}
        **Provided Context:** {context}
        {kb_context}

        **Critical Analysis Required:**
        1.  **Data Completeness & Context:** Is all necessary info present in provided context AND KB?
        2.  **Compliance Check:** Review against known rules (USA/Morocco focus). Flag potential issues.
        3.  **Risk Assessment:** Identify potential negative outcomes (financial, legal, reputational). Estimate likelihood (Low/Medium/High).
        4.  **Goal Alignment:** Does this action directly contribute to the $6k/24h or $100M goals? How? Consider KB patterns/directives.
        5.  **Efficiency:** Is this the most cost-effective way (API calls, time)? Are there relevant KB insights about efficiency?

        **Output (JSON):**
        - 'proceed' (bool): Recommendation to continue/halt.
        - 'reason' (str): Concise justification based on analysis above, referencing KB items if relevant (e.g., "Proceed, aligns with Directive #123").
        - 'risk_level' (str): Estimated risk (Low/Medium/High/Critical).
        - 'compliance_flags' (list): Specific compliance concerns, if any.
        - 'next_step' (str): Concrete, optimized next action or alternative, potentially referencing KB items.
        - 'confidence' (float): Confidence in this reflection (0.0-1.0).
        - 'log_fragment' (dict, optional): If analysis generated a new insight, provide data for log_knowledge_fragment {{'data_type': str, 'content': str|dict, 'tags': list[str], 'relevance': float}}.
        - 'update_directive' (dict, optional): If action completes/fails a directive, provide {{'directive_id': int, 'status': str, 'result_summary': str}}.
        """

        # 4. Call LLM
        reflection_json = await self._call_llm_with_retry(prompt, temperature=0.3, max_tokens=1000, is_json_output=True)

        # 5. Process response and potentially trigger KB updates
        if reflection_json:
            try:
                reflection = json.loads(reflection_json)
                # Add default values if keys are missing
                reflection.setdefault('proceed', False)
                reflection.setdefault('reason', 'Analysis failed or incomplete.')
                reflection.setdefault('risk_level', 'Unknown')
                reflection.setdefault('compliance_flags', [])
                reflection.setdefault('next_step', 'Manual review required.')
                reflection.setdefault('confidence', 0.5)

                self.logger.info(f"ThinkTool Reflection for {agent_name} ({task_description}): Proceed={reflection['proceed']}, Risk={reflection['risk_level']}, Reason={reflection['reason']}")

                # Trigger KB updates based on reflection output
                if 'log_fragment' in reflection and isinstance(reflection['log_fragment'], dict):
                    frag_data = reflection['log_fragment']
                    await self.log_knowledge_fragment(
                        agent_source="ThinkToolReflection", # Or agent_name?
                        data_type=frag_data.get('data_type', 'reflection_insight'),
                        content=frag_data.get('content', 'Missing content'),
                        tags=frag_data.get('tags'),
                        relevance_score=frag_data.get('relevance', 0.6)
                    )
                if 'update_directive' in reflection and isinstance(reflection['update_directive'], dict):
                    directive_data = reflection['update_directive']
                    await self.update_directive_status(
                        directive_id=directive_data.get('directive_id'),
                        status=directive_data.get('status', 'failed'), # Default to failed if status missing
                        result_summary=directive_data.get('result_summary')
                    )

                return reflection
            except json.JSONDecodeError:
                self.logger.error(f"ThinkTool: Failed to decode JSON reflection: {reflection_json}")
            except Exception as e:
                 self.logger.error(f"ThinkTool: Error processing reflection result: {e}", exc_info=True)

        # Fallback if LLM call fails or JSON is invalid
        return {
            "proceed": False, # Default to caution
            "reason": "ThinkTool analysis failed. LLM call error or invalid JSON.",
            "risk_level": "Critical",
            "compliance_flags": ["Analysis Failure"],
            "next_step": "Halt task and report error to Orchestrator.",
            "confidence": 0.0
        }

    async def validate_output(self, output_to_validate: str, validation_criteria: str, agent_name: str, context: str = None) -> dict:
        """Enhanced validation incorporating learned patterns."""
        self.logger.debug(f"Starting validation for {agent_name}'s output.")
        pattern_context = ""
        try:
            # 1. Fetch relevant learned patterns
            # TODO: Derive relevant tags from context/criteria
            relevant_patterns = await self.get_latest_patterns(limit=5, min_confidence=0.6) # Get reasonably confident patterns

            # 2. Format pattern context for the prompt
            if relevant_patterns:
                pattern_context += "\n\n**Relevant Learned Patterns (Consider these during validation):**\n"
                for p in relevant_patterns:
                    pattern_context += f"- ID {p.id} (Conf {p.confidence_score:.2f}): {p.pattern_description} -> Implications: {p.implications}\n"

        except Exception as e:
            self.logger.error(f"Error fetching patterns for validation: {e}", exc_info=True)
            pattern_context = "\n\n**Warning:** Failed to retrieve learned patterns for context."

        # 3. Construct the LLM prompt
        prompt = f"""
        {self.meta_prompt}
        **Agent:** {agent_name}
        **Task Context:** {context or 'Not provided'}
        **Output to Validate:** ```{output_to_validate}```
        **Validation Criteria:** {validation_criteria}
        {pattern_context}

        **Validation Checks:**
        1.  **Criteria Adherence:** Does the output meet ALL specified criteria?
        2.  **Content Validity:** Is the content plausible, coherent, and factually sound?
        3.  **Pattern Consistency:** Does the output align with or contradict known successful patterns (see above)? Explain any discrepancies.
        4.  **Usability:** Can the requesting agent directly use this output?
        5.  **Compliance/Risk:** Does the output introduce any obvious compliance issues or risks?

        **Output (JSON):**
        - 'valid' (bool): Overall validity based on criteria AND patterns.
        - 'feedback' (str): Specific reasons for pass/fail, referencing criteria and pattern IDs if relevant.
        - 'suggested_fix' (str, optional): If minor fixable issues, provide the corrected output.
        """

        # 4. Call LLM
        validation_json = await self._call_llm_with_retry(prompt, temperature=0.2, max_tokens=800, is_json_output=True)

        # 5. Process response
        if validation_json:
            try:
                validation = json.loads(validation_json)
                validation.setdefault('valid', False)
                validation.setdefault('feedback', 'Validation analysis failed.')
                self.logger.info(f"ThinkTool Validation for {agent_name}: Valid={validation['valid']}, Feedback={validation['feedback'][:100]}...")
                return validation
            except json.JSONDecodeError:
                 self.logger.error(f"ThinkTool: Failed to decode JSON validation: {validation_json}")
            except Exception as e:
                 self.logger.error(f"ThinkTool: Error processing validation result: {e}", exc_info=True)

        # Fallback
        return {"valid": False, "feedback": "ThinkTool validation failed (LLM error or invalid JSON)."}

    async def process_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes feedback collected from various agents to identify issues,
        successes, and generate actionable insights or directives.

        Args:
            feedback_data (Dict[str, Any]): A dictionary where keys are agent names
                                           and values are the feedback/insights collected
                                           from that agent (structure depends on agent's
                                           `collect_insights` method).

        Returns:
            Dict[str, Any]: A dictionary containing validated feedback or newly
                            generated directives intended for specific agents.
                            (Structure TBD based on Orchestrator's expectation).
                            For now, can return an empty dict or summary.
        """
        self.logger.info(f"ThinkTool: Processing feedback received from {len(feedback_data)} agents.")
        processed_results = {} # Placeholder for return value

        if not feedback_data:
            self.logger.warning("ThinkTool: Received empty feedback data. Skipping processing.")
            return processed_results

        # 1. Format feedback data for LLM analysis prompt
        feedback_summary = json.dumps(feedback_data, indent=2, default=str, ensure_ascii=False) # Pretty print JSON
        # Limit summary size if too large
        max_summary_len = 4000
        if len(feedback_summary) > max_summary_len:
             feedback_summary = feedback_summary[:max_summary_len] + "\n... (feedback truncated)"

        # 2. Construct LLM prompt for feedback analysis
        analysis_prompt = f"""
        {self.meta_prompt}
        **Task:** Analyze the consolidated feedback collected from various agency agents. Identify critical issues, significant successes, emerging trends, and potential areas for optimization or prompt refinement.

        **Consolidated Agent Feedback:**
        ```json
        {feedback_summary}
        ```

        **Analysis & Action Generation:**
        1.  **Synthesize Key Findings:** Summarize the most important points from the feedback (both positive and negative).
        2.  **Identify Critical Issues:** Pinpoint any recurring errors, performance bottlenecks, or strategic misalignments reported by agents.
        3.  **Recognize Successes:** Note any particularly effective strategies or high-performance areas mentioned.
        4.  **Suggest Actions:** Based on the analysis, propose concrete actions:
            *   **Directives:** Generate 1-2 high-priority `StrategicDirective` objects (JSON format matching DB model: {{ "target_agent": str, "directive_type": str, "content": str, "priority": int }}) to address critical issues or test improvements.
            *   **Prompt Critiques:** Identify specific agent prompts (agent_name/prompt_key) that may need refinement based on the feedback. List them as strings: ["AgentName/PromptKey1", "AgentName/PromptKey2"].
            *   **Knowledge Logging:** Suggest 1-2 key insights derived from the feedback to be logged as `KnowledgeFragment` objects (JSON format: {{'data_type': str, 'content': str|dict, 'tags': list[str], 'relevance': float}}).

        **Output (JSON):**
        {{
          "analysis_summary": str,
          "critical_issues_found": list[str],
          "key_successes_noted": list[str],
          "proposed_directives": [ {{ "target_agent": str, "directive_type": str, "content": str, "priority": int }} ],
          "prompts_to_critique": list[str], // e.g., ["EmailAgent/email_draft", "VoiceSalesAgent/intent_analysis"]
          "insights_to_log": [ {{'data_type': str, 'content': str|dict, 'tags': list[str], 'relevance': float}} ]
        }}
        """

        # 3. Call LLM for analysis
        self.logger.debug("Calling LLM for feedback analysis...")
        analysis_json = await self._call_llm_with_retry(analysis_prompt, temperature=0.6, max_tokens=2000, is_json_output=True)

        # 4. Process LLM response and trigger actions
        if analysis_json:
            try:
                analysis_result = json.loads(analysis_json)
                self.logger.info(f"Feedback analysis complete. Summary: {analysis_result.get('analysis_summary', 'N/A')}")

                # Create new directives
                async with self.session_maker() as session:
                     for directive_data in analysis_result.get('proposed_directives', []):
                         directive = StrategicDirective(
                             source="ThinkToolFeedback",
                             timestamp=datetime.now(timezone.utc),
                             target_agent=directive_data.get('target_agent', 'Orchestrator'),
                             directive_type=directive_data.get('directive_type', 'general_task'),
                             content=directive_data.get('content', 'N/A'),
                             priority=directive_data.get('priority', 5),
                             status='pending'
                         )
                         session.add(directive)
                         self.logger.info(f"Generated Directive from Feedback for {directive.target_agent}: {directive.directive_type}")
                     await session.commit()

                # Trigger prompt critiques (run asynchronously)
                for prompt_identifier in analysis_result.get('prompts_to_critique', []):
                    try:
                        agent_name, prompt_key = prompt_identifier.split('/', 1)
                        # Create task to run critique without blocking feedback loop
                        asyncio.create_task(self.self_critique_prompt(agent_name, prompt_key, f"Feedback analysis suggested issues: {analysis_result.get('analysis_summary', 'N/A')}"))
                        self.logger.info(f"Scheduled self-critique for prompt: {prompt_identifier}")
                    except ValueError:
                        self.logger.warning(f"Invalid format for prompt identifier in feedback analysis: {prompt_identifier}")

                # Log new knowledge fragments
                for frag_data in analysis_result.get('insights_to_log', []):
                    await self.log_knowledge_fragment(
                        agent_source="ThinkToolFeedback",
                        data_type=frag_data.get('data_type', 'feedback_insight'),
                        content=frag_data.get('content', 'Missing content'),
                        tags=frag_data.get('tags', ['feedback']),
                        relevance_score=frag_data.get('relevance', 0.7)
                    )

                # TODO: Determine what structure Orchestrator expects back
                # For now, just return a summary or the raw analysis
                processed_results = {"summary": analysis_result.get('analysis_summary', 'Processing complete.')}

            except json.JSONDecodeError:
                self.logger.error(f"ThinkTool: Failed to decode JSON feedback analysis result: {analysis_json}")
                processed_results = {"error": "Failed to decode LLM response."}
            except Exception as e:
                self.logger.error(f"ThinkTool: Error processing feedback analysis result: {e}", exc_info=True)
                processed_results = {"error": f"Internal error processing feedback: {e}"}
        else:
            self.logger.error("ThinkTool: Feedback analysis failed (LLM error).")
            processed_results = {"error": "LLM call failed during feedback analysis."}

        return processed_results

    # --- New Core Synthesis & Strategy Engines ---

    async def synthesize_insights_and_strategize(self):
        """Central cognitive loop for learning, synthesis, and strategy generation."""
        self.logger.info("ThinkTool: Starting synthesis and strategy cycle.")
        try:
            # 1. Fetch diverse data from Knowledge Base
            recent_fragments = await self.query_knowledge_base(limit=200, time_window=timedelta(days=7)) # Broad sample
            recent_patterns = await self.get_latest_patterns(limit=20)
            # Fetch performance indicators (assuming they are logged as fragments)
            perf_fragments = await self.query_knowledge_base(data_types=['performance_metric', 'profit_summary', 'error_log'], limit=50, time_window=timedelta(days=7))

            if not recent_fragments and not perf_fragments:
                self.logger.warning("ThinkTool Synthesis: Insufficient recent data to perform analysis.")
                return

            # 2. Format data for the LLM prompt (provide summaries/samples)
            fragments_summary = [{"id": f.id, "type": f.data_type, "src": f.agent_source, "preview": f.content[:80]+"..."} for f in recent_fragments[:20]] # Sample
            patterns_summary = [{"id": p.id, "desc": p.pattern_description, "conf": p.confidence_score} for p in recent_patterns]
            perf_summary = [{"id": f.id, "type": f.data_type, "src": f.agent_source, "content": f.content} for f in perf_fragments]

            # 3. Construct the synthesis prompt
            synthesis_prompt = f"""
            {self.meta_prompt}
            **Task:** Synthesize insights from recent agency activity and performance data. Identify new patterns, validate existing ones, assess goal progress, and generate actionable strategies (directives) and potential opportunities.

            **Data Streams (Samples/Summaries):**
            - Recent Knowledge Fragments ({len(recent_fragments)} total): {json.dumps(fragments_summary, indent=2)}
            - Existing Learned Patterns ({len(recent_patterns)} total): {json.dumps(patterns_summary, indent=2)}
            - Recent Performance/Error Fragments ({len(perf_fragments)} total): {json.dumps(perf_summary, indent=2)}

            **Analysis & Strategy Generation:**
            1.  **Identify Novel Patterns:** Analyze fragments for new correlations, trends, or potential causal links (e.g., "OSINT from source X leads to Y% higher email open rate", "Error type Z occurs after action W"). Describe the pattern and supporting fragment IDs (approximate if needed). Assign a confidence score (0.0-1.0).
            2.  **Validate/Refute Existing Patterns:** Assess existing patterns against recent fragments. Suggest updates (e.g., adjust confidence) or mark as obsolete if contradicted.
            3.  **Assess Goal Progress:** Evaluate performance fragments against strategic goals ($6k/24h, $100M). Identify key drivers or blockers.
            4.  **Generate Strategic Directives:** Propose 1-3 concrete, actionable directives based on findings. Specify target agent ('All', 'EmailAgent', etc.), type (e.g., 'test_strategy', 'prioritize_target', 'fix_issue'), priority (1-10, 10=urgent), and clear content/instructions.
            5.  **Identify Opportunities:** Based on all data, suggest 1-2 potential new business opportunities, service offerings, or major efficiency improvements.

            **Output (JSON):**
            {{
                "new_patterns": [ {{ "description": str, "supporting_fragment_ids": list[int], "confidence": float, "implications": str, "tags": list[str] }} ],
                "pattern_updates": [ {{ "pattern_id": int, "action": "update_confidence|mark_obsolete", "new_confidence": float, "reason": str }} ],
                "goal_assessment": str,
                "proposed_directives": [ {{ "target_agent": str, "directive_type": str, "content": str, "priority": int }} ],
                "identified_opportunities": [ {{ "description": str, "potential_value": str, "tags": list[str] }} ]
            }}
            """

            # 4. Call LLM for synthesis
            synthesis_json = await self._call_llm_with_retry(synthesis_prompt, temperature=0.7, max_tokens=2500, is_json_output=True)

            # 5. Process results and update DB
            if synthesis_json:
                try:
                    synthesis_result = json.loads(synthesis_json)
                    self.logger.info("ThinkTool Synthesis cycle completed.")

                    # Log new patterns
                    for pattern_data in synthesis_result.get('new_patterns', []):
                        await self.log_learned_pattern(
                            pattern_description=pattern_data.get('description', 'N/A'),
                            supporting_fragment_ids=pattern_data.get('supporting_fragment_ids', []),
                            confidence_score=pattern_data.get('confidence', 0.5),
                            implications=pattern_data.get('implications', 'N/A'),
                            tags=pattern_data.get('tags')
                        )

                    # TODO: Handle pattern updates (might require adding status/confidence update logic to LearnedPattern model/methods)
                    # for update_data in synthesis_result.get('pattern_updates', []):
                    #     await self.update_learned_pattern(**update_data)

                    # Create new directives
                    async with self.session_maker() as session:
                         for directive_data in synthesis_result.get('proposed_directives', []):
                             directive = StrategicDirective(
                                 source="ThinkToolSynthesis",
                                 timestamp=datetime.now(timezone.utc),
                                 target_agent=directive_data.get('target_agent', 'Orchestrator'),
                                 directive_type=directive_data.get('directive_type', 'general_task'),
                                 content=directive_data.get('content', 'N/A'),
                                 priority=directive_data.get('priority', 5),
                                 status='pending'
                             )
                             session.add(directive)
                             self.logger.info(f"Generated Strategic Directive for {directive.target_agent}: {directive.directive_type}")
                         await session.commit()


                    # Log opportunities
                    for opp_data in synthesis_result.get('identified_opportunities', []):
                        await self.log_knowledge_fragment(
                            agent_source="ThinkToolSynthesis",
                            data_type="business_opportunity_signal",
                            content={
                                "description": opp_data.get('description', 'N/A'),
                                "potential_value": opp_data.get('potential_value', 'N/A')
                            },
                            tags=opp_data.get('tags', ['opportunity']),
                            relevance_score=0.8 # High relevance
                        )

                except json.JSONDecodeError:
                    self.logger.error(f"ThinkTool: Failed to decode JSON synthesis result: {synthesis_json}")
                except Exception as e:
                    self.logger.error(f"ThinkTool: Error processing synthesis result: {e}", exc_info=True)
            else:
                self.logger.error("ThinkTool: Synthesis analysis failed (LLM error).")

        except Exception as e:
            self.logger.error(f"ThinkTool: Error during synthesis cycle: {e}", exc_info=True)
            await self.orchestrator.report_error("ThinkTool", f"Synthesis cycle failed: {e}")

    async def technology_radar(self):
        """Proactively scouts for relevant new tools, APIs, research, and techniques."""
        self.logger.info("ThinkTool: Starting technology radar cycle.")
        try:
            # 1. Define search topics relevant to agency operations
            search_topics = [
                "new AI tools for UGC video generation",
                "advances in sales automation AI",
                "open source OSINT tools 2025",
                "prompt engineering techniques for Gemini models",
                "AI workflow automation platforms",
                "latest research papers on autonomous agents arXiv"
            ]
            search_query = f"Summarize recent developments and tools related to: {random.choice(search_topics)}"

            # 2. Use BrowsingAgent via Orchestrator for web search
            # Assuming orchestrator has a way to delegate tasks like this
            browsing_agent = self.orchestrator.agents.get('browsing')
            if not browsing_agent:
                self.logger.warning("ThinkTool Radar: BrowsingAgent not available.")
                return

            # Assuming BrowsingAgent has a method like 'perform_search_and_summarize'
            # search_results_summary = await browsing_agent.perform_search_and_summarize(search_query, num_results=5)
            # --- Placeholder ---
            # Simulate getting search results for now
            self.logger.warning("ThinkTool Radar: Web search functionality needs integration with BrowsingAgent.")
            search_results_summary = f"Simulated search results for '{search_query}'. Found potential tools: 'VideoSynthX', 'SalesBot Pro'. Research paper: 'AgentCoord Strategies'."
            # --- End Placeholder ---

            if not search_results_summary:
                self.logger.info("ThinkTool Radar: No significant findings from web search.")
                return

            # 3. Analyze search results with LLM
            analysis_prompt = f"""
            {self.meta_prompt}
            **Task:** Analyze the following technology scouting report. Identify novel, high-impact tools, techniques, or research relevant to our AI agency's goals (profit, efficiency, UGC, sales). Assess potential benefits, risks, and integration effort.

            **Scouting Report Summary:**
            {search_results_summary}

            **Analysis Focus:**
            - **Novelty & Impact:** Is this genuinely new? How significantly could it impact our operations or goals?
            - **Relevance:** How directly applicable is it to UGC, sales, OSINT, or agent coordination?
            - **Benefits:** What are the specific potential advantages (e.g., cost savings, speed, quality improvement, new capabilities)?
            - **Risks:** What are the potential downsides (e.g., cost, complexity, reliability, security)?
            - **Integration Effort:** Estimate effort/time required (Low/Medium/High).
            - **Recommendation:** Should we investigate further? If yes, propose a specific next step (e.g., directive for deeper research, sandbox trial).

            **Output (JSON):**
            {{
              "analyzed_items": [
                {{
                  "item_name": str, // Tool, technique, paper name
                  "summary": str,
                  "novelty_impact": str,
                  "relevance": str,
                  "benefits": list[str],
                  "risks": list[str],
                  "integration_effort": str, // Low/Medium/High
                  "recommendation": str // e.g., "Log as fragment", "Generate investigation directive"
                }}
              ],
              "overall_assessment": str // Brief summary of the most promising findings
            }}
            """
            analysis_json = await self._call_llm_with_retry(analysis_prompt, temperature=0.4, max_tokens=1500, is_json_output=True)

            # 4. Process analysis and log findings/directives
            if analysis_json:
                try:
                    analysis_result = json.loads(analysis_json)
                    self.logger.info(f"ThinkTool Radar analysis complete. Found {len(analysis_result.get('analyzed_items', []))} items.")

                    for item in analysis_result.get('analyzed_items', []):
                        # Log finding as KnowledgeFragment
                        await self.log_knowledge_fragment(
                            agent_source="ThinkToolRadar",
                            data_type="new_tool_discovery",
                            content=item, # Store the full analysis dict
                            tags=["technology", "scouting", item.get("item_name", "unknown").lower()],
                            relevance_score=0.7 # Moderately high relevance
                        )

                        # Generate directive if recommended
                        if "directive" in item.get("recommendation", "").lower():
                             async with self.session_maker() as session:
                                directive = StrategicDirective(
                                    source="ThinkToolRadar",
                                    timestamp=datetime.now(timezone.utc),
                                    target_agent="Orchestrator", # Or specific agent like BrowsingAgent/SandboxAgent
                                    directive_type="investigate_tool",
                                    content=f"Investigate tool/technique: {item.get('item_name', 'N/A')}. Radar Summary: {item.get('summary', 'N/A')}. Recommendation: {item.get('recommendation', 'N/A')}",
                                    priority=6,
                                    status='pending'
                                )
                                session.add(directive)
                                await session.commit()
                                self.logger.info(f"Generated investigation directive for: {item.get('item_name', 'N/A')}")

                except json.JSONDecodeError:
                    self.logger.error(f"ThinkTool Radar: Failed to decode JSON analysis result: {analysis_json}")
                except Exception as e:
                    self.logger.error(f"ThinkTool Radar: Error processing analysis result: {e}", exc_info=True)
            else:
                 self.logger.error("ThinkTool Radar: Analysis failed (LLM error).")

        except Exception as e:
            self.logger.error(f"ThinkTool: Error during technology radar cycle: {e}", exc_info=True)
            await self.orchestrator.report_error("ThinkTool", f"Technology radar cycle failed: {e}")

    # --- Self-Improving Prompt Mechanism ---

    async def self_critique_prompt(self, agent_name: str, prompt_key: str, feedback_context: str):
        """Attempts to refine a prompt based on negative feedback or failure analysis."""
        self.logger.info(f"Starting self-critique for prompt: {agent_name}/{prompt_key}")
        try:
            # 1. Fetch current active prompt
            current_prompt = await self.get_prompt(agent_name, prompt_key)
            if not current_prompt:
                self.logger.error(f"Self-Critique: Cannot find active prompt for {agent_name}/{prompt_key}.")
                return

            # 2. Construct meta-prompt for critique and rewrite
            # TODO: Fetch actual error logs/feedback fragments based on feedback_context
            feedback_summary = f"Observed issues/feedback context: {feedback_context}"

            critique_prompt = f"""
            {self.meta_prompt}
            **Task:** Critique and rewrite the LLM prompt for agent '{agent_name}', task key '{prompt_key}'.
            **Goal:** Improve prompt effectiveness based on observed issues, aiming for better success rate, clarity, or desired output characteristics. Use self-instruction principles.

            **Observed Issues/Feedback:**
            {feedback_summary}

            **Current Prompt:**
            ```
            {current_prompt}
            ```

            **Analysis & Rewrite:**
            1.  **Critique:** Analyze the current prompt. What specific weaknesses might be causing the observed issues? (e.g., ambiguity, lack of constraints, poor examples, wrong persona).
            2.  **Rewrite:** Generate ONE improved version of the prompt addressing the critique. Focus on clarity, specific instructions, constraints, and desired output format. Incorporate self-instruction ("Think step-by-step...", "Ensure the output format is...", etc.) where appropriate.

            **Output (JSON):**
            {{
              "critique": str, // Brief critique of the original prompt
              "improved_prompt": str // The complete rewritten prompt text
            }}
            """

            # 3. Call LLM for critique and rewrite
            critique_json = await self._call_llm_with_retry(critique_prompt, temperature=0.6, max_tokens=2000, is_json_output=True)

            # 4. Process response and update prompt
            if critique_json:
                try:
                    critique_result = json.loads(critique_json)
                    improved_prompt = critique_result.get('improved_prompt')
                    critique_text = critique_result.get('critique')

                    if improved_prompt:
                        self.logger.info(f"Self-Critique generated improved prompt for {agent_name}/{prompt_key}. Critique: {critique_text}")

                        # Update the prompt in the database (creates new version)
                        new_template = await self.update_prompt(agent_name, prompt_key, improved_prompt, author_agent="ThinkToolCritique")

                        if new_template:
                            # 5. Generate directive to test the new prompt
                            async with self.session_maker() as session:
                                directive = StrategicDirective(
                                    source="ThinkToolCritique",
                                    timestamp=datetime.now(timezone.utc),
                                    target_agent=agent_name, # Target the agent whose prompt was updated
                                    directive_type="test_prompt_variation",
                                    content=f"Evaluate performance of new prompt version {new_template.version} for key '{prompt_key}'. Compare against previous version. Critique was: {critique_text}",
                                    priority=7, # High priority to test improvements
                                    status='pending'
                                )
                                session.add(directive)
                                await session.commit()
                                self.logger.info(f"Generated directive to test new prompt version {new_template.version} for {agent_name}/{prompt_key}")
                        else:
                             self.logger.error(f"Self-Critique: Failed to save the improved prompt for {agent_name}/{prompt_key} to DB.")

                    else:
                        self.logger.warning(f"Self-Critique for {agent_name}/{prompt_key} did not produce an improved prompt.")

                except json.JSONDecodeError:
                    self.logger.error(f"Self-Critique: Failed to decode JSON result: {critique_json}")
                except Exception as e:
                    self.logger.error(f"Self-Critique: Error processing result: {e}", exc_info=True)
            else:
                self.logger.error(f"Self-Critique: Failed to get critique/rewrite from LLM for {agent_name}/{prompt_key}.")

        except Exception as e:
            self.logger.error(f"ThinkTool: Error during self-critique for {agent_name}/{prompt_key}: {e}", exc_info=True)
            await self.orchestrator.report_error("ThinkTool", f"Self-critique failed for {agent_name}/{prompt_key}: {e}")


    # --- Agent Run Loop ---

    async def run(self):
        """Main loop for ThinkTool: Periodic analysis, strategy, and optimization."""
        self.logger.info("ThinkTool v4.0 starting run loop...")
        # Define intervals (adjust as needed)
        synthesis_interval = timedelta(hours=2) # Synthesize every 2 hours
        radar_interval = timedelta(hours=12) # Tech radar twice a day
        # Prompt critique might be triggered by events/directives rather than interval

        while True:
            try:
                current_time = datetime.now(timezone.utc)
                # Check if orchestrator has approved full operation
                # Use getattr to avoid error if orchestrator doesn't have 'approved' yet
                is_approved = getattr(self.orchestrator, 'approved', False)

                if is_approved:
                    # Run Synthesis Cycle
                    if self.last_synthesis_run is None or (current_time - self.last_synthesis_run >= synthesis_interval):
                        self.logger.info("ThinkTool: Triggering Synthesis & Strategy cycle.")
                        await self.synthesize_insights_and_strategize()
                        self.last_synthesis_run = current_time

                    # Run Technology Radar Cycle
                    if self.last_radar_run is None or (current_time - self.last_radar_run >= radar_interval):
                        self.logger.info("ThinkTool: Triggering Technology Radar cycle.")
                        await self.technology_radar()
                        self.last_radar_run = current_time

                else:
                    self.logger.debug("ThinkTool: Orchestrator not yet approved. Skipping periodic tasks.")

                # Sleep for a shorter interval to remain responsive
                await asyncio.sleep(60 * 5) # Check every 5 minutes

            except asyncio.CancelledError:
                self.logger.info("ThinkTool run loop cancelled.")
                break
            except Exception as e:
                self.logger.critical(f"ThinkTool: CRITICAL error in run loop: {e}", exc_info=True)
                await self.orchestrator.report_error("ThinkTool", f"Critical run loop error: {e}")
                # Wait longer after a critical error before retrying
                await asyncio.sleep(60 * 15) # Wait 15 mins

    # --- Placeholder/Legacy Methods (Adapt or Remove) ---
    # Keep analyze_visual and handle_quick_challenge if still needed,
    # ensuring they use _call_llm_with_retry

    async def analyze_visual(self, image_data, task_description):
        """Analyze visual data (requires appropriate LLM client/model)."""
        # This might need a specific multimodal client setup via orchestrator
        self.logger.warning("analyze_visual needs integration with a multimodal LLM client.")
        # Placeholder implementation using standard LLM (will likely fail for image data)
        prompt = f"{self.meta_prompt}\nAnalyze visual context for: {task_description}. Describe what you see."
        analysis = await self._call_llm_with_retry(prompt, max_tokens=500, is_json_output=False)
        return analysis or "Visual analysis failed or not supported."

    async def handle_quick_challenge(self, challenge_type: str, context: str) -> str:
        """Handle simple, quick challenges."""
        prompt = f"{self.meta_prompt}\nQuick challenge: {challenge_type}. Context: {context}\nProvide a concise solution or answer."
        solution = await self._call_llm_with_retry(prompt, temperature=0.4, max_tokens=200, is_json_output=False)
        if solution:
            self.logger.info(f"ThinkTool Quick challenge '{challenge_type}' solution generated.")
            return solution
        else:
            self.logger.error(f"ThinkTool: Failed to solve quick challenge '{challenge_type}'.")
            return f"Failed to solve challenge '{challenge_type}' due to API errors."

    # --- Cleanup ---
    # Add cleanup logic if needed, e.g., closing DB connections if not handled by session_maker context
    # async def cleanup(self):
    #     self.logger.info("ThinkTool shutting down.")
    #     # Add any cleanup tasks here