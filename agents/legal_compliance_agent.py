import asyncio
import logging
import json
import os
import sqlite3
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from openai import AsyncOpenAI as AsyncDeepSeekClient
import aiohttp
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from typing import Optional, Dict, Any, List, Tuple # Added
from .base_agent import GeniusAgentBase, KBInterface # Use relative import
from prompts.agent_meta_prompts import LEGAL_AGENT_META_PROMPT # Import meta prompt

# Genius-level logging for diagnostics and auditing
logger = logging.getLogger(__name__)
# Logging should be configured globally

class LegalAgent(GeniusAgentBase): # Renamed and inherited
    """
    Legal Agent (Genius Level): Proactively monitors legal landscapes, interprets regulations,
    identifies strategic compliance pathways and grey areas, and maintains the agency's legal knowledge base.
    """
    AGENT_NAME = "LegalAgent"

    # Ensure KBInterface is imported correctly at the top
    # from .base_agent import GeniusAgentBase, KBInterface

    def __init__(self, session_maker: AsyncSession, orchestrator: Any, kb_interface: KBInterface): # Accepts kb_interface
        # Pass kb_interface to the base class constructor
        super().__init__(agent_name=self.AGENT_NAME, kb_interface=kb_interface)
        # self.config is inherited from GeniusAgentBase
        # self.kb_interface is inherited from GeniusAgentBase
        self.session_maker = session_maker # Keep DB session maker if needed for operational logs? Maybe remove later.
        self.orchestrator = orchestrator # Keep orchestrator reference
        self.think_tool = orchestrator.agents.get('think') # Keep for non-core tasks initially

        # --- Internal State Initialization ---
        self.internal_state['meta_prompt'] = LEGAL_AGENT_META_PROMPT # Use imported prompt
        self.internal_state['legal_sources'] = { # Store sources in state
            "USA": [
                "https://www.federalregister.gov",
                "https://uscode.house.gov",
                "https://www.consumerfinance.gov"
            ],
            "Morocco": [
                "https://www.sgg.gov.ma",
                "https://www.bkam.ma",
                "https://www.justice.gov.ma"
            ]
        }
        self.internal_state['update_interval_seconds'] = self.config.get("LEGAL_UPDATE_INTERVAL_SECONDS", 604800) # 1 week default
        self.internal_state['similarity_threshold'] = self.config.get("LEGAL_CACHE_SIMILARITY_THRESHOLD", 0.85)
        self.internal_state['max_interpretation_tokens'] = self.config.get("LEGAL_INTERPRETATION_MAX_TOKENS", 800)
        self.internal_state['max_validation_tokens'] = self.config.get("LEGAL_VALIDATION_MAX_TOKENS", 800)
        self.internal_state['gray_area_strategies'] = {} # Store identified strategies here

        # --- Direct Attributes for Core Functionality ---
        # Using SQLite cache for simplicity as in original code. Could be moved to main DB/KB later.
        self.cache_db_path = "legal_cache.db"
        self.cache_db = sqlite3.connect(self.cache_db_path, check_same_thread=False)
        self.create_cache_tables() # Initialize cache tables
        self.tfidf_vectorizer = TfidfVectorizer() # Keep vectorizer instance

        self.logger.info(f"{self.AGENT_NAME} (Genius Level) initialized. Cache DB: {self.cache_db_path}")

    def update_contract_date(self, contract_template):
        today = datetime.now().strftime("%Y-%m-%d")  # Makes date like "2025-04-07"
        updated_contract = contract_template.replace("[CONTRACT_DATE]", today)
        return updated_contract

    async def get_w8_data(self):
        w8_data_str = await self.secure_storage.get_secret("w8ben_data")
        w8_data = json.loads(w8_data_str)
        return w8_data

    async def validate_w8_data(self):
        w8_data_str = await self.secure_storage.get_secret("w8ben_data")
        w8_data = json.loads(w8_data_str)
        required_fields = ["name", "country", "address", "tin"]
        if not all(field in w8_data for field in required_fields):
            logger.error("W-8BEN data incomplete")
            return False
        return True

    def create_cache_tables(self):
        """Initialize SQLite tables for genius-level caching."""
        cursor = self.cache_db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS legal_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country TEXT,
                content TEXT,
                timestamp DATETIME,
                hash TEXT UNIQUE
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interpretations (
                update_id INTEGER,
                category TEXT,
                interpretation TEXT,
                gray_areas TEXT,
                compliance_note TEXT,
                FOREIGN KEY(update_id) REFERENCES legal_updates(id)
            )""")
        self.cache_db.commit()

    async def fetch_legal_updates(self):
        """Asynchronously fetch legal updates with SmartProxy optimization."""
        async with aiohttp.ClientSession() as session:
            updates = {}
            for country, url in self.legal_sources.items():
                async with session.get(url) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")
                    if country == "USA":
                        updates[country] = [doc.text.strip() for doc in soup.find_all("div", class_="document-title")]
                    elif country == "Morocco":
                        updates[country] = [doc.text.strip() for doc in soup.find_all("div", class_="bulletin-entry") or soup.find_all("p")]
            logger.info(f"Fetched updates: {len(updates.get('USA', []))} US, {len(updates.get('Morocco', []))} Moroccan")
            return updates

    async def interpret_updates(self, updates: Dict[str, List[str]]) -> Dict[str, Any]:
        """Interpret legal updates using internal logic and a single LLM call."""
        interpretations = {}
        texts_to_process = []
        text_to_country_map = {}

        # Collect texts needing interpretation (not cached)
        for country, texts in updates.items():
            for text_content in texts:
                # Use strip() to handle potential leading/trailing whitespace affecting cache lookup
                stripped_text = text_content.strip()
                if not stripped_text: continue # Skip empty texts

                cached = self.get_cached_interpretation(stripped_text)
                if cached:
                    interpretations[stripped_text] = cached
                    # Update internal state directly
                    self.internal_state['gray_area_strategies'][country] = cached.get('gray_areas', [])
                    self.logger.debug(f"Cache hit for legal text snippet starting with: {stripped_text[:50]}...")
                else:
                    if stripped_text not in text_to_country_map: # Avoid duplicate processing requests
                        texts_to_process.append(stripped_text)
                        text_to_country_map[stripped_text] = country

        if not texts_to_process:
            self.logger.info("No new legal texts require interpretation.")
            return interpretations # Return cached results if any

        self.logger.info(f"Requesting interpretation for {len(texts_to_process)} new legal text snippets.")

        # Prepare context and generate dynamic prompt
        task_context = {
            "task": "Interpret legal updates and identify grey areas",
            "legal_texts": texts_to_process,
            "jurisdictions": list(set(text_to_country_map.values())), # List unique countries involved
            "desired_output_format": "JSON array, where each item corresponds to an input text and has keys: 'category', 'interpretation', 'gray_areas' (list), 'compliance_note'."
        }
        interpretation_prompt = await self.generate_dynamic_prompt(task_context)

        # Make single LLM call (using think_tool as proxy for now)
        try:
            llm_response_str = await self.think_tool._call_llm_with_retry(
                interpretation_prompt,
                model=self.config.get("OPENROUTER_LEGAL_INTERPRET_MODEL", "google/gemini-1.5-pro-latest"), # Use capable model
                temperature=0.3,
                max_tokens=self.internal_state.get('max_interpretation_tokens', 1500), # Allow more tokens for batch
                response_format={"type": "json_object"} # Expecting JSON array within a JSON object potentially
            )

            if not llm_response_str:
                raise Exception("LLM call for legal interpretation returned empty response.")

            # Parse the response (expecting a JSON array, possibly nested)
            try:
                # Find JSON array if nested or has preamble
                json_start = llm_response_str.find('[')
                json_end = llm_response_str.rfind(']') + 1
                if json_start != -1 and json_end != -1:
                    llm_response_json_str = llm_response_str[json_start:json_end]
                    results_list = json.loads(llm_response_json_str)

                    if isinstance(results_list, list) and len(results_list) == len(texts_to_process):
                        for original_text, result_data in zip(texts_to_process, results_list):
                            if isinstance(result_data, dict): # Basic validation of result structure
                                country = text_to_country_map[original_text]
                                interpretations[original_text] = result_data
                                # Update internal state directly
                                self.internal_state['gray_area_strategies'][country] = result_data.get('gray_areas', [])
                                self.cache_interpretation(original_text, result_data, country)
                            else:
                                self.logger.warning(f"Invalid result structure for text: {original_text[:50]}...")
                    else:
                        self.logger.error(f"LLM interpretation result is not a list or length mismatch. Expected {len(texts_to_process)}, Got: {type(results_list)} len {len(results_list) if isinstance(results_list, list) else 'N/A'}")

                else:
                     self.logger.error(f"Could not find JSON array '[]' in LLM interpretation response: {llm_response_str[:200]}...")

            except json.JSONDecodeError as jde:
                self.logger.error(f"Failed to decode JSON interpretation response from LLM: {jde}. Response: {llm_response_str[:200]}...")
            except Exception as parse_exc:
                 self.logger.error(f"Error parsing LLM interpretation response: {parse_exc}. Response: {llm_response_str[:200]}...")

        except Exception as e:
            self.logger.error(f"{self.AGENT_NAME}: Error during interpret_updates LLM call: {e}", exc_info=True)
            # Return only cached interpretations on failure
            return interpretations

        self.logger.info(f"Interpretation complete. Total interpretations available: {len(interpretations)}")
        return interpretations

    def get_cached_interpretation(self, text):
        """Genius-level caching with similarity detection."""
        cursor = self.cache_db.cursor()
        cursor.execute("SELECT content, interpretation FROM legal_updates JOIN interpretations ON legal_updates.id = interpretations.update_id")
        cached_data = cursor.fetchall()
        if not cached_data:
            return None

        texts = [text] + [row[0] for row in cached_data]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        if similarities.max() > 0.85:  # High threshold for precision
            idx = similarities.argmax()
            return json.loads(cached_data[idx][1])
        return None

    def cache_interpretation(self, text, interpretation, country):
        """Persist interpretations with unique hashing."""
        import hashlib
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cursor = self.cache_db.cursor()
        try:
            cursor.execute("INSERT INTO legal_updates (country, content, timestamp, hash) VALUES (?, ?, ?, ?)",
                           (country, text, datetime.now(), text_hash))
            update_id = cursor.lastrowid
            cursor.execute("INSERT INTO interpretations (update_id, category, interpretation, gray_areas, compliance_note) VALUES (?, ?, ?, ?, ?)",
                           (update_id, interpretation['category'], interpretation['interpretation'],
                            json.dumps(interpretation['gray_areas']), interpretation['compliance_note']))
            self.cache_db.commit()
        except sqlite3.IntegrityError:
            pass  # Duplicate hash; skip caching

    # FILE: agents/legal_compliance_agent.py
# Modify the validate_operation() method

    async def validate_operation(self, operation_description: str) -> dict:
        """Validate operations using internal logic, local cache, and a single LLM call."""
        self.logger.debug(f"Validating operation: {operation_description[:100]}...")
        validation_result_json = None
        # Define fallback structure here for use in error handling
        fallback_result = {
            'is_compliant': False, 'risk_score': 1.0, 'compliance_issues': ['Analysis Error'],
            'gray_area_exploitation': False, 'potential_consequences': [], 'recommendations': [],
            'proceed_recommendation': 'Halt - Analysis Failed'
        }
        try:
            # 1. Fetch relevant context from local cache DB
            cursor = self.cache_db.cursor()
            cursor.execute("SELECT interpretation, gray_areas, compliance_note FROM interpretations ORDER BY update_id DESC LIMIT 20")
            recent_interpretations = [
                {"interp": row[0], "gray_areas": json.loads(row[1] or '[]'), "note": row[2]}
                for row in cursor.fetchall()
            ]
            self.logger.debug(f"Fetched {len(recent_interpretations)} recent interpretations from cache for validation context.")

            # 2. Prepare context and generate dynamic prompt
            task_context = {
                "task": "Validate operation compliance and risk",
                "operation_description": operation_description,
                "recent_legal_context": recent_interpretations, # Provide context from cache
                "desired_output_format": "JSON object with keys: 'is_compliant', 'risk_score', 'compliance_issues', 'gray_area_exploitation', 'potential_consequences', 'recommendations', 'proceed_recommendation'."
            }
            validation_prompt = await self.generate_dynamic_prompt(task_context)

            # 3. Make single LLM call (using think_tool as proxy for now)
            validation_result_json = await self.think_tool._call_llm_with_retry(
                validation_prompt,
                model=self.config.get("OPENROUTER_LEGAL_VALIDATE_MODEL", "google/gemini-1.5-pro-latest"), # Use capable model
                temperature=0.2, # Low temp for precise analysis
                max_tokens=self.internal_state.get('max_validation_tokens', 800),
                response_format={"type": "json_object"}
            )

            if not validation_result_json:
                raise Exception("LLM call for validation returned empty response.")

            # 4. Parse and process result
            try:
                # Find JSON object if nested
                json_start = validation_result_json.find('{')
                json_end = validation_result_json.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    result_str = validation_result_json[json_start:json_end]
                    result = json.loads(result_str)
                else:
                    raise json.JSONDecodeError("No JSON object found", validation_result_json, 0)

                # Add default values for robustness
                result.setdefault('is_compliant', False)
                result.setdefault('risk_score', 1.0) # Default high risk on failure
                result.setdefault('compliance_issues', ['Analysis Failed'])
                result.setdefault('gray_area_exploitation', False)
                result.setdefault('potential_consequences', [])
                result.setdefault('recommendations', [])
                result.setdefault('proceed_recommendation', 'Halt - Analysis Failed')

                self.logger.info(f"Legal Validation for '{operation_description[:50]}...': Compliant={result['is_compliant']}, Risk Score={result['risk_score']:.2f}, Recommendation={result['proceed_recommendation']}")

                # Trigger notifications/errors via Orchestrator based on severity
                # Ensure orchestrator reference exists and has the methods before calling
                if hasattr(self.orchestrator, 'report_error') and (not result['is_compliant'] or result['risk_score'] > 0.8):
                    await self.orchestrator.report_error(
                        self.AGENT_NAME,
                        f"Operation blocked/high-risk: {operation_description[:100]}... Issues: {result['compliance_issues']}, Risk: {result['risk_score']:.2f}"
                    )
                elif hasattr(self.orchestrator, 'send_notification') and (result['risk_score'] > 0.5 or result.get('recommendations')):
                     await self.orchestrator.send_notification(
                         "LegalCompliance Advisory",
                         f"Operation: {operation_description[:100]}... Risk: {result['risk_score']:.2f}. Recommendations: {result['recommendations']}"
                     )
                return result # Return the parsed dictionary

            except json.JSONDecodeError as jde:
                 self.logger.error(f"LegalAgent: Failed to decode JSON validation: {jde}. Response: {validation_result_json}")
                 fallback_result['compliance_issues'] = ['LLM Error/Invalid JSON']
                 return fallback_result # Return fallback on JSON error
            except Exception as parse_exc:
                 self.logger.error(f"LegalAgent: Error parsing validation result: {parse_exc}")
                 fallback_result['compliance_issues'] = [f'Parsing Error: {parse_exc}']
                 return fallback_result # Return fallback on other parsing errors

        except Exception as e:
            self.logger.error(f"{self.AGENT_NAME}: Error during validate_operation: {e}", exc_info=True)
            fallback_result['compliance_issues'] = [f'Analysis Error: {e}']
            return fallback_result # Return fallback on any outer error

    async def get_invoice_legal_note(self, client_country):
        cursor = self.cache_db.cursor()
        cursor.execute("""
            SELECT compliance_note FROM interpretations
            WHERE category IN ('financial', 'tax')
            AND country = ?
            ORDER BY update_id DESC LIMIT 1
        """, (client_country,))
        row = cursor.fetchone()
        base_note = row[0] if row else "Compliant with latest financial regulations"

        bulletproof_terms = (
            "This agreement is irrevocable and binding upon acceptance. "
            "UGC Genius shall not be liable for any indirect, incidental, or consequential damages, including but not limited to loss of profits or data. "
            "All disputes shall be resolved exclusively under Moroccan law in the courts of [Your City], Morocco. "
            f"Payment to {self.orchestrator.user_name}'s account in Morocco is non-refundable and non-cancellable once services commence. "
            "Client acknowledges that UGC Genius adheres to ISO 20022 standards for secure international transactions."
        )
        return f"{base_note} | {bulletproof_terms}"


        

    # --- Learning Loop Implementation (Adapting existing run logic) ---
    async def learning_loop(self):
        """Periodic loop to fetch and interpret legal updates."""
        self.logger.info("Executing LegalAgent learning loop (fetch & interpret)...")
        try:
            # Fetch updates from configured sources
            updates = await self.fetch_legal_updates()
            if updates:
                # Interpret updates using internal logic (which caches results)
                interpretations = await self.interpret_updates(updates)

                # Potential Enhancement: Store high-level summary or identified grey areas
                # in the main Knowledge Base via self.kb_interface here.
                # Example:
                # summary = f"Processed {len(interpretations)} legal interpretations. Identified new grey areas in: {list(self.internal_state['gray_area_strategies'].keys())}"
                # if self.kb_interface:
                #     await self.kb_interface.add_knowledge(summary, 'legal_update_summary', self.AGENT_NAME, {'timestamp': datetime.now().isoformat()})

                # Notify orchestrator about new grey area opportunities found during interpretation
                # (interpret_updates already updates self.internal_state['gray_area_strategies'])
                # Consider if notification logic should live here or within interpret_updates
                for country, strategies in self.internal_state['gray_area_strategies'].items():
                     if strategies and hasattr(self.orchestrator, 'send_notification'): # Check if strategies list is not empty
                          # Maybe only notify about *newly added* strategies if possible to track
                          await self.orchestrator.send_notification(
                              "GrayAreaOpportunity",
                              f"Potential exploit(s) identified/updated for {country}: {strategies}"
                          )

                self.logger.info(f"Legal update interpretation cycle complete. Found/updated {len(interpretations)} interpretations.")
            else:
                self.logger.info("No new legal updates fetched in this cycle.")

        except Exception as e:
            self.logger.error(f"Error during LegalAgent learning loop: {e}", exc_info=True)
            if hasattr(self.orchestrator, 'report_error'):
                 await self.orchestrator.report_error(self.AGENT_NAME, f"Learning loop error: {e}")

        # Schedule next run (using internal state interval)
        try:
            interval = self.internal_state.get('update_interval_seconds', 604800)
            self.logger.debug(f"LegalAgent learning loop sleeping for {interval} seconds.")
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            self.logger.info("LegalAgent learning loop cancelled.")
            raise # Propagate cancellation

    # --- Abstract Method Implementations ---

    async def execute_task(self, task_details: Dict[str, Any]) -> Any:
        """Core method to execute the agent's primary function for a given task."""
        self.logger.info(f"execute_task received task: {task_details}")
        task_type = task_details.get('type')
        if task_type == 'validate_operation':
            description = task_details.get('description')
            if description:
                # Calls the already refactored validate_operation
                return await self.validate_operation(description)
            else:
                self.logger.error("Missing 'description' for validate_operation task.")
                return {"status": "failed", "reason": "Missing description"}
        elif task_type == 'get_invoice_note':
            country = task_details.get('country')
            if country:
                return {"status": "ok", "note": await self.get_invoice_legal_note(country)}
            else:
                self.logger.error("Missing 'country' for get_invoice_note task.")
                return {"status": "failed", "reason": "Missing country"}
        # Potential future task: 'research_specific_law'
        # elif task_type == 'research_specific_law':
        #     query = task_details.get('query')
        #     jurisdiction = task_details.get('jurisdiction')
        #     # ... call research method ...
        else:
            self.logger.warning(f"Unsupported task type for LegalAgent: {task_type}")
            return {"status": "failed", "reason": f"Unsupported task type: {task_type}"}

    async def self_critique(self) -> Dict[str, Any]:
        """Method for the agent to evaluate its own performance and strategy."""
        self.logger.info("self_critique: Placeholder - Not yet implemented.")
        # TODO: Implement logic to analyze cache hit rate, interpretation quality (if feedback available),
        # timeliness of updates, effectiveness of identified grey areas.
        # Query local cache stats, potentially KB for outcomes linked to legal advice.
        cache_stats = "N/A" # Placeholder
        critique_summary = f"Cache Stats: {cache_stats}. Interpretation quality feedback needed. Grey area effectiveness tracking needed."
        return {"status": "ok", "feedback": critique_summary}

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Constructs context-rich prompts for LLM calls (e.g., interpretation, validation)."""
        self.logger.debug(f"Generating dynamic prompt for task: {task_context.get('task')}")
        # Start with the base meta-prompt
        prompt_parts = [self.internal_state.get('meta_prompt', "Analyze legal compliance.")]

        # Add relevant context from the task
        prompt_parts.append("\n--- Current Task Context ---")
        for key, value in task_context.items():
            if key == 'recent_legal_context': # Handle potentially large context separately
                 prompt_parts.append(f"\n**Recent Legal Interpretations & Grey Areas (Cache):**")
                 # Limit the amount of context to avoid excessive prompt length
                 context_str = json.dumps(value[:10], indent=2) # Show first 10 interpretations
                 prompt_parts.append(context_str)
                 if len(value) > 10:
                     prompt_parts.append(f"... (and {len(value) - 10} more)")
            elif key == 'legal_texts':
                 prompt_parts.append(f"\n**Legal Texts to Analyze:**")
                 texts_str = json.dumps(value, indent=2)
                 prompt_parts.append(texts_str[:2000] + "..." if len(texts_str) > 2000 else texts_str) # Limit length
            elif isinstance(value, (str, int, float, bool)):
                 prompt_parts.append(f"{key.replace('_', ' ').title()}: {value}")
            # Add more specific handling for other expected context types if needed

        # Add relevant context from KB (Simulated)
        # prompt_parts.append("\n--- Relevant Knowledge (Simulated KB Retrieval) ---")
        # Example: Fetch relevant corporate strategy examples if task is grey area identification
        # if task_context.get('task') == 'Interpret legal updates and identify grey areas':
        #     simulated_kb_insight = "- Competitor X successfully used [Strategy Y] in [Jurisdiction Z]."
        #     prompt_parts.append(simulated_kb_insight)

        # Add Specific Instructions based on task
        prompt_parts.append("\n--- Instructions ---")
        if task_context.get('task') == 'Interpret legal updates and identify grey areas':
            prompt_parts.append("1. Analyze each provided legal text snippet.")
            prompt_parts.append("2. For each, determine the relevant law category.")
            prompt_parts.append("3. Summarize the core interpretation and its impact on AI agency operations (sales, marketing, data handling).")
            prompt_parts.append("4. Critically identify any ambiguities or 'grey areas' that could be strategically leveraged (or pose hidden risks).")
            prompt_parts.append("5. Generate a concise compliance note suitable for related documentation (e.g., invoices).")
            prompt_parts.append(f"6. **Output Format:** {task_context.get('desired_output_format', 'JSON array as specified previously.')}")
        elif task_context.get('task') == 'Validate operation compliance and risk':
            prompt_parts.append("1. Analyze the 'Operation to Validate' against the provided 'Recent Legal Interpretations & Grey Areas'.")
            prompt_parts.append("2. Focus on USA/Moroccan law implications and assume high agency risk tolerance for profit, but zero tolerance for severe penalties.")
            prompt_parts.append("3. Assess strict compliance ('is_compliant').")
            prompt_parts.append("4. Quantify risk ('risk_score' 0.0-1.0) considering probability and severity.")
            prompt_parts.append("5. List specific 'compliance_issues'.")
            prompt_parts.append("6. Note if 'gray_area_exploitation' is involved.")
            prompt_parts.append("7. Describe 'potential_consequences'.")
            prompt_parts.append("8. Provide actionable 'recommendations' for mitigation or compliant alternatives.")
            prompt_parts.append("9. Give an explicit 'proceed_recommendation' (Proceed, Proceed with Caution, Halt - High Risk, Halt - Non-Compliant).")
            prompt_parts.append(f"10. **Output Format:** {task_context.get('desired_output_format', 'Single JSON object with specified keys.')}")
        else:
            prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")

        prompt_parts.append("```json") # Hint for the LLM if JSON is expected

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for LegalAgent (length: {len(final_prompt)} chars)")
        return final_prompt