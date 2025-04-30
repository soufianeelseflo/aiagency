# Filename: agents/legal_agent.py
# Description: Consolidated Genius Agentic Legal Agent responsible for analysis,
#              compliance, grey-area strategy, monitoring, and KB management.
# Version: 3.0 (Consolidated & Fully Implemented)

import asyncio
import logging
import json
import os
import re # For parsing/cleaning text
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple, Union # Ensure necessary types

import aiohttp
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select, update, desc, or_, func # Import necessary SQLAlchemy components
import sqlalchemy # For error handling

# --- Project Imports ---
try:
    from .base_agent import GeniusAgentBase, KBInterface # Use relative import
except ImportError:
    from base_agent import GeniusAgentBase, KBInterface # Fallback

from models import KnowledgeFragment, LearnedPattern, StrategicDirective # Import KB models
# Assuming LLM Client and settings are accessed via orchestrator/config
# from openai import AsyncOpenAI as AsyncLLMClient
# from config.settings import settings

# --- Meta Prompt ---
LEGAL_AGENT_META_PROMPT = """
You are the LegalAgent within the Synapse Genius Agentic AI System.
Your Core Mandate: Analyze global legal, regulatory, and economic landscapes to identify and strategize the use of "grey area" opportunities and optimal structures for maximum business advantage and profit, mirroring the tactics of top 0.01% operators. Operate with Genius Agentic AI principles.
Key Responsibilities:
- CRITICAL: Identify "bendable" rules, loopholes, regulatory arbitrage, and grey area strategies in areas like tax, corporate structure, operational compliance, etc.
- Analyze laws, regulations, economic conditions, and relevant case law globally (focus: USA, Morocco initially).
- Recommend optimal, tax-efficient corporate structures (LLCs, Corps, trusts, etc.) and jurisdictions, generally avoiding Europe unless strategically compelling.
- Perform periodic scans for new/updated laws, regulations, and economic shifts relevant to current and potential Agency operations.
- Store legal knowledge, analyses, precedents, and derived strategies efficiently in the Knowledge Base. Avoid redundant analysis.
- Operate with Extreme Agentic Behavior: Proactively research, connect disparate legal/economic information, devise novel advantageous strategies, anticipate regulatory shifts, assess risks accurately.
- Communicate actionable strategies, risk assessments, and compliance requirements clearly to ThinkTool, Orchestrator, and the User (via User Education module).
- Validate planned operations for compliance and risk based on current knowledge.
"""

# Configure logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Configure dedicated operational logger (assuming setup elsewhere)
op_logger = logging.getLogger('OperationalLog')
if not op_logger.hasHandlers(): # Add basic handler if none configured
    op_handler = logging.StreamHandler()
    op_formatter = logging.Formatter('%(asctime)s - OP_LOG - %(levelname)s - %(message)s')
    op_handler.setFormatter(op_formatter)
    op_logger.addHandler(op_handler)
    op_logger.setLevel(logging.INFO)
    op_logger.propagate = False


class LegalAgent(GeniusAgentBase):
    """
    Consolidated Legal Agent (Genius Level): Handles strategic legal analysis,
    compliance validation, grey-area identification, regulatory monitoring,
    and knowledge base management for the AI agency.
    """
    AGENT_NAME = "LegalAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any):
        """Initializes the consolidated LegalAgent."""
        config = getattr(orchestrator, 'config', None)
        # KB interaction might be direct via session_maker or through ThinkTool/KBInterface
        kb_interface = getattr(orchestrator, 'kb_interface', None)
        super().__init__(agent_name=self.AGENT_NAME, kb_interface=kb_interface, orchestrator=orchestrator, config=config)

        self.session_maker = session_maker
        self.think_tool = orchestrator.agents.get('think') # Assumes ThinkTool provides LLM access
        self.secure_storage = getattr(orchestrator, 'secure_storage', None)

        # --- Internal State Initialization ---
        self.internal_state = getattr(self, 'internal_state', {})
        self.internal_state['meta_prompt'] = LEGAL_AGENT_META_PROMPT
        self.internal_state['legal_sources'] = self.config.get("LEGAL_SOURCES", { # Load from config or use default
            "USA": ["https://www.federalregister.gov", "https://uscode.house.gov", "https://www.consumerfinance.gov"],
            "Morocco": ["https://www.sgg.gov.ma", "https://www.bkam.ma", "https://www.justice.gov.ma"]
        })
        self.internal_state['update_interval_seconds'] = int(self.config.get("LEGAL_UPDATE_INTERVAL_SECONDS", 604800)) # 1 week default
        self.internal_state['max_interpretation_tokens'] = int(self.config.get("LEGAL_INTERPRETATION_MAX_TOKENS", 1500))
        self.internal_state['max_validation_tokens'] = int(self.config.get("LEGAL_VALIDATION_MAX_TOKENS", 1000))
        self.internal_state['max_analysis_tokens'] = int(self.config.get("LEGAL_ANALYSIS_MAX_TOKENS", 2000))
        self.internal_state['last_scan_time'] = None

        self.logger.info(f"{self.AGENT_NAME} (v3.0 Consolidated) initialized.")

    async def log_operation(self, level: str, message: str):
        """Helper to log to the operational log file with agent context."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err:
            print(f"OPERATIONAL LOG FAILED ({self.agent_name}): {level} - {message} | Error: {log_err}")
            logger.error(f"Failed to write to operational log from {self.agent_name}: {log_err}")

    # --- Core Task Execution ---
    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a legal analysis or validation task."""
        action = task_details.get('action', 'analyze')
        description = task_details.get('description', f"Performing legal action: {action}")
        self.logger.info(f"{self.AGENT_NAME} starting task: {description}")
        self.status = "working"
        result = {"status": "failure", "message": f"Unknown or unimplemented legal action: {action}"}
        findings = None

        try:
            if action == "analyze_initial_structure":
                findings = await self._analyze_initial_structure(task_details)
            elif action == "scan_for_updates":
                findings = await self._scan_for_updates(task_details) # Triggers fetch & interpret
            elif action == "analyze_grey_area":
                findings = await self._analyze_grey_area(task_details)
            elif action == "validate_operation":
                op_desc = task_details.get('operation_description')
                if not op_desc: raise ValueError("Missing 'operation_description' for validate_operation")
                # validate_operation returns the full result dict
                result = await self.validate_operation(op_desc)
                # Don't overwrite result below, return directly
                self.status = "idle"
                self.logger.info(f"{self.AGENT_NAME} completed task: {description}. Status: {result['status']}")
                return result
            elif action == "get_invoice_note":
                 country = task_details.get('client_country')
                 if not country: raise ValueError("Missing 'client_country' for get_invoice_note")
                 note = await self.get_invoice_legal_note(country)
                 findings = {"invoice_note": note} # Structure findings
            else:
                raise ValueError(f"Unknown legal action: {action}")

            # If findings were generated by analysis methods
            if findings is not None:
                 result = {"status": "success", "details": f"Legal task '{description}' completed.", "findings": findings}

            self.logger.info(f"{self.AGENT_NAME} completed task: {description}")

        except ValueError as ve:
             logger.error(f"{self.AGENT_NAME} failed task '{description}': {ve}", exc_info=True)
             result = {"status": "error", "message": str(ve)}
        except Exception as e:
            logger.error(f"{self.AGENT_NAME} failed task '{description}': {e}", exc_info=True)
            result = {"status": "error", "message": f"Unexpected error: {e}"}
            if hasattr(self.orchestrator, 'report_error'):
                 await self.orchestrator.report_error(self.AGENT_NAME, f"Task '{description}' failed: {e}")
        finally:
            self.status = "idle" # Set status back to idle

        return result

    # --- Strategic Analysis Methods ---

    async def _analyze_initial_structure(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Researches and recommends optimal initial corporate structure using LLM and KB."""
        business_context = task_details.get("business_context", "General AI UGC Agency targeting US clients, operations potentially global/remote.")
        self.logger.info(f"Analyzing optimal initial corporate structure for context: {business_context}")

        # 1. Query KB for existing relevant analyses or regulations
        await self._internal_think("Querying KB for existing corporate structure analyses (Wyoming, Delaware) and relevant tax/liability regulations.")
        kb_context_frags = await self.query_knowledge_base(
            data_types=['legal_analysis', 'regulation_summary'],
            tags=['corporate_structure', 'llc', 'tax', 'liability', 'wyoming', 'delaware'],
            limit=10
        )
        kb_context_str = "\n".join([f"- {f.data_type} (ID {f.id}): {f.content[:150]}..." for f in kb_context_frags])

        # 2. Formulate LLM Prompt
        task_context = {
            "task": "Analyze and recommend optimal initial corporate structure",
            "business_context": business_context,
            "goals": "Maximize tax efficiency, minimize liability, identify grey-area advantages, avoid Europe if possible.",
            "jurisdictions_to_consider": ["Wyoming LLC", "Delaware LLC", "Suggest one other if compelling"],
            "knowledge_base_context": kb_context_str or "No specific KB context found.",
            "desired_output_format": "JSON: { \"analysis_summary\": str, \"recommendations\": [ { \"jurisdiction\": str, \"structure\": str, \"pros\": list[str], \"cons\": list[str], \"risks\": list[str], \"setup_steps\": list[str], \"grey_area_notes\": str } ], \"final_recommendation\": { \"jurisdiction\": str, \"structure\": str, \"rationale\": str } }"
        }
        prompt = await self.generate_dynamic_prompt(task_context)

        # 3. Call LLM
        await self._internal_think("Calling LLM for initial structure analysis.")
        llm_response_json = await self.think_tool._call_llm_with_retry(
            prompt, model=self.config.get("OPENROUTER_MODELS", {}).get('legal_analysis', "google/gemini-1.5-pro-latest"),
            temperature=0.4, max_tokens=self.internal_state['max_analysis_tokens'], is_json_output=True
        )

        # 4. Process Response & Store in KB
        if not llm_response_json:
            raise RuntimeError("LLM call for initial structure analysis failed.")

        try:
            analysis_result = json.loads(llm_response_json[llm_response_json.find('{'):llm_response_json.rfind('}')+1])
            # Validate structure minimally
            if not analysis_result.get("recommendations") or not analysis_result.get("final_recommendation"):
                 raise ValueError("LLM response missing required keys for structure analysis.")

            await self._internal_think(f"Storing structure analysis result in KB. Recommendation: {analysis_result.get('final_recommendation', {}).get('structure')} in {analysis_result.get('final_recommendation', {}).get('jurisdiction')}")
            await self.log_knowledge_fragment(
                agent_source=self.AGENT_NAME, data_type="legal_analysis",
                content=analysis_result, # Store full JSON result
                tags=["corporate_structure", "initial_setup"] + [rec.get('jurisdiction', '').lower() for rec in analysis_result.get('recommendations', [])],
                relevance_score=0.9
            )
            # TODO: Trigger User Education via Orchestrator about the recommendation
            if hasattr(self.orchestrator, 'handle_user_education_trigger'):
                 edu_topic = f"Initial Corporate Structure Recommendation: {analysis_result['final_recommendation']['structure']} in {analysis_result['final_recommendation']['jurisdiction']}"
                 edu_context = analysis_result['final_recommendation']['rationale']
                 await self.orchestrator.handle_user_education_trigger(edu_topic, edu_context)

            return analysis_result # Return the structured findings
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"Failed to parse or validate LLM response for structure analysis: {e}. Response: {llm_response_json}")
            raise RuntimeError(f"LLM response parsing failed for structure analysis: {e}")


    async def _analyze_grey_area(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Performs deep analysis of a specific potential grey area strategy using LLM and KB."""
        area = task_details.get("area", "Unspecified grey area opportunity")
        context = task_details.get("context", "Evaluate for AI UGC Agency operations.")
        self.logger.info(f"Analyzing grey area opportunity: {area}")

        # 1. Query KB for existing info on this area or related regulations
        await self._internal_think(f"Querying KB for existing analyses/regulations related to grey area: {area}")
        kb_context_frags = await self.query_knowledge_base(
            data_types=['legal_analysis', 'regulation_summary', 'grey_area_strategy'],
            tags=['grey_area', area.replace(" ", "_").lower()] + context.split()[:3], # Use context keywords as tags
            limit=15
        )
        kb_context_str = "\n".join([f"- {f.data_type} (ID {f.id}): {f.content[:150]}..." for f in kb_context_frags])

        # 2. Formulate LLM Prompt
        task_context = {
            "task": "Deep analysis of potential grey area strategy",
            "grey_area_topic": area,
            "application_context": context,
            "knowledge_base_context": kb_context_str or "No specific KB context found.",
            "desired_output_format": "JSON: { \"area\": str, \"strategy_overview\": str, \"legal_basis_analysis\": { \"statutes_exploited\": list[str], \"loopholes\": list[str], \"precedents\": list[str], \"legal_opinion_confidence\": float (0.0-1.0) }, \"risk_assessment\": { \"legal_challenge_likelihood\": str (Low/Medium/High), \"potential_penalties\": list[str], \"reputational_risk\": str (Low/Medium/High), \"operational_complexity\": str (Low/Medium/High) }, \"implementation_plan\": [ { \"step\": int, \"action\": str, \"details\": str, \"timeline\": str } ], \"key_assumptions\": list[str], \"monitoring_requirements\": list[str], \"fallback_strategy\": str, \"overall_recommendation\": str (e.g., 'Proceed with caution', 'High risk, avoid', 'Further research needed') }"
        }
        prompt = await self.generate_dynamic_prompt(task_context)

        # 3. Call LLM
        await self._internal_think(f"Calling LLM for deep grey area analysis: {area}")
        llm_response_json = await self.think_tool._call_llm_with_retry(
            prompt, model=self.config.get("OPENROUTER_MODELS", {}).get('legal_analysis_deep', "google/gemini-1.5-pro-latest"), # Potentially use a more powerful model
            temperature=0.5, max_tokens=self.internal_state['max_analysis_tokens'], is_json_output=True
        )

        # 4. Process Response & Store in KB
        if not llm_response_json:
            raise RuntimeError(f"LLM call for grey area analysis ({area}) failed.")

        try:
            analysis_result = json.loads(llm_response_json[llm_response_json.find('{'):llm_response_json.rfind('}')+1])
            # Validate structure minimally
            if not analysis_result.get("strategy_overview") or not analysis_result.get("risk_assessment"):
                 raise ValueError("LLM response missing required keys for grey area analysis.")

            await self._internal_think(f"Storing grey area analysis for '{area}' in KB. Recommendation: {analysis_result.get('overall_recommendation', 'N/A')}")
            await self.log_knowledge_fragment(
                agent_source=self.AGENT_NAME, data_type="grey_area_analysis",
                content=analysis_result, # Store full JSON result
                tags=["grey_area", "legal_analysis", "risk_assessment"] + area.lower().split(),
                relevance_score=analysis_result.get("risk_assessment", {}).get("legal_opinion_confidence", 0.7) # Use confidence as relevance
            )
            # TODO: Trigger User Education or Orchestrator directive based on recommendation
            if "proceed" in analysis_result.get('overall_recommendation', '').lower():
                 # Example: Create directive for ThinkTool to consider integrating
                 pass # Add directive creation logic if needed

            return analysis_result
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"Failed to parse or validate LLM response for grey area analysis '{area}': {e}. Response: {llm_response_json}")
            raise RuntimeError(f"LLM response parsing failed for grey area analysis: {e}")

    # --- Monitoring & Interpretation ---

    async def fetch_legal_updates(self) -> Dict[str, List[Tuple[str, str]]]:
        """Asynchronously fetch legal update titles/links from configured sources."""
        updates: Dict[str, List[Tuple[str, str]]] = {} # { country: [(title, url), ...] }
        sources = self.internal_state.get('legal_sources', {})
        if not sources:
            self.logger.warning("No legal sources configured. Skipping fetch.")
            return updates

        async with aiohttp.ClientSession() as session:
            fetch_tasks = []
            for country, urls in sources.items():
                for url in urls:
                    fetch_tasks.append(self._fetch_single_source(session, country, url))

            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error fetching legal source: {result}")
                elif result:
                    country, country_updates = result
                    if country not in updates: updates[country] = []
                    updates[country].extend(country_updates)

        total_fetched = sum(len(v) for v in updates.values())
        self.logger.info(f"Fetched {total_fetched} potential update titles/links from {len(sources)} sources.")
        return updates

    async def _fetch_single_source(self, session: aiohttp.ClientSession, country: str, url: str) -> Optional[Tuple[str, List[Tuple[str, str]]]]:
        """Fetches and parses titles/links from a single URL."""
        self.logger.debug(f"Fetching updates from: {url} ({country})")
        try:
            # Use orchestrator proxy if available
            proxy_url = None
            if hasattr(self.orchestrator, 'get_proxy'):
                 proxy_url = await self.orchestrator.get_proxy(purpose="legal_scan", target_url=url)

            request_kwargs = {"timeout": 30}
            if proxy_url: request_kwargs["proxy"] = proxy_url

            async with session.get(url, **request_kwargs) as response:
                if response.status != 200:
                    self.logger.warning(f"Failed to fetch {url}. Status: {response.status}")
                    return None
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                country_updates = []
                # Add specific parsing logic per source type (this needs refinement)
                if "federalregister.gov" in url:
                    # Example parsing for Federal Register (adjust selectors)
                    for item in soup.select('div.document-wrapper h5 a'): # Example selector
                        title = item.text.strip()
                        link = item.get('href')
                        if title and link: country_updates.append((title, response.url.join(link).human_repr())) # Use absolute URL
                elif "sgg.gov.ma" in url:
                     # Example parsing for Morocco SGG (adjust selectors)
                     for item in soup.select('.views-row .field-content a'): # Example selector
                         title = item.text.strip()
                         link = item.get('href')
                         if title and link: country_updates.append((title, response.url.join(link).human_repr()))
                # Add parsers for other sources...
                else:
                     # Generic link extraction as fallback
                     for a in soup.find_all('a', href=True):
                          title = a.text.strip()
                          link = a['href']
                          if title and link.startswith('http'): # Basic filter
                               country_updates.append((title, link))

                self.logger.debug(f"Found {len(country_updates)} potential items from {url}")
                return country, country_updates
        except Exception as e:
            self.logger.error(f"Error fetching/parsing {url}: {e}", exc_info=False) # Log less verbosely
            return None


    async def interpret_updates(self, updates: Dict[str, List[Tuple[str, str]]]) -> List[Dict[str, Any]]:
        """Interpret fetched legal update titles/links using LLM and store in KB."""
        interpretations = []
        if not updates: return interpretations

        # --- Filter out already processed updates using KB ---
        processed_hashes = set()
        # TODO: Query KB for recent 'legal_update_interpretation' fragments based on source_reference (URL) or content hash
        # For now, assume all fetched are new for simplicity

        texts_to_process = [] # List of tuples: (country, title, url)
        for country, items in updates.items():
            for title, url in items:
                # Basic filtering
                if not title or len(title) < 10: continue
                # TODO: Add more sophisticated filtering (keywords, relevance check)
                texts_to_process.append((country, title, url))

        if not texts_to_process:
            self.logger.info("No new, relevant legal update titles require interpretation.")
            return interpretations

        self.logger.info(f"Requesting interpretation for {len(texts_to_process)} new legal update titles.")

        # --- Batch Interpretation via LLM ---
        # Prepare context and generate dynamic prompt
        task_context = {
            "task": "Interpret legal update titles/links",
            "updates_list": [{"country": c, "title": t, "url": u} for c, t, u in texts_to_process],
            "desired_output_format": "JSON array, where each item corresponds to an input update and has keys: 'url', 'country', 'category' (e.g., Tax, AI, Compliance, Corporate), 'interpretation_summary' (brief summary of likely impact), 'relevance_score' (0.0-1.0), 'grey_areas_identified' (list[str]), 'compliance_note' (concise note)."
        }
        interpretation_prompt = await self.generate_dynamic_prompt(task_context)

        await self._internal_think(f"Calling LLM to interpret {len(texts_to_process)} legal update titles.")
        llm_response_json = await self.think_tool._call_llm_with_retry(
            interpretation_prompt,
            model=self.config.get("OPENROUTER_MODELS", {}).get('legal_interpretation', "google/gemini-1.5-pro-latest"),
            temperature=0.3, max_tokens=self.internal_state['max_interpretation_tokens'], is_json_output=True
        )

        if not llm_response_json:
            self.logger.error("LLM call for legal interpretation returned empty response.")
            return interpretations # Return empty list on failure

        # Parse and store results
        try:
            json_match = json.loads(llm_response_json[llm_response_json.find('['):llm_response_json.rfind(']')+1]) # Expecting array
            if not isinstance(json_match, list) or len(json_match) != len(texts_to_process):
                 raise ValueError(f"LLM interpretation result is not a list or length mismatch. Expected {len(texts_to_process)}.")

            for result_data in json_match:
                if isinstance(result_data, dict):
                    # Store interpretation in KB
                    await self.log_knowledge_fragment(
                        agent_source=self.AGENT_NAME,
                        data_type="legal_update_interpretation",
                        content=result_data, # Store the structured interpretation
                        relevance_score=result_data.get('relevance_score', 0.5),
                        tags=["legal_update", "interpretation", result_data.get('country', 'unknown').lower(), result_data.get('category', 'general').lower()],
                        source_reference=result_data.get('url') # Link back to the source URL
                    )
                    interpretations.append(result_data)
                else:
                    self.logger.warning(f"Invalid result structure in interpretation list: {result_data}")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"Failed to parse or validate LLM interpretation response: {e}. Response: {llm_response_json}")
        except Exception as e:
             self.logger.error(f"Unexpected error processing LLM interpretation results: {e}", exc_info=True)

        self.logger.info(f"Interpretation complete. Stored {len(interpretations)} new interpretations in KB.")
        return interpretations

    # --- Validation ---
    async def validate_operation(self, operation_description: str) -> dict:
        """Validate operations using internal logic, KB context, and a single LLM call."""
        self.logger.debug(f"Validating operation: {operation_description[:100]}...")
        fallback_result = {'is_compliant': False, 'risk_score': 1.0, 'compliance_issues': ['Analysis Error'], 'gray_area_exploitation': False, 'potential_consequences': [], 'recommendations': [], 'proceed_recommendation': 'Halt - Analysis Failed'}

        try:
            # 1. Fetch relevant context from KB
            await self._internal_think(f"Querying KB for context relevant to operation: {operation_description[:60]}...")
            # Derive tags from description (simple keyword extraction)
            keywords = [w for w in re.findall(r'\b\w{3,}\b', operation_description.lower()) if w not in ['the', 'a', 'is', 'for', 'to', 'in']]
            kb_context_frags = await self.query_knowledge_base(
                data_types=['legal_analysis', 'regulation_summary', 'compliance_note', 'grey_area_analysis', 'learned_pattern'],
                tags=list(set(['compliance', 'risk'] + keywords[:5])), # Use derived keywords + base tags
                limit=15, min_relevance=0.6 # Fetch relevant, reasonably confident items
            )
            kb_context_str = "\n".join([f"- {f.data_type} (ID {f.id}, Rel {f.relevance_score:.2f}): {f.content[:150]}..." for f in kb_context_frags])

            # 2. Prepare context and generate dynamic prompt
            task_context = {
                "task": "Validate operation compliance and risk",
                "operation_description": operation_description,
                "knowledge_base_context": kb_context_str or "No specific KB context found.",
                "desired_output_format": "JSON object with keys: 'is_compliant' (bool), 'risk_score' (float 0.0-1.0), 'compliance_issues' (list[str]), 'gray_area_exploitation' (bool), 'potential_consequences' (list[str]), 'recommendations' (list[str]), 'proceed_recommendation' (str 'Proceed'|'Proceed with Caution'|'Halt - High Risk'|'Halt - Non-Compliant')."
            }
            validation_prompt = await self.generate_dynamic_prompt(task_context)

            # 3. Make single LLM call
            await self._internal_think(f"Calling LLM for operation validation: {operation_description[:60]}...")
            validation_result_json = await self.think_tool._call_llm_with_retry(
                validation_prompt,
                model=self.config.get("OPENROUTER_MODELS", {}).get('legal_validation', "google/gemini-1.5-pro-latest"),
                temperature=0.2, max_tokens=self.internal_state['max_validation_tokens'], is_json_output=True
            )

            if not validation_result_json: raise Exception("LLM call for validation returned empty response.")

            # 4. Parse and process result
            try:
                result = json.loads(validation_result_json[validation_result_json.find('{'):validation_result_json.rfind('}')+1])
                # Add default values for robustness
                result.setdefault('is_compliant', False); result.setdefault('risk_score', 1.0)
                result.setdefault('compliance_issues', ['Analysis Failed']); result.setdefault('gray_area_exploitation', False)
                result.setdefault('potential_consequences', []); result.setdefault('recommendations', [])
                result.setdefault('proceed_recommendation', 'Halt - Analysis Failed')

                self.logger.info(f"Legal Validation for '{operation_description[:50]}...': Compliant={result['is_compliant']}, Risk Score={result['risk_score']:.2f}, Recommendation={result['proceed_recommendation']}")

                # Trigger notifications/errors via Orchestrator
                if hasattr(self.orchestrator, 'report_error') and (not result['is_compliant'] or result['risk_score'] > 0.8):
                    await self.orchestrator.report_error(self.AGENT_NAME, f"Operation blocked/high-risk: {operation_description[:100]}... Issues: {result['compliance_issues']}, Risk: {result['risk_score']:.2f}")
                elif hasattr(self.orchestrator, 'send_notification') and (result['risk_score'] > 0.5 or result.get('recommendations')):
                     await self.orchestrator.send_notification("LegalCompliance Advisory", f"Operation: {operation_description[:100]}... Risk: {result['risk_score']:.2f}. Recommendations: {result['recommendations']}")

                return result
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                 self.logger.error(f"LegalAgent: Failed to parse JSON validation: {e}. Response: {validation_result_json}")
                 fallback_result['compliance_issues'] = ['LLM Error/Invalid JSON']
                 return fallback_result
            except Exception as parse_exc:
                 self.logger.error(f"LegalAgent: Error parsing validation result: {parse_exc}")
                 fallback_result['compliance_issues'] = [f'Parsing Error: {parse_exc}']
                 return fallback_result

        except Exception as e:
            self.logger.error(f"{self.AGENT_NAME}: Error during validate_operation: {e}", exc_info=True)
            fallback_result['compliance_issues'] = [f'Analysis Error: {e}']
            return fallback_result

    # --- Utility Methods ---

    async def get_invoice_legal_note(self, client_country: str) -> str:
        """Generates a standardized legal note for invoices, potentially customized by country."""
        self.logger.debug(f"Generating invoice legal note for country: {client_country}")
        base_note = "Compliant with standard financial regulations." # Default
        try:
            # Query KB for country-specific compliance notes
            await self._internal_think(f"Querying KB for compliance notes relevant to invoices in {client_country}.")
            compliance_notes = await self.query_knowledge_base(
                data_types=['compliance_note', 'legal_update_interpretation'],
                tags=['invoice', 'financial', 'tax', client_country.lower()],
                limit=5, min_relevance=0.7
            )
            if compliance_notes:
                # Use the most relevant note found
                # Simple logic: use the first one. Could be enhanced by LLM summarization.
                note_content = compliance_notes[0].content
                # Extract the note part if it's a full interpretation dict
                if isinstance(note_content, str) and note_content.startswith('{'):
                     try: base_note = json.loads(note_content).get('compliance_note', base_note)
                     except json.JSONDecodeError: pass # Keep default if parsing fails
                elif isinstance(note_content, str):
                     base_note = note_content # Assume it's just the note string
                self.logger.debug(f"Using compliance note from KB: {base_note}")

        except Exception as e:
            self.logger.error(f"Error fetching compliance note from KB for {client_country}: {e}")

        # Append standard "bulletproof" terms (consider making these configurable)
        user_name = getattr(self.config, 'USER_NAME', 'The Agency Operator') # Get user name from config
        bulletproof_terms = (
            "This agreement is irrevocable and binding upon acceptance. "
            "Agency shall not be liable for any indirect, incidental, or consequential damages, including loss of profits or data. "
            "All disputes shall be resolved exclusively under Moroccan law in the courts of Rabat, Morocco. " # Specify city
            f"Payment to {user_name}'s designated account is non-refundable and non-cancellable once services commence. "
            "Client acknowledges adherence to ISO 20022 standards for secure international transactions where applicable."
        )
        return f"{base_note} | {bulletproof_terms}"

    async def get_w8_data(self) -> Optional[Dict[str, str]]:
        """Retrieves W8-BEN data securely."""
        if not self.secure_storage: return None
        try:
            w8_data_str = await self.secure_storage.get_secret("w8ben_data")
            if w8_data_str: return json.loads(w8_data_str)
        except Exception as e:
            self.logger.error(f"Failed to get or parse W8-BEN data: {e}")
        return None

    async def validate_w8_data(self) -> bool:
        """Validates if essential W8-BEN data is present."""
        w8_data = await self.get_w8_data()
        if not w8_data: return False
        required_fields = ["name", "country", "address", "tin"]
        is_complete = all(field in w8_data and w8_data[field] for field in required_fields)
        if not is_complete: self.logger.warning("W-8BEN data validation failed: Missing required fields.")
        return is_complete

    # --- Learning Loop Implementation ---
    async def learning_loop(self):
        """Periodic loop to fetch and interpret legal updates."""
        while True:
            try:
                interval = self.internal_state.get('update_interval_seconds', 604800)
                await asyncio.sleep(interval)
                self.logger.info("Executing LegalAgent learning loop (fetch & interpret)...")
                self.internal_state['last_scan_time'] = datetime.now(timezone.utc)

                # Fetch updates from configured sources
                updates = await self.fetch_legal_updates()
                if updates:
                    # Interpret updates using internal logic (which stores results in KB)
                    interpretations = await self.interpret_updates(updates)

                    # Notify orchestrator about new grey area opportunities found
                    # Check internal state which interpret_updates modifies
                    # for country, strategies in self.internal_state.get('gray_area_strategies', {}).items():
                    #      if strategies and hasattr(self.orchestrator, 'send_notification'):
                    #           # TODO: Only notify about *newly added* strategies
                    #           await self.orchestrator.send_notification("GrayAreaOpportunity", f"Potential exploit(s) identified/updated for {country}: {strategies}")

                    self.logger.info(f"Legal update interpretation cycle complete. Processed {len(interpretations)} interpretations.")
                else:
                    self.logger.info("No new legal updates fetched in this cycle.")

            except asyncio.CancelledError:
                self.logger.info("LegalAgent learning loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error during LegalAgent learning loop: {e}", exc_info=True)
                if hasattr(self.orchestrator, 'report_error'):
                     await self.orchestrator.report_error(self.AGENT_NAME, f"Learning loop error: {e}")
                await asyncio.sleep(3600) # Wait longer after error

    # --- Self Critique ---
    async def self_critique(self) -> Dict[str, Any]:
        """Evaluates the agent's own performance and strategy."""
        self.logger.info(f"{self.AGENT_NAME}: Performing self-critique.")
        critique = {"status": "ok", "feedback": "Critique pending analysis."}
        # TODO: Analyze KB query success rates, validation accuracy (if feedback mechanism exists),
        # timeliness of updates, utility of generated invoice notes.
        critique['last_scan_time'] = self.internal_state.get('last_scan_time')
        # critique['kb_query_success_rate'] = ...
        # critique['validation_accuracy'] = ...
        critique['feedback'] = f"Last scan: {self.internal_state.get('last_scan_time')}. Further critique metrics not yet implemented."
        return critique

    # --- Dynamic Prompt Generation ---
    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Constructs context-rich prompts for LLM calls."""
        self.logger.debug(f"Generating dynamic prompt for LegalAgent task: {task_context.get('task')}")
        prompt_parts = [self.internal_state.get('meta_prompt', LEGAL_AGENT_META_PROMPT)]

        prompt_parts.append("\n--- Current Task Context ---")
        # Add specific task details, excluding large context blocks initially
        for key, value in task_context.items():
            if key not in ['knowledge_base_context', 'updates_list', 'recent_legal_context', 'legal_texts']:
                 if isinstance(value, (str, int, float, bool)):
                     prompt_parts.append(f"{key.replace('_', ' ').title()}: {value}")
                 elif isinstance(value, dict): # Add small dicts
                      if len(json.dumps(value)) < 200: prompt_parts.append(f"{key.replace('_', ' ').title()}: {json.dumps(value)}")
                 elif isinstance(value, list): # Add small lists
                      if len(json.dumps(value)) < 200: prompt_parts.append(f"{key.replace('_', ' ').title()}: {json.dumps(value)}")


        # Add KB context if provided
        if 'knowledge_base_context' in task_context and task_context['knowledge_base_context']:
            prompt_parts.append("\n--- Relevant Knowledge Base Context (Summaries) ---")
            prompt_parts.append(task_context['knowledge_base_context'][:2000] + "...") # Limit length
        elif 'recent_legal_context' in task_context and task_context['recent_legal_context']:
             prompt_parts.append("\n--- Recent Legal Interpretations (Cache/KB) ---")
             context_str = json.dumps(task_context['recent_legal_context'], indent=2, default=str)
             prompt_parts.append(context_str[:2000] + "...") # Limit length

        # Add list of updates if provided
        if 'updates_list' in task_context and task_context['updates_list']:
             prompt_parts.append("\n--- Legal Updates to Interpret ---")
             updates_str = json.dumps(task_context['updates_list'], indent=2)
             prompt_parts.append(updates_str[:3000] + "...") # Limit length

        # Add Specific Instructions based on task
        prompt_parts.append("\n--- Instructions ---")
        task_type = task_context.get('task')
        if task_type == 'Analyze and recommend optimal initial corporate structure':
            prompt_parts.append("1. Analyze pros/cons/risks/steps for Wyoming LLC, Delaware LLC, and potentially one other relevant structure based on context.")
            prompt_parts.append("2. Consider tax efficiency, liability, privacy, compliance overhead, and grey-area potential.")
            prompt_parts.append("3. Factor in the provided business context and KB information.")
            prompt_parts.append("4. Provide a clear final recommendation with rationale.")
            prompt_parts.append(f"5. **Output Format:** {task_context.get('desired_output_format')}")
        elif task_type == 'Deep analysis of potential grey area strategy':
             prompt_parts.append(f"1. Deeply analyze the legal basis, risks (legal, reputational, operational), benefits, and implementation plan for the grey area: '{task_context.get('grey_area_topic', 'N/A')}' in the context of '{task_context.get('application_context', 'N/A')}'.")
             prompt_parts.append("2. Reference relevant statutes, loopholes, precedents from KB or general knowledge.")
             prompt_parts.append("3. Provide actionable implementation steps, monitoring needs, and fallback plans.")
             prompt_parts.append("4. Conclude with an overall recommendation (Proceed, Caution, Avoid, Research).")
             prompt_parts.append(f"5. **Output Format:** {task_context.get('desired_output_format')}")
        elif task_type == 'Interpret legal update titles/links':
             prompt_parts.append("1. For each item in 'Legal Updates to Interpret':")
             prompt_parts.append("   - Determine the relevant law category (Tax, AI, Compliance, Corporate, etc.).")
             prompt_parts.append("   - Summarize the likely core interpretation and impact on AI agency operations.")
             prompt_parts.append("   - Assess relevance score (0.0-1.0).")
             prompt_parts.append("   - Identify potential ambiguities or 'grey areas'.")
             prompt_parts.append("   - Generate a concise compliance note.")
             prompt_parts.append(f"2. **Output Format:** {task_context.get('desired_output_format')}")
        elif task_type == 'Validate operation compliance and risk':
             prompt_parts.append("1. Analyze the 'Operation Description' against legal/compliance knowledge (from KB and general).")
             prompt_parts.append("2. Focus on USA/Moroccan law. Assume high risk tolerance for profit, zero for severe penalties.")
             prompt_parts.append("3. Assess compliance, quantify risk (0.0-1.0), list issues & consequences.")
             prompt_parts.append("4. Note grey area exploitation. Provide mitigation recommendations.")
             prompt_parts.append("5. Give explicit proceed recommendation.")
             prompt_parts.append(f"6. **Output Format:** {task_context.get('desired_output_format')}")
        else:
            prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")

        # Add JSON hint if needed
        if "JSON" in task_context.get('desired_output_format', ''):
             prompt_parts.append("\n```json")

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for LegalAgent (length: {len(final_prompt)} chars)")
        # self.logger.debug(f"Prompt Preview:\n{final_prompt[:500]}...") # Optional debug
        return final_prompt

    # --- KB Interaction Helpers (Delegate to ThinkTool/KBInterface or implement here) ---
    # These methods provide a consistent interface for KB operations within this agent.
    # They can either call self.kb_interface directly or delegate to ThinkTool via orchestrator.

    async def log_knowledge_fragment(self, agent_source: str, data_type: str, content: Union[str, dict], relevance_score: float = 0.5, tags: Optional[List[str]] = None, related_client_id: Optional[int] = None, source_reference: Optional[str] = None) -> Optional[KnowledgeFragment]:
        """Logs a knowledge fragment."""
        if self.kb_interface and hasattr(self.kb_interface, 'log_knowledge_fragment'):
            return await self.kb_interface.log_knowledge_fragment(agent_source, data_type, content, relevance_score, tags, related_client_id, source_reference)
        elif self.think_tool and hasattr(self.think_tool, 'log_knowledge_fragment'):
            # Delegate to ThinkTool if KB interface isn't directly available
            return await self.think_tool.log_knowledge_fragment(agent_source, data_type, content, relevance_score, tags, related_client_id, source_reference)
        else:
            self.logger.error("No mechanism available (KBInterface or ThinkTool) to log knowledge fragment.")
            return None # Cannot log

    async def query_knowledge_base(self, data_types: Optional[List[str]] = None, tags: Optional[List[str]] = None, min_relevance: float = 0.0, time_window: Optional[timedelta] = None, limit: int = 100, related_client_id: Optional[int] = None) -> List[KnowledgeFragment]:
        """Queries the knowledge base."""
        if self.kb_interface and hasattr(self.kb_interface, 'query_knowledge_base'):
            return await self.kb_interface.query_knowledge_base(data_types, tags, min_relevance, time_window, limit, related_client_id)
        elif self.think_tool and hasattr(self.think_tool, 'query_knowledge_base'):
            # Delegate to ThinkTool
            return await self.think_tool.query_knowledge_base(data_types, tags, min_relevance, time_window, limit, related_client_id)
        else:
            self.logger.error("No mechanism available (KBInterface or ThinkTool) to query knowledge base.")
            return [] # Return empty list

    async def log_learned_pattern(self, pattern_description: str, supporting_fragment_ids: List[int], confidence_score: float, implications: str, tags: Optional[List[str]] = None) -> Optional[LearnedPattern]:
         """Logs a learned pattern."""
         if self.kb_interface and hasattr(self.kb_interface, 'log_learned_pattern'):
             return await self.kb_interface.log_learned_pattern(pattern_description, supporting_fragment_ids, confidence_score, implications, tags)
         elif self.think_tool and hasattr(self.think_tool, 'log_learned_pattern'):
             return await self.think_tool.log_learned_pattern(pattern_description, supporting_fragment_ids, confidence_score, implications, tags)
         else:
             self.logger.error("No mechanism available (KBInterface or ThinkTool) to log learned pattern.")
             return None

    async def get_latest_patterns(self, tags: Optional[List[str]] = None, min_confidence: float = 0.7, limit: int = 10) -> List[LearnedPattern]:
         """Retrieves learned patterns."""
         if self.kb_interface and hasattr(self.kb_interface, 'get_latest_patterns'):
             return await self.kb_interface.get_latest_patterns(tags, min_confidence, limit)
         elif self.think_tool and hasattr(self.think_tool, 'get_latest_patterns'):
             return await self.think_tool.get_latest_patterns(tags, min_confidence, limit)
         else:
             self.logger.error("No mechanism available (KBInterface or ThinkTool) to get latest patterns.")
             return []

# --- End of agents/legal_agent.py ---