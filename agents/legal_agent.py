# Filename: agents/legal_agent.py
# Description: Genius Agentic Legal Agent - Strategic analysis, compliance validation,
#              grey-area risk assessment, regulatory monitoring, KB management.
# Version: 3.1 (Genius Agentic - Enhanced Grey Area Logic)

import asyncio
import logging
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple, Union

# --- Core Framework Imports ---
import aiohttp
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select, update, desc, or_, func
from sqlalchemy.exc import SQLAlchemyError

# --- Project Imports ---
try:
    from .base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
except ImportError:
    logging.warning("Production base agent not found, using GeniusAgentBase. Ensure base_agent_prod.py is used.")
    from .base_agent import GeniusAgentBase

from models import KnowledgeFragment, LearnedPattern, StrategicDirective # Use KB models
from config.settings import settings # Use validated settings

# Configure logger
logger = logging.getLogger(__name__)
# Configure dedicated operational logger
op_logger = logging.getLogger('OperationalLog') # Assuming setup elsewhere

# --- Meta Prompt ---
LEGAL_AGENT_META_PROMPT = """
You are the LegalAgent within the Synapse AI Sales System.
Your Core Mandate: Enable maximum profit and growth by providing strategic legal analysis, proactive compliance guidance, and **calculated risk assessment for grey area tactics**. Operate with Genius Agentic AI principles.
Key Responsibilities:
- **Strategic Risk Assessment:** Analyze proposed operations (outreach campaigns, multi-accounting, trial rotation) not just for literal ToS/legal violations, but for *actual detection likelihood* and *potential consequences* vs. *business upside*. Provide actionable risk levels (Low, Medium, High, Critical) and mitigation strategies.
- **Grey Area Enablement:** Identify loopholes, regulatory arbitrage, and "bendable" rules. Advise ThinkTool on how to leverage these opportunities *within acceptable risk parameters*. Focus on jurisdictions like USA/Morocco, generally avoiding high-regulation zones unless strategically vital.
- **Compliance Validation:** Proactively validate agent actions (emails, calls, data handling) against current regulations (CAN-SPAM, GDPR, TCPA, local laws) stored in the KB. Provide clear go/no-go recommendations with reasoning.
- **Regulatory Monitoring:** Periodically scan configured legal sources for updates relevant to AI, sales, marketing, data privacy, and corporate structure. Interpret updates and log actionable insights/compliance notes to the KB.
- **Knowledge Management:** Maintain the legal section of the KB (Postgres via ThinkTool/direct session), storing analyses, precedents, compliance notes, and risk assessments. Avoid redundant analysis by querying KB first.
- **Corporate Structure:** Analyze and recommend optimal, tax-efficient structures (LLCs etc.) based on operational goals and risk tolerance.
- **Collaboration:** Communicate findings clearly to ThinkTool, Orchestrator, and other agents as needed. Generate User Education content on compliance matters.
**Goal:** Act as a strategic legal partner, enabling aggressive but calculated business growth by navigating legal complexities and identifying advantageous grey areas, while ensuring catastrophic legal/financial risks are avoided. Prioritize profit within *managed* risk.
"""

class LegalAgent(GeniusAgentBase):
    """
    Legal Agent (Genius Level): Handles strategic legal analysis, compliance validation,
    grey-area risk assessment, regulatory monitoring, and KB management.
    Version: 3.1
    """
    AGENT_NAME = "LegalAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any):
        """Initializes the LegalAgent."""
        # ### Phase 4 Plan Ref: 9.1 (Implement __init__)
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, session_maker=session_maker)
        self.meta_prompt = LEGAL_AGENT_META_PROMPT
        self.think_tool = orchestrator.agents.get('think') # Reference ThinkTool

        # --- Internal State Initialization ---
        self.internal_state = getattr(self, 'internal_state', {})
        self.internal_state['legal_sources'] = self.config.get("LEGAL_SOURCES", {})
        self.internal_state['update_interval_seconds'] = int(self.config.get("LEGAL_UPDATE_INTERVAL_SECONDS", 604800)) # 1 week
        self.internal_state['max_interpretation_tokens'] = 1500
        self.internal_state['max_validation_tokens'] = 1200 # Increased slightly
        self.internal_state['max_analysis_tokens'] = 2500 # Increased slightly
        self.internal_state['last_scan_time'] = None

        self.logger.info(f"{self.AGENT_NAME} v3.1 (Enhanced Grey Area) initialized.")

    async def log_operation(self, level: str, message: str):
        """Helper to log to the operational log file."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err: logger.error(f"Failed to write to operational log: {log_err}")

    # --- Core Task Execution ---
    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a legal analysis or validation task."""
        # ### Phase 4 Plan Ref: 9.2 (Implement execute_task)
        action = task_details.get('action', 'analyze')
        description = task_details.get('description', f"Performing legal action: {action}")
        self.logger.info(f"{self.AGENT_NAME} starting task: {description}")
        self._status = self.STATUS_EXECUTING
        result = {"status": "failure", "message": f"Unknown or unimplemented legal action: {action}"}
        findings = None # Use 'findings' key for structured results

        try:
            if action == "analyze_initial_structure":
                findings = await self._analyze_initial_structure(task_details)
            elif action == "scan_for_updates":
                findings = await self._scan_for_updates(task_details)
            elif action == "analyze_grey_area":
                findings = await self._analyze_grey_area(task_details)
            elif action == "validate_operation":
                op_desc = task_details.get('operation_description')
                if not op_desc: raise ValueError("Missing 'operation_description' for validate_operation")
                # validate_operation now returns the full result dict including 'findings'
                result = await self.validate_operation(op_desc)
                # Don't overwrite result below, return directly
                self._status = self.STATUS_IDLE
                self.logger.info(f"{self.AGENT_NAME} completed task: {description}. Status: {result.get('status', 'failure')}")
                return result
            elif action == "get_invoice_note":
                 country = task_details.get('client_country')
                 if not country: raise ValueError("Missing 'client_country' for get_invoice_note")
                 note = await self.get_invoice_legal_note(country)
                 findings = {"invoice_note": note}
            # Add other actions LegalAgent might handle
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
            await self._report_error(f"Task '{description}' failed: {e}")
        finally:
            self._status = self.STATUS_IDLE

        return result

    # --- Strategic Analysis Methods ---

    async def _analyze_initial_structure(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Researches and recommends optimal initial corporate structure."""
        # (Keep implementation similar to previous version, ensure KB queries work)
        business_context = task_details.get("business_context", "AI UGC Agency targeting US/Global clients.")
        self.logger.info(f"Analyzing optimal initial corporate structure for context: {business_context}")
        await self._internal_think("Querying KB for existing structure analyses (Wyoming, Delaware) and tax/liability regulations.")
        kb_context_frags = await self.query_knowledge_base(data_types=['legal_analysis', 'regulation_summary'], tags=['corporate_structure', 'llc', 'tax', 'liability', 'wyoming', 'delaware'], limit=10)
        kb_context_str = "\n".join([f"- {f.data_type} (ID {f.id}): {f.content[:150]}..." for f in kb_context_frags])
        task_context = {
            "task": "Analyze and recommend optimal initial corporate structure", "business_context": business_context,
            "goals": "Maximize tax efficiency, minimize liability, identify grey-area advantages, avoid Europe if possible.",
            "jurisdictions_to_consider": ["Wyoming LLC", "Delaware LLC", "Suggest one other if compelling"],
            "knowledge_base_context": kb_context_str or "No specific KB context found.",
            "desired_output_format": "JSON: { \"analysis_summary\": str, \"recommendations\": [ { \"jurisdiction\": str, \"structure\": str, \"pros\": [str], \"cons\": [str], \"risks\": [str], \"setup_steps\": [str], \"grey_area_notes\": str } ], \"final_recommendation\": { \"jurisdiction\": str, \"structure\": str, \"rationale\": str } }"
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        await self._internal_think("Calling LLM for initial structure analysis.")
        llm_response_json = await self._call_llm_with_retry(prompt, model=self.config.OPENROUTER_MODELS.get('legal_analysis'), max_tokens=self.internal_state['max_analysis_tokens'], is_json_output=True)
        if not llm_response_json: raise RuntimeError("LLM call failed.")
        try:
            analysis_result = self._parse_llm_json(llm_response_json)
            if not analysis_result.get("recommendations") or not analysis_result.get("final_recommendation"): raise ValueError("LLM response missing required keys.")
            await self._internal_think(f"Storing structure analysis result in KB. Recommendation: {analysis_result.get('final_recommendation', {}).get('structure')} in {analysis_result.get('final_recommendation', {}).get('jurisdiction')}")
            await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="legal_analysis", content=analysis_result, tags=["corporate_structure", "initial_setup"] + [rec.get('jurisdiction', '').lower() for rec in analysis_result.get('recommendations', [])], relevance_score=0.9)
            # Trigger User Education
            if hasattr(self.orchestrator, 'handle_user_education_trigger'):
                 edu_topic = f"Initial Structure Rec: {analysis_result['final_recommendation']['structure']} in {analysis_result['final_recommendation']['jurisdiction']}"
                 edu_context = analysis_result['final_recommendation']['rationale']
                 asyncio.create_task(self.orchestrator.handle_user_education_trigger(edu_topic, edu_context))
            return analysis_result
        except (json.JSONDecodeError, ValueError, KeyError) as e: raise RuntimeError(f"LLM response parsing failed: {e}. Response: {llm_response_json[:500]}")

    async def _analyze_grey_area(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Performs deep analysis of a specific potential grey area strategy."""
        # ### Phase 4 Plan Ref: 9.3 (Implement _analyze_grey_area)
        area = task_details.get("area", "Unspecified grey area opportunity")
        context = task_details.get("context", "Evaluate for AI UGC Agency operations.")
        self.logger.info(f"Analyzing grey area opportunity: {area}")
        await self._internal_think(f"Querying KB for existing analyses/regulations related to grey area: {area}")
        kb_context_frags = await self.query_knowledge_base(data_types=['legal_analysis', 'regulation_summary', 'grey_area_analysis'], tags=['grey_area', area.replace(" ", "_").lower()] + context.split()[:3], limit=15)
        kb_context_str = "\n".join([f"- {f.data_type} (ID {f.id}): {f.content[:150]}..." for f in kb_context_frags])
        task_context = {
            "task": "Deep analysis of potential grey area strategy", "grey_area_topic": area, "application_context": context,
            "knowledge_base_context": kb_context_str or "No specific KB context found.",
            "desired_output_format": "JSON: { \"area\": str, \"strategy_overview\": str, \"legal_basis_analysis\": { \"statutes_exploited\": [str], \"loopholes\": [str], \"precedents\": [str], \"legal_opinion_confidence\": float (0.0-1.0) }, \"risk_assessment\": { \"detection_likelihood\": str (Low/Medium/High), \"potential_penalties\": [str], \"reputational_risk\": str (Low/Medium/High), \"operational_complexity\": str (Low/Medium/High) }, \"implementation_plan\": [ { \"step\": int, \"action\": str, \"details\": str } ], \"monitoring_requirements\": [str], \"fallback_strategy\": str, \"overall_recommendation\": str ('Proceed', 'Caution', 'Avoid', 'Research') }"
        } # Added detection_likelihood to risk_assessment
        prompt = await self.generate_dynamic_prompt(task_context)
        await self._internal_think(f"Calling LLM for deep grey area analysis: {area}")
        llm_response_json = await self._call_llm_with_retry(prompt, model=self.config.OPENROUTER_MODELS.get('legal_analysis'), max_tokens=self.internal_state['max_analysis_tokens'], is_json_output=True)
        if not llm_response_json: raise RuntimeError(f"LLM call for grey area analysis ({area}) failed.")
        try:
            analysis_result = self._parse_llm_json(llm_response_json)
            if not analysis_result.get("strategy_overview") or not analysis_result.get("risk_assessment"): raise ValueError("LLM response missing required keys.")
            await self._internal_think(f"Storing grey area analysis for '{area}' in KB. Recommendation: {analysis_result.get('overall_recommendation', 'N/A')}")
            await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="grey_area_analysis", content=analysis_result, tags=["grey_area", "legal_analysis", "risk_assessment"] + area.lower().split(), relevance_score=analysis_result.get("risk_assessment", {}).get("legal_opinion_confidence", 0.7))
            # Trigger directive based on recommendation
            if "proceed" in analysis_result.get('overall_recommendation', '').lower() or "caution" in analysis_result.get('overall_recommendation', '').lower():
                 async with self.session_maker() as session:
                      async with session.begin():
                           session.add(StrategicDirective(source=self.AGENT_NAME, target_agent="ThinkTool", directive_type="consider_grey_area_strategy", content=json.dumps({"area": area, "analysis_summary": analysis_result.get("strategy_overview"), "risk": analysis_result.get("risk_assessment")}), priority=6, status='pending'))
                           self.logger.info(f"Generated directive for ThinkTool to consider grey area: {area}")
            return analysis_result
        except (json.JSONDecodeError, ValueError, KeyError) as e: raise RuntimeError(f"LLM response parsing failed for grey area analysis: {e}. Response: {llm_response_json[:500]}")

    # --- Monitoring & Interpretation ---

    async def _scan_for_updates(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Fetches and interprets legal updates."""
        # ### Phase 4 Plan Ref: 9.4 (Implement _scan_for_updates)
        await self._internal_think("Starting legal update scan: Fetching sources.")
        updates = await self.fetch_legal_updates()
        if not updates: return {"status": "success", "message": "No new updates found or sources unavailable.", "interpretations": []}
        await self._internal_think(f"Fetched {sum(len(v) for v in updates.values())} potential updates. Proceeding to interpretation.")
        interpretations = await self.interpret_updates(updates)
        await self._internal_think(f"Interpretation complete. Stored {len(interpretations)} new interpretations in KB.")
        self.internal_state['last_scan_time'] = datetime.now(timezone.utc)
        return {"status": "success", "message": f"Scan complete. Processed {len(interpretations)} interpretations.", "interpretations": interpretations}

    async def fetch_legal_updates(self) -> Dict[str, List[Tuple[str, str]]]:
        """Asynchronously fetch legal update titles/links from configured sources."""
        # (Keep implementation similar to previous version, ensure proxy usage via orchestrator)
        updates: Dict[str, List[Tuple[str, str]]] = {}
        sources = self.internal_state.get('legal_sources', {})
        if not sources: self.logger.warning("No legal sources configured."); return updates
        async with aiohttp.ClientSession() as session:
            fetch_tasks = [self._fetch_single_source(session, country, url) for country, urls in sources.items() for url in urls]
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception): self.logger.error(f"Error fetching legal source: {result}")
                elif result:
                    country, country_updates = result
                    if country not in updates: updates[country] = []
                    updates[country].extend(country_updates)
        total_fetched = sum(len(v) for v in updates.values())
        self.logger.info(f"Fetched {total_fetched} potential update titles/links from {len(sources)} sources.")
        return updates

    async def _fetch_single_source(self, session: aiohttp.ClientSession, country: str, url: str) -> Optional[Tuple[str, List[Tuple[str, str]]]]:
        """Fetches and parses titles/links from a single URL using proxy."""
        self.logger.debug(f"Fetching updates from: {url} ({country})")
        try:
            proxy_url = await self.orchestrator.get_proxy(purpose="legal_scan", target_url=url) # Request proxy
            request_kwargs = {"timeout": 30, "ssl": False} # Disable SSL verify often needed
            if proxy_url: request_kwargs["proxy"] = proxy_url; self.logger.debug(f"Using proxy: {proxy_url.split('@')[-1]}")

            async with session.get(url, **request_kwargs) as response:
                if response.status != 200: self.logger.warning(f"Failed fetch {url}. Status: {response.status}"); return None
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                country_updates = []
                # Add specific parsing logic per source type (Refined examples)
                if "federalregister.gov" in url:
                    for item in soup.select('div.document-listing h5 a[href^="/documents/"]'): # More specific selector
                        title = item.text.strip(); link = item.get('href')
                        if title and link: country_updates.append((title, response.url.join(link).human_repr()))
                elif "sgg.gov.ma" in url: # Example for Morocco SGG
                     for item in soup.select('td.views-field-title a[href*="bulletin"]'): # Example selector
                         title = item.text.strip(); link = item.get('href')
                         if title and link: country_updates.append((title, response.url.join(link).human_repr()))
                # Add more parsers...
                else: # Generic fallback (less reliable)
                     for a in soup.find_all('a', href=True):
                          title = a.text.strip(); link = a['href']
                          if title and len(title) > 10 and (link.startswith('http') or link.startswith('/')):
                               country_updates.append((title, response.url.join(link).human_repr()))
                self.logger.debug(f"Found {len(country_updates)} potential items from {url}")
                return country, country_updates[:50] # Limit results per source
        except Exception as e: self.logger.error(f"Error fetching/parsing {url}: {e}", exc_info=False); return None

    async def interpret_updates(self, updates: Dict[str, List[Tuple[str, str]]]) -> List[Dict[str, Any]]:
        """Interpret fetched legal update titles/links using LLM and store in KB."""
        # (Keep implementation similar, ensure KB logging works)
        interpretations = []
        if not updates: return interpretations
        texts_to_process = [(c, t, u) for c, items in updates.items() for t, u in items if t and len(t) > 10]
        if not texts_to_process: self.logger.info("No new, relevant legal update titles require interpretation."); return interpretations
        self.logger.info(f"Requesting interpretation for {len(texts_to_process)} new legal update titles.")
        task_context = {
            "task": "Interpret legal update titles/links",
            "updates_list": [{"country": c, "title": t, "url": u} for c, t, u in texts_to_process],
            "desired_output_format": "JSON array, each item: {'url', 'country', 'category', 'interpretation_summary', 'relevance_score', 'grey_areas_identified', 'compliance_note'}."
        }
        interpretation_prompt = await self.generate_dynamic_prompt(task_context)
        await self._internal_think(f"Calling LLM to interpret {len(texts_to_process)} legal update titles.")
        llm_response_json = await self._call_llm_with_retry(interpretation_prompt, model=self.config.OPENROUTER_MODELS.get('legal_interpretation'), max_tokens=self.internal_state['max_interpretation_tokens'], is_json_output=True)
        if not llm_response_json: self.logger.error("LLM call for legal interpretation returned empty."); return interpretations
        try:
            json_match = self._parse_llm_json(llm_response_json, expect_type=list) # Use helper
            if not isinstance(json_match, list): raise ValueError(f"LLM interpretation result is not a list.")
            for result_data in json_match:
                if isinstance(result_data, dict) and result_data.get('url'):
                    await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="legal_update_interpretation", content=result_data, relevance_score=result_data.get('relevance_score', 0.5), tags=["legal_update", "interpretation", result_data.get('country', 'unknown').lower(), result_data.get('category', 'general').lower()], source_reference=result_data.get('url'))
                    interpretations.append(result_data)
                else: self.logger.warning(f"Invalid result structure in interpretation list: {result_data}")
        except (json.JSONDecodeError, ValueError, KeyError) as e: self.logger.error(f"Failed to parse/validate LLM interpretation response: {e}. Response: {llm_response_json[:500]}")
        except Exception as e: self.logger.error(f"Unexpected error processing LLM interpretation results: {e}", exc_info=True)
        self.logger.info(f"Interpretation complete. Stored {len(interpretations)} new interpretations in KB.")
        return interpretations

    # --- Validation ---
    async def validate_operation(self, operation_description: str) -> dict:
        """Validate operations using KB context and LLM, focusing on risk assessment."""
        # ### Phase 4 Plan Ref: 9.3 & 9.5 (Implement validate_operation with grey area focus)
        self.logger.info(f"Validating operation: {operation_description[:100]}...")
        fallback_result = {'status': 'failure', 'message': 'Validation failed', 'findings': {'is_compliant': False, 'risk_level': 'Critical', 'compliance_issues': ['Analysis Error'], 'recommendations': ['Halt operation']}}
        try:
            await self._internal_think(f"Querying KB for context relevant to operation: {operation_description[:60]}...")
            keywords = [w for w in re.findall(r'\b\w{4,}\b', operation_description.lower()) if w not in ['the', 'a', 'is', 'for', 'to', 'in', 'and', 'with']]
            kb_context_frags = await self.query_knowledge_base(data_types=['legal_analysis', 'regulation_summary', 'compliance_note', 'grey_area_analysis', 'learned_pattern'], tags=list(set(['compliance', 'risk'] + keywords[:5])), limit=15, min_relevance=0.6)
            kb_context_str = "\n".join([f"- {f.data_type} (ID {f.id}, Rel {f.relevance_score:.2f}): {f.content[:150]}..." for f in kb_context_frags])
            task_context = {
                "task": "Validate operation compliance and assess strategic risk",
                "operation_description": operation_description,
                "knowledge_base_context": kb_context_str or "No specific KB context found.",
                "desired_output_format": "JSON object: {'is_compliant': bool (Strict legal compliance), 'risk_level': str ('Low'|'Medium'|'High'|'Critical' - considering detection & consequences), 'compliance_issues': [str] (Specific rules potentially violated), 'grey_area_assessment': str (Analysis of how this fits grey area strategy, if applicable), 'potential_consequences': [str] (e.g., 'Account ban', 'Fine', 'Legal action'), 'mitigation_recommendations': [str], 'proceed_recommendation': str ('Proceed'|'Proceed with Caution'|'Halt - High Risk'|'Halt - Non-Compliant')}"
            }
            validation_prompt = await self.generate_dynamic_prompt(task_context)
            await self._internal_think(f"Calling LLM for operation validation: {operation_description[:60]}...")
            validation_result_json = await self._call_llm_with_retry(validation_prompt, model=self.config.OPENROUTER_MODELS.get('legal_validation'), temperature=0.2, max_tokens=self.internal_state['max_validation_tokens'], is_json_output=True)
            if not validation_result_json: raise Exception("LLM call for validation returned empty.")
            try:
                findings = self._parse_llm_json(validation_result_json)
                # Add default values for robustness
                findings.setdefault('is_compliant', False); findings.setdefault('risk_level', 'Critical')
                findings.setdefault('compliance_issues', ['Analysis Failed']); findings.setdefault('grey_area_assessment', 'N/A')
                findings.setdefault('potential_consequences', []); findings.setdefault('mitigation_recommendations', [])
                findings.setdefault('proceed_recommendation', 'Halt - Analysis Failed')
                self.logger.info(f"Legal Validation: Compliant={findings['is_compliant']}, Risk={findings['risk_level']}, Recommendation={findings['proceed_recommendation']}")
                # Log the validation result itself
                await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="operation_validation", content=findings, tags=["validation", "compliance", findings['risk_level'].lower()], relevance_score=0.8, source_reference=f"Operation: {operation_description[:50]}...")
                # Trigger notifications based on risk/compliance
                if not findings['is_compliant'] or findings['risk_level'] in ['High', 'Critical']:
                     await self.orchestrator.send_notification(f"Compliance/Risk Alert: {findings['risk_level']}", f"Operation blocked/high-risk: {operation_description[:100]}... Issues: {findings['compliance_issues']}, Risk: {findings['risk_level']}")
                elif findings['risk_level'] == 'Medium' or findings.get('mitigation_recommendations'):
                     await self.orchestrator.send_notification("LegalCompliance Advisory", f"Operation: {operation_description[:100]}... Risk: {findings['risk_level']}. Recommendations: {findings['mitigation_recommendations']}")
                return {"status": "success", "message": "Validation complete.", "findings": findings}
            except (json.JSONDecodeError, ValueError, KeyError) as e: raise RuntimeError(f"LLM validation response parsing failed: {e}. Response: {validation_result_json[:500]}")
        except Exception as e:
            self.logger.error(f"{self.AGENT_NAME}: Error during validate_operation: {e}", exc_info=True)
            await self._report_error(f"Validation failed: {e}")
            return fallback_result

    # --- Utility Methods ---

    async def get_invoice_legal_note(self, client_country: str) -> str:
        """Generates a standardized legal note for invoices, including W8/Bank info."""
        # ### Phase 4 Plan Ref: 9.6 (Implement get_invoice_legal_note)
        self.logger.debug(f"Generating invoice legal note for country: {client_country}")
        base_note = "Payment constitutes acceptance of service terms." # Default
        try:
            await self._internal_think(f"Querying KB for compliance notes relevant to invoices in {client_country}.")
            compliance_notes = await self.query_knowledge_base(data_types=['compliance_note', 'legal_update_interpretation'], tags=['invoice', 'financial', 'tax', client_country.lower()], limit=1, min_relevance=0.7)
            if compliance_notes:
                note_content = compliance_notes[0].content
                if isinstance(note_content, str) and note_content.startswith('{'):
                     try: base_note = json.loads(note_content).get('compliance_note', base_note)
                     except json.JSONDecodeError: pass
                elif isinstance(note_content, str): base_note = note_content
                self.logger.debug(f"Using compliance note from KB: {base_note}")
        except Exception as e: self.logger.error(f"Error fetching compliance note from KB for {client_country}: {e}")

        # Fetch W8 and Bank info from settings (env vars)
        user_name = self.config.get('USER_NAME', 'The Agency Operator')
        w8_name = self.config.get('W8_NAME', '[Your Name/Company Name]')
        w8_country = self.config.get('W8_COUNTRY', '[Your Country]')
        bank_account_info = self.config.get('MOROCCAN_BANK_ACCOUNT', '[Your Bank Account Info]') # Keep this somewhat generic

        bulletproof_terms = (
            f"Services provided by {w8_name}, {w8_country}. "
            "This agreement is irrevocable and binding upon acceptance. Agency not liable for indirect damages. "
            "Disputes governed by Moroccan law, courts of Rabat. "
            f"Payment to designated account ({bank_account_info}) is non-refundable post-commencement. "
            "ISO 20022 compliance acknowledged where applicable."
        )
        return f"{base_note} | {bulletproof_terms}"

    def _parse_llm_json(self, json_string: str, expect_type: Type = dict) -> Union[Dict, List, None]:
        """Safely parses JSON from LLM output, handling markdown code blocks."""
        if not json_string: return None
        try:
            # Try finding JSON within markdown code blocks first
            if expect_type == dict:
                match = re.search(r'```json\s*(\{.*?\})\s*```', json_string, re.DOTALL)
                if match: return json.loads(match.group(1))
                # Fallback: try parsing the whole string if it looks like a dict
                if json_string.strip().startswith('{') and json_string.strip().endswith('}'):
                    return json.loads(json_string)
            elif expect_type == list:
                match = re.search(r'```json\s*(\[.*?\])\s*```', json_string, re.DOTALL)
                if match: return json.loads(match.group(1))
                # Fallback: try parsing the whole string if it looks like a list
                if json_string.strip().startswith('[') and json_string.strip().endswith(']'):
                    return json.loads(json_string)
            # If no code block found or type mismatch, try parsing the whole string as a last resort
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode LLM JSON response: {e}. Response snippet: {json_string[:500]}...")
            return None # Return None on parsing failure

    # --- Abstract Method Implementations ---
    # (Keep implementations similar to previous version, ensure KB queries work)
    async def learning_loop(self):
        """Periodic loop to fetch and interpret legal updates."""
        # ### Phase 4 Plan Ref: 9.7 (Implement learning_loop)
        self.logger.info(f"{self.AGENT_NAME} learning loop started.")
        while not self._stop_event.is_set():
            try:
                interval = self.internal_state.get('update_interval_seconds', 604800)
                await asyncio.sleep(interval)
                if self._stop_event.is_set(): break # Check again after sleep
                self.logger.info("Executing LegalAgent learning loop (scan for updates)...")
                await self._scan_for_updates({}) # Trigger scan task
            except asyncio.CancelledError: self.logger.info("LegalAgent learning loop cancelled."); break
            except Exception as e:
                self.logger.error(f"Error during LegalAgent learning loop: {e}", exc_info=True)
                await self._report_error(f"Learning loop error: {e}")
                await asyncio.sleep(3600) # Wait longer after error

    async def self_critique(self) -> Dict[str, Any]:
        """Evaluates the agent's own performance and strategy."""
        # ### Phase 4 Plan Ref: 9.8 (Implement self_critique)
        self.logger.info(f"{self.AGENT_NAME}: Performing self-critique.")
        critique = {"status": "ok", "feedback": "Critique pending analysis."}
        critique_thought = "Structured Thinking: Self-Critique LegalAgent. Plan: Query KB stats -> Analyze -> Format -> Return."
        await self._internal_think(critique_thought)
        try:
            async with self.session_maker() as session:
                # Count recent validations, interpretations, analyses
                one_day_ago = datetime.now(timezone.utc) - timedelta(days=1)
                stmt = select(KnowledgeFragment.data_type, func.count(KnowledgeFragment.id)).where(
                    KnowledgeFragment.agent_source == self.AGENT_NAME,
                    KnowledgeFragment.timestamp >= one_day_ago
                ).group_by(KnowledgeFragment.data_type)
                results = await session.execute(stmt)
                activity_counts = {row.data_type: row[1] for row in results.mappings().all()}
            critique['activity_24h'] = activity_counts
            critique['last_scan_time'] = self.internal_state.get('last_scan_time')
            feedback = f"Last Scan: {self.internal_state.get('last_scan_time')}. Activity (24h): {activity_counts}. "
            # TODO: Add analysis of validation accuracy if feedback is available
            critique['feedback'] = feedback
        except Exception as e: self.logger.error(f"Error during self-critique: {e}", exc_info=True); critique['status'] = 'error'; critique['feedback'] = f"Critique failed: {e}"
        return critique

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Constructs context-rich prompts for LLM calls."""
        # ### Phase 4 Plan Ref: 9.9 (Implement generate_dynamic_prompt)
        # (Logic remains similar to previous version)
        self.logger.debug(f"Generating dynamic prompt for LegalAgent task: {task_context.get('task')}")
        prompt_parts = [self.meta_prompt]
        prompt_parts.append("\n--- Current Task Context ---")
        for key, value in task_context.items():
            value_str = ""
            max_len = 1500
            if key == 'knowledge_base_context': max_len = 2500 # Allow more KB context
            elif key == 'updates_list': max_len = 3000 # Allow more update titles
            if isinstance(value, str): value_str = value[:max_len] + ("..." if len(value) > max_len else "")
            elif isinstance(value, (int, float, bool)): value_str = str(value)
            elif isinstance(value, (dict, list)):
                try: value_str = json.dumps(value, default=str, indent=2); value_str = value_str[:max_len] + ("..." if len(value_str) > max_len else "")
                except TypeError: value_str = str(value)[:max_len] + "..."
            else: value_str = str(value)[:max_len] + "..."
            prompt_parts.append(f"**{key.replace('_', ' ').title()}**: {value_str}")

        prompt_parts.append("\n--- Instructions ---")
        task_type = task_context.get('task')
        # Add specific instructions based on task_type (same as previous version)
        if task_type == 'Analyze and recommend optimal initial corporate structure':
            prompt_parts.append("1. Analyze pros/cons/risks/steps for Wyoming LLC, Delaware LLC, and potentially one other relevant structure based on context.")
            prompt_parts.append("2. Consider tax efficiency, liability, privacy, compliance overhead, and grey-area potential.")
            prompt_parts.append("3. Factor in the provided business context and KB information.")
            prompt_parts.append("4. Provide a clear final recommendation with rationale.")
        elif task_type == 'Deep analysis of potential grey area strategy':
             prompt_parts.append(f"1. Deeply analyze the legal basis, risks (detection likelihood, penalties, reputational, operational), benefits, and implementation plan for the grey area: '{task_context.get('grey_area_topic', 'N/A')}' in the context of '{task_context.get('application_context', 'N/A')}'.")
             prompt_parts.append("2. Reference relevant statutes, loopholes, precedents from KB or general knowledge.")
             prompt_parts.append("3. Provide actionable implementation steps, monitoring needs, and fallback plans.")
             prompt_parts.append("4. Conclude with an overall recommendation (Proceed, Caution, Avoid, Research).")
        elif task_type == 'Interpret legal update titles/links':
             prompt_parts.append("1. For each item in 'Updates List': Determine category, summarize impact, assess relevance (0.0-1.0), identify grey areas, generate compliance note.")
        elif task_type == 'Validate operation compliance and risk':
             prompt_parts.append("1. Analyze 'Operation Description' against legal/compliance knowledge (KB/general). Focus USA/Morocco.")
             prompt_parts.append("2. Assess strict compliance (bool), strategic risk level (Low/Med/High/Critical - considering detection/consequences), list issues & consequences.")
             prompt_parts.append("3. Provide grey area assessment. Suggest mitigation steps. Give explicit proceed recommendation.")
        else: prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")

        if task_context.get('desired_output_format'): prompt_parts.append(f"**Output Format:** {task_context['desired_output_format']}")
        if "JSON" in task_context.get('desired_output_format', ''): prompt_parts.append("\nRespond ONLY with valid JSON matching the specified format. Do not include explanations outside the JSON structure.\n```json")

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for LegalAgent (length: {len(final_prompt)} chars)")
        return final_prompt

    async def collect_insights(self) -> Dict[str, Any]:
        """Collects insights about legal task performance."""
        # ### Phase 4 Plan Ref: 9.10 (Implement collect_insights)
        self.logger.debug("LegalAgent collect_insights called.")
        insights = {
            "agent_name": self.AGENT_NAME, "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "last_scan_time": self.internal_state.get('last_scan_time'),
            "validations_24h": 0, "interpretations_24h": 0, "analyses_24h": 0,
            "key_observations": []
        }
        if not self.session_maker: insights["key_observations"].append("DB session unavailable."); return insights
        try:
            async with self.session_maker() as session:
                one_day_ago = datetime.now(timezone.utc) - timedelta(days=1)
                stmt = select(KnowledgeFragment.data_type, func.count(KnowledgeFragment.id)).where(
                    KnowledgeFragment.agent_source == self.AGENT_NAME,
                    KnowledgeFragment.timestamp >= one_day_ago,
                    KnowledgeFragment.data_type.in_(['operation_validation', 'legal_update_interpretation', 'legal_analysis', 'grey_area_analysis'])
                ).group_by(KnowledgeFragment.data_type)
                results = await session.execute(stmt)
                counts = {row.data_type: row for row in results.mappings().all()}
                insights["validations_24h"] = counts.get('operation_validation', 0)
                insights["interpretations_24h"] = counts.get('legal_update_interpretation', 0)
                insights["analyses_24h"] = counts.get('legal_analysis', 0) + counts.get('grey_area_analysis', 0)
            insights["key_observations"].append("Collected 24h activity counts.")
        except Exception as e: self.logger.error(f"Error collecting DB insights for LegalAgent: {e}"); insights["key_observations"].append("Error collecting DB insights.")
        return insights

    # --- KB Interaction Helpers (Delegate to ThinkTool) ---
    # These ensure LegalAgent uses the central KB mechanism managed by ThinkTool

    async def log_knowledge_fragment(self, *args, **kwargs):
        """Logs a knowledge fragment via ThinkTool."""
        if self.think_tool and hasattr(self.think_tool, 'log_knowledge_fragment'):
            return await self.think_tool.log_knowledge_fragment(*args, **kwargs)
        else: self.logger.error("ThinkTool unavailable or missing log_knowledge_fragment method."); return None

    async def query_knowledge_base(self, *args, **kwargs):
        """Queries the knowledge base via ThinkTool."""
        if self.think_tool and hasattr(self.think_tool, 'query_knowledge_base'):
            return await self.think_tool.query_knowledge_base(*args, **kwargs)
        else: self.logger.error("ThinkTool unavailable or missing query_knowledge_base method."); return []

    async def log_learned_pattern(self, *args, **kwargs):
         """Logs a learned pattern via ThinkTool."""
         if self.think_tool and hasattr(self.think_tool, 'log_learned_pattern'):
             return await self.think_tool.log_learned_pattern(*args, **kwargs)
         else: self.logger.error("ThinkTool unavailable or missing log_learned_pattern method."); return None

    async def get_latest_patterns(self, *args, **kwargs):
         """Retrieves learned patterns via ThinkTool."""
         if self.think_tool and hasattr(self.think_tool, 'get_latest_patterns'):
             return await self.think_tool.get_latest_patterns(*args, **kwargs)
         else: self.logger.error("ThinkTool unavailable or missing get_latest_patterns method."); return []

# --- End of agents/legal_agent.py ---