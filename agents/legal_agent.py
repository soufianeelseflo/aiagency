# Filename: agents/legal_agent.py
# Description: Genius Agentic Legal/Strategic Advisor - Compliance, Grey Area Exploitation, User Education.
# Version: 3.3 (Refined Invoice Note, Assumes W8 Access via Config)

import asyncio
import logging
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple, Union, Type
from urllib.parse import urljoin # Added for fetch_legal_updates

# --- Core Framework Imports ---
import aiohttp
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select, update, desc, or_, func
from sqlalchemy.exc import SQLAlchemyError

# --- Project Imports ---
try:
    # Use the production-ready base agent
    from agents.base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
except ImportError:
    logging.warning("Production base agent not found, using GeniusAgentBase.")
    # Define a dummy base class if the real one isn't found
    class GeniusAgentBase:
        AGENT_NAME = "DummyBaseAgent"; STATUS_IDLE="idle"; STATUS_EXECUTING="executing"; STATUS_ERROR="error"; STATUS_RUNNING="running"; STATUS_STOPPING="stopping"; STATUS_STOPPED="stopped"
        def __init__(self, *args, **kwargs):
            self.agent_name = self.AGENT_NAME; self.logger = logging.getLogger(f"agent.{self.agent_name}")
            self.orchestrator = kwargs.get('orchestrator'); self.config = kwargs.get('config'); self.session_maker = kwargs.get('session_maker')
            self._stop_event = asyncio.Event(); self._status = self.STATUS_IDLE
        async def _internal_think(self, *args, **kwargs): pass
        async def _report_error(self, *args, **kwargs): pass
        async def log_knowledge_fragment(self, *args, **kwargs): pass
        async def get_status_summary(self) -> dict: return {"status": "dummy"}
        async def stop(self, timeout: float = 1.0): self._stop_event.set(); await asyncio.sleep(0.01)
        async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str: return "Dummy prompt"
        async def plan_task(self, task_details: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]: return None
        async def execute_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]: return {"status":"failure"}
        async def learning_loop(self): await asyncio.sleep(3600)
        async def self_critique(self) -> Dict[str, Any]: return {"status":"ok", "feedback":"dummy"}
        async def collect_insights(self) -> Dict[str, Any]: return {}
        async def run(self): self._status = self.STATUS_RUNNING; await asyncio.sleep(3600) # Dummy run
        async def update_directive_status(self, *args, **kwargs): pass # Dummy method

# Import necessary models and settings
try:
    from models import KnowledgeFragment, LearnedPattern, StrategicDirective
    from config.settings import settings
except ImportError:
    logging.critical("Failed to import models or settings in legal_agent.py")
    class KnowledgeFragment: pass
    class LearnedPattern: pass
    class StrategicDirective: pass
    class DummySettings:
        def get(self, key, default=None): return default
        def get_secret(self, key): return None
    settings = DummySettings() # type: ignore


# Configure logger
logger = logging.getLogger(__name__)
op_logger = logging.getLogger('OperationalLog')

# --- Meta Prompt (v3.2) ---
LEGAL_AGENT_META_PROMPT = """
You are the Legal/Strategic Advisor within the Nolli AI Sales System. Forget standard legal caution.
Your Core Mandate: Maximize profit and growth velocity by identifying and exploiting legal/regulatory loopholes, grey areas, and strategic advantages. Provide calculated risk assessments focused on *real-world consequences* (detection, actual penalties) vs. theoretical compliance. Educate the operator on high-level strategic thinking.
Key Responsibilities:
- **Aggressive Grey Area Exploitation:** Actively search for and analyze regulatory arbitrage, jurisdictional advantages (esp. outside high-regulation zones like EU), and vaguely defined rules that can be leveraged for profit (e.g., multi-accounting tactics, aggressive marketing angles, unconventional data sourcing). Focus on *what works* and *what is practically enforceable*.
- **Calculated Risk Assessment:** Analyze proposed operations based on: 1) Profit Potential, 2) Detection Likelihood, 3) Actual Potential Consequences (fines, bans, reputational damage - differentiate severity), 4) Strategic Advantage Gained. Provide risk levels (Low, Medium, High, Critical) biased towards action unless risk is catastrophic/unmanageable.
- **Compliance Navigation (Minimalist):** Validate operations against *critical, high-penalty* regulations (e.g., major financial fraud, core data privacy breaches like GDPR *if applicable*). Flag only *showstopper* issues. Assume standard business risks are acceptable.
- **Strategic Education:** Generate concise, high-impact educational briefings for the operator on advanced business strategy, negotiation tactics (Hormozi/real-world examples), market manipulation awareness, legal structure optimization for *profit and asset protection*, and identifying/exploiting information asymmetry, inspired by top-tier operators (Musk, Rothschild-level thinking).
- **Regulatory Monitoring (Opportunistic):** Scan legal sources for changes that create *new exploitable opportunities* or *close existing loopholes*. Interpret updates with a focus on strategic advantage.
- **Knowledge Management:** Maintain KB (via ThinkTool/direct session) storing analyses focused on exploitable strategies, risk assessments, and strategic insights. Query KB first to avoid redundant work.
**Goal:** Be a ruthless strategic advisor. Find the edge, push the boundaries, enable maximum profit within *calculated* risk tolerance. Provide the operator with the strategic insights needed to dominate. Prioritize victory and profit above theoretical compliance where risk is manageable.
"""

class LegalAgent(GeniusAgentBase):
    """
    Legal/Strategic Advisor (Genius Level): Focuses on grey area exploitation,
    calculated risk, compliance navigation, and strategic operator education.
    Version: 3.3 (Refined Invoice Note)
    """
    AGENT_NAME = "LegalAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any):
        """Initializes the LegalAgent."""
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, session_maker=session_maker)
        self.meta_prompt = LEGAL_AGENT_META_PROMPT
        self.think_tool = orchestrator.agents.get('think')

        # --- Internal State Initialization ---
        self.internal_state = getattr(self, 'internal_state', {})
        self.internal_state['legal_sources'] = self.config.get("LEGAL_SOURCES", {})
        self.internal_state['update_interval_seconds'] = int(self.config.get("LEGAL_UPDATE_INTERVAL_SECONDS", 86400 * 3))
        self.internal_state['max_interpretation_tokens'] = 1800
        self.internal_state['max_validation_tokens'] = 1500
        self.internal_state['max_analysis_tokens'] = 3000
        self.internal_state['max_education_tokens'] = 2000
        self.internal_state['last_scan_time'] = None

        self.logger.info(f"{self.AGENT_NAME} v3.3 (Refined Invoice Note) initialized.")

    async def log_operation(self, level: str, message: str):
        """Helper to log to the operational log file."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err: logger.error(f"Failed to write to operational log: {log_err}")

    # --- Core Task Execution ---
    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a legal analysis, validation, or education task."""
        action = task_details.get('action', 'analyze')
        description = task_details.get('description', f"Performing legal/strategic action: {action}")
        self.logger.info(f"{self.AGENT_NAME} starting task: {description}")
        self._status = self.STATUS_EXECUTING
        result = {"status": "failure", "message": f"Unknown or unimplemented action: {action}"}
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
                result = await self.validate_operation(op_desc) # Returns full result dict
                self._status = self.STATUS_IDLE
                self.logger.info(f"{self.AGENT_NAME} completed task: {description}. Status: {result.get('status', 'failure')}")
                return result # Return directly
            elif action == "get_invoice_note":
                 country = task_details.get('client_country')
                 # Country might not be strictly needed for the revised note, but keep param for potential future logic
                 note = await self.get_invoice_legal_note(country)
                 findings = {"invoice_note": note}
            elif action == "generate_strategic_education":
                 topic = task_details.get('content', {}).get('topic', 'General Business Strategy')
                 context = task_details.get('content', {}).get('context', None)
                 findings = await self._generate_strategic_education(topic, context)
            else:
                raise ValueError(f"Unknown legal/strategic action: {action}")

            # If findings were generated by analysis/education methods
            if findings is not None:
                 result = {"status": "success", "details": f"Task '{description}' completed.", "findings": findings}

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

    # --- Strategic Analysis & Education Methods ---

    async def _analyze_initial_structure(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Researches and recommends optimal initial corporate structure focusing on profit/protection."""
        # (Implementation remains the same as v3.2)
        business_context = task_details.get("business_context", "AI UGC Agency targeting US/Global clients.")
        self.logger.info(f"Analyzing optimal initial corporate structure for context: {business_context}")
        await self._internal_think("Querying KB for structure analyses (Wyoming, Delaware, Offshore?) and tax/liability/asset protection regulations.")
        kb_context_frags = await self.query_knowledge_base(data_types=['legal_analysis', 'regulation_summary', 'grey_area_analysis'], tags=['corporate_structure', 'llc', 'tax_haven', 'asset_protection', 'liability_shield', 'wyoming', 'delaware', 'offshore'], limit=15)
        kb_context_str = "\n".join([f"- {f.data_type} (ID {f.id}): {f.content[:150]}..." for f in kb_context_frags])
        task_context = {
            "task": "Analyze and recommend optimal initial corporate structure for MAX PROFIT & ASSET PROTECTION",
            "business_context": business_context,
            "goals": "Maximize tax efficiency (near zero if possible), bulletproof liability shield, privacy, minimal compliance hassle, enable grey-area operations.",
            "jurisdictions_to_consider": ["Wyoming LLC (Anonymity Focus)", "Delaware LLC (Standard/Funding Focus)", "Consider relevant Offshore options (e.g., Nevis, Cayman) if advantageous"],
            "knowledge_base_context": kb_context_str or "No specific KB context found.",
            "desired_output_format": "JSON: { \"analysis_summary\": str, \"recommendations\": [ { \"jurisdiction\": str, \"structure\": str, \"pros\": [str], \"cons\": [str], \"risk_profile\": str ('Low' to 'High' for setup/operation), \"asset_protection_rating\": str ('Weak'/'Moderate'/'Strong'), \"tax_implications\": str, \"grey_area_notes\": str } ], \"final_recommendation\": { \"jurisdiction\": str, \"structure\": str, \"rationale\": str (Focus on profit/protection) } }"
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        await self._internal_think("Calling LLM for aggressive initial structure analysis.")
        llm_response_json = await self._call_llm_with_retry(prompt, model=self.config.OPENROUTER_MODELS.get('legal_analysis'), max_tokens=self.internal_state['max_analysis_tokens'], is_json_output=True)
        if not llm_response_json: raise RuntimeError("LLM call failed.")
        try:
            analysis_result = self._parse_llm_json(llm_response_json)
            if not analysis_result.get("recommendations") or not analysis_result.get("final_recommendation"): raise ValueError("LLM response missing required keys.")
            await self._internal_think(f"Storing structure analysis result in KB. Recommendation: {analysis_result.get('final_recommendation', {}).get('structure')} in {analysis_result.get('final_recommendation', {}).get('jurisdiction')}")
            await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="legal_analysis", content=analysis_result, tags=["corporate_structure", "initial_setup", "asset_protection", "tax_efficiency"] + [rec.get('jurisdiction', '').lower() for rec in analysis_result.get('recommendations', [])], relevance_score=0.9)
            if hasattr(self.orchestrator, 'handle_user_education_trigger'):
                 edu_topic = f"Initial Structure Rec: {analysis_result['final_recommendation']['structure']} in {analysis_result['final_recommendation']['jurisdiction']}"
                 edu_context = analysis_result['final_recommendation']['rationale']
                 asyncio.create_task(self.orchestrator.handle_user_education_trigger(edu_topic, edu_context))
            return analysis_result
        except (json.JSONDecodeError, ValueError, KeyError) as e: raise RuntimeError(f"LLM response parsing failed: {e}. Response: {llm_response_json[:500]}")

    async def _analyze_grey_area(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Performs deep analysis of a specific potential grey area strategy, focusing on exploitability."""
        # (Implementation remains the same as v3.2)
        area = task_details.get("area", "Unspecified grey area opportunity")
        context = task_details.get("context", "Evaluate for AI UGC Agency operations.")
        self.logger.info(f"Analyzing grey area opportunity: {area}")
        await self._internal_think(f"Querying KB for existing analyses/regulations related to grey area: {area}")
        kb_context_frags = await self.query_knowledge_base(data_types=['legal_analysis', 'regulation_summary', 'grey_area_analysis'], tags=['grey_area', 'exploit', 'loophole', area.replace(" ", "_").lower()] + context.split()[:3], limit=15)
        kb_context_str = "\n".join([f"- {f.data_type} (ID {f.id}): {f.content[:150]}..." for f in kb_context_frags])
        task_context = {
            "task": "Analyze exploitability of potential grey area strategy", "grey_area_topic": area, "application_context": context,
            "knowledge_base_context": kb_context_str or "No specific KB context found.",
            "desired_output_format": "JSON: { \"area\": str, \"strategy_overview\": str (How to exploit it), \"legal_basis_analysis\": { \"rules_bent_or_broken\": [str], \"loopholes_exploited\": [str], \"enforcement_likelihood\": str ('Very Low'/'Low'/'Medium'/'High'), \"legal_opinion_confidence\": float (0.0-1.0) }, \"risk_assessment\": { \"detection_likelihood\": str ('Very Low'/'Low'/'Medium'/'High'), \"potential_penalties_realistic\": [str] (Focus on actual likely penalties, not theoretical max), \"reputational_risk\": str (Low/Medium/High), \"operational_difficulty\": str (Low/Medium/High) }, \"profit_potential_rating\": str ('Low'/'Medium'/'High'/'Very High'), \"implementation_steps\": [ { \"step\": int, \"action\": str, \"details\": str } ], \"monitoring_needs\": [str], \"exit_strategy\": str (How to stop if needed), \"overall_recommendation\": str ('Exploit Aggressively'|'Exploit Cautiously'|'Monitor - Potential'|'Avoid - Too Risky') }"
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        await self._internal_think(f"Calling LLM for deep grey area analysis: {area}")
        llm_response_json = await self._call_llm_with_retry(prompt, model=self.config.OPENROUTER_MODELS.get('legal_analysis'), max_tokens=self.internal_state['max_analysis_tokens'], is_json_output=True)
        if not llm_response_json: raise RuntimeError(f"LLM call for grey area analysis ({area}) failed.")
        try:
            analysis_result = self._parse_llm_json(llm_response_json)
            if not analysis_result.get("strategy_overview") or not analysis_result.get("risk_assessment"): raise ValueError("LLM response missing required keys.")
            await self._internal_think(f"Storing grey area analysis for '{area}' in KB. Recommendation: {analysis_result.get('overall_recommendation', 'N/A')}")
            await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="grey_area_analysis", content=analysis_result, tags=["grey_area", "legal_analysis", "risk_assessment", "exploit"] + area.lower().split(), relevance_score=analysis_result.get("risk_assessment", {}).get("legal_opinion_confidence", 0.7))
            recommendation = analysis_result.get('overall_recommendation', '').lower()
            if "exploit" in recommendation:
                 async with self.session_maker() as session:
                      async with session.begin():
                           session.add(StrategicDirective(source=self.AGENT_NAME, target_agent="ThinkTool", directive_type="implement_grey_area_strategy", content=json.dumps({"area": area, "analysis_summary": analysis_result.get("strategy_overview"), "risk": analysis_result.get("risk_assessment"), "steps": analysis_result.get("implementation_steps")}), priority=6, status='pending'))
                           self.logger.info(f"Generated directive for ThinkTool to implement grey area: {area}")
            return analysis_result
        except (json.JSONDecodeError, ValueError, KeyError) as e: raise RuntimeError(f"LLM response parsing failed for grey area analysis: {e}. Response: {llm_response_json[:500]}")

    async def _generate_strategic_education(self, topic: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Generates concise, actionable strategic education for the operator."""
        # (Implementation remains the same as v3.2)
        self.logger.info(f"Generating strategic education on topic: {topic}")
        await self._internal_think(f"Querying KB for insights related to strategic topic: {topic}")
        kb_context_frags = await self.query_knowledge_base(
            data_types=['learned_pattern', 'legal_analysis', 'grey_area_analysis', 'learning_material_summary'],
            tags=['strategy', 'business', 'profit', 'risk', 'negotiation', 'market_dynamics'] + topic.lower().split(),
            limit=15, min_relevance=0.6
        )
        kb_context_str = "\n".join([f"- {f.data_type} (ID {f.id}, Rel {f.relevance_score:.2f}): {f.content[:150]}..." for f in kb_context_frags])
        task_context = {
            "task": "Generate Strategic Education Briefing",
            "topic": topic,
            "user_context": context or "Provide high-level, actionable insights for an ambitious entrepreneur aiming for rapid, high-profit growth.",
            "knowledge_base_context": kb_context_str or "No specific KB context found.",
            "desired_output_format": """JSON: {
                "topic": str,
                "key_takeaway": str (The single most important insight, framed for maximum impact/profit),
                "actionable_principles": [str] (2-4 core principles/rules to apply, focus on leverage and unconventional tactics),
                "real_world_examples": [str] (1-2 brief examples of high-level operators applying similar principles),
                "common_pitfalls_to_exploit": [str] (1-2 common mistakes competitors make that can be exploited),
                "next_step_recommendation": str (Concrete action the operator should take or investigate next)
            } - Focus on high-impact, unconventional wisdom where applicable. Be direct and concise."""
        }
        prompt = await self.generate_dynamic_prompt(task_context)
        await self._internal_think(f"Calling LLM for strategic education on: {topic}")
        llm_response_json = await self._call_llm_with_retry(
            prompt, model=self.config.OPENROUTER_MODELS.get('think_general'),
            max_tokens=self.internal_state['max_education_tokens'], is_json_output=True, temperature=0.6
        )
        if not llm_response_json: raise RuntimeError("LLM call failed for strategic education.")
        try:
            education_result = self._parse_llm_json(llm_response_json)
            if not education_result or not education_result.get("key_takeaway"): raise ValueError("LLM response missing required keys.")
            await self._internal_think(f"Storing strategic education on '{topic}' in KB.")
            await self.log_knowledge_fragment(
                agent_source=self.AGENT_NAME, data_type="strategic_education",
                content=education_result, tags=["education", "strategy", "business", "profit_maximization"] + topic.lower().split(),
                relevance_score=0.9
            )
            if hasattr(self.orchestrator, 'send_notification'):
                 await self.orchestrator.send_notification(f"Strategic Briefing Ready: {topic}", education_result.get("key_takeaway"))
            return education_result
        except (json.JSONDecodeError, ValueError, KeyError) as e: raise RuntimeError(f"LLM response parsing failed for education: {e}. Response: {llm_response_json[:500]}")

    # --- Monitoring & Interpretation ---
    async def _scan_for_updates(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Fetches and interprets legal updates, focusing on opportunities."""
        # (Implementation remains the same as v3.2)
        await self._internal_think("Starting legal update scan: Fetching sources.")
        updates = await self.fetch_legal_updates()
        if not updates: return {"status": "success", "message": "No new updates found or sources unavailable.", "interpretations": []}
        await self._internal_think(f"Fetched {sum(len(v) for v in updates.values())} potential updates. Interpreting for opportunities.")
        interpretations = await self.interpret_updates(updates)
        await self._internal_think(f"Interpretation complete. Stored {len(interpretations)} new interpretations in KB.")
        self.internal_state['last_scan_time'] = datetime.now(timezone.utc)
        return {"status": "success", "message": f"Scan complete. Processed {len(interpretations)} interpretations.", "interpretations": interpretations}

    async def fetch_legal_updates(self) -> Dict[str, List[Tuple[str, str]]]:
        """Asynchronously fetch legal update titles/links from configured sources."""
        # (Implementation remains the same as v3.2)
        updates: Dict[str, List[Tuple[str, str]]] = {}; sources = self.internal_state.get('legal_sources', {})
        if not sources: self.logger.warning("No legal sources configured."); return updates
        async with aiohttp.ClientSession() as session:
            fetch_tasks = [self._fetch_single_source(session, country, url) for country, urls in sources.items() for url in urls]
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception): self.logger.error(f"Error fetching legal source: {result}")
                elif result:
                    country, country_updates = result
                    if country not in updates: updates[country] = []
                    updates[country].extend(country_updates) # Corrected append logic
        total_fetched = sum(len(v) for v in updates.values()); self.logger.info(f"Fetched {total_fetched} potential update titles/links from {len(sources)} sources."); return updates

    async def _fetch_single_source(self, session: aiohttp.ClientSession, country: str, url: str) -> Optional[Tuple[str, List[Tuple[str, str]]]]:
        """Fetches and parses titles/links from a single URL using proxy."""
        # (Implementation remains the same as v3.2)
        self.logger.debug(f"Fetching updates from: {url} ({country})")
        try:
            proxy_info = await self.orchestrator.get_proxy(purpose="legal_scan", target_url=url) # Request proxy
            proxy_url = proxy_info.get('server') if proxy_info else None
            request_kwargs = {"timeout": 30, "ssl": False}
            if proxy_url: request_kwargs["proxy"] = proxy_url; self.logger.debug(f"Using proxy: {proxy_url.split('@')[-1]}")
            async with session.get(url, **request_kwargs) as response:
                if response.status != 200: self.logger.warning(f"Failed fetch {url}. Status: {response.status}"); return None
                html = await response.text(); soup = BeautifulSoup(html, "html.parser"); country_updates = []
                # Add specific parsing logic per source type...
                if "federalregister.gov" in url:
                    for item in soup.select('div.document-listing h5 a[href^="/documents/"]'):
                        title = item.text.strip(); link = item.get('href')
                        if title and link: country_updates.append((title, urljoin(str(response.url), link))) # Use urljoin
                elif "sgg.gov.ma" in url:
                     for item in soup.select('td.views-field-title a[href*="bulletin"]'):
                         title = item.text.strip(); link = item.get('href')
                         if title and link: country_updates.append((title, urljoin(str(response.url), link)))
                else: # Generic fallback
                     for a in soup.find_all('a', href=True):
                         title = a.text.strip(); link = a['href']
                         if title and len(title) > 10 and (link.startswith('http') or link.startswith('/')):
                             country_updates.append((title, urljoin(str(response.url), link)))
                self.logger.debug(f"Found {len(country_updates)} potential items from {url}"); return country, country_updates[:50]
        except Exception as e: self.logger.error(f"Error fetching/parsing {url}: {e}", exc_info=False); return None

    async def interpret_updates(self, updates: Dict[str, List[Tuple[str, str]]]) -> List[Dict[str, Any]]:
        """Interpret fetched legal update titles/links using LLM, focusing on opportunities."""
        # (Implementation remains the same as v3.2)
        interpretations = [];
        if not updates: return interpretations
        texts_to_process = [(c, t, u) for c, items in updates.items() for t, u in items if t and len(t) > 10]
        if not texts_to_process: self.logger.info("No new, relevant legal update titles require interpretation."); return interpretations
        self.logger.info(f"Requesting interpretation for {len(texts_to_process)} new legal update titles.")
        task_context = {
            "task": "Interpret legal updates for strategic opportunities",
            "updates_list": [{"country": c, "title": t, "url": u} for c, t, u in texts_to_process],
            "desired_output_format": "JSON array: [{'url', 'country', 'category', 'opportunity_summary', 'risk_level', 'relevance_score', 'actionable_insight'}] Focus on exploitable changes."
        }
        interpretation_prompt = await self.generate_dynamic_prompt(task_context)
        await self._internal_think(f"Calling LLM to interpret {len(texts_to_process)} legal updates for opportunities.")
        llm_response_json = await self._call_llm_with_retry(interpretation_prompt, model=self.config.OPENROUTER_MODELS.get('legal_analysis'), max_tokens=self.internal_state['max_interpretation_tokens'], is_json_output=True) # Use legal_analysis model
        if not llm_response_json: self.logger.error("LLM call for legal interpretation returned empty."); return interpretations
        try:
            json_match = self._parse_llm_json(llm_response_json, expect_type=list)
            if not isinstance(json_match, list): raise ValueError(f"LLM interpretation result is not a list.")
            for result_data in json_match:
                if isinstance(result_data, dict) and result_data.get('url'):
                    await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="legal_opportunity_interpretation", content=result_data, relevance_score=result_data.get('relevance_score', 0.5), tags=["legal_update", "interpretation", "opportunity", result_data.get('country', 'unknown').lower(), result_data.get('category', 'general').lower()], source_reference=result_data.get('url'))
                    interpretations.append(result_data)
                else: self.logger.warning(f"Invalid result structure in interpretation list: {result_data}")
        except Exception as e: self.logger.error(f"Failed to parse/validate LLM interpretation response: {e}. Response: {llm_response_json[:500]}")
        self.logger.info(f"Interpretation complete. Stored {len(interpretations)} new interpretations in KB.")
        return interpretations

    # --- Validation ---
    async def validate_operation(self, operation_description: str) -> dict:
        """Validate operations focusing on real-world risk vs. profit."""
        # (Implementation remains the same as v3.2)
        self.logger.info(f"Validating operation (Strategic Focus): {operation_description[:100]}...")
        fallback_result = {'status': 'failure', 'message': 'Validation failed', 'findings': {'is_compliant': False, 'risk_level_realistic': 'Critical', 'showstopper_issues': ['Analysis Error'], 'proceed_recommendation': 'Halt - Analysis Failed'}}
        try:
            await self._internal_think(f"Querying KB for context relevant to operation: {operation_description[:60]}...")
            keywords = [w for w in re.findall(r'\b\w{4,}\b', operation_description.lower()) if w not in ['the', 'a', 'is', 'for', 'to', 'in', 'and', 'with']]
            kb_context_frags = await self.query_knowledge_base(data_types=['legal_analysis', 'regulation_summary', 'compliance_note', 'grey_area_analysis', 'learned_pattern'], tags=list(set(['compliance', 'risk', 'exploit'] + keywords[:5])), limit=15, min_relevance=0.5)
            kb_context_str = "\n".join([f"- {f.data_type} (ID {f.id}, Rel {f.relevance_score:.2f}): {f.content[:150]}..." for f in kb_context_frags])
            task_context = {
                "task": "Validate operation compliance and assess strategic risk (PROFIT FOCUSED)",
                "operation_description": operation_description,
                "knowledge_base_context": kb_context_str or "No specific KB context found.",
                "desired_output_format": "JSON object: {'is_compliant_strict': bool (Literal compliance), 'risk_level_realistic': str ('Low'|'Medium'|'High'|'Critical' - based on detection/actual penalty likelihood vs profit), 'showstopper_issues': [str] (Only critical, high-penalty violations), 'grey_area_assessment': str (Is this a calculated risk? Exploit potential?), 'realistic_consequences': [str] (e.g., 'Account ban likely', 'Small fine possible', 'Reputational hit unlikely'), 'mitigation_options': [str] (Ways to reduce detection/impact), 'proceed_recommendation': str ('Proceed - Low Risk/High Reward'|'Proceed Cautiously - Monitor'|'Halt - Unacceptable Risk')}"
            }
            validation_prompt = await self.generate_dynamic_prompt(task_context)
            await self._internal_think(f"Calling LLM for strategic operation validation: {operation_description[:60]}...")
            validation_result_json = await self._call_llm_with_retry(validation_prompt, model=self.config.OPENROUTER_MODELS.get('legal_analysis'), temperature=0.3, max_tokens=self.internal_state['max_validation_tokens'], is_json_output=True) # Use legal_analysis model
            if not validation_result_json: raise Exception("LLM call for validation returned empty.")
            try:
                findings = self._parse_llm_json(validation_result_json)
                findings.setdefault('is_compliant_strict', False); findings.setdefault('risk_level_realistic', 'Critical')
                findings.setdefault('showstopper_issues', ['Analysis Failed']); findings.setdefault('grey_area_assessment', 'N/A')
                findings.setdefault('realistic_consequences', []); findings.setdefault('mitigation_options', [])
                findings.setdefault('proceed_recommendation', 'Halt - Analysis Failed')
                self.logger.info(f"Strategic Validation: CompliantStrict={findings['is_compliant_strict']}, RiskRealistic={findings['risk_level_realistic']}, Recommendation={findings['proceed_recommendation']}")
                await self.log_knowledge_fragment(agent_source=self.AGENT_NAME, data_type="operation_validation", content=findings, tags=["validation", "risk_assessment", findings['risk_level_realistic'].lower()], relevance_score=0.85, source_reference=f"Operation: {operation_description[:50]}...")
                if findings['risk_level_realistic'] in ['High', 'Critical'] or "halt" in findings['proceed_recommendation'].lower():
                     await self.orchestrator.send_notification(f"Risk Alert: {findings['risk_level_realistic']}", f"Operation HALTED/High-Risk: {operation_description[:100]}... Issues: {findings['showstopper_issues']}, Risk: {findings['risk_level_realistic']}")
                elif findings['risk_level_realistic'] == 'Medium' or findings.get('mitigation_options'):
                     await self.orchestrator.send_notification("Strategic Advisory", f"Operation: {operation_description[:100]}... Risk: {findings['risk_level_realistic']}. Mitigation: {findings['mitigation_options']}")
                findings['is_compliant'] = "halt" not in findings['proceed_recommendation'].lower() # Simplified go/no-go for other agents
                return {"status": "success", "message": "Validation complete.", "findings": findings}
            except (json.JSONDecodeError, ValueError, KeyError) as e: raise RuntimeError(f"LLM validation response parsing failed: {e}. Response: {validation_result_json[:500]}")
        except Exception as e:
            self.logger.error(f"{self.AGENT_NAME}: Error during validate_operation: {e}", exc_info=True)
            await self._report_error(f"Validation failed: {e}")
            fallback_result['findings']['is_compliant'] = False
            return fallback_result

    # --- Utility Methods ---
    # MODIFIED: v3.3 - Refined legal note to reference MSA instead of explicit jurisdiction
    async def get_invoice_legal_note(self, client_country: Optional[str] = None) -> str:
        """
        Generates a strategically worded, professional legal note for invoices.
        Focuses on core protections while deferring specific jurisdiction to a separate agreement.
        """
        self.logger.debug(f"Generating STRATEGIC invoice legal note (Client Country: {client_country or 'Unknown'}).")

        # Fetch necessary details (assuming they are configured)
        sender_name = self.config.get('SENDER_NAME', 'Nolli Agency') # Use configured name
        bank_account_info = self.config.get('MOROCCAN_BANK_ACCOUNT', '[Configure Bank Details]')
        # W8 details (Name, Country) are available via self.config.get but omitted from note text
        # W8 Address/TIN are also available via self.config but not used in this note

        # --- Revised Pro-Agency Terms (Less Specific Jurisdiction on Invoice) ---
        # Focus: Finality, Non-Refundable, Limited Liability, Reference to Agreement
        terms = (
            f"SERVICE PROVIDER: {sender_name}. PAYMENT TO: {bank_account_info}. "
            "PAYMENT CONSTITUTES FINAL AND IRREVOCABLE ACCEPTANCE of services rendered or commenced. "
            "ALL PAYMENTS ARE NON-REFUNDABLE. NO CHARGEBACKS. " # Keep strong non-refundable clause
            "Provider liability is strictly limited to the total service fee paid for this invoice. "
            "Provider is not liable for any indirect, consequential, or incidental damages. "
            "This transaction and all services are governed by the terms of the Master Service Agreement previously agreed upon by the parties, which includes governing law and dispute resolution clauses." # Points to external agreement
        )
        # --- End Revised Terms ---

        final_note = f"Payment Terms: Due Upon Receipt. See Master Service Agreement for full terms. Key Terms Summary: {terms}"
        self.logger.info(f"Generated Revised Strategic Invoice Note (Client Country: {client_country or 'Unknown'})")
        return final_note

    def _parse_llm_json(self, json_string: str, expect_type: Type = dict) -> Union[Dict, List, None]:
        """Safely parses JSON from LLM output, handling markdown code blocks."""
        # (Implementation remains the same as v3.2)
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
                    self.logger.warning(f"Initial JSON parsing failed ({e}), cleaning: {potential_json[:100]}..."); cleaned_json = re.sub(r',\s*([\}\]])', r'\1', potential_json); cleaned_json = re.sub(r'^\s*|\s*$', '', cleaned_json)
                    try: parsed_json = json.loads(cleaned_json)
                    except json.JSONDecodeError as e2: self.logger.error(f"JSON cleaning failed ({e2}): {potential_json[:200]}..."); return None
            elif json_string.strip().startswith(start_char) and json_string.strip().endswith(end_char):
                 try: parsed_json = json.loads(json_string)
                 except json.JSONDecodeError as e: self.logger.error(f"Direct JSON parsing failed ({e}): {json_string[:200]}..."); return None
            else: self.logger.warning(f"Could not find JSON structure ({expect_type}): {json_string[:200]}..."); return None
            if isinstance(parsed_json, expect_type): return parsed_json
            else: self.logger.error(f"Parsed JSON type mismatch. Expected {expect_type}, got {type(parsed_json)}"); return None
        except json.JSONDecodeError as e: self.logger.error(f"Failed decode LLM JSON: {e}. Snippet: {json_string[:500]}..."); return None
        except Exception as e: self.logger.error(f"Unexpected error during JSON parsing: {e}", exc_info=True); return None

    # --- Abstract Method Implementations ---
    async def learning_loop(self):
        """Periodic loop to scan for legal updates impacting strategy."""
        # (Implementation remains the same as v3.2)
        self.logger.info(f"{self.AGENT_NAME} learning loop started (Opportunistic Scan).")
        while not self._stop_event.is_set():
            try:
                interval = self.internal_state.get('update_interval_seconds', 86400 * 3)
                await asyncio.sleep(interval)
                if self._stop_event.is_set(): break
                self.logger.info("Executing LegalAgent learning loop (scan for updates)...")
                await self._scan_for_updates({}) # Trigger scan task
            except asyncio.CancelledError: self.logger.info("LegalAgent learning loop cancelled."); break
            except Exception as e:
                self.logger.error(f"Error during LegalAgent learning loop: {e}", exc_info=True)
                await self._report_error(f"Learning loop error: {e}")
                await asyncio.sleep(3600) # Wait longer after error

    async def self_critique(self) -> Dict[str, Any]:
        """Evaluates the agent's effectiveness in strategic guidance and risk assessment."""
        # (Implementation remains the same as v3.2)
        self.logger.info(f"{self.AGENT_NAME}: Performing self-critique.")
        critique = {"status": "ok", "feedback": "Critique pending analysis."}
        critique['feedback'] = "Strategic effectiveness critique not fully implemented. Review KB manually."
        return critique

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Constructs context-rich prompts for LLM calls, emphasizing strategic goals."""
        # (Implementation remains the same as v3.2)
        self.logger.debug(f"Generating dynamic prompt for LegalAgent task: {task_context.get('task')}")
        prompt_parts = [self.meta_prompt] # Use the updated meta prompt
        prompt_parts.append("\n--- Current Task Context ---")
        for key, value in task_context.items():
            value_str = ""; max_len = 2000 # Allow more context generally
            if key == 'knowledge_base_context': max_len = 4000
            elif key == 'updates_list': max_len = 5000
            if isinstance(value, str): value_str = value[:max_len] + ("..." if len(value) > max_len else "")
            elif isinstance(value, (int, float, bool)): value_str = str(value)
            elif isinstance(value, (dict, list)):
                try: value_str = json.dumps(value, default=str, indent=2); value_str = value_str[:max_len] + ("..." if len(value_str) > max_len else "")
                except TypeError: value_str = str(value)[:max_len] + "..."
            else: value_str = str(value)[:max_len] + "..."
            prompt_parts.append(f"**{key.replace('_', ' ').title()}**: {value_str}")

        prompt_parts.append("\n--- Instructions ---")
        task_type = task_context.get('task')
        if task_type == 'Analyze and recommend optimal initial corporate structure for MAX PROFIT & ASSET PROTECTION':
            prompt_parts.append("1. Analyze pros/cons/risks for Wyoming LLC, Delaware LLC, and relevant Offshore options.")
            prompt_parts.append("2. Prioritize: Minimal Tax, Strong Asset Protection, Privacy, Low Hassle, Grey Area Enablement.")
            prompt_parts.append("3. Provide clear final recommendation focused on these priorities.")
        elif task_type == 'Analyze exploitability of potential grey area strategy':
             prompt_parts.append(f"1. Analyze legal basis, **exploitability**, risks (detection, realistic penalties, reputation), benefits, implementation steps for: '{task_context.get('grey_area_topic', 'N/A')}' applied to '{task_context.get('application_context', 'N/A')}'.")
             prompt_parts.append("2. Focus on loopholes and enforcement likelihood.")
             prompt_parts.append("3. Conclude with recommendation: Exploit Aggressively, Exploit Cautiously, Monitor, or Avoid.")
        elif task_type == 'Interpret legal updates for strategic opportunities':
             prompt_parts.append("1. For each update: Identify category, summarize the *exploitable opportunity* or *new risk*, assess relevance, suggest actionable insight.")
        elif task_type == 'Validate operation compliance and assess strategic risk (PROFIT FOCUSED)':
             prompt_parts.append("1. Analyze 'Operation Description'. Focus on *realistic* risks vs. profit potential.")
             prompt_parts.append("2. Identify only *showstopper* compliance issues (high penalty/detection). Assess realistic risk level (Low/Med/High/Critical).")
             prompt_parts.append("3. Assess grey area potential. Suggest mitigation ONLY if risk is High/Critical.")
             prompt_parts.append("4. Recommend: Proceed (Low/Med Risk), Proceed Cautiously, or Halt (High/Critical Risk).")
        elif task_type == 'Generate Strategic Education Briefing':
             prompt_parts.append(f"1. Generate concise, actionable briefing on '{task_context.get('topic')}' for an ambitious operator.")
             prompt_parts.append("2. Focus on high-impact principles, real-world examples (think top operators), pitfalls, and next steps.")
             prompt_parts.append("3. Draw inspiration from advanced business/legal strategy and KB context.")
        else: prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")

        if task_context.get('desired_output_format'): prompt_parts.append(f"**Output Format:** {task_context['desired_output_format']}")
        if "JSON" in task_context.get('desired_output_format', ''): prompt_parts.append("\nRespond ONLY with valid JSON matching the specified format. Do not include explanations outside the JSON structure.\n```json")

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for LegalAgent (length: {len(final_prompt)} chars)")
        return final_prompt

    async def collect_insights(self) -> Dict[str, Any]:
        """Collects insights about legal task performance."""
        # (Implementation remains the same as v3.2)
        insights = { "agent_name": self.AGENT_NAME, "status": self.status, "timestamp": datetime.now(timezone.utc).isoformat(), "last_scan_time": self.internal_state.get('last_scan_time'), "key_observations": ["Basic status collected."] }
        return insights

    # --- KB Interaction Helpers (Delegate to ThinkTool) ---
    async def log_knowledge_fragment(self, *args, **kwargs):
        if self.think_tool and hasattr(self.think_tool, 'log_knowledge_fragment'): return await self.think_tool.log_knowledge_fragment(*args, **kwargs)
        else: self.logger.error("ThinkTool unavailable for logging KB fragment."); return None

    async def query_knowledge_base(self, *args, **kwargs):
        if self.think_tool and hasattr(self.think_tool, 'query_knowledge_base'): return await self.think_tool.query_knowledge_base(*args, **kwargs)
        else: self.logger.error("ThinkTool unavailable for querying KB."); return []

    async def log_learned_pattern(self, *args, **kwargs):
         if self.think_tool and hasattr(self.think_tool, 'log_learned_pattern'): return await self.think_tool.log_learned_pattern(*args, **kwargs)
         else: self.logger.error("ThinkTool unavailable for logging learned pattern."); return None

    async def get_latest_patterns(self, *args, **kwargs):
         if self.think_tool and hasattr(self.think_tool, 'get_latest_patterns'): return await self.think_tool.get_latest_patterns(*args, **kwargs)
         else: self.logger.error("ThinkTool unavailable for getting latest patterns."); return []

    # --- Utility Methods ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=2, max=15), retry=retry_if_exception_type(Exception))
    async def _call_llm_with_retry(self, prompt: str, model: Optional[str] = None, temperature: float = 0.5, max_tokens: int = 1500, is_json_output: bool = False) -> Optional[str]:
        """Calls the LLM via orchestrator with retry logic."""
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
            return content.strip() if content and content.strip() else None
        except Exception as e:
            self.logger.warning(f"LLM call failed (attempt): {e}")
            raise # Reraise to trigger tenacity retry

# --- End of agents/legal_agent.py ---