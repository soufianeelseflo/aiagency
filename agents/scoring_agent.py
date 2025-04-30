# Filename: agents/scoring_agent.py
# Description: Agentic Scoring Agent capable of calculating profit/scores
#              and autonomously learning/adapting its scoring model.
# Version: 3.0 (Agentic Learning Implemented)

import asyncio
import logging
import json
from datetime import datetime, timedelta, timezone
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import text as sql_text, select, case, func, Float # Import Float for casting
from sqlalchemy.exc import SQLAlchemyError
from collections import Counter
import random # For jitter

# --- Project Imports ---
try:
    from .base_agent import GeniusAgentBase, KBInterface # Use relative import
except ImportError:
    from base_agent import GeniusAgentBase, KBInterface # Fallback

from models import Metric, Client, EmailLog, CallLog, Invoice, KnowledgeFragment, LearnedPattern # Import necessary models
# Secure storage might not be needed if profit isn't encrypted for return
# from utils.secure_storage import SecureStorage
from typing import Optional, Dict, Any, List, Tuple # Ensure necessary types

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

class ScoringAgent(GeniusAgentBase):
    """
    Scoring Agent (Genius Level): Calculates profitability, client engagement scores,
    and provides system performance insights. Autonomously learns and adapts scoring
    parameters based on performance data.
    Version: 3.0
    """
    AGENT_NAME = "ScoringAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any):
        """Initializes the ScoringAgent."""
        config = getattr(orchestrator, 'config', None)
        kb_interface = getattr(orchestrator, 'kb_interface', None)
        super().__init__(agent_name=self.AGENT_NAME, kb_interface=kb_interface, orchestrator=orchestrator, config=config)

        self.session_maker = session_maker
        self.think_tool = orchestrator.agents.get('think') # For LLM calls if needed

        # --- Internal State Initialization ---
        self.internal_state = getattr(self, 'internal_state', {})
        # Scoring Parameters (Load defaults, potentially load from KB/DB later)
        self.internal_state['weights'] = self.config.get("SCORING_WEIGHTS", {"email_response": 1.0, "call_success": 2.5, "invoice_paid": 5.0})
        self.internal_state['decay_rate'] = float(self.config.get("SCORING_DECAY_RATE_PER_DAY", 0.05))
        self.internal_state['learning_interval_seconds'] = int(self.config.get("SCORING_LEARNING_INTERVAL_S", 3600 * 4)) # Learn every 4 hours
        self.internal_state['last_learning_run'] = None
        self.internal_state['last_critique_run'] = None
        self.internal_state['critique_interval_seconds'] = int(self.config.get("SCORING_CRITIQUE_INTERVAL_S", 3600 * 24)) # Critique daily

        self.logger.info(f"{self.AGENT_NAME} v3.0 (Agentic Learning) initialized. Decay: {self.internal_state['decay_rate']}, Weights: {self.internal_state['weights']}")

    async def log_operation(self, level: str, message: str):
        """Helper to log to the operational log file with agent context."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err:
            print(f"OPERATIONAL LOG FAILED ({self.agent_name}): {level} - {message} | Error: {log_err}")
            logger.error(f"Failed to write to operational log from {self.agent_name}: {log_err}")

    async def calculate_total_profit(self, session: AsyncSession) -> float:
        """Calculate the total profit from paid invoices. Returns raw float value."""
        profit = 0.0
        try:
            stmt = select(func.sum(Invoice.amount)).where(Invoice.status == 'paid')
            result = await session.execute(stmt)
            profit = result.scalar() or 0.0
            self.logger.info(f"Calculated total profit: ${profit:.2f}")
            # Metric logging happens in get_insights after commit
        except SQLAlchemyError as e:
            self.logger.error(f"Database error calculating total profit: {e}", exc_info=True)
            raise # Propagate
        except Exception as e:
            self.logger.error(f"Unexpected error calculating total profit: {e}", exc_info=True)
            raise # Propagate
        return float(profit)

    async def score_client(self, client_id: int) -> float:
        """
        Compute a client’s score based on recent interactions using current weights/decay.
        """
        await self._internal_think(f"Calculating engagement score for Client ID: {client_id} using current parameters.")
        score = 0.0
        now_utc = datetime.now(timezone.utc)
        # Use current parameters from internal state
        weights = self.internal_state['weights']
        decay_rate = self.internal_state['decay_rate']
        try:
            async with self.session_maker() as session:
                # Email responses
                email_stmt = select(EmailLog.timestamp).where(
                    EmailLog.client_id == client_id, EmailLog.status == 'responded'
                )
                email_results = await session.execute(email_stmt)
                for row in email_results.mappings().all():
                    days_ago = max(0, (now_utc - row.timestamp).total_seconds() / 86400.0) # Ensure non-negative
                    score += weights.get("email_response", 1.0) * np.exp(-decay_rate * days_ago)

                # Successful calls
                call_stmt = select(CallLog.timestamp).where(
                    CallLog.client_id == client_id, CallLog.outcome == 'success'
                )
                call_results = await session.execute(call_stmt)
                for row in call_results.mappings().all():
                    days_ago = max(0, (now_utc - row.timestamp).total_seconds() / 86400.0)
                    score += weights.get("call_success", 2.5) * np.exp(-decay_rate * days_ago)

                # Paid invoices
                invoice_stmt = select(Invoice.timestamp).where(
                    Invoice.client_id == client_id, Invoice.status == 'paid'
                )
                invoice_results = await session.execute(invoice_stmt)
                for row in invoice_results.mappings().all():
                    days_ago = max(0, (now_utc - row.timestamp).total_seconds() / 86400.0)
                    score += weights.get("invoice_paid", 5.0) * np.exp(-decay_rate * days_ago)

            self.logger.debug(f"Calculated score for Client {client_id}: {score:.3f}")
            return round(score, 3)

        except SQLAlchemyError as e:
            self.logger.error(f"Database error scoring client {client_id}: {e}", exc_info=True)
            return 0.0
        except Exception as e:
            self.logger.error(f"Unexpected error scoring client {client_id}: {e}", exc_info=True)
            return 0.0

    async def score_all_clients(self) -> Dict[int, float]:
        """Efficiently compute scores for all clients using current parameters."""
        await self._internal_think("Calculating scores for all clients using current parameters.")
        scores: Dict[int, float] = {}
        now_utc = datetime.now(timezone.utc)
        weights = self.internal_state['weights']
        decay_rate = self.internal_state['decay_rate']
        try:
            async with self.session_maker() as session:
                # Email responses
                email_stmt = select(EmailLog.client_id, EmailLog.timestamp).where(EmailLog.status == 'responded')
                email_results = await session.execute(email_stmt)
                for row in email_results.mappings().all():
                    client_id = row.client_id; ts = row.timestamp
                    if client_id is None or ts is None: continue
                    days_ago = max(0, (now_utc - ts).total_seconds() / 86400.0)
                    scores[client_id] = scores.get(client_id, 0.0) + weights.get("email_response", 1.0) * np.exp(-decay_rate * days_ago)

                # Successful calls
                call_stmt = select(CallLog.client_id, CallLog.timestamp).where(CallLog.outcome == 'success')
                call_results = await session.execute(call_stmt)
                for row in call_results.mappings().all():
                    client_id = row.client_id; ts = row.timestamp
                    if client_id is None or ts is None: continue
                    days_ago = max(0, (now_utc - ts).total_seconds() / 86400.0)
                    scores[client_id] = scores.get(client_id, 0.0) + weights.get("call_success", 2.5) * np.exp(-decay_rate * days_ago)

                # Paid invoices
                invoice_stmt = select(Invoice.client_id, Invoice.timestamp).where(Invoice.status == 'paid')
                invoice_results = await session.execute(invoice_stmt)
                for row in invoice_results.mappings().all():
                    client_id = row.client_id; ts = row.timestamp
                    if client_id is None or ts is None: continue
                    days_ago = max(0, (now_utc - ts).total_seconds() / 86400.0)
                    scores[client_id] = scores.get(client_id, 0.0) + weights.get("invoice_paid", 5.0) * np.exp(-decay_rate * days_ago)

            rounded_scores = {cid: round(s, 3) for cid, s in scores.items()}
            self.logger.info(f"Scored {len(rounded_scores)} clients.")
            return rounded_scores

        except SQLAlchemyError as e:
            self.logger.error(f"Database error scoring all clients: {e}", exc_info=True)
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error scoring all clients: {e}", exc_info=True)
            return {}

    async def get_insights(self) -> Dict[str, Any]:
        """Provide insights including total profit and average client score."""
        await self._internal_think("Generating insights: Calculating total profit and average client score.")
        total_profit = 0.0
        avg_score = 0.0
        scores = {}
        try:
            async with self.session_maker() as session:
                async with session.begin(): # Use transaction for profit calc + metric log
                    total_profit = await self.calculate_total_profit(session)
                    # Commit happens automatically

            scores = await self.score_all_clients()
            if scores:
                avg_score = np.mean(list(scores.values())) if scores else 0.0

        except Exception as e:
            self.logger.error(f"Failed to generate full insights due to error: {e}")

        insights = {
            "agent_name": self.AGENT_NAME, "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_profit": round(total_profit, 2), "average_client_score": round(avg_score, 3),
            "scored_clients_count": len(scores)
        }
        self.logger.info(f"Insights generated: Profit=${insights['total_profit']}, AvgScore={insights['average_client_score']}, Count={insights['scored_clients_count']}")
        return insights

    # --- Abstract Method Implementations ---

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handles scoring-related tasks."""
        action = task_details.get('action')
        self.logger.info(f"{self.AGENT_NAME} received task: {action}")
        self.status = "working"
        result = {"status": "failure", "message": f"Unsupported action for ScoringAgent: {action}"}

        try:
            if action == 'get_client_score':
                client_id = task_details.get('client_id')
                if client_id:
                    score = await self.score_client(client_id)
                    result = {"status": "success", "score": score}
                else: result["message"] = "Missing client_id for get_client_score"
            elif action == 'get_system_insights':
                 insights = await self.get_insights()
                 result = {"status": "success", "insights": insights}
            else: self.logger.warning(f"Unsupported action '{action}' for ScoringAgent.")

        except Exception as e:
             self.logger.error(f"Error executing task '{action}': {e}", exc_info=True)
             result = {"status": "error", "message": str(e)}

        self.status = "idle"
        return result

    async def learning_loop(self):
        """Agentic Learning Loop: Periodically analyzes performance to adapt scoring."""
        self.logger.info(f"{self.AGENT_NAME} learning loop started.")
        while True:
            try:
                interval = self.internal_state.get('learning_interval_seconds', 3600 * 4)
                await asyncio.sleep(interval)
                current_time = datetime.now(timezone.utc)
                self.internal_state['last_learning_run'] = current_time
                self.logger.info("ScoringAgent Learning Loop: Starting analysis cycle.")

                # --- 1. Fetch Performance Data ---
                # Get recent client interactions and outcomes (e.g., last 30 days)
                await self._internal_think("Fetching recent client interactions and outcomes for scoring model adaptation.")
                performance_data = [] # List of tuples: (client_id, score_at_time, outcome_value)
                analysis_start_date = current_time - timedelta(days=30)

                async with self.session_maker() as session:
                    # Example: Get clients with recent paid invoices
                    stmt_paid = select(Invoice.client_id, Invoice.timestamp).where(
                        Invoice.status == 'paid', Invoice.timestamp >= analysis_start_date
                    )
                    paid_results = await session.execute(stmt_paid)
                    clients_paid = {row.client_id: row.timestamp for row in paid_results.mappings().all()}

                    # Example: Get clients with recent successful calls
                    stmt_calls = select(CallLog.client_id, CallLog.timestamp).where(
                        CallLog.outcome == 'success', CallLog.timestamp >= analysis_start_date
                    )
                    call_results = await session.execute(stmt_calls)
                    clients_called_success = {row.client_id: row.timestamp for row in call_results.mappings().all()}

                    # Example: Get clients with recent email responses
                    stmt_emails = select(EmailLog.client_id, EmailLog.timestamp).where(
                        EmailLog.status == 'responded', EmailLog.timestamp >= analysis_start_date
                    )
                    email_results = await session.execute(stmt_emails)
                    clients_replied = {row.client_id: row.timestamp for row in email_results.mappings().all()}

                    # Combine and fetch scores (this could be optimized)
                    all_relevant_clients = set(clients_paid.keys()) | set(clients_called_success.keys()) | set(clients_replied.keys())
                    if not all_relevant_clients:
                         self.logger.info("Learning Loop: No relevant client activity found in the last 30 days.")
                         continue

                    # Fetch scores for these clients (could be computationally intensive)
                    # Ideally, we'd have historical scores, but let's recalculate for now
                    client_scores = await self.score_all_clients() # Get current scores

                    for client_id in all_relevant_clients:
                        score = client_scores.get(client_id, 0.0)
                        # Define outcome value (e.g., 1 for paid invoice, 0.5 for successful call/reply, 0 otherwise)
                        outcome_value = 0.0
                        if client_id in clients_paid: outcome_value = 1.0
                        elif client_id in clients_called_success: outcome_value = 0.5
                        elif client_id in clients_replied: outcome_value = 0.2 # Lower value for reply only
                        performance_data.append({'client_id': client_id, 'score': score, 'outcome': outcome_value})

                if not performance_data:
                    self.logger.info("Learning Loop: No performance data points generated for analysis.")
                    continue

                self.logger.info(f"Learning Loop: Analyzing {len(performance_data)} performance data points.")

                # --- 2. Correlation Analysis & Parameter Adjustment ---
                # Simple analysis: Check if higher scores correlate with better outcomes
                # More advanced: Regression analysis to find optimal weights
                await self._internal_think("Analyzing correlation between scores and outcomes to potentially adjust scoring weights.")
                # Placeholder for analysis logic - This is complex
                # Example: If avg score of clients with outcome > 0 is significantly higher than avg score of others, model is okay.
                # If not, maybe adjust weights. If avg score of paid clients is low, increase invoice_paid weight?
                # This requires statistical analysis or potentially an LLM call to interpret the data.

                # --- Simulated Adjustment ---
                # Let's simulate a small adjustment based on simple observation
                paid_scores = [p['score'] for p in performance_data if p['outcome'] == 1.0]
                avg_paid_score = np.mean(paid_scores) if paid_scores else 0
                avg_overall_score = np.mean([p['score'] for p in performance_data]) if performance_data else 0

                adjustment_made = False
                if avg_paid_score < avg_overall_score * 1.1 and len(paid_scores) > 5: # If paid clients aren't scoring significantly higher
                    old_weight = self.internal_state['weights'].get('invoice_paid', 5.0)
                    new_weight = min(old_weight * 1.1, 10.0) # Increase weight slightly, cap at 10
                    if abs(new_weight - old_weight) > 0.1:
                        self.internal_state['weights']['invoice_paid'] = new_weight
                        adjustment_made = True
                        self.logger.info(f"Learning Loop: Adjusted 'invoice_paid' weight to {new_weight:.2f} (Avg paid score {avg_paid_score:.2f} vs overall {avg_overall_score:.2f})")
                        await self.log_learned_pattern(
                            pattern_description=f"Adjusted scoring weight for 'invoice_paid' to {new_weight:.2f} based on correlation analysis.",
                            supporting_fragment_ids=[], # Link to analysis fragment if logged
                            confidence_score=0.6,
                            implications="Scoring model adapted to better reflect payment success.",
                            tags=["scoring_model", "adaptation", "learning_loop"]
                        )

                if not adjustment_made:
                    self.logger.info("Learning Loop: No significant scoring parameter adjustments made this cycle.")

                # TODO: Persist updated weights if needed beyond agent memory

            except asyncio.CancelledError:
                self.logger.info(f"{self.AGENT_NAME} learning loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in ScoringAgent learning loop: {e}", exc_info=True)
                await asyncio.sleep(60 * 15) # Wait longer after error

    async def self_critique(self) -> Dict[str, Any]:
        """Evaluates the accuracy and performance impact of scoring."""
        self.logger.info(f"{self.AGENT_NAME}: Performing self-critique.")
        critique = {"status": "ok", "feedback": "Critique pending analysis."}
        try:
            insights = await self.get_insights() # Get current metrics
            # --- LLM-based Critique ---
            await self._internal_think("Generating self-critique using LLM based on current insights and scoring parameters.")
            critique_context = {
                "task": "Critique Scoring Agent Performance",
                "current_insights": insights,
                "current_weights": self.internal_state['weights'],
                "current_decay_rate": self.internal_state['decay_rate'],
                "last_learning_run": self.internal_state.get('last_learning_run'),
                "desired_output_format": "JSON: { \"overall_assessment\": str, \"strengths\": list[str], \"weaknesses\": list[str], \"suggestions_for_learning_loop\": list[str] }"
            }
            critique_prompt = await self.generate_dynamic_prompt(critique_context)
            critique_json = await self.think_tool._call_llm_with_retry(
                 critique_prompt, model=self.config.get("OPENROUTER_MODELS", {}).get('agent_critique', "google/gemini-pro"),
                 temperature=0.5, max_tokens=500, is_json_output=True
            )

            if critique_json:
                 try:
                     critique_result = json.loads(critique_json[critique_json.find('{'):critique_json.rfind('}')+1])
                     critique['feedback'] = critique_result.get('overall_assessment', 'LLM critique generated.')
                     critique['details'] = critique_result # Store full critique
                     self.logger.info(f"Self-Critique Assessment: {critique['feedback']}")
                     # Log critique summary
                     await self.log_knowledge_fragment(
                         agent_source=self.AGENT_NAME, data_type="self_critique_summary",
                         content=critique_result, tags=["critique", "scoring"], relevance_score=0.8
                     )
                 except (json.JSONDecodeError, ValueError, KeyError) as e:
                      self.logger.error(f"Failed to parse self-critique LLM response: {e}")
                      critique['feedback'] += " Failed to parse LLM critique."
            else:
                 critique['feedback'] += " LLM critique call failed."
                 critique['status'] = 'error'

        except Exception as e:
            self.logger.error(f"Error during self-critique: {e}", exc_info=True)
            critique['status'] = 'error'
            critique['feedback'] = f"Self-critique failed: {e}"

        return critique

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Generates prompts for LLM calls if ScoringAgent needed them (e.g., for critique)."""
        self.logger.debug(f"Generating dynamic prompt for ScoringAgent task: {task_context.get('task')}")
        # ScoringAgent primarily uses data, but might use LLM for critique/analysis interpretation
        prompt_parts = [f"You are acting as part of the ScoringAgent ({self.AGENT_NAME}). Your goal is data analysis and performance assessment."]

        prompt_parts.append("\n--- Current Task Context ---")
        for key, value in task_context.items():
             if isinstance(value, (str, int, float, bool)): prompt_parts.append(f"{key.replace('_', ' ').title()}: {value}")
             elif isinstance(value, dict) and len(json.dumps(value)) < 500: prompt_parts.append(f"{key.replace('_', ' ').title()}: {json.dumps(value, indent=2)}")
             elif isinstance(value, list) and len(json.dumps(value)) < 500: prompt_parts.append(f"{key.replace('_', ' ').title()}: {json.dumps(value)}")
             # Add more specific handling if needed

        prompt_parts.append("\n--- Instructions ---")
        task_type = task_context.get('task')
        if task_type == 'Critique Scoring Agent Performance':
            prompt_parts.append("1. Analyze the provided insights, weights, and decay rate.")
            prompt_parts.append("2. Assess the overall effectiveness of the current scoring model based on profit and average score trends (if inferrable).")
            prompt_parts.append("3. Identify strengths (e.g., high correlation if data provided) and weaknesses (e.g., low avg score despite high profit, potential bias in weights).")
            prompt_parts.append("4. Suggest specific areas or metrics for the next learning loop cycle to focus on.")
            prompt_parts.append(f"5. **Output Format:** {task_context.get('desired_output_format')}")
        else:
            prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")

        if "JSON" in task_context.get('desired_output_format', ''): prompt_parts.append("\n```json")

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for ScoringAgent (length: {len(final_prompt)} chars)")
        return final_prompt

    # --- KB Interaction Helpers (Delegate or Direct) ---
    async def log_knowledge_fragment(self, *args, **kwargs):
        """Logs a knowledge fragment using the best available method."""
        if self.kb_interface and hasattr(self.kb_interface, 'log_knowledge_fragment'):
            return await self.kb_interface.log_knowledge_fragment(*args, **kwargs)
        elif self.think_tool and hasattr(self.think_tool, 'log_knowledge_fragment'):
            return await self.think_tool.log_knowledge_fragment(*args, **kwargs)
        else: self.logger.error("No mechanism available to log knowledge fragment."); return None

    async def query_knowledge_base(self, *args, **kwargs):
        """Queries the knowledge base using the best available method."""
        if self.kb_interface and hasattr(self.kb_interface, 'query_knowledge_base'):
            return await self.kb_interface.query_knowledge_base(*args, **kwargs)
        elif self.think_tool and hasattr(self.think_tool, 'query_knowledge_base'):
            return await self.think_tool.query_knowledge_base(*args, **kwargs)
        else: self.logger.error("No mechanism available to query knowledge base."); return []

    async def log_learned_pattern(self, *args, **kwargs):
         """Logs a learned pattern using the best available method."""
         if self.kb_interface and hasattr(self.kb_interface, 'log_learned_pattern'):
             return await self.kb_interface.log_learned_pattern(*args, **kwargs)
         elif self.think_tool and hasattr(self.think_tool, 'log_learned_pattern'):
             return await self.think_tool.log_learned_pattern(*args, **kwargs)
         else: self.logger.error("No mechanism available to log learned pattern."); return None

