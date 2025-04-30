# Filename: agents/scoring_agent.py
# Description: Agent responsible for calculating profit, scoring client engagement, and providing insights.
# Version: 1.1 (Verified and Refined Implementation)

import asyncio
import logging
from datetime import datetime, timedelta, timezone
import numpy as np
import json # Added for logging insights dict
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import text, select, func, case # Added func, case
from sqlalchemy.exc import SQLAlchemyError

# Assuming utils/database.py and models.py exist as provided
# from utils.secure_storage import SecureStorage # Removed - Not encrypting profit value directly
from models import Metric, Client, EmailLog, CallLog # Import necessary models

# Base Agent Import
try:
    from .base_agent import GeniusAgentBase, KBInterface # Use relative import if applicable
except ImportError:
    from base_agent import GeniusAgentBase, KBInterface # Fallback

# Configure logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class ScoringAgent(GeniusAgentBase):
    """
    Scoring Agent (Genius Level): Calculates total profit, scores clients based on engagement,
    and provides insights for optimization. Fully integrated with the Orchestrator and database.
    Version: 1.1
    """
    AGENT_NAME = "ScoringAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any):
        """Initializes the ScoringAgent."""
        config = getattr(orchestrator, 'config', None)
        # Scoring agent might not need direct KB access, gets data from DB
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, config=config, session_maker=session_maker)

        # Load weights and decay rate from config or use defaults
        self.weights = getattr(self.config, 'SCORING_WEIGHTS', {"email": 1.0, "call": 2.5}) # e.g., Calls worth more
        self.decay_rate = float(getattr(self.config, 'SCORING_DECAY_RATE_PER_DAY', 0.05)) # Slower decay e.g. 5% per day
        self.score_log_interval_seconds = int(getattr(self.config, 'SCORING_LOG_INTERVAL_S', 3600)) # Log insights hourly

        # SecureStorage removed - not encrypting profit directly
        # self.secure_storage = SecureStorage()

        self.logger.info(f"ScoringAgent initialized. Weights: {self.weights}, Decay Rate: {self.decay_rate}")

    async def calculate_total_profit(self, session: AsyncSession) -> float:
        """
        Calculate the total profit from paid invoices. Logs the profit metric.

        Args:
            session: Active database session.

        Returns:
            float: Total profit. Returns 0.0 if no paid invoices or on error.
        """
        profit = 0.0
        try:
            # Ensure Invoice model is imported and table exists
            stmt = select(func.sum(Invoice.amount)).where(Invoice.status == 'paid')
            result = await session.execute(stmt)
            profit = result.scalar_one_or_none() or 0.0 # Handle None case gracefully
            profit = float(profit) # Ensure float type

            self.logger.info(f"Calculated total profit: ${profit:.2f}")

            # Log profit metric to database
            metric = Metric(
                agent_name=self.AGENT_NAME,
                timestamp=datetime.now(timezone.utc),
                metric_name="total_profit",
                value=str(profit) # Store as string for flexibility
            )
            session.add(metric)
            # Commit happens within the calling context (e.g., get_insights or run loop)
            # await session.commit() # Avoid commit here if called within another transaction

        except SQLAlchemyError as e:
            self.logger.error(f"Database error calculating total profit: {e}", exc_info=True)
            # Do not raise, return 0.0 and let caller handle potential issues
        except Exception as e:
            self.logger.error(f"Unexpected error calculating total profit: {e}", exc_info=True)

        return profit

    async def score_client(self, client_id: int) -> float:
        """
        Compute a client’s score based on recent successful interactions using exponential decay.
        Assumes EmailLog and CallLog tables have indexes on client_id, status/outcome, timestamp.

        Args:
            client_id: The client’s unique identifier.

        Returns:
            float: The client’s score. Returns 0.0 on error.
        """
        score = 0.0
        try:
            async with self.session_maker() as session:
                now_utc = datetime.now(timezone.utc)

                # Fetch successful email interactions (responded)
                email_stmt = select(EmailLog.timestamp).where(
                    EmailLog.client_id == client_id,
                    EmailLog.status == 'responded'
                )
                email_results = await session.execute(email_stmt)
                email_timestamps = email_results.scalars().all()

                # Fetch successful call interactions
                call_stmt = select(CallLog.timestamp).where(
                    CallLog.client_id == client_id,
                    CallLog.outcome == 'success' # Assuming 'success' indicates a positive outcome
                )
                call_results = await session.execute(call_stmt)
                call_timestamps = call_results.scalars().all()

                # Calculate score component for emails
                for ts in email_timestamps:
                    if isinstance(ts, datetime): # Ensure it's a datetime object
                        # Ensure timestamp is timezone-aware (assume UTC if not)
                        ts_aware = ts if ts.tzinfo else pytz.utc.localize(ts)
                        days_ago = (now_utc - ts_aware).total_seconds() / 86400.0
                        score += self.weights.get("email", 1.0) * np.exp(-self.decay_rate * max(0, days_ago)) # Ensure days_ago is non-negative
                    else:
                        self.logger.warning(f"Invalid timestamp type found in email_logs for client {client_id}: {type(ts)}")


                # Calculate score component for calls
                for ts in call_timestamps:
                     if isinstance(ts, datetime):
                        ts_aware = ts if ts.tzinfo else pytz.utc.localize(ts)
                        days_ago = (now_utc - ts_aware).total_seconds() / 86400.0
                        score += self.weights.get("call", 2.0) * np.exp(-self.decay_rate * max(0, days_ago))
                     else:
                        self.logger.warning(f"Invalid timestamp type found in call_logs for client {client_id}: {type(ts)}")

                self.logger.debug(f"Calculated score for Client {client_id}: {score:.4f}")

                # Optional: Update the client's score in the database directly?
                # update_stmt = update(Client).where(Client.id == client_id).values(engagement_score=score)
                # await session.execute(update_stmt)
                # await session.commit()

        except SQLAlchemyError as e:
            self.logger.error(f"Database error scoring client {client_id}: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Unexpected error scoring client {client_id}: {e}", exc_info=True)

        return score

    async def score_all_clients(self) -> dict:
        """
        Efficiently compute scores for all clients in a single batch query.
        Assumes indexes on client_id, status/outcome, timestamp for log tables.

        Returns:
            dict: Mapping of client_id to their respective scores. Returns empty dict on error.
        """
        scores = {}
        try:
            async with self.session_maker() as session:
                now_utc = datetime.now(timezone.utc)

                # Combine queries for efficiency using UNION ALL
                # Cast timestamp to a common type if necessary (e.g., TIMESTAMPTZ)
                interaction_query = text("""
                    SELECT client_id, 'email' as type, timestamp FROM email_logs WHERE status = 'responded'
                    UNION ALL
                    SELECT client_id, 'call' as type, timestamp FROM call_logs WHERE outcome = 'success'
                """)
                # Note: This query might become slow with very large log tables.
                # Consider adding time window limits (e.g., last 90 days) if performance degrades.
                # WHERE timestamp >= :ninety_days_ago

                interaction_results = await session.execute(interaction_query)
                interactions = interaction_results.mappings().all() # Fetch all results

                # Process interactions in Python
                client_scores_temp = {} # Use temp dict for accumulation
                for interaction in interactions:
                    client_id = interaction['client_id']
                    interaction_type = interaction['type']
                    ts = interaction['timestamp']

                    if not isinstance(ts, datetime):
                        self.logger.warning(f"Invalid timestamp type found during batch scoring for client {client_id}: {type(ts)}")
                        continue

                    ts_aware = ts if ts.tzinfo else pytz.utc.localize(ts) # Ensure timezone aware
                    days_ago = (now_utc - ts_aware).total_seconds() / 86400.0
                    weight = self.weights.get(interaction_type, 0.0) # Get weight, default 0 if type unknown
                    decayed_score = weight * np.exp(-self.decay_rate * max(0, days_ago))

                    client_scores_temp[client_id] = client_scores_temp.get(client_id, 0.0) + decayed_score

                scores = client_scores_temp # Assign calculated scores
                self.logger.info(f"Scored {len(scores)} clients via batch query.")

                # Optional: Batch update client scores in DB (can be resource intensive)
                # if scores:
                #     update_tasks = []
                #     for cid, score_val in scores.items():
                #         update_tasks.append({'id': cid, 'engagement_score': score_val})
                #     if update_tasks:
                #         await session.execute(update(Client).where(Client.id == text(':id')), update_tasks)
                #         await session.commit()
                #         self.logger.info(f"Batch updated engagement scores for {len(scores)} clients.")

        except SQLAlchemyError as e:
            self.logger.error(f"Database error scoring all clients: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Unexpected error scoring all clients: {e}", exc_info=True)

        return scores

    async def dynamic_pricing(self, client_id):
        """Placeholder for dynamic pricing logic."""
        # This requires more complex inputs (client score, OSINT data, service details)
        # Likely delegated to ThinkTool or a dedicated PricingAgent in a full implementation.
        self.logger.warning("dynamic_pricing called, but uses placeholder logic.")
        base_price = float(self.config.get("BASE_UGC_PRICE", 5000.0))
        score = await self.score_client(client_id) # Get individual score
        # Simple example adjustment - replace with sophisticated logic
        adjusted_price = base_price * (1 + (score / 100.0) * 0.2) # Max 20% increase based on score
        return max(base_price * 0.8, min(adjusted_price, base_price * 1.5)) # Clamp between 80% and 150% of base

    async def get_insights(self) -> dict:
        """Provide actionable insights: total profit and average client score."""
        total_profit = 0.0
        avg_score = 0.0
        scores = {}
        try:
            # Use a single session for consistency
            async with self.session_maker() as session:
                total_profit = await self.calculate_total_profit(session)
                # Need to commit the metric logged in calculate_total_profit
                await session.commit()

            # score_all_clients opens its own session
            scores = await self.score_all_clients()
            if scores:
                avg_score = np.mean(list(scores.values()))

            insights = {"total_profit": round(total_profit, 2), "average_client_score": round(avg_score, 4)}
            self.logger.info(f"Insights generated: {insights}")
            return insights
        except Exception as e:
             self.logger.error(f"Failed to generate insights: {e}", exc_info=True)
             return {"total_profit": 0.0, "average_client_score": 0.0} # Return defaults on error

    async def run(self):
        """Run the ScoringAgent continuously, logging insights periodically."""
        self.status = "running"
        self.logger.info(f"{self.AGENT_NAME} run loop starting. Logging insights every {self.score_log_interval_seconds} seconds.")
        while self.status == "running":
            try:
                is_approved = getattr(self.orchestrator, 'approved', False)
                if is_approved:
                    insights = await self.get_insights()

                    # Log average score metric separately here
                    async with self.session_maker() as session:
                        metric = Metric(
                            agent_name=self.AGENT_NAME,
                            timestamp=datetime.now(timezone.utc),
                            metric_name="average_client_score",
                            value=str(insights.get("average_client_score", 0.0))
                        )
                        session.add(metric)
                        await session.commit()
                    self.logger.info(f"Periodic scoring cycle completed. Avg Score: {insights.get('average_client_score', 0.0):.4f}")
                else:
                    self.logger.info("ScoringAgent awaiting orchestrator approval.")

                await asyncio.sleep(self.score_log_interval_seconds)

            except asyncio.CancelledError:
                self.logger.info(f"{self.AGENT_NAME} run loop cancelled.")
                self.status = "stopped"
                break
            except Exception as e:
                self.logger.error(f"{self.AGENT_NAME} run loop error: {e}", exc_info=True)
                self.status = "error"
                if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"Critical run loop error: {e}")
                await asyncio.sleep(60 * 5) # Wait longer after critical error before potentially restarting loop

        self.logger.info(f"{self.AGENT_NAME} run loop finished.")

    # --- Abstract Method Implementations (Placeholders) ---
    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.warning(f"ScoringAgent received execute_task (not primary function): {task_details}")
        # Primarily analytical, might handle specific calculation requests
        action = task_details.get('action')
        if action == 'get_client_score':
            client_id = task_details.get('client_id')
            if client_id:
                score = await self.score_client(client_id)
                return {"status": "success", "client_id": client_id, "score": score}
            else:
                return {"status": "failure", "message": "Missing client_id"}
        return {"status": "warning", "message": "ScoringAgent primarily runs analytical loops."}

    async def learning_loop(self):
        self.logger.debug(f"{self.AGENT_NAME} learning_loop: No specific learning implemented. Relies on data analysis.")
        await asyncio.sleep(3600 * 24) # Sleep long

    async def self_critique(self) -> Dict[str, Any]:
        self.logger.debug(f"{self.AGENT_NAME} self_critique: Performing basic critique.")
        insights = await self.get_insights()
        feedback = f"Critique: Profit=${insights.get('total_profit', 0):.2f}, AvgScore={insights.get('average_client_score', 0):.4f}."
        # Add checks - e.g., if avg score is declining
        return {"status": "ok", "feedback": feedback}

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        self.logger.warning(f"{self.AGENT_NAME} generate_dynamic_prompt called - not typically used.")
        return f"Context: {json.dumps(task_context)}. Scoring agent does not typically use LLM prompts directly."

# --- End of agents/scoring_agent.py ---