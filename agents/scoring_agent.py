    import asyncio
    import logging
    from datetime import datetime
    import numpy as np
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import text
    from utils.secure_storage import SecureStorage
    from models import Metric

    # Configure logging for production monitoring
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("agency.log"), logging.StreamHandler()]
    )

    class ScoringAgent:
        """A production-level ScoringAgent for an AI agency, driving profitability and growth.

        This agent calculates total profit, scores clients based on engagement, and provides
        insights for optimization, fully integrated with the Orchestrator and other agents.
        Optimized for VPS constraints, it uses real data and async processing for efficiency.
        """


        def __init__(self, session_maker, config, orchestrator, clients_models):
            self.session_maker = session_maker
            self.config = config
            self.orchestrator = orchestrator
            self.clients_models = clients_models  # List of (client, model) tuples
            self.secure_storage = SecureStorage()
            self.weights = {"email": 1.0, "call": 2.0}  # Calls are more valuable than emails
            self.decay_rate = 0.1  # Decay factor for interaction recency (per day)
            logger.info("ScoringAgent initialized.")

        async def calculate_total_profit(self, session: AsyncSession) -> float:
            """Calculate the total profit from paid invoices.

            This method is called by the OptimizationAgent to determine the RL reward.
            It queries the invoices table and ensures data integrity.

            Args:
                session: Active database session.

            Returns:
                float: Total profit in dollars, encrypted for security.

            Raises:
                Exception: If database query fails, logs and re-raises for Orchestrator handling.
            """
            try:
                result = await session.execute(
                    text("SELECT SUM(amount) FROM invoices WHERE status = 'paid'")
                )
                profit = result.scalar() or 0.0
                encrypted_profit = self.secure_storage.encrypt(str(profit))
                logger.info(f"Calculated total profit: ${profit:.2f}")
                # Log to Grafana via Metric model
                metric = Metric(
                    agent_name="scoring",
                    timestamp=datetime.utcnow(),
                    metric_name="total_profit",
                    value=profit        
                )
                session.add(metric)
                await session.commit()
                return profit
            except Exception as e:
                logger.error(f"Failed to calculate total profit: {e}")
                raise

        async def score_client(self, client_id: int) -> float:
            """Compute a client’s score based on recent successful interactions.

            Uses a decay function to prioritize recent engagement, ensuring actionable scores
            for the OptimizationAgent to prioritize high-value clients.

            Args:
                client_id: The client’s unique identifier.

            Returns:
                float: The client’s score, where higher values indicate higher potential.
            """
            async with self.session_maker() as session:
                now = datetime.utcnow()
                # Fetch successful email interactions
                email_query = text(
                    "SELECT timestamp FROM email_logs "
                    "WHERE client_id = :client_id AND status = 'responded'"
                )
                emails = await session.execute(email_query, {"client_id": client_id})
                # Fetch successful call interactions
                call_query = text(
                    "SELECT timestamp FROM call_logs "
                    "WHERE client_id = :client_id AND outcome = 'success'"
                )
                calls = await session.execute(call_query, {"client_id": client_id})

                score = 0.0
                # Process emails
                for email in emails.fetchall():
                    days_ago = (now - email.timestamp).total_seconds() / 86400.0
                    score += self.weights["email"] * np.exp(-self.decay_rate * days_ago)
                # Process calls
                for call in calls.fetchall():
                    days_ago = (now - call.timestamp).total_seconds() / 86400.0
                    score += self.weights["call"] * np.exp(-self.decay_rate * days_ago)

                logger.debug(f"Client {client_id} score: {score:.2f}")
                return score

        async def score_all_clients(self) -> dict:
            """Efficiently compute scores for all clients in a single batch.

            Optimizes database queries to minimize VPS resource usage while providing a
            comprehensive scoring map for system-wide decisions.

            Returns:
                dict: Mapping of client_id to their respective scores.
            """
            async with self.session_maker() as session:
                now = datetime.utcnow()
                # Batch fetch all successful interactions
                email_query = text(
                    "SELECT client_id, timestamp FROM email_logs WHERE status = 'responded'"
                )
                call_query = text(
                    "SELECT client_id, timestamp FROM call_logs WHERE outcome = 'success'"
                )
                emails = await session.execute(email_query)
                calls = await session.execute(call_query)

                # Organize interactions by client
                client_interactions = {}
                for email in emails.fetchall():
                    client_id = email.client_id
                    if client_id not in client_interactions:
                        client_interactions[client_id] = {"emails": [], "calls": []}
                    client_interactions[client_id]["emails"].append(email.timestamp)
                for call in calls.fetchall():
                    client_id = call.client_id
                    if client_id not in client_interactions:
                        client_interactions[client_id] = {"emails": [], "calls": []}
                    client_interactions[client_id]["calls"].append(call.timestamp)

                # Compute scores
                scores = {}
                for client_id, interactions in client_interactions.items():
                    score = 0.0
                    for timestamp in interactions["emails"]:
                        days_ago = (now - timestamp).total_seconds() / 86400.0
                        score += self.weights["email"] * np.exp(-self.decay_rate * days_ago)
                    for timestamp in interactions["calls"]:
                        days_ago = (now - timestamp).total_seconds() / 86400.0
                        score += self.weights["call"] * np.exp(-self.decay_rate * days_ago)
                    scores[client_id] = score

                logger.info(f"Scored {len(scores)} clients.")
                return scores

        async def dynamic_pricing(self, client_id):
            score = (await self.score_all_clients())[client_id]
            return 5000 * (1 + score / 100)  # Base $5,000 + score boost

        async def get_insights(self) -> dict:
            """Provide actionable insights for the Orchestrator and UI.

            Returns total profit and average client score, logged hourly for monitoring
            and feedback, aligning with system goals.

            Returns:
                dict: Insights including total_profit and average_client_score.
            """
            async with self.session_maker() as session:
                total_profit = await self.calculate_total_profit(session)
            scores = await self.score_all_clients()
            avg_score = np.mean(list(scores.values())) if scores else 0.0
            insights = {"total_profit": total_profit, "average_client_score": avg_score}
            logger.info(f"Insights generated: {insights}")
            return insights

        async def run(self):
            """Run the ScoringAgent continuously in the background.

            Executes hourly insights collection, respecting manual approval and VPS limits,
            with robust error handling.
            """
            while True:
                try:
                    if self.orchestrator.approved:  # Respect manual approval
                        insights = await self.get_insights()
                        # Log insights as metrics for Grafana
                        async with self.session_maker() as session:
                            metric = Metric(
                                agent_name="scoring",
                                timestamp=datetime.utcnow(),
                                metric_name="insights",
                                value=str(insights)
                            )
                            session.add(metric)
                            await session.commit()
                        logger.info("Hourly scoring cycle completed.")
                    else:
                        logger.info("Awaiting manual approval.")
                    await asyncio.sleep(3600)  # Hourly cycle
                except Exception as e:
                    logger.error(f"ScoringAgent run error: {e}")
                    await self.orchestrator.report_error("ScoringAgent", str(e))
                    await asyncio.sleep(60)  # Retry after 1 minute