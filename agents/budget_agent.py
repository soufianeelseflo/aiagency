
import asyncio
import logging
from datetime import datetime
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from utils.secure_storage import SecureStorage
from models import Metric
import psutil

# Configure production-grade logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("agency.log"), logging.StreamHandler()]
)

class BudgetAgent:
    """A genius-level BudgetAgent for an AI agency, ensuring financial precision and growth.

    Manages budgets, tracks expenses, and optimizes costs in real-time, fully integrated
    with the Orchestrator and other agents. Designed for a real system targeting
    $6,000 in 24 hours and $100M in 9 months, optimized for VPS constraints.
    """

    def __init__(self, session_maker, config, orchestrator, clients_models):
        self.session_maker = session_maker
        self.config = config
        self.orchestrator = orchestrator
        self.clients_models = clients_models  # List of (client, model) tuples
        self.secure_storage = SecureStorage()
        self.total_budget = 50.0
        self.alert_threshold = 0.8
        self.expense_categories = ["API", "Proxy", "Email", "Legal", "Voice", "Concurrency", "Other"]
        self.category_budgets = {
            cat: self.total_budget / len(self.expense_categories)
            for cat in self.expense_categories
        }
        self.expenditure_cache = 0.0
        logger.info("BudgetAgent initialized with total budget ${:.2f}".format(self.total_budget))

    async def track_expense(self, amount: float, category: str, description: str):
        """Track an expense with real-time validation and database persistence.

        Args:
            amount: Expense amount in dollars.
            category: Expense category (e.g., 'API', 'Proxy').
            description: Details of the expense.

        Raises:
            ValueError: If category is invalid or budget exceeded.
        """
        if category not in self.expense_categories:
            raise ValueError(f"Invalid category: {category}")

        async with self.session_maker() as session:
            # Validate category budget
            current_expenditure = await self.get_category_expenditure(session, category)
            if current_expenditure + amount > self.category_budgets[category]:
                logger.error(f"Budget exceeded for {category}: ${current_expenditure + amount:.2f} > ${self.category_budgets[category]:.2f}")
                raise ValueError(f"Budget exceeded for {category}")

            # Log expense with encrypted description
            encrypted_desc = self.secure_storage.encrypt(description)
            await session.execute(
                text("""
                    INSERT INTO expenses (amount, category, description, timestamp)
                    VALUES (:amount, :category, :description, :timestamp)
                """),
                {
                    "amount": amount,
                    "category": category,
                    "description": encrypted_desc,
                    "timestamp": datetime.utcnow()
                }
            )
            await session.commit()

            # Update cache and check thresholds
            self.expenditure_cache += amount
            total_expenditure = await self.get_total_expenditure(session)
            if total_expenditure >= self.total_budget * self.alert_threshold:
                await self.send_budget_alert(total_expenditure)
            logger.info(f"Tracked expense: ${amount:.2f} in {category} - {description}")

    async def allocate_budget(self, category: str, amount: float):
        """Dynamically allocate budget to a category, maintaining total budget integrity.

        Args:
            category: Target category.
            amount: New budget amount.

        Raises:
            ValueError: If category invalid or allocation exceeds total budget.
        """
        if category not in self.expense_categories:
            raise ValueError(f"Invalid category: {category}")

        other_budgets = sum(
            self.category_budgets[cat] for cat in self.expense_categories if cat != category
        )
        if other_budgets + amount > self.total_budget:
            raise ValueError(f"Allocation of ${amount:.2f} to {category} exceeds total budget")

        self.category_budgets[category] = amount
        logger.info(f"Allocated ${amount:.2f} to {category}")

    async def get_budget_status(self):
        """Retrieve real-time budget status for monitoring and decision-making.

        Returns:
            dict: Total and category-specific budget details.
        """
        async with self.session_maker() as session:
            total_expenditure = await self.get_total_expenditure(session)
            category_expenditures = {}
            for category in self.expense_categories:
                category_expenditures[category] = await self.get_category_expenditure(session, category)

            status = {
                "total_budget": self.total_budget,
                "total_expenditure": total_expenditure,
                "remaining_budget": self.total_budget - total_expenditure,
                "category_budgets": self.category_budgets,
                "category_expenditures": category_expenditures
            }
            logger.debug(f"Budget status: {status}")
            return status

    async def send_budget_alert(self, total_expenditure: float):
        """Notify Orchestrator when budget threshold is reached.

        Args:
            total_expenditure: Current total expenditure.
        """
        subject = "Budget Alert: Threshold Reached"
        body = f"Total expenditure: ${total_expenditure:.2f}, Budget: ${self.total_budget:.2f}, Threshold: {self.alert_threshold * 100}%"
        await self.orchestrator.send_notification(subject, body)
        logger.warning("Budget alert triggered: " + body)

        async def optimize_costs(self):
            """Dynamically adjust budgets based on system performance."""
            async with self.session_maker() as session:
                expenses = await session.execute(
                    text("SELECT category, SUM(amount) as total FROM expenses GROUP BY category")
                )
                expense_data = {row.category: row.total for row in expenses.fetchall()}
                
                # Get system insights
                optimization_insights = await self.orchestrator.agents['optimization'].get_insights()
                concurrency_limits = optimization_insights.get("concurrency_limits", {})
                scoring_insights = await self.orchestrator.agents['scoring'].get_insights()
                profit = scoring_insights.get("total_profit", 0.0)
                
                # Adjust budgets based on concurrency and profit
                total_current_budget = sum(self.category_budgets.values())
                for category in self.expense_categories:
                    current_expense = expense_data.get(category, 0.0)
                    predicted = current_expense * 1.1  # 10% growth assumption
                    
                    if category == "Concurrency":
                        active_agents = len([agent for agent, limit in concurrency_limits.items() if limit > 1])
                        predicted += active_agents * 0.5  # Extra $0.50 per active agent
                    
                    if predicted > self.category_budgets[category] and profit > total_current_budget:
                        new_budget = min(predicted * 1.2, self.total_budget * 0.3)  # Cap at 30% of total
                        await self.allocate_budget(category, new_budget)
                        logger.info(f"Adjusted {category} budget to ${new_budget:.2f} due to demand.")
                    elif current_expense < self.category_budgets[category] * 0.5:
                        new_budget = max(self.total_budget / len(self.expense_categories), current_expense * 1.1)
                        await self.allocate_budget(category, new_budget)
                        logger.info(f"Reduced {category} budget to ${new_budget:.2f} due to low usage.")

    async def get_total_expenditure(self, session: AsyncSession) -> float:
        """Calculate total expenditure from database.

        Args:
            session: Active database session.

        Returns:
            float: Total expenditure.
        """
        result = await session.execute(text("SELECT SUM(amount) FROM expenses"))
        total = result.scalar() or 0.0
        self.expenditure_cache = total  # Sync cache
        return total

    async def get_category_expenditure(self, session: AsyncSession, category: str) -> float:
        """Calculate expenditure for a specific category.

        Args:
            session: Active database session.
            category: Target category.

        Returns:
            float: Category expenditure.
        """
        result = await session.execute(
            text("SELECT SUM(amount) FROM expenses WHERE category = :category"),
            {"category": category}
        )
        return result.scalar() or 0.0

    async def adjust_concurrency(self):
        """Dynamically adjust operations based on VPS resource usage."""
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        if cpu_usage > 70 or memory_usage > 70:
            return min(5, self.orchestrator.concurrency_limit)
        elif cpu_usage < 30 and memory_usage < 30:
            return min(15, self.orchestrator.concurrency_limit)
        return self.orchestrator.concurrency_limit

    async def run(self):
        """Run BudgetAgent continuously with hourly insights and optimization.

        Respects manual approval and VPS limits, with robust error handling.
        """
        while True:
            try:
                if self.orchestrator.approved:  # Honor your manual approval requirement
                    concurrency = await self.adjust_concurrency()
                    async with asyncio.Semaphore(concurrency):
                        await self.optimize_costs()
                        status = await self.get_budget_status()
                        # Log metrics for Grafana
                        async with self.session_maker() as session:
                            metric = Metric(
                                agent_name="budget",
                                timestamp=datetime.utcnow(),
                                metric_name="budget_status",
                                value=str(status)
                            )
                            session.add(metric)
                            await session.commit()
                    logger.info("Budget cycle completed.")
                else:
                    logger.info("Awaiting manual approval from Orchestrator.")
                await asyncio.sleep(3600)  # Hourly, practical for insights
            except Exception as e:
                logger.error(f"BudgetAgent run error: {e}")
                await self.orchestrator.report_error("BudgetAgent", str(e))
                await asyncio.sleep(60)  # Retry after 1 minute