import asyncio
import logging
from datetime import datetime, timedelta # Ensure timedelta is imported if needed elsewhere, added datetime
from typing import Dict, Any, Optional, List # Added imports
from decimal import Decimal, ROUND_HALF_UP # Added imports
import json # Added import

# Removed unused imports: numpy, sqlalchemy, SecureStorage, Metric, psutil
# Note: If sqlalchemy, SecureStorage, Metric, psutil are used by other methods not modified here (like optimize_costs, run), they should be re-added.
# Assuming they are NOT needed for the core budget logic implemented now.

# Configure production-grade logging
logger = logging.getLogger(__name__)
# Assuming logging is configured elsewhere or keeping the basic config
if not logger.hasHandlers(): # Avoid adding handlers multiple times if imported elsewhere
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s", # Improved format
        handlers=[logging.FileHandler("agency.log"), logging.StreamHandler()]
    )

class BudgetAgent:
    """A genius-level BudgetAgent for an AI agency, ensuring financial precision and growth.

    Manages budgets, tracks expenses, and provides budget checks.
    """

    def __init__(self, session_maker, config, orchestrator, clients_models):
        """Initializes the BudgetAgent."""
        self.session_maker = session_maker # Keep if needed for other methods like optimize_costs
        self.config = config
        self.orchestrator = orchestrator # Keep for potential notifications
        self.clients_models = clients_models # Keep if used elsewhere
        # self.secure_storage = SecureStorage() # Removed, not used in new methods

        self.budgets: Dict[str, Decimal] = {} # Category -> Limit
        self.expenses: Dict[str, Decimal] = {} # Category -> Spent
        self.expense_log: List[Dict[str, Any]] = [] # Log of individual expenses

        # Load initial budgets (example - refine with actual config structure)
        initial_budgets = self.config.get('BUDGETS', {
            "LLM": Decimal("100.00"),
            "API": Decimal("50.00"),
            "Resource": Decimal("50.00"), # e.g., Twilio numbers
            "Proxy": Decimal("20.00"),
            "Default": Decimal("10.00") # Fallback category
        })
        for category, limit in initial_budgets.items():
            try:
                # Ensure limit is Decimal, converting from string or float if necessary
                if isinstance(limit, (int, float)):
                    limit_decimal = Decimal(str(limit)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)
                elif isinstance(limit, str):
                     limit_decimal = Decimal(limit).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)
                elif isinstance(limit, Decimal):
                     limit_decimal = limit.quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)
                else:
                    logger.warning(f"Invalid type for budget limit '{limit}' for category '{category}'. Skipping.")
                    continue

                category_lower = category.lower()
                self.budgets[category_lower] = limit_decimal
                self.expenses[category_lower] = Decimal("0.00") # Initialize expenses
            except Exception as e:
                 logger.error(f"Error processing initial budget for category '{category}' with limit '{limit}': {e}")

        # Ensure default category exists if not provided
        if "default" not in self.budgets:
             default_limit = Decimal("10.00") # Default fallback limit
             self.budgets["default"] = default_limit
             self.expenses["default"] = Decimal("0.00")
             logger.info(f"Default budget category not found in config, added with limit: {default_limit}")


        logger.info(f"BudgetAgent initialized with budgets: { {k: float(v) for k, v in self.budgets.items()} }") # Log floats for readability
        # Removed old initializations: total_budget, alert_threshold, expense_categories, category_budgets

    async def record_expense(self, agent_name: str, amount: float, category: str, description: str) -> bool:
        """Records an expense against a budget category."""
        category_lower = category.lower()
        try:
            # Use high precision for intermediate calculations, round for storage if needed
            amount_decimal = Decimal(str(amount)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        except Exception as e:
             logger.error(f"Invalid amount format '{amount}' for expense recording: {e}")
             return False

        if amount_decimal <= 0:
            logger.warning(f"Attempted to record non-positive expense ({amount_decimal}) for {category}. Skipping.")
            return False

        # Use default category if provided one doesn't exist
        if category_lower not in self.budgets:
            original_category = category_lower
            logger.warning(f"Unknown budget category '{category}'. Using 'default'.")
            category_lower = "default"
            if category_lower not in self.budgets:
                 # This case should be prevented by __init__, but check defensively
                 logger.error("Default budget category not found. Cannot record expense.")
                 return False # Cannot record if even default is missing

        # Ensure the category exists in expenses (might be newly defaulted)
        if category_lower not in self.expenses:
            self.expenses[category_lower] = Decimal("0.00")

        self.expenses[category_lower] += amount_decimal
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": agent_name,
            "amount": float(amount_decimal), # Store as float in log for easier JSON/readability
            "category": category_lower,
            "original_category": original_category if category_lower == "default" and original_category != "default" else None, # Log original if defaulted
            "description": description,
            "cumulative_spent": float(self.expenses[category_lower]) # Store as float
        }
        self.expense_log.append(log_entry)

        # Log with precision
        log_amount_str = f"{amount_decimal:.6f}".rstrip('0').rstrip('.') # Format nicely
        log_total_str = f"{self.expenses[category_lower]:.6f}".rstrip('0').rstrip('.')
        logger.info(f"Recorded expense: {log_amount_str} in '{category_lower}' for '{agent_name}'. New total: {log_total_str}")

        # Check if budget exceeded and log/notify
        if self.expenses[category_lower] > self.budgets[category_lower]:
             limit_str = f"{self.budgets[category_lower]:.2f}"
             spent_str = f"{self.expenses[category_lower]:.6f}".rstrip('0').rstrip('.')
             logger.warning(f"Budget exceeded for category '{category_lower}'! Limit: {limit_str}, Spent: {spent_str}")
             # TODO: Trigger notification via Orchestrator?
             # Example: await self.orchestrator.notify_budget_exceeded(category_lower, self.budgets[category_lower], self.expenses[category_lower])
        return True

    async def check_budget(self, category: str, proposed_amount: float) -> bool:
        """Checks if a proposed expense is within the budget for a category."""
        category_lower = category.lower()
        try:
            amount_decimal = Decimal(str(proposed_amount)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        except Exception as e:
             logger.error(f"Invalid proposed amount format '{proposed_amount}' for budget check: {e}")
             return False

        if amount_decimal <= 0:
             logger.debug(f"Proposed amount ({amount_decimal}) is non-positive. Budget check trivially passes.")
             return True # Spending nothing is always within budget

        original_category = category_lower
        if category_lower not in self.budgets:
            logger.warning(f"Checking budget for unknown category '{category}'. Using 'default'.")
            category_lower = "default"
            if category_lower not in self.budgets:
                 logger.error("Default budget category not found. Cannot check budget.")
                 return False # Cannot approve if category doesn't exist

        limit = self.budgets[category_lower]
        # Use .get() for expenses, as a category might exist in budgets but have no expenses yet
        current_spent = self.expenses.get(category_lower, Decimal("0.00"))
        remaining = limit - current_spent

        # Format for logging
        prop_amt_str = f"{amount_decimal:.6f}".rstrip('0').rstrip('.')
        remain_str = f"{remaining:.6f}".rstrip('0').rstrip('.')

        if amount_decimal <= remaining:
            new_remain_str = f"{remaining - amount_decimal:.6f}".rstrip('0').rstrip('.')
            log_cat = f"'{category_lower}'"
            if original_category != category_lower:
                log_cat += f" (originally '{original_category}')"
            logger.debug(f"Budget check PASSED for {prop_amt_str} in {log_cat}. Remaining after proposed: {new_remain_str} (Current remaining: {remain_str})")
            return True
        else:
            log_cat = f"'{category_lower}'"
            if original_category != category_lower:
                log_cat += f" (originally '{original_category}')"
            logger.warning(f"Budget check FAILED for {prop_amt_str} in {log_cat}. Proposed amount exceeds remaining budget. Remaining: {remain_str}")
            return False

    async def get_budget_summary(self) -> Dict[str, Any]:
        """Returns a summary of current budget status."""
        summary = {}
        total_limit = Decimal("0.00")
        total_spent = Decimal("0.00")

        # Iterate through defined budgets to ensure all categories are included
        for category, limit in self.budgets.items():
            # Expenses might not exist for a category if nothing spent yet
            spent = self.expenses.get(category, Decimal("0.00"))
            remaining = limit - spent
            summary[category] = {
                # Convert Decimals to float for JSON compatibility/simpler display
                "limit": float(limit.quantize(Decimal("0.00"))),
                "spent": float(spent.quantize(Decimal("0.00"))),
                "remaining": float(remaining.quantize(Decimal("0.00")))
            }
            total_limit += limit
            total_spent += spent # Accumulate total spent from actual expenses dict

        # Calculate overall summary
        summary["overall"] = {
             "limit": float(total_limit.quantize(Decimal("0.00"))),
             "spent": float(total_spent.quantize(Decimal("0.00"))),
             "remaining": float((total_limit - total_spent).quantize(Decimal("0.00")))
        }
        logger.info("Generated budget summary.")
        # Consider logging the summary itself at DEBUG level if needed
        # logger.debug(f"Budget Summary: {json.dumps(summary, indent=2)}")
        return summary


    # --- Keeping other methods for now, but they might need refactoring ---
    # --- to use the new self.budgets/self.expenses structure ---
    # --- or be removed if their functionality is fully replaced. ---

    async def allocate_budget(self, category: str, amount: float):
        """Dynamically allocate budget to a category. (NEEDS REFACTORING for Decimal)"""
        # TODO: Refactor this method to use self.budgets (Dict[str, Decimal])
        # and ensure amount is handled as Decimal.
        category_lower = category.lower()
        try:
            amount_decimal = Decimal(str(amount)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)
        except Exception as e:
            logger.error(f"Invalid amount format '{amount}' for budget allocation: {e}")
            return # Or raise ValueError

        if category_lower not in self.budgets:
             logger.error(f"Cannot allocate budget: Invalid category '{category}'")
             return # Or raise ValueError

        # Check against total budget limit if one is defined/tracked separately
        # current_total_limit = sum(self.budgets.values())
        # potential_new_total = current_total_limit - self.budgets[category_lower] + amount_decimal
        # if potential_new_total > SOME_OVERALL_LIMIT:
        #     raise ValueError("Allocation exceeds overall budget limit")

        self.budgets[category_lower] = amount_decimal
        logger.info(f"Allocated {amount_decimal:.2f} to {category_lower}")


    async def send_budget_alert(self, category: str, limit: Decimal, spent: Decimal):
        """Notify Orchestrator when a specific budget category threshold is reached or exceeded."""
        # Modified to be more specific
        subject = f"Budget Alert: Category '{category}' Threshold Reached"
        body = f"Budget category '{category}' exceeded limit. Limit: {limit:.2f}, Spent: {spent:.2f}"
        # Assuming orchestrator has a method like this
        if hasattr(self.orchestrator, 'send_notification'):
             await self.orchestrator.send_notification(subject, body, level="WARNING") # Example
        else:
             logger.warning("Orchestrator does not have send_notification method.")
        logger.warning(f"Budget alert triggered for category '{category}': Limit={limit:.2f}, Spent={spent:.2f}")


    async def optimize_costs(self):
        """Dynamically adjust budgets based on system performance. (NEEDS REFACTORING)"""
        # TODO: Refactor this method significantly.
        # - It should use self.expenses (Dict[str, Decimal]) for current spending.
        # - It relies on orchestrator agents ('optimization', 'scoring') which might change.
        # - Database interaction for expenses is removed in the new model (in-memory log).
        # - The logic needs to be adapted to the new in-memory structure.
        logger.warning("optimize_costs method needs refactoring for the new budget structure.")
        # Placeholder logic or disable until refactored
        pass


    # Removed old DB-dependent methods: get_total_expenditure, get_category_expenditure
    # Removed old track_expense, get_budget_status

    async def run(self):
        """Run BudgetAgent continuously. (NEEDS REFACTORING)"""
        # TODO: Refactor this method.
        # - `optimize_costs` needs refactoring first.
        # - `get_budget_status` is replaced by `get_budget_summary`.
        # - Database interaction for metrics needs review (is Metric model still used?).
        # - Logic relying on orchestrator.approved might need review.
        logger.warning("BudgetAgent run loop needs refactoring for the new budget structure.")
        while True:
            try:
                # Example: Periodically log summary
                summary = await self.get_budget_summary()
                logger.info(f"Periodic Budget Summary: {json.dumps(summary)}")

                # Placeholder for potential future optimization calls
                # if getattr(self.orchestrator, 'approved', False): # Check approval if needed
                #    await self.optimize_costs() # Needs refactor

                await asyncio.sleep(3600)  # Hourly cycle (or configurable)

            except asyncio.CancelledError:
                 logger.info("BudgetAgent run loop cancelled.")
                 break
            except Exception as e:
                logger.error(f"BudgetAgent run error: {e}", exc_info=True)
                # Report error if orchestrator supports it
                if hasattr(self.orchestrator, 'report_error'):
                    try:
                        await self.orchestrator.report_error("BudgetAgent", str(e))
                    except Exception as report_e:
                        logger.error(f"Failed to report error to orchestrator: {report_e}")
                await asyncio.sleep(60) # Wait before retrying loop
