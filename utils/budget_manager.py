import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BudgetManager:
    def __init__(self):
        """Initialize budget manager with hardcoded OpenRouter and DeepSeek R1 pricing."""
        self.total_budget = 20.0  # Hardcoded $20
        self.input_cost_per_million = 0.80  # $0.80/M input tokens
        self.output_cost_per_million = 2.40  # $2.40/M output tokens
        self.used_budget = 0.0
        logging.info(f"BudgetManager initialized with total budget: ${self.total_budget}, "
                     f"input: ${self.input_cost_per_million}/M, output: ${self.output_cost_per_million}/M")

    def can_afford(self, input_tokens: int = 0, output_tokens: int = 0) -> bool:
        """Check if the budget allows the requested token usage."""
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
        total_cost = input_cost + output_cost
        can_afford = self.used_budget + total_cost <= self.total_budget
        if not can_afford:
            logging.warning(f"Budget check failed: ${self.used_budget + total_cost:.4f} exceeds ${self.total_budget}")
        return can_afford

    def log_usage(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Log token usage and update budget."""
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
        total_cost = input_cost + output_cost
        self.used_budget += total_cost
        logging.info(f"Logged usage: {input_tokens} input tokens, {output_tokens} output tokens, "
                     f"cost: ${total_cost:.4f}, remaining: ${self.get_remaining_budget():.2f}")

    def get_remaining_budget(self) -> float:
        """Return the remaining budget in dollars."""
        return self.total_budget - self.used_budget