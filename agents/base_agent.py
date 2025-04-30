# Filename: agents/base_agent.py
# Description: Abstract Base Class for Genius Agents with core functionalities.
# Version: 1.2

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union # Added Union
from datetime import datetime, timezone

# Configure a base logger for the module
# Specific agent instances will get their own named logger in __init__
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
module_logger = logging.getLogger(__name__)

# Define KBInterface placeholder for type hinting if not centrally defined/imported
# In a real setup, this might be imported from a shared types module
class KBInterfacePlaceholder:
    async def add_knowledge(self, *args, **kwargs): pass
    async def get_knowledge(self, *args, **kwargs): return []
    async def add_email_composition(self, *args, **kwargs): pass
    async def log_learned_pattern(self, *args, **kwargs): pass
    # Add other methods expected by agents if any

KBInterface = KBInterfacePlaceholder

class GeniusAgentBase(ABC):
    """
    Abstract base class for all specialized genius agents within the Synapse system.
    Provides a common interface, shared utilities, internal reflection capability,
    and a default run loop structure.
    Version: 1.2
    """
    def __init__(self, agent_name: str, orchestrator: Optional[Any] = None, config: Optional[Any] = None, kb_interface: Optional[KBInterface] = None, session_maker: Optional[Any] = None):
        """
        Initializes the GeniusAgentBase.

        Args:
            agent_name (str): The unique name of the agent instance.
            orchestrator (Optional[Any]): Reference to the main Orchestrator instance.
                                          Provides access to config, other agents, tools, etc.
            config (Optional[Any]): Configuration object (often accessed via orchestrator).
            kb_interface (Optional[KBInterface]): An interface object for interacting
                                                  with the knowledge base (can be None).
            session_maker (Optional[Any]): Async session maker for database access,
                                           if needed directly by the agent.
        """
        if not agent_name:
            raise ValueError("Agent name cannot be empty.")

        self.agent_name = agent_name
        self.orchestrator = orchestrator
        # Prefer accessing config via orchestrator if available
        self.config = config if config is not None else getattr(orchestrator, 'config', None)
        self.kb_interface = kb_interface # May be None
        self.session_maker = session_maker # May be None

        self.status = "idle" # Common status attribute: idle, running, working, error, stopped
        # Initialize internal state dictionary for subclasses
        self.internal_state: Dict[str, Any] = {}

        # Use a logger specific to the concrete agent class instance
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}({self.agent_name})")
        self.logger.info(f"Agent '{self.agent_name}' initialized.")

    @abstractmethod
    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method for executing a specific task assigned by the Orchestrator.
        This is the primary entry point for reactive task execution.

        Args:
            task_details (dict): Contains task description, parameters, target, etc.

        Returns:
            dict: Contains execution status ('success'/'failure'), message, and results/artifacts.
        """
        pass

    @abstractmethod
    async def learning_loop(self):
        """
        Abstract method for the agent's autonomous learning and adaptation cycle.
        Implementations should contain the core logic for periodic analysis,
        strategy refinement, KB interaction, etc., and handle their own timing/sleep.
        This loop runs in the background, managed by the `run` method.
        """
        self.logger.warning(f"learning_loop not implemented for agent '{self.agent_name}'.")
        # Default behavior: do nothing periodically to prevent blocking if called
        while True:
            await asyncio.sleep(3600) # Sleep for an hour if not implemented

    @abstractmethod
    async def self_critique(self) -> Dict[str, Any]:
        """
        Abstract method for the agent to evaluate its own performance, strategy,
        or recent actions based on internal metrics, KB data, or directives.

        Returns:
            Dict[str, Any]: Critique summary, identified issues, suggested improvements.
        """
        self.logger.warning(f"self_critique not implemented for agent '{self.agent_name}'.")
        return {"status": "warning", "feedback": "Self-critique not implemented."}

    @abstractmethod
    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """
        Abstract method to construct context-rich prompts for LLM calls,
        incorporating agent state, KB insights, and task specifics.

        Args:
            task_context (Dict[str, Any]): Data relevant to the specific LLM call needed.

        Returns:
            str: The fully constructed prompt string.
        """
        self.logger.warning(f"generate_dynamic_prompt not implemented for agent '{self.agent_name}'. Returning basic prompt.")
        return f"Task Context: {json.dumps(task_context)}. Please process."

    # --- Internal Reflection Method ---
    async def _internal_think(self, thought: str):
        """
        Logs an internal thought or reflection point during task execution.
        This simulates the "think" tool concept from the article - a dedicated
        step for structured thinking/verification without external effects.

        Args:
            thought (str): A description of the internal thought process,
                           checklist verification, or reflection point.
                           Should be concise but informative.
        """
        # Use the agent's specific logger instance for this internal thought process
        # Log at INFO level to ensure visibility in standard logs.
        self.logger.info(f"[Internal Reflection] {thought.strip()}")
        # Intentionally does not perform external actions or change state beyond logging.

    # --- Default Run Loop ---
    async def run(self):
        """
        Default main run loop for an agent. Starts background learning loop (if implemented)
        and processes an internal task queue (if present in self.internal_state).
        Can be overridden by subclasses for different operational models.
        """
        if self.status == "running":
            self.logger.warning(f"Agent '{self.agent_name}' run() called while already running.")
            return

        self.logger.info(f"Agent '{self.agent_name}' run loop starting.")
        self.status = "running"
        learning_task = None

        try:
            # Start the learning loop as a background task if implemented
            if hasattr(self, 'learning_loop') and callable(self.learning_loop):
                 learning_task = asyncio.create_task(self.learning_loop(), name=f"{self.agent_name}_LearningLoop")
                 self.logger.info(f"Started background learning loop for {self.agent_name}.")
            else:
                 self.logger.info(f"No learning_loop method implemented for {self.agent_name}. Skipping background learning.")

            # Check for and process an internal task queue if the agent uses one
            if 'task_queue' in self.internal_state and isinstance(self.internal_state['task_queue'], asyncio.PriorityQueue):
                task_queue = self.internal_state['task_queue']
                self.logger.info(f"Agent '{self.agent_name}' now processing internal task queue.")
                while self.status == "running": # Continue while agent is supposed to be running
                    try:
                        # Wait for a task indefinitely until queue is non-empty or loop is cancelled
                        priority, task_data = await task_queue.get()
                        self.logger.info(f"{self.agent_name} dequeued internal task with priority {priority:.3f}: {task_data.get('type', 'Unknown')}")
                        # Execute the task in the background to keep the loop responsive
                        # Use task_details structure expected by execute_task
                        asyncio.create_task(self.execute_task(task_data), name=f"{self.agent_name}_Task_{task_data.get('id', random.randint(1000,9999))}")
                        task_queue.task_done() # Mark task done after spawning handler
                    except asyncio.QueueEmpty:
                        # Should not happen with await task_queue.get() unless queue is closed elsewhere
                        self.logger.debug(f"Agent '{self.agent_name}' task queue empty, waiting.")
                        await asyncio.sleep(1) # Short sleep before checking again
                    except Exception as task_e:
                         self.logger.error(f"Error processing task from internal queue for {self.agent_name}: {task_e}", exc_info=True)
                         # Avoid tight loop on persistent queue errors
                         await asyncio.sleep(5)

            else:
                # If no task queue, just keep agent alive and running (e.g., for background learning)
                self.logger.info(f"Agent '{self.agent_name}' entering idle monitoring state (no internal task queue).")
                while self.status == "running":
                    await asyncio.sleep(60) # Check status periodically

        except asyncio.CancelledError:
            self.logger.info(f"Agent '{self.agent_name}' run loop cancelled.")
        except Exception as e:
            self.logger.error(f"Agent '{self.agent_name}' run loop encountered critical error: {e}", exc_info=True)
            self.status = "error"
            # Report critical error to orchestrator if possible
            if self.orchestrator and hasattr(self.orchestrator, 'report_error'):
                try:
                    await self.orchestrator.report_error(self.agent_name, f"Critical run loop error: {e}")
                except Exception as report_err:
                    self.logger.error(f"Failed to report run loop error to orchestrator: {report_err}")
        finally:
            self.status = "stopped"
            if learning_task and not learning_task.done():
                learning_task.cancel()
                self.logger.info(f"Cancelled background learning loop for {self.agent_name}.")
            self.logger.info(f"Agent '{self.agent_name}' run loop finished.")

    def get_status(self) -> dict:
        """Returns the current status and basic info of the agent."""
        queue_size = -1 # Indicate N/A
        if 'task_queue' in self.internal_state and isinstance(self.internal_state['task_queue'], asyncio.PriorityQueue):
            queue_size = self.internal_state['task_queue'].qsize()

        return {
            "agent": self.agent_name,
            "status": self.status,
            "internal_queue_size": queue_size
            }

    async def collect_insights(self) -> Dict[str, Any]:
        """
        Provides a structured placeholder for agent insights.
        Subclasses MUST override this to provide meaningful operational data.
        """
        self.logger.debug(f"Agent '{self.agent_name}' collect_insights called (base placeholder).")
        return {
            "agent_name": self.agent_name,
            "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "key_observations": ["Base agent insight collection - override in subclass."],
            "errors_encountered_count": 0, # Example metric subclasses should populate
            "tasks_processed_count": 0 # Example metric
        }

# --- End of agents/base_agent.py ---