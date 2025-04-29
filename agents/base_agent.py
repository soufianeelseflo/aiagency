import logging
from abc import ABC, abstractmethod
import asyncio

# Configure logging
# Note: Agents might configure their own specific loggers,
# but having a base configuration can be useful.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents within the Synapse system.
    Provides a common interface and potentially shared utilities.
    """
    def __init__(self, orchestrator):
        """
        Initializes the BaseAgent.

        Args:
            orchestrator: An instance of the main Orchestrator (or similar)
                          class that provides access to shared resources like
                          tool execution, configuration, and potentially other agents.
        """
        if orchestrator is None:
            # In a real implementation, the orchestrator would be crucial.
            # For initial structure, we might allow None but log a warning.
            logger.warning("BaseAgent initialized without a valid orchestrator instance.")
        self.orchestrator = orchestrator
        logger.debug(f"{self.__class__.__name__} initialized.")

    @abstractmethod
    async def execute_task(self, task_details: dict) -> dict:
        """
        Abstract method for executing a task specific to the agent.

        Args:
            task_details (dict): A dictionary containing the details of the task
                                 to be executed, including description, parameters, etc.

        Returns:
            dict: A dictionary containing the result of the task execution,
                  including status ('success' or 'error'), details, and any
                  artifacts produced.
        """
        pass

    @abstractmethod
    def get_status(self) -> dict:
        """
        Abstract method to get the current status of the agent.

        Returns:
            dict: A dictionary representing the agent's current state (e.g.,
                  'idle', 'working', 'error', current_task).
        """
        pass
    @abstractmethod
    async def collect_insights(self) -> Dict[str, Any]:
        """
        Abstract method for collecting insights, performance data, or feedback
        from the agent's recent operations.

        This method will be called periodically by the Orchestrator's feedback loop.

        Returns:
            Dict[str, Any]: A dictionary containing relevant insights. The structure
                            can vary per agent but should be serializable and
                            understandable by ThinkTool. Examples:
                            {'errors': [...], 'success_rate': 0.85, 'key_observations': [...]}
        """
        pass

    async def run(self):
        """
        Default main run loop for an agent.
        Specific agents should override this if they need a continuous background process.
        This default implementation simply logs startup and does nothing further.
        """
        logger.info(f"{self.__class__.__name__} run loop started. No default background tasks.")
        # Option 1: Do nothing further (agent primarily reacts to execute_task)
        # Option 2: Basic sleep loop if some minimal background check is ever needed by default
        # try:
        #     while True:
        #         await asyncio.sleep(3600) # Example: sleep for an hour
        # except asyncio.CancelledError:
        #     logger.info(f"{self.__class__.__name__} run loop cancelled.")
        # except Exception as e:
        #     logger.error(f"{self.__class__.__name__} default run loop error: {e}", exc_info=True)
        # For now, let's just log and finish, assuming agents are task-driven unless they override run().
        pass

    # Potential shared utility methods could be added here later, e.g.,
    # async def _call_llm(self, prompt):
    #     if not self.orchestrator:
    #         raise RuntimeError("Orchestrator not available for LLM call.")
    #     # Assuming orchestrator has a method to handle LLM calls
    #     return await self.orchestrator.llm_call(prompt)
    #
    # async def _use_tool(self, tool_name, args):
    #     if not self.orchestrator:
    #         raise RuntimeError("Orchestrator not available for tool use.")
    #     # Assuming orchestrator has a method to handle tool use
    #     return await self.orchestrator.use_tool(tool_name, args)