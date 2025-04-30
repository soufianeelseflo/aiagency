# Filename: agents/base_agent_prod.py
# Description: Production-Ready Abstract Base Class for Genius Agents.
# Version: 2.1 (Production Hardening)

import asyncio
import logging
import json
import uuid
import traceback # For detailed error logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple, AsyncGenerator, Type
from datetime import datetime, timezone, timedelta

# --- Type Hinting & Interfaces (Define expected structure) ---

class KBInterface:
    """Defines the expected interface for interacting with the Knowledge Base."""
    async def log_knowledge_fragment(self, *args, **kwargs) -> Optional[Any]: raise NotImplementedError
    async def query_knowledge_base(self, *args, **kwargs) -> List[Any]: raise NotImplementedError
    async def log_learned_pattern(self, *args, **kwargs) -> Optional[Any]: raise NotImplementedError
    async def get_latest_patterns(self, *args, **kwargs) -> List[Any]: raise NotImplementedError
    # Add other necessary KB interaction methods

class OrchestratorInterface:
    """Defines the expected interface for the Orchestrator."""
    config: Any # Access to settings
    agents: Dict[str, Any] # Access to other agents
    async def report_error(self, agent_name: str, error_message: str): raise NotImplementedError
    async def use_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]: raise NotImplementedError
    # Add other necessary Orchestrator interaction methods (e.g., delegate_task, get_proxy)

ConfigInterface = Any # Placeholder for actual config class/dict structure
SessionMakerInterface = Any # Placeholder for async session maker type

# --- Base Agent Class ---

class GeniusAgentBase_ProdReady(ABC):
    """
    Production-Ready Abstract Base Class for Genius Agents (v2.1).
    Provides a hardened foundation for agentic behavior: planning, reasoning,
    robust execution, learning, and operational lifecycle management.
    Designed to enable "First-Try Deployment" of derived agent implementations.
    """
    AGENT_NAME: str = "UnnamedGeniusAgent_ProdReady" # Subclasses MUST override

    # Agent Status Constants
    STATUS_INITIALIZING = "initializing"
    STATUS_IDLE = "idle"
    STATUS_PLANNING = "planning"
    STATUS_EXECUTING = "executing"
    STATUS_LEARNING = "learning" # If learning loop is active
    STATUS_STOPPING = "stopping"
    STATUS_STOPPED = "stopped"
    STATUS_ERROR = "error"

    def __init__(self,
                 agent_name: str,
                 orchestrator: OrchestratorInterface,
                 session_maker: Optional[SessionMakerInterface] = None,
                 kb_interface: Optional[KBInterface] = None,
                 config: Optional[ConfigInterface] = None):
        """
        Initializes the Production-Ready GeniusAgentBase.

        Args:
            agent_name (str): The unique name of the agent instance. MUST be overridden in subclass.
            orchestrator (OrchestratorInterface): Reference to the main Orchestrator. REQUIRED.
            session_maker (Optional[SessionMakerInterface]): Async session maker for DB access. Recommended.
            kb_interface (Optional[KBInterface]): Interface for KB interaction. Recommended for advanced agents.
            config (Optional[ConfigInterface]): Configuration object (usually accessed via orchestrator).
        """
        if not agent_name or agent_name == "UnnamedGeniusAgent_ProdReady":
            raise ValueError("Agent name must be set and unique in the subclass.")
        if orchestrator is None:
            raise ValueError("Orchestrator instance is required.")

        self.agent_name = agent_name
        self.orchestrator = orchestrator
        self.session_maker = session_maker
        self.kb_interface = kb_interface
        self.config = config if config is not None else getattr(orchestrator, 'config', None)
        if self.config is None:
             # Log is configured below, use print for critical init failure
             print(f"CRITICAL WARNING: Agent '{self.agent_name}' initialized without configuration.")

        self._status = self.STATUS_INITIALIZING
        self._run_lock = asyncio.Lock() # Prevent concurrent run calls
        self._stop_event = asyncio.Event() # Signal for graceful shutdown
        self._background_tasks = set() # Store background asyncio tasks

        # Standardized Internal State
        self.internal_state: Dict[str, Any] = {
            "task_queue": asyncio.Queue(), # Default internal task queue
            "current_task_id": None,
            "current_plan": None,        # Plan for the current task
            "errors_encountered_session": 0, # Errors since last start
            "tasks_processed_session": 0,  # Tasks processed since last start
            "last_error_ts": None,
            "last_error_details": None,
            "last_task_completion_ts": None,
            "last_learning_cycle_ts": None,
            "last_critique_ts": None,
        }

        # Agent-specific logger instance
        self.logger = logging.getLogger(f"agent.{self.agent_name}") # Standardized naming
        self.logger.info(f"Initializing (v2.1 Production Ready)...")

        # Validate essential dependencies
        if not hasattr(self.orchestrator, 'report_error'):
            self.logger.warning("Orchestrator does not have 'report_error' method. Error reporting limited.")
        if not hasattr(self.orchestrator, 'use_tool'):
            self.logger.warning("Orchestrator does not have 'use_tool' method. Tool usage disabled.")

        self._status = self.STATUS_IDLE
        self.logger.info(f"Initialization complete. Status: {self._status}")

    # --- Core Agent Lifecycle & Status ---

    @property
    def status(self) -> str:
        """Gets the current agent status."""
        return self._status

    async def start(self):
        """Starts the agent's main run loop if not already running."""
        async with self._run_lock:
            if self._status == self.STATUS_RUNNING:
                self.logger.warning("Start requested but agent is already running.")
                return
            if self._status == self.STATUS_STOPPING:
                self.logger.warning("Start requested but agent is currently stopping.")
                return

            self._status = self.STATUS_RUNNING
            self._stop_event.clear()
            self.internal_state["errors_encountered_session"] = 0
            self.internal_state["tasks_processed_session"] = 0
            self.internal_state["last_error_ts"] = None
            self.internal_state["last_error_details"] = None

            # Create and store the main run task
            run_task = asyncio.create_task(self._run_main_loop(), name=f"{self.agent_name}_MainLoop")
            self._background_tasks.add(run_task)
            self.logger.info("Main run loop started.")
            # Optionally wait for the task to finish if start() should be blocking
            # await run_task

    async def stop(self, timeout: float = 30.0):
        """Requests graceful shutdown of the agent and waits for tasks to complete."""
        if self._status in [self.STATUS_STOPPING, self.STATUS_STOPPED]:
            self.logger.info(f"Stop requested but agent is already {self._status}.")
            return

        self.logger.info(f"Stop requested. Initiating graceful shutdown (timeout: {timeout}s)...")
        self._status = self.STATUS_STOPPING
        self._stop_event.set() # Signal loops to stop

        # Wait for background tasks to finish
        tasks_to_await = list(self._background_tasks)
        if tasks_to_await:
            self.logger.debug(f"Waiting for {len(tasks_to_await)} background tasks to complete...")
            done, pending = await asyncio.wait(tasks_to_await, timeout=timeout, return_when=asyncio.ALL_COMPLETED)

            if pending:
                self.logger.warning(f"{len(pending)} background tasks did not finish within timeout:")
                for task in pending:
                    self.logger.warning(f"  - Task '{task.get_name()}' still pending. Cancelling.")
                    task.cancel()
                # Optionally wait a short extra time for cancellations
                await asyncio.sleep(1)
        else:
            self.logger.info("No active background tasks to wait for.")

        self._status = self.STATUS_STOPPED
        self._background_tasks.clear() # Clear tracked tasks
        self.logger.info("Shutdown complete.")

    # --- Core Abstract Methods (Subclasses MUST Implement) ---

    @abstractmethod
    async def plan_task(self, task_details: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        **REQUIRED:** Analyzes task details and generates a structured execution plan.
        This embodies the agent's reasoning about *how* to achieve the goal.
        Return None if the task is simple and requires no plan (handle in execute_task).
        """
        pass

    @abstractmethod
    async def execute_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        **REQUIRED:** Executes a single step from the plan.
        Handles tool execution via orchestrator, state updates, and error checking for the step.
        Must return a dictionary with at least {'status': 'success'|'failure', 'message': str}.
        """
        pass

    @abstractmethod
    async def learning_loop(self):
        """
        **REQUIRED:** Autonomous learning/adaptation cycle.
        Analyzes performance, updates strategies, interacts with KB. Handles own timing.
        Must check `self._stop_event.is_set()` periodically for graceful shutdown.
        """
        pass

    @abstractmethod
    async def self_critique(self) -> Dict[str, Any]:
        """
        **REQUIRED:** Evaluates own performance and strategy.
        Leverages internal metrics, KB data, potentially LLM analysis.
        Returns structured critique: {'status': 'ok'|'warning'|'error', 'feedback': str, 'metrics': {...}}
        """
        pass

    @abstractmethod
    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """
        **REQUIRED:** Constructs context-rich prompts for LLM calls.
        Incorporates agent state, KB insights, task specifics, strategic directives.
        """
        pass

    @abstractmethod
    async def collect_insights(self) -> Dict[str, Any]:
        """
        **REQUIRED:** Collects key operational insights for monitoring/feedback.
        Returns structured data (success rates, queue size, resource usage, etc.).
        """
        pass

    # --- Core Agentic Workflow (Handles Planning & Execution Flow) ---

    async def handle_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Primary reactive task handler: Plans, Executes Plan, Handles Results.
        Triggered by the task queue processor or directly by Orchestrator.
        """
        task_id = task_details.get('id', str(uuid.uuid4()))
        task_desc = task_details.get('description', 'Unnamed Task')
        self.internal_state['current_task_id'] = task_id
        self.logger.info(f"Handling task: {task_id} - {task_desc}")
        final_result = {"status": "failure", "message": "Task handling failed.", "task_id": task_id}
        plan = None

        try:
            # 1. Planning Phase
            self._status = self.STATUS_PLANNING
            await self._internal_think(f"Initiating planning for task {task_id}: {task_desc}")
            plan = await self.plan_task(task_details)
            self.internal_state['current_plan'] = plan # Store for visibility

            # Validate plan structure (basic check)
            if plan is not None and not isinstance(plan, list):
                 self.logger.error(f"Task {task_id}: plan_task returned invalid type ({type(plan)}), expected list or None.")
                 raise TypeError(f"Invalid plan format returned for task {task_id}.")

            if plan:
                self.logger.info(f"Task {task_id}: Plan generated with {len(plan)} steps.")
            else:
                # If plan is None, assume it's a simple task handled directly by execute_step
                # or that planning failed and an error should be raised by plan_task.
                # We proceed assuming execute_step can handle a 'None' plan or a single implicit step.
                self.logger.info(f"Task {task_id}: No explicit plan generated. Proceeding with direct execution logic if implemented.")
                # Create a dummy single step if needed by execute_step logic
                plan = [{"step": 1, "action": "Direct Execution", "tool": "agent_internal", "params": task_details}]


            # 2. Execution Phase
            self._status = self.STATUS_EXECUTING
            execution_summary = []
            all_steps_successful = True

            for step in plan:
                if self._stop_event.is_set():
                    self.logger.warning(f"Task {task_id}: Execution cancelled during step {step.get('step', '?')} due to stop signal.")
                    final_result["message"] = "Task execution cancelled."
                    final_result["status"] = "cancelled"
                    all_steps_successful = False
                    break # Exit loop if agent is stopping

                step_num = step.get("step", "?")
                step_action = step.get("action", "Unknown Step")
                self.logger.debug(f"Task {task_id}: Executing step {step_num} - {step_action}")

                await self._internal_think(f"Pre-execution check for step {step_num}: {step_action}.")
                step_result = await self.execute_step(step, task_details) # Delegate step execution
                execution_summary.append({"step": step_num, "action": step_action, "result": step_result})

                if not isinstance(step_result, dict) or 'status' not in step_result:
                     self.logger.error(f"Task {task_id}: Step {step_num} returned invalid result format: {step_result}")
                     step_result = {'status': 'failure', 'message': 'Invalid step result format'} # Standardize error

                if step_result.get("status") != "success":
                    self.logger.error(f"Task {task_id}: Step {step_num} failed: {step_result.get('message', 'Unknown step error')}")
                    all_steps_successful = False
                    final_result["message"] = f"Task failed at step {step_num}: {step_result.get('message', 'Unknown step error')}"
                    # Default: Stop plan execution on first failure
                    break

            # 3. Finalize Result
            if self._stop_event.is_set() and final_result["status"] != "cancelled":
                 final_result["status"] = "cancelled"
                 final_result["message"] = "Task cancelled during execution."
            elif all_steps_successful:
                final_result["status"] = "success"
                final_result["message"] = f"Task {task_id} completed successfully."
                # Capture final output if provided by the last step
                if execution_summary: final_result["final_output"] = execution_summary[-1].get("result", {}).get("result_data") # Example key
            # else: message already set during step failure

            self.internal_state["tasks_processed_session"] += 1
            self.internal_state["last_task_completion_ts"] = datetime.now(timezone.utc)

        except Exception as e:
            self.logger.error(f"Critical error during task {task_id} handling: {e}", exc_info=True)
            self._status = self.STATUS_ERROR # Set agent status to error
            self.internal_state["errors_encountered_session"] += 1
            self.internal_state["last_error_ts"] = datetime.now(timezone.utc)
            self.internal_state["last_error_details"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            final_result["status"] = "error"
            final_result["message"] = f"Critical error handling task {task_id}: {e}"
            await self._report_error(f"Critical task handling error: {e}", task_id) # Report to orchestrator
        finally:
            # Reset status to idle only if not in error or stopping
            if self._status not in [self.STATUS_ERROR, self.STATUS_STOPPING, self.STATUS_STOPPED]:
                 self._status = self.STATUS_IDLE
            self.internal_state['current_task_id'] = None
            self.internal_state['current_plan'] = None # Clear plan
            self.logger.info(f"Finished handling task: {task_id}. Final Status: {final_result['status']}")

        return final_result

    # --- Internal Helper Methods ---

    async def _internal_think(self, thought: str, details: Optional[Dict] = None):
        """Logs internal reasoning steps for clarity and traceability."""
        log_message = f"[Internal Reflection] {thought.strip()}"
        if details:
            try:
                details_str = json.dumps(details, indent=2, default=str)
                log_message += f"\n--- Details ---\n{details_str}\n---------------"
            except Exception: log_message += f" | Details: {str(details)}"
        self.logger.info(log_message)
        # Potential future enhancement: Log thoughts to KB for meta-analysis
        # if self.kb_interface:
        #     await self.kb_interface.log_knowledge_fragment(...)

    async def _report_error(self, error_message: str, task_id: Optional[str] = None):
        """Standardized internal error reporting to Orchestrator."""
        self.internal_state["errors_encountered_session"] += 1
        self.internal_state["last_error_ts"] = datetime.now(timezone.utc)
        self.internal_state["last_error_details"] = error_message # Store raw message
        log_msg = f"ERROR reported: {error_message}"
        if task_id: log_msg += f" (Task: {task_id})"
        self.logger.error(log_msg)

        if hasattr(self.orchestrator, 'report_error'):
            try:
                await self.orchestrator.report_error(self.agent_name, error_message)
            except Exception as report_err:
                self.logger.error(f"Failed to report error to orchestrator: {report_err}")
        else:
            self.logger.warning("Orchestrator unavailable or lacks report_error method.")

    async def _execute_tool(self, tool_name: str, params: Dict[str, Any], step_num: Optional[int] = None) -> Dict[str, Any]:
        """Safely executes a tool via the orchestrator."""
        step_id = f"Step {step_num}" if step_num else "Tool Execution"
        if not hasattr(self.orchestrator, 'use_tool'):
            err_msg = f"{step_id}: Orchestrator tool execution unavailable."
            self.logger.error(err_msg)
            return {"status": "failure", "message": err_msg}
        try:
            await self._internal_think(f"{step_id}: Executing tool '{tool_name}'", details=params)
            result = await self.orchestrator.use_tool(tool_name, params)
            if not isinstance(result, dict) or 'status' not in result:
                 self.logger.error(f"{step_id}: Tool '{tool_name}' returned invalid result format: {result}")
                 return {"status": "failure", "message": "Invalid tool result format"}
            self.logger.debug(f"{step_id}: Tool '{tool_name}' result status: {result.get('status')}")
            return result
        except Exception as e:
            self.logger.error(f"{step_id}: Exception during tool '{tool_name}' execution: {e}", exc_info=True)
            await self._report_error(f"Exception executing tool '{tool_name}': {e}")
            return {"status": "failure", "message": f"Exception executing tool '{tool_name}': {e}"}

    # --- Main Run Loop (Handles Background Tasks & Task Queue) ---

    async def _run_main_loop(self):
        """The core asynchronous loop managing background tasks and the internal queue."""
        self.logger.info("Main run loop entered.")
        learning_task = None
        queue_processor_task = None

        try:
            # --- Start Background Learning Loop ---
            if hasattr(self, 'learning_loop') and callable(self.learning_loop):
                 learning_task = asyncio.create_task(self._learning_loop_wrapper(), name=f"{self.agent_name}_LearningLoop")
                 self._background_tasks.add(learning_task)
                 self.logger.info("Started background learning loop.")
            else:
                 self.logger.info("No learning_loop method implemented. Skipping background learning.")

            # --- Start Task Queue Processor ---
            task_queue = self.internal_state.get('task_queue')
            if isinstance(task_queue, asyncio.Queue):
                queue_processor_task = asyncio.create_task(self._process_task_queue(task_queue), name=f"{self.agent_name}_QueueProcessor")
                self._background_tasks.add(queue_processor_task)
                self.logger.info("Started internal task queue processor.")
            else:
                self.logger.info("No internal task queue found. Agent will rely on direct calls or learning loop.")

            # --- Keep Loop Alive & Monitor Stop Event ---
            while not self._stop_event.is_set():
                # Check status of background tasks (optional, for restarting failed tasks)
                # Monitor external signals or perform periodic checks if needed
                await asyncio.sleep(1) # Check stop event periodically

            self.logger.info("Stop event received, exiting main loop.")

        except asyncio.CancelledError:
            self.logger.info("Main run loop cancelled.")
        except Exception as e:
            self.logger.critical(f"CRITICAL error in main run loop: {e}", exc_info=True)
            self._status = self.STATUS_ERROR
            await self._report_error(f"Critical run loop error: {e}")
        finally:
            self.logger.info("Main run loop exiting.")
            # Status is set in the stop() method or if an error occurred

    async def _learning_loop_wrapper(self):
        """Wraps the learning loop to handle errors and stop signals."""
        self.logger.info("Learning loop wrapper started.")
        try:
            while not self._stop_event.is_set():
                # Check if agent is in error state, maybe pause learning?
                if self._status == self.STATUS_ERROR:
                    self.logger.warning("Agent in error state, pausing learning loop.")
                    await asyncio.sleep(60) # Wait before checking again
                    continue

                self._status = self.STATUS_LEARNING
                await self.learning_loop() # Call the subclass implementation
                # learning_loop should handle its own sleep/timing
                if self._status == self.STATUS_LEARNING: # Reset status if learning loop didn't change it
                    self._status = self.STATUS_RUNNING
        except asyncio.CancelledError:
            self.logger.info("Learning loop cancelled.")
        except Exception as e:
            self.logger.error(f"Error in learning loop: {e}", exc_info=True)
            self._status = self.STATUS_ERROR
            await self._report_error(f"Error in learning loop: {e}")
        finally:
            self.logger.info("Learning loop wrapper finished.")
            if self._status == self.STATUS_LEARNING: self._status = self.STATUS_RUNNING # Ensure status reset if loop exits unexpectedly

    async def _process_task_queue(self, task_queue: asyncio.Queue):
        """Continuously processes tasks from the internal queue."""
        self.logger.info("Internal task queue processor started.")
        while not self._stop_event.is_set():
            try:
                # Wait for a task, but check stop event periodically
                task_data = await asyncio.wait_for(task_queue.get(), timeout=5.0)

                # Check stop event again after getting a task
                if self._stop_event.is_set():
                    self.logger.info("Stop event received while task pending, requeueing task.")
                    # Requeue the task if stopping gracefully
                    await task_queue.put(task_data)
                    break

                task_id = task_data.get('id', 'N/A')
                self.logger.info(f"Dequeued internal task {task_id}: {task_data.get('action', 'Unknown Action')}")

                # Execute task safely in the background (don't block queue processing)
                # The handle_task method contains its own robust error handling.
                asyncio.create_task(self.handle_task(task_data), name=f"{self.agent_name}_Task_{task_id}")
                # task_queue.task_done() # Only needed for JoinableQueue

            except asyncio.TimeoutError:
                continue # No task received, loop and check stop event
            except asyncio.CancelledError:
                self.logger.info("Task queue processor cancelled.")
                break
            except Exception as e:
                 # This catches errors in the queue logic itself (e.g., getting from queue)
                 self.logger.error(f"Error in task queue processor: {e}", exc_info=True)
                 self._status = self.STATUS_ERROR
                 await self._report_error(f"Error in task queue processor: {e}")
                 await asyncio.sleep(5) # Avoid tight loop on queue errors
        self.logger.info("Internal task queue processor stopped.")

    # --- Standard Utility Methods ---

    def get_status_summary(self) -> dict:
        """Returns a standardized status summary dictionary."""
        queue_size = -1
        task_queue = self.internal_state.get('task_queue')
        if isinstance(task_queue, asyncio.Queue):
            queue_size = task_queue.qsize()

        return {
            "agent": self.agent_name,
            "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_task_id": self.internal_state.get("current_task_id"),
            "queue_size": queue_size,
            "tasks_processed_session": self.internal_state.get("tasks_processed_session", 0),
            "errors_encountered_session": self.internal_state.get("errors_encountered_session", 0),
            "last_error_ts": self.internal_state.get("last_error_ts"),
        }

# --- End of agents/base_agent_prod.py ---