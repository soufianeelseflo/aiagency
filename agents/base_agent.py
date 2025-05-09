# Filename: agents/base_agent.py
# Description: Production-Ready Abstract Base Class for Genius Agents.
# Version: 2.2 (Level 30+ Transmutation - Enhanced for PLEA, ARAA, MRE)

import asyncio
import logging
import json
import uuid
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple, AsyncGenerator, Type, Callable
from datetime import datetime, timezone, timedelta

# --- Type Hinting & Interfaces ---
class KBInterface:
    async def log_knowledge_fragment(self, *args, **kwargs) -> Optional[Any]: raise NotImplementedError
    async def query_knowledge_base(self, *args, **kwargs) -> List[Any]: raise NotImplementedError
    async def log_learned_pattern(self, *args, **kwargs) -> Optional[Any]: raise NotImplementedError
    async def get_latest_patterns(self, *args, **kwargs) -> List[Any]: raise NotImplementedError

class OrchestratorInterface:
    config: Any
    agents: Dict[str, Any]
    async def report_error(self, agent_name: str, error_message: str, task_id: Optional[str] = None): raise NotImplementedError # Added task_id
    async def use_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]: raise NotImplementedError
    async def delegate_task(self, agent_name: str, task_details: Dict[str, Any]) -> Optional[Dict[str, Any]]: raise NotImplementedError
    async def get_proxy(self, purpose: str, quality_level: str = 'standard', target_url: Optional[str] = None, specific_hint: Optional[str] = None) -> Optional[Dict[str,Any]]: raise NotImplementedError
    async def report_proxy_status(self, proxy_server_url: str, success: bool): raise NotImplementedError
    async def call_llm(self, agent_name: str, prompt: str, temperature: float = 0.5, max_tokens: int = 1024, is_json_output: bool = False, model_preference: Optional[List[str]] = None, image_data: Optional[bytes] = None, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]: raise NotImplementedError
    async def send_notification(self, title: str, message: str, level: str = "info"): raise NotImplementedError
    # Secure storage access methods (MRE)
    secure_storage: Any # Placeholder for a secure storage interface
    async def report_expense(self, agent_name: str, amount: float, category: str, description: str, task_id_reference: Optional[str]=None): raise NotImplementedError # MRE


ConfigInterface = Any
SessionMakerInterface = Callable[[], AsyncGenerator[Any, None]] # More specific type for async session maker

class GeniusAgentBase(ABC): # Renamed for clarity if used directly
    AGENT_NAME: str = "UnnamedGeniusAgent_L30"

    STATUS_INITIALIZING = "initializing"
    STATUS_IDLE = "idle"
    STATUS_PLANNING = "planning"
    STATUS_EXECUTING = "executing"
    STATUS_LEARNING = "learning"
    STATUS_STOPPING = "stopping"
    STATUS_STOPPED = "stopped"
    STATUS_ERROR = "error"
    STATUS_AWAITING_APPROVAL = "awaiting_approval" # For PCOF on critical actions

    def __init__(self,
                 agent_name: str,
                 orchestrator: OrchestratorInterface,
                 session_maker: Optional[SessionMakerInterface] = None,
                 config: Optional[ConfigInterface] = None):

        if not agent_name or agent_name == "UnnamedGeniusAgent_L30":
            raise ValueError("Agent name must be set and unique in the subclass.")
        if orchestrator is None:
            raise ValueError("Orchestrator instance is required.")

        self.agent_name = agent_name
        self.orchestrator = orchestrator
        self.session_maker = session_maker
        self.config = config if config is not None else getattr(orchestrator, 'config', None)
        if self.config is None:
            print(f"CRITICAL WARNING: Agent '{self.agent_name}' initialized without configuration.")

        self._status = self.STATUS_INITIALIZING
        self._run_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._background_tasks: set[asyncio.Task] = set() # Explicitly type hint
        self._task_queue_processor_task: Optional[asyncio.Task] = None
        self._learning_loop_task: Optional[asyncio.Task] = None

        self.internal_state: Dict[str, Any] = {
            "task_queue": asyncio.Queue(),
            "current_task_id": None,
            "current_plan": None,
            "errors_encountered_session": 0,
            "tasks_processed_session": 0,
            "last_error_ts": None,
            "last_error_details": None,
            "last_task_completion_ts": None,
            "last_learning_cycle_ts": None,
            "last_critique_ts": None,
            "meta_prompt_version": "1.0", # PLEA
            "strategy_effectiveness_rating": 0.5, # PLEA (0.0 to 1.0)
        }

        self.logger = logging.getLogger(f"agent.{self.agent_name}")
        self.op_logger = logging.getLogger('OperationalLog') # Direct access to operational logger
        self.logger.info(f"Initializing (v2.2 L30+ Transmutation)...")

        if not hasattr(self.orchestrator, 'report_error'): self.logger.warning("Orchestrator missing 'report_error'.")
        if not hasattr(self.orchestrator, 'use_tool'): self.logger.warning("Orchestrator missing 'use_tool'.")
        if not hasattr(self.orchestrator, 'delegate_task'): self.logger.warning("Orchestrator missing 'delegate_task'.")
        if not hasattr(self.orchestrator, 'call_llm'): self.logger.warning("Orchestrator missing 'call_llm'.")

        self._status = self.STATUS_IDLE
        self.logger.info(f"Initialization complete. Status: {self._status}")

    @property
    def status(self) -> str:
        return self._status

    async def start(self):
        async with self._run_lock:
            if self._status == self.STATUS_RUNNING: self.logger.warning("Start requested but agent is already running."); return
            if self._status == self.STATUS_STOPPING: self.logger.warning("Start requested but agent is currently stopping."); return

            self._status = self.STATUS_RUNNING
            self._stop_event.clear()
            self.internal_state.update({"errors_encountered_session": 0, "tasks_processed_session": 0, "last_error_ts": None, "last_error_details": None})

            # Standardized task startup
            self._task_queue_processor_task = asyncio.create_task(self._process_task_queue(self.internal_state['task_queue']), name=f"{self.agent_name}_QueueProcessor")
            self._background_tasks.add(self._task_queue_processor_task)

            if hasattr(self, 'learning_loop') and callable(self.learning_loop):
                self._learning_loop_task = asyncio.create_task(self._learning_loop_wrapper(), name=f"{self.agent_name}_LearningLoop")
                self._background_tasks.add(self._learning_loop_task)
            else: self.logger.info("No learning_loop method implemented or not callable.")

            self.logger.info(f"Agent started with {len(self._background_tasks)} background tasks.")

    async def run(self): # Default run implementation for agents that primarily process tasks or have learning loops
        """
        Default main run method. Starts background tasks if not already started.
        Agents with more complex main loops can override this.
        """
        if self._status != self.STATUS_RUNNING: # If start() wasn't called explicitly
            await self.start()
        # The main work is done in background tasks. This loop just keeps agent "alive" if needed.
        while not self._stop_event.is_set():
            await asyncio.sleep(1) # Keep alive, check stop event
        self.logger.info(f"{self.agent_name} run method exiting due to stop signal.")


    async def stop(self, timeout: float = 30.0):
        if self._status in [self.STATUS_STOPPING, self.STATUS_STOPPED]:
            self.logger.info(f"Stop requested but agent {self.agent_name} is already {self._status}.")
            return

        self.logger.info(f"{self.agent_name} stop requested. Initiating graceful shutdown (timeout: {timeout}s)...")
        self._status = self.STATUS_STOPPING
        self._stop_event.set()

        tasks_to_await = list(self._background_tasks)
        # Add specific tasks if they were managed separately and might not be in _background_tasks
        if self._task_queue_processor_task and self._task_queue_processor_task not in tasks_to_await:
            tasks_to_await.append(self._task_queue_processor_task)
        if self._learning_loop_task and self._learning_loop_task not in tasks_to_await:
            tasks_to_await.append(self._learning_loop_task)

        # Filter out None or already completed tasks before attempting to cancel/await
        valid_tasks_to_await = [task for task in tasks_to_await if task and not task.done()]

        if valid_tasks_to_await:
            self.logger.debug(f"{self.agent_name}: Attempting to cancel {len(valid_tasks_to_await)} background tasks...")
            for task in valid_tasks_to_await:
                task.cancel()
            
            done, pending = await asyncio.wait(valid_tasks_to_await, timeout=timeout, return_when=asyncio.ALL_COMPLETED)

            if pending:
                self.logger.warning(f"{self.agent_name}: {len(pending)} background tasks did not finish within timeout:")
                for task_p in pending: self.logger.warning(f"  - Task '{task_p.get_name()}' still pending.")
            if done:
                 for task_d in done:
                     try: task_d.result() # Check for exceptions in completed tasks
                     except asyncio.CancelledError: self.logger.debug(f"Task '{task_d.get_name()}' successfully cancelled.")
                     except Exception as e_task: self.logger.error(f"Task '{task_d.get_name()}' raised an exception during shutdown: {e_task}")
        else:
            self.logger.info(f"{self.agent_name}: No active background tasks to wait for.")

        self._status = self.STATUS_STOPPED
        self._background_tasks.clear()
        self.logger.info(f"{self.agent_name} shutdown complete.")

    @abstractmethod
    async def plan_task(self, task_details: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        pass

    @abstractmethod
    async def execute_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def learning_loop(self): # Specific learning logic for the agent
        pass

    @abstractmethod
    async def self_critique(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    async def collect_insights(self) -> Dict[str, Any]:
        pass

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default task execution flow if a subclass doesn't override.
        This is a simplified version; complex agents should override `handle_task` or this.
        """
        task_id = task_details.get('id', str(uuid.uuid4()))
        task_desc = task_details.get('description', 'Unnamed Task')
        self.logger.info(f"Agent {self.agent_name} executing simple task: {task_id} - {task_desc}")
        
        # For simple tasks, plan_task might return None or a single-step plan.
        # We'll treat it as if there's one step which is the task itself.
        step_result = await self.execute_step({"action": "DirectExecutionOfTask", "params": task_details}, task_details)
        
        if step_result.get("status") == "success":
            return {"status": "success", "message": f"Task {task_id} completed by direct execution.", "result_data": step_result.get("result_data")}
        else:
            return {"status": "failure", "message": f"Task {task_id} failed direct execution: {step_result.get('message')}"}


    async def handle_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task_details.get('id', str(uuid.uuid4()))
        task_desc = task_details.get('description', f"Task type: {task_details.get('action', 'Unknown')}")
        self.internal_state['current_task_id'] = task_id
        self.logger.info(f"Handling task: {task_id} - {task_desc}")
        final_result: Dict[str, Any] = {"status": "failure", "message": "Task handling failed.", "task_id": task_id}
        plan = None; original_agent_status = self.status

        try:
            # 0. Pre-Execution Approval Check (PCOF) - Example for critical actions
            if task_details.get("requires_approval", False) and not getattr(self.orchestrator, 'approved', False):
                self.logger.warning(f"Task {task_id} requires approval, but agency is not approved. Halting.")
                final_result = {"status": "halted_awaiting_approval", "message": "Task requires agency operational approval."}
                if hasattr(self, 'update_directive_status') and task_details.get('directive_id'):
                    await self.update_directive_status(task_details['directive_id'], 'awaiting_approval', final_result["message"])
                return final_result # Early exit

            self._status = self.STATUS_PLANNING
            await self._internal_think(f"Initiating planning for task {task_id}: {task_desc}", details=task_details)
            plan = await self.plan_task(task_details)
            self.internal_state['current_plan'] = plan

            if plan is not None and not isinstance(plan, list):
                 self.logger.error(f"Task {task_id}: plan_task returned invalid type ({type(plan)}).")
                 raise TypeError(f"Invalid plan format returned for task {task_id}.")
            if not plan:
                self.logger.info(f"Task {task_id}: No explicit plan. Attempting direct execution via execute_task.")
                # This ensures execute_task (which calls execute_step usually) is the main point of action
                final_result = await self.execute_task(task_details) # Use the agent's specific execute_task
                self.internal_state["tasks_processed_session"] += 1
                self.internal_state["last_task_completion_ts"] = datetime.now(timezone.utc)
                return final_result

            self.logger.info(f"Task {task_id}: Plan generated with {len(plan)} steps.")
            self._status = self.STATUS_EXECUTING
            execution_summary = []
            all_steps_successful = True

            for step_data in plan:
                if self._stop_event.is_set():
                    self.logger.warning(f"Task {task_id}: Execution cancelled during step {step_data.get('step', '?')} due to stop signal.")
                    final_result.update({"message": "Task execution cancelled.", "status": "cancelled"})
                    all_steps_successful = False; break

                step_num = step_data.get("step", "?"); step_action = step_data.get("action", "Unknown Step")
                self.logger.debug(f"Task {task_id}: Executing step {step_num} - {step_action}")
                await self._internal_think(f"Pre-execution check for step {step_num}: {step_action}.", details=step_data)
                step_result = await self.execute_step(step_data, task_details) # Pass full task_details as context
                execution_summary.append({"step": step_num, "action": step_action, "result": step_result})

                if not isinstance(step_result, dict) or 'status' not in step_result:
                     self.logger.error(f"Task {task_id}: Step {step_num} returned invalid result: {step_result}")
                     step_result = {'status': 'failure', 'message': 'Invalid step result format'}

                if step_result.get("status") != "success":
                    msg = step_result.get('message', 'Unknown step error')
                    self.logger.error(f"Task {task_id}: Step {step_num} failed: {msg}")
                    all_steps_successful = False
                    final_result["message"] = f"Task failed at step {step_num} ('{step_action}'): {msg}"
                    break
                # Propagate results from step to task_context if needed for subsequent steps
                if step_result.get("result_data"):
                    task_details.setdefault("step_outputs", {})[f"step_{step_num}_output"] = step_result["result_data"]


            if self._stop_event.is_set() and final_result["status"] != "cancelled":
                 final_result.update({"status": "cancelled", "message": "Task cancelled during execution."})
            elif all_steps_successful:
                final_result.update({"status": "success", "message": f"Task {task_id} completed successfully."})
                if execution_summary: final_result["final_output"] = execution_summary[-1].get("result", {}).get("result_data")
            
            self.internal_state["tasks_processed_session"] += 1
            self.internal_state["last_task_completion_ts"] = datetime.now(timezone.utc)

        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"Critical error during task {task_id} ('{task_desc}') handling: {error_type} - {e}"
            self.logger.error(error_msg, exc_info=True)
            self._status = self.STATUS_ERROR
            self.internal_state["errors_encountered_session"] += 1
            self.internal_state["last_error_ts"] = datetime.now(timezone.utc)
            self.internal_state["last_error_details"] = f"{error_type}: {e}\n{traceback.format_exc()}"
            final_result.update({"status": "error", "message": error_msg})
            await self._report_error(error_msg, task_id)
        finally:
            if self._status not in [self.STATUS_ERROR, self.STATUS_STOPPING, self.STATUS_STOPPED]:
                 self._status = original_agent_status if original_agent_status != self.STATUS_PLANNING else self.STATUS_IDLE
            self.internal_state['current_task_id'] = None
            self.internal_state['current_plan'] = None
            self.logger.info(f"Finished handling task: {task_id}. Final Status: {final_result.get('status')}")
            # Update directive status if this task was part of a directive
            if task_details.get('directive_id') and hasattr(self, 'update_directive_status'): # check for method
                await self.update_directive_status(task_details['directive_id'], final_result['status'], final_result['message'])

        return final_result

    async def _internal_think(self, thought: str, details: Optional[Dict] = None):
        log_message = f"[{self.agent_name} Internal Reflection] {thought.strip()}"
        if details:
            try: details_str = json.dumps(details, default=str, indent=None, separators=(',', ':'))[:500] # Compact
            except Exception: details_str = str(details)[:500]
            log_message += f" | Details: {details_str}..."
        self.logger.debug(log_message) # Changed to DEBUG for less verbosity in main logs

    async def _report_error(self, error_message: str, task_id: Optional[str] = None):
        self.internal_state["errors_encountered_session"] += 1
        ts = datetime.now(timezone.utc)
        self.internal_state["last_error_ts"] = ts
        self.internal_state["last_error_details"] = error_message
        log_msg = f"ERROR reported: {error_message}"
        if task_id: log_msg += f" (Task: {task_id})"
        self.logger.error(log_msg)
        if hasattr(self.orchestrator, 'report_error'):
            try: await self.orchestrator.report_error(self.agent_name, error_message, task_id)
            except Exception as report_err: self.logger.error(f"Failed to report error to orchestrator: {report_err}")

    async def _execute_tool(self, tool_name: str, params: Dict[str, Any], step_num: Optional[Union[int,str]] = None) -> Dict[str, Any]:
        step_id = f"Step {step_num}" if step_num else "Tool Execution"
        if not hasattr(self.orchestrator, 'use_tool'):
            return {"status": "failure", "message": f"{step_id}: Orchestrator tool execution unavailable."}
        try:
            await self._internal_think(f"{step_id}: Executing tool '{tool_name}'", details=params)
            result = await self.orchestrator.use_tool(tool_name, params)
            if not isinstance(result, dict) or 'status' not in result:
                 return {"status": "failure", "message": f"{step_id}: Tool '{tool_name}' returned invalid result format."}
            return result
        except Exception as e:
            await self._report_error(f"Exception executing tool '{tool_name}': {e}", task_id=self.internal_state.get('current_task_id'))
            return {"status": "failure", "message": f"Exception executing tool '{tool_name}': {e}"}

    async def _learning_loop_wrapper(self):
        self.logger.info(f"{self.agent_name} learning loop wrapper started.")
        try:
            while not self._stop_event.is_set():
                if self._status == self.STATUS_ERROR:
                    self.logger.warning(f"{self.agent_name} in error state, pausing learning loop."); await asyncio.sleep(60); continue
                original_status = self._status
                self._status = self.STATUS_LEARNING
                await self.learning_loop()
                if self._status == self.STATUS_LEARNING: self._status = original_status # Revert if learning_loop doesn't set it
        except asyncio.CancelledError: self.logger.info(f"{self.agent_name} learning loop cancelled.")
        except Exception as e:
            self.logger.error(f"Error in {self.agent_name} learning loop: {e}", exc_info=True)
            self._status = self.STATUS_ERROR; await self._report_error(f"Error in learning loop: {e}")
        finally:
            self.logger.info(f"{self.agent_name} learning loop wrapper finished.")
            if self._status == self.STATUS_LEARNING: self._status = self.STATUS_IDLE

    async def _process_task_queue(self, task_queue: asyncio.Queue):
        self.logger.info(f"{self.agent_name} internal task queue processor started.")
        while not self._stop_event.is_set():
            try:
                task_data = await asyncio.wait_for(task_queue.get(), timeout=2.0)
                if self._stop_event.is_set(): await task_queue.put(task_data); break
                task_id = task_data.get('id', str(uuid.uuid4()))
                task_data['id'] = task_id # Ensure ID is set
                self.logger.debug(f"Dequeued internal task {task_id} for {self.agent_name}: {task_data.get('action','N/A')}")
                # Create a fire-and-forget task for handling, as handle_task is now robust
                asyncio.create_task(self.handle_task(task_data), name=f"{self.agent_name}_Task_{task_id}")
            except asyncio.TimeoutError: continue
            except asyncio.CancelledError: self.logger.info(f"{self.agent_name} task queue processor cancelled."); break
            except Exception as e:
                 self.logger.error(f"Error in {self.agent_name} task queue processor: {e}", exc_info=True)
                 self._status = self.STATUS_ERROR; await self._report_error(f"Error in task queue processor: {e}")
                 await asyncio.sleep(5)
        self.logger.info(f"{self.agent_name} internal task queue processor stopped.")

    def get_status_summary(self) -> dict:
        queue_size = -1; task_queue = self.internal_state.get('task_queue')
        if isinstance(task_queue, asyncio.Queue): queue_size = task_queue.qsize()
        return {
            "agent_name": self.agent_name, "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_task_id": self.internal_state.get("current_task_id"),
            "queue_size": queue_size,
            "tasks_processed_session": self.internal_state.get("tasks_processed_session", 0),
            "errors_encountered_session": self.internal_state.get("errors_encountered_session", 0),
            "last_error_ts": self.internal_state.get("last_error_ts").isoformat() if self.internal_state.get("last_error_ts") else None,
            "last_learning_cycle_ts": self.internal_state.get("last_learning_cycle_ts").isoformat() if self.internal_state.get("last_learning_cycle_ts") else None,
            "strategy_effectiveness_rating": self.internal_state.get("strategy_effectiveness_rating", 0.5)
        }

    async def update_directive_status(self, directive_id: int, status: str, result_summary: Optional[str] = None):
        """Helper to update directive status via Orchestrator, if available."""
        if hasattr(self.orchestrator, 'update_directive_status'):
            try:
                await self.orchestrator.update_directive_status(directive_id, status, result_summary)
            except Exception as e:
                self.logger.error(f"Failed to update directive {directive_id} status via orchestrator: {e}")
        else:
            self.logger.warning("Orchestrator does not have 'update_directive_status' method. Cannot update.")

    def _parse_llm_json(self, json_string: str, expect_type: Type = dict) -> Union[Dict, List, None]:
        """Safely parses JSON from LLM output, handling markdown code blocks and common issues."""
        if not json_string: self.logger.warning("Attempted to parse empty JSON string."); return None
        try:
            # Common: ```json\n{...}\n``` or ```\n{...}\n``` or just {...}
            match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', json_string, re.DOTALL)
            if match: json_to_parse = match.group(1).strip()
            else: json_to_parse = json_string.strip()

            # Attempt to fix common errors like trailing commas before parsing
            # For objects: remove comma before closing brace
            json_to_parse = re.sub(r',\s*([}\]])', r'\1', json_to_parse)
            # For arrays: remove comma before closing bracket (less common from LLMs for top level)

            parsed_json = json.loads(json_to_parse)

            if isinstance(parsed_json, expect_type): return parsed_json
            else: self.logger.error(f"Parsed JSON type mismatch. Expected {expect_type}, got {type(parsed_json)}. Input: {json_string[:200]}"); return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode LLM JSON response: {e}. Response snippet: {json_string[:500]}...")
            # Try a more aggressive cleaning for common LLM mishaps (e.g. Python dict syntax)
            try:
                import ast
                # This is a more risky eval, ensure it's only used on LLM output expected to be dict/list-like
                # A safer approach would be regex replacements for common Pythonisms (True -> true)
                potentially_python_dict_str = json_to_parse.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                # Attempt to convert to dict using ast.literal_eval
                evaluated_obj = ast.literal_eval(potentially_python_dict_str)
                if isinstance(evaluated_obj, expect_type):
                    self.logger.warning("Successfully parsed JSON-like string using AST literal_eval after initial JSONDecodeError.")
                    # Convert back to JSON string then load to ensure it's valid JSON for downstream
                    return json.loads(json.dumps(evaluated_obj))
            except Exception as ast_e:
                self.logger.error(f"AST eval also failed after JSONDecodeError: {ast_e}. Original error: {e}")
            return None
        except Exception as e_gen:
            self.logger.error(f"Unexpected error during JSON parsing: {e_gen}. Snippet: {json_string[:200]}", exc_info=True)
            return None

# --- End of agents/base_agent.py ---