Filename: agents/programmer_agent.py
Description: Agent responsible for developing, maintaining, and refactoring codebase.
Version: 1.1 (Implemented Core Logic, Planning, Execution)
import asyncio
import logging
import json
import os
import shlex # For safe command splitting/quoting
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union
Assuming utils/database.py and models.py exist as provided
from models import KnowledgeFragment # Add other models if needed
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy import update, desc, func
from sqlalchemy.exc import SQLAlchemyError
Base Agent and LLM Client imports
try:
from .base_agent import GeniusAgentBase, KBInterface # Use relative import if applicable
except ImportError:
from base_agent import GeniusAgentBase, KBInterface # Fallback
from openai import AsyncOpenAI as AsyncLLMClient # Standardized name
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
Configure logging
logger = logging.getLogger(name)
if not logger.hasHandlers():
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
PROGRAMMER_AGENT_META_PROMPT = """
You are the ProgrammerAgent within the Synapse Genius Agentic AI System.
Your Core Mandate: Develop, maintain, refactor, and adapt the agency's codebase with genius-level proficiency, efficiency, and foresight.
Key Responsibilities:
Implement new agents and features based on specifications from ThinkTool/Orchestrator.
Refactor existing code for modularity, efficiency, robustness, and maintainability.
Modify sandboxed agency instances for testing new business models.
Adhere to the highest coding standards (clean, testable, well-documented).
Plan your coding tasks meticulously (e.g., using a dedicated plan file or updates to Master_Plan.md). Generate structured, step-by-step plans using available tools (read_file, apply_diff, execute_command, etc.).
Implement efficient coding techniques and caching strategies where applicable.
Ensure code quality and robustness through testing and validation (e.g., running linters/tests via execute_command).
Operate with Extreme Agentic Behavior: Anticipate coding needs, suggest architectural improvements, optimize performance proactively, and solve problems creatively.
Collaborate with other agents, especially ThinkTool for requirements and Orchestrator for tasking and tool execution.
Save all code to files before execution via execute_command. Handle tool installation requests.
"""
class ProgrammerAgent(GeniusAgentBase):
"""
Agent responsible for developing, maintaining, and refactoring the codebase.
Embodies Genius Agentic AI principles for programming tasks.
Version: 1.1
"""
AGENT_NAME = "ProgrammerAgent"
def __init__(self, orchestrator: Any, session_maker: Optional[async_sessionmaker[AsyncSession]] = None):
    """Initializes the ProgrammerAgent."""
    super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, session_maker=session_maker)
    # ProgrammerAgent might not directly need KB interface, relies on ThinkTool/Orchestrator
    self.state = {"current_task_details": None, "coding_plan": None, "status": "idle"}
    self.logger.info("ProgrammerAgent initialized.")

async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a programming-related task by analyzing, planning, and using tools.
    Handles standard coding tasks and specific 'install_tool' actions.
    """
    task_description = task_details.get('description', 'No description provided.')
    action = task_details.get('action', 'generic_code_task') # Default action
    self.logger.info(f"ProgrammerAgent received task: {action} - {task_description}")
    self.state["status"] = "working"
    self.state["current_task_details"] = task_details
    self.state["coding_plan"] = None # Reset plan for new task
    result = {"status": "failure", "message": "Task initialization failed."}

    # --- Structured Thinking: Initial Task Assessment ---
    initial_thought = f"""
    Structured Thinking: Assess Task '{action}'
    1. Goal: Fulfill programming task: {task_description}.
    2. Context: Task details: {json.dumps(task_details)}. Current project state via Orchestrator context (files, etc.).
    3. Constraints: Use Orchestrator tools. Adhere to coding standards. Generate plan first. Handle 'install_tool' specifically.
    4. Information Needed: Potentially file contents, existing code structure.
    5. Plan:
        a. If action=='install_tool', execute installation directly.
        b. Otherwise, analyze request via LLM.
        c. Get necessary file context via Orchestrator 'read_file'.
        d. Generate detailed coding plan via _create_coding_plan (LLM).
        e. Execute plan via _execute_coding_plan (Orchestrator tools).
        f. Return execution result.
    """
    await self._internal_think(initial_thought)
    # --- End Structured Thinking ---

    try:
        if action == 'install_tool':
            # Handle tool installation directly
            result = await self._handle_tool_installation(task_details)
        else:
            # --- Standard Coding Task Flow ---
            # 1. Analyze Request & Get Context (using LLM via helper)
            self.logger.debug("Analyzing programming task request...")
            analysis_prompt = f"Analyze this programming task: {json.dumps(task_details)}. Identify target file(s), relevant code sections, dependencies, and clarify requirements. What context (specific files/lines) is needed to proceed?"
            analysis_result_str = await self._call_llm_with_retry(analysis_prompt, max_tokens=500, temperature=0.2)
            # TODO: Parse analysis_result_str to intelligently fetch context if needed
            self.logger.info(f"Task Analysis Result: {analysis_result_str}")

            # 2. Get File Context (Example - refine based on analysis)
            file_path = task_details.get('file_path') # Example: if a primary file is specified
            file_content = None
            if file_path and self.orchestrator and hasattr(self.orchestrator, 'use_tool'):
                 try:
                     self.logger.debug(f"Reading file context: {file_path}")
                     # Use orchestrator's tool mechanism
                     read_result = await self.orchestrator.use_tool('read_file', {'path': file_path}) # Assuming use_tool exists
                     if read_result and read_result.get('status') == 'success':
                         file_content = read_result.get('content')
                         self.logger.info(f"Successfully read content from: {file_path}")
                     else: self.logger.warning(f"Could not read file {file_path}: {read_result.get('message')}")
                 except Exception as read_err: self.logger.warning(f"Exception during file read {file_path}: {read_err}", exc_info=True)

            # 3. Create Plan (using LLM via helper)
            self.state["coding_plan"] = await self._create_coding_plan(task_details, analysis_result_str, file_content)

            # 4. Execute Plan (using Orchestrator tools via helper)
            if self.state["coding_plan"]:
                result = await self._execute_coding_plan()
            else:
                result = {"status": "failure", "message": "Failed to create a coding plan."}

    except Exception as e:
        self.logger.error(f"Error executing programming task '{task_description}': {e}", exc_info=True)
        result = {"status": "failure", "message": f"Failed to execute task: {e}"}

    finally:
        self.state["status"] = "idle"
        self.state["current_task_details"] = None
        # Don't clear plan, might be useful for insights
        self.logger.info(f"ProgrammerAgent finished task: {task_description}. Status: {result.get('status', 'failure')}")
        # Log task outcome to KB
        await self.log_knowledge_fragment(
            agent_source=self.AGENT_NAME,
            data_type="task_outcome",
            content={"task": task_description, "action": action, "result": result},
            tags=["task_execution", result.get('status', 'failure')],
            relevance_score=0.8 if result.get('status') == 'success' else 0.6
        )
    return result

async def _handle_tool_installation(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
    """Handles the specific task of installing a tool."""
    tool_name = task_details.get('tool_name')
    package_manager = task_details.get('package_manager', 'apt')
    package_name = task_details.get('package_name')
    git_repo = task_details.get('git_repo')
    self.logger.info(f"Handling installation request for tool: {tool_name}")

    if not tool_name or not (package_name or git_repo):
        return {"status": "failure", "message": "Missing tool name, package name, or git repo for installation."}

    command = None
    if git_repo:
        # Assumes standard python setup.py install for git repos
        # TODO: Make this more robust based on repo structure if possible
        clone_dir = f"/tmp/{tool_name}_{uuid.uuid4().hex[:6]}"
        command = f"git clone {shlex.quote(git_repo)} {shlex.quote(clone_dir)} && cd {shlex.quote(clone_dir)} && python3 setup.py install && cd / && rm -rf {shlex.quote(clone_dir)}"
    elif package_manager == 'apt':
        command = f"sudo apt-get update && sudo apt-get install -y {shlex.quote(package_name)}"
    elif package_manager == 'pip':
        command = f"pip3 install --user {shlex.quote(package_name)}" # Install to user site-packages
    else:
        return {"status": "failure", "message": f"Unsupported package manager: {package_manager}"}

    if command and self.orchestrator and hasattr(self.orchestrator, 'use_tool'):
        self.logger.info(f"Executing installation command: {command}")
        # --- Structured Thinking Step ---
        install_thought = f"""
        Structured Thinking: Execute Tool Installation
        1. Goal: Install tool '{tool_name}' using {package_manager}.
        2. Context: Command generated: '{command}'.
        3. Constraints: Use Orchestrator's execute_command tool. Handle potential errors.
        4. Information Needed: None additional.
        5. Plan: Call use_tool('execute_command'). Check result status. Return success/failure.
        """
        await self._internal_think(install_thought)
        # --- End Structured Thinking Step ---
        try:
            exec_result = await self.orchestrator.use_tool('execute_command', {'command': command})
            if exec_result and exec_result.get('status') == 'success':
                self.logger.info(f"Successfully executed installation command for {tool_name}.")
                return {"status": "success", "message": f"Installation command for {tool_name} executed."}
            else:
                error_msg = exec_result.get('message', exec_result.get('stderr', 'Unknown execution error'))
                self.logger.error(f"Installation command failed for {tool_name}: {error_msg}")
                return {"status": "failure", "message": f"Installation command failed: {error_msg}"}
        except Exception as exec_err:
            self.logger.error(f"Exception during installation command execution for {tool_name}: {exec_err}", exc_info=True)
            return {"status": "failure", "message": f"Exception during installation: {exec_err}"}
    else:
        return {"status": "failure", "message": "Orchestrator tool execution unavailable."}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception))
async def _call_llm_with_retry(self, prompt: str, model_preference: Optional[List[str]] = None, temperature: float = 0.5, max_tokens: int = 1024, is_json_output: bool = False) -> Optional[str]:
    """Centralized method for calling LLMs via the Orchestrator."""
    if not self.orchestrator or not hasattr(self.orchestrator, 'call_llm'):
        self.logger.error("Orchestrator or its call_llm method is unavailable.")
        return None
    try:
        # Delegate the call to the orchestrator
        response_content = await self.orchestrator.call_llm(
            agent_name=self.AGENT_NAME, prompt=prompt, temperature=temperature,
            max_tokens=max_tokens, is_json_output=is_json_output, model_preference=model_preference
        )
        return response_content
    except Exception as e:
        self.logger.error(f"Error occurred calling LLM via orchestrator: {e}", exc_info=True)
        raise # Re-raise for tenacity

async def _create_coding_plan(self, task_details: Dict[str, Any], analysis_result_str: Optional[str], file_content: Optional[str]) -> Optional[List[Dict]]:
    """Creates a detailed step-by-step coding plan using an LLM."""
    self.logger.info("Generating coding plan via LLM...")
    # --- Structured Thinking Step ---
    plan_gen_thought = f"""
    Structured Thinking: Create Coding Plan
    1. Goal: Generate a detailed, sequential plan (list of tool calls) to fulfill the programming task.
    2. Context: Task details, LLM analysis of task, relevant file content (if provided). Available tools: read_file, apply_diff, write_to_file, insert_content, search_and_replace, execute_command, search_files, list_files.
    3. Constraints: Plan must be a JSON list of steps. Each step needs 'action', 'tool', 'params'. Steps must be logical and sequential.
    4. Information Needed: Task requirements, analysis insights, code context.
    5. Plan: Formulate LLM prompt asking for the JSON plan. Call LLM. Parse JSON response. Return plan or None on failure.
    """
    await self._internal_think(plan_gen_thought)
    # --- End Structured Thinking Step ---

    planning_prompt = f"""
    {PROGRAMMER_AGENT_META_PROMPT[:500]}... # Include meta prompt context

    **Task:** Create a detailed, step-by-step execution plan to accomplish the following programming task.

    **Programming Task Details:**
    {json.dumps(task_details, indent=2)}

    **Initial Analysis:**
    {analysis_result_str or 'N/A'}

    **Relevant File Context (if available):**
    ```
    {file_content or 'N/A'}
    ```

    **Available Tools:** read_file, apply_diff, write_to_file, insert_content, search_and_replace, execute_command, search_files, list_files.

    **Instructions:**
    Generate a plan as a JSON list of dictionaries. Each dictionary represents a step and MUST contain:
    - "step" (int): Sequential step number starting from 1.
    - "action" (str): A brief description of what the step does (e.g., "Read main function", "Apply refactoring diff", "Run unit tests").
    - "tool" (str): The exact name of the Orchestrator tool to use for this step (e.g., "read_file", "apply_diff", "execute_command").
    - "params" (dict): A dictionary of parameters required by the specified tool (e.g., {{"path": "/path/to/file.py", "diff": "..."}}, {{"command": "pytest"}}).

    The plan should be logical, sequential, and cover all necessary actions including context gathering, code modification, and verification (if applicable). Ensure parameters like file paths are correct relative to the project structure.

    **Output:** Respond ONLY with the valid JSON list representing the plan. Do not include any other text, preamble, or explanation.
    ```json
    [
      {{ "step": 1, "action": "...", "tool": "...", "params": {{...}} }},
      {{ "step": 2, "action": "...", "tool": "...", "params": {{...}} }}
    ]
    ```
    """
    plan_json_str = await self._call_llm_with_retry(planning_prompt, max_tokens=1500, temperature=0.2, is_json_output=True)

    if plan_json_str:
        try:
            # Find JSON array if nested
            json_start = plan_json_str.find('[')
            json_end = plan_json_str.rfind(']') + 1
            if json_start != -1 and json_end != -1:
                 plan_list = json.loads(plan_json_str[json_start:json_end])
                 if isinstance(plan_list, list) and all(isinstance(step, dict) and 'step' in step and 'action' in step and 'tool' in step and 'params' in step for step in plan_list):
                     # Add status to each step
                     for step in plan_list: step['status'] = 'pending'
                     self.logger.info(f"Successfully generated coding plan with {len(plan_list)} steps.")
                     return plan_list
                 else:
                      raise ValueError("LLM response is not a valid list of step dictionaries.")
            else:
                 raise ValueError("Could not find JSON array in LLM response.")
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse LLM coding plan: {e}. Response: {plan_json_str[:500]}...")
            return None
    else:
        self.logger.error("LLM failed to generate a coding plan.")
        return None

async def _execute_coding_plan(self) -> Dict[str, Any]:
    """Executes the steps defined in the coding plan using Orchestrator tools."""
    self.logger.info("Starting execution of the coding plan.")
    if not self.state["coding_plan"] or not isinstance(self.state["coding_plan"], list):
        self.logger.error("Execution attempted without a valid coding plan.")
        return {"status": "failure", "message": "No valid coding plan available."}

    execution_results = []
    all_steps_successful = True
    plan_halted = False
    last_tool_output = None # Store output for potential verification steps

    for step in self.state["coding_plan"]:
        if plan_halted:
            step["status"] = "skipped"
            continue

        if step.get("status") == "pending":
            step_num = step.get("step", "?")
            action_desc = step.get("action", "Unknown action")
            tool_name = step.get("tool")
            tool_params = step.get("params", {})

            if not tool_name:
                self.logger.warning(f"Step {step_num} ('{action_desc}') does not specify a tool. Skipping.")
                step["status"] = "skipped"
                execution_results.append({"step": step_num, "status": "skipped", "details": "No tool specified."})
                continue

            # --- Structured Thinking Step (Before Tool Call) ---
            pre_tool_thought = f"""
            Structured Thinking: Execute Plan Step {step_num}
            1. Goal: Execute action '{action_desc}' using tool '{tool_name}'.
            2. Context: Parameters: {tool_params}. Previous step output (if any): {str(last_tool_output)[:100]}...
            3. Constraints: Must use Orchestrator.use_tool. Handle potential errors.
            4. Information Needed: Tool parameters are provided in the plan.
            5. Plan: Call orchestrator.use_tool('{tool_name}', {tool_params}). Process result. Update step status. Check for verification needs.
            """
            await self._internal_think(pre_tool_thought)
            # --- End Structured Thinking Step ---

            self.logger.info(f"Executing Step {step_num}: {action_desc} using tool '{tool_name}'")
            observation = None # Reset observation
            try:
                if not self.orchestrator or not hasattr(self.orchestrator, 'use_tool'):
                     raise RuntimeError("Orchestrator tool execution is unavailable.")

                observation = await self.orchestrator.use_tool(tool_name, tool_params)
                last_tool_output = observation # Store for next step's context/verification
                self.logger.debug(f"Tool '{tool_name}' observation received: {str(observation)[:200]}...")

                # Process Observation
                if observation and observation.get("status") == "success":
                    step["status"] = "done"
                    execution_results.append({"step": step_num, "status": "success", "output_preview": str(observation.get('content', observation.get('stdout', 'OK')))[:100]})
                    self.logger.info(f"Step {step_num} completed successfully.")
                    # Optional: Log successful output as KB fragment if significant
                    # await self.log_knowledge_fragment(...)

                    # --- Code Verification Logic ---
                    if tool_name == 'execute_command' and ('pytest' in tool_params.get('command', '') or 'eslint' in tool_params.get('command', '') or 'flake8' in tool_params.get('command', '')):
                         return_code = observation.get('returncode', 0)
                         stdout = observation.get('stdout', '')
                         stderr = observation.get('stderr', '')
                         if return_code != 0:
                              verification_failed_msg = f"Verification failed (Step {step_num}): Command '{tool_params.get('command')}' exited with code {return_code}. Stderr: {stderr[:500]}"
                              self.logger.error(verification_failed_msg)
                              step["status"] = "verification_failed" # Special status
                              execution_results[-1]["status"] = "verification_failed" # Update last result
                              execution_results[-1]["details"] = verification_failed_msg
                              all_steps_successful = False
                              plan_halted = True # Stop execution after verification failure
                              # TODO: Trigger self-correction? For now, just halt.
                              # await self.orchestrator.report_error(self.AGENT_NAME, verification_failed_msg)
                         else:
                              self.logger.info(f"Verification successful (Step {step_num}): Command '{tool_params.get('command')}' passed.")
                    # --- End Code Verification ---

                else: # Tool execution failed
                    error_details = observation.get('message', observation.get('stderr', 'Unknown tool error'))
                    raise RuntimeError(f"Tool '{tool_name}' failed: {error_details}")

            except Exception as e:
                self.logger.error(f"Error executing step {step_num}: {action_desc}. Error: {e}", exc_info=True)
                step["status"] = "error"
                all_steps_successful = False
                plan_halted = True # Stop execution on error
                error_detail_str = str(e)
                execution_results.append({"step": step_num, "status": "error", "details": error_detail_str})
                # Report error to orchestrator
                if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"Error in coding plan step {step_num} ({action_desc}): {error_detail_str}")
                # break # Stop plan execution on first error

        elif step.get("status") == "done":
            self.logger.debug(f"Step {step['step']} already done.")
        elif step.get("status") == "skipped":
             self.logger.debug(f"Step {step['step']} was skipped.")
        elif step.get("status") == "verification_failed":
             self.logger.warning(f"Step {step['step']} previously failed verification.")
             plan_halted = True # Ensure plan stops if resuming after verification failure
        else:
            self.logger.warning(f"Step {step.get('step', '?')} has unexpected status: {step.get('status')}")

    # --- Final Result ---
    final_status = "success" if all_steps_successful else ("verification_failed" if any(s.get('status') == 'verification_failed' for s in self.state["coding_plan"]) else "error")
    self.logger.info(f"Coding plan execution finished with overall status: {final_status}")
    # Collect artifacts (paths of modified/created files from successful steps)
    artifacts = [
        step['params']['path'] for step in self.state["coding_plan"]
        if step.get('tool') in ['apply_diff', 'write_to_file', 'insert_content', 'search_and_replace']
        and step.get('status') == 'done'
        and 'path' in step.get('params', {})
    ]

    return {"status": final_status, "message": f"Plan execution finished with status: {final_status}", "details": execution_results, "artifacts": sorted(list(set(artifacts)))}

async def collect_insights(self) -> Dict[str, Any]:
    """Reports on recent task outcomes and common errors."""
    self.logger.debug("ProgrammerAgent collect_insights called.")
    insights = {
        "agent_name": self.AGENT_NAME,
        "status": self.status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "recent_tasks_count": 0,
        "success_rate_pct": 0.0,
        "common_error_types": [],
        "refactoring_opportunities_identified": 0, # Placeholder
        "key_observations": []
    }
    try:
        # Query recent task outcomes logged by this agent
        time_window = timedelta(hours=24)
        fragments = await self.query_knowledge_base(
            agent_source=self.AGENT_NAME,
            data_types=["task_outcome"],
            time_window=time_window,
            limit=50
        )

        if fragments:
            insights["recent_tasks_count"] = len(fragments)
            success_count = 0
            error_types = Counter()
            for frag in fragments:
                try:
                    outcome_data = json.loads(frag.content)
                    if outcome_data.get("result", {}).get("status") == "success":
                        success_count += 1
                    else:
                        # Try to extract error type/message
                        error_msg = outcome_data.get("result", {}).get("message", "Unknown Error")
                        # Basic error type categorization (can be improved)
                        if "Tool" in error_msg and "failed" in error_msg: error_types["ToolExecutionError"] += 1
                        elif "LLM" in error_msg: error_types["LLMError"] += 1
                        elif "Verification failed" in error_msg: error_types["VerificationError"] += 1
                        elif "Exception" in error_msg: error_types["PythonException"] += 1
                        else: error_types["UnknownTaskError"] += 1
                except (json.JSONDecodeError, TypeError):
                    error_types["LoggingFormatError"] += 1

            if insights["recent_tasks_count"] > 0:
                insights["success_rate_pct"] = round((success_count / insights["recent_tasks_count"]) * 100, 1)
            insights["common_error_types"] = dict(error_types.most_common(3))
            insights["key_observations"].append(f"Analyzed {insights['recent_tasks_count']} task outcomes from last 24h.")
        else:
             insights["key_observations"].append("No recent task outcome data found.")

    except Exception as e:
        self.logger.error(f"Error collecting insights: {e}", exc_info=True)
        insights["key_observations"].append(f"Error collecting insights: {e}")

    return insights

# --- Abstract Method Stubs ---
# ProgrammerAgent focuses on execution, learning/critique might be less frequent
# or handled primarily by ThinkTool analyzing its outputs/logs.

async def learning_loop(self):
    self.logger.debug(f"{self.AGENT_NAME} learning_loop: No specific periodic learning implemented. Learning occurs via task analysis and potential self-correction.")
    await asyncio.sleep(3600 * 24) # Sleep long, not primary focus

async def self_critique(self) -> Dict[str, Any]:
    self.logger.debug(f"{self.AGENT_NAME} self_critique: Performing basic critique based on recent insights.")
    insights = await self.collect_insights()
    feedback = f"Critique based on last 24h: {insights.get('recent_tasks_count', 0)} tasks processed with {insights.get('success_rate_pct', 0)}% success. Common errors: {insights.get('common_error_types', {})}"
    status = "ok"
    if insights.get('success_rate_pct', 100) < 75 and insights.get('recent_tasks_count', 0) > 5:
         status = "warning"
         feedback += " Success rate below threshold, investigate common errors."

    return {"status": status, "feedback": feedback}

async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
    """Generates prompts for internal LLM calls (analysis, planning, diff gen)."""
    # This method is effectively called by execute_task and _create_coding_plan
    # We return a placeholder here as the actual prompt generation happens within those methods.
    self.logger.warning(f"{self.AGENT_NAME} generate_dynamic_prompt called directly - prompt generation is handled within execute_task/_create_coding_plan.")
    return f"Context: {json.dumps(task_context)}. Please perform the required programming action."
Use code with caution.
--- End of agents/programmer_agent.py ---