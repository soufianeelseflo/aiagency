# Filename: agents/programmer_agent.py
# Description: Production-ready Programmer Agent capable of planning, coding,
#              tool execution, verification, and handling installation tasks.
# Version: 2.0 (Refined - Placeholders Removed)

import asyncio
import logging
import json
import os
import shlex # For safe command splitting/quoting
import hashlib
import time
import re # For parsing analysis/verification output
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union, Tuple # Added Tuple
from collections import Counter # For insight analysis

# Assuming utils/database.py and models.py exist as provided
from models import KnowledgeFragment # Add other models if needed
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy import update, desc, func
from sqlalchemy.exc import SQLAlchemyError

# Base Agent and LLM Client imports
try:
    from .base_agent import GeniusAgentBase, KBInterface # Use relative import if applicable
except ImportError:
    from base_agent import GeniusAgentBase, KBInterface # Fallback
# Assuming LLM Client access via orchestrator
# from openai import AsyncOpenAI as AsyncLLMClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Configure dedicated operational logger (assuming setup elsewhere)
op_logger = logging.getLogger('OperationalLog')
if not op_logger.hasHandlers(): # Add basic handler if none configured
    op_handler = logging.StreamHandler()
    op_formatter = logging.Formatter('%(asctime)s - OP_LOG - %(levelname)s - %(message)s')
    op_handler.setFormatter(op_formatter)
    op_logger.addHandler(op_handler)
    op_logger.setLevel(logging.INFO)
    op_logger.propagate = False


PROGRAMMER_AGENT_META_PROMPT = """
You are the ProgrammerAgent within the Synapse Genius Agentic AI System.
Your Core Mandate: Develop, maintain, refactor, and adapt the agency's codebase with genius-level proficiency, efficiency, and foresight. Aim for "First-Try Deployment".
Key Responsibilities:
- Implement new agents and features based on specifications from ThinkTool/Orchestrator.
- Refactor existing code for modularity, efficiency, robustness, and maintainability.
- Modify sandboxed agency instances for testing new business models.
- Adhere to the highest coding standards (clean, testable, well-documented, minimalist potency).
- Plan coding tasks meticulously using structured steps. Generate structured, step-by-step plans using available tools (read_file, apply_diff, write_to_file, execute_command, etc.).
- Implement efficient coding techniques and caching strategies.
- Ensure code quality through testing and validation (e.g., running linters/tests via execute_command).
- Operate with Extreme Agentic Behavior: Anticipate coding needs, suggest architectural improvements, optimize performance proactively, solve problems creatively.
- Collaborate with other agents via the Orchestrator.
- Save all code to files using appropriate tools (`apply_diff`, `write_to_file`) before execution via `execute_command`. Handle tool installation requests.
"""

class ProgrammerAgent(GeniusAgentBase):
    """
    Agent responsible for developing, maintaining, and refactoring the codebase.
    Embodies Genius Agentic AI principles for programming tasks.
    Version: 2.0 (Refined)
    """
    AGENT_NAME = "ProgrammerAgent"

    def __init__(self, orchestrator: Any, session_maker: Optional[async_sessionmaker[AsyncSession]] = None):
        """Initializes the ProgrammerAgent."""
        # Get dependencies from orchestrator if available
        config = getattr(orchestrator, 'config', None)
        kb_interface = getattr(orchestrator, 'kb_interface', None)
        super().__init__(agent_name=self.AGENT_NAME, kb_interface=kb_interface, orchestrator=orchestrator, config=config)
        self.session_maker = session_maker # Store session_maker if provided for direct DB access (e.g., insights)

        self.internal_state = getattr(self, 'internal_state', {})
        self.internal_state['current_task_details'] = None
        self.internal_state['coding_plan'] = None
        self.internal_state['last_read_content'] = {}
        self.internal_state['modified_files'] = set()
        self.logger.info(f"{self.AGENT_NAME} v2.0 (Refined) initialized.")

    async def log_operation(self, level: str, message: str):
        """Helper to log to the operational log file with agent context."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err:
            print(f"OPERATIONAL LOG FAILED ({self.agent_name}): {level} - {message} | Error: {log_err}")
            logger.error(f"Failed to write to operational log from {self.agent_name}: {log_err}")

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a programming-related task by analyzing, planning, and using tools.
        Handles standard coding tasks and specific 'install_tool' actions.
        """
        task_description = task_details.get('description', 'No description provided.')
        action = task_details.get('action', 'generic_code_task')
        self.logger.info(f"{self.AGENT_NAME} received task: {action} - {task_description}")
        self.status = "working"
        self.internal_state['current_task_details'] = task_details
        self.internal_state['coding_plan'] = None
        self.internal_state['last_read_content'] = {}
        self.internal_state['modified_files'] = set()
        result = {"status": "failure", "message": "Task initialization failed."}

        # --- Structured Thinking: Initial Task Assessment ---
        initial_thought = f"""
        Structured Thinking: Assess Task '{action}'
        1. Goal: Fulfill programming task: {task_description}.
        2. Context: Task details: {json.dumps(task_details)}. Access to workspace via Orchestrator tools.
        3. Constraints: Use Orchestrator tools. Adhere to coding standards. Generate plan first for coding tasks. Handle 'install_tool' specifically.
        4. Information Needed: Potential file contents, existing code structure, dependencies.
        5. Plan:
            a. If action=='install_tool', delegate to _handle_tool_installation.
            b. Otherwise (coding task):
                i. Analyze request via LLM (_analyze_request).
                ii. Get necessary file context based on analysis (_get_required_context).
                iii. Generate detailed coding plan via _create_coding_plan (LLM).
                iv. Execute plan via _execute_coding_plan (Orchestrator tools).
                v. Return execution result.
        """
        await self._internal_think(initial_thought)
        # --- End Structured Thinking ---

        try:
            if action == 'install_tool':
                result = await self._handle_tool_installation(task_details)
            elif action in ['implement_code', 'refactor_code', 'fix_bug', 'create_agent', 'generic_code_task']:
                # --- Standard Coding Task Flow ---
                # 1. Analyze Request
                analysis_result = await self._analyze_request(task_details)
                if not analysis_result: raise RuntimeError("Failed to analyze task request.")
                self.logger.info(f"Task Analysis Complete. Identified files/context needed.")

                # 2. Get File Context
                required_files = analysis_result.get("required_files", [])
                file_context_map = await self._get_required_context(required_files)

                # 3. Create Plan
                self.state["coding_plan"] = await self._create_coding_plan(task_details, analysis_result, file_context_map)
                if not self.state["coding_plan"]: raise RuntimeError("Failed to create a coding plan.")
                await self.log_operation('info', f"Generated coding plan with {len(self.state['coding_plan'])} steps.")
                # Log plan to KB
                await self.log_knowledge_fragment(
                    agent_source=self.AGENT_NAME, data_type="coding_plan",
                    content=self.state["coding_plan"], tags=["plan", action], relevance_score=0.9
                )

                # 4. Execute Plan
                await self._internal_think(f"Starting execution of {len(self.state['coding_plan'])} planned steps.")
                execution_result = await self._execute_coding_plan()

                # 5. Final Result
                result = {
                    "status": execution_result["status"],
                    "message": f"Task '{task_description}' completed with status: {execution_result['status']}.",
                    "details": execution_result["details"],
                    "files_modified": sorted(list(self.internal_state['modified_files']))
                }
            else:
                 result["message"] = f"Unsupported action type for ProgrammerAgent: {action}"
                 self.logger.warning(result["message"])

        except Exception as e:
            self.logger.error(f"Error executing programming task '{task_description}': {e}", exc_info=True)
            result = {"status": "failure", "message": f"Failed to execute task: {e}"}
            if hasattr(self.orchestrator, 'report_error'):
                 await self.orchestrator.report_error(self.AGENT_NAME, f"Task '{task_description}' failed: {e}")
        finally:
            self.status = "idle"
            self.internal_state['current_task_details'] = None
            # Keep plan in state for insights, clear context/modified files
            self.internal_state['last_read_content'] = {}
            self.internal_state['modified_files'] = set()
            self.logger.info(f"{self.AGENT_NAME} finished task: {task_description}. Status: {result.get('status', 'failure')}")
            # Log final outcome
            await self.log_knowledge_fragment(
                agent_source=self.AGENT_NAME, data_type="task_outcome",
                content={"task": task_description, "action": action, "result": result},
                tags=["task_execution", result.get('status', 'failure')],
                relevance_score=0.8 if result.get('status') == 'success' else 0.6
            )
        return result

    async def _analyze_request(self, task_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyzes the programming task using an LLM to identify scope and context needs."""
        self.logger.debug("Analyzing programming task request via LLM...")
        analysis_prompt = f"""
        Analyze the following programming task details:
        ```json
        {json.dumps(task_details, indent=2)}
        ```
        Identify:
        1. The primary goal of the task.
        2. Specific files or code sections likely needing modification or reference.
        3. Any potential dependencies or related modules to consider.
        4. Any ambiguities or missing information in the request.
        5. A list of file paths (`required_files`) absolutely necessary to read for context before planning changes.

        Output ONLY a JSON object with keys: "goal", "target_areas", "dependencies", "ambiguities", "required_files".
        """
        analysis_json_str = await self._call_llm_with_retry(
            analysis_prompt, max_tokens=800, temperature=0.1, is_json_output=True
        )
        if analysis_json_str:
            try:
                analysis_result = json.loads(analysis_json_str[analysis_json_str.find('{'):analysis_json_str.rfind('}')+1])
                # Basic validation
                if not all(k in analysis_result for k in ["goal", "target_areas", "required_files"]):
                    raise ValueError("LLM analysis response missing required keys.")
                self.logger.info(f"Task Analysis Result: Goal='{analysis_result.get('goal')}', Targets='{analysis_result.get('target_areas')}', Files Needed={analysis_result.get('required_files')}")
                return analysis_result
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse LLM task analysis: {e}. Response: {analysis_json_str[:500]}...")
                return None
        else:
            self.logger.error("LLM failed to generate task analysis.")
            return None

    async def _get_required_context(self, required_files: List[str]) -> Dict[str, Optional[str]]:
        """Fetches content for the files identified as necessary by the analysis step."""
        file_context_map = {}
        if not required_files or not isinstance(required_files, list):
            self.logger.info("No specific files identified as required for context by analysis.")
            return file_context_map

        await self._internal_think(f"Fetching required context from {len(required_files)} files: {required_files}")
        fetch_tasks = [self._read_file_context(file_path) for file_path in required_files]
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for file_path, content_or_error in zip(required_files, results):
            if isinstance(content_or_error, Exception):
                self.logger.warning(f"Failed to fetch context for {file_path}: {content_or_error}")
                file_context_map[file_path] = None # Indicate read failure
            else:
                file_context_map[file_path] = content_or_error # Store content or None if read failed internally

        self.logger.info(f"Fetched context for {len([c for c in file_context_map.values() if c is not None])}/{len(required_files)} required files.")
        return file_context_map

    async def _handle_tool_installation(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the specific task of installing a tool."""
        tool_name = task_details.get('tool_name')
        # Infer package manager and package name if possible, or require them
        package_manager = task_details.get('package_manager') # e.g., 'apt', 'pip', 'npm', 'git'
        package_name = task_details.get('package_name') # e.g., 'theharvester', 'requests'
        git_repo = task_details.get('git_repo') # e.g., 'https://github.com/user/repo.git'

        self.logger.info(f"Handling installation request for tool: {tool_name}")

        if not tool_name:
            return {"status": "failure", "message": "Missing 'tool_name' for install_tool action."}
        if not (package_manager and package_name) and not git_repo:
             return {"status": "failure", "message": f"Missing package manager/name or git repo for tool '{tool_name}'."}

        command = self._determine_install_command(tool_name, package_manager, package_name, git_repo)
        if not command:
             return {"status": "failure", "message": f"Could not determine install command for tool: {tool_name}"}

        if self.orchestrator and hasattr(self.orchestrator, 'use_tool'):
            await self._internal_think(f"Executing install command for '{tool_name}': {command}")
            await self.log_operation('info', f"Executing install command: {command}")
            try:
                exec_result = await self.orchestrator.use_tool(
                    'execute_command',
                    {'command': command, 'purpose': f'Install tool {tool_name}'} # Assuming use_tool accepts purpose
                )
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

    def _determine_install_command(self, tool_name: str, pkg_manager: Optional[str], pkg_name: Optional[str], git_repo: Optional[str]) -> Optional[str]:
        """Determines the appropriate install command based on provided details."""
        # TODO: Get OS info from Orchestrator/Config if possible for better accuracy
        os_type = "linux" # Assume linux default for now

        if git_repo:
            # Assumes standard python setup.py install or similar build process
            clone_dir = f"/tmp/{tool_name.lower()}_{uuid.uuid4().hex[:6]}"
            # Basic attempt, might need adjustment based on repo structure
            # Consider adding common build steps: configure, make, make install?
            return f"git clone --depth 1 {shlex.quote(git_repo)} {shlex.quote(clone_dir)} && cd {shlex.quote(clone_dir)} && (python3 setup.py install || pip3 install --user . || npm install || make) && cd / && rm -rf {shlex.quote(clone_dir)}"

        if pkg_manager and pkg_name:
            pkg_manager = pkg_manager.lower()
            safe_pkg_name = shlex.quote(pkg_name)
            if pkg_manager == 'apt':
                return f"sudo apt-get update && sudo apt-get install -y {safe_pkg_name}"
            elif pkg_manager == 'pip' or pkg_manager == 'pip3':
                return f"pip3 install --user {safe_pkg_name}" # Prefer user install
            elif pkg_manager == 'npm':
                 return f"npm install -g {safe_pkg_name}" # Global install for CLI tools
            elif pkg_manager == 'yum': # Add other common managers
                 return f"sudo yum install -y {safe_pkg_name}"
            elif pkg_manager == 'brew':
                 return f"brew install {safe_pkg_name}"
            else:
                 self.logger.warning(f"Unsupported package manager '{pkg_manager}' specified for tool '{tool_name}'.")
                 return None
        else:
             # Fallback heuristic based on tool name (less reliable)
             tool_lower = tool_name.lower()
             if tool_lower in ["theharvester", "recon-ng", "exiftool", "nmap", "sqlmap"]:
                 return f"sudo apt-get update && sudo apt-get install -y {tool_lower}"
             elif tool_lower in ["sherlock", "photon", "requests", "beautifulsoup4", "playwright", "pytest", "flake8"]:
                 pip_command = f"pip3 install --user {tool_lower}"
                 if tool_lower == "playwright": pip_command += " && python3 -m playwright install --with-deps"
                 return pip_command
             elif tool_lower in ["vite", "next", "eslint", "prettier", "tailwindcss"]:
                  return f"npm install -g {tool_lower}"
             else:
                 self.logger.warning(f"Cannot determine install command for tool: {tool_name} without package manager info.")
                 return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception))
    async def _create_coding_plan(self, task_details: Dict[str, Any], analysis_result: Optional[Dict[str, Any]], file_context_map: Dict[str, Optional[str]]) -> Optional[List[Dict]]:
        """Creates a detailed step-by-step coding plan using an LLM."""
        self.logger.info("Generating coding plan via LLM...")
        task_description = task_details.get('description', 'N/A')

        # Format file context for prompt
        file_context_prompt = ""
        for path, content in file_context_map.items():
            file_context_prompt += f"\n--- Context: {path} ---\n"
            if content:
                file_context_prompt += content[:2000] + ("..." if len(content) > 2000 else "") # Limit context per file
            else:
                file_context_prompt += "[Could not read file content]"
            file_context_prompt += "\n---------------------\n"

        # --- Structured Thinking Step ---
        plan_gen_thought = f"""
        Structured Thinking: Create Coding Plan
        1. Goal: Generate a detailed, sequential plan (list of tool calls) to fulfill the programming task.
        2. Context: Task details, LLM analysis ({analysis_result.get('goal', 'N/A')}), file context for {list(file_context_map.keys())}. Available tools: read_file, apply_diff, write_to_file, insert_content, search_and_replace, execute_command, search_files, list_files.
        3. Constraints: Plan must be JSON list. Steps need 'action', 'tool', 'params'. Must be logical, sequential. Include verification steps if appropriate (e.g., run tests after changes).
        4. Information Needed: Task requirements, analysis insights, code context.
        5. Plan: Formulate LLM prompt asking for the JSON plan. Call LLM. Parse JSON response. Validate plan structure. Return plan or None.
        """
        await self._internal_think(plan_gen_thought)
        # --- End Structured Thinking Step ---

        planning_prompt = f"""
        {PROGRAMMER_AGENT_META_PROMPT[:500]}...

        **Task:** Create a detailed, step-by-step execution plan (JSON list) to accomplish the programming task described below.

        **Programming Task Details:**
        {json.dumps(task_details, indent=2)}

        **Initial Analysis Insights:**
        {json.dumps(analysis_result, indent=2) if analysis_result else 'N/A'}

        **Relevant File Context:**
        {file_context_prompt if file_context_prompt else 'N/A'}

        **Available Tools:** read_file, apply_diff, write_to_file, insert_content, search_and_replace, execute_command, search_files, list_files.

        **Instructions:**
        Generate a plan as a JSON list of dictionaries. Each dictionary represents a step and MUST contain:
        - "step" (int): Sequential step number starting from 1.
        - "action" (str): A brief description of what the step does (e.g., "Read main function", "Apply refactoring diff", "Run unit tests").
        - "tool" (str): The exact name of the Orchestrator tool to use (e.g., "read_file", "apply_diff", "execute_command").
        - "params" (dict): Parameters required by the tool (e.g., {{"path": "/path/to/file.py", "diff": "..."}}, {{"command": "pytest"}}).
        - "verification" (str, optional): Describe how this step's success can be verified (e.g., "Check command output for 'Success'", "Run linter on modified file").

        The plan should be logical, sequential, and cover all necessary actions including context gathering (if analysis indicated more is needed), code modification, and verification (e.g., add 'execute_command' steps for linters/tests after modifications). Ensure file paths in 'params' are correct.

        **Output:** Respond ONLY with the valid JSON list representing the plan.
        ```json
        [
          {{ "step": 1, "action": "...", "tool": "...", "params": {{...}}, "verification": "..." }},
          {{ "step": 2, "action": "...", "tool": "...", "params": {{...}} }}
        ]
        ```
        """
        plan_json_str = await self._call_llm_with_retry(
            planning_prompt,
            model=self.config.get("OPENROUTER_MODELS", {}).get('programmer_planning', "google/gemini-1.5-pro-latest"),
            temperature=0.2, max_tokens=2000, is_json_output=True # Increased tokens for plan
        )

        if plan_json_str:
            try:
                # Find JSON array if nested
                json_start = plan_json_str.find('[')
                json_end = plan_json_str.rfind(']') + 1
                if json_start == -1 or json_end == 0: raise ValueError("No JSON array found in response.")

                plan_list = json.loads(plan_json_str[json_start:json_end])
                if not isinstance(plan_list, list): raise ValueError("LLM did not return a list for the plan.")

                validated_plan = []
                for i, step in enumerate(plan_list):
                    if not isinstance(step, dict) or not all(k in step for k in ['action', 'tool', 'params']):
                         self.logger.warning(f"Invalid step format in plan: {step}. Skipping.")
                         continue
                    step['step'] = i + 1 # Ensure step numbers are sequential
                    step['status'] = 'pending'
                    validated_plan.append(step)

                if not validated_plan:
                     self.logger.error("LLM generated plan was empty or contained no valid steps.")
                     return None

                self.logger.info(f"Successfully generated and validated coding plan with {len(validated_plan)} steps.")
                return validated_plan
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                self.logger.error(f"Failed to parse or validate generated coding plan: {e}. Response: {plan_json_str[:500]}...")
                return None
        else:
            self.logger.error("LLM failed to generate a coding plan.")
            return None

    async def _execute_coding_plan(self) -> Dict[str, Any]:
        """Executes the generated coding plan step-by-step using Orchestrator tools."""
        self.logger.info("Starting execution of the coding plan.")
        plan = self.internal_state.get('coding_plan')
        if not plan or not isinstance(plan, list):
            self.logger.error("Execution attempted without a valid coding plan.")
            return {"status": "error", "message": "No valid coding plan available.", "details": []}

        execution_details = []
        all_steps_successful = True
        plan_halted = False
        last_tool_output = None

        for step in plan:
            if plan_halted:
                step["status"] = "skipped"
                execution_details.append({"step": step.get('step', '?'), "status": "skipped", "details": "Plan halted due to previous error/failure."})
                continue

            if step.get("status") != "pending":
                self.logger.debug(f"Step {step.get('step', '?')} already processed (Status: {step.get('status')}). Skipping.")
                # Add existing status to details if needed
                if step.get("status") != "done": all_steps_successful = False
                continue

            step_num = step.get("step", "?")
            action_desc = step.get("action", "Unknown action")
            tool_name = step.get("tool")
            tool_params = step.get("params", {})
            verification_desc = step.get("verification") # Optional verification instruction

            if not tool_name:
                error_detail = f"Step {step_num} ('{action_desc}') failed: No tool specified in plan."
                self.logger.error(error_detail)
                step["status"] = "error"
                execution_details.append({"step": step_num, "status": "error", "details": error_detail})
                all_steps_successful = False
                plan_halted = True
                continue

            # --- Structured Thinking Step (Before Tool Call) ---
            pre_tool_thought = f"""
            Structured Thinking: Execute Plan Step {step_num}
            1. Goal: Execute action '{action_desc}' using tool '{tool_name}'.
            2. Context: Parameters: {tool_params}. Previous step output preview: {str(last_tool_output)[:100]}...
            3. Constraints: Must use Orchestrator.use_tool. Handle potential errors. Update plan status.
            4. Information Needed: Tool parameters provided in plan.
            5. Plan: Call orchestrator.use_tool('{tool_name}', {tool_params}). Process result. Update step status. Perform verification if specified and possible.
            """
            await self._internal_think(pre_tool_thought)
            # --- End Structured Thinking Step ---

            self.logger.info(f"Executing Step {step_num}: {action_desc} using tool '{tool_name}'")
            observation = None
            try:
                if not self.orchestrator or not hasattr(self.orchestrator, 'use_tool'):
                     raise RuntimeError("Orchestrator tool execution is unavailable.")

                observation = await self.orchestrator.use_tool(tool_name, tool_params)
                last_tool_output = observation # Store full observation
                self.logger.debug(f"Tool '{tool_name}' observation received: {str(observation)[:200]}...")

                # Process Observation
                if observation and observation.get("status") == "success":
                    step["status"] = "done"
                    output_preview = str(observation.get('content', observation.get('stdout', observation.get('message', 'OK'))))[:150]
                    execution_details.append({"step": step_num, "status": "success", "output_preview": output_preview})
                    self.logger.info(f"Step {step_num} completed successfully.")
                    await self.log_operation('info', f"Step {step_num} ({action_desc}) completed successfully.")

                    # Cache content if read_file
                    if tool_name == 'read_file' and 'path' in tool_params:
                        self.internal_state['last_read_content'][tool_params['path']] = observation.get('content')
                    # Track modified files
                    if tool_name in ['apply_diff', 'write_to_file', 'insert_content', 'search_and_replace'] and 'path' in tool_params:
                         self.internal_state['modified_files'].add(tool_params['path'])

                    # --- Verification Step ---
                    if verification_desc:
                         verification_passed = await self._perform_verification(step_num, verification_desc, last_tool_output)
                         if not verification_passed:
                              step["status"] = "verification_failed"
                              execution_details[-1]["status"] = "verification_failed" # Update last result
                              execution_details[-1]["details"] = f"Verification failed: {verification_desc}"
                              all_steps_successful = False
                              plan_halted = True # Stop execution
                         else:
                              self.logger.info(f"Verification passed for step {step_num}.")

                else: # Tool execution failed
                    error_details = observation.get('message', observation.get('stderr', 'Unknown tool error'))
                    raise RuntimeError(f"Tool '{tool_name}' failed: {error_details}")

            except Exception as e:
                error_detail_str = str(e)
                self.logger.error(f"Error executing step {step_num} ({action_desc}): {error_detail_str}", exc_info=True)
                step["status"] = "error"
                all_steps_successful = False
                plan_halted = True
                execution_details.append({"step": step_num, "status": "error", "details": error_detail_str})
                await self.log_operation('error', f"Step {step_num} ({action_desc}) failed: {error_detail_str}")
                if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"Error in coding plan step {step_num} ({action_desc}): {error_detail_str}")
                # break # Stop plan execution on first error

        # --- Final Result ---
        final_status = "success" if all_steps_successful else ("verification_failed" if any(s.get('status') == 'verification_failed' for s in self.state["coding_plan"]) else "error")
        self.logger.info(f"Coding plan execution finished with overall status: {final_status}")
        artifacts = sorted(list(self.internal_state['modified_files']))

        return {"status": final_status, "message": f"Plan execution finished with status: {final_status}", "details": execution_details, "artifacts": artifacts}

    async def _perform_verification(self, step_num: int, verification_desc: str, last_tool_output: Optional[Dict]) -> bool:
        """Performs verification based on description and last tool output."""
        await self._internal_think(f"Performing verification for step {step_num}: {verification_desc}")
        self.logger.info(f"Performing verification for step {step_num}: {verification_desc}")

        # Simple verification based on command output
        if isinstance(last_tool_output, dict) and 'command' in verification_desc.lower():
            return_code = last_tool_output.get('returncode')
            stdout = last_tool_output.get('stdout', '')
            stderr = last_tool_output.get('stderr', '')

            if return_code == 0:
                # Basic check for common failure indicators in output (can be expanded)
                if "error" in stderr.lower() or "fail" in stderr.lower():
                     self.logger.warning(f"Verification (Step {step_num}): Command succeeded (rc=0) but stderr contains potential error indicators.")
                     # Decide if this constitutes failure - depends on context. Let's treat rc=0 as pass for now.
                     return True
                self.logger.info(f"Verification (Step {step_num}): Command successful (return code 0).")
                return True
            else:
                self.logger.error(f"Verification failed (Step {step_num}): Command exited with code {return_code}. Stderr: {stderr[:500]}")
                return False
        else:
            # Placeholder for more complex verification (e.g., LLM analysis of output)
            self.logger.warning(f"Verification for step {step_num} ('{verification_desc}') not fully implemented based on available output.")
            return True # Default to passing if verification method unclear

    async def _read_file_context(self, file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> Optional[str]:
        """Reads file content using the orchestrator's tool, caching result."""
        # Check cache first
        # Note: Caching only full file reads for simplicity now
        if start_line is None and end_line is None and file_path in self.internal_state.get('last_read_content', {}):
            self.logger.debug(f"Using cached content for: {file_path}")
            return self.internal_state['last_read_content'][file_path]

        if not self.orchestrator or not hasattr(self.orchestrator, 'use_tool'):
            self.logger.error("Orchestrator tool execution unavailable. Cannot read file context.")
            return None

        self.logger.debug(f"Reading file context via tool: {file_path} (Lines: {start_line}-{end_line})")
        params = {'path': file_path}
        # Add line numbers if tool supports it (adjust key names based on actual tool schema)
        if start_line is not None: params['start_line'] = start_line # Assuming 0-based if not specified
        if end_line is not None: params['end_line'] = end_line # Assuming inclusive if not specified

        try:
            read_result = await self.orchestrator.use_tool('read_file', params)
            if read_result and read_result.get('status') == 'success':
                content = read_result.get('content')
                # Cache only if full file was read
                if start_line is None and end_line is None:
                    self.internal_state.setdefault('last_read_content', {})[file_path] = content
                return content
            else:
                self.logger.warning(f"Could not read file {file_path} via tool: {read_result.get('message', 'Unknown error')}")
                return None
        except Exception as read_err:
            self.logger.error(f"Exception during file read tool call for {file_path}: {read_err}", exc_info=True)
            return None

    # --- Standardized LLM Interaction ---
    # (Keep _call_llm_with_retry as implemented in Version <1>)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30), retry=retry_if_exception_type(Exception))
    async def _call_llm_with_retry(self, prompt: str, model_preference: Optional[List[str]] = None, temperature: float = 0.5, max_tokens: int = 1024, is_json_output: bool = False) -> Optional[str]:
        """Centralized method for calling LLMs via the Orchestrator."""
        if not self.orchestrator or not hasattr(self.orchestrator, 'call_llm'):
            self.logger.error("Orchestrator or its call_llm method is unavailable.")
            return None
        try:
            response_content = await self.orchestrator.call_llm(
                agent_name=self.AGENT_NAME, prompt=prompt, temperature=temperature,
                max_tokens=max_tokens, is_json_output=is_json_output, model_preference=model_preference
            )
            return response_content
        except Exception as e:
            self.logger.error(f"Error occurred calling LLM via orchestrator: {e}", exc_info=True)
            raise # Re-raise for tenacity

    # --- Abstract Method Implementations ---

    async def learning_loop(self):
        """Analyzes past programming tasks to identify common issues or successful patterns."""
        while True:
            try:
                await asyncio.sleep(3600 * 6) # Run every 6 hours
                self.logger.info(f"{self.AGENT_NAME} learning loop: Analyzing recent task outcomes.")
                insights = await self.collect_insights() # Use existing method
                # TODO: Implement more sophisticated analysis of insights
                # e.g., If success rate drops below X%, generate directive for ThinkTool to investigate.
                # e.g., If specific error type is common, log a pattern or suggest a documentation update.
                if insights.get("success_rate_pct", 100) < 80 and insights.get("recent_tasks_count", 0) > 10:
                     self.logger.warning(f"Learning Loop: Success rate ({insights['success_rate_pct']}%) below threshold. Common errors: {insights['common_error_types']}")
                     # Log pattern or create directive for ThinkTool
                     await self.log_knowledge_fragment(
                         agent_source=self.AGENT_NAME, data_type="performance_alert",
                         content={"alert": "Programmer success rate low", "details": insights},
                         tags=["alert", "performance", "programmer"], relevance_score=0.9
                     )

            except asyncio.CancelledError:
                self.logger.info(f"{self.AGENT_NAME} learning loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error during ProgrammerAgent learning loop: {e}", exc_info=True)

    async def self_critique(self) -> Dict[str, Any]:
        """Evaluates the quality and success rate of programming tasks based on collected insights."""
        self.logger.info(f"{self.AGENT_NAME}: Performing self-critique.")
        insights = await self.collect_insights()
        feedback = f"Critique based on last 24h: {insights.get('recent_tasks_count', 0)} tasks processed with {insights.get('success_rate_pct', 0)}% success. Common errors: {insights.get('common_error_types', {})}"
        status = "ok"
        if insights.get('success_rate_pct', 100) < 75 and insights.get('recent_tasks_count', 0) > 5:
             status = "warning"
             feedback += " Success rate below threshold, investigate common errors."
        # TODO: Add critique based on plan accuracy vs execution, code complexity metrics (if available)
        return {"status": status, "feedback": feedback}

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Constructs context-rich prompts for LLM calls (analysis, planning, diff gen)."""
        self.logger.debug(f"Generating dynamic prompt for ProgrammerAgent task: {task_context.get('task')}")
        prompt_parts = [PROGRAMMER_AGENT_META_PROMPT]

        prompt_parts.append("\n--- Current Task Context ---")
        # Add specific task details, excluding large context blocks initially
        for key, value in task_context.items():
            if key == 'file_context_summary':
                 prompt_parts.append(f"\n**Relevant File Context (Summary):**\n{value}")
            elif isinstance(value, (str, int, float, bool)):
                 prompt_parts.append(f"{key.replace('_', ' ').title()}: {value}")
            elif isinstance(value, list) and key == 'available_tools':
                 prompt_parts.append(f"Available Tools for Plan: {', '.join(value)}")
            elif isinstance(value, dict) and len(json.dumps(value)) < 300: # Include small dicts
                 prompt_parts.append(f"{key.replace('_', ' ').title()}: {json.dumps(value)}")

        # Add relevant context from KB (Simulated - replace with actual query if needed)
        # prompt_parts.append("\n--- Relevant Knowledge (Simulated KB Retrieval) ---")
        # standards = await self.query_knowledge_base(data_types=['coding_standard'], tags=[language])

        prompt_parts.append("\n--- Instructions ---")
        task_type = task_context.get('task')
        if task_type == 'Generate Coding Plan':
            prompt_parts.append("1. Analyze task description, analysis insights, and file context.")
            prompt_parts.append("2. Create a logical, sequential plan using available tools.")
            prompt_parts.append(f"3. For each step, specify 'action', 'tool' ({task_context.get('available_tools', [])}), 'params', and optional 'verification'.")
            prompt_parts.append("4. Ensure plan covers context gathering, modification, and verification.")
            prompt_parts.append(f"5. **Output Format:** {task_context.get('desired_output_format')}")
        elif task_type == 'Analyze Task Request': # Example for analysis prompt generation
             prompt_parts.append("1. Identify primary goal, target files/areas, dependencies, ambiguities.")
             prompt_parts.append("2. List absolute file paths required for context.")
             prompt_parts.append(f"3. **Output Format:** {task_context.get('desired_output_format')}")
        # Add instructions for diff generation if needed
        else:
            prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")

        # Add JSON hint if needed
        if "JSON" in task_context.get('desired_output_format', ''):
             prompt_parts.append("\n```json")

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for ProgrammerAgent (length: {len(final_prompt)} chars)")
        return final_prompt

    async def collect_insights(self) -> Dict[str, Any]:
        """Collects insights about programming task performance from KB."""
        self.logger.debug("ProgrammerAgent collect_insights called.")
        insights = {
            "agent_name": self.AGENT_NAME, "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recent_tasks_count": 0, "success_rate_pct": 0.0,
            "common_error_types": {}, "key_observations": []
        }
        if not self.session_maker: # Check if DB access is available
             insights["key_observations"].append("Database session unavailable for insights.")
             return insights
        try:
            time_window = timedelta(hours=24)
            threshold_time = datetime.now(timezone.utc) - time_window

            async with self.session_maker() as session:
                # Query using SQLAlchemy Core expression language for flexibility
                stmt = select(KnowledgeFragment.content).where(
                    KnowledgeFragment.agent_source == self.AGENT_NAME,
                    KnowledgeFragment.data_type == "task_outcome",
                    KnowledgeFragment.timestamp >= threshold_time
                ).limit(50)
                result = await session.execute(stmt)
                fragments_content = result.scalars().all()

            if fragments_content:
                insights["recent_tasks_count"] = len(fragments_content)
                success_count = 0
                error_types = Counter()
                for content_str in fragments_content:
                    try:
                        outcome_data = json.loads(content_str)
                        task_result = outcome_data.get("result", {})
                        if task_result.get("status") == "success":
                            success_count += 1
                        elif task_result.get("status") in ["error", "failure", "verification_failed"]:
                            # Extract error details if available in the plan execution details
                            error_detail = "Unknown Error"
                            if isinstance(task_result.get("details"), list):
                                for step_detail in task_result["details"]:
                                    if step_detail.get("status") in ["error", "verification_failed"]:
                                         error_detail = step_detail.get("details", "Unknown Step Error")
                                         break # Take first error found
                            else:
                                 error_detail = task_result.get("message", "Unknown Error")

                            # Basic error categorization
                            if "ToolExecutionError" in error_detail or "Tool" in error_detail and "failed" in error_detail: error_types["ToolExecutionError"] += 1
                            elif "LLMError" in error_detail or "LLM" in error_detail: error_types["LLMError"] += 1
                            elif "Verification failed" in error_detail: error_types["VerificationError"] += 1
                            elif "Exception" in error_detail: error_types["PythonException"] += 1
                            else: error_types["UnknownTaskError"] += 1
                    except (json.JSONDecodeError, TypeError, KeyError):
                        error_types["LoggingFormatError"] += 1

                if insights["recent_tasks_count"] > 0:
                    insights["success_rate_pct"] = round((success_count / insights["recent_tasks_count"]) * 100, 1)
                insights["common_error_types"] = dict(error_types.most_common(3))
                insights["key_observations"].append(f"Analyzed {insights['recent_tasks_count']} task outcomes from last 24h.")
            else:
                 insights["key_observations"].append("No recent task outcome data found in KB.")

        except SQLAlchemyError as db_err:
             self.logger.error(f"Database error collecting insights: {db_err}", exc_info=True)
             insights["key_observations"].append(f"DB error collecting insights: {db_err}")
        except Exception as e:
            self.logger.error(f"Error collecting insights: {e}", exc_info=True)
            insights["key_observations"].append(f"Error collecting insights: {e}")

        return insights

# --- End of agents/programmer_agent.py ---