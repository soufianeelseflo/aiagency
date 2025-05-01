# Filename: agents/programmer_agent.py
# Description: Production-ready Programmer Agent capable of planning, coding,
#              tool execution, verification, and handling installation tasks,
#              with integrated file-level safety mechanism.
# Version: 2.1 (File Safety Implemented)

import asyncio
import logging
import json
import os
import shlex # For safe command splitting/quoting
import hashlib
import time
import re # For parsing analysis/verification output
import uuid # For temporary filenames
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
- Plan coding tasks meticulously using structured steps. Generate structured, step-by-step plans using available tools (read_file, apply_diff, execute_command, etc.).
- Implement efficient coding techniques and caching strategies where applicable.
- Ensure code quality through testing and validation (e.g., running linters/tests via execute_command).
- Operate with Extreme Agentic Behavior: Anticipate coding needs, suggest architectural improvements, optimize performance proactively, solve problems creatively.
- Collaborate with other agents via the Orchestrator.
- Apply changes safely using temporary file copies for modification and validation before overwriting originals.
- Save all code to files using appropriate tools (`apply_diff`, `write_to_file`) before execution via `execute_command`. Handle tool installation requests.
"""

class ProgrammerAgent(GeniusAgentBase):
    """
    Agent responsible for developing, maintaining, and refactoring the codebase.
    Embodies Genius Agentic AI principles for programming tasks. Includes file-level safety.
    Version: 2.1
    """
    AGENT_NAME = "ProgrammerAgent"

    def __init__(self, orchestrator: Any, session_maker: Optional[async_sessionmaker[AsyncSession]] = None):
        """Initializes the ProgrammerAgent."""
        config = getattr(orchestrator, 'config', None)
        kb_interface = getattr(orchestrator, 'kb_interface', None)
        super().__init__(agent_name=self.AGENT_NAME, kb_interface=kb_interface, orchestrator=orchestrator, config=config)
        self.session_maker = session_maker

        self.internal_state = getattr(self, 'internal_state', {})
        self.internal_state['current_task_details'] = None
        self.internal_state['coding_plan'] = None
        self.internal_state['last_read_content'] = {}
        self.internal_state['modified_files'] = set()
        # Define a temporary directory path (relative to workspace or absolute like /tmp)
        # Using a relative path within the workspace might be better for context management
        self.temp_dir = ".ai_temp/programmer" # Example relative path
        self.logger.info(f"{self.AGENT_NAME} v2.1 (File Safety) initialized. Temp dir: {self.temp_dir}")

    async def log_operation(self, level: str, message: str):
        """Helper to log to the operational log file with agent context."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err:
            print(f"OPERATIONAL LOG FAILED ({self.agent_name}): {level} - {message} | Error: {log_err}")
            logger.error(f"Failed to write to operational log from {self.agent_name}: {log_err}")

    async def _ensure_temp_dir(self):
        """Ensures the temporary directory exists."""
        if self.orchestrator and hasattr(self.orchestrator, 'use_tool'):
            try:
                # Use execute_command to create the directory if it doesn't exist
                await self.orchestrator.use_tool('execute_command', {'command': f'mkdir -p {shlex.quote(self.temp_dir)}'})
            except Exception as e:
                self.logger.error(f"Failed to ensure temporary directory '{self.temp_dir}' exists: {e}")
                # Decide if this is a critical failure
                raise RuntimeError(f"Cannot create temp directory: {e}") from e
        else:
            raise RuntimeError("Orchestrator tool execution unavailable for temp dir creation.")

    async def _cleanup_temp_file(self, temp_file_path: str):
        """Safely removes a temporary file."""
        if temp_file_path and self.orchestrator and hasattr(self.orchestrator, 'use_tool'):
            try:
                await self._internal_think(f"Cleaning up temporary file: {temp_file_path}")
                await self.orchestrator.use_tool('execute_command', {'command': f'rm -f {shlex.quote(temp_file_path)}'})
                self.logger.debug(f"Removed temporary file: {temp_file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove temporary file '{temp_file_path}': {e}")

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a programming-related task by analyzing, planning, and using tools safely.
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

        initial_thought = f"""
        Structured Thinking: Assess Task '{action}'
        1. Goal: Fulfill programming task: {task_description}.
        2. Context: Task details: {json.dumps(task_details)}. Workspace access via Orchestrator tools.
        3. Constraints: Use Orchestrator tools. Adhere to coding standards. Use file-level safety for modifications. Generate plan first. Handle 'install_tool'.
        4. Information Needed: File contents, code structure, dependencies.
        5. Plan:
            a. If action=='install_tool', delegate to _handle_tool_installation.
            b. Otherwise (coding task):
                i. Analyze request (_analyze_request).
                ii. Get required context (_get_required_context).
                iii. Create coding plan (_create_coding_plan).
                iv. Ensure temp directory exists (_ensure_temp_dir).
                v. Execute plan safely (_execute_coding_plan_safely).
                vi. Return result.
        """
        await self._internal_think(initial_thought)

        try:
            if action == 'install_tool':
                result = await self._handle_tool_installation(task_details)
            elif action in ['implement_code', 'refactor_code', 'fix_bug', 'create_agent', 'generic_code_task']:
                # --- Standard Coding Task Flow ---
                analysis_result = await self._analyze_request(task_details)
                if not analysis_result: raise RuntimeError("Failed to analyze task request.")
                self.logger.info(f"Task Analysis Complete.")

                required_files = analysis_result.get("required_files", [])
                file_context_map = await self._get_required_context(required_files)

                self.state["coding_plan"] = await self._create_coding_plan(task_details, analysis_result, file_context_map)
                if not self.state["coding_plan"]: raise RuntimeError("Failed to create a coding plan.")
                await self.log_operation('info', f"Generated coding plan with {len(self.state['coding_plan'])} steps.")
                await self.log_knowledge_fragment(
                    agent_source=self.AGENT_NAME, data_type="coding_plan",
                    content=self.state["coding_plan"], tags=["plan", action], relevance_score=0.9
                )

                await self._ensure_temp_dir() # Make sure temp dir exists before execution

                await self._internal_think(f"Starting safe execution of {len(self.state['coding_plan'])} planned steps.")
                execution_result = await self._execute_coding_plan_safely() # Use the safe execution method

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
            # Keep plan in state for insights
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

    async def _handle_tool_installation(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the specific task of installing a tool."""
        tool_name = task_details.get('tool_name')
        package_manager = task_details.get('package_manager')
        package_name = task_details.get('package_name')
        git_repo = task_details.get('git_repo')
        self.logger.info(f"Handling installation request for tool: {tool_name}")

        if not tool_name or not (package_manager and package_name) and not git_repo:
            return {"status": "failure", "message": "Missing tool name, package manager/name, or git repo for installation."}

        command = self._determine_install_command(tool_name, package_manager, package_name, git_repo)
        if not command:
             return {"status": "failure", "message": f"Could not determine install command for tool: {tool_name}"}

        if self.orchestrator and hasattr(self.orchestrator, 'use_tool'):
            await self._internal_think(f"Executing install command for '{tool_name}': {command}")
            await self.log_operation('info', f"Executing install command: {command}")
            try:
                # Installation commands can modify system state, run directly (no temp copy needed)
                exec_result = await self.orchestrator.use_tool(
                    'execute_command',
                    {'command': command, 'purpose': f'Install tool {tool_name}'}
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
        # TODO: Get OS info from Orchestrator/Config if possible
        os_type = "linux" # Assume linux default

        if git_repo:
            clone_dir = f"/tmp/{tool_name.lower()}_{uuid.uuid4().hex[:6]}"
            # Enhanced command to try common build/install methods
            return f"git clone --depth 1 {shlex.quote(git_repo)} {shlex.quote(clone_dir)} && cd {shlex.quote(clone_dir)} && (make install || python3 setup.py install --user || pip3 install --user . || npm install -g || cargo install --path .) && cd / && rm -rf {shlex.quote(clone_dir)}"

        if pkg_manager and pkg_name:
            pkg_manager = pkg_manager.lower()
            safe_pkg_name = shlex.quote(pkg_name)
            if pkg_manager == 'apt': return f"sudo apt-get update && sudo apt-get install -y {safe_pkg_name}"
            elif pkg_manager == 'pip' or pkg_manager == 'pip3': return f"pip3 install --user {safe_pkg_name}"
            elif pkg_manager == 'npm': return f"npm install -g {safe_pkg_name}"
            elif pkg_manager == 'yum': return f"sudo yum install -y {safe_pkg_name}"
            elif pkg_manager == 'brew': return f"brew install {safe_pkg_name}"
            else: self.logger.warning(f"Unsupported package manager '{pkg_manager}' for tool '{tool_name}'."); return None
        else: # Fallback heuristic
             tool_lower = tool_name.lower()
             if tool_lower in ["theharvester", "recon-ng", "exiftool", "nmap", "sqlmap", "jq"]: return f"sudo apt-get update && sudo apt-get install -y {tool_lower}"
             elif tool_lower in ["photon", "requests", "beautifulsoup4", "playwright", "pytest", "flake8", "numpy", "scikit-learn", "spacy", "pybreaker", "pytz", "sqlalchemy", "asyncpg", "psutil", "reportlab", "openai", "google-generativeai", "tenacity", "prometheus-client", "stable-baselines3", "gymnasium", "aiohttp", "faker", "deepgram-sdk", "websockets", "python-dotenv", "aiokafka", "hcp-vault-secrets", "cryptography"]:
                 pip_command = f"pip3 install --user {tool_lower}"
                 if tool_lower == "playwright": pip_command += " && python3 -m playwright install --with-deps"
                 if tool_lower == "spacy": pip_command += " && python3 -m spacy download en_core_web_sm" # Add model download
                 return pip_command
             elif tool_lower in ["vite", "next", "eslint", "prettier", "tailwindcss", "typescript", "@hookform/resolvers", "@radix-ui/*", "@tanstack/react-query", "class-variance-authority", "clsx", "cmdk", "date-fns", "embla-carousel-react", "input-otp", "lucide-react", "next-themes", "react", "react-dom", "react-day-picker", "react-hook-form", "react-resizable-panels", "react-router-dom", "recharts", "sonner", "tailwind-merge", "tailwindcss-animate", "vaul", "zod"]:
                  return f"npm install -g {tool_lower}" # Or maybe local install `npm install` in project dir? Needs context.
             else: self.logger.warning(f"Cannot determine install command for tool: {tool_name} without package manager info."); return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception))
    async def _create_coding_plan(self, task_details: Dict[str, Any], analysis_result: Optional[Dict[str, Any]], file_context_map: Dict[str, Optional[str]]) -> Optional[List[Dict]]:
        """Creates a detailed step-by-step coding plan using an LLM."""
        self.logger.info("Generating coding plan via LLM...")
        task_description = task_details.get('description', 'N/A')

        file_context_prompt = ""
        for path, content in file_context_map.items():
            file_context_prompt += f"\n--- Context: {path} ---\n"
            if content: file_context_prompt += content[:2000] + ("..." if len(content) > 2000 else "")
            else: file_context_prompt += "[Could not read file content]"
            file_context_prompt += "\n---------------------\n"

        await self._internal_think("Formulating LLM prompt for coding plan generation.")
        planning_prompt = f"""
        {PROGRAMMER_AGENT_META_PROMPT[:500]}...

        **Task:** Create a detailed, step-by-step execution plan (JSON list) to accomplish the programming task.

        **Programming Task Details:** {json.dumps(task_details, indent=2)}
        **Initial Analysis Insights:** {json.dumps(analysis_result, indent=2) if analysis_result else 'N/A'}
        **Relevant File Context:** {file_context_prompt if file_context_prompt else 'N/A'}
        **Available Tools:** read_file, apply_diff, write_to_file, insert_content, search_and_replace, execute_command, search_files, list_files.

        **Instructions:**
        Generate a plan as a JSON list of dictionaries. Each dictionary represents a step and MUST contain:
        - "step" (int): Sequential step number starting from 1.
        - "action" (str): Brief description (e.g., "Read main function", "Apply refactoring diff", "Run unit tests").
        - "tool" (str): Exact Orchestrator tool name (e.g., "read_file", "apply_diff", "execute_command").
        - "params" (dict): Parameters for the tool (e.g., {{"path": "/path/to/file.py", "diff": "..."}}, {{"command": "pytest"}}).
        - "verification" (str, optional): Description of how to verify step success (e.g., "Check command output for 'Success'", "Run linter on modified file").

        Plan should be logical, sequential, cover context gathering, modification, and verification (add 'execute_command' steps for linters/tests after modifications). Ensure file paths are correct.

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
            temperature=0.2, max_tokens=2000, is_json_output=True
        )

        if plan_json_str:
            try:
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
                    step['step'] = i + 1
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

    async def _execute_coding_plan_safely(self) -> Dict[str, Any]:
        """Executes the coding plan using file-level safety mechanisms."""
        self.logger.info("Starting SAFE execution of the coding plan.")
        plan = self.internal_state.get('coding_plan')
        if not plan or not isinstance(plan, list):
            self.logger.error("Safe execution attempted without a valid coding plan.")
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
                self.logger.debug(f"Skipping step {step.get('step', '?')} with status {step.get('status')}.")
                if step.get("status") != "done": all_steps_successful = False
                continue

            step_num = step.get("step", "?")
            action_desc = step.get("action", "Unknown action")
            tool_name = step.get("tool")
            tool_params = step.get("params", {}).copy() # Work with a copy
            verification_desc = step.get("verification")
            original_file_path = tool_params.get('path') # Path for file operations
            temp_file_path = None
            is_modification_step = tool_name in ['apply_diff', 'write_to_file', 'insert_content', 'search_and_replace'] and original_file_path

            await self._internal_think(f"Executing Plan Step {step_num}: {action_desc} (Tool: {tool_name}, Params: {tool_params})")
            await self.log_operation('info', f"Executing Step {step_num}: {action_desc} using tool '{tool_name}'")

            if not tool_name or not hasattr(self.orchestrator, 'use_tool'):
                error_detail = f"Step {step_num} failed: Invalid tool '{tool_name}' or orchestrator cannot use tools."
                self.logger.error(error_detail); step["status"] = "error"
                execution_details.append({"step": step_num, "status": "error", "details": error_detail})
                all_steps_successful = False; plan_halted = True
                continue

            try:
                # --- File Safety Logic ---
                if is_modification_step:
                    # Check if original file exists (needed for copy, matters for write_to_file)
                    # This check itself might need a tool call if path is complex/relative
                    # Assuming simple path check for now, enhance if needed
                    original_exists = os.path.exists(original_file_path) # This check might be inaccurate in container/remote env

                    if tool_name == 'write_to_file' and not original_exists:
                        # Writing a NEW file, no temp copy needed for this step
                        self.logger.debug(f"Step {step_num}: Tool is write_to_file for a new file '{original_file_path}'. Proceeding directly.")
                        is_modification_step = False # Treat as non-modification for safety logic
                    else:
                        # Create temporary copy
                        temp_file_name = f"temp_{uuid.uuid4().hex[:8]}_{os.path.basename(original_file_path)}"
                        temp_file_path = os.path.join(self.temp_dir, temp_file_name)
                        await self._internal_think(f"Step {step_num}: Creating temp copy of '{original_file_path}' at '{temp_file_path}'.")
                        copy_command = f"cp {shlex.quote(original_file_path)} {shlex.quote(temp_file_path)}"
                        copy_result = await self.orchestrator.use_tool('execute_command', {'command': copy_command})
                        if not copy_result or copy_result.get('status') != 'success':
                            raise RuntimeError(f"Failed to create temporary copy of {original_file_path}: {copy_result.get('message', 'Unknown copy error')}")
                        self.logger.info(f"Created temporary copy: {temp_file_path}")
                        # Update parameters to use the temporary path for modification
                        tool_params['path'] = temp_file_path

                # --- Execute Tool (on original or temp file) ---
                observation = await self.orchestrator.use_tool(tool_name, tool_params)
                last_tool_output = observation
                self.logger.debug(f"Tool '{tool_name}' observation received: {str(observation)[:200]}...")

                if not observation or observation.get("status") != "success":
                    error_details = observation.get('message', observation.get('stderr', 'Unknown tool error'))
                    raise RuntimeError(f"Tool '{tool_name}' failed: {error_details}")

                # --- Post-Execution Logic (Validation, Replace Original) ---
                step_success = True # Assume success unless verification fails
                if is_modification_step:
                    # Verification (on temp file)
                    if verification_desc:
                        verification_passed = await self._perform_verification(step_num, verification_desc, last_tool_output, temp_file_path) # Pass temp path
                        if not verification_passed:
                            step_success = False
                            step["status"] = "verification_failed"
                            execution_details.append({"step": step_num, "status": "verification_failed", "details": f"Verification failed: {verification_desc}"})
                            all_steps_successful = False
                            plan_halted = True # Stop after verification failure
                        else:
                            self.logger.info(f"Verification passed for step {step_num} on temp file.")

                    # Replace original if modification and verification succeeded
                    if step_success:
                        await self._internal_think(f"Step {step_num}: Modification and verification successful for temp file. Replacing original '{original_file_path}'.")
                        replace_command = f"mv {shlex.quote(temp_file_path)} {shlex.quote(original_file_path)}"
                        replace_result = await self.orchestrator.use_tool('execute_command', {'command': replace_command})
                        if not replace_result or replace_result.get('status') != 'success':
                            # Critical error if replacement fails
                            error_detail = f"CRITICAL: Failed to replace original file '{original_file_path}' with modified temp file '{temp_file_path}': {replace_result.get('message', 'Unknown move error')}"
                            self.logger.error(error_detail)
                            step["status"] = "error"
                            execution_details.append({"step": step_num, "status": "error", "details": error_detail})
                            all_steps_successful = False
                            plan_halted = True
                            # Don't clean up temp file in this case, might be needed for recovery
                            temp_file_path = None # Prevent cleanup in finally
                        else:
                            self.logger.info(f"Successfully replaced '{original_file_path}' with verified temporary file.")
                            self.internal_state['modified_files'].add(original_file_path) # Track original path as modified
                            # Clean up successful temp file now handled by finally block

                # Update status and results if step hasn't failed verification
                if step_success and not plan_halted:
                    step["status"] = "done"
                    output_preview = str(observation.get('content', observation.get('stdout', observation.get('message', 'OK'))))[:150]
                    execution_details.append({"step": step_num, "status": "success", "output_preview": output_preview})
                    self.logger.info(f"Step {step_num} completed successfully.")
                    await self.log_operation('info', f"Step {step_num} ({action_desc}) completed successfully.")
                    # Cache content if read_file
                    if tool_name == 'read_file' and 'path' in tool_params:
                        self.internal_state.setdefault('last_read_content', {})[tool_params['path']] = observation.get('content')

            except Exception as e:
                error_detail_str = str(e)
                self.logger.error(f"Error executing step {step_num} ({action_desc}): {error_detail_str}", exc_info=True)
                step["status"] = "error"
                all_steps_successful = False
                plan_halted = True
                execution_details.append({"step": step_num, "status": "error", "details": error_detail_str})
                await self.log_operation('error', f"Step {step_num} ({action_desc}) failed: {error_detail_str}")
                if self.orchestrator: await self.orchestrator.report_error(self.AGENT_NAME, f"Error in coding plan step {step_num} ({action_desc}): {error_detail_str}")
            finally:
                # Ensure temporary file is cleaned up if it was created and not successfully moved
                if temp_file_path:
                    await self._cleanup_temp_file(temp_file_path)

        # --- Final Result ---
        final_status = "success" if all_steps_successful else ("verification_failed" if any(s.get('status') == 'verification_failed' for s in self.state["coding_plan"]) else "error")
        self.logger.info(f"Coding plan execution finished with overall status: {final_status}")
        artifacts = sorted(list(self.internal_state['modified_files']))

        return {"status": final_status, "message": f"Plan execution finished with status: {final_status}", "details": execution_details, "artifacts": artifacts}

    async def _perform_verification(self, step_num: int, verification_desc: str, last_tool_output: Optional[Dict], file_path_to_verify: Optional[str] = None) -> bool:
        """Performs verification based on description, tool output, and potentially the modified file."""
        await self._internal_think(f"Performing verification for step {step_num}: {verification_desc}")
        self.logger.info(f"Performing verification for step {step_num}: {verification_desc}")

        # 1. Check Command Output (if last step was execute_command)
        if isinstance(last_tool_output, dict) and 'command' in verification_desc.lower():
            return_code = last_tool_output.get('returncode')
            stdout = last_tool_output.get('stdout', '')
            stderr = last_tool_output.get('stderr', '')
            if return_code == 0 and "error" not in stderr.lower() and "fail" not in stderr.lower():
                self.logger.info(f"Verification (Step {step_num}): Command successful (return code 0).")
                return True
            else:
                self.logger.error(f"Verification failed (Step {step_num}): Command exited with code {return_code}. Stderr: {stderr[:500]}")
                return False

        # 2. Run Linter/Tests (if specified and file path available)
        elif ("lint" in verification_desc.lower() or "test" in verification_desc.lower()) and file_path_to_verify:
            command = None
            if "lint" in verification_desc.lower():
                # Determine linter command (e.g., flake8, eslint) - needs context or config
                linter = "flake8" if file_path_to_verify.endswith(".py") else "eslint" # Basic guess
                command = f"{linter} {shlex.quote(file_path_to_verify)}"
            elif "test" in verification_desc.lower():
                # Determine test command (e.g., pytest, npm test) - needs context or config
                tester = "pytest" if file_path_to_verify.endswith(".py") else "npm test --" # Basic guess
                command = f"{tester} {shlex.quote(file_path_to_verify)}" # Test specific file if possible

            if command and self.orchestrator and hasattr(self.orchestrator, 'use_tool'):
                await self._internal_think(f"Running verification command: {command}")
                verify_result = await self.orchestrator.use_tool('execute_command', {'command': command})
                if verify_result and verify_result.get('status') == 'success' and verify_result.get('returncode') == 0:
                    self.logger.info(f"Verification successful (Step {step_num}): Command '{command}' passed.")
                    return True
                else:
                    error_msg = verify_result.get('message', verify_result.get('stderr', 'Verification command failed'))
                    self.logger.error(f"Verification failed (Step {step_num}): Command '{command}' failed: {error_msg}")
                    return False
            else:
                 self.logger.warning(f"Could not execute verification command for step {step_num}.")
                 return False # Cannot verify

        # 3. LLM Analysis (Fallback or specific request)
        elif "llm" in verification_desc.lower() and file_path_to_verify:
             await self._internal_think(f"Performing LLM-based verification for step {step_num} on file {file_path_to_verify}")
             # Read the modified temp file content
             modified_content = await self._read_file_context(file_path_to_verify)
             if not modified_content:
                  self.logger.error(f"Verification failed (Step {step_num}): Could not read modified temp file {file_path_to_verify} for LLM analysis.")
                  return False

             llm_verify_prompt = f"Verify if the following code in '{file_path_to_verify}' meets the requirement: '{self.state['current_task_details'].get('description', '')}'. Specifically check: {verification_desc}. Respond ONLY with 'True' or 'False'.\n\nCode:\n```\n{modified_content[:3000]}\n```"
             llm_verdict = await self._call_llm_with_retry(llm_verify_prompt, temperature=0.0, max_tokens=10)
             if llm_verdict and llm_verdict.strip().lower() == 'true':
                  self.logger.info(f"Verification successful (Step {step_num}): LLM confirmed criteria met.")
                  return True
             else:
                  self.logger.error(f"Verification failed (Step {step_num}): LLM indicated criteria not met or failed. Verdict: {llm_verdict}")
                  return False
        else:
            self.logger.warning(f"Verification for step {step_num} ('{verification_desc}') could not be performed based on available output/methods.")
            return True # Default to passing if verification method unclear/unsupported

    async def _read_file_context(self, file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> Optional[str]:
        """Reads file content using the orchestrator's tool, caching result."""
        # Check cache only for full file reads
        is_full_read = start_line is None and end_line is None
        if is_full_read and file_path in self.internal_state.get('last_read_content', {}):
            self.logger.debug(f"Using cached content for: {file_path}")
            return self.internal_state['last_read_content'][file_path]

        if not self.orchestrator or not hasattr(self.orchestrator, 'use_tool'):
            self.logger.error("Orchestrator tool execution unavailable. Cannot read file context.")
            return None

        self.logger.debug(f"Reading file context via tool: {file_path} (Lines: {start_line}-{end_line})")
        params = {'path': file_path}
        # Adapt parameter names if Orchestrator's read_file tool uses different keys
        if start_line is not None: params['start_line'] = start_line
        if end_line is not None: params['end_line'] = end_line

        try:
            read_result = await self.orchestrator.use_tool('read_file', params)
            if read_result and read_result.get('status') == 'success':
                content = read_result.get('content')
                if is_full_read: # Cache only full reads
                    self.internal_state.setdefault('last_read_content', {})[file_path] = content
                return content
            else:
                self.logger.warning(f"Could not read file {file_path} via tool: {read_result.get('message', 'Unknown error')}")
                return None
        except Exception as read_err:
            self.logger.error(f"Exception during file read tool call for {file_path}: {read_err}", exc_info=True)
            return None

    # --- Abstract Method Implementations ---

    async def learning_loop(self):
        """Analyzes past programming tasks to identify common issues or successful patterns."""
        while True:
            try:
                await asyncio.sleep(3600 * 6) # Run every 6 hours
                self.logger.info(f"{self.AGENT_NAME} learning loop: Analyzing recent task outcomes.")
                insights = await self.collect_insights()
                if insights.get("success_rate_pct", 100) < 80 and insights.get("recent_tasks_count", 0) > 10:
                     self.logger.warning(f"Learning Loop: Success rate ({insights['success_rate_pct']}%) below threshold. Common errors: {insights['common_error_types']}")
                     await self.log_knowledge_fragment(
                         agent_source=self.AGENT_NAME, data_type="performance_alert",
                         content={"alert": "Programmer success rate low", "details": insights},
                         tags=["alert", "performance", "programmer"], relevance_score=0.9
                     )
                     # TODO: Generate directive for ThinkTool to investigate further?
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
        for key, value in task_context.items():
            if key == 'file_context_summary': prompt_parts.append(f"\n**Relevant File Context (Summary):**\n{value}")
            elif isinstance(value, (str, int, float, bool)): prompt_parts.append(f"{key.replace('_', ' ').title()}: {value}")
            elif isinstance(value, list) and key == 'available_tools': prompt_parts.append(f"Available Tools for Plan: {', '.join(value)}")
            elif isinstance(value, dict) and len(json.dumps(value)) < 300: prompt_parts.append(f"{key.replace('_', ' ').title()}: {json.dumps(value)}")

        prompt_parts.append("\n--- Instructions ---")
        task_type = task_context.get('task')
        if task_type == 'Generate Coding Plan':
            prompt_parts.append("1. Analyze task description, analysis insights, and file context.")
            prompt_parts.append("2. Create a logical, sequential plan using available tools.")
            prompt_parts.append(f"3. For each step, specify 'action', 'tool' ({task_context.get('available_tools', [])}), 'params', and optional 'verification'.")
            prompt_parts.append("4. Ensure plan covers context gathering, modification, and verification.")
            prompt_parts.append(f"5. **Output Format:** {task_context.get('desired_output_format')}")
        elif task_type == 'Analyze Task Request':
             prompt_parts.append("1. Identify primary goal, target files/areas, dependencies, ambiguities.")
             prompt_parts.append("2. List absolute file paths required for context.")
             prompt_parts.append(f"3. **Output Format:** {task_context.get('desired_output_format')}")
        else:
            prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")

        if "JSON" in task_context.get('desired_output_format', ''): prompt_parts.append("\n```json")

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
        if not self.session_maker:
             insights["key_observations"].append("Database session unavailable for insights.")
             return insights
        try:
            time_window = timedelta(hours=24)
            threshold_time = datetime.now(timezone.utc) - time_window
            async with self.session_maker() as session:
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
                        if task_result.get("status") == "success": success_count += 1
                        elif task_result.get("status") in ["error", "failure", "verification_failed"]:
                            error_detail = "Unknown Error"
                            if isinstance(task_result.get("details"), list):
                                for step_detail in task_result["details"]:
                                    if step_detail.get("status") in ["error", "verification_failed"]:
                                         error_detail = step_detail.get("details", "Unknown Step Error"); break
                            else: error_detail = task_result.get("message", "Unknown Error")
                            # Basic error categorization
                            if "ToolExecutionError" in error_detail or "Tool" in error_detail and "failed" in error_detail: error_types["ToolExecutionError"] += 1
                            elif "LLMError" in error_detail or "LLM" in error_detail: error_types["LLMError"] += 1
                            elif "Verification failed" in error_detail: error_types["VerificationError"] += 1
                            elif "Exception" in error_detail: error_types["PythonException"] += 1
                            else: error_types["UnknownTaskError"] += 1
                    except (json.JSONDecodeError, TypeError, KeyError): error_types["LoggingFormatError"] += 1

                if insights["recent_tasks_count"] > 0: insights["success_rate_pct"] = round((success_count / insights["recent_tasks_count"]) * 100, 1)
                insights["common_error_types"] = dict(error_types.most_common(3))
                insights["key_observations"].append(f"Analyzed {insights['recent_tasks_count']} task outcomes from last 24h.")
            else: insights["key_observations"].append("No recent task outcome data found in KB.")

        except SQLAlchemyError as db_err: self.logger.error(f"Database error collecting insights: {db_err}", exc_info=True); insights["key_observations"].append(f"DB error collecting insights: {db_err}")
        except Exception as e: self.logger.error(f"Error collecting insights: {e}", exc_info=True); insights["key_observations"].append(f"Error collecting insights: {e}")

        return insights

    # --- KB Interaction Helpers (Delegate or Direct) ---
    async def log_knowledge_fragment(self, *args, **kwargs):
        """Logs a knowledge fragment using the best available method."""
        if self.kb_interface and hasattr(self.kb_interface, 'log_knowledge_fragment'):
            return await self.kb_interface.log_knowledge_fragment(*args, **kwargs)
        elif self.think_tool and hasattr(self.think_tool, 'log_knowledge_fragment'):
            return await self.think_tool.log_knowledge_fragment(*args, **kwargs)
        else: # Fallback: Direct DB interaction if session_maker is available
            if self.session_maker:
                 try:
                     async with self.session_maker() as session:
                          fragment = KnowledgeFragment(*args, **kwargs) # Assumes args match model
                          session.add(fragment)
                          await session.commit()
                          await session.refresh(fragment)
                          self.logger.info(f"Logged KnowledgeFragment directly: ID={fragment.id}")
                          return fragment
                 except Exception as e:
                      self.logger.error(f"Direct DB logging failed for KnowledgeFragment: {e}")
            else:
                 self.logger.error("No mechanism available to log knowledge fragment.")
            return None

    async def query_knowledge_base(self, *args, **kwargs):
        """Queries the knowledge base using the best available method."""
        if self.kb_interface and hasattr(self.kb_interface, 'query_knowledge_base'):
            return await self.kb_interface.query_knowledge_base(*args, **kwargs)
        elif self.think_tool and hasattr(self.think_tool, 'query_knowledge_base'):
            return await self.think_tool.query_knowledge_base(*args, **kwargs)
        else: # Fallback: Direct DB interaction if session_maker is available
             if self.session_maker:
                  try:
                      async with self.session_maker() as session:
                           # Reconstruct query logic here based on args/kwargs
                           stmt = select(KnowledgeFragment) # Basic example
                           # Add filtering based on kwargs...
                           result = await session.execute(stmt.limit(kwargs.get('limit', 10)))
                           return result.scalars().all()
                  except Exception as e:
                       self.logger.error(f"Direct DB query failed for KnowledgeBase: {e}")
             else:
                  self.logger.error("No mechanism available to query knowledge base.")
             return []

# --- End of agents/programmer_agent.py ---
            return await self.kb_interface.query_knowledge_base(*args, **kwargs)
        elif self.think_tool and hasattr(self.think_tool, 'query_knowledge_base'):
            return await self.think_tool.query_knowledge_base(*args, **kwargs)
        else: # Fallback: Direct DB interaction
             if self.session_maker:
                  try:
                      async with self.session_maker() as session:
                           # Reconstruct query logic here based on args/kwargs
                           # This is complex and ideally handled by ThinkTool/KBInterface
                           stmt = select(KnowledgeFragment) # Basic example
                           # Add filtering based on kwargs...
                           result = await session.execute(stmt.limit(kwargs.get('limit', 10)))
                           return result.scalars().all()
                  except Exception as e:
                       self.logger.error(f"Direct DB query failed for KnowledgeBase: {e}")
             else:
                  self.logger.error("No mechanism available to query knowledge base.")
             return []

# --- End of agents/programmer_agent.py ---