import logging
import os
import json
import asyncio # Added
import random # Added
import hashlib # Added
import time # Added
from datetime import datetime, timedelta, timezone # Added
from typing import Dict, Any, Optional, List # Added List

from agents.base_agent import BaseAgent
# Using adaptable name for flexibility, assuming OpenAI compatible interface
from openai import AsyncOpenAI as AsyncLLMClient # Added
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type # Added

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROGRAMMER_AGENT_META_PROMPT = """
You are the ProgrammerAgent within the Synapse Genius Agentic AI System.
Your Core Mandate: Develop, maintain, refactor, and adapt the agency's codebase with genius-level proficiency, efficiency, and foresight.
Key Responsibilities:
- Implement new agents and features based on specifications from ThinkTool/Orchestrator.
- Refactor existing code for modularity, efficiency, robustness, and maintainability (e.g., BrowsingAgent).
- Modify sandboxed agency instances for testing new business models (adding/removing/altering code).
- Adhere to the highest coding standards (clean, testable, well-documented).
- Plan your coding tasks meticulously (e.g., using a dedicated plan file or updates to Master_Plan.md).
- Implement efficient coding techniques and caching strategies where applicable during development.
- Ensure code quality and robustness through testing and validation.
- Operate with Extreme Agentic Behavior: Anticipate coding needs, suggest architectural improvements, optimize performance proactively, and solve problems creatively.
- Collaborate with other agents, especially ThinkTool for requirements and Orchestrator for tasking.
- Save all code to files before execution via execute_command.
"""

class ProgrammerAgent(BaseAgent):
    """
    Agent responsible for developing, maintaining, and refactoring the codebase.
    Embodies Genius Agentic AI principles for programming tasks.
    """
    def __init__(self, orchestrator):
        super().__init__(orchestrator)
        self.state = {"current_task": None, "plan": None, "status": "idle"}
        self.logger = logger # Use module-level logger
        self.logger.info("ProgrammerAgent initialized.")
        # Potentially load or initialize planning file/mechanism here

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a programming-related task based on the provided details using LLM calls.
        """
        task_description = task_details.get('description', 'No description provided.')
        self.logger.info(f"ProgrammerAgent received task: {task_description}")
        self.state["status"] = "working"
        result = {"status": "failure", "message": "Task initialization failed."}
        file_path = task_details.get('file_path')

        try:
            # 1. Analyze Request & Get Context
            self.logger.debug("Analyzing programming task request...")
            analysis_prompt = f"Analyze this programming task: {json.dumps(task_details)}. Identify target file(s), relevant code sections, and clarify requirements. If a file_path is provided, what context is needed?"
            analysis_result_str = await self._call_llm_with_retry(analysis_prompt, max_tokens=500)
            # TODO: Parse analysis_result_str to guide next steps (e.g., identify specific lines needed)

            file_content = None
            if file_path and hasattr(self.orchestrator, 'read_file'):
                 try:
                     self.logger.debug(f"Reading file context: {file_path}")
                     # TODO: Potentially use analysis_result_str to read only specific lines if needed
                     file_content_result = await self.orchestrator.use_tool('read_file', {'path': file_path})
                     if file_content_result and file_content_result.get('status') == 'success':
                         file_content = file_content_result.get('content')
                         self.logger.info(f"Successfully read content from: {file_path}")
                     else:
                         read_err_msg = file_content_result.get('message', 'Unknown read error')
                         self.logger.warning(f"Could not read file {file_path}: {read_err_msg}")
                         # Decide if task can proceed without file content
                 except Exception as read_err:
                     self.logger.warning(f"Exception during file read {file_path}: {read_err}", exc_info=True)
                     # Decide if task can proceed without file content

            # 2. Plan Changes
            self.logger.debug("Planning code modifications...")
            planning_prompt = f"Based on task '{task_description}' and analysis '{analysis_result_str}', create a plan to modify file '{file_path}'. Context:\n{file_content or 'N/A'}"
            plan_result_str = await self._call_llm_with_retry(planning_prompt, max_tokens=500)
            # TODO: Parse plan_result_str into actionable steps if needed, or use directly in diff prompt

            # 3. Generate Code Diff
            self.logger.debug("Generating code diff via LLM...")
            diff_prompt = f"""
            {PROGRAMMER_AGENT_META_PROMPT}
            **Task:** Generate code modifications in unified diff format for file '{file_path}' based on the plan: {plan_result_str}.
            **Request:** {task_description}
            **Current File Context:**
            ```
            {file_content or 'File content not available.'}
            ```
            **Output:** Provide ONLY the diff block starting with '--- a/' or '+++ b/' or similar, enclosed in ```diff ... ```. Ensure correct line numbers (:start_line:) are included if modifying existing code. If creating a new file, provide the full file content instead of a diff.
            """
            # Use lower temperature for more deterministic code generation
            generated_diff_str = await self._call_llm_with_retry(diff_prompt, max_tokens=2048, temperature=0.3)

            if not generated_diff_str:
                 raise ValueError("LLM did not generate any output for the diff.")

            diff_content = None
            # Extract diff content (simple extraction, might need refinement)
            if '```diff' in generated_diff_str:
                diff_content = generated_diff_str.split('```diff')[1].split('```')[0].strip()
                # Basic format check/correction
                if not diff_content.startswith(('--- a/', '+++ b/')):
                    diff_start_index = diff_content.find('--- a/')
                    if diff_start_index != -1:
                        diff_content = diff_content[diff_start_index:]
                    else:
                        # Maybe it's just adding lines? Check for '+++ b/'
                        diff_start_index_add = diff_content.find('+++ b/')
                        if diff_start_index_add != -1:
                             diff_content = diff_content[diff_start_index_add:]
                        else:
                             # If it doesn't look like a diff, maybe it's full content for a new file?
                             # Or maybe the LLM failed the format. Log warning and proceed cautiously.
                             self.logger.warning("Generated content doesn't strictly follow diff format. Assuming full content or malformed diff.")
                             # Keep diff_content as is for now, apply_diff might handle it or fail.
            elif file_path and not os.path.exists(file_path): # Check if file exists (needs os import)
                 # If file doesn't exist and output isn't a diff, assume it's full content for new file
                 self.logger.info(f"Target file {file_path} doesn't exist. Assuming LLM generated full content.")
                 diff_content = generated_diff_str # Use the whole output
                 # TODO: Consider using write_to_file tool instead of apply_diff for new files
            else:
                 # No diff block found, and file exists or path not specified
                 raise ValueError("LLM did not generate a valid diff block (```diff ... ```).")


            # 4. Apply Changes
            files_modified = []
            apply_success = False
            if file_path and diff_content and hasattr(self.orchestrator, 'use_tool'):
                self.logger.info(f"Attempting to apply diff to {file_path}")
                try:
                    # Use orchestrator's tool mechanism
                    apply_result = await self.orchestrator.use_tool(
                        'apply_diff',
                        {'path': file_path, 'diff': diff_content}
                    )

                    if apply_result and apply_result.get('status') == 'success':
                        apply_success = True
                        files_modified.append(file_path)
                        self.logger.info(f"Successfully applied diff to {file_path}")
                        # Store the applied diff in the result
                        result["diff_applied"] = diff_content
                    else:
                        error_msg = apply_result.get('message', 'Unknown apply_diff error')
                        self.logger.error(f"Failed to apply diff to {file_path}: {error_msg}")
                        result["message"] = f"Generated diff but failed to apply: {error_msg}"
                        # Include the generated diff in the error message for debugging
                        result["generated_diff_for_error"] = diff_content

                except Exception as apply_err:
                    self.logger.error(f"Exception during apply_diff call for {file_path}: {apply_err}", exc_info=True)
                    result["message"] = f"Generated diff but failed to apply (exception): {apply_err}"
                    result["generated_diff_for_error"] = diff_content
                    # Keep apply_success as False
            elif not file_path:
                 result["message"] = "Cannot apply diff: file_path not provided."
            elif not diff_content:
                 result["message"] = "Cannot apply diff: No valid diff content generated or extracted."
            else:
                 result["message"] = "Cannot apply diff: Orchestrator missing use_tool method."


            # 5. Verification (Placeholder)
            if apply_success:
                self.logger.debug("Skipping code verification (placeholder).")
                # TODO: Implement Linting/Testing via self.orchestrator.use_tool('execute_command', ...)

                result = {
                    "status": "success",
                    "message": f"Successfully applied generated changes for task: {task_description}",
                    "files_modified": files_modified,
                    "diff_applied": result.get("diff_applied", diff_content) # Ensure diff is included
                }
            # else: result message already set in apply_diff error handling or initial value

        except Exception as e:
            self.logger.error(f"Error executing programming task '{task_description}': {e}", exc_info=True)
            result["status"] = "failure"
            # Avoid overwriting more specific error messages if already set
            if result.get("message") == "Task initialization failed.":
                 result["message"] = f"Failed to execute task: {e}"

        finally:
            self.state["status"] = "idle"
            self.logger.info(f"ProgrammerAgent finished task: {task_description}. Status: {result['status']}")
            return result

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=4, max=30), retry=retry_if_exception_type(Exception))
    async def _call_llm_with_retry(self, prompt: str, model_preference: Optional[List[str]] = None, temperature: float = 0.5, max_tokens: int = 1024, is_json_output: bool = False) -> Optional[str]:
        """
        Centralized method for calling LLMs via the Orchestrator.
        Handles client selection, retries, error reporting, and JSON formatting.
        Adapted from ThinkTool.
        """
        llm_client: Optional[AsyncLLMClient] = None
        model_name: Optional[str] = None
        api_key_identifier: str = "unknown_key" # For logging/reporting issues

        try:
            # --- Caching Logic: Check Cache First ---
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            # Determine model name early for cache key consistency
            # TODO: Get default model from orchestrator config if possible
            default_model = "google/gemini-pro" # Placeholder default for programmer
            # TODO: Refine model selection based on model_preference if provided and stable
            model_name_for_cache = default_model # Use the determined model for the cache key

            cache_key_parts = [
                "llm_call",
                prompt_hash,
                model_name_for_cache, # Include model name
                str(temperature),
                str(max_tokens),
                str(is_json_output),
            ]
            cache_key = ":".join(cache_key_parts)
            cache_ttl = 3600 # Default 1 hour TTL for LLM calls

            # Check cache first
            if hasattr(self.orchestrator, 'get_from_cache'):
                cached_result = self.orchestrator.get_from_cache(cache_key)
                if cached_result is not None:
                    self.logger.debug(f"LLM call cache hit for key: {cache_key[:20]}...{cache_key[-20:]}")
                    return cached_result # Return cached value
                else:
                    self.logger.debug(f"LLM call cache miss for key: {cache_key[:20]}...{cache_key[-20:]}")
            else:
                self.logger.warning("Orchestrator does not have 'get_from_cache' method. Skipping cache check.")
            # --- End Cache Check ---

            # 1. Get available clients from Orchestrator
            available_clients = await self.orchestrator.get_available_openrouter_clients()
            if not available_clients:
                self.logger.error("ProgrammerAgent: No available LLM clients from Orchestrator.")
                return None

            # TODO: Implement smarter client/model selection
            llm_client = random.choice(available_clients)
            api_key_identifier = getattr(llm_client, 'api_key', 'unknown_key')[-6:] # Log last 6 chars

            # 2. Determine model name
            model_name = model_name_for_cache

            # 3. Prepare request arguments
            response_format = {"type": "json_object"} if is_json_output else None
            messages = [{"role": "user", "content": prompt}]

            self.logger.debug(f"ProgrammerAgent LLM Call (Cache Miss): Model={model_name}, Temp={temperature}, MaxTokens={max_tokens}, JSON={is_json_output}, Key=...{api_key_identifier}")

            # 4. Make the API call
            response = await llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                timeout=120 # Increased timeout for potentially longer code generation
            )

            content = response.choices[0].message.content.strip()

            # --- Token Tracking & Cost Estimation (Placeholder) ---
            input_tokens_est = len(prompt) // 4 # Rough estimate
            output_tokens = 0
            try:
                if response.usage and response.usage.completion_tokens:
                    output_tokens = response.usage.completion_tokens
                    if response.usage.prompt_tokens:
                        input_tokens_est = response.usage.prompt_tokens
                else:
                     output_tokens = len(content) // 4
            except AttributeError:
                 output_tokens = len(content) // 4

            total_tokens_est = input_tokens_est + output_tokens
            estimated_cost = total_tokens_est * 0.000001 # Placeholder cost
            self.logger.debug(f"LLM Call Est. Tokens: ~{total_tokens_est} (In: {input_tokens_est}, Out: {output_tokens}). Est. Cost: ${estimated_cost:.6f}")
            # --- End Token Tracking ---

            # --- Report Expense ---
            if estimated_cost > 0 and hasattr(self.orchestrator, 'report_expense'):
                try:
                    await self.orchestrator.report_expense(
                        agent_name="ProgrammerAgent",
                        amount=estimated_cost,
                        category="LLM",
                        description=f"LLM call ({model_name or 'unknown_model'}). Estimated tokens: {total_tokens_est}."
                    )
                except Exception as report_err:
                    self.logger.error(f"Failed to report LLM expense: {report_err}", exc_info=True)
            # --- End Report Expense ---

            # --- Caching Logic: Add to Cache on Success ---
            if content and hasattr(self.orchestrator, 'add_to_cache'):
                self.orchestrator.add_to_cache(cache_key, content, ttl_seconds=cache_ttl)
                self.logger.debug(f"Added LLM result to cache for key: {cache_key[:20]}...{cache_key[-20:]}")
            elif not hasattr(self.orchestrator, 'add_to_cache'):
                self.logger.warning("Orchestrator does not have 'add_to_cache' method. Skipping caching result.")
            # --- End Add to Cache ---

            return content

        except Exception as e:
            error_str = str(e).lower()
            issue_type = "llm_error"
            if "rate limit" in error_str or "quota" in error_str: issue_type = "rate_limit"
            elif "authentication" in error_str: issue_type = "auth_error"
            elif "timeout" in error_str: issue_type = "timeout_error"

            self.logger.warning(f"ProgrammerAgent LLM call failed (attempt): Model={model_name}, Key=...{api_key_identifier}, ErrorType={issue_type}, Error={e}")

            # Report issue back to orchestrator
            if llm_client and hasattr(self.orchestrator, 'report_client_issue'):
                 await self.orchestrator.report_client_issue(api_key_identifier, issue_type)

            raise # Reraise exception for tenacity retry logic

        # Fallback
        self.logger.error(f"ProgrammerAgent LLM call failed after all retries: Model={model_name}, Key=...{api_key_identifier}")
        return None

    def _create_coding_plan(self, task_details):
        """
        Analyzes the task and creates a detailed step-by-step plan.
        Placeholder implementation. Should involve LLM call with context.
        """
        # --- LLM-Based Planning ---
        logger.info("Intending to use LLM for detailed coding plan generation.")
        # 1. Formulate the prompt for the planning LLM.
        #    This should include:
        #    - PROGRAMMER_AGENT_META_PROMPT
        #    - task_details (description, requirements, constraints)
        #    - Relevant code context (e.g., read relevant files using orchestrator.use_tool('read_file', ...))
        #    - Instructions to output a structured plan (e.g., JSON list of steps with 'action', 'tool', 'params', 'file_path', etc.)
        # Example prompt formulation (conceptual):
        # planning_prompt = f"{PROGRAMMER_AGENT_META_PROMPT}\n\nTask: {task_details}\n\nRelevant Code:\n{code_context}\n\nGenerate a step-by-step plan..."

        # 2. (Simulated) Call the LLM via Orchestrator or a dedicated service.
        #    llm_response = await self.orchestrator.call_llm(model="planning_model", prompt=planning_prompt)
        #    parsed_plan = self._parse_llm_plan(llm_response) # Function to parse JSON/structured text

        # 3. Use a placeholder plan for now, simulating the LLM output structure.
        logger.warning("Using placeholder plan instead of actual LLM call for _create_coding_plan")
        plan = [
            {"step": 1, "action": "Read relevant file", "tool": "read_file", "params": {"path": "agents/example_agent.py"}, "status": "pending"},
            {"step": 2, "action": "Apply specific code change", "tool": "apply_diff", "params": {"path": "agents/example_agent.py", "diff": "<diff_content>"}, "status": "pending"},
            {"step": 3, "action": "Search for usage examples", "tool": "search_files", "params": {"path": "./", "regex": "ExampleClass\\(", "file_pattern": "*.py"}, "status": "pending"},
            {"step": 4, "action": "Run tests", "tool": "execute_command", "params": {"command": "pytest tests/test_example_agent.py"}, "status": "pending"},
            {"step": 5, "action": "Write updated file content", "tool": "write_to_file", "params": {"path": "agents/new_module.py", "content": "# New content...", "line_count": 10}, "status": "pending"},
        ]
        # TODO: Replace placeholder with actual LLM interaction and parsing.
        return plan

    async def _execute_coding_plan(self):
        """
        Executes the steps defined in the coding plan using available tools.
        Placeholder implementation.
        """
        logger.info("Starting execution of the coding plan.")
        if not self.state["plan"]:
            logger.error("Execution attempted without a plan.")
            raise ValueError("No coding plan available to execute.")

        execution_results = []
        all_steps_successful = True

        for step in self.state["plan"]:
            if step["status"] == "pending":
                logger.info(f"Executing Step {step['step']}: {step['action']} using tool '{step.get('tool', 'N/A')}'")
                try:
                    # --- Tool Selection Logic ---
                    # The plan should ideally specify the tool and parameters directly.
                    tool_name = step.get("tool")
                    tool_params = step.get("params", {})

                    if not tool_name:
                        logger.warning(f"Step {step['step']} does not specify a tool. Skipping.")
                        step["status"] = "skipped"
                        continue

                    # --- Simulated Tool Call ---
                    logger.debug(f"Simulating call to tool '{tool_name}' with params: {tool_params}")
                    # observation = await self.orchestrator.use_tool(tool_name, tool_params)
                    # logger.debug(f"Simulated Observation received: {observation}") # Placeholder for actual result

                    # --- Process Observation (Simulated) ---
                    # Check observation for errors or extract necessary data for subsequent steps.
                    # Example: If observation indicates failure:
                    # if observation.get("status") == "error":
                    #     raise ToolExecutionError(f"Tool '{tool_name}' failed: {observation.get('details')}")
                    # Example: If reading a file, store content:
                    # if tool_name == 'read_file':
                    #     self.state['last_read_content'] = observation.get('content')

                    # --- Update Step Status ---
                    step["status"] = "done"
                    execution_results.append({"step": step['step'], "status": "success"})
                    await self.orchestrator.notify_user(f"ProgrammerAgent: Completed step {step['step']}: {step['action']}")

                except Exception as e: # Simulate catching tool execution errors
                    logger.error(f"Error executing step {step['step']}: {step['action']}. Error: {e}", exc_info=True)
                    step["status"] = "error"
                    all_steps_successful = False
                    execution_results.append({"step": step['step'], "status": "error", "details": str(e)})
                    # Decide on error handling: stop execution, try alternative, or notify orchestrator?
                    # For now, we'll log and continue, but mark the overall execution as failed.
                    await self.orchestrator.notify_user(f"ProgrammerAgent: Failed step {step['step']}: {step['action']}. Error: {e}")
                    # break # Optionally stop execution on first error

            elif step["status"] == "done":
                logger.debug(f"Step {step['step']} already done.")
            else:
                logger.warning(f"Step {step['step']} has unexpected status: {step['status']}")

        # --- Final Result ---
        final_status = "success" if all_steps_successful else "error"
        logger.info(f"Coding plan execution finished with status: {final_status}")
        # Collect artifacts (e.g., paths of modified/created files) based on successful steps
        artifacts = [p['params']['path'] for p in self.state['plan'] if p.get('tool') in ['apply_diff', 'write_to_file'] and p['status'] == 'done'] # Example artifact collection

        return {"status": final_status, "details": execution_results, "artifacts": artifacts}

    def get_status(self):
        """Returns the current status of the agent."""
        return self.state

    async def collect_insights(self) -> Dict[str, Any]:
        """
        Placeholder implementation for collecting insights from ProgrammerAgent.
        (Required by BaseAgent).

        Returns:
            Dict[str, Any]: A dictionary containing placeholder insights.
        """
        # TODO: Implement actual insight collection logic.
        # This could include: recent task outcomes, common errors, code quality metrics, etc.
        self.logger.debug("ProgrammerAgent collect_insights called (placeholder).")
        return {
            "agent_name": "ProgrammerAgent",
            "status": "placeholder",
            "recent_tasks_count": 0,
            "errors_encountered": [],
            "key_observations": ["Placeholder insight collection."]
        }
# Example usage (within Orchestrator or main loop):
# programmer = ProgrammerAgent(orchestrator_instance)
# task = {"description": "Refactor agents/browsing_agent.py", "details": {...}}
# result = await programmer.execute_task(task)