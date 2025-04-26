# Evolution Plan: agents/browsing_agent.py (v1.0)

## 1. Objective
Evolve the `BrowsingAgent` towards the Genius Agentic Standard by enhancing its learning capabilities, implementing robust phone verification, completing the UGC workflow, and improving overall efficiency and reliability.

## 2. Identified Gaps (from Baselines & Evolution_Plan_Overall.md)
- **Brittle Strategy Inference:** Relies solely on initial LLM inference (`infer_signup_steps`) for signup strategies, which can fail or become outdated quickly due to website changes.
- **Rudimentary Learning:** Strategy caching (`strategy_cache.db`) only tracks success rates (basic moving average) and doesn't adapt the *content* of the strategies based on failures.
- **Missing/Insecure Phone Verification:** Logic is commented out and relied on insecure public SMS services. Requires a complete, secure redesign.
- **Incomplete UGC Workflow:** Core methods for video generation (`_generate_avatar_video`), editing (`_edit_video`), login (`_login_to_service`), and downloading (`_download_asset`) are non-functional placeholders.
- **Inefficiency:** Over-reliance on browser automation even when APIs might exist. No mechanism to discover or prioritize API usage.
- **Error Handling:** Could be more granular to distinguish specific Playwright/network errors from website structure issues, enabling better strategy adaptation.
- **Account Management Integration:** Uses `utils.account_manager` but needs robust handling of Vault paths and potential credential updates.

## 3. Proposed Superior Solutions & Evolutionary Path

**Phase 2.B.1: Enhanced Strategy Learning & Adaptation**
1.  **Detailed Failure Logging:**
    *   **Solution:** Modify `create_account`'s error handling. When a step fails (e.g., selector not found, timeout on element, unexpected content), log detailed context: the failed step details (action, selector, value), the step number, the specific error message/type, and potentially a snippet of the page HTML around the problematic selector. Store this failure context in `attempt_logs` table or link it.
    *   **Action:** Refactor `except` blocks within the signup loop in `create_account`. Add columns to `attempt_logs` or create a related table for failure details.
2.  **LLM-Powered Strategy Refinement:**
    *   **Solution:** Create a new async method `refine_strategy(service: str, failed_strategy_id: int, failure_context: dict)`. This method will:
        *   Fetch the failed strategy steps from `strategy_cache`.
        *   Fetch recent failure details for this strategy/service from `attempt_logs`.
        *   Construct a prompt for `ThinkTool` (or a dedicated LLM call) including the original steps, failure details (error, step number, HTML snippet), and ask it to propose *modified* steps (JSON format) to overcome the specific failure. Examples: "Suggest alternative CSS selectors", "Add a wait step before this click", "Handle potential CAPTCHA at this stage".
        *   Parse the LLM response. If valid modified steps are proposed, save them as a *new* strategy in `strategy_cache` (linked to the original or marked as a refinement). Mark the failed strategy with a lower success rate or 'needs_refinement' status.
    *   **Action:** Implement `refine_strategy`. Modify `create_account` to call `refine_strategy` upon specific, potentially recoverable signup logic failures (not proxy errors). Update `strategy_cache` schema if needed to link refined strategies.
3.  **Prioritize Learned Strategies:**
    *   **Solution:** Modify `create_account` to first try loading strategies from `strategy_cache` (`load_strategies`), ordered by success rate. Only call `infer_signup_steps` if no cached strategies exist or all cached strategies have very low success rates.
    *   **Action:** Update the beginning of `create_account` logic.

**Phase 2.B.2: Secure Phone Verification**
1.  **Research & Select Service:**
    *   **Solution:** Use `ThinkTool` or manual research (if necessary, via `ask_followup_question`) to identify reliable, API-driven temporary/virtual phone number services suitable for verification (considering cost, country availability, API quality, ethical/legal constraints). Prioritize services allowing programmatic number purchase/release and SMS retrieval. Twilio is a strong candidate already partially integrated.
    *   **Action:** Perform research/analysis. Update `Evolution_Plan_Overall.md` with chosen service(s).
2.  **Implement Service Integration:**
    *   **Solution:** Create new private methods in `BrowsingAgent` (e.g., `_acquire_virtual_number(country)`, `_release_virtual_number(number_sid)`, `_get_sms_code_from_service(number_sid, timeout)`). These methods will use `aiohttp` (with appropriate headers/auth fetched from Vault via `SecureStorage`) to interact with the chosen service's API.
    *   **Action:** Implement API interaction methods. Add necessary secrets (API keys/tokens for the chosen service) to Vault and update `Orchestrator`'s `initialize_agents` to fetch and pass them to `BrowsingAgent`.
3.  **Integrate into `create_account`:**
    *   **Solution:** Replace the commented-out phone verification logic within the signup step loop. If a step requires phone verification (`verification_type == 'phone'`):
        *   Call `_acquire_virtual_number`.
        *   Fill the number into the webpage form.
        *   Call `_get_sms_code_from_service`, waiting for the SMS.
        *   Fill the code into the verification field.
        *   Call `_release_virtual_number` in the `finally` block or after successful verification.
    *   **Action:** Modify the signup loop in `create_account`. Add robust error handling for number acquisition/SMS retrieval failures.

**Phase 2.B.3: Implement UGC Workflow**
1.  **Implement Placeholder Methods:**
    *   **Solution:** Fill in the logic for `_login_to_service`, `_generate_avatar_video`, `_edit_video`, and `_download_asset`. This will involve detailed Playwright scripting: navigating to specific URLs, finding buttons/fields using selectors (potentially inferred or configured), uploading files (script, source video, assets), waiting for processes, and triggering/handling downloads. Use the `page` object passed to these methods. Leverage `infer_signup_steps` logic or dedicated LLM calls if complex interactions need dynamic planning.
    *   **Action:** Implement the detailed Playwright logic within each placeholder method. Add necessary configuration (e.g., selectors, upload paths) to `config/settings.py` or fetch dynamically if needed.
2.  **Refine Workflow Orchestration:**
    *   **Solution:** Review the task dispatching logic in `run` and the interaction with `Orchestrator.report_ugc_step_complete`. Ensure state is passed correctly, errors are handled gracefully, and the workflow progresses logically between account acquisition, generation, editing, and potential downloading/storage.
    *   **Action:** Refine the `elif` blocks for UGC actions (`generate_ugc_video`, `edit_ugc_video`) within the `run` method.

**Phase 2.B.4: Efficiency & API Discovery**
1.  **API Discovery Task:**
    *   **Solution:** Add a new task type (e.g., `discover_api`) for `BrowsingAgent`. When triggered for a service URL, the agent uses Playwright's network monitoring capabilities (`page.on('request')`, `page.on('response')`) during a typical interaction (like login or a core action) to capture API calls made by the website's frontend. Log potential API endpoints, request methods, and payload structures.
    *   **Action:** Implement the `discover_api` task handling in the `run` method. Use Playwright network interception. Store findings in KB via `kb_interface`.
2.  **Prioritize API Usage (Future):**
    *   **Solution:** Once APIs are discovered and documented (in KB), subsequent tasks for that service should prioritize using direct `aiohttp` calls (via a dedicated method or potentially another agent) instead of `BrowsingAgent` where feasible.
    *   **Action:** This is a longer-term goal requiring significant changes to task dispatching and potentially new agent capabilities. Note this in `Evolution_Plan_Overall.md`.

## 5. Next Steps
- Implement Phase 2.B.1: Enhanced Strategy Learning & Adaptation (Detailed failure logging, `refine_strategy` method, prioritize cached strategies).
- Proceed with Phase 2.B.2: Secure Phone Verification (Research service, implement API interaction, integrate into `create_account`).
- Implement Phase 2.B.3: UGC Workflow placeholder methods.
- Implement Phase 2.B.4: API Discovery task.