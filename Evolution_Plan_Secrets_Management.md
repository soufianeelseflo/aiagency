# Evolution Plan: Secrets Management (v1.0)

## 1. Objective
Enhance the security and robustness of secrets management by transitioning critical secrets from environment variables to HCP Vault, improving the `SecureStorage` utility, and ensuring consistent usage across the application, aligning with the Genius Agentic Standard.

## 2. Identified Gaps (from Baselines & Evolution_Plan_Overall.md)
- **Over-reliance on Environment Variables:** Numerous sensitive secrets (API keys, passwords for SMTP, IMAP, Twilio, etc.) are loaded directly from environment variables via `config/settings.py`. This increases the attack surface and makes management/rotation harder.
- **`SecureStorage` Limitations:**
    - Lacks resilience patterns (retries, circuit breaking) for HCP Vault API calls.
    - Error handling is basic (generic `ValueError`).
    - `add_credential` uses an inefficient and potentially unsafe read-modify-write pattern.
    - API version is hardcoded.
- **Inconsistent Usage:** While `SecureStorage` exists and HCP Vault settings are configured, it's unclear how extensively it's used versus direct environment variable access (e.g., `Orchestrator` checks env vars directly, `Notifications` uses config object likely backed by env vars).
- **Root Token Security:** The `HCP_API_TOKEN` itself, loaded from an environment variable, is a critical dependency requiring secure handling.

## 3. Proposed Superior Solutions

1.  **Transition Secrets to Vault:**
    *   Identify all sensitive credentials currently loaded from environment variables in `config/settings.py` (e.g., `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `HOSTINGER_SMTP_PASS`, `HOSTINGER_IMAP_PASS`, `TWILIO_AUTH_TOKEN`, `SMARTPROXY_PASSWORD`, `DEEPGRAM_API_KEY`, etc.).
    *   Define a clear naming convention for secrets within HCP Vault (e.g., `service-api-key`, `smtp-password`, `twilio-auth-token`).
    *   Store these secrets securely in the configured HCP Vault application (`HCP_APP_NAME`).
    *   Modify `config/settings.py` to load *fewer* secrets directly. Instead, it should primarily load configuration needed to *access* Vault (Org ID, Project ID, App Name, API Token).
    *   Modify relevant components (Orchestrator, Agents, Utilities like `Notifications`) to fetch required secrets *at runtime* from Vault using the `SecureStorage` interface.
2.  **Enhance `SecureStorage` (`utils/secure_storage.py`):**
    *   **Resilience:** Implement `tenacity` for retries with exponential backoff on Vault API calls (`get_secret`, `set_secret`). Consider adding a `pybreaker` circuit breaker.
    *   **Error Handling:** Catch specific `aiohttp` exceptions (e.g., `ClientConnectorError`, `ClientResponseError`) and provide more informative error messages or custom exceptions instead of generic `ValueError`. Log API response details on failure.
    *   **`add_credential` Improvement:** Investigate if HCP Vault API offers atomic operations for updating secrets or parts of secrets (like appending to a list stored in JSON). If not, implement locking or a safer update mechanism if concurrent writes are a risk (though less likely in the current single orchestrator model). For simplicity initially, retain the read-modify-write but add robust error handling.
    *   **API Version:** Consider making the API version configurable or dynamically discoverable if possible, although `2023-11-28` might be stable for now.
3.  **Secure Root Token (`HCP_API_TOKEN`):**
    *   While loading from an environment variable is common, emphasize in documentation/deployment procedures the need to inject this variable securely (e.g., via Coolify's secrets management, not hardcoded in Dockerfiles).
    *   Recommend using short-lived HCP tokens or authentication methods if feasible in the deployment environment. (This is more an operational recommendation than a code change).
4.  **Refactor Secret Usage:**
    *   Audit codebase (`Orchestrator`, agents, `utils/notifications.py`, etc.) for direct `os.getenv()` calls or `config.get()` calls retrieving secrets that *should* now come from Vault.
    *   Refactor these components to use `orchestrator.secure_storage.get_secret()` or equivalent. Pass the `SecureStorage` instance where needed.

## 4. Detailed Implementation Steps

1.  **Vault Setup & Secret Population:**
    *   **Action (Manual/Ops):** Ensure the HCP Vault is configured correctly. Define and populate the necessary secrets (e.g., `openrouter-api-key`, `deepseek-api-key`, `hostinger-smtp-pass`, `twilio-auth-token`, etc.) within the Vault application specified by `HCP_ORGANIZATION_ID`, `HCP_PROJECT_ID`, `HCP_APP_NAME`.
2.  **Enhance `SecureStorage` (`utils/secure_storage.py`):**
    *   **Action:** Add `@retry` decorator (from `tenacity`) to `get_secret` and `set_secret` methods. Configure appropriate wait strategies and retry conditions (e.g., retry on 5xx errors or connection errors).
    *   **Action:** Add `@circuitbreaker` decorator (from `pybreaker`) to `get_secret` and `set_secret`.
    *   **Action:** Refine `try...except` blocks to catch specific `aiohttp` exceptions and log/raise more informative errors.
    *   **Action:** Review `add_credential` - initially keep read-modify-write but ensure robust error handling around JSON parsing and `set_secret` calls.
3.  **Refactor `config/settings.py`:**
    *   **Action:** Remove the direct loading (`os.getenv`) of secrets that will now be fetched from Vault. Keep only essential non-secret config and Vault access details.
    *   **Action:** Ensure the `Settings` class still correctly initializes attributes (they might become `None` initially if not fetched from Vault yet). Code relying on settings must adapt.
4.  **Refactor `Orchestrator` (`agents/orchestrator.py`):**
    *   **Action:** Remove direct `os.getenv` checks for secrets during initialization (lines 154-166). Validation should happen when secrets are *fetched* from Vault or rely on Vault's presence check.
    *   **Action:** Modify `initialize_clients` to fetch `OPENROUTER_API_KEY` and `DEEPSEEK_API_KEY` from `self.secure_storage.get_secret()`. Handle potential errors during fetch.
    *   **Action:** Pass `self.secure_storage` instance to agents that need it during their initialization, or provide a method for agents to access it via the orchestrator.
5.  **Refactor `utils/notifications.py`:**
    *   **Action:** Modify `send_notification` to accept `smtp_pass` as an argument or fetch it directly using a passed-in `SecureStorage` instance, instead of relying on `config.get("HOSTINGER_SMTP_PASS")`. Update the call site in `Orchestrator.send_notification`.
6.  **Refactor Agents (e.g., `BrowsingAgent`, `EmailAgent`, `VoiceSalesAgent`):**
    *   **Action:** Audit agents for direct secret access via `os.getenv` or `self.config`.
    *   **Action:** Modify agents to receive necessary secrets (e.g., `SMARTPROXY_PASSWORD`, `DEEPGRAM_API_KEY`, IMAP/SMTP passwords) during initialization (passed from Orchestrator after fetching from Vault) or provide them access to the `SecureStorage` instance.
7.  **Testing:**
    *   Unit test `SecureStorage` enhancements (retries, error handling).
    *   Integration test key workflows (agent initialization, notifications, browsing tasks) ensuring secrets are correctly fetched from Vault and used.

## 5. Next Steps
- Implement enhancements in `utils/secure_storage.py` (retries, circuit breaker, error handling).
- Refactor `config/settings.py` to remove direct secret loading.
- Refactor `Orchestrator` and other components (`Notifications`, Agents) to fetch secrets via `SecureStorage`.
- Requires manual step: Populate secrets in HCP Vault.