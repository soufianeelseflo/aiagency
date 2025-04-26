import aiohttp
import os
import json
import logging
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pybreaker

logger = logging.getLogger(__name__)

# --- Custom Exception ---
class VaultError(Exception):
    """Custom exception for HCP Vault interaction errors."""
    def __init__(self, message, status_code=None, details=None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details

# --- Resilience Configuration ---
vault_breaker = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=60 * 2) # 2 min timeout

# Define retry conditions for Vault API calls
def is_retryable_vault_exception(exception):
    """Return True if the exception is a retryable network or server error."""
    if isinstance(exception, (aiohttp.ClientConnectorError, asyncio.TimeoutError)):
        return True
    if isinstance(exception, VaultError):
        # Retry on 5xx server errors or 429 rate limiting
        return exception.status_code is not None and (exception.status_code >= 500 or exception.status_code == 429)
    return False

class SecureStorage:
    def __init__(self):
        self.base_url = "https://api.cloud.hashicorp.com"
        self.org_id = os.getenv("HCP_ORGANIZATION_ID")
        self.project_id = os.getenv("HCP_PROJECT_ID")
        self.app_name = os.getenv("HCP_APP_NAME")
        self.api_token = os.getenv("HCP_API_TOKEN")
        self.path_template = "/secrets/2023-11-28/organizations/{}/projects/{}/apps/{}/secrets/{}" # API version hardcoded for now

        if not all([self.org_id, self.project_id, self.app_name, self.api_token]):
            logger.critical("HCP Vault configuration missing (ORG_ID, PROJECT_ID, APP_NAME, API_TOKEN). SecureStorage will fail.")
            # Consider raising an error here to prevent startup if Vault is essential
            # raise ValueError("HCP Vault configuration incomplete.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), retry=retry_if_exception_type(is_retryable_vault_exception))
    @vault_breaker
    async def get_secret(self, secret_name):
        """Retrieve a secret from HCP Vault Secrets with resilience."""
        if not all([self.org_id, self.project_id, self.app_name, self.api_token]):
            raise VaultError("HCP Vault configuration is incomplete.")

        path = self.path_template.format(self.org_id, self.project_id, self.app_name, secret_name)
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            # Use a shared session potentially? For now, create per request. Add timeout.
            timeout = aiohttp.ClientTimeout(total=15) # 15 second total timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            # Validate expected structure
                            if "secret" in data and "value" in data["secret"]:
                                return data["secret"]["value"]
                            else:
                                logger.error(f"Vault response for {secret_name} missing expected structure: {data}")
                                raise VaultError(f"Invalid response structure for secret {secret_name}", status_code=response.status)
                        except aiohttp.ContentTypeError as json_err:
                            logger.error(f"Failed to decode JSON response for {secret_name}: {json_err}")
                            raise VaultError(f"Invalid JSON response for secret {secret_name}", status_code=response.status, details=await response.text())
                    elif response.status == 404:
                        logger.warning(f"Secret '{secret_name}' not found in Vault (404).")
                        # Return None or raise specific error? Returning None might be safer for optional secrets.
                        # For now, let's raise a specific error for clarity.
                        raise VaultError(f"Secret '{secret_name}' not found", status_code=404)
                    else:
                        details = await response.text()
                        logger.error(f"Failed to retrieve secret {secret_name} (Status: {response.status}): {details}")
                        raise VaultError(f"Failed to retrieve secret {secret_name}", status_code=response.status, details=details)
        except asyncio.TimeoutError:
            logger.error(f"Timeout retrieving secret {secret_name} from Vault.")
            raise VaultError(f"Timeout retrieving secret {secret_name}")
        except aiohttp.ClientConnectorError as conn_err:
            logger.error(f"Connection error retrieving secret {secret_name}: {conn_err}")
            raise VaultError(f"Connection error retrieving secret {secret_name}", details=str(conn_err))
        except VaultError: # Re-raise specific VaultErrors
            raise
        except Exception as e: # Catch unexpected errors
            logger.exception(f"Unexpected error retrieving secret {secret_name}: {e}")
            raise VaultError(f"Unexpected error retrieving secret {secret_name}", details=str(e))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), retry=retry_if_exception_type(is_retryable_vault_exception))
    @vault_breaker
    async def set_secret(self, secret_name, value):
        """Set or update a secret in HCP Vault Secrets with resilience."""
        if not all([self.org_id, self.project_id, self.app_name, self.api_token]):
            raise VaultError("HCP Vault configuration is incomplete.")

        # Path for setting secrets might differ slightly (often targets the app, not a specific secret name in path)
        # Assuming the provided path template is correct for POST/PUT to create/update.
        # Check HCP Vault API docs for exact endpoint for create/update.
        # Let's assume POST to the app's secrets path with name in body is correct.
        path = self.path_template.format(self.org_id, self.project_id, self.app_name, secret_name) # Using name in path for PUT/PATCH style update
        # Alternative for POST create:
        # path = f"/secrets/2023-11-28/organizations/{self.org_id}/projects/{self.project_id}/apps/{self.app_name}/secrets"
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        # Structure for creating/updating might vary. Assuming 'value' field. Check API docs.
        data = {"value": value} # Assuming PUT/PATCH to update existing secret by name in path
        # Alternative for POST create:
        # data = {"name": secret_name, "value": value}

        try:
            timeout = aiohttp.ClientTimeout(total=20) # Longer timeout for writes
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Use PUT for create/update by name in path (idempotent)
                async with session.put(url, headers=headers, json=data) as response:
                # Alternative: Use POST to create if name not in path
                # async with session.post(url, headers=headers, json=data) as response:

                    # Check for success codes (e.g., 200 OK for update, 201 Created for new)
                    if response.status in [200, 201]:
                        logger.info(f"Successfully stored/updated secret {secret_name}")
                    else:
                        details = await response.text()
                        logger.error(f"Failed to store secret {secret_name} (Status: {response.status}): {details}")
                        raise VaultError(f"Failed to store secret {secret_name}", status_code=response.status, details=details)
        except asyncio.TimeoutError:
            logger.error(f"Timeout storing secret {secret_name} in Vault.")
            raise VaultError(f"Timeout storing secret {secret_name}")
        except aiohttp.ClientConnectorError as conn_err:
            logger.error(f"Connection error storing secret {secret_name}: {conn_err}")
            raise VaultError(f"Connection error storing secret {secret_name}", details=str(conn_err))
        except VaultError: # Re-raise specific VaultErrors
            raise
        except Exception as e: # Catch unexpected errors
            logger.exception(f"Unexpected error storing secret {secret_name}: {e}")
            raise VaultError(f"Unexpected error storing secret {secret_name}", details=str(e))


    async def get_credentials(self, service):
        """Retrieves and parses credentials stored as JSON for a service."""
        secret_name = f"{service}_credentials"
        try:
            creds_str = await self.get_secret(secret_name)
            if creds_str:
                try:
                    return json.loads(creds_str)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON credentials for service '{service}' from secret '{secret_name}'. Content: {creds_str[:100]}...")
                    # Raise error or return empty list? Raising might be better to signal corruption.
                    raise VaultError(f"Corrupted JSON credentials for service '{service}'")
            else:
                # If get_secret returned None (e.g., 404 handled gracefully, though current impl raises error)
                return []
        except VaultError as e:
            # If get_secret raised VaultError (e.g., 404 or other API error)
            if e.status_code == 404:
                logger.info(f"Credentials secret '{secret_name}' not found for service '{service}'. Returning empty list.")
                return []
            else:
                logger.error(f"Vault error getting credentials for service '{service}': {e}")
                raise # Re-raise other Vault errors

    async def add_credential(self, service, credential):
        """Adds a credential object to the list stored for a service (read-modify-write)."""
        # Warning: This read-modify-write pattern is not atomic.
        # Consider alternative approaches if concurrent writes are possible.
        secret_name = f"{service}_credentials"
        try:
            # Get current credentials, handling potential 404 or errors
            try:
                creds = await self.get_credentials(service)
                if not isinstance(creds, list):
                    logger.warning(f"Existing credentials for '{service}' is not a list. Overwriting with new list.")
                    creds = []
            except VaultError as e:
                # If secret doesn't exist (404 handled in get_credentials) or other error getting it
                if e.status_code == 404:
                    creds = [] # Start fresh if secret didn't exist
                else:
                    raise # Re-raise other errors encountered during get

            creds.append(credential)
            await self.set_secret(secret_name, json.dumps(creds, indent=2)) # Add indent for readability in Vault UI
            logger.info(f"Successfully added credential for service '{service}'.")
        except VaultError as e:
            logger.error(f"Failed to add credential for service '{service}' due to Vault error: {e}")
            raise # Re-raise VaultError from set_secret or get_credentials
        except Exception as e:
            logger.exception(f"Unexpected error adding credential for service '{service}': {e}")
            raise VaultError(f"Unexpected error adding credential for '{service}'", details=str(e))