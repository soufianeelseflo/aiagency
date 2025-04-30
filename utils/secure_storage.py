# Filename: utils/secure_storage.py
# Description: Securely interacts with HCP Vault Secrets API.
# Version: 2.1 (Production Ready - Enhanced Resilience & Error Handling)

import aiohttp
import os
import json
import logging
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
import pybreaker
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# --- Custom Exception ---
class VaultError(Exception):
    """Custom exception for HCP Vault interaction errors."""
    def __init__(self, message, status_code=None, details=None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details

# --- Resilience Configuration ---
vault_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60 * 3, name="VaultAPI") # 3 min timeout, 5 failures

# Define retry conditions for Vault API calls
def is_retryable_vault_exception(exception: BaseException) -> bool:
    """Return True if the exception is a retryable network or server error."""
    if isinstance(exception, (aiohttp.ClientConnectorError, asyncio.TimeoutError)):
        logger.debug(f"Retrying Vault operation due to network/timeout error: {type(exception).__name__}")
        return True
    if isinstance(exception, VaultError):
        # Retry on 5xx server errors or 429 rate limiting
        should_retry = exception.status_code is not None and (exception.status_code >= 500 or exception.status_code == 429)
        if should_retry:
            logger.debug(f"Retrying Vault operation due to VaultError status {exception.status_code}")
        return should_retry
    # Don't retry other exceptions by default (e.g., 404 Not Found, 401 Unauthorized, JSON errors)
    return False

class SecureStorage:
    """
    Provides an interface to securely store and retrieve secrets from HCP Vault.
    Includes retry logic and circuit breaking for resilience.
    """
    def __init__(self):
        self.base_url = "https://api.cloud.hashicorp.com"
        self.org_id = os.getenv("HCP_ORGANIZATION_ID")
        self.project_id = os.getenv("HCP_PROJECT_ID")
        self.app_name = os.getenv("HCP_APP_NAME")
        self.api_token = os.getenv("HCP_API_TOKEN")
        # API version hardcoded for stability, update if necessary
        self.secrets_path_template = "/secrets/2023-11-28/organizations/{}/projects/{}/apps/{}/secrets"
        self.secret_path_template = self.secrets_path_template + "/{}" # For GET/PUT/DELETE by name

        if not all([self.org_id, self.project_id, self.app_name, self.api_token]):
            logger.critical("HCP Vault configuration missing (ORG_ID, PROJECT_ID, APP_NAME, API_TOKEN). SecureStorage cannot operate.")
            # This is critical, raise error to prevent insecure operation
            raise ValueError("HCP Vault configuration incomplete. Cannot initialize SecureStorage.")
        logger.info("SecureStorage initialized with HCP Vault configuration.")

    def _get_headers(self) -> Dict[str, str]:
        """Returns standard headers for Vault API requests."""
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type(is_retryable_vault_exception))
    @vault_breaker
    async def get_secret(self, secret_name: str) -> Optional[str]:
        """
        Retrieve a secret's value from HCP Vault Secrets with resilience.
        Returns the secret value string or None if not found (after handling 404).
        Raises VaultError for other API or connection issues after retries.
        """
        if not secret_name: raise ValueError("secret_name cannot be empty")

        path = self.secret_path_template.format(self.org_id, self.project_id, self.app_name, secret_name)
        url = f"{self.base_url}{path}"
        headers = self._get_headers()
        logger.debug(f"Attempting to GET secret: {secret_name} from {url}")

        try:
            timeout = aiohttp.ClientTimeout(total=15) # 15 second total timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            if isinstance(data, dict) and "secret" in data and "value" in data["secret"]:
                                logger.info(f"Successfully retrieved secret: {secret_name}")
                                return data["secret"]["value"]
                            else:
                                logger.error(f"Vault GET response for {secret_name} missing expected structure: {str(data)[:200]}...")
                                raise VaultError(f"Invalid response structure for secret {secret_name}", status_code=response.status)
                        except aiohttp.ContentTypeError:
                            resp_text = await response.text()
                            logger.error(f"Failed to decode JSON response for {secret_name}. Status: {response.status}. Response: {resp_text[:200]}...")
                            raise VaultError(f"Invalid JSON response for secret {secret_name}", status_code=response.status, details=resp_text)
                    elif response.status == 404:
                        logger.warning(f"Secret '{secret_name}' not found in Vault (404). Returning None.")
                        return None # Explicitly return None for not found
                    else:
                        details = await response.text()
                        logger.error(f"Failed to retrieve secret {secret_name} (Status: {response.status}): {details[:500]}...")
                        # Raise VaultError to trigger retry logic if applicable (e.g., 5xx)
                        raise VaultError(f"Failed to retrieve secret {secret_name}", status_code=response.status, details=details)
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout retrieving secret {secret_name} from Vault.")
            raise VaultError(f"Timeout retrieving secret {secret_name}") from e
        except aiohttp.ClientError as e: # Catches ClientConnectorError and others
            logger.error(f"Network/Connection error retrieving secret {secret_name}: {e}")
            raise VaultError(f"Connection error retrieving secret {secret_name}", details=str(e)) from e
        except VaultError: raise # Re-raise specific VaultErrors
        except Exception as e: # Catch unexpected errors
            logger.exception(f"Unexpected error retrieving secret {secret_name}: {e}")
            raise VaultError(f"Unexpected error retrieving secret {secret_name}", details=str(e)) from e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15), retry=retry_if_exception_type(is_retryable_vault_exception))
    @vault_breaker
    async def set_secret(self, secret_name: str, value: str):
        """
        Set or update a secret in HCP Vault Secrets with resilience.
        Uses PUT for idempotency (create or update).
        Raises VaultError on failure after retries.
        """
        if not secret_name: raise ValueError("secret_name cannot be empty")
        if value is None: raise ValueError("secret value cannot be None") # Use delete_secret instead

        path = self.secret_path_template.format(self.org_id, self.project_id, self.app_name, secret_name)
        url = f"{self.base_url}{path}"
        headers = self._get_headers()
        # Structure for PUT request (check API docs if this changes)
        data = {"value": value}
        logger.debug(f"Attempting to PUT secret: {secret_name} to {url}")

        try:
            timeout = aiohttp.ClientTimeout(total=20) # Longer timeout for writes
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.put(url, headers=headers, json=data) as response:
                    # Check for success codes (200 OK for update, 201 Created - though PUT usually returns 200 or 204)
                    if response.status in [200, 201, 204]: # 204 No Content might also be possible
                        logger.info(f"Successfully stored/updated secret: {secret_name}")
                    else:
                        details = await response.text()
                        logger.error(f"Failed to store secret {secret_name} (Status: {response.status}): {details[:500]}...")
                        # Raise VaultError to trigger retry logic if applicable (e.g., 5xx)
                        raise VaultError(f"Failed to store secret {secret_name}", status_code=response.status, details=details)
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout storing secret {secret_name} in Vault.")
            raise VaultError(f"Timeout storing secret {secret_name}") from e
        except aiohttp.ClientError as e:
            logger.error(f"Network/Connection error storing secret {secret_name}: {e}")
            raise VaultError(f"Connection error storing secret {secret_name}", details=str(e)) from e
        except VaultError: raise # Re-raise specific VaultErrors
        except Exception as e: # Catch unexpected errors
            logger.exception(f"Unexpected error storing secret {secret_name}: {e}")
            raise VaultError(f"Unexpected error storing secret {secret_name}", details=str(e)) from e

    async def get_credentials(self, service: str) -> List[Dict[str, Any]]:
        """
        Retrieves and parses credentials stored as a JSON list for a service.
        Returns an empty list if the secret is not found or cannot be parsed.
        Raises VaultError for other API/connection issues.
        """
        secret_name = f"{service}_credentials" # Standardized naming convention
        try:
            creds_str = await self.get_secret(secret_name)
            if creds_str is None:
                # get_secret now returns None for 404
                logger.info(f"Credentials secret '{secret_name}' not found for service '{service}'. Returning empty list.")
                return []
            else:
                try:
                    creds_list = json.loads(creds_str)
                    if isinstance(creds_list, list):
                        return creds_list
                    else:
                        logger.error(f"Credentials for service '{service}' (secret: {secret_name}) is not a JSON list. Found type: {type(creds_list)}. Returning empty list.")
                        return []
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON credentials for service '{service}' from secret '{secret_name}'. Content: {creds_str[:100]}...")
                    # Consider raising an error here to signal data corruption
                    raise VaultError(f"Corrupted JSON credentials for service '{service}'")
        except VaultError as e:
            # Re-raise errors other than 'not found' which was handled above
            logger.error(f"Vault error getting credentials for service '{service}': {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in get_credentials for service '{service}': {e}")
            raise VaultError(f"Unexpected error getting credentials for '{service}'", details=str(e)) from e

    async def add_credential(self, service: str, credential: Dict[str, Any]):
        """
        Adds a credential object to the list stored for a service (read-modify-write).

        WARNING: This operation is NOT atomic. Concurrent calls to this method for the
                 same service can lead to data loss. Use with caution or implement
                 external locking if concurrency is expected.
        """
        if not isinstance(credential, dict):
            raise ValueError("Credential to add must be a dictionary.")

        secret_name = f"{service}_credentials"
        logger.warning(f"Attempting non-atomic read-modify-write for secret '{secret_name}'. Ensure no concurrent writes.")
        try:
            # Get current credentials, returns empty list if not found or handles other errors
            creds = await self.get_credentials(service)

            # Append the new credential
            creds.append(credential)

            # Set the updated list back into the vault
            await self.set_secret(secret_name, json.dumps(creds, indent=2)) # Add indent for readability
            logger.info(f"Successfully added credential for service '{service}'.")
        except VaultError as e:
            logger.error(f"Failed to add credential for service '{service}' due to Vault error: {e}")
            raise # Re-raise VaultError from set_secret or get_credentials
        except Exception as e:
            logger.exception(f"Unexpected error adding credential for service '{service}': {e}")
            # Wrap unexpected errors in VaultError for consistency
            raise VaultError(f"Unexpected error adding credential for '{service}'", details=str(e)) from e

# --- End of utils/secure_storage.py ---