import aiohttp
import os
import json
import logging

logger = logging.getLogger(__name__)

class SecureStorage:
    def __init__(self):
        self.base_url = "https://api.cloud.hashicorp.com"
        self.org_id = os.getenv("HCP_ORGANIZATION_ID")
        self.project_id = os.getenv("HCP_PROJECT_ID")
        self.app_name = os.getenv("HCP_APP_NAME")
        self.api_token = os.getenv("HCP_API_TOKEN")
        self.path_template = "/secrets/2023-11-28/organizations/{}/projects/{}/apps/{}/secrets/{}"

    async def get_secret(self, secret_name):
        """Retrieve a secret from HCP Vault Secrets."""
        path = self.path_template.format(self.org_id, self.project_id, self.app_name, secret_name)
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["secret"]["value"]
                else:
                    logger.error(f"Failed to retrieve secret {secret_name}: {await response.text()}")
                    raise ValueError(f"Secret {secret_name} not found or inaccessible")

    async def set_secret(self, secret_name, value):
        """Set or update a secret in HCP Vault Secrets."""
        path = self.path_template.format(self.org_id, self.project_id, self.app_name, "")
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        data = {"name": secret_name, "value": value}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 201:
                    logger.info(f"Successfully stored secret {secret_name}")
                else:
                    logger.error(f"Failed to store secret {secret_name}: {await response.text()}")
                    raise ValueError(f"Failed to store secret {secret_name}")

    async def get_credentials(self, service):
        creds_str = await self.get_secret(f"{service}_credentials")
        return json.loads(creds_str) if creds_str else []

    async def add_credential(self, service, credential):
        creds = await self.get_credentials(service)
        creds.append(credential)
        await self.set_secret(f"{service}_credentials", json.dumps(creds))