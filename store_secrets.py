import requests
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# HCP API configuration
BASE_URL = "https://api.cloud.hashicorp.com"
PATH_TEMPLATE = "/secrets/2023-11-28/organizations/{}/projects/{}/apps/{}/secrets"

ORG_ID = os.getenv("HCP_ORGANIZATION_ID")
PROJECT_ID = os.getenv("HCP_PROJECT_ID")
APP_NAME = os.getenv("HCP_APP_NAME")
API_TOKEN = os.getenv("HCP_API_TOKEN")

# Define secrets to store (optimized for the 25-secret limit)
secrets = {
    "bank_details": json.dumps({
        "account": os.getenv("MOROCCAN_BANK_ACCOUNT"),
        "swift": os.getenv("MOROCCAN_SWIFT_CODE"),
        "name": os.getenv("BANK_NAME", "Attijariwafa Bank"),
        "address": os.getenv("BANK_ADDRESS", "123 Avenue Mohammed V, Rabat, Morocco")
    }),
    "w8ben_data": json.dumps({
        "name": os.getenv("W8_NAME", "Your Name"),
        "country": os.getenv("W8_COUNTRY", "Morocco"),
        "address": os.getenv("W8_ADDRESS", "Your Address"),
        "tin": os.getenv("W8_TIN", "Your TIN")
    }),
    "openrouter_key": os.getenv("OPENROUTER_API_KEY"),
    "deepseek_key": os.getenv("DEEPSEEK_API_KEY"),
    "twilio_sid": os.getenv("TWILIO_ACCOUNT_SID"),
    "twilio_token": os.getenv("TWILIO_AUTH_TOKEN"),
    "hostinger_email": os.getenv("HOSTINGER_EMAIL"),
    "hostinger_smtp_pass": os.getenv("HOSTINGER_SMTP_PASS"),
    "smartproxy_user": os.getenv("SMARTPROXY_USERNAME"),
    "smartproxy_pass": os.getenv("SMARTPROXY_PASSWORD"),
    "user_email": os.getenv("USER_EMAIL"),
    "user_whatsapp": os.getenv("USER_WHATSAPP_NUMBER"),
    "twilio_whatsapp": os.getenv("TWILIO_WHATSAPP_NUMBER"),
    "inboxes_key": os.getenv("INBOXES_API_KEY")
}

def store_secret(secret_name, secret_value):
    """Store a secret in HCP Vault Secrets."""
    path = PATH_TEMPLATE.format(ORG_ID, PROJECT_ID, APP_NAME)
    url = f"{BASE_URL}{path}"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "name": secret_name,
        "value": secret_value
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print(f"Successfully stored secret: {secret_name}")
    else:
        print(f"Failed to store secret {secret_name}: {response.text}")

if __name__ == "__main__":
    for name, value in secrets.items():
        if value:  # Ensure no empty values are sent
            store_secret(name, value)
        else:
            print(f"Skipping {name}: value not provided in environment variables")