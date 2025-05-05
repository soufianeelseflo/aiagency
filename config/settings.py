# Filename: config/settings.py
# Description: Configuration settings for the Synapse AI Sales System,
#              validated using Pydantic. Secrets loaded from environment variables.
# Version: 2.2 (Removed Unused SMTP Config)

import os
import json
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import (
    Field, PostgresDsn, HttpUrl, DirectoryPath, EmailStr, validator, AnyUrl
)
from typing import Dict, List, Optional, Any, Union

# Configure logger for settings loading issues
logger = logging.getLogger(__name__)

# --- Helper Functions ---
# ... (get_env_variable function remains the same) ...
def get_env_variable(var_name: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieves an environment variable, logging a warning if not found and no default is set."""
    value = os.environ.get(var_name)
    if value is None:
        if default is None:
            # Log only once per variable to avoid spamming logs during startup
            if not hasattr(get_env_variable, 'logged_warnings'):
                get_env_variable.logged_warnings = set() # type: ignore
            if var_name not in get_env_variable.logged_warnings: # type: ignore
                logger.warning(f"Environment variable '{var_name}' not set and no default provided.")
                get_env_variable.logged_warnings.add(var_name) # type: ignore
        return default
    return value

class Settings(BaseSettings):
    """
    Main configuration class using Pydantic BaseSettings.
    Reads from environment variables automatically (case-insensitive).
    """
    # --- Core Application Settings ---
    APP_NAME: str = Field(default="Synapse AI Sales System", description="Name of the application.")
    APP_VERSION: str = Field(default="3.2-Genius-Cleaned", description="Version of the application.") # Updated version
    DEBUG: bool = Field(default=False, description="Enable debug logging and potentially other debug features.")
    AGENCY_BASE_URL: AnyUrl = Field(..., description="Base URL where the agency is hosted (e.g., for webhooks, asset hosting). Must include scheme. Example: 'http://yourdomain.com:5000'") # Required, use AnyUrl for flexibility

    # --- Database Configuration ---
    DATABASE_URL: PostgresDsn = Field(..., description="Async PostgreSQL connection string. Load from env var 'DATABASE_URL'.")
    DATABASE_ENCRYPTION_KEY: str = Field(..., min_length=32, description="Master key for database field encryption (MUST be strong, >= 32 chars, and kept secret). Load from env var 'DATABASE_ENCRYPTION_KEY'.")

    # --- LLM / OpenRouter Configuration ---
    OPENROUTER_API_KEY: Optional[str] = Field(default=None, description="Primary OpenRouter API Key. Load from env var 'OPENROUTER_API_KEY'.")
    OPENROUTER_API_KEY_1: Optional[str] = Field(default=None, description="Additional OpenRouter API Key 1. Load from env var 'OPENROUTER_API_KEY_1'.")
    OPENROUTER_API_KEY_2: Optional[str] = Field(default=None, description="Additional OpenRouter API Key 2. Load from env var 'OPENROUTER_API_KEY_2'.")
    OPENROUTER_MODELS: Dict[str, str] = {
        # --- High Power (Keep Pro/Opus/GPT-4o) ---
        "think_synthesize": "anthropic/claude-3-opus-20240229",
        "think_strategize": "anthropic/claude-3-opus-20240229",
        "think_critique": "anthropic/claude-3-sonnet-20240229",
        "legal_analysis": "anthropic/claude-3-sonnet-20240229",
        "browsing_visual_analysis": "google/gemini-1.5-pro-latest",
        "email_draft": "anthropic/claude-3-sonnet-20240229",

        # --- Medium Power (Sonnet / Gemini Pro / GPT-4o-mini?) ---
        "think_radar": "anthropic/claude-3-sonnet-20240229",
        "browsing_extract": "anthropic/claude-3-sonnet-20240229",

        # --- Fast & Cheap (Haiku / Gemini Flash) ---
        "default_llm": "anthropic/claude-3-haiku-20240307",
        "think_validate": "anthropic/claude-3-haiku-20240307",
        "think_user_education": "anthropic/claude-3-haiku-20240307",
        "email_humanize": "anthropic/claude-3-haiku-20240307",
        "voice_intent": "anthropic/claude-3-haiku-20240307",
        "voice_response": "anthropic/claude-3-haiku-20240307",
        "browsing_summarize": "anthropic/claude-3-haiku-20240307",
    }
    OPENROUTER_API_TIMEOUT_S: float = Field(default=120.0, gt=0, description="Timeout in seconds for OpenRouter API calls.")

    # --- Email Agent Configuration ---
    HOSTINGER_EMAIL: EmailStr = Field(..., description="Your verified sending/receiving email address. Load from env var 'HOSTINGER_EMAIL'.")
    # HOSTINGER_SMTP_HOST: Optional[str] = Field(default=None, description="Hostinger SMTP server address (Fallback Only). Load from env var 'HOSTINGER_SMTP_HOST'.") # REMOVED/OPTIONAL
    # HOSTINGER_SMTP_PASS: Optional[str] = Field(default=None, description="Hostinger SMTP password (Fallback Only). Load from env var 'HOSTINGER_SMTP_PASS'.") # REMOVED/OPTIONAL
    HOSTINGER_IMAP_HOST: str = Field(default="imap.hostinger.com", description="Hostinger IMAP server address (For Replies). Load from env var 'HOSTINGER_IMAP_HOST'.")
    HOSTINGER_IMAP_PASS: str = Field(..., description="Hostinger IMAP password (For Replies). Load from env var 'HOSTINGER_IMAP_PASS'.") # Still required for replies
    HOSTINGER_IMAP_USER: Optional[EmailStr] = Field(default=None, description="Hostinger IMAP username (defaults to HOSTINGER_EMAIL if not set). Load from env var 'HOSTINGER_IMAP_USER'.")
    # SMTP_PORT: Optional[int] = Field(default=None, description="SMTP port (Fallback Only).") # REMOVED/OPTIONAL
    HOSTINGER_IMAP_PORT: int = Field(default=993, description="IMAP port (usually 993 for SSL).")
    SENDER_NAME: str = Field(default="Alex Reed", description="Name to use in the 'From' field of emails.")
    SENDER_TITLE: Optional[str] = Field(default="Growth Strategist", description="Optional title for sender in email signature.")
    SENDER_COMPANY_ADDRESS: str = Field(..., description="Physical company address required by CAN-SPAM. Load from env var 'SENDER_COMPANY_ADDRESS'.") # Required by CAN-SPAM
    EMAIL_AGENT_MAX_CONCURRENCY: int = Field(default=10, gt=0, description="Max concurrent email sending tasks.")
    EMAIL_AGENT_MAX_PER_DAY: int = Field(default=500, ge=0, description="Global daily email sending limit for the agent.")
    # SMTP_ACCOUNT_DAILY_LIMIT: int = Field(default=50, ge=0, description="Daily sending limit per individual SMTP account (Fallback Only).") # REMOVED/OPTIONAL
    IMAP_CHECK_INTERVAL_S: int = Field(default=300, ge=30, description="Interval in seconds to check IMAP for replies/opt-outs.")

    # --- Voice Agent Configuration ---
    # ... (Voice Agent settings remain the same) ...
    TWILIO_ACCOUNT_SID: str = Field(..., description="Twilio Account SID. Load from env var 'TWILIO_ACCOUNT_SID'.")
    TWILIO_AUTH_TOKEN: str = Field(..., description="Twilio Auth Token. Load from env var 'TWILIO_AUTH_TOKEN'.")
    TWILIO_VOICE_NUMBER: str = Field(..., description="Twilio phone number used for making calls (E.164 format recommended). Load from env var 'TWILIO_VOICE_NUMBER'.")
    DEEPGRAM_API_KEY: str = Field(..., description="Deepgram API Key. Load from env var 'DEEPGRAM_API_KEY'.")
    DEEPGRAM_AURA_VOICE: str = Field(default="aura-asteria-en", description="Deepgram Aura voice model ID for TTS.")
    DEEPGRAM_STT_MODEL: str = Field(default="nova-2-general", description="Deepgram model for STT.")
    VOICE_TARGET_COUNTRY: str = Field(default="US", description="Default target country code for voice operations.")
    PAYMENT_TERMS: str = Field(default="Payment due upon receipt.", description="Default payment terms mentioned in calls/invoices.")
    BASE_UGC_PRICE: float = Field(default=7000.0, ge=0, description="Base price for UGC package.") # Updated base price
    VOICE_INTENT_CONFIDENCE_THRESHOLD: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum confidence score to accept LLM intent classification.")
    DEEPGRAM_RECEIVE_TIMEOUT_S: float = Field(default=60.0, gt=0, description="Timeout for waiting for Deepgram transcriptions.")
    OPENROUTER_INTENT_TIMEOUT_S: float = Field(default=10.0, gt=0, description="Timeout for LLM intent classification calls.")
    OPENROUTER_RESPONSE_TIMEOUT_S: float = Field(default=15.0, gt=0, description="Timeout for LLM response generation calls.")


    # --- Browsing Agent Configuration ---
    # ... (Browsing Agent settings remain the same) ...
    SMARTPROXY_USER: Optional[str] = Field(default=None, description="Smartproxy username. Load from env var 'SMARTPROXY_USER'.")
    SMARTPROXY_PASSWORD: Optional[str] = Field(default=None, description="Smartproxy password. Load from env var 'SMARTPROXY_PASSWORD'.")
    SMARTPROXY_HOST: Optional[str] = Field(default=None, description="Smartproxy hostname (e.g., gate.smartproxy.com). Load from env var 'SMARTPROXY_HOST'.")
    SMARTPROXY_PORT: Optional[int] = Field(default=None, description="Smartproxy port (e.g., 7000). Load from env var 'SMARTPROXY_PORT'.")
    BROWSER_USER_AGENT: str = Field(default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36", description="User agent string for the browser.")
    BROWSER_MAX_CONCURRENT_PAGES: int = Field(default=5, gt=0, description="Maximum concurrent browser pages allowed.") # Renamed from CONCURRENT_CONTEXTS for clarity
    BROWSER_DEFAULT_TIMEOUT_MS: int = Field(default=60000, gt=0, description="Default navigation/action timeout for Playwright in milliseconds.")
    BROWSER_LEARNING_INTERVAL_S: int = Field(default=14400, ge=60, description="Interval (seconds) for BrowsingAgent learning cycle (e.g., proxy analysis).")
    SERVICE_GMAIL_SIGNUP_URL: Optional[str] = Field(default="https://accounts.google.com/signup", description="URL for Gmail signup page.")
    SERVICE_DESCRIPT_LOGIN_URL: Optional[str] = Field(default="https://web.descript.com/login", description="URL for Descript login.")


    # --- ThinkTool Configuration ---
    # ... (ThinkTool settings remain the same) ...
    THINKTOOL_SYNTHESIS_INTERVAL_SECONDS: int = Field(default=3600, ge=60, description="Interval for ThinkTool's main synthesis cycle.")
    THINKTOOL_RADAR_INTERVAL_SECONDS: int = Field(default=21600, ge=300, description="Interval for ThinkTool's technology radar cycle.")
    THINKTOOL_FEEDBACK_INTERVAL_SECONDS: int = Field(default=300, ge=60, description="Interval for Orchestrator to collect and send feedback to ThinkTool.")
    SCORING_WEIGHTS: Dict[str, float] = Field(default={"email_response": 1.0, "call_success": 2.5, "invoice_paid": 5.0})
    SCORING_DECAY_RATE_PER_DAY: float = Field(default=0.05, ge=0.0, le=1.0, description="Daily decay rate for engagement scores.")


    # --- Data Management ---
    # ... (Data Management settings remain the same) ...
    DATA_PURGE_DAYS_THRESHOLD: int = Field(default=90, ge=1, description="Age in days threshold for purging old Knowledge Fragments based on last access time.")
    DATA_PURGE_INTERVAL_SECONDS: int = Field(default=86400, ge=3600, description="Interval for running the data purge check.")
    LEARNING_MATERIALS_DIR: str = Field(default="learning_for_AI", description="Directory containing learning material files for ThinkTool.")


    # --- Clay.com API ---
    CLAY_API_KEY: Optional[str] = Field(default=None, description="API Key for Clay.com. Load from env var 'CLAY_API_KEY'.")

    # --- MailerSend / MailerCheck ---
    MAILERSEND_API_KEY: Optional[str] = Field(default=None, description="API Key for MailerSend. Load from env var 'MAILERSEND_API_KEY'.")
    MAILERCHECK_API_KEY: Optional[str] = Field(default=None, description="API Key for MailerCheck. Load from env var 'MAILERCHECK_API_KEY'.")

    # --- Operational ---
    # ... (Operational settings remain the same) ...
    META_PROMPT: str = Field(default="You are a component of the Synapse AI Sales System. Your goal is profit maximization.", description="Default meta prompt fallback.")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    LOG_FILE_PATH: Optional[str] = Field(default="/app/logs/synapse_agent.log", description="Path to the main log file. Set to empty string or None to disable file logging.")
    OPERATIONAL_LOG_FILE_PATH: Optional[str] = Field(default="/app/logs/operational.log", description="Path to the operational/human-readable log file. Set to empty or None to disable.")
    TEMP_AUDIO_DIR: str = Field(default="/app/temp_audio", description="Directory for temporary audio files (TTS). Needs write permissions.")
    USER_EMAIL: Optional[EmailStr] = Field(default=None, description="Operator's email for notifications. Load from env var 'USER_EMAIL'.")
    DOWNLOAD_PASSWORD: str = Field(default="changeme123", description="Password for downloading data via UI (change this!). Load from env var 'DOWNLOAD_PASSWORD'.")


    # --- Financials & Legal (Example - Load from Env Vars) ---
    # ... (Financials & Legal settings remain the same) ...
    LEGAL_NOTE: str = Field(default="Governed by laws of Morocco.", description="Default legal note. Load from env var 'LEGAL_NOTE'.")
    MOROCCAN_BANK_ACCOUNT: Optional[str] = Field(default=None, description="Identifier for Moroccan bank account. Load from env var 'MOROCCAN_BANK_ACCOUNT'.")
    W8_NAME: Optional[str] = Field(default=None, description="Name for W8 form. Load from env var 'W8_NAME'.")
    W8_COUNTRY: Optional[str] = Field(default="Morocco", description="Country for W8 form. Load from env var 'W8_COUNTRY'.")
    W8_ADDRESS: Optional[str] = Field(default=None, description="Address for W8 form. Load from env var 'W8_ADDRESS'.")
    W8_TIN: Optional[str] = Field(default=None, description="Tax ID Number for W8 form. Load from env var 'W8_TIN'.")


    # --- Pydantic Settings Configuration ---
    model_config = SettingsConfigDict(
        env_file='.env.local', # Load .env.local file if present
        env_file_encoding='utf-8',
        extra='ignore', # Ignore extra fields from env vars
        case_sensitive=False # Environment variables are case-insensitive
    )

    # --- Custom Validators ---
    @validator('*', pre=True, allow_reuse=True)
    def load_from_env_first(cls, v, field):
        """Ensure environment variable takes precedence over .env file or defaults if set."""
        # Use field.validation_alias if available (Pydantic v2), else field.alias
        alias = getattr(field, 'validation_alias', getattr(field, 'alias', field.name))
        env_val = os.environ.get(alias)
        # Also check uppercase version for compatibility
        if env_val is None:
             env_val = os.environ.get(alias.upper())
        return env_val if env_val is not None else v

    # MODIFIED: Removed HOSTINGER_SMTP_PASS from required check
    @validator('DATABASE_ENCRYPTION_KEY', 'OPENROUTER_API_KEY', 'HOSTINGER_IMAP_PASS', 'TWILIO_AUTH_TOKEN', 'DEEPGRAM_API_KEY', 'SENDER_COMPANY_ADDRESS', 'MAILERSEND_API_KEY', 'MAILERCHECK_API_KEY', pre=True, always=True)
    def check_required_secrets(cls, v, field):
        """Ensures critical secrets are loaded, potentially from environment."""
        alias = getattr(field, 'validation_alias', getattr(field, 'alias', field.name))
        # Allow certain secrets to be optional based on usage
        is_required = True
        # Make MailerSend/Check/Clay/Smartproxy optional at startup, log warning if missing
        if field.name in ['MAILERSEND_API_KEY', 'MAILERCHECK_API_KEY', 'CLAY_API_KEY', 'SMARTPROXY_PASSWORD']:
             if not v:
                 logger.warning(f"Optional secret '{field.name}' (env var '{alias.upper()}') is not set. Related features will be disabled.")
                 is_required = False # Treat as not strictly required for startup

        # Check strictly required secrets
        elif not v:
            # Only raise error if it's one of the truly essential secrets
            essential_secrets = ['DATABASE_ENCRYPTION_KEY', 'OPENROUTER_API_KEY', 'HOSTINGER_IMAP_PASS', 'TWILIO_AUTH_TOKEN', 'DEEPGRAM_API_KEY', 'SENDER_COMPANY_ADDRESS']
            if field.name in essential_secrets:
                 raise ValueError(f"CRITICAL: Required secret '{field.name}' (env var '{alias.upper()}') is not set.")
            else:
                 # This case shouldn't be hit often due to the optional check above, but acts as a safeguard
                 logger.warning(f"Secret '{field.name}' is missing but not deemed critical for startup.")


        if field.name == 'DATABASE_ENCRYPTION_KEY' and v and len(v) < 32:
             raise ValueError(f"CRITICAL: '{field.name}' must be at least 32 characters long.")
        return v

    @validator('DATABASE_URL', 'AGENCY_BASE_URL', 'HOSTINGER_EMAIL', 'TWILIO_ACCOUNT_SID', 'TWILIO_VOICE_NUMBER', pre=True, always=True)
    def check_required_config(cls, v, field):
        """Ensures non-secret but critical config is loaded."""
        alias = getattr(field, 'validation_alias', getattr(field, 'alias', field.name))
        if not v:
            raise ValueError(f"CRITICAL: Required setting '{field.name}' (env var '{alias.upper()}') is not set.")
        return v

    @validator('HOSTINGER_IMAP_USER', pre=True, always=True)
    def set_default_imap_user(cls, v, values):
        """Set default IMAP user to HOSTINGER_EMAIL if not provided."""
        # 'values' contains already validated/loaded fields
        return v or values.data.get('HOSTINGER_EMAIL') # Use values.data in Pydantic v2

    # --- Secret Management Helper ---
    def get_secret(self, secret_name: str) -> Optional[str]:
        """
        Safely retrieve a validated secret attribute by its field name.
        Returns the value if the secret exists and is validated, otherwise None.
        """
        # List of fields considered secrets (adjust as needed)
        # REMOVED HOSTINGER_SMTP_PASS
        secret_fields = {
            'DATABASE_ENCRYPTION_KEY', 'OPENROUTER_API_KEY',
            'HOSTINGER_IMAP_PASS', 'TWILIO_AUTH_TOKEN', 'DEEPGRAM_API_KEY',
            'SENDER_COMPANY_ADDRESS', 'CLAY_API_KEY', 'SMARTPROXY_PASSWORD',
            'MAILERSEND_API_KEY', 'MAILERCHECK_API_KEY', # Added Mailer keys
            'OPENROUTER_API_KEY_1', 'OPENROUTER_API_KEY_2' # Add other optional secret keys here
        }
        # Check if the attribute exists on the instance first
        if hasattr(self, secret_name):
            value = getattr(self, secret_name)
            # Return the value only if it's considered a secret and is not None/empty
            if secret_name in secret_fields and value:
                return value
            elif secret_name not in secret_fields:
                 logger.warning(f"Attempted to get non-secret attribute '{secret_name}' via get_secret.")
                 return value # Return non-secret value if it exists
            else:
                 # Secret field exists but has no value (was optional or validation failed softly)
                 return None
        else:
            logger.warning(f"Attempted to get non-existent attribute '{secret_name}' via get_secret.")
            return None # Return None if attribute doesn't exist

# --- Instantiate Settings ---
# This instance will be imported by other modules
try:
    settings = Settings()
    # Log some non-sensitive settings on startup
    logger.info(f"Settings loaded for App: {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Log Level: {settings.LOG_LEVEL}, Debug Mode: {settings.DEBUG}")
    # Log only host part of DB URL for security
    db_host = getattr(settings.DATABASE_URL, 'host', 'N/A') if settings.DATABASE_URL else 'N/A'
    logger.info(f"Database URL Host: {db_host}")
    logger.info(f"Base Agency URL: {settings.AGENCY_BASE_URL}")
except ValueError as e:
    logger.critical(f"CRITICAL ERROR: Failed to initialize settings due to validation errors: {e}", exc_info=False) # Don't log full traceback for validation errors
    # Exit or raise critical error if settings fail validation
    raise SystemExit(f"Settings validation failed: {e}")
except Exception as e:
    logger.critical(f"CRITICAL ERROR: Unexpected error during settings initialization: {e}", exc_info=True)
    raise SystemExit(f"Unexpected settings initialization error: {e}")

# --- End of config/settings.py ---