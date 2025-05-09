# Filename: config/settings.py
# Description: Configuration settings for the Nolli AI Sales System,
#              validated using Pydantic. Secrets loaded from environment variables.
# Version: 2.6 (Clarified env loading, minor Pydantic V2 syntax consistency)

import os
import json # Not directly used for loading here, but often for defaults
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import (
    Field, PostgresDsn, HttpUrl, DirectoryPath, EmailStr, field_validator, model_validator, ValidationInfo, AnyUrl
)
from typing import Dict, List, Optional, Any, Union

# Configure logger for settings loading issues (will be configured by main.py first)
logger = logging.getLogger(__name__) # Standard logger

class Settings(BaseSettings):
    """
    Main configuration class using Pydantic BaseSettings.
    Reads from environment variables automatically (case-insensitive).
    Also supports .env.local for local development overrides.
    """
    # --- Core Application Settings ---
    APP_NAME: str = Field(default="Nolli AI Sales System", description="Name of the application.")
    APP_VERSION: str = Field(default="3.5-Genius-Hardened", description="Version of the application.") # Update version
    DEBUG: bool = Field(default=False, description="Enable debug logging and potentially other debug features.")
    AGENCY_BASE_URL: AnyUrl = Field(..., description="Base URL where the agency is hosted (e.g., for webhooks, asset hosting). Must include scheme. Example: 'https://agency.nichenova.store'")

    # --- Database Configuration ---
    DATABASE_URL: PostgresDsn = Field(..., description="Async PostgreSQL connection string. Load from env var 'DATABASE_URL'. Example: 'postgresql+asyncpg://user:pass@host:port/db'")
    DATABASE_ENCRYPTION_KEY: str = Field(..., min_length=32, description="Master key for database field encryption (MUST be strong, >= 32 chars, and kept secret). Load from env var 'DATABASE_ENCRYPTION_KEY'.")

    # --- LLM / OpenRouter Configuration ---
    OPENROUTER_API_KEY: Optional[str] = Field(default=None, description="Primary OpenRouter API Key. Load from env var 'OPENROUTER_API_KEY'.")
    OPENROUTER_MODELS: Dict[str, str] = {
        "think_synthesize": "google/gemini-pro-1.5", # Updated to a generally available strong model
        "think_strategize": "google/gemini-pro-1.5",
        "think_critique": "google/gemini-pro-1.5",
        "legal_analysis": "google/gemini-pro-1.5",
        "Browse_visual_analysis": "openai/gpt-4-vision-preview", # Explicitly visual
        "email_draft": "mistralai/mistral-large-latest", # Good for creative text
        "think_radar": "anthropic/claude-3-opus-20240229", # Strong reasoning for radar
        "Browse_extract": "mistralai/mistral-small-latest", # Efficient for extraction
        "default_llm": "mistralai/mistral-small-latest",
        "think_validate": "mistralai/mistral-small-latest",
        "think_user_education": "mistralai/mistral-small-latest",
        "email_humanize": "mistralai/mistral-small-latest",
        "voice_intent": "google/gemini-flash-1.5", # Fast for intent
        "voice_response": "google/gemini-flash-1.5", # Fast for response
        "Browse_summarize": "mistralai/mistral-small-latest",
    }
    OPENROUTER_API_TIMEOUT_S: float = Field(default=120.0, gt=0, description="Timeout in seconds for OpenRouter API calls.")

    # --- Email Agent Configuration ---
    HOSTINGER_EMAIL: EmailStr = Field(..., description="Your verified sending/receiving email address. Load from env var 'HOSTINGER_EMAIL'.")
    HOSTINGER_IMAP_HOST: str = Field(default="imap.hostinger.com", description="Hostinger IMAP server address (For Replies). Load from env var 'HOSTINGER_IMAP_HOST'.")
    HOSTINGER_IMAP_PASS: str = Field(..., description="Hostinger IMAP password (For Replies). Load from env var 'HOSTINGER_IMAP_PASS'.")
    HOSTINGER_IMAP_USER: Optional[EmailStr] = Field(default=None, description="Hostinger IMAP username (defaults to HOSTINGER_EMAIL if not set). Load from env var 'HOSTINGER_IMAP_USER'.")
    HOSTINGER_IMAP_PORT: int = Field(default=993, description="IMAP port (usually 993 for SSL).")
    SENDER_NAME: str = Field(default="Alex Reed", description="Name to use in the 'From' field of emails.")
    SENDER_TITLE: Optional[str] = Field(default="Growth Strategist", description="Optional title for sender in email signature.")
    SENDER_COMPANY_ADDRESS: str = Field(..., description="Physical company address required by CAN-SPAM. Load from env var 'SENDER_COMPANY_ADDRESS'.")
    EMAIL_AGENT_MAX_CONCURRENCY: int = Field(default=10, gt=0, description="Max concurrent email sending tasks.")
    EMAIL_AGENT_MAX_PER_DAY: int = Field(default=500, ge=0, description="Global daily email sending limit for the agent.")
    IMAP_CHECK_INTERVAL_S: int = Field(default=300, ge=30, description="Interval in seconds to check IMAP for replies/opt-outs.")

    # --- Voice Agent Configuration ---
    TWILIO_ACCOUNT_SID: str = Field(..., description="Twilio Account SID. Load from env var 'TWILIO_ACCOUNT_SID'.")
    TWILIO_AUTH_TOKEN: str = Field(..., description="Twilio Auth Token. Load from env var 'TWILIO_AUTH_TOKEN'.")
    TWILIO_VOICE_NUMBER: str = Field(..., description="Twilio phone number used for making calls (E.164 format recommended). Load from env var 'TWILIO_VOICE_NUMBER'.")
    DEEPGRAM_API_KEY: str = Field(..., description="Deepgram API Key. Load from env var 'DEEPGRAM_API_KEY'.")
    DEEPGRAM_AURA_VOICE: str = Field(default="aura-asteria-en", description="Deepgram Aura voice model ID for TTS.")
    DEEPGRAM_STT_MODEL: str = Field(default="nova-2-general", description="Deepgram model for STT.")
    VOICE_TARGET_COUNTRY: str = Field(default="US", description="Default target country code for voice operations.")
    PAYMENT_TERMS: str = Field(default="Payment due upon receipt.", description="Default payment terms mentioned in calls/invoices.")
    BASE_UGC_PRICE: float = Field(default=7000.0, ge=0, description="Base price for UGC package.")
    VOICE_INTENT_CONFIDENCE_THRESHOLD: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum confidence score to accept LLM intent classification.")
    DEEPGRAM_RECEIVE_TIMEOUT_S: float = Field(default=60.0, gt=0, description="Timeout for waiting for Deepgram transcriptions.")
    OPENROUTER_INTENT_TIMEOUT_S: float = Field(default=10.0, gt=0, description="Timeout for LLM intent classification calls.")
    OPENROUTER_RESPONSE_TIMEOUT_S: float = Field(default=15.0, gt=0, description="Timeout for LLM response generation calls.")

    # --- Browse Agent Configuration ---
    SMARTPROXY_USER: Optional[str] = Field(default=None, description="Smartproxy username. Load from env var 'SMARTPROXY_USER'.")
    SMARTPROXY_PASSWORD: Optional[str] = Field(default=None, description="Smartproxy password. Load from env var 'SMARTPROXY_PASSWORD'.")
    SMARTPROXY_HOST: Optional[str] = Field(default=None, description="Smartproxy hostname (e.g., gate.smartproxy.com). Load from env var 'SMARTPROXY_HOST'.")
    SMARTPROXY_PORT: Optional[int] = Field(default=None, description="Smartproxy port (e.g., 7000). Load from env var 'SMARTPROXY_PORT'.")
    BROWSER_USER_AGENT: str = Field(default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36", description="User agent string for the browser.") # Updated Chrome
    BROWSER_MAX_CONCURRENT_PAGES: int = Field(default=5, gt=0, description="Maximum concurrent browser pages allowed.")
    BROWSER_DEFAULT_TIMEOUT_MS: int = Field(default=60000, gt=0, description="Default navigation/action timeout for Playwright in milliseconds.")
    BROWSER_LEARNING_INTERVAL_S: int = Field(default=14400, ge=60, description="Interval (seconds) for BrowseAgent learning cycle.")
    SERVICE_GMAIL_SIGNUP_URL: Optional[str] = Field(default="https://accounts.google.com/signup", description="URL for Gmail signup page.")
    SERVICE_DESCRIPT_LOGIN_URL: Optional[str] = Field(default="https://web.descript.com/login", description="URL for Descript login.")
    TEMP_DOWNLOAD_DIR: str = Field(default="/app/temp_downloads", description="Directory for temporary file downloads by BrowseAgent.") # Added

    # --- ThinkTool Configuration ---
    THINKTOOL_SYNTHESIS_INTERVAL_SECONDS: int = Field(default=3600, ge=60, description="Interval for ThinkTool's main synthesis cycle.")
    THINKTOOL_RADAR_INTERVAL_SECONDS: int = Field(default=21600, ge=300, description="Interval for ThinkTool's technology radar cycle.")
    THINKTOOL_FEEDBACK_INTERVAL_SECONDS: int = Field(default=300, ge=60, description="Interval for Orchestrator to collect and send feedback to ThinkTool.")
    SCORING_WEIGHTS: Dict[str, float] = Field(default={"email_response": 1.0, "call_success": 2.5, "invoice_paid": 5.0})
    SCORING_DECAY_RATE_PER_DAY: float = Field(default=0.05, ge=0.0, le=1.0, description="Daily decay rate for engagement scores.")

    # --- Data Management ---
    DATA_PURGE_DAYS_THRESHOLD: int = Field(default=90, ge=1, description="Age in days threshold for purging old Knowledge Fragments based on last access time.")
    DATA_PURGE_INTERVAL_SECONDS: int = Field(default=86400, ge=3600, description="Interval for running the data purge check.")
    LEARNING_MATERIALS_DIR: str = Field(default="learning_for_AI", description="Directory containing learning material files for ThinkTool (relative to project root).")

    # --- Clay.com API ---
    CLAY_API_KEY: Optional[str] = Field(default=None, description="API Key for Clay.com. Load from env var 'CLAY_API_KEY'.")

    # --- MailerSend / MailerCheck ---
    MAILERSEND_API_KEY: Optional[str] = Field(default=None, description="API Key for MailerSend. Load from env var 'MAILERSEND_API_KEY'.")
    MAILERCHECK_API_KEY: Optional[str] = Field(default=None, description="API Key for MailerCheck. Load from env var 'MAILERCHECK_API_KEY'.")

    # --- Operational ---
    META_PROMPT: str = Field(default="You are a component of the Nolli AI Sales System. Your goal is profit maximization within ethical and legal boundaries defined by LegalAgent.", description="Default meta prompt fallback.") # Refined
    LOG_LEVEL: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    LOG_FILE_PATH: Optional[str] = Field(default=None, description="Path to the main log file. Set to empty string or None to disable file logging.")
    OPERATIONAL_LOG_FILE_PATH: Optional[str] = Field(default=None, description="Path to the operational/human-readable log file. Set to empty or None to disable.")
    TEMP_AUDIO_DIR: str = Field(default="/app/temp_audio", description="Directory for temporary audio files (TTS). Needs write permissions.")
    USER_EMAIL: Optional[EmailStr] = Field(default=None, description="Operator's email for notifications. Load from env var 'USER_EMAIL'.")
    DOWNLOAD_PASSWORD: str = Field(default="changeme123", min_length=12, description="Password for downloading data via UI (MUST CHANGE THIS!). Load from env var 'DOWNLOAD_PASSWORD'.")

    # --- Financials & Legal (Example - Load from Env Vars) ---
    LEGAL_NOTE: str = Field(default="Governed by laws of Morocco.", description="Default legal note. Load from env var 'LEGAL_NOTE'.")
    MOROCCAN_BANK_ACCOUNT: Optional[str] = Field(default=None, description="Identifier for Moroccan bank account (IBAN, SWIFT). Load from env var 'MOROCCAN_BANK_ACCOUNT'.")
    W8_NAME: Optional[str] = Field(default=None, description="Name for W8 form. Load from env var 'W8_NAME'.")
    W8_COUNTRY: Optional[str] = Field(default="Morocco", description="Country for W8 form. Load from env var 'W8_COUNTRY'.")
    W8_ADDRESS: Optional[str] = Field(default=None, description="Address for W8 form. Load from env var 'W8_ADDRESS'.")
    W8_TIN: Optional[str] = Field(default=None, description="Tax ID Number for W8 form. Load from env var 'W8_TIN'.")

    # --- Pydantic Settings Configuration ---
    # Pydantic V2 uses model_config instead of class Config
    model_config = SettingsConfigDict(
        env_file=('.env.local' if os.path.exists(os.path.join(os.path.dirname(__file__), '..', '.env.local')) else None), # Load .env.local if it exists at project root
        env_file_encoding='utf-8',
        extra='ignore',       # Ignore extra fields from env vars not defined in the model
        case_sensitive=False  # Environment variables are typically case-insensitive
    )

    # --- Custom Validators (Pydantic V2 Syntax) ---
    @field_validator(
        'DATABASE_ENCRYPTION_KEY', 'OPENROUTER_API_KEY', 'HOSTINGER_IMAP_PASS',
        'TWILIO_AUTH_TOKEN', 'DEEPGRAM_API_KEY', 'SENDER_COMPANY_ADDRESS',
        mode='before', check_fields=False # check_fields=False for Pydantic V2 group validators
    )
    @classmethod
    def check_essential_secrets(cls, v: Any, info: ValidationInfo) -> Any:
        """Ensures critical secrets are loaded."""
        field_name = info.field_name
        # Value 'v' here is the value attempting to be set (could be from env or default)
        # If it's from env, it's already loaded. If it's a default being applied because env var is missing,
        # Pydantic handles 'Field(..., description="...")' as required.
        # This validator primarily checks length for specific fields if they ARE provided.
        if field_name == 'DATABASE_ENCRYPTION_KEY' and v and len(str(v)) < 32:
            # This error will be caught by Pydantic if the field is required and not provided.
            # If it *is* provided but too short, this validator catches it.
            raise ValueError(f"CRITICAL: '{field_name}' must be at least 32 characters long.")
        # For other essential secrets, Pydantic's `Field(...)` without a default makes them required.
        # Pydantic will raise a `ValidationError` if they are missing from the environment and have no default.
        # We can add a warning here if an *optional* secret is missing.
        return v # Always return the value

    @field_validator(
        'MAILERSEND_API_KEY', 'MAILERCHECK_API_KEY', 'CLAY_API_KEY', 'SMARTPROXY_PASSWORD', 'SMARTPROXY_USER',
        mode='before', check_fields=False
    )
    @classmethod
    def check_optional_secrets(cls, v: Any, info: ValidationInfo) -> Any:
        """Logs warnings for optional secrets if not set."""
        field_name = info.field_name
        if not v: # If the value is None or empty (meaning not set in env and no default that's non-empty)
            logger.warning(f"Optional secret '{field_name}' (env var '{field_name.upper()}') is not set. Related features might be disabled or limited.")
        return v

    @field_validator(
        'DATABASE_URL', 'AGENCY_BASE_URL', 'HOSTINGER_EMAIL',
        'TWILIO_ACCOUNT_SID', 'TWILIO_VOICE_NUMBER',
        mode='before', check_fields=False
    )
    @classmethod
    def check_required_config(cls, v: Any, info: ValidationInfo) -> Any:
        """Ensures non-secret but critical config is loaded (Pydantic handles this with `Field(...)`)."""
        # Pydantic's default behavior for fields defined with `Field(...)` (no default)
        # already makes them required. This validator is more for explicit logging or custom checks if needed.
        # If `v` is None here, it means it wasn't found in env and has no default, Pydantic will raise error.
        if not v:
             # This situation should ideally be caught by Pydantic itself if no default is provided for '...' fields.
             # However, an explicit check can be useful for immediate logging.
             logger.critical(f"CRITICAL SETTING MISSING: '{info.field_name}' (env var '{info.field_name.upper()}') is not set and is required.")
             # Pydantic will raise the ValidationError, so we don't need to raise here.
        return v

    @model_validator(mode='after')
    def set_default_imap_user(self) -> 'Settings':
        """Set default IMAP user to HOSTINGER_EMAIL if not provided."""
        if self.HOSTINGER_IMAP_USER is None and self.HOSTINGER_EMAIL:
            self.HOSTINGER_IMAP_USER = self.HOSTINGER_EMAIL
            logger.debug(f"Defaulting HOSTINGER_IMAP_USER to HOSTINGER_EMAIL: {self.HOSTINGER_EMAIL}")
        return self

    # --- Secret Management Helper ---
    def get_secret(self, secret_name: str) -> Optional[str]:
        """
        Safely retrieve a validated secret attribute by its field name.
        Returns the value if the secret exists and is validated, otherwise None.
        """
        secret_fields_defined_in_model = {
            'DATABASE_ENCRYPTION_KEY', 'OPENROUTER_API_KEY',
            'HOSTINGER_IMAP_PASS', 'TWILIO_AUTH_TOKEN', 'DEEPGRAM_API_KEY',
            'SENDER_COMPANY_ADDRESS', # Not a secret, but handled by check_essential_secrets logic before
            'CLAY_API_KEY', 'SMARTPROXY_PASSWORD', 'SMARTPROXY_USER',
            'MAILERSEND_API_KEY', 'MAILERCHECK_API_KEY'
        }
        if hasattr(self, secret_name):
            value = getattr(self, secret_name)
            if value: # Ensure value is not None or empty
                return str(value) # Pydantic types like SecretStr will cast to str
            elif secret_name in secret_fields_defined_in_model:
                 # If it's a known secret field but has no value (None), it means it wasn't set
                 # and Pydantic should have caught it if it was truly required (no default).
                 # If it had a default=None and wasn't set, this is expected.
                 logger.debug(f"Secret attribute '{secret_name}' is present but has no value (None/empty).")
                 return None
        logger.warning(f"Attempted to get non-existent or non-set attribute '{secret_name}' via get_secret, or it's an optional secret that's not set.")
        return None

# --- Instantiate Settings ---
# This is where the Pydantic validation error (DATABASE_URL missing) originates if env vars aren't set.
try:
    settings = Settings()
    logger.info(f"Settings loaded successfully for App: {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Log Level set to: {settings.LOG_LEVEL}, Debug Mode: {settings.DEBUG}")
    db_host = getattr(settings.DATABASE_URL, 'host', 'N/A') if settings.DATABASE_URL else 'N/A'
    logger.info(f"Database URL Host (from settings object): {db_host}")
    logger.info(f"Base Agency URL (from settings object): {settings.AGENCY_BASE_URL}")
except Exception as e: # Catches Pydantic's ValidationError among others
    # This logger message might not appear if basicConfig in main.py hasn't run due to this error.
    # The print statements in main.py's except block are crucial for seeing this error during startup.
    logger.critical(f"CRITICAL ERROR in settings.py: Failed to initialize Settings object: {e}", exc_info=True)
    # Re-raise to ensure main.py catches it and exits, preventing the app from trying to run with invalid settings.
    # This also ensures the deployment system (Coolify) sees a failure.
    raise SystemExit(f"Settings initialization failed: {e}")

# --- End of config/settings.py ---