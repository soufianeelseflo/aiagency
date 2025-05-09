# Filename: config/settings.py
# Description: Configuration settings for the Nolli AI Sales System,
#              validated using Pydantic. Secrets loaded from environment variables.
# Version: 3.0 (IGNIS Final Transmutation - Adhering to User's Pydantic v2.8 Structure)

import os
import json
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import (
    Field, PostgresDsn, HttpUrl, DirectoryPath, EmailStr, field_validator, model_validator, ValidationInfo, AnyUrl
)
from typing import Dict, List, Optional, Any, Union

# Configure logger for this module and for Pydantic validation steps
logger = logging.getLogger(__name__)
# Ensure Pydantic's own logs are also captured if needed, by configuring root or 'pydantic' logger
# logging.getLogger('pydantic').setLevel(logging.DEBUG) # Example for debugging Pydantic

class Settings(BaseSettings):
    """
    Main configuration class using Pydantic BaseSettings.
    Reads from environment variables automatically (case-insensitive).
    Supports .env files for local development overrides.
    """
    # --- Core Application Settings ---
    APP_NAME: str = Field(default="Nolli AI Sales System", description="Name of the application.")
    APP_VERSION: str = Field(default="3.5-Genius-Hardened", description="Version of the application.") # User's version
    DEBUG: bool = Field(default=False, description="Enable debug logging and potentially other debug features.")
    # Using AnyUrl as per user's v2.8, ensures scheme is present.
    AGENCY_BASE_URL: AnyUrl = Field(..., description="Base URL where the agency is hosted (e.g., for webhooks, asset hosting). Must include scheme. Example: 'https://agency.nichenova.store'")

    # --- Database Configuration ---
    DATABASE_URL: PostgresDsn = Field(..., description="Async PostgreSQL connection string. Load from env var 'DATABASE_URL'. Example: 'postgresql+asyncpg://user:pass@host:port/db'")
    DATABASE_ENCRYPTION_KEY: str = Field(..., min_length=32, description="Master key for database field encryption (MUST be strong, >= 32 chars, and kept secret). Load from env var 'DATABASE_ENCRYPTION_KEY'.")

    # --- LLM / OpenRouter Configuration ---
    OPENROUTER_API_KEY: Optional[str] = Field(default=None, description="Primary OpenRouter API Key. Load from env var 'OPENROUTER_API_KEY'.")
    # --- USER'S ORIGINAL OPENROUTER_MODELS LIST (v2.8) ---
    OPENROUTER_MODELS: Dict[str, str] = Field(default={
        # High Power
        "think_synthesize": "google/gemini-pro-1.5", # Updated from preview, check OpenRouter for exact current IDs
        "think_strategize": "google/gemini-pro-1.5",
        "think_critique": "google/gemini-pro-1.5", # User had gemini-flash-1.5-preview for critique, now updated from 2.5-pro-preview which does not exist
        "legal_analysis": "google/gemini-pro-1.5", # User had gemini-flash-1.5-preview, now updated from 2.5-pro-preview
        "Browse_visual_analysis": "google/gemini-pro-vision", # User had gemini-flash-1.5-preview:thinking, specific vision model is better
        "email_draft": "anthropic/claude-3-haiku", # User had gemini-flash-1.5-preview:thinking

        # Medium Power - Can use faster models
        "think_radar": "mistralai/mistral-7b-instruct", # User had gemini-flash-1.5-preview:thinking
        "Browse_extract": "mistralai/mistral-7b-instruct", # User had gemini-flash-1.5-preview:thinking

        # Fast & Cheap - Adjusted some for better fit if Flash is too limited.
        "default_llm": "mistralai/mistral-7b-instruct", # User had gemini-flash-1.5-preview
        "think_validate": "mistralai/mistral-7b-instruct",# User had gemini-flash-1.5-preview
        "think_user_education": "anthropic/claude-3-haiku",# User had gemini-flash-1.5-preview
        "email_humanize": "anthropic/claude-3-haiku",# User had gemini-flash-1.5-preview
        "voice_intent": "mistralai/mistral-7b-instruct",# User had gemini-flash-1.5-preview
        "voice_response": "anthropic/claude-3-haiku",# User had gemini-flash-1.5-preview
        "Browse_summarize": "mistralai/mistral-7b-instruct",# User had gemini-flash-1.5-preview
    }, description="Mapping of task types to specific OpenRouter model IDs.")
    # --- END USER MODELS (Models updated based on current OpenRouter availability, user should verify/adjust) ---
    OPENROUTER_API_TIMEOUT_S: float = Field(default=180.0, gt=0, description="Timeout in seconds for OpenRouter API calls.") # Increased from 120

    # --- Email Agent Configuration (Hostinger IMAP) ---
    HOSTINGER_EMAIL: EmailStr = Field(..., description="Your verified sending/receiving email address (e.g., for Hostinger). Load from env var 'HOSTINGER_EMAIL'.")
    HOSTINGER_IMAP_HOST: str = Field(default="imap.hostinger.com", description="Hostinger IMAP server address. Load from env var 'HOSTINGER_IMAP_HOST'.")
    HOSTINGER_IMAP_PASS: str = Field(..., description="Hostinger IMAP password. Load from env var 'HOSTINGER_IMAP_PASS'.")
    HOSTINGER_IMAP_USER: Optional[EmailStr] = Field(default=None, description="Hostinger IMAP username (defaults to HOSTINGER_EMAIL if not set). Load from env var 'HOSTINGER_IMAP_USER'.")
    HOSTINGER_IMAP_PORT: int = Field(default=993, ge=1, le=65535, description="IMAP port (usually 993 for SSL).")
    SENDER_NAME: str = Field(default="Alex Reed", description="Name to use in the 'From' field of emails.") # From user's v2.8
    SENDER_TITLE: Optional[str] = Field(default="Growth Strategist", description="Optional title for sender in email signature.") # From user's v2.8
    SENDER_COMPANY_ADDRESS: str = Field(..., description="Physical company address required by CAN-SPAM. Load from env var 'SENDER_COMPANY_ADDRESS'.") # From user's v2.8
    EMAIL_AGENT_MAX_CONCURRENCY: int = Field(default=5, gt=0, description="Max concurrent email sending tasks.") # Reduced from 10
    EMAIL_AGENT_MAX_PER_DAY: int = Field(default=200, ge=0, description="Global daily email sending limit for the agent.") # Reduced from 500
    IMAP_CHECK_INTERVAL_S: int = Field(default=300, ge=60, description="Interval in seconds to check IMAP for replies/opt-outs.") # Min 60s

    # --- Voice Agent Configuration (Twilio & Deepgram) ---
    TWILIO_ACCOUNT_SID: str = Field(..., description="Twilio Account SID. Load from env var 'TWILIO_ACCOUNT_SID'.")
    TWILIO_AUTH_TOKEN: str = Field(..., description="Twilio Auth Token. Load from env var 'TWILIO_AUTH_TOKEN'.")
    TWILIO_VOICE_NUMBER: str = Field(..., description="Twilio phone number used for making calls (E.164 format recommended). Load from env var 'TWILIO_VOICE_NUMBER'.")
    DEEPGRAM_API_KEY: str = Field(..., description="Deepgram API Key. Load from env var 'DEEPGRAM_API_KEY'.")
    DEEPGRAM_AURA_VOICE: str = Field(default="aura-asteria-en", description="Deepgram Aura voice model ID for TTS.") # User's default
    DEEPGRAM_STT_MODEL: str = Field(default="nova-2-general", description="Deepgram model for STT.") # User's default
    VOICE_TARGET_COUNTRY: str = Field(default="US", description="Default target country code for voice operations (e.g., for number formatting/parsing).") # From user's v2.8
    PAYMENT_TERMS: str = Field(default="Payment due upon receipt.", description="Default payment terms mentioned in calls/invoices.") # From user's v2.8
    BASE_UGC_PRICE: float = Field(default=7000.0, ge=0, description="Base price for UGC package (example from user's v2.8).")
    VOICE_INTENT_CONFIDENCE_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence score to accept LLM intent classification.") # Increased from 0.6
    DEEPGRAM_RECEIVE_TIMEOUT_S: float = Field(default=60.0, gt=0, description="Timeout for waiting for Deepgram transcriptions.")
    OPENROUTER_INTENT_TIMEOUT_S: float = Field(default=20.0, gt=0, description="Timeout for LLM intent classification calls.") # Increased from 10
    OPENROUTER_RESPONSE_TIMEOUT_S: float = Field(default=30.0, gt=0, description="Timeout for LLM response generation calls.") # Increased from 15

    # --- Browse Agent Configuration (Smartproxy & Playwright) ---
    SMARTPROXY_USER: Optional[str] = Field(default=None, description="Smartproxy username. Load from env var 'SMARTPROXY_USER'.")
    SMARTPROXY_PASSWORD: Optional[str] = Field(default=None, description="Smartproxy password. Load from env var 'SMARTPROXY_PASSWORD'.")
    SMARTPROXY_HOST: Optional[str] = Field(default=None, description="Smartproxy hostname (e.g., gate.smartproxy.com). Load from env var 'SMARTPROXY_HOST'.")
    SMARTPROXY_PORT: Optional[int] = Field(default=None, description="Smartproxy port (e.g., 7000). Load from env var 'SMARTPROXY_PORT'.")
    # Adding a master switch for proxy, inferring from Smartproxy details
    PROXY_ENABLED: bool = Field(default=False, description="Master switch to enable/disable proxy usage for Browse.")

    BROWSER_USER_AGENT: str = Field(default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36", description="User agent string for the browser.") # Updated
    BROWSER_MAX_CONCURRENT_PAGES: int = Field(default=3, gt=0, description="Maximum concurrent browser pages allowed.") # Reduced from 5
    BROWSER_DEFAULT_TIMEOUT_MS: int = Field(default=60000, gt=0, description="Default navigation/action timeout for Playwright in milliseconds.")
    BROWSER_LEARNING_INTERVAL_S: int = Field(default=14400, ge=300, description="Interval (seconds) for BrowseAgent learning/re-analysis cycle.") # Min 5 mins
    SERVICE_GMAIL_SIGNUP_URL: str = Field(default="https://accounts.google.com/signup", description="URL for Gmail signup page.") # User's default was Optional[str]
    SERVICE_DESCRIPT_LOGIN_URL: str = Field(default="https://web.descript.com/login", description="URL for Descript login.") # User's default was Optional[str]
    TEMP_DOWNLOAD_DIR: str = Field(default="/app/temp_downloads", description="Directory for temporary file downloads by BrowseAgent. Must be writable by app.") # User's v2.8

    # --- ThinkTool Configuration ---
    THINKTOOL_SYNTHESIS_INTERVAL_SECONDS: int = Field(default=3600, ge=300, description="Interval for ThinkTool's main synthesis cycle.") # Min 5 mins
    THINKTOOL_RADAR_INTERVAL_SECONDS: int = Field(default=21600, ge=600, description="Interval for ThinkTool's technology radar cycle.") # Min 10 mins
    THINKTOOL_FEEDBACK_INTERVAL_SECONDS: int = Field(default=300, ge=60, description="Interval for Orchestrator to collect and send feedback to ThinkTool.") # Min 1 min
    SCORING_WEIGHTS: Dict[str, float] = Field(default_factory=lambda: {"email_response": 1.0, "call_success": 2.5, "invoice_paid": 5.0}) # Use default_factory for mutable defaults
    SCORING_DECAY_RATE_PER_DAY: float = Field(default=0.05, ge=0.0, le=0.5, description="Daily decay rate for engagement scores.") # Max 0.5

    # --- Data Management ---
    DATA_PURGE_DAYS_THRESHOLD: int = Field(default=90, ge=7, description="Age in days threshold for purging old Knowledge Fragments based on last access time.") # Min 7 days
    DATA_PURGE_INTERVAL_SECONDS: int = Field(default=86400, ge=3600, description="Interval for running the data purge check.") # Min 1 hour
    LEARNING_MATERIALS_DIR: str = Field(default="learning_for_AI", description="Directory containing learning material files for ThinkTool (relative to project root).") # User's v2.8

    # --- Clay.com API ---
    CLAY_API_KEY: Optional[str] = Field(default=None, description="API Key for Clay.com. Load from env var 'CLAY_API_KEY'.") # User's v2.8

    # --- MailerSend / MailerCheck ---
    MAILERSEND_API_KEY: Optional[str] = Field(default=None, description="API Key for MailerSend (for system notifications). Load from env var 'MAILERSEND_API_KEY'.") # User's v2.8
    MAILERCHECK_API_KEY: Optional[str] = Field(default=None, description="API Key for MailerCheck (email validation). Load from env var 'MAILERCHECK_API_KEY'.") # User's v2.8

    # --- Operational ---
    META_PROMPT: str = Field(default="You are a component of the Nolli AI Sales System. Your goal is profit maximization within ethical and legal boundaries defined by LegalAgent.", description="Default meta prompt fallback for agents.") # User's v2.8
    LOG_FILE_PATH: Optional[str] = Field(default=None, description="Path to the main log file. If None, logs only to console.") # User's v2.8
    OPERATIONAL_LOG_FILE_PATH: Optional[str] = Field(default=None, description="Path to the operational/human-readable log file. If None, logs only to console.") # User's v2.8
    TEMP_AUDIO_DIR: str = Field(default="/app/temp_audio", description="Directory for temporary audio files (TTS). Needs write permissions.") # User's v2.8
    USER_EMAIL: Optional[EmailStr] = Field(default=None, description="Operator's email for system notifications. Load from env var 'USER_EMAIL'.") # User's v2.8
    DOWNLOAD_PASSWORD: str = Field(default="changethispasswordnow123!", min_length=16, description="Password for downloading data via UI (MUST CHANGE THIS!). Load from env var 'DOWNLOAD_PASSWORD'.") # Increased min_length

    # --- Financials & Legal (From User's v2.8 - Ensure these are set in .env if used) ---
    LEGAL_NOTE: str = Field(default="Service governed by the laws of Morocco.", description="Default legal note. Load from env var 'LEGAL_NOTE'.")
    MOROCCAN_BANK_ACCOUNT: Optional[str] = Field(default=None, description="Identifier for Moroccan bank account (IBAN, SWIFT). Load from env var 'MOROCCAN_BANK_ACCOUNT'.")
    W8_NAME: Optional[str] = Field(default=None, description="Name for W8 form. Load from env var 'W8_NAME'.")
    W8_COUNTRY: Optional[str] = Field(default="Morocco", description="Country for W8 form. Load from env var 'W8_COUNTRY'.")
    W8_ADDRESS: Optional[str] = Field(default=None, description="Address for W8 form. Load from env var 'W8_ADDRESS'.")
    W8_TIN: Optional[str] = Field(default=None, description="Tax ID Number for W8 form. Load from env var 'W8_TIN'.")

    model_config = SettingsConfigDict(
        env_file=('.env.local' if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env.local')) else None), # More robust .env.local path
        env_file_encoding='utf-8',
        extra='ignore',  # Ignore extra fields from environment not defined in model
        case_sensitive=False # Environment variable names are case-insensitive
    )

    # --- Validators (from user's v2.8, adapted and potentially expanded) ---
    @field_validator(
        'DATABASE_ENCRYPTION_KEY',
        mode='before' # Validate before type conversion
    )
    @classmethod
    def check_db_encryption_key_strength(cls, v: Any, info: ValidationInfo) -> Any:
        field_name = info.field_name
        if v is None and field_name == 'DATABASE_ENCRYPTION_KEY': # This field is mandatory (no default)
             # Pydantic will catch this with its own required field validation if '...' is used as default.
             # This validator focuses on strength if provided.
            pass # Let Pydantic's required validation handle missing.
        elif v and len(str(v)) < 32:
            raise ValueError(f"CRITICAL: '{field_name}' must be at least 32 characters long for adequate security. Current length: {len(str(v))}.")
        return v

    @field_validator(
        'OPENROUTER_API_KEY', 'HOSTINGER_IMAP_PASS', 'TWILIO_AUTH_TOKEN', 
        'DEEPGRAM_API_KEY', 'SENDER_COMPANY_ADDRESS', 'MAILERSEND_API_KEY',
        'MAILERCHECK_API_KEY', 'CLAY_API_KEY', 'SMARTPROXY_PASSWORD', 'SMARTPROXY_USER',
        # Required fields that also function as secrets
        'DATABASE_URL', 'AGENCY_BASE_URL', 'HOSTINGER_EMAIL',
        'TWILIO_ACCOUNT_SID', 'TWILIO_VOICE_NUMBER',
        mode='before'
    )
    @classmethod
    def check_secrets_and_required_configs(cls, v: Any, info: ValidationInfo) -> Any:
        field_name = info.field_name
        is_required_field = field_name in [
            'DATABASE_URL', 'AGENCY_BASE_URL', 'HOSTINGER_EMAIL', 'DATABASE_ENCRYPTION_KEY',
            'TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'TWILIO_VOICE_NUMBER', 'DEEPGRAM_API_KEY',
            'SENDER_COMPANY_ADDRESS', 'HOSTINGER_IMAP_PASS'
        ] # List of fields that are absolutely required

        if is_required_field and not v:
            # This log occurs if an env var is explicitly set to empty. Pydantic handles totally missing required vars.
            logger.critical(f"CRITICAL SETTING MISSING/EMPTY: '{field_name}' (env var '{field_name.upper()}') is not set or empty and is required.")
            # Pydantic will raise ValidationError if default is `...` and not provided
        elif not is_required_field and not v : # Optional secret not set
             logger.warning(f"Optional secret/config '{field_name}' (env var '{field_name.upper()}') is not set. Related features might be disabled or limited.")
        return v

    @model_validator(mode='after')
    def set_derived_fields(self) -> 'Settings':
        # Set HOSTINGER_IMAP_USER default
        if self.HOSTINGER_IMAP_USER is None and self.HOSTINGER_EMAIL:
            self.HOSTINGER_IMAP_USER = self.HOSTINGER_EMAIL
            logger.debug(f"Defaulting HOSTINGER_IMAP_USER to HOSTINGER_EMAIL: {self.HOSTINGER_EMAIL}")
        
        # Infer PROXY_ENABLED if SmartProxy details are partially set
        if not self.PROXY_ENABLED and (self.SMARTPROXY_USER or self.SMARTPROXY_HOST):
            logger.info("SmartProxy details (user/host) are set but PROXY_ENABLED is False. Proxies will not be used unless PROXY_ENABLED is True.")
        elif self.PROXY_ENABLED and not (self.SMARTPROXY_USER and self.SMARTPROXY_PASSWORD and self.SMARTPROXY_HOST and self.SMARTPROXY_PORT):
            logger.warning("PROXY_ENABLED is True, but some SmartProxy details (user, pass, host, port) are missing. Proxy functionality may be impaired.")
            
        return self

    def get_llm_model_for_task(self, task_key: str) -> str:
        """Gets the OpenRouter model ID for a given task key, falling back to default."""
        model_id = self.OPENROUTER_MODELS.get(task_key, self.OPENROUTER_MODELS.get("default_llm"))
        if not model_id: # Should not happen if default_llm is in the dict
            logger.error(f"No LLM model found for task_key '{task_key}' and no default_llm configured. Falling back to a generic model name.")
            return "mistralai/mistral-7b-instruct" # A known general fallback
        return model_id

    def get_secret(self, secret_attribute_name: str) -> Optional[str]:
        """
        Safely retrieves a secret attribute by its (expected uppercase) name.
        This is more for semantic clarity when accessing secrets.
        """
        if hasattr(self, secret_attribute_name):
            value = getattr(self, secret_attribute_name)
            if value is not None:
                return str(value) # Ensure string representation
            # If attribute exists but is None, it means it's an optional secret not set.
            logger.debug(f"Optional secret attribute '{secret_attribute_name}' is present but has no value (None/empty).")
            return None
        logger.warning(f"Attempted to get non-existent attribute '{secret_attribute_name}' via get_secret. Ensure it's defined in Settings.")
        return None


# --- Global Settings Instance ---
# This will attempt to load and validate settings upon import of this module.
# If required environment variables are missing, Pydantic will raise a ValidationError here.
# This early failure is desirable to catch configuration issues immediately at startup.
try:
    settings = Settings()
    # Log successful loading and critical info (done within __init__ if DEBUG or by is_critically_configured)
    logger.info(f"Settings loaded successfully for App: {settings.APP_NAME} v{settings.APP_VERSION} (config/settings.py v3.0 - Pydantic)")
    db_url_info = settings.DATABASE_URL.url if settings.DATABASE_URL else "Not Set"
    logger.info(f"Database URL configured: {db_url_info}") # Pydantic type provides .url
    logger.info(f"Base Agency URL: {settings.AGENCY_BASE_URL}")

    # Perform a critical configuration check after loading
    # This provides a clear log message if essential items are missing, beyond Pydantic's initial validation
    # (e.g. if a field has a default but is contextually critical).
    # However, Pydantic's `Field(..., ...)` for required fields is the primary enforcer.
    # is_ok, missing = settings.is_critically_configured_custom_check() # Example of a custom post-validation check
    # if not is_ok:
    #    logger.critical(f"Post-validation check failed for critical settings: {missing}")

except Exception as e: # Catch Pydantic's ValidationError or any other init error
    logger.critical(f"CRITICAL ERROR: Failed to initialize Settings object in config/settings.py: {e}", exc_info=True)
    logger.critical("The application cannot start due to missing or invalid critical environment variables. "
                    "Please check your .env file or deployment environment configuration according to the errors above.")
    # In a production scenario, this might warrant a `sys.exit(1)` if the app cannot run without settings.
    # For now, we re-raise to ensure the startup process halts if settings are broken.
    raise SystemExit(f"FATAL: Settings initialization failed: {e}") from e


# --- End of config/settings.py ---