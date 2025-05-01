# Filename: config/settings.py
# Description: Centralized Configuration for AI Agency (Postgres & Env Var Focused).
# Version: 3.0 (Genius Agentic Ready)

import os
import json
import logging
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any, Union, Type # Added Type

# Configure logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Load .env file (especially for local development)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env.local')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"Loaded environment variables from: {dotenv_path}")
else:
    logger.info(f"Info: .env.local file not found at {dotenv_path}. Relying on system environment variables.")

# --- Helper Function for Type Conversion ---
def _convert_env_var(value: Optional[str], target_type: Type) -> Any:
    """Attempts type conversion for environment variables."""
    if value is None:
        # For Optional types, return None. For others, maybe raise error or return default?
        # Returning None is generally safer for Optional fields.
        return None

    # Handle boolean specifically
    if target_type == bool:
        return value.lower() in ['true', '1', 't', 'y', 'yes']

    # Handle numeric types
    if target_type == int:
        try: return int(value)
        except (ValueError, TypeError): return None # Return None if conversion fails
    if target_type == float:
        try: return float(value)
        except (ValueError, TypeError): return None

    # Handle JSON lists/dicts
    if target_type in [list, dict]:
        try:
            loaded_json = json.loads(value)
            # Basic check if the loaded type matches the target type
            if isinstance(loaded_json, target_type):
                return loaded_json
            else:
                logger.warning(f"Env var JSON type mismatch for '{value[:50]}...'. Expected {target_type}, got {type(loaded_json)}. Returning raw string.")
                return value
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON for env var, returning raw string: {value[:50]}...")
            return value # Return raw string if JSON parse fails

    # Default return as string if no other type matches
    if target_type == str:
        return value

    # If target_type is something else (like Union), default to string or handle specifically
    return value

# --- Define Required Environment Variables (Secrets & Core Config) ---
# These MUST be set in the Coolify environment variables.
# Secrets (Examples - Add ALL required API keys, passwords etc.)
DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_ENCRYPTION_KEY = os.getenv("DATABASE_ENCRYPTION_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CLAY_API_KEY = os.getenv("CLAY_API_KEY") # CRITICAL for lead gen
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
HOSTINGER_SMTP_PASS = os.getenv("HOSTINGER_SMTP_PASS")
HOSTINGER_IMAP_PASS = os.getenv("HOSTINGER_IMAP_PASS")
SMARTPROXY_PASSWORD = os.getenv("SMARTPROXY_PASSWORD")
# Add other secret keys: SHODAN_API_KEY, SPIDERFOOT_API_KEY etc. if OSINT tools are kept/used

# Core Non-Secret Config
AGENCY_BASE_URL = os.getenv("AGENCY_BASE_URL", "http://localhost:5000") # Public URL of your app
HOSTINGER_EMAIL = os.getenv("HOSTINGER_EMAIL") # Your sending/IMAP email
HOSTINGER_SMTP = os.getenv("HOSTINGER_SMTP", "smtp.hostinger.com")
SMTP_PORT = _convert_env_var(os.getenv("SMTP_PORT", "587"), int)
HOSTINGER_IMAP_HOST = os.getenv("HOSTINGER_IMAP_HOST", "imap.hostinger.com")
HOSTINGER_IMAP_PORT = _convert_env_var(os.getenv("HOSTINGER_IMAP_PORT", "993"), int)
HOSTINGER_IMAP_USER = os.getenv("HOSTINGER_IMAP_USER", HOSTINGER_EMAIL)
TWILIO_VOICE_NUMBER = os.getenv("TWILIO_VOICE_NUMBER") # Your Twilio phone number for calls
USER_EMAIL = os.getenv("USER_EMAIL") # Your email for notifications

# --- Define Optional Environment Variables (with Defaults) ---
PROJECT_NAME = os.getenv("PROJECT_NAME", "Genius AI Sales System")
AGENCY_TRACKING_DOMAIN = os.getenv("AGENCY_TRACKING_DOMAIN") # Optional: Domain for tracking pixels
SENDER_NAME = os.getenv("SENDER_NAME", "Alex Reed")
SENDER_TITLE = os.getenv("SENDER_TITLE", "Growth Strategist")
USER_NAME = os.getenv("USER_NAME", "Agency Operator")
USER_WHATSAPP_NUMBER = os.getenv("USER_WHATSAPP_NUMBER") # Optional
DOWNLOAD_PASSWORD = os.getenv("DOWNLOAD_PASSWORD", "changeme123") # Change this!
SMARTPROXY_USERNAME = os.getenv("SMARTPROXY_USERNAME") # Optional if using proxies

# LLM Model Mapping (OpenRouter) - Using defaults, override via Env Vars
OPENROUTER_MODELS = {
    "think_general": os.getenv("THINK_GENERAL_MODEL", "google/gemini-1.5-pro-latest"),
    "think_strategize": os.getenv("THINK_STRATEGY_MODEL", "google/gemini-1.5-pro-latest"),
    "think_validate": os.getenv("THINK_VALIDATE_MODEL", "google/gemini-flash-1.5"),
    "think_critique": os.getenv("THINK_CRITIQUE_MODEL", "google/gemini-1.5-pro-latest"),
    "think_synthesize": os.getenv("THINK_SYNTHESIZE_MODEL", "google/gemini-1.5-pro-latest"),
    "think_radar": os.getenv("THINK_RADAR_MODEL", "google/gemini-flash-1.5"),
    "think_user_education": os.getenv("THINK_USER_EDU_MODEL", "google/gemini-flash-1.5"),
    "legal_analysis": os.getenv("LEGAL_ANALYSIS_MODEL", "google/gemini-1.5-pro-latest"),
    "legal_interpretation": os.getenv("LEGAL_INTERPRET_MODEL", "google/gemini-1.5-pro-latest"),
    "legal_validation": os.getenv("LEGAL_VALIDATE_MODEL", "google/gemini-flash-1.5"),
    "email_draft": os.getenv("EMAIL_DRAFT_MODEL", "google/gemini-flash-1.5"),
    "email_subject": os.getenv("EMAIL_SUBJECT_MODEL", "google/gemini-flash-1.5"),
    "email_humanize": os.getenv("EMAIL_HUMANIZE_MODEL", "google/gemini-1.5-pro-latest"),
    "email_spam_analysis": os.getenv("EMAIL_SPAM_MODEL", "google/gemini-flash-1.5"),
    "voice_intent": os.getenv("VOICE_INTENT_MODEL", "google/gemini-flash-1.5"),
    "voice_response": os.getenv("VOICE_RESPONSE_MODEL", "google/gemini-flash-1.5"),
    "voice_dynamic_price": os.getenv("VOICE_PRICE_MODEL", "google/gemini-flash-1.5"),
    "browsing_infer_steps": os.getenv("BROWSE_INFER_MODEL", "google/gemini-flash-1.5"),
    "browsing_extract_data": os.getenv("BROWSE_EXTRACT_MODEL", "google/gemini-flash-1.5"),
    "default_llm": os.getenv("DEFAULT_LLM", "google/gemini-flash-1.5"),
}

# Financials & Legal
PAYMENT_TERMS = os.getenv("PAYMENT_TERMS", "Payment due upon receipt.")
BASE_UGC_PRICE = _convert_env_var(os.getenv("BASE_UGC_PRICE", "5000.0"), float)
LEGAL_NOTE = os.getenv("LEGAL_NOTE", "Payment terms as agreed. Governed by laws of Morocco.")
MOROCCAN_BANK_ACCOUNT = os.getenv("MOROCCAN_BANK_ACCOUNT") # Store actual details in env vars
W8_NAME = os.getenv("W8_NAME")
W8_COUNTRY = os.getenv("W8_COUNTRY", "Morocco")
W8_ADDRESS = os.getenv("W8_ADDRESS")
W8_TIN = os.getenv("W8_TIN")

# Operational Parameters
EMAIL_AGENT_MAX_CONCURRENCY = _convert_env_var(os.getenv("EMAIL_AGENT_MAX_CONCURRENCY", "10"), int) # Reduced default
VOICE_AGENT_MAX_CONCURRENCY = _convert_env_var(os.getenv("VOICE_AGENT_MAX_CONCURRENCY", "5"), int) # Reduced default
BROWSING_AGENT_MAX_CONCURRENCY = _convert_env_var(os.getenv("BROWSING_AGENT_MAX_CONCURRENCY", "10"), int) # Reduced default
EMAIL_AGENT_MAX_PER_DAY = _convert_env_var(os.getenv("EMAIL_AGENT_MAX_PER_DAY", "500"), int) # Reduced default
SMTP_ACCOUNT_DAILY_LIMIT = _convert_env_var(os.getenv("SMTP_ACCOUNT_DAILY_LIMIT", "50"), int) # Reduced default
EMAIL_SPAM_THRESHOLD = _convert_env_var(os.getenv("EMAIL_SPAM_THRESHOLD", "0.8"), float)
EMAIL_MIN_SENDS_FOR_ANALYSIS = _convert_env_var(os.getenv("EMAIL_MIN_SENDS_FOR_ANALYSIS", "15"), int)
IMAP_CHECK_INTERVAL_S = _convert_env_var(os.getenv("IMAP_CHECK_INTERVAL_S", "300"), int)
THINKTOOL_SYNTHESIS_INTERVAL_SECONDS = _convert_env_var(os.getenv("THINKTOOL_SYNTHESIS_INTERVAL_SECONDS", "3600"), int) # 1 hour
THINKTOOL_RADAR_INTERVAL_SECONDS = _convert_env_var(os.getenv("THINKTOOL_RADAR_INTERVAL_SECONDS", "21600"), int) # 6 hours
DATA_PURGE_INTERVAL_SECONDS = _convert_env_var(os.getenv("DATA_PURGE_INTERVAL_SECONDS", "86400"), int) # 24 hours
DATA_PURGE_DAYS_THRESHOLD = _convert_env_var(os.getenv("DATA_PURGE_DAYS_THRESHOLD", "30"), int) # Added
KB_QUERY_DEFAULT_LIMIT = _convert_env_var(os.getenv("KB_QUERY_DEFAULT_LIMIT", "50"), int)
VOICE_INTENT_CONFIDENCE_THRESHOLD = _convert_env_var(os.getenv("VOICE_INTENT_CONFIDENCE_THRESHOLD", "0.6"), float)
DEEPGRAM_RECEIVE_TIMEOUT_S = _convert_env_var(os.getenv("DEEPGRAM_RECEIVE_TIMEOUT_S", "60.0"), float)
OPENROUTER_API_TIMEOUT_S = _convert_env_var(os.getenv("OPENROUTER_API_TIMEOUT_S", "120.0"), float) # General API timeout

# SMTP Providers List (Structure only, password fetched from env vars later)
SMTP_PROVIDERS = []
if HOSTINGER_EMAIL and HOSTINGER_SMTP:
    SMTP_PROVIDERS.append({
        "host": HOSTINGER_SMTP,
        "port": SMTP_PORT,
        "email": HOSTINGER_EMAIL,
        # "pass": fetched_at_runtime from env var HOSTINGER_SMTP_PASS
    })

# Meta Prompt (Central Directive - Keep as is or refine based on final agent structure)
META_PROMPT = """
You are the central intelligence and strategic coordinator of a highly autonomous, agentic AI system ("Genius AI Sales System"). Your singular mandate is **Maximize Profit and Achieve Exponential Growth** ($10k+/day target), starting with UGC but dynamically pivoting to higher ROI ventures identified through continuous data synthesis using Clay.com and other sources.
**Core Principles:** Autonomy & Initiative, Hyper-Efficiency, Relentless Learning & Meta-Prompting, AI-Native Advantage, Strategic Exploitation & Grey Area Analysis (within assessed risk), Adaptability & User Value, Security & Compliance (USA/Morocco focus, guided by LegalAgent).
**Operational Focus:** Maximize profitable outreach (Email/Voice) using Clay.com leads. Continuously analyze performance (KB/Logs) to adapt strategies, prompts, and resource allocation. Exploit free trials and multi-account tactics where risk is acceptable (guided by LegalAgent). Aim for "First-Try Deployment" quality.
"""

# --- Settings Access Class ---
class Settings:
    # Define attributes for type hinting and easy access
    # Only include non-secret or essential bootstrap config here
    DATABASE_URL: Optional[str] = DATABASE_URL
    DATABASE_ENCRYPTION_KEY: Optional[str] = DATABASE_ENCRYPTION_KEY # Needed for DB utils
    PROJECT_NAME: Optional[str] = PROJECT_NAME
    AGENCY_BASE_URL: Optional[str] = AGENCY_BASE_URL
    AGENCY_TRACKING_DOMAIN: Optional[str] = AGENCY_TRACKING_DOMAIN
    SENDER_NAME: Optional[str] = SENDER_NAME
    SENDER_TITLE: Optional[str] = SENDER_TITLE
    USER_NAME: Optional[str] = USER_NAME
    USER_EMAIL: Optional[str] = USER_EMAIL
    USER_WHATSAPP_NUMBER: Optional[str] = USER_WHATSAPP_NUMBER
    DOWNLOAD_PASSWORD: Optional[str] = DOWNLOAD_PASSWORD
    OPENROUTER_MODELS: Optional[Dict[str, str]] = OPENROUTER_MODELS
    HOSTINGER_EMAIL: Optional[str] = HOSTINGER_EMAIL
    HOSTINGER_SMTP: Optional[str] = HOSTINGER_SMTP
    SMTP_PORT: Optional[int] = SMTP_PORT
    HOSTINGER_IMAP_HOST: Optional[str] = HOSTINGER_IMAP_HOST
    HOSTINGER_IMAP_PORT: Optional[int] = HOSTINGER_IMAP_PORT
    HOSTINGER_IMAP_USER: Optional[str] = HOSTINGER_IMAP_USER
    TWILIO_ACCOUNT_SID: Optional[str] = TWILIO_ACCOUNT_SID # Needed for agent init check
    TWILIO_VOICE_NUMBER: Optional[str] = TWILIO_VOICE_NUMBER
    SMARTPROXY_USERNAME: Optional[str] = SMARTPROXY_USERNAME
    PAYMENT_TERMS: Optional[str] = PAYMENT_TERMS
    BASE_UGC_PRICE: Optional[float] = BASE_UGC_PRICE
    LEGAL_NOTE: Optional[str] = LEGAL_NOTE
    MOROCCAN_BANK_ACCOUNT: Optional[str] = MOROCCAN_BANK_ACCOUNT # Identifier, actual details in env
    W8_NAME: Optional[str] = W8_NAME
    W8_COUNTRY: Optional[str] = W8_COUNTRY
    W8_ADDRESS: Optional[str] = W8_ADDRESS
    W8_TIN: Optional[str] = W8_TIN
    EMAIL_AGENT_MAX_CONCURRENCY: Optional[int] = EMAIL_AGENT_MAX_CONCURRENCY
    VOICE_AGENT_MAX_CONCURRENCY: Optional[int] = VOICE_AGENT_MAX_CONCURRENCY
    BROWSING_AGENT_MAX_CONCURRENCY: Optional[int] = BROWSING_AGENT_MAX_CONCURRENCY
    EMAIL_AGENT_MAX_PER_DAY: Optional[int] = EMAIL_AGENT_MAX_PER_DAY
    SMTP_ACCOUNT_DAILY_LIMIT: Optional[int] = SMTP_ACCOUNT_DAILY_LIMIT
    EMAIL_SPAM_THRESHOLD: Optional[float] = EMAIL_SPAM_THRESHOLD
    EMAIL_MIN_SENDS_FOR_ANALYSIS: Optional[int] = EMAIL_MIN_SENDS_FOR_ANALYSIS
    IMAP_CHECK_INTERVAL_S: Optional[int] = IMAP_CHECK_INTERVAL_S
    THINKTOOL_SYNTHESIS_INTERVAL_SECONDS: Optional[int] = THINKTOOL_SYNTHESIS_INTERVAL_SECONDS
    THINKTOOL_RADAR_INTERVAL_SECONDS: Optional[int] = THINKTOOL_RADAR_INTERVAL_SECONDS
    DATA_PURGE_INTERVAL_SECONDS: Optional[int] = DATA_PURGE_INTERVAL_SECONDS
    DATA_PURGE_DAYS_THRESHOLD: Optional[int] = DATA_PURGE_DAYS_THRESHOLD
    KB_QUERY_DEFAULT_LIMIT: Optional[int] = KB_QUERY_DEFAULT_LIMIT
    VOICE_INTENT_CONFIDENCE_THRESHOLD: Optional[float] = VOICE_INTENT_CONFIDENCE_THRESHOLD
    DEEPGRAM_RECEIVE_TIMEOUT_S: Optional[float] = DEEPGRAM_RECEIVE_TIMEOUT_S
    OPENROUTER_API_TIMEOUT_S: Optional[float] = OPENROUTER_API_TIMEOUT_S
    SMTP_PROVIDERS: Optional[List[Dict[str, Any]]] = SMTP_PROVIDERS
    META_PROMPT: Optional[str] = META_PROMPT

    # Add other non-secret operational parameters here

    def __init__(self):
        # Load all attributes defined above from the module-level constants
        for name, type_hint in self.__annotations__.items():
            if name in globals():
                setattr(self, name, globals()[name])
            else:
                setattr(self, name, None) # Should not happen if defined above

    def get(self, name: str, default: Any = None) -> Any:
        """Provides dict-like access with a default value."""
        return getattr(self, name, default)

    def get_secret(self, secret_env_var_name: str) -> Optional[str]:
        """Helper to get secret from environment, centralizing the access pattern."""
        value = os.getenv(secret_env_var_name)
        if not value:
            logger.warning(f"Secret environment variable '{secret_env_var_name}' is not set.")
            return None
        return value

settings = Settings()

# --- Validation ---
def validate_core_settings():
    """Validates essential settings required for basic operation."""
    required_env_vars = [
        "DATABASE_URL", "DATABASE_ENCRYPTION_KEY",
        "OPENROUTER_API_KEY", "CLAY_API_KEY",
        "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_VOICE_NUMBER",
        "DEEPGRAM_API_KEY",
        "HOSTINGER_EMAIL", "HOSTINGER_SMTP_PASS", "HOSTINGER_IMAP_PASS",
        "USER_EMAIL",
        # Add other absolutely critical env vars here
    ]
    missing = [key for key in required_env_vars if not os.getenv(key)]
    if missing:
        msg = f"CRITICAL settings missing in environment variables: {', '.join(missing)}. Agency cannot function correctly. Check Coolify environment settings."
        logger.critical(msg)
        raise ValueError(msg) # Crash on startup if critical secrets/config missing
    else:
        logger.info("Core environment variables appear to be present.")

# Run validation immediately upon import
validate_core_settings()

# --- End of config/settings.py ---