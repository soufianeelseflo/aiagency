# Filename: config/settings.py
# Description: Centralized Configuration for AI Agency.
# Version: 3.0 (Genius Agentic - Env Var Driven, No Vault)

import os
import json
import logging
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any, Union, Type # Added Type

# Configure logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Load .env file (especially for local development)
# ### Phase 1 Plan Ref: 1.1 (Define env vars) - Loading happens here
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
        return None # Keep None as None

    # Handle boolean first
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
        # Only attempt parse if it looks like JSON
        if isinstance(value, str) and value.strip().startswith(('[', '{')) and value.strip().endswith((']', '}')):
            try: return json.loads(value)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON for env var expecting {target_type}, returning raw string: {value[:50]}...")
                return value # Return raw string if JSON parse fails
        else:
             # If it doesn't look like JSON but list/dict expected, return default empty or handle error
             logger.warning(f"Env var value '{value[:50]}...' does not look like JSON, but {target_type} expected. Returning empty {target_type}.")
             return [] if target_type == list else {}

    # Default return as string if no other type matches
    return value

# --- Configuration Values (Loaded from Env Vars or Defaults) ---
# ### Phase 1 Plan Ref: 1.1 (Define env vars) - All variables defined here

# Core Infrastructure
DATABASE_URL = os.getenv("DATABASE_URL") # e.g., "postgresql+asyncpg://user:pass@host:port/dbname"
DATABASE_ENCRYPTION_KEY = os.getenv("DATABASE_ENCRYPTION_KEY") # MUST be strong

# Agency / User Info
PROJECT_NAME = os.getenv("PROJECT_NAME", "Genius AI Sales System")
AGENCY_BASE_URL = os.getenv("AGENCY_BASE_URL", "http://localhost:5000") # Public URL of deployed app
AGENCY_TRACKING_DOMAIN = os.getenv("AGENCY_TRACKING_DOMAIN") # Optional: Domain for tracking pixels
SENDER_NAME = os.getenv("SENDER_NAME", "Alex Reed")
SENDER_TITLE = os.getenv("SENDER_TITLE", "Growth Strategist")
USER_NAME = os.getenv("USER_NAME", "Agency Operator")
USER_EMAIL = os.getenv("USER_EMAIL") # For notifications
# USER_WHATSAPP_NUMBER = os.getenv("USER_WHATSAPP_NUMBER") # Optional: Format +1xxxxxxxxxx
DOWNLOAD_PASSWORD = os.getenv("DOWNLOAD_PASSWORD", "pleas€changeme!") # Password for UI data download

# LLM API Keys (Read directly from env vars)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") # Add if DeepSeek is used
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Add if Google AI Studio/Vertex is used directly

# LLM Model Mapping (OpenRouter)
DEFAULT_OPENROUTER_MODELS = {
    "think_general": "google/gemini-1.5-pro-latest",
    "think_strategize": "google/gemini-1.5-pro-latest",
    "think_validate": "google/gemini-flash-1.5",
    "think_critique": "google/gemini-1.5-pro-latest",
    "think_synthesize": "google/gemini-1.5-pro-latest",
    "think_radar": "google/gemini-flash-1.5",
    "think_user_education": "google/gemini-flash-1.5",
    "legal_analysis": "google/gemini-1.5-pro-latest",
    "legal_interpretation": "google/gemini-1.5-pro-latest",
    "legal_validation": "google/gemini-flash-1.5",
    "osint_analyze": "google/gemini-1.5-pro-latest",
    "email_draft": "google/gemini-flash-1.5",
    "email_subject": "google/gemini-flash-1.5",
    "email_humanize": "google/gemini-1.5-pro-latest",
    "email_spam_analysis": "google/gemini-flash-1.5",
    "voice_intent": "google/gemini-flash-1.5",
    "voice_response": "google/gemini-flash-1.5",
    "voice_dynamic_price": "google/gemini-flash-1.5",
    "programmer_planning": "google/gemini-1.5-pro-latest",
    "programmer_diff_gen": "google/gemini-1.5-pro-latest",
    "programmer_verification": "google/gemini-flash-1.5",
    "browsing_infer_steps": "google/gemini-flash-1.5",
    "browsing_extract_data": "google/gemini-flash-1.5",
    "social_content_gen": "google/gemini-flash-1.5",
    "social_plan_gen": "google/gemini-1.5-pro-latest",
    "default_llm": "google/gemini-flash-1.5",
}
OPENROUTER_MODELS = _convert_env_var(os.getenv("OPENROUTER_MODELS"), dict) or DEFAULT_OPENROUTER_MODELS

# External Service Credentials (Read directly from env vars)
# SMTP (Hostinger Example)
HOSTINGER_EMAIL = os.getenv("HOSTINGER_EMAIL")
HOSTINGER_SMTP_PASS = os.getenv("HOSTINGER_SMTP_PASS")
HOSTINGER_SMTP = os.getenv("HOSTINGER_SMTP", "smtp.hostinger.com")
SMTP_PORT = _convert_env_var(os.getenv("SMTP_PORT", "587"), int)

# IMAP (Hostinger Example)
HOSTINGER_IMAP_HOST = os.getenv("HOSTINGER_IMAP_HOST", "imap.hostinger.com")
HOSTINGER_IMAP_PORT = _convert_env_var(os.getenv("HOSTINGER_IMAP_PORT", "993"), int)
HOSTINGER_IMAP_USER = os.getenv("HOSTINGER_IMAP_USER", HOSTINGER_EMAIL)
HOSTINGER_IMAP_PASS = os.getenv("HOSTINGER_IMAP_PASS")

# Twilio
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
# TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER") # Optional
TWILIO_VOICE_NUMBER = os.getenv("TWILIO_VOICE_NUMBER") # Required for VoiceSalesAgent

# Deepgram
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Smartproxy (or other proxy)
SMARTPROXY_USERNAME = os.getenv("SMARTPROXY_USERNAME")
SMARTPROXY_PASSWORD = os.getenv("SMARTPROXY_PASSWORD")
# Add SMARTPROXY_ENDPOINT if needed, e.g., "gate.smartproxy.com:7000"

# Clay.com
CLAY_API_KEY = os.getenv("CLAY_API_KEY")

# OSINT Tools (Optional Keys)
SHODAN_API_KEY = os.getenv("SHODAN_API_KEY")
SPIDERFOOT_API_KEY = os.getenv("SPIDERFOOT_API_KEY")

# Financials & Legal
PAYMENT_TERMS = os.getenv("PAYMENT_TERMS", "Standard terms: 50% upfront, 50% upon completion.")
BASE_UGC_PRICE = _convert_env_var(os.getenv("BASE_UGC_PRICE", "5000.0"), float)
LEGAL_NOTE = os.getenv("LEGAL_NOTE", "Payment terms as agreed. All sales final upon service commencement. No refunds. Data managed securely. Agency not liable for indirect damages. Governed by laws of Morocco.")
# W8 / Bank Details (Read directly from env vars)
W8_NAME = os.getenv("W8_NAME")
W8_COUNTRY = os.getenv("W8_COUNTRY")
W8_ADDRESS = os.getenv("W8_ADDRESS")
W8_TIN = os.getenv("W8_TIN")
MOROCCAN_BANK_ACCOUNT = os.getenv("MOROCCAN_BANK_ACCOUNT")
# Add other bank details if needed (SWIFT, Bank Name, Address)
# LEMON_SQUEEZY_PAYMENT_URL = os.getenv("LEMON_SQUEEZY_PAYMENT_URL") # Optional

# Operational Parameters (with defaults)
BASE_CONCURRENCY = _convert_env_var(os.getenv("BASE_CONCURRENCY", "5"), int)
EMAIL_AGENT_MAX_CONCURRENCY = _convert_env_var(os.getenv("EMAIL_AGENT_MAX_CONCURRENCY", "25"), int)
VOICE_AGENT_MAX_CONCURRENCY = _convert_env_var(os.getenv("VOICE_AGENT_MAX_CONCURRENCY", "10"), int)
BROWSING_AGENT_MAX_CONCURRENCY = _convert_env_var(os.getenv("BROWSING_AGENT_MAX_CONCURRENCY", "15"), int)
OPTIMIZATION_MAX_CONCURRENCY_PER_AGENT = _convert_env_var(os.getenv("OPTIMIZATION_MAX_CONCURRENCY_PER_AGENT", "50"), int) # For RL agent if kept

# Email Agent Specific
EMAIL_AGENT_MAX_PER_DAY = _convert_env_var(os.getenv("EMAIL_AGENT_MAX_PER_DAY", "1000"), int)
SMTP_ACCOUNT_DAILY_LIMIT = _convert_env_var(os.getenv("SMTP_ACCOUNT_DAILY_LIMIT", "100"), int)
EMAIL_SPAM_THRESHOLD = _convert_env_var(os.getenv("EMAIL_SPAM_THRESHOLD", "0.8"), float)
EMAIL_MIN_SENDS_FOR_ANALYSIS = _convert_env_var(os.getenv("EMAIL_MIN_SENDS_FOR_ANALYSIS", "20"), int)
IMAP_CHECK_INTERVAL_S = _convert_env_var(os.getenv("IMAP_CHECK_INTERVAL_S", "300"), int)

# ThinkTool Specific
THINKTOOL_SYNTHESIS_INTERVAL_SECONDS = _convert_env_var(os.getenv("THINKTOOL_SYNTHESIS_INTERVAL_SECONDS", "3600"), int)
THINKTOOL_RADAR_INTERVAL_SECONDS = _convert_env_var(os.getenv("THINKTOOL_RADAR_INTERVAL_SECONDS", "21600"), int)
DATA_PURGE_INTERVAL_SECONDS = _convert_env_var(os.getenv("DATA_PURGE_INTERVAL_SECONDS", "86400"), int)
KB_QUERY_DEFAULT_LIMIT = _convert_env_var(os.getenv("KB_QUERY_DEFAULT_LIMIT", "50"), int)

# OSINT Agent Specific (if logic kept in ThinkTool/Browsing)
OSINT_ENABLE_GOOGLE_DORKING = _convert_env_var(os.getenv("OSINT_ENABLE_GOOGLE_DORKING", "True"), bool)
OSINT_ENABLE_SOCIAL_SCRAPING = _convert_env_var(os.getenv("OSINT_ENABLE_SOCIAL_SCRAPING", "True"), bool)
OSINT_ENABLE_LEAK_MONITORING = _convert_env_var(os.getenv("OSINT_ENABLE_LEAK_MONITORING", "True"), bool)
SPIDERFOOT_URL = os.getenv("SPIDERFOOT_URL", "http://localhost:5001") # If using Spiderfoot tool
SHODAN_RESULT_LIMIT = _convert_env_var(os.getenv("SHODAN_RESULT_LIMIT", "100"), int)
RECONNG_MODULE = os.getenv("RECONNG_MODULE", "recon/domains-hosts/hackertarget")

# Legal Agent Specific
DEFAULT_LEGAL_SOURCES = {"USA": ["https://www.federalregister.gov"], "Morocco": ["https://www.sgg.gov.ma"]}
LEGAL_SOURCES = _convert_env_var(os.getenv("LEGAL_SOURCES"), dict) or DEFAULT_LEGAL_SOURCES
LEGAL_UPDATE_INTERVAL_SECONDS = _convert_env_var(os.getenv("LEGAL_UPDATE_INTERVAL_SECONDS", "604800"), int) # 1 week

# Voice Agent Specific
VOICE_TARGET_COUNTRY = os.getenv("VOICE_TARGET_COUNTRY", "US")
DEEPGRAM_AURA_VOICE = os.getenv("DEEPGRAM_AURA_VOICE", "aura-asteria-en")
DEEPGRAM_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-2-general")
VOICE_INTENT_CONFIDENCE_THRESHOLD = _convert_env_var(os.getenv("VOICE_INTENT_CONFIDENCE_THRESHOLD", "0.6"), float)
DEEPGRAM_RECEIVE_TIMEOUT_S = _convert_env_var(os.getenv("DEEPGRAM_RECEIVE_TIMEOUT_S", "60.0"), float)
OPENROUTER_INTENT_TIMEOUT_S = _convert_env_var(os.getenv("OPENROUTER_INTENT_TIMEOUT_S", "10.0"), float)
OPENROUTER_RESPONSE_TIMEOUT_S = _convert_env_var(os.getenv("OPENROUTER_RESPONSE_TIMEOUT_S", "15.0"), float)

# Scoring Logic Specific (if integrated into ThinkTool)
DEFAULT_SCORING_WEIGHTS = {"email_response": 1.0, "call_success": 2.5, "invoice_paid": 5.0}
SCORING_WEIGHTS = _convert_env_var(os.getenv("SCORING_WEIGHTS"), dict) or DEFAULT_SCORING_WEIGHTS
SCORING_DECAY_RATE_PER_DAY = _convert_env_var(os.getenv("SCORING_DECAY_RATE_PER_DAY", "0.05"), float)

# Budget Logic Specific (if integrated into Orchestrator)
DEFAULT_BUDGETS = {"LLM": "100.00", "API": "50.00", "Resource": "50.00", "Proxy": "20.00", "Default": "10.00"}
BUDGETS = _convert_env_var(os.getenv("BUDGETS"), dict) or DEFAULT_BUDGETS

# SMTP Providers List (Structure only, passwords fetched from env vars)
SMTP_PROVIDERS = []
if HOSTINGER_EMAIL and HOSTINGER_SMTP:
    SMTP_PROVIDERS.append({
        "host": HOSTINGER_SMTP,
        "port": SMTP_PORT,
        "email": HOSTINGER_EMAIL,
        # "pass": HOSTINGER_SMTP_PASS # Password read directly from env var when needed
    })
# Add logic here to load more providers from environment if needed

# Meta Prompt (Central Directive) - Keep as defined previously or load from env
META_PROMPT = os.getenv("META_PROMPT", """
You are the central intelligence and strategic coordinator... [Your Full Meta Prompt Here]
""")


# --- Settings Access Class ---
class Settings:
    # Define attributes for type hinting and easy access
    # Only include non-secret attributes here that are loaded above
    DATABASE_URL: Optional[str]
    DATABASE_ENCRYPTION_KEY: Optional[str]
    PROJECT_NAME: Optional[str]
    AGENCY_BASE_URL: Optional[str]
    AGENCY_TRACKING_DOMAIN: Optional[str]
    SENDER_NAME: Optional[str]
    SENDER_TITLE: Optional[str]
    USER_NAME: Optional[str]
    USER_EMAIL: Optional[str]
    # USER_WHATSAPP_NUMBER: Optional[str]
    DOWNLOAD_PASSWORD: Optional[str]
    OPENROUTER_MODELS: Optional[Dict[str, str]]
    HOSTINGER_EMAIL: Optional[str]
    HOSTINGER_SMTP: Optional[str]
    SMTP_PORT: Optional[int]
    HOSTINGER_IMAP_HOST: Optional[str]
    HOSTINGER_IMAP_PORT: Optional[int]
    HOSTINGER_IMAP_USER: Optional[str]
    TWILIO_ACCOUNT_SID: Optional[str]
    # TWILIO_WHATSAPP_NUMBER: Optional[str]
    TWILIO_VOICE_NUMBER: Optional[str]
    SMARTPROXY_USERNAME: Optional[str]
    PAYMENT_TERMS: Optional[str]
    BASE_UGC_PRICE: Optional[float]
    LEGAL_NOTE: Optional[str]
    MOROCCAN_BANK_ACCOUNT: Optional[str]
    # LEMON_SQUEEZY_PAYMENT_URL: Optional[str]
    BASE_CONCURRENCY: Optional[int]
    EMAIL_AGENT_MAX_CONCURRENCY: Optional[int]
    VOICE_AGENT_MAX_CONCURRENCY: Optional[int]
    BROWSING_AGENT_MAX_CONCURRENCY: Optional[int]
    OPTIMIZATION_MAX_CONCURRENCY_PER_AGENT: Optional[int]
    EMAIL_AGENT_MAX_PER_DAY: Optional[int]
    SMTP_ACCOUNT_DAILY_LIMIT: Optional[int]
    EMAIL_SPAM_THRESHOLD: Optional[float]
    EMAIL_MIN_SENDS_FOR_ANALYSIS: Optional[int]
    IMAP_CHECK_INTERVAL_S: Optional[int]
    THINKTOOL_SYNTHESIS_INTERVAL_SECONDS: Optional[int]
    THINKTOOL_RADAR_INTERVAL_SECONDS: Optional[int]
    DATA_PURGE_INTERVAL_SECONDS: Optional[int]
    KB_QUERY_DEFAULT_LIMIT: Optional[int]
    OSINT_ENABLE_GOOGLE_DORKING: Optional[bool]
    OSINT_ENABLE_SOCIAL_SCRAPING: Optional[bool]
    OSINT_ENABLE_LEAK_MONITORING: Optional[bool]
    SPIDERFOOT_URL: Optional[str]
    SPIDERFOOT_MODULES: Optional[List[str]]
    SHODAN_RESULT_LIMIT: Optional[int]
    RECONNG_MODULE: Optional[str]
    LEGAL_SOURCES: Optional[Dict[str, List[str]]]
    LEGAL_UPDATE_INTERVAL_SECONDS: Optional[int]
    # Add other non-secret operational params...
    VOICE_TARGET_COUNTRY: Optional[str]
    DEEPGRAM_AURA_VOICE: Optional[str]
    DEEPGRAM_STT_MODEL: Optional[str]
    VOICE_INTENT_CONFIDENCE_THRESHOLD: Optional[float]
    DEEPGRAM_RECEIVE_TIMEOUT_S: Optional[float]
    OPENROUTER_INTENT_TIMEOUT_S: Optional[float]
    OPENROUTER_RESPONSE_TIMEOUT_S: Optional[float]
    SCORING_WEIGHTS: Optional[Dict[str, float]]
    SCORING_DECAY_RATE_PER_DAY: Optional[float]
    SCORING_LEARNING_INTERVAL_S: Optional[int]
    SCORING_CRITIQUE_INTERVAL_S: Optional[int]
    BUDGETS: Optional[Dict[str, str]]
    SMTP_PROVIDERS: Optional[List[Dict[str, Any]]]
    META_PROMPT: Optional[str]
    W8_NAME: Optional[str]
    W8_COUNTRY: Optional[str]
    W8_ADDRESS: Optional[str]
    W8_TIN: Optional[str]

    # Secrets that are loaded from env but NOT exposed via this class instance
    # They should be accessed directly via os.getenv() when needed (e.g., in Orchestrator init)
    _secrets = {
        "OPENROUTER_API_KEY", "CLAY_API_KEY", "TWILIO_AUTH_TOKEN", "DEEPGRAM_API_KEY",
        "HOSTINGER_SMTP_PASS", "HOSTINGER_IMAP_PASS", "SMARTPROXY_PASSWORD",
        "SHODAN_API_KEY", "SPIDERFOOT_API_KEY"
        # Add other secret keys here
    }

    def __init__(self):
        # Load all attributes defined above from the module-level constants/env vars
        for name, type_hint in self.__annotations__.items():
            if name != '_secrets': # Skip the internal secrets set
                # Get value from globals() first (which loaded from os.getenv or default)
                value = globals().get(name)
                # Set the attribute on the instance
                setattr(self, name, value)

    def get(self, name: str, default: Any = None) -> Any:
        """Provides dict-like access with a default value."""
        # Prevent access to secrets via get()
        if name in self._secrets:
            logger.warning(f"Attempted to access secret '{name}' via settings.get(). Access secrets directly via os.getenv() when needed.")
            return default
        return getattr(self, name, default)

    # Method to explicitly get a secret (used internally by Orchestrator/Agents)
    # This emphasizes that secrets are handled differently
    def get_secret(self, secret_name: str) -> Optional[str]:
         if secret_name not in self._secrets:
              logger.warning(f"'{secret_name}' is not defined as a known secret key.")
              # Fallback to checking general env vars, but log warning
              val = os.getenv(secret_name)
              if val: logger.warning(f"Found '{secret_name}' in env vars, but it wasn't in the predefined secrets list.")
              return val
         val = os.getenv(secret_name)
         if not val: logger.error(f"Required secret '{secret_name}' is not set in environment variables.")
         return val


settings = Settings()

# --- Validation ---
def validate_core_settings():
    """Validates essential settings required for basic operation."""
    # ### Phase 1 Plan Ref: 1.3 (Implement critical validation)
    required_settings = [
        "DATABASE_URL", "DATABASE_ENCRYPTION_KEY",
        "OPENROUTER_API_KEY", # Needed for LLM calls
        "CLAY_API_KEY", # Needed for core lead gen
        "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_VOICE_NUMBER", # Needed for Voice
        "DEEPGRAM_API_KEY", # Needed for Voice
        "HOSTINGER_EMAIL", "HOSTINGER_SMTP_PASS", "HOSTINGER_IMAP_PASS", # Needed for Email/Notifications/Verification
        "USER_EMAIL", # Needed for notifications
        "W8_NAME", "W8_COUNTRY", "W8_ADDRESS", "W8_TIN", # Needed for compliance/payouts
        "MOROCCAN_BANK_ACCOUNT", # Needed for payouts
    ]
    missing = [key for key in required_settings if not os.getenv(key)] # Check env directly for secrets
    if missing:
        msg = f"CRITICAL settings missing: {', '.join(missing)}. Agency cannot function. Check environment variables."
        logger.critical(msg)
        raise ValueError(msg)
    else:
        logger.info("Core settings appear to be present and validated.")

# Run validation immediately upon import
validate_core_settings()

# --- End of config/settings.py ---