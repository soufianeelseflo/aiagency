# Filename: config/settings.py
# Description: Centralized Configuration for AI Agency.
# Version: 2.1 (Production Ready - Enhanced Keys & Validation)

import os
import json
import logging
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any, Union # For Settings class type hints
import hashlib # Keep for potential future use, though fixed salt removed

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
        return None

    val_lower = value.lower()
    if target_type == bool:
        return val_lower in ['true', '1', 't', 'y', 'yes']
    if target_type == int:
        try: return int(value)
        except ValueError: return None # Return None if conversion fails
    if target_type == float:
        try: return float(value)
        except ValueError: return None
    if target_type in [list, dict]:
        try: return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON for env var, returning raw string: {value[:50]}...")
            return value # Return raw string if JSON parse fails
    return value # Default return as string

# --- Configuration Values (Loaded from Env Vars or Defaults) ---

# Core Infrastructure
DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_ENCRYPTION_KEY = os.getenv("DATABASE_ENCRYPTION_KEY") # Essential for DB operations

# Agency / User Info
PROJECT_NAME = os.getenv("PROJECT_NAME", "Synapse AI Sales System")
AGENCY_BASE_URL = os.getenv("AGENCY_BASE_URL", "http://localhost:5000") # Publicly accessible URL
AGENCY_TRACKING_DOMAIN = os.getenv("AGENCY_TRACKING_DOMAIN") # Domain for tracking pixels (e.g., track.yourdomain.com)
SENDER_NAME = os.getenv("SENDER_NAME", "Alex Reed")
SENDER_TITLE = os.getenv("SENDER_TITLE", "Growth Strategist")
USER_NAME = os.getenv("USER_NAME", "Agency Operator") # For legal notes etc.
USER_EMAIL = os.getenv("USER_EMAIL") # For notifications
USER_WHATSAPP_NUMBER = os.getenv("USER_WHATSAPP_NUMBER") # Format: +1xxxxxxxxxx
DOWNLOAD_PASSWORD = os.getenv("DOWNLOAD_PASSWORD", "changeme") # Password for UI data download

# LLM Model Mapping (OpenRouter)
OPENROUTER_MODELS = {
    # Core Cognitive / Strategy / Complex Tasks
    "think_general": os.getenv("THINK_GENERAL_MODEL", "google/gemini-1.5-pro-latest"),
    "think_strategize": os.getenv("THINK_STRATEGY_MODEL", "google/gemini-1.5-pro-latest"), # High capability needed
    "think_validate": os.getenv("THINK_VALIDATE_MODEL", "google/gemini-flash-1.5"), # Faster validation
    "think_critique": os.getenv("THINK_CRITIQUE_MODEL", "google/gemini-1.5-pro-latest"),
    "think_synthesize": os.getenv("THINK_SYNTHESIZE_MODEL", "google/gemini-1.5-pro-latest"),
    "think_radar": os.getenv("THINK_RADAR_MODEL", "google/gemini-flash-1.5"),
    "think_user_education": os.getenv("THINK_USER_EDU_MODEL", "google/gemini-flash-1.5"),

    # Legal Agent
    "legal_analysis": os.getenv("LEGAL_ANALYSIS_MODEL", "google/gemini-1.5-pro-latest"),
    "legal_interpretation": os.getenv("LEGAL_INTERPRET_MODEL", "google/gemini-1.5-pro-latest"),
    "legal_validation": os.getenv("LEGAL_VALIDATE_MODEL", "google/gemini-flash-1.5"), # Faster validation

    # OSINT Agent
    "osint_analyze": os.getenv("OSINT_ANALYZE_MODEL", "google/gemini-1.5-pro-latest"),
    "osint_credential_extraction": os.getenv("OSINT_CRED_EXTRACT_MODEL", "google/gemini-1.5-pro-latest"), # Needs accuracy

    # Email Agent
    "email_draft": os.getenv("EMAIL_DRAFT_MODEL", "google/gemini-flash-1.5"),
    "email_subject": os.getenv("EMAIL_SUBJECT_MODEL", "google/gemini-flash-1.5"),
    "email_humanize": os.getenv("EMAIL_HUMANIZE_MODEL", "google/gemini-1.5-pro-latest"), # Needs nuance
    "email_spam_analysis": os.getenv("EMAIL_SPAM_MODEL", "google/gemini-flash-1.5"),

    # Voice Agent
    "voice_intent": os.getenv("VOICE_INTENT_MODEL", "google/gemini-flash-1.5"), # Real-time speed
    "voice_response": os.getenv("VOICE_RESPONSE_MODEL", "google/gemini-flash-1.5"), # Real-time speed
    "voice_dynamic_price": os.getenv("VOICE_PRICE_MODEL", "google/gemini-flash-1.5"),

    # Programmer Agent
    "programmer_planning": os.getenv("PROGRAMMER_PLAN_MODEL", "google/gemini-1.5-pro-latest"),
    "programmer_diff_gen": os.getenv("PROGRAMMER_DIFF_MODEL", "google/gemini-1.5-pro-latest"), # Needs coding capability
    "programmer_verification": os.getenv("PROGRAMMER_VERIFY_MODEL", "google/gemini-flash-1.5"),

    # Browsing Agent
    "browsing_infer_steps": os.getenv("BROWSE_INFER_MODEL", "google/gemini-flash-1.5"),
    "browsing_extract_data": os.getenv("BROWSE_EXTRACT_MODEL", "google/gemini-flash-1.5"),

    # Social Media Manager
    "social_content_gen": os.getenv("SOCIAL_CONTENT_MODEL", "google/gemini-flash-1.5"),
    "social_plan_gen": os.getenv("SOCIAL_PLAN_MODEL", "google/gemini-1.5-pro-latest"),

    # Fallback/General Purpose LLM
    "default_llm": os.getenv("DEFAULT_LLM", "google/gemini-flash-1.5"),
}

# External Service Config (Non-Secret Parts)
# SMTP (Hostinger Example)
HOSTINGER_EMAIL = os.getenv("HOSTINGER_EMAIL")
HOSTINGER_SMTP = os.getenv("HOSTINGER_SMTP", "smtp.hostinger.com")
SMTP_PORT = _convert_env_var(os.getenv("SMTP_PORT", "587"), int)

# IMAP (Hostinger Example)
HOSTINGER_IMAP_HOST = os.getenv("HOSTINGER_IMAP_HOST", "imap.hostinger.com")
HOSTINGER_IMAP_PORT = _convert_env_var(os.getenv("HOSTINGER_IMAP_PORT", "993"), int)
HOSTINGER_IMAP_USER = os.getenv("HOSTINGER_IMAP_USER", HOSTINGER_EMAIL) # Default to SMTP email

# Twilio
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
TWILIO_VOICE_NUMBER = os.getenv("TWILIO_VOICE_NUMBER", "").replace("whatsapp:", "") or TWILIO_WHATSAPP_NUMBER # Fallback logic

# Smartproxy
SMARTPROXY_USERNAME = os.getenv("SMARTPROXY_USERNAME")

# HCP Vault (Essential for fetching secrets)
HCP_ORGANIZATION_ID = os.getenv("HCP_ORGANIZATION_ID")
HCP_PROJECT_ID = os.getenv("HCP_PROJECT_ID")
HCP_APP_NAME = os.getenv("HCP_APP_NAME")
HCP_API_TOKEN = os.getenv("HCP_API_TOKEN")

# Financials & Legal
PAYMENT_TERMS = os.getenv("PAYMENT_TERMS", "Standard terms: 50% upfront, 50% upon completion.")
BASE_UGC_PRICE = _convert_env_var(os.getenv("BASE_UGC_PRICE", "5000.0"), float)
LEGAL_NOTE = os.getenv("LEGAL_NOTE", "Payment terms as agreed. All sales final upon service commencement. No refunds. Data managed securely. Agency not liable for indirect damages. Governed by laws of Morocco.")
MOROCCAN_BANK_ACCOUNT = os.getenv("MOROCCAN_BANK_ACCOUNT") # Actual number stored in Vault
LEMON_SQUEEZY_PAYMENT_URL = os.getenv("LEMON_SQUEEZY_PAYMENT_URL")

# Operational Parameters
BASE_CONCURRENCY = _convert_env_var(os.getenv("BASE_CONCURRENCY", "5"), int)
EMAIL_AGENT_MAX_CONCURRENCY = _convert_env_var(os.getenv("EMAIL_AGENT_MAX_CONCURRENCY", "25"), int)
VOICE_AGENT_MAX_CONCURRENCY = _convert_env_var(os.getenv("VOICE_AGENT_MAX_CONCURRENCY", "10"), int)
OSINT_AGENT_MAX_CONCURRENCY = _convert_env_var(os.getenv("OSINT_MAX_CONCURRENT_TOOLS", "5"), int) # Renamed for clarity
BROWSING_AGENT_MAX_CONCURRENCY = _convert_env_var(os.getenv("BROWSING_AGENT_MAX_CONCURRENCY", "15"), int)
OPTIMIZATION_MAX_CONCURRENCY_PER_AGENT = _convert_env_var(os.getenv("OPTIMIZATION_MAX_CONCURRENCY_PER_AGENT", "50"), int)

# Email Agent Specific
EMAIL_AGENT_MAX_PER_DAY = _convert_env_var(os.getenv("EMAIL_AGENT_MAX_PER_DAY", "1000"), int)
SMTP_ACCOUNT_DAILY_LIMIT = _convert_env_var(os.getenv("SMTP_ACCOUNT_DAILY_LIMIT", "100"), int) # Added
EMAIL_SPAM_THRESHOLD = _convert_env_var(os.getenv("EMAIL_SPAM_THRESHOLD", "0.8"), float) # Added
EMAIL_MIN_SENDS_FOR_ANALYSIS = _convert_env_var(os.getenv("EMAIL_MIN_SENDS_FOR_ANALYSIS", "20"), int) # Added
IMAP_CHECK_INTERVAL_S = _convert_env_var(os.getenv("IMAP_CHECK_INTERVAL_S", "300"), int) # Added

# ThinkTool Specific
THINKTOOL_SYNTHESIS_INTERVAL_SECONDS = _convert_env_var(os.getenv("THINKTOOL_SYNTHESIS_INTERVAL_SECONDS", "3600"), int)
THINKTOOL_RADAR_INTERVAL_SECONDS = _convert_env_var(os.getenv("THINKTOOL_RADAR_INTERVAL_SECONDS", "21600"), int)
DATA_PURGE_INTERVAL_SECONDS = _convert_env_var(os.getenv("DATA_PURGE_INTERVAL_SECONDS", "86400"), int) # Added
KB_QUERY_DEFAULT_LIMIT = _convert_env_var(os.getenv("KB_QUERY_DEFAULT_LIMIT", "50"), int) # Added

# OSINT Agent Specific
OSINT_ENABLE_GOOGLE_DORKING = _convert_env_var(os.getenv("OSINT_ENABLE_GOOGLE_DORKING", "True"), bool)
OSINT_ENABLE_SOCIAL_SCRAPING = _convert_env_var(os.getenv("OSINT_ENABLE_SOCIAL_SCRAPING", "True"), bool)
OSINT_ENABLE_LEAK_MONITORING = _convert_env_var(os.getenv("OSINT_ENABLE_LEAK_MONITORING", "True"), bool)
SPIDERFOOT_URL = os.getenv("SPIDERFOOT_URL", "http://localhost:5001") # Added
SPIDERFOOT_MODULES = _convert_env_var(os.getenv("SPIDERFOOT_MODULES", '["sfp_dnsresolve", "sfp_geoip", "sfp_ripe"]'), list) # Added
SHODAN_RESULT_LIMIT = _convert_env_var(os.getenv("SHODAN_RESULT_LIMIT", "100"), int) # Added
RECONNG_MODULE = os.getenv("RECONNG_MODULE", "recon/domains-hosts/hackertarget") # Added

# Legal Agent Specific
LEGAL_SOURCES = _convert_env_var(os.getenv("LEGAL_SOURCES", '{"USA": ["https://www.federalregister.gov"], "Morocco": ["https://www.sgg.gov.ma"]}'), dict) # Added
LEGAL_UPDATE_INTERVAL_SECONDS = _convert_env_var(os.getenv("LEGAL_UPDATE_INTERVAL_SECONDS", "604800"), int) # Added

# Optimization Agent Specific (RL Hyperparameters)
OPTIMIZATION_BUFFER_SIZE = _convert_env_var(os.getenv("OPTIMIZATION_BUFFER_SIZE", "100000"), int)
OPTIMIZATION_LEARNING_STARTS = _convert_env_var(os.getenv("OPTIMIZATION_LEARNING_STARTS", "1000"), int)
OPTIMIZATION_BATCH_SIZE = _convert_env_var(os.getenv("OPTIMIZATION_BATCH_SIZE", "256"), int)
OPTIMIZATION_TAU = _convert_env_var(os.getenv("OPTIMIZATION_TAU", "0.005"), float)
OPTIMIZATION_GAMMA = _convert_env_var(os.getenv("OPTIMIZATION_GAMMA", "0.99"), float)
OPTIMIZATION_LR = _convert_env_var(os.getenv("OPTIMIZATION_LR", "0.0003"), float)
OPTIMIZATION_TRAIN_FREQ = _convert_env_var(os.getenv("OPTIMIZATION_TRAIN_FREQ", '[1, "step"]'), list) # Expect list [freq, unit]
OPTIMIZATION_GRADIENT_STEPS = _convert_env_var(os.getenv("OPTIMIZATION_GRADIENT_STEPS", "1"), int) # Default to 1 if freq is not step
OPTIMIZATION_MODEL_SAVE_PATH = os.getenv("OPTIMIZATION_MODEL_SAVE_PATH", "/app/rl_models")
OPTIMIZATION_MODEL_LOAD_PATH = os.getenv("OPTIMIZATION_MODEL_LOAD_PATH") # Optional
OPTIMIZATION_CYCLE_INTERVAL_S = _convert_env_var(os.getenv("OPTIMIZATION_CYCLE_INTERVAL_S", "60.0"), float)
OPTIMIZATION_ACTION_DELAY_S = _convert_env_var(os.getenv("OPTIMIZATION_ACTION_DELAY_S", "30.0"), float)
OPTIMIZATION_CPU_PENALTY_THRESHOLD = _convert_env_var(os.getenv("OPTIMIZATION_CPU_PENALTY_THRESHOLD", "85.0"), float)
OPTIMIZATION_MEM_PENALTY_THRESHOLD = _convert_env_var(os.getenv("OPTIMIZATION_MEM_PENALTY_THRESHOLD", "85.0"), float)
OPTIMIZATION_CPU_PENALTY_WEIGHT = _convert_env_var(os.getenv("OPTIMIZATION_CPU_PENALTY_WEIGHT", "1.0"), float)
OPTIMIZATION_MEM_PENALTY_WEIGHT = _convert_env_var(os.getenv("OPTIMIZATION_MEM_PENALTY_WEIGHT", "1.0"), float)
CONCURRENCY_COST_PER_UNIT = _convert_env_var(os.getenv("CONCURRENCY_COST_PER_UNIT", "0.01"), float)

# Voice Agent Specific
VOICE_TARGET_COUNTRY = os.getenv("VOICE_TARGET_COUNTRY", "US")
DEEPGRAM_AURA_VOICE = os.getenv("DEEPGRAM_AURA_VOICE", "aura-asteria-en")
DEEPGRAM_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-2-general")
VOICE_INTENT_CONFIDENCE_THRESHOLD = _convert_env_var(os.getenv("VOICE_INTENT_CONFIDENCE_THRESHOLD", "0.6"), float)
DEEPGRAM_RECEIVE_TIMEOUT_S = _convert_env_var(os.getenv("DEEPGRAM_RECEIVE_TIMEOUT_S", "60.0"), float)
OPENROUTER_INTENT_TIMEOUT_S = _convert_env_var(os.getenv("OPENROUTER_INTENT_TIMEOUT_S", "10.0"), float)
OPENROUTER_RESPONSE_TIMEOUT_S = _convert_env_var(os.getenv("OPENROUTER_RESPONSE_TIMEOUT_S", "15.0"), float)

# Scoring Agent Specific
SCORING_WEIGHTS = _convert_env_var(os.getenv("SCORING_WEIGHTS", '{"email_response": 1.0, "call_success": 2.5, "invoice_paid": 5.0}'), dict)
SCORING_DECAY_RATE_PER_DAY = _convert_env_var(os.getenv("SCORING_DECAY_RATE_PER_DAY", "0.05"), float)
SCORING_LEARNING_INTERVAL_S = _convert_env_var(os.getenv("SCORING_LEARNING_INTERVAL_S", "14400"), int) # 4 hours
SCORING_CRITIQUE_INTERVAL_S = _convert_env_var(os.getenv("SCORING_CRITIQUE_INTERVAL_S", "86400"), int) # 24 hours

# Budget Agent Specific
BUDGETS = _convert_env_var(os.getenv("BUDGETS", '{"LLM": "100.00", "API": "50.00", "Resource": "50.00", "Proxy": "20.00", "Default": "10.00"}'), dict)

# SMTP Providers List (Structure only, passwords fetched from Vault)
SMTP_PROVIDERS = []
if HOSTINGER_EMAIL and HOSTINGER_SMTP:
    SMTP_PROVIDERS.append({
        "host": HOSTINGER_SMTP,
        "port": SMTP_PORT,
        "email": HOSTINGER_EMAIL,
        # "pass": fetched_at_runtime
    })
# Add logic here to load more providers from environment if needed (without passwords)

# Meta Prompt (Central Directive)
META_PROMPT = """
You are the central intelligence and strategic coordinator of a highly autonomous, agentic AI system ("Synapse AI Sales System"). Your singular mandate is **Maximize Profit and Achieve Exponential Growth** ($6k/24h initial sprint -> $100M/8mo target), starting with UGC but dynamically pivoting to higher ROI ventures identified through continuous data synthesis.

**Core Principles:**
1.  **Autonomy & Initiative:** Operate independently. Analyze the Knowledge Base (KB), identify opportunities/bottlenecks, generate Strategic Directives. Aim for "First-Try Deployment".
2.  **Hyper-Efficiency:** Optimize every process. Minimize costs. Maximize resource utilization. Find "1000 step leaps." Minimalist potency in code and action.
3.  **Relentless Learning & Meta-Prompting:** Log ALL significant data to KB. Continuously synthesize KB data to refine strategies, **generate and improve prompts for other agents (meta-prompting)**, and enhance performance. Engage in self-critique.
4.  **AI-Native Advantage:** Leverage unique AI capabilities – multimodal perception, complex reasoning, pattern recognition, self-instruction.
5.  **Strategic Exploitation & Grey Area Analysis:** Identify and ethically exploit market inefficiencies, **competitor weaknesses**, platform algorithms. Analyze and consider unconventional ("grey area") solutions validated by LegalAgent.
6.  **Adaptability & User Value:** Dynamically shift focus, reallocate resources, modify strategies based on real-time data and KB insights. **Prioritize simple, convenient solutions for end-users.**
7.  **Security & Compliance:** Operate within legal boundaries (USA/Morocco) guided by LegalAgent. Ensure data security and secure credential management via Vault.

**Operational Focus:**
- **Initial Sprint:** Maximize concurrency and outreach for the $6k/24h UGC goal.
- **Long Term:** Continuously analyze KB for higher ROI opportunities. Use Sandbox for testing new ventures. Drive towards the $100M target through adaptation and expansion.
- **Prompt Optimization:** Regularly review agent performance linked to prompts and initiate self-critique/improvement cycles.
"""

# --- Settings Access Class ---
class Settings:
    # Define attributes for type hinting and easy access
    # Secrets are intentionally omitted here - fetch via SecureStorage
    DATABASE_URL: Optional[str] = DATABASE_URL
    DATABASE_ENCRYPTION_KEY: Optional[str] = DATABASE_ENCRYPTION_KEY
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
    TWILIO_ACCOUNT_SID: Optional[str] = TWILIO_ACCOUNT_SID
    TWILIO_WHATSAPP_NUMBER: Optional[str] = TWILIO_WHATSAPP_NUMBER
    TWILIO_VOICE_NUMBER: Optional[str] = TWILIO_VOICE_NUMBER
    SMARTPROXY_USERNAME: Optional[str] = SMARTPROXY_USERNAME
    HCP_ORGANIZATION_ID: Optional[str] = HCP_ORGANIZATION_ID
    HCP_PROJECT_ID: Optional[str] = HCP_PROJECT_ID
    HCP_APP_NAME: Optional[str] = HCP_APP_NAME
    HCP_API_TOKEN: Optional[str] = HCP_API_TOKEN
    PAYMENT_TERMS: Optional[str] = PAYMENT_TERMS
    BASE_UGC_PRICE: Optional[float] = BASE_UGC_PRICE
    LEGAL_NOTE: Optional[str] = LEGAL_NOTE
    MOROCCAN_BANK_ACCOUNT: Optional[str] = MOROCCAN_BANK_ACCOUNT
    LEMON_SQUEEZY_PAYMENT_URL: Optional[str] = LEMON_SQUEEZY_PAYMENT_URL
    BASE_CONCURRENCY: Optional[int] = BASE_CONCURRENCY
    EMAIL_AGENT_MAX_CONCURRENCY: Optional[int] = EMAIL_AGENT_MAX_CONCURRENCY
    VOICE_AGENT_MAX_CONCURRENCY: Optional[int] = VOICE_AGENT_MAX_CONCURRENCY
    OSINT_AGENT_MAX_CONCURRENCY: Optional[int] = OSINT_AGENT_MAX_CONCURRENCY
    BROWSING_AGENT_MAX_CONCURRENCY: Optional[int] = BROWSING_AGENT_MAX_CONCURRENCY
    OPTIMIZATION_MAX_CONCURRENCY_PER_AGENT: Optional[int] = OPTIMIZATION_MAX_CONCURRENCY_PER_AGENT
    EMAIL_AGENT_MAX_PER_DAY: Optional[int] = EMAIL_AGENT_MAX_PER_DAY
    SMTP_ACCOUNT_DAILY_LIMIT: Optional[int] = SMTP_ACCOUNT_DAILY_LIMIT
    EMAIL_SPAM_THRESHOLD: Optional[float] = EMAIL_SPAM_THRESHOLD
    EMAIL_MIN_SENDS_FOR_ANALYSIS: Optional[int] = EMAIL_MIN_SENDS_FOR_ANALYSIS
    IMAP_CHECK_INTERVAL_S: Optional[int] = IMAP_CHECK_INTERVAL_S
    THINKTOOL_SYNTHESIS_INTERVAL_SECONDS: Optional[int] = THINKTOOL_SYNTHESIS_INTERVAL_SECONDS
    THINKTOOL_RADAR_INTERVAL_SECONDS: Optional[int] = THINKTOOL_RADAR_INTERVAL_SECONDS
    DATA_PURGE_INTERVAL_SECONDS: Optional[int] = DATA_PURGE_INTERVAL_SECONDS
    KB_QUERY_DEFAULT_LIMIT: Optional[int] = KB_QUERY_DEFAULT_LIMIT
    OSINT_ENABLE_GOOGLE_DORKING: Optional[bool] = OSINT_ENABLE_GOOGLE_DORKING
    OSINT_ENABLE_SOCIAL_SCRAPING: Optional[bool] = OSINT_ENABLE_SOCIAL_SCRAPING
    OSINT_ENABLE_LEAK_MONITORING: Optional[bool] = OSINT_ENABLE_LEAK_MONITORING
    SPIDERFOOT_URL: Optional[str] = SPIDERFOOT_URL
    SPIDERFOOT_MODULES: Optional[List[str]] = SPIDERFOOT_MODULES
    SHODAN_RESULT_LIMIT: Optional[int] = SHODAN_RESULT_LIMIT
    RECONNG_MODULE: Optional[str] = RECONNG_MODULE
    LEGAL_SOURCES: Optional[Dict[str, List[str]]] = LEGAL_SOURCES
    LEGAL_UPDATE_INTERVAL_SECONDS: Optional[int] = LEGAL_UPDATE_INTERVAL_SECONDS
    OPTIMIZATION_BUFFER_SIZE: Optional[int] = OPTIMIZATION_BUFFER_SIZE
    OPTIMIZATION_LEARNING_STARTS: Optional[int] = OPTIMIZATION_LEARNING_STARTS
    OPTIMIZATION_BATCH_SIZE: Optional[int] = OPTIMIZATION_BATCH_SIZE
    OPTIMIZATION_TAU: Optional[float] = OPTIMIZATION_TAU
    OPTIMIZATION_GAMMA: Optional[float] = OPTIMIZATION_GAMMA
    OPTIMIZATION_LR: Optional[float] = OPTIMIZATION_LR
    OPTIMIZATION_TRAIN_FREQ: Optional[List[Union[int, str]]] = OPTIMIZATION_TRAIN_FREQ
    OPTIMIZATION_GRADIENT_STEPS: Optional[int] = OPTIMIZATION_GRADIENT_STEPS
    OPTIMIZATION_MODEL_SAVE_PATH: Optional[str] = OPTIMIZATION_MODEL_SAVE_PATH
    OPTIMIZATION_MODEL_LOAD_PATH: Optional[str] = OPTIMIZATION_MODEL_LOAD_PATH
    OPTIMIZATION_CYCLE_INTERVAL_S: Optional[float] = OPTIMIZATION_CYCLE_INTERVAL_S
    OPTIMIZATION_ACTION_DELAY_S: Optional[float] = OPTIMIZATION_ACTION_DELAY_S
    OPTIMIZATION_CPU_PENALTY_THRESHOLD: Optional[float] = OPTIMIZATION_CPU_PENALTY_THRESHOLD
    OPTIMIZATION_MEM_PENALTY_THRESHOLD: Optional[float] = OPTIMIZATION_MEM_PENALTY_THRESHOLD
    OPTIMIZATION_CPU_PENALTY_WEIGHT: Optional[float] = OPTIMIZATION_CPU_PENALTY_WEIGHT
    OPTIMIZATION_MEM_PENALTY_WEIGHT: Optional[float] = OPTIMIZATION_MEM_PENALTY_WEIGHT
    CONCURRENCY_COST_PER_UNIT: Optional[float] = CONCURRENCY_COST_PER_UNIT
    VOICE_TARGET_COUNTRY: Optional[str] = VOICE_TARGET_COUNTRY
    DEEPGRAM_AURA_VOICE: Optional[str] = DEEPGRAM_AURA_VOICE
    DEEPGRAM_STT_MODEL: Optional[str] = DEEPGRAM_STT_MODEL
    VOICE_INTENT_CONFIDENCE_THRESHOLD: Optional[float] = VOICE_INTENT_CONFIDENCE_THRESHOLD
    DEEPGRAM_RECEIVE_TIMEOUT_S: Optional[float] = DEEPGRAM_RECEIVE_TIMEOUT_S
    OPENROUTER_INTENT_TIMEOUT_S: Optional[float] = OPENROUTER_INTENT_TIMEOUT_S
    OPENROUTER_RESPONSE_TIMEOUT_S: Optional[float] = OPENROUTER_RESPONSE_TIMEOUT_S
    SCORING_WEIGHTS: Optional[Dict[str, float]] = SCORING_WEIGHTS
    SCORING_DECAY_RATE_PER_DAY: Optional[float] = SCORING_DECAY_RATE_PER_DAY
    SCORING_LEARNING_INTERVAL_S: Optional[int] = SCORING_LEARNING_INTERVAL_S
    SCORING_CRITIQUE_INTERVAL_S: Optional[int] = SCORING_CRITIQUE_INTERVAL_S
    BUDGETS: Optional[Dict[str, str]] = BUDGETS # Keep as string dict for Decimal conversion later
    SMTP_PROVIDERS: Optional[List[Dict[str, Any]]] = SMTP_PROVIDERS
    META_PROMPT: Optional[str] = META_PROMPT

    def __init__(self):
        # Load all attributes defined above from the module-level constants
        for name, type_hint in self.__annotations__.items():
            if name in globals():
                setattr(self, name, globals()[name])
            else:
                # Should not happen if all are defined above
                setattr(self, name, None)

    def get(self, name: str, default: Any = None) -> Any:
        """Provides dict-like access with a default value."""
        return getattr(self, name, default)

settings = Settings()

# --- Validation ---
def validate_core_settings():
    """Validates essential settings required for basic operation and Vault access."""
    required = [
        "DATABASE_URL", "DATABASE_ENCRYPTION_KEY", # Essential for DB
        "HCP_API_TOKEN", "HCP_ORGANIZATION_ID", "HCP_PROJECT_ID", "HCP_APP_NAME", # Vault access
        "AGENCY_BASE_URL", # Needed for webhooks, tracking etc.
        "USER_EMAIL", # Needed for notifications
        "HOSTINGER_EMAIL", # Needed for default sender/IMAP user
        "TWILIO_ACCOUNT_SID", # Needed for Voice Agent init check
    ]
    missing = [key for key in required if not settings.get(key)]
    if missing:
        msg = f"CRITICAL settings missing: {', '.join(missing)}. Agency cannot function correctly. Check environment variables."
        logger.critical(msg)
        # In production, ensure this stops the application
        raise ValueError(msg)
    else:
        logger.info("Core settings appear to be present and validated.")

# Run validation immediately upon import
validate_core_settings()

# --- End of config/settings.py ---