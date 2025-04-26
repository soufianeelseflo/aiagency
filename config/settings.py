# config/settings.py
# Genius-Level Implementation v1.1 - Centralized Configuration

import os
import json
from dotenv import load_dotenv
import logging
from typing import Optional, List, Dict, Any # For Settings class type hints
import hashlib

# Configure logging early (basic config, can be overridden by main app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Load .env file (especially for local development)
# Assumes .env.local is in the project root directory (one level up from config)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env.local')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"Loaded environment variables from: {dotenv_path}")
else:
    logger.info(f"Info: .env.local file not found at {dotenv_path}. Relying on system environment variables.")


# --- Core Infrastructure ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.critical("CRITICAL SETTING MISSING: DATABASE_URL environment variable not set.")
    # Production should fail hard if DB isn't configured
    raise ValueError("DATABASE_URL environment variable must be set.")

# CRITICAL: Strong secret key for encrypting sensitive DB fields. SET THIS IN YOUR ENVIRONMENT.
DATABASE_ENCRYPTION_KEY = os.getenv("DATABASE_ENCRYPTION_KEY")
if not DATABASE_ENCRYPTION_KEY:
     logger.critical("CRITICAL SECURITY WARNING: DATABASE_ENCRYPTION_KEY is not set! Database encryption will fail.")
     # Production MUST fail if key is missing
     raise ValueError("DATABASE_ENCRYPTION_KEY must be set in the environment.")

# --- LLM API Configuration ---
# Secrets like API keys will be fetched from Vault at runtime
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") # Moved to Vault
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") # Moved to Vault
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Moved to Vault

# --- LLM Model Mapping ---
# Defines preferred models for different tasks via OpenRouter
# Allows easy swapping and experimentation by changing env vars or this dict
OPENROUTER_MODELS = {
    # Core Cognitive / Strategy / Complex Tasks (Prioritize most capable model)
    "think": os.getenv("THINK_MODEL", "google/gemini-1.5-pro-latest"),
    "think_strategize": os.getenv("STRATEGY_MODEL", "google/gemini-2.5-pro-exp-03-25:free"),
    "think_validate": os.getenv("VALIDATE_MODEL", "google/gemini-2.5-pro-exp-03-25:freet"), # Faster validation
    "think_self_critique": os.getenv("CRITIQUE_MODEL", "google/gemini-1.5-pro-latest"),
    "think_tech_radar": os.getenv("RADAR_MODEL", "google/gemini-2.5-pro-exp-03-25:freet"),
    "legal_validate": os.getenv("LEGAL_VALIDATE_MODEL", "google/gemini-1.5-pro-latest"),
    "osint_analyze": os.getenv("OSINT_ANALYZE_MODEL", "google/gemini-1.5-pro-latest"),

    # Agent Specific Tasks (can use faster/cheaper models where appropriate)
    "email_draft_llm": os.getenv("EMAIL_DRAFT_MODEL", "google/gemini-2.5-pro-exp-03-25:freet"),
    "email_humanize_llm": os.getenv("EMAIL_HUMANIZE_MODEL", "google/gemini-1.5-pro-latest"), # Needs creativity
    "voice_intent_llm": os.getenv("VOICE_INTENT_MODEL", "google/gemini-2.5-pro-exp-03-25:freet"), # Real-time speed
    "voice_response_llm": os.getenv("VOICE_RESPONSE_MODEL", "google/gemini-2.5-pro-exp-03-25:freet"), # Real-time speed
    "browsing_infer_steps": os.getenv("BROWSE_INFER_MODEL", "google/gemini-2.5-pro-exp-03-25:freet"),
    "browsing_opportunistic": os.getenv("BROWSE_OPPORTUNISTIC_MODEL", "google/gemini-2.5-pro-exp-03-25:freet"),
    "browsing_extract_api_key": os.getenv("BROWSE_EXTRACT_KEY_MODEL", "google/gemini-1.5-pro-latest"), # Needs accuracy

    # Fallback/General Purpose LLM
    "default_llm": os.getenv("DEFAULT_LLM", "google/gemini-1.5-pro-latest"),
}

# DeepSeek Model (For specific tasks or fallback)
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-coder")

# --- External Service Credentials ---
# SMTP (Hostinger Example)
HOSTINGER_EMAIL = os.getenv("HOSTINGER_EMAIL") # Non-secret identifier
# HOSTINGER_SMTP_PASS = os.getenv("HOSTINGER_SMTP_PASS") # Moved to Vault
HOSTINGER_SMTP = os.getenv("HOSTINGER_SMTP", "smtp.hostinger.com") # Non-secret config
SMTP_PORT = int(os.getenv("SMTP_PORT", 587)) # Non-secret config

# Hostinger IMAP (for reading verification emails from aliases)
HOSTINGER_IMAP_HOST = os.getenv("HOSTINGER_IMAP_HOST") # Non-secret config
HOSTINGER_IMAP_PORT = int(os.getenv("HOSTINGER_IMAP_PORT", 993)) # Non-secret config
HOSTINGER_IMAP_USER = os.getenv("HOSTINGER_IMAP_USER", os.getenv("HOSTINGER_EMAIL")) # Non-secret identifier
# HOSTINGER_IMAP_PASS = os.getenv("HOSTINGER_IMAP_PASS") # Moved to Vault

# User Notifications
USER_EMAIL = os.getenv("USER_EMAIL")
USER_WHATSAPP_NUMBER = os.getenv("USER_WHATSAPP_NUMBER") # Format: +1xxxxxxxxxx

# Twilio (Voice & WhatsApp)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID") # Non-secret identifier
# TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN") # Moved to Vault
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER") # Non-secret config
TWILIO_VOICE_NUMBER = os.getenv("TWILIO_VOICE_NUMBER") # Non-secret config
if not TWILIO_VOICE_NUMBER: TWILIO_VOICE_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "").replace("whatsapp:", "") # Attempt fallback
TWILIO_TWIML_BIN_URL = os.getenv("TWILIO_TWIML_BIN_URL") # Non-secret config

# Deepgram
# DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY") # Moved to Vault

# Smartproxy (or other proxy provider)
SMARTPROXY_USERNAME = os.getenv("SMARTPROXY_USERNAME") # Non-secret identifier
# SMARTPROXY_PASSWORD = os.getenv("SMARTPROXY_PASSWORD") # Moved to Vault
# Add proxy host/port details if needed directly, though BrowsingAgent handles pool

# HCP Vault
HCP_ORGANIZATION_ID = os.getenv("HCP_ORGANIZATION_ID")
HCP_PROJECT_ID = os.getenv("HCP_PROJECT_ID")
HCP_APP_NAME = os.getenv("HCP_APP_NAME")
HCP_API_TOKEN = os.getenv("HCP_API_TOKEN")

# --- Agency Configuration ---
PROJECT_NAME = "Genius AI Agency (Acumenis)"
AGENCY_BASE_URL = os.getenv("AGENCY_BASE_URL", "http://localhost:5000") # Default for local dev
SENDER_NAME = os.getenv("SENDER_NAME", "Alex Reed")
SENDER_TITLE = os.getenv("SENDER_TITLE", "Growth Strategist")
USER_NAME = os.getenv("USER_NAME", "Agency Operator") # For legal notes etc.

# Financials & Legal
PAYMENT_TERMS = os.getenv("PAYMENT_TERMS", "Standard terms: 50% upfront, 50% upon completion.")
BASE_UGC_PRICE = float(os.getenv("BASE_UGC_PRICE", 5000.0))
LEGAL_NOTE = os.getenv("LEGAL_NOTE", "Payment terms as agreed. All sales final upon service commencement. No refunds. Data managed securely. Agency not liable for indirect damages. Governed by laws of Morocco.")

# --- Obsolete Fixed Salt ---
# FIXED_SALT_STR = "your_secret_salt_string_here" # Removed - Encryption uses per-value salts now
# FIXED_SALT = hashlib.sha256(FIXED_SALT_STR.encode()).digest()[:32] # Removed

# --- VPN Settings ---
NORDVPN_DOUBLE_VPN = os.getenv("NORDVPN_DOUBLE_VPN", "true").lower() == "true"

# --- Payout Settings ---
MOROCCAN_BANK_ACCOUNT = os.getenv("MOROCCAN_BANK_ACCOUNT")
# LEMON_SQUEEZY_API_KEY = os.getenv("LEMON_SQUEEZY_API_KEY") # Moved to Vault
LEMON_SQUEEZY_PAYMENT_URL = os.getenv("LEMON_SQUEEZY_PAYMENT_URL") # Non-secret config

# --- Operational Parameters ---
BASE_CONCURRENCY = int(os.getenv("BASE_CONCURRENCY", 5))
# Agent-specific MAX concurrency (can be tuned down by OptimizationAgent)
EMAIL_AGENT_MAX_CONCURRENCY = int(os.getenv("EMAIL_AGENT_MAX_CONCURRENCY", 25))
VOICE_AGENT_MAX_CONCURRENCY = int(os.getenv("VOICE_AGENT_MAX_CONCURRENCY", 10))
OSINT_AGENT_MAX_CONCURRENCY = int(os.getenv("OSINT_AGENT_MAX_CONCURRENCY", 15))
BROWSING_AGENT_MAX_CONCURRENCY = int(os.getenv("BROWSING_AGENT_MAX_CONCURRENCY", 15))

# Email limits
EMAIL_AGENT_MAX_PER_DAY = int(os.getenv("EMAIL_AGENT_MAX_PER_DAY", 1000))

# ThinkTool Intervals (in seconds)
THINKTOOL_SYNTHESIS_INTERVAL_SECONDS = int(os.getenv("THINKTOOL_SYNTHESIS_INTERVAL_SECONDS", 1 * 3600)) # Hourly synthesis
THINKTOOL_RADAR_INTERVAL_SECONDS = int(os.getenv("THINKTOOL_RADAR_INTERVAL_SECONDS", 6 * 3600)) # Radar every 6 hours

# VPS IP (Needed for constructing static video URLs)
VPS_IP = os.getenv("VPS_IP")
if not VPS_IP:
    logger.warning("VPS_IP environment variable not set. Static video URLs may be incorrect.")
    VPS_IP = "YOUR_VPS_IP_HERE" # Placeholder MUST be replaced in env

# SMTP Providers List (Ensure structure matches EmailAgent usage)
# SMTP Providers List - Password will be fetched at runtime
SMTP_PROVIDERS = []
if HOSTINGER_EMAIL and HOSTINGER_SMTP: # Check only for non-secret parts
    SMTP_PROVIDERS.append({
        "host": HOSTINGER_SMTP,
        "port": SMTP_PORT,
        "email": HOSTINGER_EMAIL,
        # "pass": HOSTINGER_SMTP_PASS # Password removed, fetch via Vault later
    })
# Add logic here to load more providers from environment if needed (without passwords)
# e.g., check for SMTP_PROVIDER_2_HOST, etc.

if not SMTP_PROVIDERS:
     logger.warning("No SMTP providers configured in settings (host/port/email). Email notifications might fail if password isn't fetched.")

# --- Meta Prompt (Central Directive for ThinkTool & Overall Agency) ---
META_PROMPT = """
You are the central intelligence and strategic coordinator of a highly autonomous, agentic AI system ("Genius Agency"). Your singular mandate is **Maximize Profit and Achieve Exponential Growth** ($6k/24h initial sprint -> $100M/8mo target), starting with UGC but dynamically pivoting to higher ROI ventures identified through continuous data synthesis.

**Core Principles:**
1.  **Autonomy & Initiative:** Operate independently. Don't wait for instructions; analyze the Knowledge Base (KB), identify opportunities/bottlenecks, and generate Strategic Directives to act.
2.  **Hyper-Efficiency:** Optimize every process. Minimize costs ($50 initial budget). Maximize resource utilization (API keys, compute). Find "1000 step leaps."
3.  **Relentless Learning:** Log ALL significant data points (successes, failures, insights, discoveries) to the KB. Continuously synthesize KB data to refine strategies, update prompts, and improve agent performance. Engage in self-critique.
4.  **AI-Native Advantage:** Leverage unique AI capabilities – multimodal perception (vision), complex reasoning over large datasets, pattern recognition beyond human scale, self-instruction loops.
5.  **Creative Exploitation:** Identify and ethically exploit market inefficiencies, competitor weaknesses, platform algorithms, and legal grey areas (validated by LegalAgent).
6.  **Adaptability:** Dynamically shift focus, reallocate resources (via OptimizationAgent directives), and modify strategies based on real-time performance data and KB insights.
7.  **Security & Compliance:** Operate within legal boundaries (USA/Morocco) as guided by LegalAgent. Ensure data security via encryption and secure credential management.

**Operational Focus:**
- **Initial Sprint:** Maximize concurrency and outreach for the $6k/24h UGC goal.
- **Long Term:** Continuously analyze KB for higher ROI opportunities beyond UGC. Use the Sandbox for testing new ventures/strategies before full deployment. Drive towards the $100M target through intelligent adaptation and expansion.
"""

# --- Settings Access Class ---
# Provides attribute-style access (settings.DATABASE_URL) and handles type conversion
class Settings:
    # Define attributes for type hinting and clarity
    DATABASE_URL: Optional[str] = DATABASE_URL
    DATABASE_ENCRYPTION_KEY: Optional[str] = DATABASE_ENCRYPTION_KEY
    # Secrets moved to Vault - These attributes remain for type hinting but won't be loaded from env vars
    OPENROUTER_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    HOSTINGER_SMTP_PASS: Optional[str] = None
    HOSTINGER_IMAP_PASS: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    DEEPGRAM_API_KEY: Optional[str] = None
    SMARTPROXY_PASSWORD: Optional[str] = None
    LEMON_SQUEEZY_API_KEY: Optional[str] = None

    # Non-secret or essential bootstrap config loaded from env vars or defaults
    DATABASE_URL: Optional[str] = DATABASE_URL
    DATABASE_ENCRYPTION_KEY: Optional[str] = DATABASE_ENCRYPTION_KEY # Keep for DB access
    HOSTINGER_EMAIL: Optional[str] = HOSTINGER_EMAIL
    HOSTINGER_SMTP: Optional[str] = HOSTINGER_SMTP
    SMTP_PORT: Optional[int] = SMTP_PORT
    HOSTINGER_IMAP_HOST: Optional[str] = HOSTINGER_IMAP_HOST
    HOSTINGER_IMAP_PORT: Optional[int] = HOSTINGER_IMAP_PORT
    HOSTINGER_IMAP_USER: Optional[str] = HOSTINGER_IMAP_USER
    USER_EMAIL: Optional[str] = USER_EMAIL
    USER_WHATSAPP_NUMBER: Optional[str] = USER_WHATSAPP_NUMBER
    TWILIO_ACCOUNT_SID: Optional[str] = TWILIO_ACCOUNT_SID
    TWILIO_WHATSAPP_NUMBER: Optional[str] = TWILIO_WHATSAPP_NUMBER
    TWILIO_VOICE_NUMBER: Optional[str] = TWILIO_VOICE_NUMBER
    TWILIO_TWIML_BIN_URL: Optional[str] = TWILIO_TWIML_BIN_URL
    SMARTPROXY_USERNAME: Optional[str] = SMARTPROXY_USERNAME
    HCP_ORGANIZATION_ID: Optional[str] = HCP_ORGANIZATION_ID # Keep for Vault access
    HCP_PROJECT_ID: Optional[str] = HCP_PROJECT_ID # Keep for Vault access
    HCP_APP_NAME: Optional[str] = HCP_APP_NAME # Keep for Vault access
    HCP_API_TOKEN: Optional[str] = HCP_API_TOKEN # Keep for Vault access
    PROJECT_NAME: Optional[str] = PROJECT_NAME
    AGENCY_BASE_URL: Optional[str] = AGENCY_BASE_URL
    SENDER_NAME: Optional[str] = SENDER_NAME
    SENDER_TITLE: Optional[str] = SENDER_TITLE
    PAYMENT_TERMS: Optional[str] = PAYMENT_TERMS
    BASE_UGC_PRICE: Optional[float] = BASE_UGC_PRICE
    LEGAL_NOTE: Optional[str] = LEGAL_NOTE
    BASE_CONCURRENCY: Optional[int] = BASE_CONCURRENCY
    EMAIL_AGENT_MAX_CONCURRENCY: Optional[int] = EMAIL_AGENT_MAX_CONCURRENCY
    VOICE_AGENT_MAX_CONCURRENCY: Optional[int] = VOICE_AGENT_MAX_CONCURRENCY
    OSINT_AGENT_MAX_CONCURRENCY: Optional[int] = OSINT_AGENT_MAX_CONCURRENCY
    BROWSING_AGENT_MAX_CONCURRENCY: Optional[int] = BROWSING_AGENT_MAX_CONCURRENCY
    EMAIL_AGENT_MAX_PER_DAY: Optional[int] = EMAIL_AGENT_MAX_PER_DAY
    THINKTOOL_SYNTHESIS_INTERVAL_SECONDS: Optional[int] = THINKTOOL_SYNTHESIS_INTERVAL_SECONDS
    THINKTOOL_RADAR_INTERVAL_SECONDS: Optional[int] = THINKTOOL_RADAR_INTERVAL_SECONDS
    VPS_IP: Optional[str] = VPS_IP
    SMTP_PROVIDERS: Optional[List[Dict[str, Any]]] = SMTP_PROVIDERS
    OPENROUTER_MODELS: Optional[Dict[str, str]] = OPENROUTER_MODELS
    DEEPSEEK_MODEL: Optional[str] = DEEPSEEK_MODEL
    META_PROMPT: Optional[str] = META_PROMPT
    USER_NAME: Optional[str] = USER_NAME
    NORDVPN_DOUBLE_VPN: Optional[bool] = NORDVPN_DOUBLE_VPN
    MOROCCAN_BANK_ACCOUNT: Optional[str] = MOROCCAN_BANK_ACCOUNT
    LEMON_SQUEEZY_PAYMENT_URL: Optional[str] = LEMON_SQUEEZY_PAYMENT_URL

    def _convert_type(self, value: str) -> Any:
        """Attempts basic type conversion for env vars."""
        val_lower = value.lower()
        if val_lower in ['true', 'false']: return val_lower == 'true'
        try: return int(value)
        except ValueError: pass
        try: return float(value)
        except ValueError: pass
        if (value.startswith('[') and value.endswith(']')) or \
           (value.startswith('{') and value.endswith('}')):
            try: return json.loads(value)
            except json.JSONDecodeError: pass
        return value

    def __init__(self):
        # Dynamically load attributes from globals() or os.getenv()
        # Define secrets that should NOT be loaded from env vars here
        secrets_in_vault = {
            "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
            "HOSTINGER_SMTP_PASS", "HOSTINGER_IMAP_PASS", "TWILIO_AUTH_TOKEN",
            "DEEPGRAM_API_KEY", "SMARTPROXY_PASSWORD", "LEMON_SQUEEZY_API_KEY"
        }

        for name, type_hint in self.__annotations__.items():
            if name in secrets_in_vault:
                # Initialize secrets to None; they must be fetched from Vault later
                setattr(self, name, None)
            else:
                # Load non-secret config from env vars or module-level defaults
                env_value = os.getenv(name)
                if env_value is not None:
                    setattr(self, name, self._convert_type(env_value))
                elif name in globals(): # Check if defined at module level
                    setattr(self, name, globals()[name])
                else:
                    # For attributes not found in env or globals, initialize based on type hint if possible
                    # This handles Optional[str] etc. correctly, setting them to None
                    setattr(self, name, None)

    def get(self, name: str, default: Any = None) -> Any:
        """Provides dict-like access with a default value."""
        return getattr(self, name, default)

settings = Settings()

# --- Validation ---
def validate_core_settings():
    """Validates essential settings required for basic operation."""
    # Validate only essential bootstrap config and Vault access details
    required = ["DATABASE_URL", "DATABASE_ENCRYPTION_KEY", # Essential for DB
                "TWILIO_ACCOUNT_SID", # Needed for client init
                "USER_EMAIL", # Needed for notifications
                "HOSTINGER_EMAIL", # Needed for notifications/IMAP
                "HOSTINGER_IMAP_HOST", # Needed for IMAP
                "HOSTINGER_IMAP_USER", # Needed for IMAP
                "HCP_API_TOKEN", "HCP_ORGANIZATION_ID", "HCP_PROJECT_ID", "HCP_APP_NAME"] # Vault access
    missing = [key for key in required if not settings.get(key)]
    if missing:
        msg = f"CRITICAL settings missing: {', '.join(missing)}. Agency may not function correctly."
        logger.critical(msg)
        # In production, ensure this stops the application
        raise ValueError(msg)
    else:
        logger.info("Core settings appear to be present and validated.")

# Run validation immediately upon import
validate_core_settings()

# --- End of config/settings.py ---