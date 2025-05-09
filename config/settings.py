# Filename: config/settings.py
# Description: Configuration settings for the AI Agency, loaded from environment variables.
# Version: 1.2.1 (IGNIS - Ensured SecretStr is imported from pydantic)

import os
from typing import Optional, List, Dict, Any # Keep general typing imports
from enum import Enum

# --- Pydantic Imports ---
# Ensure SecretStr is imported here!
from pydantic import (
    BaseModel, 
    Field, 
    validator,  # Retained for compatibility if older Pydantic features are used elsewhere
    SecretStr,  # <<<< THE CRUCIAL IMPORT
    AnyHttpUrl, 
    PostgresDsn,
    EmailStr,
    HttpUrl,     # Added HttpUrl explicitly if used
    DirectoryPath # Added DirectoryPath explicitly if used
    # field_validator, model_validator, ValidationInfo, AnyUrl # These are for Pydantic v2 style validation
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# Determine the root directory of the project
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class LogLevelEnum(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class Settings(BaseSettings):
    """
    Defines the application settings, loaded from environment variables.
    Utilizes Pydantic BaseSettings for automatic loading and validation.
    """
    # --- Core Infrastructure & API Keys ---
    DATABASE_URL: PostgresDsn 
    DATABASE_ENCRYPTION_KEY: SecretStr 
    OPENROUTER_API_KEY: Optional[SecretStr] = None
    AGENCY_BASE_URL: AnyHttpUrl

    # --- Logging ---
    LOG_LEVEL: LogLevelEnum = LogLevelEnum.INFO
    LOG_FILE_PATH: Optional[str] = None 
    OPERATIONAL_LOG_FILE_PATH: Optional[str] = None

    # --- Email Configuration ---
    HOSTINGER_EMAIL: Optional[EmailStr] = None # Use EmailStr for validation
    HOSTINGER_IMAP_PASS: Optional[SecretStr] = None
    HOSTINGER_IMAP_SERVER: str = "imap.hostinger.com"
    HOSTINGER_IMAP_PORT: int = 993
    HOSTINGER_IMAP_USER: Optional[EmailStr] = None # Use EmailStr
    HOSTINGER_SMTP_SERVER: str = "smtp.hostinger.com"
    HOSTINGER_SMTP_PORT: int = 465 
    HOSTINGER_SMTP_USER: Optional[EmailStr] = None # Use EmailStr
    HOSTINGER_SMTP_PASS: Optional[SecretStr] = None

    # --- Twilio (Voice Calls) ---
    TWILIO_ACCOUNT_SID: Optional[SecretStr] = None
    TWILIO_AUTH_TOKEN: Optional[SecretStr] = None
    TWILIO_VOICE_NUMBER: Optional[str] = None # E.164 format

    # --- Deepgram (Speech-to-Text) ---
    DEEPGRAM_API_KEY: Optional[SecretStr] = None

    # --- Proxy Configuration (Decodo/Smartproxy) ---
    SMARTPROXY_USER: Optional[SecretStr] = None
    SMARTPROXY_PASSWORD: Optional[SecretStr] = None
    SMARTPROXY_HOST: Optional[str] = None 
    SMARTPROXY_PORT: Optional[int] = None

    # --- Optional External Services ---
    CLAY_API_KEY: Optional[SecretStr] = None
    CLAY_CALLBACK_SECRET: Optional[SecretStr] = None
    MAILERSEND_API_KEY: Optional[SecretStr] = None
    MAILERCHECK_API_KEY: Optional[SecretStr] = None
    DECODO_WEBHOOK_SECRET: Optional[SecretStr] = None # For securing incoming Decodo webhooks

    # --- Application Behavior & Defaults ---
    SENDER_NAME: str = "AI Agency"
    USER_EMAIL: Optional[EmailStr] = None # Operator's email
    DOWNLOAD_PASSWORD: SecretStr 
    MAX_CONCURRENT_CALLS: int = Field(default=3, ge=1, le=50)
    AUTO_APPROVE_OPERATIONS: bool = False

    # --- Paths ---
    TEMP_AUDIO_DIR: str = "/app/temp_audio"
    TEMP_DOWNLOAD_DIR: str = "/app/temp_downloads"
    ASSETS_DIR: str = "/app/assets"

    # --- AI Model Configuration ---
    AI_SMART_MODEL: str = Field(default="openai/gpt-4-turbo-preview", alias="SMART_MODEL")
    AI_FAST_MODEL: str = Field(default="openai/gpt-3.5-turbo", alias="FAST_MODEL")
    AI_DEFAULT_MODEL_FALLBACKS: List[str] = Field(default=["anthropic/claude-3-haiku", "google/gemini-pro"], alias="DEFAULT_MODEL_FALLBACKS")
    AI_DEFAULT_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    AI_DEFAULT_MAX_TOKENS: int = Field(default=2048, gt=0)

    # --- Static Business Information ---
    SENDER_COMPANY_NAME: Optional[str] = "Your AI Agency LLC"
    SENDER_COMPANY_ADDRESS: Optional[str] = "123 Future Drive, AI City, ST 90210"
    W8_NAME: Optional[str] = Field(default=None, alias="W8_FORM_NAME")
    W8_ADDRESS: Optional[str] = Field(default=None, alias="W8_FORM_ADDRESS")
    W8_COUNTRY: Optional[str] = Field(default=None, alias="W8_FORM_COUNTRY")
    MOROCCAN_BANK_ACCOUNT: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False
    )

    @validator('HOSTINGER_IMAP_USER', 'HOSTINGER_SMTP_USER', pre=True, always=True)
    def default_hostinger_user(cls, v, values):
        hostinger_email = values.get('HOSTINGER_EMAIL')
        if v is None and hostinger_email:
            return hostinger_email
        return v

    @validator('HOSTINGER_SMTP_PASS', pre=True, always=True)
    def default_hostinger_smtp_pass(cls, v, values):
        hostinger_imap_pass = values.get('HOSTINGER_IMAP_PASS')
        if v is None and hostinger_imap_pass:
            return hostinger_imap_pass
        return v
    
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def get_secret(self, key: str) -> Optional[str]:
        value = getattr(self, key, None)
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        return value if isinstance(value, (str, type(None))) else None

try:
    settings = Settings()
except Exception as e_settings_init:
    import sys
    print(f"CRITICAL ERROR: Failed to initialize Pydantic Settings: {e_settings_init}", file=sys.stderr)
    sys.exit("Settings initialization failed. Application cannot start.")
