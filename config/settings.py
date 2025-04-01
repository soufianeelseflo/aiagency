# config/settings.py

import os

# Database configuration (from Coolify PostgreSQL)
DATABASE_URL = "postgresql+asyncpg://user:password@localhost/agency_db"  # Update with your Coolify credentials

# OpenRouter model mappings (using Gemini models)
OPENROUTER_MODELS = {
    "think": "google/gemini-2.5-pro-exp-03-25:free",  
    "email": "google/gemini-2.5-pro-exp-03-25:free",
    "legal": "google/gemini-2.5-pro-exp-03-25:free",
    "osint": "google/gemini-2.5-pro-exp-03-25:free",
    "scoring": "google/gemini-2.5-pro-exp-03-25:free",
    "voice_sales": "google/gemini-2.0-flash-exp:free",
    "optimization": "google/gemini-2.5-pro-exp-03-25:free",
}

# DeepSeek model for fallback and BrowsingAgent tasks
DEEPSEEK_MODEL = "deepseek/deepseek-r1:free"  

# SMTP settings for notifications (Hostinger-based)
HOSTINGER_EMAIL = os.getenv("HOSTINGER_EMAIL")
HOSTINGER_SMTP_PASS = os.getenv("HOSTINGER_SMTP_PASS")
HOSTINGER_SMTP = "smtp.hostinger.com"
SMTP_PORT = 587

# User email for notifications
USER_EMAIL = os.getenv("USER_EMAIL")

# Twilio configuration for WhatsApp and voice
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
USER_WHATSAPP_NUMBER = os.getenv("USER_WHATSAPP_NUMBER")

# Moroccan banking details
MOROCCAN_BANK_ACCOUNT = os.getenv("MOROCCAN_BANK_ACCOUNT")
MOROCCAN_SWIFT_CODE = os.getenv("MOROCCAN_SWIFT_CODE")
