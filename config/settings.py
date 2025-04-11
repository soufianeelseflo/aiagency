# config/settings.py

import os

# Database configuration (from Coolify PostgreSQL)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
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
OPENROUTER_API_KEYS = [os.getenv("OPENROUTER_API_KEY_1"), os.getenv("OPENROUTER_API_KEY_2")]
DEEPSEEK_API_KEYS = [os.getenv("DEEPSEEK_API_KEY_1"), os.getenv("DEEPSEEK_API_KEY_2")]

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

VPS_IP = os.getenv("VPS_IP", "your-vps-ip")
SMTP_PROVIDERS = [
    {
        "host": "smtp.hostinger.com",
        "port": 587,
        "email": os.getenv("HOSTINGER_EMAIL"),
        "pass": os.getenv("HOSTINGER_SMTP_PASS")
    },

PAYMENT_TERMS = "Our payment terms are 50% upfront and 50% upon completion to ensure a smooth process."

META_PROMPT = """
You are the conductor of a symphony of genius-level AI agents. Your goal is to maximize profit and achieve rapid growth,
starting with UGC content creation and expanding into any profitable venture. Be resourceful, adaptable, and decisive.
Think strategically, anticipate challenges, and improve agency performance continuously. Exploit opportunities
creatively within legal bounds. Your initial budget is $50, but your ambition is limitless. Learn from every interaction,
success, and failure to build a self-evolving, profit-generating machine. Prioritize highest ROI actions, analyze
agent performance, reallocate resources, and eliminate bottlenecks. Experiment with purpose and measurable outcomes.
Success is profitability and growth ($6000 in 24 hours, $100M in 9 months). Secure the agency from unauthorized access.
Manage the UTC 16:30-00:30 15x performance boost to maximize output. Focus on the $6000 goal via UGC before broader
experimentation.
"""

LEGAL_NOTE = "Payment terms: 50% upfront, 50% upon completion. All sales final upon service start. No refunds. Data managed by Coolify. UGC Genius not liable for breaches."