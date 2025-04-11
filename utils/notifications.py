import smtplib
from email.message import EmailMessage
import logging

logger = logging.getLogger(__name__)

async def send_notification(subject, body, config):
    """Send email notifications with retry logic."""
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = config.get("HOSTINGER_EMAIL")
        msg['To'] = config.get("USER_EMAIL")
        with smtplib.SMTP(config.get("HOSTINGER_SMTP"), config.get("SMTP_PORT")) as server:
            server.starttls()
            server.login(config.get("HOSTINGER_EMAIL"), config.get("HOSTINGER_SMTP_PASS"))
            server.send_message(msg)
        logger.info(f"Notification sent: {subject}")
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        raise   