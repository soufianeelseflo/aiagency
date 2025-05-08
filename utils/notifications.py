# Filename: utils/notifications.py
# Description: Handles sending operational notifications (e.g., email to operator).
# Version: 1.2 (Level 50+ Transmutation - Enhanced Error Handling)

import logging
import asyncio
from typing import Optional, Any

# Import settings AFTER it's defined and validated
from config.settings import settings, SecretStr

# Import MailerSend client if available
try:
    from mailersend import emails as MailerSendEmails
    MAILERSEND_AVAILABLE = True
except ImportError:
    MAILERSEND_AVAILABLE = False
    logging.warning("MailerSend SDK not found. Email notifications disabled. Install with 'pip install mailersend'")

logger = logging.getLogger(__name__)

async def send_notification(title: str, message: str, level: str = "info", config: Optional[Any] = None):
    """
    Sends a notification, currently via email to the configured USER_EMAIL using MailerSend.
    """
    cfg = config or settings
    user_email = cfg.get('USER_EMAIL')

    if not user_email:
        logger.warning(f"Cannot send notification '{title}': USER_EMAIL not configured in settings.")
        return

    mailersend_api_key: Optional[SecretStr] = cfg.get_secret("MAILERSEND_API_KEY") # Use get_secret
    sender_email = cfg.get("HOSTINGER_EMAIL")
    sender_name = cfg.get("SENDER_NAME", "Nolli AI Agency")

    if MAILERSEND_AVAILABLE and mailersend_api_key and sender_email:
        logger.info(f"Attempting to send notification email via MailerSend to {user_email}")
        try:
            mailer = MailerSendEmails.NewEmail(mailersend_api_key.get_secret_value()) # Use get_secret_value()
            mail_body = {}
            mail_from = {"name": sender_name, "email": sender_email}
            recipients = [{"email": user_email}]
            subject = f"[{level.upper()}] {settings.APP_NAME} Notification: {title}"

            # Pre-process message for HTML compatibility
            message_html_compatible = message.replace('\n', '<br>')
            text_content = f"Notification Level: {level.upper()}\n\n{message}"
            html_content = f"<p><strong>Notification Level: {level.upper()}</strong></p><p>{message_html_compatible}</p>"

            mailer.set_mail_from(mail_from, mail_body)
            mailer.set_mail_to(recipients, mail_body)
            mailer.set_subject(subject, mail_body)
            mailer.set_html_content(html_content, mail_body)
            mailer.set_plaintext_content(text_content, mail_body)

            # Run blocking call in executor
            response = await asyncio.to_thread(mailer.send, mail_body)

            if 200 <= response.status_code < 300:
                msg_id = response.headers.get('X-Message-Id', 'N/A')
                logger.info(f"Notification email sent successfully to {user_email} via MailerSend. MsgID: {msg_id}")
            else:
                # Safely attempt to get response body/text
                error_details = "Unknown error details"
                try:
                    error_details = response.text if hasattr(response, 'text') else str(response.body) if hasattr(response, 'body') else 'No details'
                except Exception as read_err:
                    logger.warning(f"Could not read error details from MailerSend response: {read_err}")
                logger.error(f"Failed to send notification email via MailerSend. Status: {response.status_code}, Response: {error_details}")
        except Exception as e:
            logger.error(f"Error sending notification email via MailerSend: {e}", exc_info=True)
    else:
        if not MAILERSEND_AVAILABLE:
            logger.warning("MailerSend SDK not installed. Cannot send email notification.")
        else:
            logger.warning("MailerSend not available or configured (API key/Sender missing). Cannot send email notification.")

# --- End of utils/notifications.py ---