# Filename: utils/notifications.py
# Description: Handles sending operational notifications (e.g., email to operator).
# Version: 1.1 (SyntaxError Fix for f-string backslash)

import logging
import asyncio
from typing import Optional, Any

# Import settings AFTER it's defined and validated
from config.settings import settings

# Import MailerSend client if available
try:
    from mailersend import emails as MailerSendEmails
    MAILERSEND_AVAILABLE = True
except ImportError:
    MAILERSEND_AVAILABLE = False

logger = logging.getLogger(__name__)

async def send_notification(title: str, message: str, level: str = "info", config: Optional[Any] = None):
    """
    Sends a notification, currently via email to the configured USER_EMAIL.
    """
    cfg = config or settings
    user_email = cfg.get('USER_EMAIL')

    if not user_email:
        logger.warning(f"Cannot send notification '{title}': USER_EMAIL not configured in settings.")
        return

    mailersend_api_key = cfg.get_secret("MAILERSEND_API_KEY")
    sender_email = cfg.get("HOSTINGER_EMAIL")
    sender_name = cfg.get("SENDER_NAME", "Synapse AI Agency")

    if MAILERSEND_AVAILABLE and mailersend_api_key and sender_email:
        logger.info(f"Attempting to send notification email via MailerSend to {user_email}")
        try:
            mailer = MailerSendEmails.NewEmail(mailersend_api_key)
            mail_body = {}
            mail_from = {"name": sender_name, "email": sender_email}
            recipients = [{"email": user_email}]
            subject = f"[{level.upper()}] Synapse Notification: {title}"
            
            # --- SYNTAX FIX IS HERE ---
            # Pre-process the message for HTML replacement outside the f-string expression part
            message_html_compatible = message.replace('\n', '<br>')
            text_content = f"Notification Level: {level.upper()}\n\n{message}"
            html_content = f"<p><strong>Notification Level: {level.upper()}</strong></p><p>{message_html_compatible}</p>"
            # --- END OF SYNTAX FIX ---

            mailer.set_mail_from(mail_from, mail_body)
            mailer.set_mail_to(recipients, mail_body)
            mailer.set_subject(subject, mail_body)
            mailer.set_html_content(html_content, mail_body)
            mailer.set_plaintext_content(text_content, mail_body)

            response = await asyncio.to_thread(mailer.send, mail_body)

            if 200 <= response.status_code < 300:
                msg_id = response.headers.get('X-Message-Id', 'N/A')
                logger.info(f"Notification email sent successfully to {user_email} via MailerSend. MsgID: {msg_id}")
            else:
                error_details = await asyncio.to_thread(getattr, response, 'text', 'No details')
                logger.error(f"Failed to send notification email via MailerSend. Status: {response.status_code}, Response: {error_details}")
        except Exception as e:
            logger.error(f"Error sending notification email via MailerSend: {e}", exc_info=True)
    else:
        logger.warning("MailerSend not available or configured. Cannot send email notification.")
# --- End of utils/notifications.py ---