# Filename: utils/notifications.py
# Description: Handles sending operational notifications (e.g., email to operator).
# Version: 1.2 (IGNIS Transmutation - Enhanced Logging & Robustness)

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
    logging.getLogger(__name__).warning("MailerSend SDK not found. Email notifications via MailerSend will be unavailable.")

logger = logging.getLogger(__name__)
op_logger = logging.getLogger('OperationalLog')


async def send_notification(title: str, message: str, level: str = "info", config: Optional[Any] = None):
    """
    Sends a notification, currently via email to the configured USER_EMAIL.
    Enhanced for robustness and clear logging.
    """
    cfg = config or settings
    user_email = cfg.get('USER_EMAIL')

    if not user_email:
        logger.warning(f"Cannot send notification '{title}': USER_EMAIL not configured in settings.")
        return

    mailersend_api_key = cfg.get_secret("MAILERSEND_API_KEY")
    sender_email = cfg.get("HOSTINGER_EMAIL") # Assuming this is the verified sender for MailerSend
    sender_name = cfg.get("SENDER_NAME", "Nolli AI Agency")

    log_entry = f"Notification: [{level.upper()}] Title: '{title}' Message: '{message[:100]}...'"
    logger.info(log_entry)
    op_logger.info(f"Attempting to send notification to operator ({user_email}): {title}")


    if MAILERSEND_AVAILABLE and mailersend_api_key and sender_email:
        logger.info(f"Attempting to send notification email via MailerSend to {user_email}")
        try:
            mailer = MailerSendEmails.NewEmail(mailersend_api_key)
            mail_body = {}
            # Ensure sender_name and sender_email are valid
            if not sender_name or not sender_email:
                logger.error("MailerSend: Sender name or email is missing. Cannot send notification.")
                return

            mail_from = {"name": sender_name, "email": sender_email}
            recipients = [{"email": user_email}]
            subject = f"[{level.upper()}] Nolli Notification: {title}"

            message_html_compatible = message.replace('\n', '<br>')
            text_content = f"Notification Level: {level.upper()}\n\n{message}"
            html_content = f"<h2>Nolli AI Agency Notification</h2><p><strong>Level: {level.upper()}</strong></p><p><strong>Title: {title}</strong></p><hr><p>{message_html_compatible}</p><hr><p><small>Timestamp: {datetime.now(timezone.utc).isoformat()}</small></p>"

            mailer.set_mail_from(mail_from, mail_body)
            mailer.set_mail_to(recipients, mail_body)
            mailer.set_subject(subject, mail_body)
            mailer.set_html_content(html_content, mail_body)
            mailer.set_plaintext_content(text_content, mail_body)

            response = await asyncio.to_thread(mailer.send, mail_body) # Run synchronous send in a thread

            if 200 <= response.status_code < 300:
                msg_id = response.headers.get('X-Message-Id', 'N/A')
                success_msg = f"Notification email sent successfully to {user_email} via MailerSend. MsgID: {msg_id}"
                logger.info(success_msg)
                op_logger.info(success_msg)
            else:
                error_details = "Unknown error"
                try:
                    error_details_json = await asyncio.to_thread(response.json) # Attempt to get JSON error
                    error_details = json.dumps(error_details_json)
                except: # Fallback to text if not JSON
                    error_details = await asyncio.to_thread(getattr, response, 'text', 'No error details available')
                
                fail_msg = f"Failed to send notification email via MailerSend. Status: {response.status_code}, Response: {error_details[:500]}"
                logger.error(fail_msg)
                op_logger.error(fail_msg)
        except Exception as e:
            logger.error(f"Error sending notification email via MailerSend: {e}", exc_info=True)
            op_logger.error(f"Exception sending notification email: {e}")
    else:
        logger.warning("MailerSend not available or not configured. Cannot send email notification.")
        op_logger.warning("MailerSend unavailable/unconfigured. Notification NOT sent by email.")

# --- End of utils/notifications.py ---