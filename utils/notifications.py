# Filename: utils/notifications.py
# Description: Handles sending operational notifications (e.g., email to operator).
# Version: 2.0 (IGNIS Final Transmutation - Production Grade, Async MailerSend, Full Configuration via Settings)

import logging
import asyncio
import json # For parsing MailerSend error responses
from typing import Optional, Any
from datetime import datetime, timezone

# --- Settings Import ---
# This mechanism ensures that if settings.py has an issue or is missing,
# a fallback is used, but errors are logged, and functionality might be impaired.
# The primary expectation is that config.settings is correctly configured and available.
try:
    from config.settings import settings
    SETTINGS_AVAILABLE = True
except ImportError as e_settings_import:
    # Fallback for critical failure of settings import
    class DummySettingsProvider:
        USER_EMAIL: Optional[str] = None
        MAILERSEND_API_KEY: Optional[str] = None
        HOSTINGER_EMAIL: Optional[str] = None # This should be a verified sender for MailerSend
        SENDER_NAME: str = "Nolli AI Agency (Config Error)"
        DEBUG: bool = True # Default to debug if settings are broken to get more logs

        def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
            return getattr(self, key.upper(), default)

        def get_secret(self, key: str) -> Optional[str]:
            return getattr(self, key.upper(), None)

    settings = DummySettingsProvider() # type: ignore
    SETTINGS_AVAILABLE = False
    # Log this critical failure immediately
    critical_msg = f"CRITICAL FAILURE: config.settings module could not be imported in utils/notifications.py. Cause: {e_settings_import}. Notifications will likely fail or be severely limited."
    logging.basicConfig(level=logging.ERROR) # Ensure basicConfig is called if not already
    logging.getLogger(__name__).critical(critical_msg)


# --- MailerSend SDK Import ---
# This allows the system to function (log-only notifications) if MailerSend isn't installed,
# but logs a clear warning.
try:
    from mailersend import emails as MailerSendEmails
    MAILERSEND_SDK_AVAILABLE = True
except ImportError:
    MAILERSEND_SDK_AVAILABLE = False
    # Log this at warning level as it's a missing dependency for a feature.
    logging.getLogger(__name__).warning(
        "MailerSend SDK (mailersend package) not found. "
        "Email notifications via MailerSend will be unavailable. "
        "Please install 'mailersend' if email notifications are required."
    )

logger = logging.getLogger(__name__)
op_logger = logging.getLogger('OperationalLog') # Dedicated logger for operational events

async def send_notification(title: str, message: str, level: str = "info") -> None:
    """
    Sends a notification. Primarily attempts email via MailerSend if configured.
    Always logs to the operational log.
    `level` corresponds to logging levels (e.g., "info", "warning", "error", "critical").
    """
    # Determine the logging level method
    log_method = getattr(op_logger, level.lower(), op_logger.info)
    
    # Log the notification event to operational log first and foremost
    operational_log_message = f"Notification Event: Level='{level.upper()}', Title='{title}', Message='{message[:250]}...'"
    log_method(operational_log_message)

    # Proceed to attempt email notification if possible
    if not SETTINGS_AVAILABLE:
        logger.error(
            f"Email notification for '{title}' aborted: Settings module was not available during initialization. "
            "This is a critical configuration error."
        )
        return

    user_email_to_notify: Optional[str] = settings.get("USER_EMAIL")
    if not user_email_to_notify:
        logger.warning(
            f"Email notification for '{title}' cannot be sent: USER_EMAIL is not configured in settings. "
            "The notification has been logged operationally."
        )
        return

    mailersend_api_key: Optional[str] = settings.get_secret("MAILERSEND_API_KEY")
    # HOSTINGER_EMAIL is assumed to be the verified sender email for MailerSend
    sender_email_address: Optional[str] = settings.get("HOSTINGER_EMAIL")
    sender_display_name: str = settings.get("SENDER_NAME", "Nolli AI Agency")

    if not MAILERSEND_SDK_AVAILABLE:
        logger.warning(
            f"Email notification for '{title}' to {user_email_to_notify} skipped: MailerSend SDK is not installed. "
            "The notification has been logged operationally."
        )
        return

    if not mailersend_api_key:
        logger.warning(
            f"Email notification for '{title}' to {user_email_to_notify} skipped: MAILERSEND_API_KEY is not configured. "
            "The notification has been logged operationally."
        )
        return

    if not sender_email_address:
        logger.warning(
            f"Email notification for '{title}' to {user_email_to_notify} skipped: Sender email (e.g., HOSTINGER_EMAIL) "
            "for MailerSend is not configured. The notification has been logged operationally."
        )
        return

    logger.info(f"Attempting to send '{level.upper()}' email notification titled '{title}' to {user_email_to_notify} via MailerSend.")

    try:
        mailer = MailerSendEmails.NewEmail(mailersend_api_key)
        
        mail_body_params = {} # MailerSend SDK expects a dict for mail_body
        
        mail_from_details = {
            "name": sender_display_name,
            "email": sender_email_address,
        }
        
        recipients_list = [{"email": user_email_to_notify}]
        
        email_subject = f"[{level.upper()}] Nolli AI Notification: {title}"
        
        # Prepare content (ensure HTML compatibility for message)
        message_html_compatible = message.replace('\n', '<br>')
        current_timestamp_iso = datetime.now(timezone.utc).isoformat()

        text_content = (
            f"Nolli AI Agency Notification\n"
            f"Level: {level.upper()}\n"
            f"Title: {title}\n"
            f"Timestamp: {current_timestamp_iso}\n\n"
            f"{message}"
        )
        
        html_content = (
            f"<html><body style='font-family: Arial, sans-serif; color: #333;'>"
            f"<div style='border: 1px solid #ddd; padding: 20px; max-width: 600px; margin: auto;'>"
            f"<h2 style='color: #0056b3;'>Nolli AI Agency Notification</h2>"
            f"<p style='font-size: 1.1em;'><strong>Level: <span style='color: {('red' if level in ['error', 'critical'] else 'orange' if level == 'warning' else 'green')}'>{level.upper()}</span></strong></p>"
            f"<p style='font-size: 1.1em;'><strong>Title: {title}</strong></p>"
            f"<p style='font-size: 0.9em; color: #777;'>Timestamp: {current_timestamp_iso}</p>"
            f"<hr style='border: 0; border-top: 1px solid #eee; margin: 20px 0;'>"
            f"<div style='font-size: 1em; line-height: 1.6;'>{message_html_compatible}</div>"
            f"<hr style='border: 0; border-top: 1px solid #eee; margin: 20px 0;'>"
            f"<p style='font-size: 0.8em; color: #aaa;'>This is an automated notification from the Nolli AI Agency system.</p>"
            f"</div></body></html>"
        )

        mailer.set_mail_from(mail_from_details, mail_body_params)
        mailer.set_mail_to(recipients_list, mail_body_params)
        mailer.set_subject(email_subject, mail_body_params)
        mailer.set_html_content(html_content, mail_body_params)
        mailer.set_plaintext_content(text_content, mail_body_params)

        # MailerSend's send method is synchronous, so run it in a thread
        # to avoid blocking the asyncio event loop.
        response = await asyncio.to_thread(mailer.send, mail_body_params)

        # MailerSend API typically returns 202 Accepted on success
        if response and 200 <= response.status_code < 300: # Check for response presence
            message_id_header = response.headers.get('X-Message-Id', 'N/A (Header Missing)')
            success_message = (
                f"Email notification '{title}' sent successfully to {user_email_to_notify} via MailerSend. "
                f"Status: {response.status_code}. Message ID: {message_id_header}"
            )
            logger.info(success_message)
            op_logger.info(f"Email Delivery Success: '{title}' to {user_email_to_notify}. MsgID: {message_id_header}")
        else:
            error_details = "Unknown error or no response object."
            status_code_info = "N/A"
            if response:
                status_code_info = str(response.status_code)
                try:
                    # Attempt to parse JSON error response if available
                    error_json = await asyncio.to_thread(response.json)
                    error_details = json.dumps(error_json)
                except Exception:
                    # Fallback to text content if not JSON or other error
                    error_details = await asyncio.to_thread(getattr, response, 'text', 'No error details available from response.text')
            
            failure_message = (
                f"Failed to send email notification '{title}' to {user_email_to_notify} via MailerSend. "
                f"Status Code: {status_code_info}. Response: {error_details[:500]}..." # Truncate long responses
            )
            logger.error(failure_message)
            op_logger.error(f"Email Delivery Failure: '{title}' to {user_email_to_notify}. Status: {status_code_info}. Details: {error_details[:200]}...")

    except Exception as e:
        # Catch any other exceptions during the process
        exception_message = (
            f"An unexpected error occurred while sending email notification '{title}' to {user_email_to_notify} via MailerSend: {e}"
        )
        logger.error(exception_message, exc_info=settings.get("DEBUG", False)) # Show exc_info if in debug
        op_logger.error(f"Email Sending Exception: '{title}' to {user_email_to_notify}. Error: {e}")

# --- End of utils/notifications.py ---