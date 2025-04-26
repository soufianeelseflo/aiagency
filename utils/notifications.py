import smtplib
from email.message import EmailMessage
import logging
import asyncio # Added for to_thread
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type # Added for retry
import pybreaker # Added for circuit breaker

logger = logging.getLogger(__name__)

# Define a circuit breaker for SMTP operations
smtp_notification_breaker = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=60 * 5) # 5 min timeout

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(smtplib.SMTPException))
@smtp_notification_breaker
async def send_notification(subject: str, body: str, config, smtp_password: str):
    """
    Send email notifications with retry logic and circuit breaker.
    Accepts SMTP password directly instead of relying on config object for it.
    """
    sender_email = config.get("HOSTINGER_EMAIL")
    recipient_email = config.get("USER_EMAIL")
    smtp_host = config.get("HOSTINGER_SMTP")
    smtp_port = config.get("SMTP_PORT")
    # smtp_pass is now passed as an argument

    if not all([sender_email, recipient_email, smtp_host, smtp_port, smtp_password]):
        # Check for missing password specifically
        missing = [item for item, value in {
            "sender_email": sender_email, "recipient_email": recipient_email,
            "smtp_host": smtp_host, "smtp_port": smtp_port, "smtp_password": smtp_password
        }.items() if not value]
        error_msg = f"Notification failed: Missing SMTP configuration or password: {', '.join(missing)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        msg = EmailMessage()
        msg.set_content(body) # Set plain text content
        # Optionally add HTML alternative if body is HTML: msg.add_alternative(body, subtype='html')
        msg['Subject'] = subject
        msg['From'] = sender_email # Use configured sender email
        msg['To'] = recipient_email
        msg['Date'] = smtplib.email.utils.formatdate(localtime=True)
        msg['Message-ID'] = smtplib.email.utils.make_msgid()

        # Use asyncio.to_thread for synchronous smtplib operations
        await asyncio.to_thread(
            _smtp_send,
            smtp_host,
            smtp_port,
            sender_email,
            smtp_password, # Pass the argument here
            msg
        )
        logger.info(f"Notification sent to {recipient_email}: {subject}")
    except Exception as e:
        logger.error(f"Failed to send notification '{subject}' to {recipient_email}: {e}", exc_info=True)
        # Re-raise wrapped as SMTPException if not already, to ensure tenacity retries SMTP-related issues
        if not isinstance(e, smtplib.SMTPException):
             raise smtplib.SMTPException(f"Notification sending failed: {e}")
        else:
             raise # Re-raise original SMTPException

def _smtp_send(host, port, sender_email, sender_pass, msg):
    """Synchronous helper function to perform SMTP operations."""
    # Context manager ensures server.quit() is called
    logger.debug(f"Connecting to SMTP: {host}:{port}")
    with smtplib.SMTP(host, port, timeout=30) as server: # Added timeout
        server.set_debuglevel(0) # Set to 1 for detailed SMTP logs if needed
        # Use STARTTLS for port 587 (common practice)
        if port == 587:
            logger.debug("Issuing STARTTLS")
            server.starttls()
            logger.debug("STARTTLS successful")
        # Add logic here if port 465 (SMTPS/SSL) is used - requires smtplib.SMTP_SSL
        # elif port == 465:
        #    # server = smtplib.SMTP_SSL(...) # Requires different class
        #    pass
        logger.debug(f"Logging in as {sender_email}")
        server.login(sender_email, sender_pass)
        logger.debug("SMTP Login successful")
        server.send_message(msg)
        logger.debug(f"Message sent successfully via {host}")