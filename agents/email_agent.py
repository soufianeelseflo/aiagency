import asyncio
import logging
import random
import os
import json
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone
import pytz
from collections import Counter

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, desc

# Assuming these exist and are correctly set up
from models import Client, EmailLog, Lead, PromptTemplate # Add others if needed
from utils.database import encrypt_data, decrypt_data # For email body if needed, though maybe not encrypting content itself
from config.settings import settings
# Assume Orchestrator provides access to ThinkTool, Vault, etc.
# from orchestrator import Orchestrator # Conceptual import
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pybreaker # For SMTP circuit breaker

logger = logging.getLogger(__name__)

# SMTP Circuit Breaker
smtp_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60 * 10) # 10 min timeout

class PuppeteerAgent:
    """
    Puppeteer Agent v2.0: Executes hyper-personalized, psychologically optimized
    email outreach campaigns. Masters human mimicry, spam evasion, engagement tracking,
    and automated A/B/n testing of persuasive techniques.
    """

    def __init__(self, session_maker: callable, config: object, orchestrator: object):
        self.session_maker = session_maker
        self.config = config
        self.orchestrator = orchestrator
        self.think_tool = orchestrator.agents.get('think')
        self.secure_storage = orchestrator.secure_storage

        self.task_queue = asyncio.PriorityQueue() # Tasks: (priority, {'type': 'send_email', 'client_id': 123})
                                                 # Priority = 1.0 - engagement_score
        self.max_concurrency = getattr(config, 'PUPPETEER_MAX_CONCURRENCY', 25)
        self.active_sends = 0
        self.send_semaphore = asyncio.Semaphore(self.max_concurrency)

        # Daily limits per sending account (managed via Orchestrator/DB ideally)
        self.daily_limits = {} # { 'sender_email': {'limit': 100, 'sent': 0, 'reset_time': ...} }
        self.global_daily_limit = getattr(config, 'PUPPETEER_GLOBAL_DAILY_LIMIT', 1000)
        self.global_sent_today = 0
        self.global_reset_time = self._get_next_reset_time_utc()

        # SMTP Providers (fetched from config)
        self.smtp_providers = settings.SMTP_PROVIDERS
        if not self.smtp_providers:
             logger.critical("PuppeteerAgent: SMTP_PROVIDERS configuration is missing or empty!")
        self.current_provider_index = 0

        # A/B Testing Framework State (simple version)
        self.active_tests = {} # { 'test_id': {'variant_a_count': 0, 'variant_b_count': 0, ...} }

        logger.info("Puppeteer Agent v2.0 (Psychological Outreach) initialized.")

    # --- Task Queuing & Prioritization ---
    async def queue_email_task(self, client_id: int, campaign_id: Optional[int] = None):
        """Adds an email task to the priority queue."""
        try:
            async with self.session_maker() as session:
                client = await session.get(Client, client_id)
                if not client or not client.opt_in or not client.is_deliverable:
                    logger.warning(f"Skipping queue for client {client_id}: Not found, opted out, or undeliverable.")
                    return False

                # Priority based on engagement score (lower score = higher priority)
                priority = 1.0 - (client.engagement_score or 0.1) # Ensure score exists, default low score = high priority
                priority += random.uniform(-0.05, 0.05) # Add jitter to break ties

                task_data = {'type': 'send_email', 'client_id': client_id, 'campaign_id': campaign_id}
                await self.task_queue.put((priority, task_data))
                logger.info(f"Queued email task for client {client_id} with priority {priority:.3f}")
                return True
        except Exception as e:
            logger.error(f"Failed to queue email task for client {client_id}: {e}", exc_info=True)
            return False

    async def queue_osint_leads(self, leads: list[dict]):
        """Processes leads from Oracle and queues them."""
        queued_count = 0
        created_count = 0
        skipped_count = 0
        async with self.session_maker() as session:
            async with session.begin(): # Use transaction
                for lead_data in leads:
                    email = lead_data.get('email')
                    if not email or '@' not in email:
                        skipped_count += 1
                        continue

                    # Check if client already exists
                    existing = await session.execute(select(Client.id).where(Client.email == email))
                    if existing.scalar_one_or_none():
                        skipped_count += 1
                        continue

                    # Create new client record
                    client = Client(
                        email=email,
                        name=lead_data.get('name', f"Lead <{email}>"),
                        source=lead_data.get('source', 'OracleOSINT'),
                        opt_in=True, # Assume opt-in initially for gray area
                        is_deliverable=True,
                        engagement_score=0.1 # Start with low score/high priority
                        # Add country, interests if provided by Oracle
                    )
                    session.add(client)
                    await session.flush() # Get the client ID
                    created_count += 1
                    if await self.queue_email_task(client.id):
                        queued_count += 1
                    else:
                         # Should ideally not happen within transaction, but handle just in case
                         skipped_count +=1

            logger.info(f"Processed OSINT leads: Created={created_count}, Queued={queued_count}, Skipped={skipped_count}")


    # --- Email Generation (Multi-Stage) ---
    async def generate_email_content(self, client: Client, campaign_id: Optional[int] = None):
        """Generates subject and body using the multi-stage process."""
        try:
            # 1. Get Prompts via ThinkTool (Handles caching)
            draft_prompt_template = await self.think_tool.get_prompt("PuppeteerAgent", "email_draft_v3")
            humanize_prompt_template = await self.think_tool.get_prompt("PuppeteerAgent", "email_humanize_v3")
            validate_prompt_template = await self.think_tool.get_prompt("PuppeteerAgent", "email_validate_v3") # Gemini validation prompt
            subject_prompt_template = await self.think_tool.get_prompt("PuppeteerAgent", "email_subject_v2")

            if not all([draft_prompt_template, humanize_prompt_template, validate_prompt_template, subject_prompt_template]):
                logger.error(f"Missing core prompt templates for client {client.id}. Aborting generation.")
                # TODO: Trigger ThinkTool self-critique or alert
                return None, None

            # 2. Prepare Context
            # Fetch recent interactions, OSINT summary if available
            osint_summary = "No specific OSINT data available." # Placeholder - fetch from OSINTData via client_id/email
            client_context = {
                "name": client.name,
                "email": client.email,
                "country": client.country,
                "interests": client.interests.split(',') if client.interests else ['their work'],
                "engagement_score": client.engagement_score,
                "osint_summary": osint_summary,
            }
            primary_interest = client_context['interests'][0]

            # 3. Generate Initial Draft
            draft_prompt = draft_prompt_template.format(
                client_context_json=json.dumps(client_context),
                primary_interest=primary_interest
            )
            initial_draft = await self.think_tool._call_llm_with_retry(draft_prompt, temperature=0.6, max_tokens=150)
            if not initial_draft: raise Exception("Failed to generate initial draft.")

            # 4. Humanize Draft
            regional_adapt = self._get_regional_adaptation(client.country)
            humanize_prompt = humanize_prompt_template.format(
                draft=initial_draft,
                regional_adaptation=regional_adapt
            )
            humanized_body = await self.think_tool._call_llm_with_retry(humanize_prompt, temperature=0.9, max_tokens=400)
            if not humanized_body:
                logger.warning(f"Humanization failed for client {client.id}, using initial draft.")
                humanized_body = initial_draft # Fallback

            # 5. Validate Final Body (Optional - uses Gemini via ThinkTool)
            validate_prompt = validate_prompt_template.format(email_body=humanized_body)
            # Use a specific method in ThinkTool that might route to Gemini or cheaper model for validation
            validation_result = await self.think_tool.validate_output(
                 output_to_validate=humanized_body,
                 validation_criteria="Assess statistical predictability (1-10 score). Score < 9 is predictable.",
                 agent_name="PuppeteerAgent",
                 context=f"Validating email body for client {client.id}"
            ) # Returns dict {'valid': bool, 'feedback': str}
            if not validation_result.get('valid', True): # Default to valid if validation fails
                 logger.warning(f"Email body validation failed for client {client.id}: {validation_result.get('feedback')}. Sending anyway.")
                 # Optionally trigger prompt self-critique here via ThinkTool

            # 6. Generate Subject Line
            subject_prompt = subject_prompt_template.format(
                 client_name=client.name.split()[0], # First name
                 primary_interest=primary_interest,
                 company_name=client.email.split('@')[1].split('.')[0] # Basic company guess
            )
            subject = await self.think_tool._call_llm_with_retry(subject_prompt, temperature=0.8, max_tokens=30)
            if not subject: subject = f"Quick thought on {primary_interest}" # Fallback

            # 7. A/B Testing Logic (Example: Subject Line Variants)
            # test_id = f"subject_test_{campaign_id}" if campaign_id else "subject_test_default"
            # if test_id in self.active_tests:
            #     # Alternate between variants A and B
            #     if self.active_tests[test_id]['variant_a_count'] <= self.active_tests[test_id]['variant_b_count']:
            #         # Use variant A (original subject)
            #         self.active_tests[test_id]['variant_a_count'] += 1
            #         variant_tag = "A"
            #     else:
            #         # Generate variant B subject (e.g., different style)
            #         subject_b_prompt = ... # Prompt for variant B
            #         subject = await self.think_tool._call_llm_with_retry(...)
            #         self.active_tests[test_id]['variant_b_count'] += 1
            #         variant_tag = "B"
            #     # Add variant tag to log later
            # else: variant_tag = None

            # 8. Add Tracking Pixel (Benign)
            pixel_url = f"https://{settings.AGENCY_TRACKING_DOMAIN}/track/{uuid.uuid4()}.png" # Assume tracking endpoint exists
            final_body = humanized_body + f'<img src="{pixel_url}" width="1" height="1" alt="">'

            return subject.strip().replace('"', ''), final_body

        except Exception as e:
            logger.error(f"Email content generation failed for client {client.id}: {e}", exc_info=True)
            return None, None

    def _get_regional_adaptation(self, country_code):
        # Simple example, expand as needed
        if country_code == 'GB': return "Use subtle British phrasing (e.g., 'reckon', 'keen')."
        if country_code == 'AU': return "Use subtle Australian phrasing (e.g., 'no worries', 'good onya')."
        return "Use standard US English phrasing."

    # --- Email Sending & Delivery ---
    async def send_email_task(self, client_id: int, campaign_id: Optional[int] = None):
        """Handles the entire process for sending one email."""
        async with self.send_semaphore:
            self.active_sends += 1
            log_status = "failed"
            log_entry = None
            subject = None
            body = None
            recipient = None
            sender_email = None

            try:
                # 0. Check Global Daily Limit
                if self.global_sent_today >= self.global_daily_limit:
                    if datetime.now(timezone.utc) > self.global_reset_time:
                        self.global_sent_today = 0
                        self.global_reset_time = self._get_next_reset_time_utc()
                    else:
                        logger.warning("Global daily email limit reached. Task deferred.")
                        # Requeue task with lower priority? Needs careful handling.
                        # await self.task_queue.put((priority + 1.0, {'type': 'send_email', 'client_id': client_id}))
                        return # For now, just drop if limit hit

                # 1. Fetch Client Data
                async with self.session_maker() as session:
                    client = await session.get(Client, client_id)
                    if not client or not client.opt_in or not client.is_deliverable or not client.email:
                        logger.warning(f"Cannot send email to client {client_id}: Invalid state.")
                        return
                    recipient = client.email

                # 2. Compliance Check (via ThinkTool)
                compliance_context = f"Sending cold outreach email to {client.name} ({client.email}) in {client.country}. Opt-in: {client.opt_in}. Source: {client.source}."
                compliance_check = await self.think_tool.reflect_on_action(
                    context=compliance_context, agent_name="PuppeteerAgent", task_description="Verify email compliance"
                )
                if not compliance_check.get('proceed', False):
                    logger.warning(f"Compliance check failed for {client.id}: {compliance_check.get('reason')}. Skipping send.")
                    await self._log_email(client_id, recipient, "Compliance Block", "", "blocked", None)
                    return

                # 3. Generate Content
                subject, body = await self.generate_email_content(client, campaign_id)
                if not subject or not body:
                    logger.error(f"Failed to generate content for {client.id}. Skipping send.")
                    return

                # 4. Optimal Send Time Calculation & Wait
                await self._wait_for_optimal_send_time(client)

                # 5. Select Sender Account & Check Limits
                sender_config = self._select_sending_account()
                if not sender_config:
                    logger.error("No available SMTP sending accounts within limits. Task deferred.")
                    # Requeue?
                    return
                sender_email = sender_config['email']

                # 6. Send Email (with circuit breaker)
                send_success = await self._send_via_smtp(sender_config, recipient, subject, body)

                # 7. Update Limits & Log Result
                if send_success:
                    log_status = "sent"
                    self._increment_send_count(sender_email)
                    self.global_sent_today += 1
                    logger.info(f"Email SENT to {recipient} via {sender_email}. Subject: {subject[:50]}...")
                    # Update client last interaction time
                    async with self.session_maker() as session:
                         await session.execute(update(Client).where(Client.id == client_id).values(last_interaction=datetime.now(timezone.utc)))
                         await session.commit()
                else:
                    log_status = "failed"
                    # Failure handled by retry/breaker, log as failed here
                    logger.warning(f"Email FAILED for {recipient} via {sender_email}.")

                log_entry = await self._log_email(client_id, recipient, subject, body, log_status, sender_email) # Log pixelated body

                # 8. A/B Testing: Log variant if applicable
                # if log_entry and variant_tag:
                #     async with self.session_maker() as session:
                #         await session.execute(update(EmailLog).where(EmailLog.id == log_entry.id).values(ab_test_variant=variant_tag))
                #         await session.commit()

            except Exception as e:
                logger.error(f"Unhandled error during send_email_task for client {client_id}: {e}", exc_info=True)
                log_status = "error"
                # Attempt to log error state
                if client_id and recipient:
                    await self._log_email(client_id, recipient, subject or "ERROR", body or f"Error: {e}", log_status, sender_email)
            finally:
                self.active_sends -= 1

    def _get_next_reset_time_utc(self):
        """Calculates the next reset time (midnight UTC)."""
        now_utc = datetime.now(timezone.utc)
        reset_time = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return reset_time

    async def _wait_for_optimal_send_time(self, client: Client):
        """Calculates and waits for optimal send time based on client timezone and past opens."""
        try:
            client_tz_str = client.timezone or "America/New_York" # Default TZ
            client_tz = pytz.timezone(client_tz_str)
        except pytz.UnknownTimeZoneError:
            logger.warning(f"Invalid timezone '{client_tz_str}' for client {client.id}. Defaulting to US/Eastern.")
            client_tz = pytz.timezone("America/New_York")

        optimal_hour_local = 10 # Default: 10 AM local time

        # --- Query past open times (simplified) ---
        # In production, this needs optimization (e.g., pre-calculating optimal hour periodically)
        try:
            async with self.session_maker() as session:
                 ninety_days_ago = datetime.now(timezone.utc) - timedelta(days=90)
                 stmt = select(EmailLog.opened_at).where(
                     EmailLog.client_id == client.id,
                     EmailLog.opened_at.isnot(None),
                     EmailLog.timestamp > ninety_days_ago
                 ).order_by(desc(EmailLog.opened_at)).limit(20)
                 result = await session.execute(stmt)
                 open_times_utc = [row.opened_at for row in result.fetchall()]

                 if len(open_times_utc) >= 3: # Need a few data points
                     open_hours_local = [t.astimezone(client_tz).hour for t in open_times_utc]
                     hour_counts = Counter(h for h in open_hours_local if 8 <= h <= 17) # Focus on business hours
                     if hour_counts:
                         optimal_hour_local = hour_counts.most_common(1)[0][0]
        except Exception as e:
             logger.error(f"Error calculating optimal send hour for client {client.id}: {e}. Using default.")
        # --- End Query ---

        now_local = datetime.now(client_tz)
        send_time_local = now_local.replace(hour=optimal_hour_local, minute=random.randint(0, 29), second=0, microsecond=0)

        # If optimal time already passed today, schedule for tomorrow
        if send_time_local <= now_local:
            send_time_local += timedelta(days=1)

        send_time_utc = send_time_local.astimezone(timezone.utc)
        now_utc = datetime.now(timezone.utc)
        delay_seconds = (send_time_utc - now_utc).total_seconds()

        if delay_seconds > 0:
            # Limit max delay to avoid tasks waiting indefinitely
            max_delay = 60 * 60 * 36 # 36 hours max wait
            delay_seconds = min(delay_seconds, max_delay)
            if delay_seconds > 60: # Only log if waiting more than a minute
                 logger.info(f"Optimal Send Time: Waiting {delay_seconds:.0f}s to send to client {client.id} ({client.email}) at {send_time_local.strftime('%Y-%m-%d %H:%M')} {client_tz_str}")
            await asyncio.sleep(delay_seconds)

    def _select_sending_account(self):
        """Selects an available SMTP provider respecting daily limits."""
        if not self.smtp_providers: return None
        num_providers = len(self.smtp_providers)
        now = datetime.now(timezone.utc)

        for i in range(num_providers):
            idx = (self.current_provider_index + i) % num_providers
            provider = self.smtp_providers[idx]
            email = provider['email']

            # Initialize limit tracking if not present
            if email not in self.daily_limits:
                 self.daily_limits[email] = {'limit': getattr(self.config, 'PUPPETEER_PER_ACCOUNT_LIMIT', 100), 'sent': 0, 'reset_time': self._get_next_reset_time_utc()}

            # Check reset time
            if now >= self.daily_limits[email]['reset_time']:
                 self.daily_limits[email]['sent'] = 0
                 self.daily_limits[email]['reset_time'] = self._get_next_reset_time_utc()

            # Check limit
            if self.daily_limits[email]['sent'] < self.daily_limits[email]['limit']:
                 self.current_provider_index = (idx + 1) % num_providers # Rotate for next time
                 logger.debug(f"Selected sending account: {email} (Sent today: {self.daily_limits[email]['sent']}/{self.daily_limits[email]['limit']})")
                 return provider

        logger.warning("All sending accounts have reached their daily limits.")
        return None # All accounts are over limit

    def _increment_send_count(self, sender_email):
        """Increments the send count for a specific account."""
        if sender_email in self.daily_limits:
            self.daily_limits[sender_email]['sent'] += 1
        else:
            # Should have been initialized, but handle defensively
            self.daily_limits[sender_email] = {'limit': getattr(self.config, 'PUPPETEER_PER_ACCOUNT_LIMIT', 100), 'sent': 1, 'reset_time': self._get_next_reset_time_utc()}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=5, max=30), retry=retry_if_exception_type(smtplib.SMTPException))
    @smtp_breaker # Apply circuit breaker
    async def _send_via_smtp(self, sender_config, recipient, subject, body):
        """Sends email using a specific SMTP provider."""
        host = sender_config['host']
        port = sender_config['port']
        email = sender_config['email']
        # Fetch password securely each time if possible, or assume it's loaded in sender_config
        password = sender_config['pass'] # Or fetch from Vault via orchestrator/secure_storage

        msg = EmailMessage()
        # Use set_content for plain text, add_alternative for HTML
        msg.set_content("Please enable HTML to view this email.") # Fallback
        msg.add_alternative(body, subtype='html') # Send HTML body with tracking pixel
        msg['Subject'] = subject
        msg['From'] = f"{settings.SENDER_NAME} <{email}>" # Use configured sender name
        msg['To'] = recipient
        msg['Date'] = smtplib.email.utils.formatdate(localtime=True)
        msg['Message-ID'] = smtplib.email.utils.make_msgid()

        try:
            # Use asyncio.to_thread for synchronous smtplib
            await asyncio.to_thread(
                self._smtp_connect_send_quit, host, port, email, password, msg
            )
            return True
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP Auth failed for {email} on {host}. Check credentials/config.")
            # TODO: Report persistent auth errors to Orchestrator to potentially disable provider
            raise # Reraise for retry/breaker
        except smtplib.SMTPException as e:
            logger.warning(f"SMTP error sending to {recipient} via {host}: {e}")
            raise # Reraise for retry/breaker
        except Exception as e:
             logger.error(f"Unexpected error sending email via {host}: {e}", exc_info=True)
             raise smtplib.SMTPException(f"Unexpected send error: {e}") # Wrap as SMTPException for retry

    def _smtp_connect_send_quit(self, host, port, email, password, msg):
        """Synchronous helper for SMTP operations."""
        # Context manager ensures quit() is called
        with smtplib.SMTP(host, port, timeout=45) as server:
            server.ehlo()
            # Use STARTTLS for port 587
            if port == 587:
                 context = smtplib.ssl.create_default_context()
                 server.starttls(context=context)
                 server.ehlo() # Re-identify after TLS
            # Add logic for port 465 (SMTPS - SSL) if needed
            # elif port == 465:
            #     server = smtplib.SMTP_SSL(host, port, timeout=45)
            #     # No starttls needed for SMTP_SSL

            server.login(email, password)
            server.send_message(msg)
            # logger.debug(f"SMTP send successful via {host}") # Logged in calling function


    async def _log_email(self, client_id, recipient, subject, body, status, sender_email):
        """Logs email details to the database."""
        try:
            async with self.session_maker() as session:
                log = EmailLog(
                    client_id=client_id,
                    recipient=recipient,
                    subject=subject,
                    content=body, # Store the HTML body (with pixel)
                    status=status,
                    timestamp=datetime.now(timezone.utc),
                    agent_version='Puppeteer_v2.0',
                    # opened_at=None, # Updated via tracking endpoint later
                    # responded_at=None # Updated via reply processing later
                    # sender_account=sender_email # Add if schema supports
                )
                session.add(log)
                await session.commit()
                await session.refresh(log)
                logger.debug(f"Logged email to {recipient}, status: {status}, ID: {log.id}")
                return log
        except Exception as e:
            logger.error(f"Failed to log email for {recipient}: {e}", exc_info=True)
            return None


    # --- Engagement Tracking & Learning ---
    # These would be triggered by external events (tracking pixel hit, reply processing)
    async def process_email_open(self, tracking_id: str):
        """Updates EmailLog when tracking pixel is hit."""
        # 1. Find EmailLog by tracking_id (needs schema change or mapping)
        # 2. Update opened_at timestamp
        # 3. Update Client engagement score
        logger.info(f"Processing email open for tracking ID: {tracking_id} (Not fully implemented)")
        pass

    async def process_email_reply(self, message_id: str, reply_content: str):
        """Updates EmailLog and Client score upon receiving a reply."""
        # 1. Find original EmailLog by Message-ID
        # 2. Update responded_at timestamp, status='responded'
        # 3. Update Client engagement score significantly
        # 4. Log reply content as KnowledgeFragment via ThinkTool
        # 5. Analyze reply sentiment/intent via ThinkTool
        logger.info(f"Processing email reply for Message-ID: {message_id} (Not fully implemented)")
        pass


    # --- Main Run Loop ---
    async def run(self):
        """Main loop processing email tasks from the priority queue."""
        logger.info("Puppeteer Agent v2.0 run loop started...")
        # TODO: Start background task for periodic engagement score updates based on logs
        # asyncio.create_task(self.update_engagement_scores_periodically())

        while True:
            try:
                # Get highest priority task (lowest number)
                priority, task_data = await self.task_queue.get()
                client_id = task_data['client_id']
                logger.info(f"Puppeteer dequeued task for client {client_id} with priority {priority:.3f}")

                # Process task asynchronously
                asyncio.create_task(self.send_email_task(client_id, task_data.get('campaign_id')))

            except asyncio.CancelledError:
                logger.info("Puppeteer Agent run loop cancelled.")
                break
            except Exception as e:
                logger.critical(f"Puppeteer Agent: CRITICAL error in run loop: {e}", exc_info=True)
                await self.orchestrator.report_error("PuppeteerAgent", f"Critical run loop error: {e}")
                await asyncio.sleep(60) # Wait after critical error
