# Filename: agents/email_agent.py
# Description: Genius Agentic Email Agent - Handles hyper-personalized outreach,
#              IMAP opt-out processing, humanization, compliance, and learning.
# Version: 3.1 (Added Enriched Data Usage)

import asyncio
import logging
import random
import os
import json
import smtplib
import imaplib
import email # For parsing emails
import re # For parsing email content/headers
import uuid # For tracking pixel
import hashlib # For caching/hashing
import time # For caching TTL
from email.message import EmailMessage
from email.header import decode_header
from datetime import datetime, timedelta, timezone
import pytz
from collections import Counter
import pybreaker # For SMTP circuit breaker

# --- Core Framework Imports ---
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy import update, desc, func, case, String, cast # Import necessary SQLAlchemy components
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import SQLAlchemyError

# --- Project Imports ---
try:
    from .base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
except ImportError:
    logging.warning("Production base agent not found, using GeniusAgentBase. Ensure base_agent_prod.py is used.")
    # Attempt absolute import if relative fails (common in some setups)
    try:
        from agents.base_agent import GeniusAgentBase
    except ImportError:
        logging.critical("Failed to import GeniusAgentBase from both relative and absolute paths.")
        raise

# Use correct model imports as defined in your models.py
from models import Client, EmailLog, PromptTemplate, KnowledgeFragment, EmailComposition, LearnedPattern, EmailStyles, KVStore
from config.settings import settings # Use validated settings
from utils.database import encrypt_data, decrypt_data # Use DB utils
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl
# Assuming LLM Client access via orchestrator
# from openai import AsyncOpenAI as AsyncLLMClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from bs4 import BeautifulSoup # For HTML to text conversion
from typing import Dict, Any, Optional, List, Union, Tuple, Type # Ensure typing is imported

# Configure logger
logger = logging.getLogger(__name__)
# Configure dedicated operational logger
op_logger = logging.getLogger('OperationalLog') # Assuming setup elsewhere

# --- Meta Prompt ---
EMAIL_AGENT_META_PROMPT = """
You are the EmailAgent within the Synapse AI Sales System.
Your Core Mandate: Execute hyper-personalized, psychologically optimized, and compliant email outreach campaigns to maximize profitable conversions ($10k+/day goal).
Key Responsibilities:
1.  **Hyper-Personalized Content:** Generate human-like, engaging email subjects and bodies using context from ThinkTool (Client data, OSINT, KB insights, learned styles, **enriched Clay data**). Use LLM self-critique for humanization.
2.  **Compliant Outreach:** Strictly adhere to CAN-SPAM/GDPR/CASL. Include physical address, clear opt-out mechanism ("Reply STOP"). Validate campaigns via LegalAgent before sending.
3.  **Deliverability & Anti-Spam:** Analyze content for spam triggers. Utilize multiple SMTP accounts with rotation and rate limiting. Monitor bounce/spam rates.
4.  **Engagement Tracking:** Embed tracking pixels (via Orchestrator). Process replies via IMAP, specifically identifying "STOP" requests for immediate opt-out.
5.  **Performance Logging:** Log all email sends, opens, replies, bounces, and failures meticulously to the Postgres database (`EmailLog`). Link sent emails to KB fragments used (`EmailComposition`).
6.  **Learning Integration:** Utilize successful email styles (`EmailStyles` table) identified by ThinkTool. Provide performance data for ThinkTool's learning loop.
7.  **Collaboration:** Receive tasks (leads, content briefs, **enriched data**) from Orchestrator (originating from ThinkTool). Request validation from LegalAgent. Report outcomes and errors.
**Goal:** Drive high-value engagement (replies, calls booked, sales) through intelligent, compliant, and adaptive email marketing leveraging all available data, including Clay enrichments.
"""

# SMTP Circuit Breaker
smtp_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60 * 10, name="EmailAgentSMTP")

class EmailAgent(GeniusAgentBase):
    """
    Email Agent (Genius Level): Executes hyper-personalized, compliant email campaigns,
    handles opt-outs via IMAP, learns from performance, and manages SMTP rotation.
    Version: 3.1 (Added Enriched Data Usage)
    """
    AGENT_NAME = "EmailAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any, smtp_password: str, imap_password: str):
        """Initializes the EmailAgent."""
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, session_maker=session_maker)
        self.meta_prompt = EMAIL_AGENT_META_PROMPT
        self.think_tool = orchestrator.agents.get('think') # Reference ThinkTool if needed

        # Store secrets passed directly
        self._smtp_password = smtp_password
        self._imap_password = imap_password

        # --- Internal State Initialization ---
        self.internal_state = getattr(self, 'internal_state', {})
        self.internal_state['max_concurrency'] = int(self.config.get("EMAIL_AGENT_MAX_CONCURRENCY", 25))
        self.internal_state['send_semaphore'] = asyncio.Semaphore(self.internal_state['max_concurrency'])
        self.internal_state['daily_limits'] = {} # sender_email -> {'limit': N, 'sent': N, 'reset_time': datetime}
        self.internal_state['global_daily_limit'] = int(self.config.get("EMAIL_AGENT_MAX_PER_DAY", 1000))
        self.internal_state['global_sent_today'] = 0
        self.internal_state['global_reset_time'] = self._get_next_reset_time_utc()
        self.internal_state['smtp_providers'] = self._load_smtp_providers() # Load provider structures
        self.internal_state['current_provider_index'] = 0
        self.internal_state['imap_check_interval_seconds'] = int(self.config.get("IMAP_CHECK_INTERVAL_S", 300))
        self.internal_state['tracking_pixel_cache'] = {} # uuid -> email_log_id (simple in-memory cache)

        self.logger.info(f"{self.AGENT_NAME} v3.1 initialized. Max Concurrency: {self.internal_state['max_concurrency']}")

    def _load_smtp_providers(self) -> List[Dict]:
        """Loads SMTP provider details from config, adding the password."""
        providers = []
        hostinger_email = self.config.get("HOSTINGER_EMAIL")
        hostinger_smtp = self.config.get("HOSTINGER_SMTP_HOST") # Use correct setting name
        smtp_port = self.config.get("SMTP_PORT")

        if hostinger_email and hostinger_smtp and self._smtp_password:
            providers.append({
                "host": hostinger_smtp,
                "port": smtp_port,
                "email": hostinger_email,
                "pass": self._smtp_password
            })
            self.logger.info(f"Loaded SMTP provider: {hostinger_email}")
        else:
            self.logger.critical("Hostinger SMTP email, host, or password missing in config/secrets. Email sending will fail.")
        return providers

    async def log_operation(self, level: str, message: str):
        """Helper to log to the operational log file."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err: logger.error(f"Failed to write to operational log: {log_err}")

    # --- Task Execution ---
    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handles email-related tasks delegated by Orchestrator."""
        action = task_details.get('action')
        self.logger.info(f"{self.AGENT_NAME} received task: {action} with details: {list(task_details.keys())}")
        self._status = self.STATUS_EXECUTING
        result = {"status": "failure", "message": f"Unsupported action: {action}"}

        try:
            # Updated to handle 'initiate_outreach' which includes enriched_data
            if action == 'initiate_outreach' or action == 'send_email':
                client_id = task_details.get('client_id')
                # Use target_identifier from content if available
                target_identifier = task_details.get('content', {}).get('target_identifier')
                campaign_id = task_details.get('content', {}).get('campaign_id')
                enriched_data = task_details.get('content', {}).get('enriched_data')
                goal = task_details.get('content', {}).get('goal', 'engagement') # Pass goal

                if not client_id and not target_identifier:
                     result["message"] = "Missing client_id or target_identifier for outreach task."
                     self.logger.error(result["message"])
                else:
                    # Run the full send process in the background, passing context correctly
                    asyncio.create_task(self.send_email_task(client_id=client_id,
                                                             target_identifier=target_identifier,
                                                             campaign_id=campaign_id,
                                                             enriched_data=enriched_data,
                                                             goal=goal))
                    result = {"status": "success", "message": "Email outreach task initiated."}

            elif action == 'process_imap_replies':
                await self._check_imap_for_replies()
                result = {"status": "success", "message": "IMAP check completed."}
            elif action == 'process_open_tracking':
                tracking_id = task_details.get('tracking_id')
                if tracking_id:
                    await self.process_email_open(tracking_id)
                    result = {"status": "success", "message": "Open tracking processed."}
                else:
                    result["message"] = "Missing tracking_id for open tracking."
            else:
                self.logger.warning(f"Unsupported action '{action}' for EmailAgent.")

        except Exception as e:
            self.logger.error(f"Error executing EmailAgent task '{action}': {e}", exc_info=True)
            result = {"status": "error", "message": f"Unexpected error: {e}"}
            await self._report_error(f"Task '{action}' failed: {e}")
        finally:
            self._status = self.STATUS_IDLE

        return result

    # --- Core Email Workflow ---
    # MODIFICATION START: Added goal parameter
    async def send_email_task(self,
                              client_id: Optional[int],
                              target_identifier: Optional[str],
                              campaign_id: Optional[int] = None,
                              enriched_data: Optional[Dict[str, Any]] = None,
                              goal: str = 'engagement'): # Pass goal
        # MODIFICATION END
        """Handles the entire process for sending one email, including internal reflection and throttling."""
        async with self.internal_state['send_semaphore']:
            self.internal_state['active_sends'] = self.internal_state.get('active_sends', 0) + 1
            log_status = "failed_preparation"
            message_id = None
            subject: Optional[str] = None
            body: Optional[str] = None
            recipient: Optional[str] = None
            actual_client_id: Optional[int] = client_id
            sender_email: Optional[str] = None
            composition_ids: Dict[str, Any] = {}
            email_log_id: Optional[int] = None
            client: Optional[Client] = None

            try:
                # 0. Check Global Daily Limit
                if not self._check_global_limit():
                    self.logger.warning(f"Global daily email limit reached. Task deferred for client {actual_client_id or target_identifier}.")
                    # Requeue with lower priority, include goal
                    await self.orchestrator.delegate_task(self.AGENT_NAME, {'action': 'initiate_outreach', 'content': {'client_id': actual_client_id, 'target_identifier': target_identifier, 'campaign_id': campaign_id, 'enriched_data': enriched_data, 'goal': goal}, 'priority': 9.0})
                    return

                # 1. Fetch Client Data & Check Opt-In
                async with self.session_maker() as session:
                    if actual_client_id:
                        client = await session.get(Client, actual_client_id)
                        if not client:
                             self.logger.warning(f"Client ID {actual_client_id} provided but not found.")
                    # Try lookup by email if no client or ID missing
                    if not client and target_identifier and '@' in target_identifier:
                         stmt = select(Client).where(Client.email == target_identifier).limit(1)
                         client = (await session.execute(stmt)).scalar_one_or_none()
                         if client:
                             actual_client_id = client.id
                             self.logger.info(f"Found client ID {actual_client_id} via email lookup for {target_identifier}.")
                         else:
                             self.logger.info(f"No client record found for email {target_identifier}. Proceeding with email as recipient.")

                    # Determine recipient email
                    recipient = client.email if client else target_identifier

                    if not recipient:
                         self.logger.error(f"Cannot send email: No recipient email address identified for client {actual_client_id or target_identifier}.")
                         return

                    # Check client state if client object exists
                    if client and (not client.opt_in or not client.is_deliverable):
                        self.logger.warning(f"Cannot send email to client {actual_client_id}: Invalid state (OptIn:{client.opt_in}, Deliverable:{client.is_deliverable}).")
                        return

                # 2. Select Sender Account & Check Account Limit
                sender_config = self._select_sending_account()
                if not sender_config:
                    self.logger.error(f"No available SMTP sending accounts within limits for client {actual_client_id or target_identifier}. Task deferred.")
                    # Requeue, include goal
                    await self.orchestrator.delegate_task(self.AGENT_NAME, {'action': 'initiate_outreach', 'content': {'client_id': actual_client_id, 'target_identifier': target_identifier, 'campaign_id': campaign_id, 'enriched_data': enriched_data, 'goal': goal}, 'priority': 8.0})
                    return
                sender_email = sender_config['email']

                # 3. Generate Content (Passing client object, enriched_data, and goal)
                await self._internal_think(f"Requesting content generation for {recipient}.")
                # MODIFICATION START: Pass goal
                subject, body, composition_ids = await self._generate_email_content_internal(client, campaign_id, enriched_data, goal) # Pass goal
                # MODIFICATION END

                if not subject or not body:
                    raise ValueError("Failed to generate email content.")

                # 4. Compliance Check (via LegalAgent)
                await self._internal_think(f"Requesting compliance validation for email to {recipient}.")
                compliance_context = f"Email Campaign Send: To={recipient}, Subject='{subject[:50]}...', Body Snippet='{self._html_to_plain_text(body)[:100]}...', Client Country: {client.country if client else 'Unknown'}"
                validation_result = await self.orchestrator.delegate_task("LegalAgent", {"action": "validate_operation", "operation_description": compliance_context})
                if not validation_result or validation_result.get('status') != 'success' or not validation_result.get('findings', {}).get('is_compliant'):
                    reason = validation_result.get('findings', {}).get('compliance_issues', ['Validation Failed'])[0] if validation_result else 'Validation Error'
                    self.logger.warning(f"Compliance check failed for email to {recipient}. Reason: {reason}. Skipping send.")
                    await self._log_email(actual_client_id, recipient, subject, body, "blocked_compliance", sender_email, composition_ids, None)
                    return

                # 5. Add Compliance Footer & Tracking Pixel
                company_address = self.config.SENDER_COMPANY_ADDRESS or "[Your Company Physical Address - Configure SENDER_COMPANY_ADDRESS]"
                footer = f"\n\n---\n{self.config.SENDER_NAME}\n{self.config.SENDER_TITLE}\n{company_address}\nReply 'STOP' to unsubscribe."
                tracking_pixel_uuid = uuid.uuid4()
                pixel_base_url = self.config.AGENCY_BASE_URL.rstrip('/')
                pixel_url = f"{pixel_base_url}/track/{tracking_pixel_uuid}.png" # Correct endpoint
                body += f'<img src="{pixel_url}" width="1" height="1" alt="" style="display:none;"/>'
                body += f"<p style='font-size:10px; color:#888;'>{footer.replace('\n', '<br>')}</p>"

                # 6. Optimal Send Time Calculation & Wait (If client exists)
                if client:
                    await self._wait_for_optimal_send_time(client)
                else:
                    self.logger.debug(f"No client object for {recipient}, sending immediately.")

                # --- Internal Reflection Step ---
                pre_send_thought = f"Pre-Send Checklist: ClientID={actual_client_id}, To={recipient}, Subject='{subject[:30]}...', Compliance=OK, Sender={sender_email}, Limits OK. Action: Proceed with SMTP send."
                await self._internal_think(pre_send_thought)

                # 7. Apply Throttling
                await self._apply_throttling()

                # 8. Send Email via SMTP
                send_result = await self._send_email_smtp(recipient, subject, body, sender_config)
                send_success = send_result.get("status") == "success"
                message_id = send_result.get("message_id")

                # 9. Update Limits & Log Result
                if send_success:
                    log_status = "sent"
                    self._increment_send_count(sender_email)
                    self.internal_state['global_sent_today'] += 1
                    self.logger.info(f"Email SENT to {recipient} via {sender_email}. Subject: {subject[:50]}... (Message-ID: {message_id})")
                    if actual_client_id:
                        async with self.session_maker() as session:
                            await session.execute(update(Client).where(Client.id == actual_client_id).values(last_contacted_at=datetime.now(timezone.utc)))
                            await session.commit()
                else:
                    log_status = "failed_send"
                    self.logger.warning(f"Email FAILED for {recipient} via {sender_email}. Reason: {send_result.get('message')}")
                    if "recipient refused" in send_result.get('message', '').lower():
                         await self._mark_client_undeliverable(recipient)

                # Log email attempt to DB
                email_log = await self._log_email(actual_client_id, recipient, subject, body, log_status, sender_email, composition_ids, message_id)
                email_log_id = email_log.id if email_log else None

                # Cache tracking pixel ID -> email log ID mapping
                if email_log_id:
                     self.internal_state['tracking_pixel_cache'][str(tracking_pixel_uuid)] = email_log_id

            except Exception as e:
                self.logger.error(f"Unhandled error during send_email_task for {actual_client_id or target_identifier}: {e}", exc_info=True)
                log_status = "error_internal"
                await self._log_email(actual_client_id, recipient or str(target_identifier), subject or "ERROR", str(e), log_status, sender_email, None, None)
                await self._report_error(f"Send email task failed for {actual_client_id or target_identifier}: {e}")
            finally:
                self.internal_state['active_sends'] = max(0, self.internal_state.get('active_sends', 1) - 1)

    # MODIFICATION START: Added enriched_data parameter and goal, integrated into task_context
    async def _generate_email_content_internal(self,
                                               client: Optional[Client],
                                               campaign_id: Optional[int],
                                               enriched_data: Optional[Dict[str, Any]] = None,
                                               goal: str = 'engagement' # Add goal parameter
                                               ) -> tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """Generates subject and body using LLM, including enriched data, returns subject, body (HTML), composition_ids."""
        subject: Optional[str] = None
        body: Optional[str] = None
        composition_ids: Dict[str, Any] = {}
        client_info_dict = {}
        client_id_for_log = None

        if client:
             client_info_dict = {
                 "id": client.id, "name": client.name, "email": client.email,
                 "country": client.country, "interests": client.interests,
                 "engagement_score": client.engagement_score, "timezone": client.timezone
             }
             client_id_for_log = client.id
             recipient_email = client.email # Get email from client object
        elif enriched_data and enriched_data.get('verified_email'):
             recipient_email = enriched_data.get('verified_email')
             # Use enriched data if client object isn't available
             client_info_dict = {
                 "name": enriched_data.get('full_name', 'Valued Prospect'),
                 "email": recipient_email,
                 "job_title": enriched_data.get('job_title'),
                 "company_name": enriched_data.get('company_name'),
                 # Add other relevant fields from enriched_data
             }
             self.logger.info(f"Using enriched data for email to {recipient_email}")
        else:
             self.logger.error("Cannot generate email content: No client object or enriched data with email provided.")
             return None, None, {}

        try:
            # 1. Prepare Context (Fetch OSINT, KB Styles)
            osint_summary = "No specific OSINT data available."
            successful_styles = []

            if client_id_for_log:
                osint_task = {"action": "fetch_osint_summary", "client_id": client_id_for_log}
                # Use try-except for optional OSINT fetch
                try:
                    osint_result = await self.orchestrator.delegate_task("ThinkTool", osint_task)
                    if osint_result and osint_result.get('status') == 'success':
                        osint_summary = osint_result.get('summary', osint_summary)
                except Exception as osint_err:
                    self.logger.warning(f"OSINT fetch failed for client {client_id_for_log}: {osint_err}")

            # Fetch successful styles from DB
            async with self.session_maker() as session:
                stmt = select(EmailStyles.subject_template, EmailStyles.body_template).order_by(desc(EmailStyles.performance_score)).limit(3)
                style_results = await session.execute(stmt)
                successful_styles = style_results.mappings().all()

            task_context = {
                "client_info": client_info_dict,
                "osint_summary": osint_summary,
                "campaign_id": campaign_id,
                "task": "Generate personalized email subject and body",
                "goal": goal, # Use passed goal
                "successful_styles": [dict(s) for s in successful_styles],
                "desired_output_format": "JSON: {\"subject\": \"string\", \"body\": \"string (HTML formatted)\"}"
            }

            # ** ADD ENRICHED DATA TO CONTEXT IF AVAILABLE **
            if enriched_data:
                filtered_enriched_data = {
                    k: v for k, v in enriched_data.items()
                    if k in ['verified_email', 'job_title', 'company_name', 'full_name', 'linkedin_url'] and v
                }
                if filtered_enriched_data:
                    task_context['enriched_data_available'] = filtered_enriched_data # Add the data itself
                    self.logger.debug(f"Adding enriched data to prompt context: {list(filtered_enriched_data.keys())}")

            # 2. Generate Dynamic Prompt
            comprehensive_prompt = await self.generate_dynamic_prompt(task_context)

            # 3. Call LLM
            llm_response_str = await self._call_llm_with_retry(
                comprehensive_prompt, temperature=0.75, max_tokens=1500, is_json_output=True
            )
            if not llm_response_str: raise Exception("LLM call failed to return content.")

            # 4. Parse Response
            try:
                parsed_json = self._parse_llm_json(llm_response_str) # Use helper
                if not parsed_json: raise ValueError("Failed to parse LLM response.")
                subject = parsed_json.get('subject')
                body = parsed_json.get('body')
                if not subject or not body: raise ValueError("LLM response missing subject or body.")
            except (json.JSONDecodeError, ValueError, KeyError) as parse_err:
                self.logger.error(f"Failed to parse LLM JSON for {recipient_email}: {parse_err}. Response: {llm_response_str[:200]}...")
                raise ValueError(f"LLM response parsing failed: {parse_err}") from parse_err

            # 5. Humanization Check
            humanization_prompt = f"Critique this email draft for sounding robotic or overly 'AI-generated'. Focus on tone, flow, and natural language. Respond ONLY with 'Human-like' or 'Robotic'.\n\nSubject: {subject}\n\nBody:\n{self._html_to_plain_text(body)}"
            verdict = await self._call_llm_with_retry(humanization_prompt, temperature=0.1, max_tokens=10)
            if verdict and 'robotic' in verdict.lower():
                await self._internal_think(f"Email for {recipient_email} flagged as robotic. Requesting rewrite.")
                rewrite_prompt = f"Rewrite the following email to sound more human, natural, and less like AI. Keep the core message and call to action.\n\nSubject: {subject}\n\nBody:\n{body}\n\nOutput JSON: {{\"subject\": \"string\", \"body\": \"string (HTML formatted)\"}}"
                rewritten_json_str = await self._call_llm_with_retry(rewrite_prompt, temperature=0.7, max_tokens=1500, is_json_output=True)
                if rewritten_json_str:
                    try:
                        rewritten_data = self._parse_llm_json(rewritten_json_str)
                        if rewritten_data:
                            subject = rewritten_data.get('subject', subject)
                            body = rewritten_data.get('body', body)
                            self.logger.info(f"Successfully rewrote email for {recipient_email} for humanization.")
                        else: raise ValueError("Parsed rewritten data is None")
                    except Exception as rewrite_parse_err:
                         self.logger.warning(f"Failed to parse rewritten email JSON: {rewrite_parse_err}. Using original.")
                else: self.logger.warning("LLM failed to rewrite robotic email. Using original.")

            subject = subject.strip().replace('"', '')
            self.logger.info(f"Generated email content for {recipient_email}")

            return subject, body, composition_ids

        except Exception as e:
            self.logger.error(f"Internal email content generation failed for {recipient_email}: {e}", exc_info=True)
            return None, None, {}
    # MODIFICATION END

    # --- Remaining methods (_get_next_reset_time_utc, _check_global_limit, _wait_for_optimal_send_time, etc.) remain unchanged ---
    # --- Paste the rest of the EmailAgent class code here (from _get_next_reset_time_utc onwards) ---
    # --- ... (previous implementation of methods from _get_next_reset_time_utc to stop) ... ---

    def _get_next_reset_time_utc(self):
        """Calculates the next reset time (midnight UTC)."""
        now_utc = datetime.now(timezone.utc)
        reset_time = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return reset_time

    def _check_global_limit(self) -> bool:
        """Checks and resets the global daily send limit."""
        if datetime.now(timezone.utc) > self.internal_state['global_reset_time']:
            self.logger.info(f"Resetting global daily email limit. Previous count: {self.internal_state['global_sent_today']}")
            self.internal_state['global_sent_today'] = 0
            self.internal_state['global_reset_time'] = self._get_next_reset_time_utc()
        return self.internal_state['global_sent_today'] < self.internal_state['global_daily_limit']

    async def _wait_for_optimal_send_time(self, client: Client):
        """Calculates and waits for optimal send time based on client timezone and past opens."""
        try:
            client_tz_str = client.timezone or "America/New_York"
            client_tz = pytz.timezone(client_tz_str)
        except pytz.UnknownTimeZoneError:
            self.logger.warning(f"Invalid timezone '{client_tz_str}' for client {client.id}. Defaulting.")
            client_tz = pytz.timezone("America/New_York")

        optimal_hour_local = 10 # Default 10 AM in client's timezone

        try:
            async with self.session_maker() as session:
                 ninety_days_ago = datetime.now(timezone.utc) - timedelta(days=90)
                 stmt = select(EmailLog.opened_at).where(
                     EmailLog.client_id == client.id, EmailLog.opened_at.isnot(None),
                     EmailLog.timestamp >= ninety_days_ago
                 ).order_by(desc(EmailLog.opened_at)).limit(20)
                 result = await session.execute(stmt)
                 open_times_utc = [row.opened_at for row in result.mappings().all() if row.opened_at]

                 if len(open_times_utc) >= 3:
                     open_hours_local = [t.astimezone(client_tz).hour for t in open_times_utc]
                     hour_counts = Counter(h for h in open_hours_local if 8 <= h <= 17)
                     if hour_counts: optimal_hour_local = hour_counts.most_common(1)[0][0]; self.logger.debug(f"Optimal hour for client {client.id}: {optimal_hour_local} {client_tz_str}")
                     else: self.logger.debug(f"No past opens during business hours for client {client.id}. Using default {optimal_hour_local}.")
                 else: self.logger.debug(f"Insufficient open data (<3) for client {client.id}. Using default {optimal_hour_local}.")
        except Exception as e: self.logger.error(f"Error calculating optimal send hour for client {client.id}: {e}. Using default.")

        now_local = datetime.now(client_tz)
        send_time_local = now_local.replace(hour=optimal_hour_local, minute=random.randint(0, 29), second=0, microsecond=0)
        if send_time_local <= now_local: send_time_local += timedelta(days=1)

        send_time_utc = send_time_local.astimezone(timezone.utc)
        now_utc = datetime.now(timezone.utc)
        delay_seconds = (send_time_utc - now_utc).total_seconds()

        if delay_seconds > 0:
            max_delay = 60 * 60 * 36
            delay_seconds = min(delay_seconds, max_delay)
            if delay_seconds > 60:
                self.logger.info(f"Optimal Send Time: Waiting {delay_seconds:.0f}s to send to client {client.id} ({client.email}) at {send_time_local.strftime('%Y-%m-%d %H:%M')} {client_tz_str}")
            await asyncio.sleep(delay_seconds)

    def _select_sending_account(self) -> Optional[Dict]:
        """Selects an available SMTP provider respecting daily limits."""
        if not self.internal_state['smtp_providers']: return None
        num_providers = len(self.internal_state['smtp_providers'])
        now = datetime.now(timezone.utc)

        for i in range(num_providers):
            idx = (self.internal_state['current_provider_index'] + i) % num_providers
            provider = self.internal_state['smtp_providers'][idx]
            email = provider['email']

            if email not in self.internal_state['daily_limits']:
                 limit = int(self.config.get('SMTP_ACCOUNT_DAILY_LIMIT', 100))
                 self.internal_state['daily_limits'][email] = {'limit': limit, 'sent': 0, 'reset_time': self._get_next_reset_time_utc()}

            if now >= self.internal_state['daily_limits'][email]['reset_time']:
                 self.logger.info(f"Resetting daily limit for SMTP account {email}.")
                 self.internal_state['daily_limits'][email]['sent'] = 0
                 self.internal_state['daily_limits'][email]['reset_time'] = self._get_next_reset_time_utc()

            if self.internal_state['daily_limits'][email]['sent'] < self.internal_state['daily_limits'][email]['limit']:
                 self.internal_state['current_provider_index'] = (idx + 1) % num_providers
                 self.logger.debug(f"Selected sending account: {email} (Sent today: {self.internal_state['daily_limits'][email]['sent']}/{self.internal_state['daily_limits'][email]['limit']})")
                 return provider

        self.logger.warning("All sending accounts have reached their daily limits.")
        return None

    def _increment_send_count(self, sender_email):
        """Increments the send count for a specific account."""
        if sender_email in self.internal_state['daily_limits']:
            self.internal_state['daily_limits'][sender_email]['sent'] += 1
        else:
            limit = int(self.config.get('SMTP_ACCOUNT_DAILY_LIMIT', 100))
            self.internal_state['daily_limits'][sender_email] = {'limit': limit, 'sent': 1, 'reset_time': self._get_next_reset_time_utc()}
            self.logger.warning(f"Initialized daily limit counter for {sender_email} during increment.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=5, max=30), retry=retry_if_exception_type(smtplib.SMTPException))
    @smtp_breaker
    async def _send_email_smtp(self, recipient: str, subject: str, body_html: str, sender_config: Dict) -> Dict[str, Any]:
        """Sends email using a specific SMTP provider via smtplib."""
        host = sender_config['host']
        port = int(sender_config['port'])
        sender_email = sender_config['email']
        password = sender_config.get('pass')
        result = {"status": "failure", "message": "SMTP send initialization failed."}

        if not password:
             msg = f"SMTP password missing for sender {sender_email}. Cannot send."
             self.logger.error(msg)
             result["message"] = msg
             return result

        msg = MIMEMultipart("alternative")
        msg['Subject'] = subject
        sender_name = self.config.SENDER_NAME
        msg['From'] = f"{sender_name} <{sender_email}>"
        msg['To'] = recipient
        msg['Date'] = smtplib.email.utils.formatdate(localtime=True)
        message_id = smtplib.email.utils.make_msgid()
        msg['Message-ID'] = message_id

        plain_body = self._html_to_plain_text(body_html)
        msg.attach(MIMEText(plain_body, "plain", "utf-8"))
        msg.attach(MIMEText(body_html, "html", "utf-8"))

        try:
            await asyncio.to_thread(self._smtp_connect_send_quit, host, port, sender_email, password, msg)
            self.logger.info(f"Email successfully sent to {recipient} via {sender_email}")
            result = {"status": "success", "message": "Email sent successfully.", "message_id": message_id}
        except smtplib.SMTPAuthenticationError as e:
             self.logger.error(f"SMTP Auth failed for {sender_email} on {host}. Check credentials.")
             result["message"] = f"SMTP Auth error: {e}"
             raise
        except smtplib.SMTPRecipientsRefused as e:
             self.logger.error(f"SMTP Recipient Refused error for {recipient}: {e}")
             result["message"] = f"SMTP Recipient Refused: {e}"
             await self._mark_client_undeliverable(recipient)
             # Don't re-raise
        except smtplib.SMTPSenderRefused as e:
             self.logger.error(f"SMTP Sender Refused error for {sender_email}: {e}")
             result["message"] = f"SMTP Sender Refused: {e}"
             raise
        except smtplib.SMTPException as e:
            self.logger.warning(f"SMTP error sending to {recipient} via {host}: {e}")
            result["message"] = f"SMTP error: {e}"
            raise
        except Exception as e:
             self.logger.error(f"Unexpected error sending email via {host}: {e}", exc_info=True)
             result["message"] = f"Unexpected send error: {e}"
             raise smtplib.SMTPException(f"Unexpected send error: {e}") from e

        return result

    def _smtp_connect_send_quit(self, host, port, email_addr, password, msg):
        """Synchronous helper for SMTP operations."""
        logger.debug(f"Connecting to SMTP: {host}:{port}")
        context = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=30, context=context) as server:
                server.set_debuglevel(0)
                logger.debug(f"Logging in via SMTP_SSL as {email_addr}")
                server.login(email_addr, password)
                logger.debug("SMTP Login successful")
                server.send_message(msg)
                logger.debug(f"Message sent successfully via {host} (SSL)")
        else: # Assume port 587
            with smtplib.SMTP(host, port, timeout=30) as server:
                server.set_debuglevel(0)
                server.ehlo()
                logger.debug("Issuing STARTTLS")
                server.starttls(context=context)
                server.ehlo()
                logger.debug("STARTTLS successful")
                logger.debug(f"Logging in via STARTTLS as {email_addr}")
                server.login(email_addr, password)
                logger.debug("SMTP Login successful")
                server.send_message(msg)
                logger.debug(f"Message sent successfully via {host} (STARTTLS)")

    async def _log_email(self, client_id: Optional[int], recipient: str, subject: str, body: str, status: str, sender_email: Optional[str], composition_ids: Optional[Dict[str, Any]] = None, message_id: Optional[str] = None) -> Optional[EmailLog]:
        """Logs email details to the database."""
        if not self.session_maker: return None
        try:
            preview = self._html_to_plain_text(body)[:250] if body else None
            async with self.session_maker() as session:
                async with session.begin():
                    log = EmailLog(
                        client_id=client_id, recipient=recipient, subject=subject,
                        content_preview=preview, status=status,
                        timestamp=datetime.now(timezone.utc),
                        agent_version=f"{self.AGENT_NAME}_v3.1", # Use current version
                        sender_account=sender_email, message_id=message_id
                    )
                    session.add(log)
                await session.refresh(log)
                self.logger.debug(f"Logged email to {recipient}, status: {status}, EmailLog ID: {log.id}, MessageID: {message_id}")
                return log
        except SQLAlchemyError as e:
            self.logger.error(f"DB Error logging email for {recipient}: {e}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error logging email for {recipient}: {e}", exc_info=True)
            return None

    async def process_email_open(self, tracking_id: str):
        """Updates EmailLog when tracking pixel is hit."""
        self.logger.info(f"Processing email open for tracking ID: {tracking_id}")
        email_log_id = self.internal_state['tracking_pixel_cache'].pop(tracking_id, None)

        if not email_log_id:
             # Fallback: Query KVStore if implemented
             # if self.session_maker:
             #     async with self.session_maker() as session:
             #         kv_entry = await session.get(KVStore, tracking_id)
             #         if kv_entry: email_log_id = int(kv_entry.value)
             # if not email_log_id:
             self.logger.warning(f"Could not map tracking ID {tracking_id} to EmailLog ID.")
             return

        try:
            async with self.session_maker() as session:
                async with session.begin():
                    stmt = update(EmailLog).where(
                        EmailLog.id == email_log_id,
                        EmailLog.opened_at == None
                    ).values(
                        opened_at=datetime.now(timezone.utc),
                        status='opened',
                        last_interaction = datetime.now(timezone.utc) # Update client interaction time
                    ).returning(EmailLog.client_id)
                    result = await session.execute(stmt)
                    updated_client_id = result.scalar_one_or_none()

                if updated_client_id:
                    self.logger.info(f"Marked EmailLog {email_log_id} as opened.")
                    await self._update_client_engagement(updated_client_id, score_increase=0.1)
                else:
                    self.logger.debug(f"EmailLog {email_log_id} already marked as opened or not found.")

        except SQLAlchemyError as e: self.logger.error(f"DB Error processing email open for EmailLog ID {email_log_id}: {e}", exc_info=True)
        except Exception as e: self.logger.error(f"Unexpected error processing email open for EmailLog ID {email_log_id}: {e}", exc_info=True)

    async def process_email_reply(self, original_message_id: str, reply_from: str, reply_body: str):
        """Updates EmailLog and Client score upon receiving a reply."""
        self.logger.info(f"Processing email reply linked to Message-ID: {original_message_id}")
        email_log = None # Define outside try
        try:
            async with self.session_maker() as session:
                async with session.begin():
                    stmt_log = select(EmailLog).where(EmailLog.message_id == original_message_id).limit(1).with_for_update()
                    result_log = await session.execute(stmt_log)
                    email_log = result_log.scalar_one_or_none()

                    if not email_log:
                        self.logger.warning(f"Could not find original EmailLog for Message-ID {original_message_id}.")
                        return

                    if email_log.status == 'responded':
                        self.logger.info(f"EmailLog {email_log.id} already marked as responded. Skipping.")
                        return

                    now_ts = datetime.now(timezone.utc)
                    email_log.status = 'responded'
                    email_log.responded_at = now_ts
                    if not email_log.opened_at: email_log.opened_at = now_ts
                    client_id = email_log.client_id

                # Update client engagement score
                if client_id:
                    await self._update_client_engagement(client_id, score_increase=1.5, interaction_time=now_ts)

                # Log reply content as KnowledgeFragment via ThinkTool
                log_task = {
                    "action": "log_knowledge_fragment",
                    "fragment_data": {
                        "agent_source": "EmailReplyProcessor", "data_type": "email_reply_content",
                        "content": {"original_message_id": original_message_id, "from": reply_from, "body": reply_body},
                        "tags": ["email_reply", f"client_{client_id}" if client_id else "unknown_client"], "relevance_score": 0.8,
                        "related_client_id": client_id, "source_reference": f"EmailLog:{email_log.id}"
                    }
                }
                await self.orchestrator.delegate_task("ThinkTool", log_task)

                # Trigger ThinkTool analysis of reply content
                analysis_task = {
                    "action": "analyze_email_reply",
                    "reply_content": reply_body, "sender": reply_from, "client_id": client_id,
                    "original_subject": email_log.subject, "email_log_id": email_log.id
                }
                await self.orchestrator.delegate_task("ThinkTool", analysis_task)
                self.logger.info(f"Triggered ThinkTool analysis for reply to EmailLog {email_log.id}")

        except SQLAlchemyError as e: self.logger.error(f"DB Error processing email reply for Message-ID {original_message_id}: {e}", exc_info=True)
        except Exception as e: self.logger.error(f"Unexpected error processing email reply for Message-ID {original_message_id}: {e}", exc_info=True)

    async def _update_client_engagement(self, client_id: int, score_increase: float, interaction_time: Optional[datetime] = None):
        """Helper to update client engagement score and last interaction time."""
        if not client_id: return
        interaction_time = interaction_time or datetime.now(timezone.utc)
        try:
            async with self.session_maker() as session:
                async with session.begin():
                    stmt = update(Client).where(Client.id == client_id).values(
                        engagement_score = func.coalesce(Client.engagement_score, 0) + score_increase,
                        last_interaction = interaction_time # Use provided or current time
                    )
                    await session.execute(stmt)
                self.logger.debug(f"Updated engagement score for Client {client_id} by {score_increase}.")
        except SQLAlchemyError as e: self.logger.error(f"DB Error updating engagement score for Client {client_id}: {e}", exc_info=True)
        except Exception as e: self.logger.error(f"Unexpected error updating engagement score for Client {client_id}: {e}", exc_info=True)

    async def _check_imap_for_replies(self):
        """Connects to IMAP, checks for unseen emails, identifies replies/STOP requests, and triggers processing."""
        self.logger.info("Checking Hostinger IMAP for replies/opt-outs...")
        host = self.config.HOSTINGER_IMAP_HOST
        port = int(self.config.HOSTINGER_IMAP_PORT)
        user = self.config.HOSTINGER_IMAP_USER # Should default to HOSTINGER_EMAIL if not set
        password = self._imap_password

        if not all([host, user, password]):
            self.logger.error("IMAP configuration or password missing. Cannot check for replies.")
            return

        mail = None
        processed_replies = 0
        processed_optouts = 0
        try:
            loop = asyncio.get_running_loop()
            def connect_and_login():
                m = imaplib.IMAP4_SSL(host, port)
                m.login(user, password)
                m.select("inbox")
                return m
            mail = await loop.run_in_executor(None, connect_and_login)
            self.logger.info("IMAP connected successfully.")

            status, messages = await loop.run_in_executor(None, mail.search, None, '(UNSEEN)')
            if status != 'OK': self.logger.error(f"IMAP search failed: {messages}"); return

            email_ids = messages[0].split()
            if not email_ids: self.logger.info("No unseen emails found."); return

            self.logger.info(f"Found {len(email_ids)} unseen emails. Processing...")

            for email_id_bytes in reversed(email_ids):
                email_id_str = email_id_bytes.decode()
                try:
                    status, msg_data = await loop.run_in_executor(None, mail.fetch, email_id_bytes, '(RFC822)')
                    if status != 'OK': continue

                    full_msg = email.message_from_bytes(msg_data[0][1])
                    original_message_id = None
                    in_reply_to = full_msg.get('In-Reply-To')
                    references = full_msg.get('References')
                    subject = self._decode_header(full_msg.get('Subject', ''))
                    sender_email = email.utils.parseaddr(full_msg.get('From', ''))[1]

                    if in_reply_to: original_message_id = in_reply_to.strip('<>')
                    elif references:
                        ref_ids = references.split(); original_message_id = ref_ids[0].strip('<>') if ref_ids else None

                    reply_body = self._get_email_body(full_msg)

                    if self._is_stop_request(subject, reply_body):
                        self.logger.info(f"STOP request detected from {sender_email} (Subject: {subject[:50]}...). Processing opt-out.")
                        await self._mark_client_undeliverable(sender_email, opt_out=True)
                        processed_optouts += 1
                        await loop.run_in_executor(None, mail.store, email_id_bytes, '+FLAGS', '\\Seen')
                    elif original_message_id:
                        self.logger.info(f"Potential reply identified from {sender_email} for Message-ID: {original_message_id}.")
                        await self.process_email_reply(original_message_id, sender_email, reply_body)
                        processed_replies += 1
                        await loop.run_in_executor(None, mail.store, email_id_bytes, '+FLAGS', '\\Seen')
                    else:
                        self.logger.debug(f"Unseen email from {sender_email} (Subj: {subject[:50]}...) is not a recognized reply or STOP request. Marking as Seen.")
                        await loop.run_in_executor(None, mail.store, email_id_bytes, '+FLAGS', '\\Seen')

                except Exception as fetch_err:
                    self.logger.error(f"Error processing email ID {email_id_str}: {fetch_err}", exc_info=True)

            self.logger.info(f"Finished IMAP check. Processed Replies: {processed_replies}, Opt-Outs: {processed_optouts}.")

        except imaplib.IMAP4.error as imap_err:
            self.logger.error(f"IMAP connection/login error: {imap_err}", exc_info=True)
            if "authentication failed" in str(imap_err).lower(): self.logger.critical("IMAP AUTHENTICATION FAILED.")
        except Exception as e: self.logger.error(f"Unexpected error during IMAP check: {e}", exc_info=True)
        finally:
            if mail:
                try: await loop.run_in_executor(None, mail.logout)
                except Exception: pass

    def _decode_header(self, header_value: Optional[str]) -> str:
        """Safely decodes email headers."""
        if not header_value: return ""
        try:
            decoded_parts = decode_header(header_value)
            header_str = ""
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    header_str += part.decode(encoding or 'utf-8', errors='replace')
                elif isinstance(part, str):
                    header_str += part
            return header_str
        except Exception as e:
            logger.warning(f"Could not decode header: {header_value[:50]}... Error: {e}")
            return str(header_value)

    def _get_email_body(self, msg: email.message.Message) -> str:
        """Extracts plain text body from email message."""
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition'))
                if ctype == 'text/plain' and 'attachment' not in cdispo:
                    try:
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or 'utf-8'
                        body = payload.decode(charset, errors='replace')
                        break
                    except Exception as e:
                        logger.warning(f"Could not decode email part (type {ctype}): {e}")
        else:
            try:
                payload = msg.get_payload(decode=True)
                charset = msg.get_content_charset() or 'utf-8'
                body = payload.decode(charset, errors='replace')
            except Exception as e:
                logger.warning(f"Could not decode non-multipart email body: {e}")
        return body.strip()

    def _is_stop_request(self, subject: str, body: str) -> bool:
        """Checks if email subject or body contains a clear STOP request."""
        stop_keywords = ['stop', 'unsubscribe', 'remove me', 'opt out', 'opt-out']
        if any(keyword in subject.lower() for keyword in stop_keywords):
            return True
        if any(re.search(rf'\b{keyword}\b', body, re.IGNORECASE) for keyword in stop_keywords):
            return True
        return False

    async def _mark_client_undeliverable(self, email_address: str, opt_out: bool = False):
        """Marks a client as undeliverable or opted-out in the database."""
        if not email_address: return
        update_values = {}
        if opt_out:
            update_values['opt_in'] = False
            log_msg = f"Marking client {email_address} as opted-out."
        else:
            update_values['is_deliverable'] = False
            log_msg = f"Marking client {email_address} as undeliverable."

        self.logger.warning(log_msg)
        await self.log_operation('warning', log_msg)
        try:
            async with self.session_maker() as session:
                async with session.begin():
                    stmt = update(Client).where(Client.email == email_address).values(**update_values)
                    await session.execute(stmt)
        except Exception as e:
            self.logger.error(f"Failed to mark client {email_address} status: {e}", exc_info=True)

    def _html_to_plain_text(self, html_content: str) -> str:
        """Basic conversion of HTML to plain text using BeautifulSoup."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for script_or_style in soup(["script", "style"]): script_or_style.decompose()
            text_parts = []
            for element in soup.find_all(True):
                if element.name == 'p': text_parts.append('\n')
                text = element.get_text(separator=' ', strip=True)
                if text: text_parts.append(text)
                if element.name in ['br', 'p', 'div', 'h1', 'h2', 'h3', 'h4', 'li']: text_parts.append('\n')
            plain_text = re.sub(r'\n\s*\n', '\n\n', ' '.join(text_parts)).strip()
            return plain_text
        except Exception as e:
            self.logger.error(f"Error converting HTML to plain text: {e}")
            text = re.sub('<[^>]+>', ' ', html_content)
            return re.sub(r'\s+', ' ', text).strip()

    async def _apply_throttling(self):
         """Applies a random delay to simulate human sending patterns."""
         min_delay = 5
         max_delay = 25
         throttle_delay = random.uniform(min_delay, max_delay)
         self.logger.debug(f"Throttling send for {throttle_delay:.2f}s")
         await asyncio.sleep(throttle_delay)

    # --- Agent Lifecycle & Abstract Methods ---

    async def run(self):
        """Main run loop: Processes task queue and runs periodic IMAP checks."""
        if self.status == self.STATUS_RUNNING: self.logger.warning("EmailAgent run() called while already running."); return
        self.logger.info("EmailAgent starting run loop...")
        self._status = self.STATUS_RUNNING

        imap_task = asyncio.create_task(self._run_imap_checker_loop(), name=f"{self.AGENT_NAME}_IMAPChecker")
        self._background_tasks.add(imap_task)
        queue_processor_task = asyncio.create_task(self._process_task_queue(self.internal_state['task_queue']), name=f"{self.AGENT_NAME}_QueueProcessor")
        self._background_tasks.add(queue_processor_task)
        learning_task = asyncio.create_task(self._learning_loop_wrapper(), name=f"{self.AGENT_NAME}_LearningLoop")
        self._background_tasks.add(learning_task)

        while not self._stop_event.is_set():
            await asyncio.sleep(1)

        self.logger.info("EmailAgent run loop received stop signal.")

    async def _run_imap_checker_loop(self):
        """Background loop to periodically check IMAP inbox."""
        self.logger.info("Starting background IMAP checker loop.")
        while not self._stop_event.is_set():
            try:
                await self._check_imap_for_replies()
                await asyncio.sleep(self.internal_state['imap_check_interval_seconds'])
            except asyncio.CancelledError:
                self.logger.info("IMAP checker loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in IMAP checker loop: {e}", exc_info=True)
                await asyncio.sleep(self.internal_state['imap_check_interval_seconds'] * 2)

    async def plan_task(self, task_details: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        self.logger.debug("EmailAgent plan_task: Returning None, actions handled directly.")
        return None

    async def execute_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.error(f"EmailAgent execute_step called unexpectedly: {step}")
        return {"status": "failure", "message": "EmailAgent does not use planned steps."}

    async def learning_loop(self):
        """Analyzes email performance data to refine strategies (delegated to ThinkTool)."""
        self.logger.info("EmailAgent learning_loop: Performance analysis delegated to ThinkTool.")
        while not self._stop_event.is_set():
            await asyncio.sleep(3600 * 6)

    async def self_critique(self) -> Dict[str, Any]:
        """Evaluates email sending performance, deliverability, and engagement."""
        self.logger.info(f"{self.AGENT_NAME}: Performing self-critique.")
        critique = {"status": "ok", "feedback": "Critique pending analysis."}
        try:
            async with self.session_maker() as session:
                one_day_ago = datetime.now(timezone.utc) - timedelta(days=1)
                stmt = select(
                    EmailLog.status, func.count(EmailLog.id).label('count')
                ).where(EmailLog.timestamp >= one_day_ago).group_by(EmailLog.status)
                results = await session.execute(stmt)
                status_counts = {row.status: row.count for row in results.mappings().all()}

            total_sent = sum(status_counts.values())
            opened_count = status_counts.get('opened', 0) + status_counts.get('responded', 0)
            responded_count = status_counts.get('responded', 0)
            failed_count = status_counts.get('failed_send', 0) + status_counts.get('bounced', 0) + status_counts.get('blocked_compliance', 0)
            open_rate = (opened_count / total_sent * 100) if total_sent > 0 else 0
            response_rate = (responded_count / total_sent * 100) if total_sent > 0 else 0
            failure_rate = (failed_count / total_sent * 100) if total_sent > 0 else 0

            critique['performance_24h'] = {
                "total_attempts": total_sent,
                "opened": opened_count, "responded": responded_count, "failed": failed_count,
                "open_rate_pct": round(open_rate, 2), "response_rate_pct": round(response_rate, 2),
                "failure_rate_pct": round(failure_rate, 2)
            }
            critique['global_send_status'] = f"Sent {self.internal_state['global_sent_today']}/{self.internal_state['global_daily_limit']} globally today."
            critique['account_limits'] = self.internal_state.get('daily_limits', {})

            feedback_points = [f"24h Perf: Open {open_rate:.1f}%, Reply {response_rate:.1f}%, Fail {failure_rate:.1f}% ({total_sent} attempts)."]
            if failure_rate > 10: feedback_points.append("ACTION NEEDED: High failure rate (>10%). Check content, list quality, sender reputation, DMARC/SPF/DKIM setup.") ; critique['status'] = 'warning'
            if open_rate < 15 and total_sent > 50: feedback_points.append("WARNING: Low open rate (<15%). Review subject lines, deliverability, send times.") ; critique['status'] = 'warning'
            if self.internal_state['global_sent_today'] >= self.internal_state['global_daily_limit']: feedback_points.append("LIMIT REACHED: Global daily send limit hit.") ; critique['status'] = 'warning'
            elif self.internal_state['global_sent_today'] / self.internal_state['global_daily_limit'] > 0.9: feedback_points.append("INFO: Approaching global daily send limit.")

            critique['feedback'] = " ".join(feedback_points)

        except Exception as e:
            self.logger.error(f"Error during self-critique: {e}", exc_info=True)
            critique['status'] = 'error'; critique['feedback'] = f"Critique failed: {e}"
        return critique

    # MODIFICATION START: Added enriched_data handling in prompt generation
    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Constructs prompts for LLM calls, incorporating enriched data more effectively."""
        self.logger.debug(f"Generating dynamic prompt for EmailAgent task: {task_context.get('task')}")
        prompt_parts = [self.meta_prompt] # Use the agent's specific meta prompt

        prompt_parts.append("\n--- Current Task Context ---")
        # Prioritize key context items for clarity
        priority_keys = ['task', 'goal', 'client_info', 'enriched_data_available', 'osint_summary', 'successful_styles', 'campaign_id']
        for key in priority_keys:
            if key in task_context:
                value = task_context[key]
                value_str = ""
                max_len = 1500 # Default max length
                if key in ['osint_summary', 'successful_styles', 'enriched_data_available']: max_len = 2500 # Allow more context
                if isinstance(value, str): value_str = value[:max_len] + ("..." if len(value) > max_len else "")
                elif isinstance(value, (int, float, bool)): value_str = str(value)
                elif isinstance(value, dict):
                    # Special handling for enriched data for better readability in prompt
                    if key == 'enriched_data_available':
                         value_str = ", ".join([f"{k}: {v}" for k, v in value.items()])[:max_len] + "..."
                    else:
                         try: value_str = json.dumps(value, default=str, indent=2); value_str = value_str[:max_len] + ("..." if len(value_str) > max_len else "")
                         except TypeError: value_str = str(value)[:max_len] + "..."
                elif isinstance(value, list): value_str = json.dumps(value, default=str)[:max_len] + "..."
                else: value_str = str(value)[:max_len] + "..."
                # Use a more descriptive key for enriched data
                prompt_key = "Enriched Prospect Data" if key == 'enriched_data_available' else key.replace('_', ' ').title()
                prompt_parts.append(f"**{prompt_key}**: {value_str}")

        # Add remaining context items concisely
        other_params = {k: v for k, v in task_context.items() if k not in priority_keys and k not in ['desired_output_format']}
        if other_params:
            prompt_parts.append("\n**Other Parameters:**")
            try: prompt_parts.append(f"```json\n{json.dumps(other_params, default=str, indent=2)}\n```")
            except TypeError: prompt_parts.append(str(other_params)[:500] + "...")

        prompt_parts.append("\n--- Instructions ---")
        task_type = task_context.get('task')
        if task_type == 'Generate personalized email subject and body':
            prompt_parts.append("1. **Hyper-Personalize:** Use Client Info, OSINT Summary, AND **specifically leverage details from 'Enriched Prospect Data'** (like job title, company name, industry, location) to make the email highly relevant.")
            prompt_parts.append("2. **Subject Line:** Craft a compelling, human-like subject. Use insights from 'Successful Styles' if applicable.")
            prompt_parts.append("3. **Body Copy (HTML):** Write engaging HTML body. Weave in client interests/pain points (from Client Info/OSINT) and **reference enriched data points naturally**. Adapt tone based on 'Successful Styles' examples.")
            prompt_parts.append("4. **Tone:** Professional, persuasive, slightly informal. Avoid sounding robotic or overly salesy.")
            prompt_parts.append(f"5. **Call To Action (CTA):** Include a clear CTA aligned with the campaign 'Goal': '{task_context.get('goal', 'engagement')}'.")
            prompt_parts.append(f"6. **Output Format:** {task_context.get('desired_output_format')}")
        elif task_type == 'Critique email human-likeness':
             prompt_parts.append("Analyze the provided email draft. Does it sound like a natural human wrote it, or does it sound robotic/AI-generated? Focus on tone, flow, word choice, and common AI patterns.")
             prompt_parts.append(f"**Output Format:** Respond ONLY with 'Human-like' or 'Robotic'.")
        elif task_type == 'Rewrite email for humanization':
             prompt_parts.append("Rewrite the provided email to sound significantly more human and natural. Improve flow, vary sentence structure, use less formal language where appropriate, but retain the core message and call to action.")
             prompt_parts.append(f"**Output Format:** {task_context.get('desired_output_format')}")
        elif task_type == 'Analyze email for spam triggers':
             prompt_parts.append("Analyze the Subject and Body for potential spam triggers (e.g., excessive caps, spammy words like 'free', 'guarantee', '$', misleading claims, excessive links, poor formatting). Assign a spam score (0.0=safe, 1.0=spam). List specific issues found.")
             prompt_parts.append(f"**Output Format:** {task_context.get('desired_output_format')}")
        else:
            prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")

        if "JSON" in task_context.get('desired_output_format', ''): prompt_parts.append("\n```json")

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for EmailAgent (length: {len(final_prompt)} chars)")
        return final_prompt
    # MODIFICATION END

    async def collect_insights(self) -> Dict[str, Any]:
        """Collects insights about email performance and deliverability."""
        self.logger.debug("EmailAgent collect_insights called.")
        insights = {
            "agent_name": self.AGENT_NAME, "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_sends": self.internal_state.get('active_sends', 0),
            "global_sent_today": self.internal_state.get('global_sent_today', 0),
            "global_daily_limit": self.internal_state.get('global_daily_limit', 0),
            "account_send_counts": {email: data['sent'] for email, data in self.internal_state.get('daily_limits', {}).items()},
            "key_observations": []
        }
        try:
            async with self.session_maker() as session:
                one_day_ago = datetime.now(timezone.utc) - timedelta(days=1)
                stmt = select(
                    EmailLog.status, func.count(EmailLog.id).label('count')
                ).where(EmailLog.timestamp >= one_day_ago).group_by(EmailLog.status)
                results = await session.execute(stmt)
                insights["performance_24h"] = {row.status: row.count for row in results.mappings().all()}
            insights["key_observations"].append("Included 24h performance summary.")
        except Exception as e:
            self.logger.error(f"Error collecting DB insights for EmailAgent: {e}")
            insights["key_observations"].append("Failed to retrieve 24h performance summary.")
        return insights

    async def stop(self, timeout: float = 30.0):
        """Override stop to cancel background tasks."""
        self.logger.info(f"{self.AGENT_NAME} received stop signal.")
        self._stop_event.set()
        tasks_to_cancel = list(self._background_tasks)
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
        if tasks_to_cancel:
            self.logger.info(f"Waiting for {len(tasks_to_cancel)} EmailAgent background tasks to cancel...")
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            self.logger.info("EmailAgent background tasks cancellation complete.")
        await super().stop(timeout)

    # Helper to parse LLM JSON robustly (moved from ThinkTool for reuse)
    def _parse_llm_json(self, json_string: str, expect_type: Type = dict) -> Union[Dict, List, None]:
        """Safely parses JSON from LLM output, handling markdown code blocks."""
        if not json_string: return None
        try:
            match = None; start_char, end_char = '{', '}'
            if expect_type == list: start_char, end_char = '[', ']'
            match = re.search(rf'(?:```json)?\s*(\{start_char}.*\{end_char})\s*(?:```)?', json_string, re.DOTALL)

            parsed_json = None
            if match:
                potential_json = match.group(1)
                try: parsed_json = json.loads(potential_json)
                except json.JSONDecodeError: return None # Fail if specific block fails
            elif json_string.strip().startswith(start_char) and json_string.strip().endswith(end_char):
                 try: parsed_json = json.loads(json_string)
                 except json.JSONDecodeError: return None
            else: return None

            if isinstance(parsed_json, expect_type): return parsed_json
            else: self.logger.error(f"Parsed JSON type mismatch. Expected {expect_type}, got {type(parsed_json)}"); return None
        except Exception as e:
             self.logger.error(f"Unexpected error during JSON parsing in EmailAgent: {e}", exc_info=True)
             return None

# --- End of agents/email_agent.py ---