# Filename: agents/email_agent.py
# Description: Enhanced EmailAgent with agentic capabilities, IMAP receiving, and internal reflection.
# Version: 1.2 (Agentic Enhancements, Internal Think Step, IMAP, Learning)

import asyncio
import logging
import random
import os
import json
import smtplib
import imaplib
import email # For parsing emails
import re # For parsing email content/headers
from email.message import EmailMessage
from email.header import decode_header
from datetime import datetime, timedelta, timezone
import pytz
from collections import Counter
import pybreaker # For SMTP circuit breaker
import uuid # For tracking pixel
import hashlib # For caching
import time # For caching TTL

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy import update, desc, func, case, String, cast # Import func, case, String, cast
from sqlalchemy.dialects.postgresql import ARRAY # Assuming PostgreSQL for array operations
from sqlalchemy.exc import SQLAlchemyError

# Assuming these exist and are correctly set up
from typing import Optional, Dict, Any, List, Union, Tuple
from models import Client, EmailLog, PromptTemplate, KnowledgeFragment, EmailComposition, LearnedPattern # Added KB models
# Assuming utils/database.py provides encryption if needed, but not used directly here
# from utils.database import encrypt_data, decrypt_data
from config.settings import settings # Assuming settings object is available
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl
from openai import AsyncOpenAI as AsyncLLMClient # Assuming this is how LLM client is accessed via orchestrator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import the base class and KBInterface placeholder
try:
    from .base_agent import GeniusAgentBase
    # Define KBInterface placeholder if needed for type hinting, or get from orchestrator
    class KBInterfacePlaceholder:
        async def add_knowledge(self, *args, **kwargs): pass
        async def get_knowledge(self, *args, **kwargs): return []
        async def add_email_composition(self, *args, **kwargs): pass
        async def log_learned_pattern(self, *args, **kwargs): pass
        async def get_fragments_by_ids(self, *args, **kwargs): return [] # Add if needed
    KBInterface = KBInterfacePlaceholder
except ImportError:
    from base_agent import GeniusAgentBase
    class KBInterfacePlaceholder:
        async def add_knowledge(self, *args, **kwargs): pass
        async def get_knowledge(self, *args, **kwargs): return []
        async def add_email_composition(self, *args, **kwargs): pass
        async def log_learned_pattern(self, *args, **kwargs): pass
        async def get_fragments_by_ids(self, *args, **kwargs): return []
    KBInterface = KBInterfacePlaceholder

# Import meta prompt
try:
    # Assuming prompts are stored centrally or passed via config/orchestrator
    EMAIL_AGENT_META_PROMPT = getattr(settings, 'EMAIL_AGENT_META_PROMPT', """
You are an Email Agent focused on hyper-personalized, effective outreach.
Key Principles: Human-like tone, value focus, spam avoidance, A/B testing.
Goal: Drive engagement (opens, clicks, replies) and support sales objectives.
Operational Flow: Receive task -> Fetch context (Client, OSINT, KB) -> Generate Content (Subject, Body) -> Check Compliance/Deliverability -> Schedule/Send -> Log -> Learn from results. Process incoming replies/opens.
    """)
except ImportError:
    EMAIL_AGENT_META_PROMPT = "Default Email Agent Meta Prompt: Focus on hyper-personalized, effective outreach."
    logging.warning("Could not import EMAIL_AGENT_META_PROMPT. Using default.")


# Use module-level logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# SMTP Circuit Breaker
smtp_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60 * 10) # 10 min timeout

class EmailAgent(GeniusAgentBase):
    """
    Email Agent (Genius Level): Executes hyper-personalized, psychologically optimized
    email outreach campaigns. Masters human mimicry, spam evasion, engagement tracking,
    internal copywriting, and automated A/B/n testing of persuasive techniques.
    Processes incoming replies and open tracking data.
    Version: 1.2
    """
    AGENT_NAME = "EmailAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: object, smtp_password: str, imap_password: str):
        """Initializes the EmailAgent."""
        config = getattr(orchestrator, 'config', settings)
        # KB interaction might be mediated via ThinkTool/Orchestrator
        kb_interface = getattr(orchestrator, 'kb_interface', None)
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, config=config, kb_interface=kb_interface, session_maker=session_maker)

        self.think_tool = orchestrator.agents.get('think')
        self.secure_storage = getattr(orchestrator, 'secure_storage', None)

        self._smtp_password = smtp_password
        self._imap_password = imap_password

        # --- Internal State Initialization ---
        self.internal_state = getattr(self, 'internal_state', {})
        self.internal_state['task_queue'] = asyncio.PriorityQueue()
        self.internal_state['max_concurrency'] = int(self.config.get("EMAIL_AGENT_MAX_CONCURRENCY", 25))
        self.internal_state['active_sends'] = 0
        self.internal_state['send_semaphore'] = asyncio.Semaphore(self.internal_state['max_concurrency'])
        self.internal_state['daily_limits'] = {}
        self.internal_state['global_daily_limit'] = int(self.config.get("EMAIL_AGENT_MAX_PER_DAY", 1000))
        self.internal_state['global_sent_today'] = 0
        self.internal_state['global_reset_time'] = self._get_next_reset_time_utc()
        self.internal_state['smtp_providers'] = []
        smtp_providers_config = getattr(self.config, 'SMTP_PROVIDERS', [])
        hostinger_email = getattr(self.config, 'HOSTINGER_EMAIL', None)
        if smtp_providers_config:
            for provider_config in smtp_providers_config:
                if provider_config.get('email') == hostinger_email:
                    provider_config['pass'] = self._smtp_password
                    self.internal_state['smtp_providers'].append(provider_config)
                else: self.logger.warning(f"Skipping SMTP provider config for {provider_config.get('email')}")
        else: self.logger.warning("SMTP_PROVIDERS configuration missing or empty.")

        if not self.internal_state['smtp_providers']:
             self.logger.critical("EmailAgent: SMTP_PROVIDERS configuration is missing, empty, or password could not be added!")

        self.internal_state['current_provider_index'] = 0
        self.internal_state['active_tests'] = {}
        self.internal_state['campaign_stats'] = {}
        self.internal_state['preferred_subject_template'] = "Quick question about [Interest]"
        self.internal_state['min_sends_for_analysis'] = int(self.config.get("EMAIL_MIN_SENDS_FOR_ANALYSIS", 20)) # Min sends for learning loop
        self.internal_state['imap_check_interval_seconds'] = int(self.config.get("EMAIL_IMAP_CHECK_INTERVAL_S", 300)) # Check every 5 mins

        self.logger.info("Email Agent (Genius Level) v1.2 initialized.")

    # --- Task Queuing & Prioritization ---
    async def queue_email_task(self, client_id: int, campaign_id: Optional[int] = None, priority_override: Optional[float] = None):
        """Adds an email task to the priority queue."""
        try:
            async with self.session_maker() as session:
                client = await session.get(Client, client_id)
                if not client or not client.opt_in or not client.is_deliverable:
                    self.logger.warning(f"Skipping queue for client {client_id}: Not found, opted out, or undeliverable.")
                    return False

                if priority_override is not None:
                    priority = priority_override
                else:
                    priority = 1.0 - (client.engagement_score or 0.1)
                    priority += random.uniform(-0.05, 0.05)

                task_data = {'type': 'send_email', 'client_id': client_id, 'campaign_id': campaign_id}
                await self.internal_state['task_queue'].put((priority, task_data))
                self.logger.info(f"Queued email task for client {client_id} with priority {priority:.3f}")
                return True
        except SQLAlchemyError as e:
             self.logger.error(f"DB Error queuing email task for client {client_id}: {e}", exc_info=True)
             return False
        except Exception as e:
            self.logger.error(f"Failed to queue email task for client {client_id}: {e}", exc_info=True)
            return False

    async def queue_osint_leads(self, leads: list[dict]):
        """Processes leads from OSINTAgent and queues them."""
        queued_count = 0
        created_count = 0
        skipped_count = 0
        async with self.session_maker() as session:
            async with session.begin():
                for lead_data in leads:
                    email = lead_data.get('email')
                    if not email or '@' not in email:
                        skipped_count += 1
                        continue

                    existing = await session.execute(select(Client.id).where(Client.email == email))
                    if existing.scalar_one_or_none():
                        skipped_count += 1
                        continue

                    client = Client(
                        email=email, name=lead_data.get('name', f"Lead <{email}>"),
                        source=lead_data.get('source', 'OSINTLead'), opt_in=True,
                        is_deliverable=True, engagement_score=0.1
                    )
                    session.add(client)
                    await session.flush()
                    created_count += 1
                    if await self.queue_email_task(client.id):
                        queued_count += 1
                    else:
                         skipped_count +=1
            self.logger.info(f"Processed OSINT leads: Created={created_count}, Queued={queued_count}, Skipped={skipped_count}")

    # --- Email Generation (Internal Logic) ---
    async def generate_email_content(self, client: Client, campaign_id: Optional[int] = None) -> tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """Generates subject and body using LLM, returns subject, body (HTML), composition_ids."""
        subject: Optional[str] = None
        body: Optional[str] = None
        composition_ids: Dict[str, Any] = {}
        try:
            # 1. Prepare Context
            osint_summary = "No specific OSINT data available."
            # Fetch OSINT via KB if interface exists
            if self.kb_interface and hasattr(self.kb_interface, 'get_knowledge'):
                osint_fragments = await self.kb_interface.get_knowledge(
                    related_client_id=client.id, data_types=['osint_summary'], limit=1
                )
                if osint_fragments: osint_summary = osint_fragments[0].content

            task_context = {
                "client_info": { "id": client.id, "name": client.name, "email": client.email, "country": client.country, "interests": client.interests.split(',') if client.interests else ['their work'], "engagement_score": client.engagement_score, "timezone": client.timezone },
                "osint_summary": osint_summary, "campaign_id": campaign_id,
                "task": "Generate personalized email subject and body",
                "desired_outcome": "Compelling email driving engagement (open, click, reply)",
                "learned_preferences": { "preferred_subject": self.internal_state.get('preferred_subject_template') }
            }

            # 2. Generate Dynamic Prompt
            comprehensive_prompt = await self.generate_dynamic_prompt(task_context)

            # 3. Call LLM
            llm_response_str = await self._call_llm_with_retry(
                comprehensive_prompt, temperature=0.75, max_tokens=600, is_json_output=True
            )
            if not llm_response_str: raise Exception("LLM call failed to return content.")

            # 4. Parse Response
            try:
                json_match = json.loads(llm_response_str[llm_response_str.find('{'):llm_response_str.rfind('}')+1])
                subject = json_match.get('subject')
                body = json_match.get('body') # Expecting HTML
                if not subject: subject = f"Quick thought for {client.name.split()[0] if client.name else 'you'}"; self.logger.warning(f"LLM missing subject for client {client.id}. Using fallback.")
                if not body: raise Exception("LLM response parsed, but body content is missing.")
            except (json.JSONDecodeError, KeyError, ValueError) as parse_err:
                self.logger.error(f"Failed to parse LLM JSON for client {client.id}: {parse_err}. Response: {llm_response_str[:200]}...")
                subject = f"Following up with {client.name.split()[0] if client.name else 'you'}"
                body = f"<p>Hi {client.name.split()[0] if client.name else ''},</p><p>Just wanted to follow up regarding our UGC services. Let me know if you have any questions.</p><p>Best,<br>{getattr(self.config, 'SENDER_NAME', 'Support')}</p>"

            # 6. Add Tracking Pixel
            pixel_url = f"http://localhost:8000/track/{uuid.uuid4()}.png" # Placeholder
            tracking_domain = getattr(self.config, 'AGENCY_TRACKING_DOMAIN', None)
            if tracking_domain: pixel_url = f"https://{tracking_domain}/track/{uuid.uuid4()}.png"
            else: self.logger.warning("AGENCY_TRACKING_DOMAIN not found. Using placeholder pixel URL.")
            final_body = body + f'<img src="{pixel_url}" width="1" height="1" alt="" style="display:none;"/>'

            subject = subject.strip().replace('"', '')
            self.logger.info(f"Generated email content internally for client {client.id}")

            # 7. Store in KB (if interface exists)
            if self.kb_interface and hasattr(self.kb_interface, 'add_knowledge') and subject and final_body:
                try:
                    subject_kf_id = await self.kb_interface.add_knowledge(
                        content=subject, data_type='email_subject', source_agent=self.agent_name,
                        metadata={'client_id': client.id, 'campaign_id': campaign_id}
                    )
                    if subject_kf_id: composition_ids['subject_kf_id'] = subject_kf_id

                    body_kf_id = await self.kb_interface.add_knowledge(
                        content=final_body, data_type='email_body_snippet', source_agent=self.agent_name,
                        metadata={'client_id': client.id, 'campaign_id': campaign_id, 'snippet_type': 'full_body_html'}
                    )
                    if body_kf_id: composition_ids['body_snippets_kf_ids'] = [body_kf_id]
                except Exception as kb_err: self.logger.error(f"Error storing generated content to KB for client {client.id}: {kb_err}")

            return subject, final_body, composition_ids

        except Exception as e:
            self.logger.error(f"Internal email content generation failed for client {client.id}: {e}", exc_info=True)
            return None, None, {}

    def _get_regional_adaptation(self, country_code):
        if country_code == 'GB': return "Use subtle British phrasing (e.g., 'reckon', 'keen')."
        if country_code == 'AU': return "Use subtle Australian phrasing (e.g., 'no worries', 'good onya')."
        return "Use standard US English phrasing."

    # --- Standardized LLM Interaction ---
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=4, max=30), retry=retry_if_exception_type(Exception))
    async def _call_llm_with_retry(self, prompt: str, model_preference: Optional[List[str]] = None, temperature: float = 0.5, max_tokens: int = 1024, is_json_output: bool = False) -> Optional[str]:
        """Centralized method for calling LLMs via the Orchestrator."""
        if not self.orchestrator or not hasattr(self.orchestrator, 'call_llm'):
            self.logger.error("Orchestrator or its call_llm method is unavailable.")
            return None
        try:
            response_content = await self.orchestrator.call_llm(
                agent_name=self.AGENT_NAME, prompt=prompt, temperature=temperature,
                max_tokens=max_tokens, is_json_output=is_json_output, model_preference=model_preference
            )
            return response_content
        except Exception as e:
            self.logger.error(f"Error occurred calling LLM via orchestrator: {e}", exc_info=True)
            raise

    # --- Core Email Sending Logic ---
    async def send_email(self, to_address: str, subject: str, body: str, from_address: Optional[str] = None) -> Dict[str, Any]:
        """Sends an email using configured SMTP settings with deliverability checks."""
        self.logger.info(f"Preparing to send email to {to_address} with subject: {subject[:50]}...")
        result = {"status": "failure", "message": "Email sending initialization failed."}

        try:
            # 1. Content Analysis
            self.logger.debug("Analyzing email content for spam triggers...")
            analysis_prompt = f"Analyze the following email subject and body for potential spam triggers (e.g., excessive caps, spammy words like 'free', 'guarantee', misleading claims). Suggest improvements if needed.\nSubject: {subject}\nBody:\n{body}\n\nOutput JSON: {{ \"spam_score\": float (0.0-1.0), \"issues\": [str], \"suggestions\": str }}"
            analysis_result_json = await self._call_llm_with_retry(analysis_prompt, max_tokens=300, is_json_output=True)
            spam_score = 0.0
            if analysis_result_json:
                try:
                    analysis_data = json.loads(analysis_result_json)
                    spam_score = analysis_data.get('spam_score', 0.0)
                    issues = analysis_data.get('issues', [])
                    self.logger.info(f"Content analysis result: Score={spam_score:.2f}, Issues={issues}")
                    spam_threshold = float(getattr(self.config, 'EMAIL_SPAM_THRESHOLD', 0.8))
                    if spam_score >= spam_threshold:
                         raise ValueError(f"High spam score ({spam_score:.2f}). Issues: {issues}")
                except (json.JSONDecodeError, KeyError, ValueError) as json_err:
                     self.logger.warning(f"Could not parse spam analysis JSON: {json_err}. Proceeding cautiously.")
            else:
                 self.logger.warning("Spam analysis LLM call failed. Proceeding cautiously.")

            # 2. Deliverability Checks (Implementations)
            self.logger.debug("Performing deliverability checks...")
            sender_email = from_address or self.internal_state['smtp_providers'][0]['email'] # Get sender for checks
            sender_reputation_ok = await self._check_sender_reputation(sender_email)
            recipient_valid = await self._check_recipient_validity(to_address)
            is_suppressed = await self._check_suppression_list(to_address)

            can_send = sender_reputation_ok and recipient_valid and not is_suppressed
            if not can_send:
                 reason = "Suppressed" if is_suppressed else ("Invalid Recipient" if not recipient_valid else "Sender Reputation Issue")
                 self.logger.warning(f"Deliverability pre-checks failed for {to_address}. Reason: {reason}")
                 raise ValueError(f"Deliverability pre-checks failed: {reason}")

            # 3. SMTP Connection & Sending
            sender_config = self._select_sending_account() # Ensure we use an available account
            if not sender_config: raise ValueError("No available sending account.")
            host = sender_config['host']
            port = int(sender_config['port'])
            username = sender_config['email']
            password = sender_config['pass'] # Password is now in the config dict
            sender = username # Use the selected account's email

            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            sender_name = getattr(self.config, 'SENDER_NAME', sender)
            message["From"] = f"{sender_name} <{sender}>"
            message["To"] = to_address
            message['Date'] = smtplib.email.utils.formatdate(localtime=True)
            message_id = smtplib.email.utils.make_msgid() # Generate and store Message-ID
            message['Message-ID'] = message_id

            plain_body = self._html_to_plain_text(body) # Generate plain text version
            message.attach(MIMEText(plain_body, "plain", "utf-8")) # Specify encoding
            message.attach(MIMEText(body, "html", "utf-8")) # Specify encoding

            context = ssl.create_default_context()
            self.logger.debug(f"Connecting to SMTP server: {host}:{port}")

            await asyncio.to_thread(self._smtp_send, host, port, username, password, message)

            self.logger.info(f"Email successfully sent to {to_address}")
            result = {"status": "success", "message": "Email sent successfully.", "message_id": message_id} # Return Message-ID

        except smtplib.SMTPAuthenticationError as e:
             self.logger.error(f"SMTP Auth error sending email to {to_address}: {e}", exc_info=True)
             result["message"] = f"SMTP Auth error: {e}"
             # Report persistent auth errors
             if self.orchestrator: await self.orchestrator.report_client_issue(f"SMTP_{sender_email}", "auth_error")
        except smtplib.SMTPSenderRefused as e:
             self.logger.error(f"SMTP Sender Refused error for {sender_email}: {e}", exc_info=True)
             result["message"] = f"SMTP Sender Refused: {e}"
             # Report sender issue
             if self.orchestrator: await self.orchestrator.report_client_issue(f"SMTP_{sender_email}", "sender_issue")
        except smtplib.SMTPRecipientsRefused as e:
             self.logger.error(f"SMTP Recipient Refused error for {to_address}: {e}", exc_info=True)
             result["message"] = f"SMTP Recipient Refused: {e}"
             # Mark recipient as undeliverable
             await self._mark_client_undeliverable(to_address)
        except smtplib.SMTPException as smtp_err:
             self.logger.error(f"Generic SMTP error sending email to {to_address}: {smtp_err}", exc_info=True)
             result["message"] = f"SMTP error: {smtp_err}"
        except ValueError as ve:
             self.logger.error(f"Configuration or pre-check error sending email to {to_address}: {ve}")
             result["message"] = str(ve)
        except Exception as e:
            self.logger.error(f"Unexpected error sending email to {to_address}: {e}", exc_info=True)
            result["message"] = f"Failed to send email due to unexpected error: {e}"

        return result

    # --- Deliverability Check Placeholders ---
    async def _check_sender_reputation(self, sender_email: str) -> bool:
        self.logger.debug(f"Checking sender reputation for {sender_email} (Placeholder - returning True)")
        # TODO: Implement check using external service or internal tracking
        return True

    async def _check_recipient_validity(self, recipient_email: str) -> bool:
        self.logger.debug(f"Checking recipient validity for {recipient_email} (Placeholder - returning True)")
        # TODO: Implement check using validation API or bounce history from DB
        return True

    async def _check_suppression_list(self, recipient_email: str) -> bool:
        self.logger.debug(f"Checking suppression list for {recipient_email} (Placeholder - returning False)")
        # TODO: Implement check against internal suppression list in DB
        return False

    async def _mark_client_undeliverable(self, email_address: str):
        """Marks a client as undeliverable in the database."""
        self.logger.warning(f"Marking client with email {email_address} as undeliverable.")
        try:
            async with self.session_maker() as session:
                stmt = update(Client).where(Client.email == email_address).values(is_deliverable=False)
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            self.logger.error(f"Failed to mark client {email_address} as undeliverable: {e}", exc_info=True)

    def _html_to_plain_text(self, html_content: str) -> str:
        """Basic conversion of HTML to plain text."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            # Get text, handling paragraphs and line breaks better
            text_parts = []
            for element in soup.find_all(True):
                if element.name == 'p':
                    text_parts.append('\n') # Add space before paragraphs
                text = element.get_text(separator=' ', strip=True)
                if text:
                    text_parts.append(text)
                if element.name in ['br', 'p', 'div', 'h1', 'h2', 'h3', 'h4', 'li']:
                    text_parts.append('\n') # Add line breaks after block elements

            # Join parts and clean up excessive newlines
            plain_text = re.sub(r'\n\s*\n', '\n\n', ' '.join(text_parts)).strip()
            return plain_text
        except ImportError:
            self.logger.warning("BeautifulSoup4 not installed. Plain text conversion will be basic.")
            # Basic fallback without BeautifulSoup
            text = re.sub('<style.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub('<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub('<[^>]+>', ' ', text) # Remove HTML tags
            text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
            return text
        except Exception as e:
            self.logger.error(f"Error converting HTML to plain text: {e}")
            return "Could not generate plain text version." # Fallback

    # --- Email Sending & Delivery Task ---
    async def send_email_task(self, client_id: int, campaign_id: Optional[int] = None):
        """Handles the entire process for sending one email, including internal reflection and throttling."""
        async with self.internal_state['send_semaphore']:
            self.internal_state['active_sends'] += 1
            log_status = "failed"
            message_id = None # Store message ID for logging
            subject: Optional[str] = None
            body: Optional[str] = None
            recipient: Optional[str] = None
            sender_email: Optional[str] = None
            composition_ids: Dict[str, Any] = {}

            try:
                # 0. Check Global Daily Limit
                if self.internal_state['global_sent_today'] >= self.internal_state['global_daily_limit']:
                    if datetime.now(timezone.utc) > self.internal_state['global_reset_time']:
                        self.internal_state['global_sent_today'] = 0
                        self.internal_state['global_reset_time'] = self._get_next_reset_time_utc()
                    else:
                        self.logger.warning(f"Global daily email limit ({self.internal_state['global_daily_limit']}) reached. Task deferred for client {client_id}.")
                        await self.queue_email_task(client_id, campaign_id, priority_override=9.0)
                        return

                # 1. Fetch Client Data
                async with self.session_maker() as session:
                    client = await session.get(Client, client_id)
                    if not client or not client.opt_in or not client.is_deliverable or not client.email:
                        self.logger.warning(f"Cannot send email to client {client_id}: Invalid state (OptIn:{getattr(client, 'opt_in', 'N/A')}, Deliverable:{getattr(client, 'is_deliverable', 'N/A')}).")
                        return
                    recipient = client.email

                # 2. Compliance Check (Placeholder)
                compliance_ok = True # Assume OK
                if not compliance_ok:
                    self.logger.warning(f"Compliance check failed for {client.id}. Skipping send.")
                    await self._log_email(client_id, recipient, "Compliance Block", "", "blocked", None, None)
                    return

                # 3. Generate Content
                subject, body, composition_ids = await self.generate_email_content(client, campaign_id)
                if not subject or not body:
                    self.logger.error(f"Failed to generate content for {client.id}. Skipping send.")
                    await self._log_email(client_id, recipient, "Content Gen Failed", "", "failed_generation", sender_email, None)
                    return

                # 4. Optimal Send Time Calculation & Wait
                await self._wait_for_optimal_send_time(client)

                # 5. Select Sender Account
                sender_config = self._select_sending_account()
                if not sender_config:
                    self.logger.error(f"No available SMTP sending accounts within limits for client {client_id}. Task deferred.")
                    await self.queue_email_task(client_id, campaign_id, priority_override=8.0)
                    return
                sender_email = sender_config['email']

                # --- Internal Reflection Step ---
                pre_send_thought = f"""
                Pre-Send Checklist for Client ID {client_id} ({recipient}):
                - Compliance Check Passed: {'Yes (Placeholder)' if compliance_ok else 'No'}
                - Content Generated: Yes (Subject: '{subject[:30]}...')
                - Optimal Send Time Reached: Yes
                - Sending Account Selected: {sender_email}
                - Account Daily Limit OK: Yes
                - Global Daily Limit OK: Yes
                - Action: Proceed with sending via send_email method. Applying throttle.
                """
                await self._internal_think(pre_send_thought)
                # --- End Internal Reflection Step ---

                # 6. Apply Throttling
                # Example: 1000 emails/day = ~1 email every 86.4 seconds. Add jitter.
                min_delay = 70
                max_delay = 100
                throttle_delay = random.uniform(min_delay, max_delay)
                self.logger.debug(f"Throttling send for {throttle_delay:.2f}s before sending to {recipient}")
                await asyncio.sleep(throttle_delay)

                # 7. Send Email
                send_result = await self.send_email(recipient, subject, body, sender_email)
                send_success = send_result.get("status") == "success"
                message_id = send_result.get("message_id") # Get Message-ID if returned

                # 8. Update Limits & Log Result
                if send_success:
                    log_status = "sent"
                    self._increment_send_count(sender_email)
                    self.internal_state['global_sent_today'] += 1
                    self.logger.info(f"Email SENT to {recipient} via {sender_email}. Subject: {subject[:50]}... (Message-ID: {message_id})")
                    async with self.session_maker() as session:
                         await session.execute(update(Client).where(Client.id == client_id).values(last_interaction=datetime.now(timezone.utc)))
                         await session.commit()
                         self.logger.debug(f"Updated last_interaction time for client {client_id}")
                else:
                    log_status = "failed"
                    self.logger.warning(f"Email FAILED for {recipient} via {sender_email}. Reason: {send_result.get('message')}")

                log_entry = await self._log_email(client_id, recipient, subject, body, log_status, sender_email, composition_ids, message_id)

            except Exception as e:
                self.logger.error(f"Unhandled error during send_email_task for client {client_id}: {e}", exc_info=True)
                log_status = "error"
                if client_id and recipient:
                    await self._log_email(client_id, recipient, subject or "ERROR", body or f"Error: {e}", log_status, sender_email, None, None)
            finally:
                self.internal_state['active_sends'] -= 1


    def _get_next_reset_time_utc(self):
        """Calculates the next reset time (midnight UTC)."""
        now_utc = datetime.now(timezone.utc)
        reset_time = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return reset_time

    async def _wait_for_optimal_send_time(self, client: Client):
        """Calculates and waits for optimal send time based on client timezone and past opens."""
        try:
            client_tz_str = client.timezone or "America/New_York"
            client_tz = pytz.timezone(client_tz_str)
        except pytz.UnknownTimeZoneError:
            self.logger.warning(f"Invalid timezone '{client_tz_str}' for client {client.id}. Defaulting to US/Eastern.")
            client_tz = pytz.timezone("America/New_York")

        optimal_hour_local = 10 # Default

        try:
            async with self.session_maker() as session:
                 ninety_days_ago = datetime.now(timezone.utc) - timedelta(days=90)
                 stmt = select(EmailLog.opened_at).where(
                     EmailLog.client_id == client.id, EmailLog.opened_at.isnot(None),
                     EmailLog.timestamp >= ninety_days_ago # Corrected condition
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
            if delay_seconds > 60: self.logger.info(f"Optimal Send Time: Waiting {delay_seconds:.0f}s to send to client {client.id} ({client.email}) at {send_time_local.strftime('%Y-%m-%d %H:%M')} {client_tz_str}")
            await asyncio.sleep(delay_seconds)

    def _select_sending_account(self):
        """Selects an available SMTP provider respecting daily limits."""
        if not self.internal_state['smtp_providers']: return None
        num_providers = len(self.internal_state['smtp_providers'])
        now = datetime.now(timezone.utc)

        for i in range(num_providers):
            idx = (self.internal_state['current_provider_index'] + i) % num_providers
            provider = self.internal_state['smtp_providers'][idx]
            email = provider['email']

            if email not in self.internal_state['daily_limits']:
                 limit = int(getattr(self.config, 'PUPPETEER_PER_ACCOUNT_LIMIT', 100))
                 self.internal_state['daily_limits'][email] = {'limit': limit, 'sent': 0, 'reset_time': self._get_next_reset_time_utc()}

            if now >= self.internal_state['daily_limits'][email]['reset_time']:
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
            limit = int(getattr(self.config, 'PUPPETEER_PER_ACCOUNT_LIMIT', 100))
            self.internal_state['daily_limits'][sender_email] = {'limit': limit, 'sent': 1, 'reset_time': self._get_next_reset_time_utc()}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=5, max=30), retry=retry_if_exception_type(smtplib.SMTPException))
    @smtp_breaker
    async def _send_via_smtp(self, sender_config, recipient, subject, body):
        """Sends email using a specific SMTP provider."""
        host = sender_config['host']
        port = int(sender_config['port'])
        email_addr = sender_config['email'] # Renamed variable to avoid conflict
        password = sender_config.get('pass')
        if not password:
             self.logger.error(f"SMTP password missing for sender {email_addr}. Cannot send.")
             raise smtplib.SMTPAuthenticationError("Password not available for SMTP account.")

        msg = EmailMessage()
        msg.set_content("Please enable HTML to view this email.")
        msg.add_alternative(body, subtype='html')
        msg['Subject'] = subject
        sender_name = getattr(self.config, 'SENDER_NAME', email_addr)
        msg['From'] = f"{sender_name} <{email_addr}>"
        msg['To'] = recipient
        msg['Date'] = smtplib.email.utils.formatdate(localtime=True)
        message_id = smtplib.email.utils.make_msgid() # Generate Message-ID
        msg['Message-ID'] = message_id

        try:
            await asyncio.to_thread(self._smtp_connect_send_quit, host, port, email_addr, password, msg)
            return True, message_id # Return success and message_id
        except smtplib.SMTPAuthenticationError as e:
            self.logger.error(f"SMTP Auth failed for {email_addr} on {host}. Check credentials/config.")
            raise
        except smtplib.SMTPException as e:
            self.logger.warning(f"SMTP error sending to {recipient} via {host}: {e}")
            raise
        except Exception as e:
             self.logger.error(f"Unexpected error sending email via {host}: {e}", exc_info=True)
             raise smtplib.SMTPException(f"Unexpected send error: {e}")
        return False, None # Return failure

    def _smtp_connect_send_quit(self, host, port, email_addr, password, msg):
        """Synchronous helper for SMTP operations."""
        with smtplib.SMTP(host, port, timeout=45) as server:
            server.ehlo()
            if port == 587:
                 context = ssl.create_default_context()
                 server.starttls(context=context)
                 server.ehlo()
            server.login(email_addr, password)
            server.send_message(msg)


    async def _log_email(self, client_id: Optional[int], recipient: str, subject: str, body: str, status: str, sender_email: Optional[str], composition_ids: Optional[Dict[str, Any]] = None, message_id: Optional[str] = None):
        """Logs email details to the database and links composition."""
        try:
            async with self.session_maker() as session:
                log = EmailLog(
                    client_id=client_id, recipient=recipient, subject=subject,
                    content_preview=body[:250] if body else None, # Store preview
                    status=status, timestamp=datetime.now(timezone.utc),
                    agent_version='EmailAgent_Genius_v1.2', # Updated version
                    sender_account=sender_email,
                    message_id=message_id # Store Message-ID
                )
                session.add(log)
                await session.commit()
                await session.refresh(log)
                self.logger.debug(f"Logged email to {recipient}, status: {status}, EmailLog ID: {log.id}, MessageID: {message_id}")

                # Link composition via KBInterface
                if composition_ids and self.kb_interface and hasattr(self.kb_interface, 'add_email_composition') and log.id:
                    try:
                        composition_details = {
                            "subject_kf_id": composition_ids.get('subject_kf_id'),
                            "body_snippets_kf_ids": composition_ids.get('body_snippets_kf_ids'),
                        }
                        cleaned_composition_details = {k: v for k, v in composition_details.items() if v is not None}
                        if cleaned_composition_details:
                            await self.kb_interface.add_email_composition(
                                email_log_id=log.id, composition_details=cleaned_composition_details
                            )
                            self.logger.debug(f"Linked EmailLog {log.id} to composition items via KBInterface.")
                    except AttributeError: self.logger.error("KBInterface missing 'add_email_composition'.")
                    except Exception as comp_e: self.logger.error(f"Failed link composition for EmailLog {log.id}: {comp_e}", exc_info=True)
                return log
        except SQLAlchemyError as e:
            self.logger.error(f"DB Error logging email for {recipient}: {e}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error logging email for {recipient}: {e}", exc_info=True)
            return None


    # --- Engagement Tracking & Learning ---
    async def process_email_open(self, tracking_id: str):
        """Updates EmailLog when tracking pixel is hit."""
        self.logger.info(f"Processing email open for tracking ID: {tracking_id}")
        # Assuming tracking_id is the UUID part of the pixel URL, which needs to be mapped back to EmailLog.id
        # This requires storing the UUID -> EmailLog.id mapping somewhere temporarily (e.g., Redis cache)
        # or embedding EmailLog.id directly in the pixel URL (less secure).
        # --- Placeholder Logic ---
        email_log_id = None
        # email_log_id = await self.get_log_id_from_tracking_id(tracking_id) # Needs implementation
        if not email_log_id:
             self.logger.warning(f"Could not map tracking ID {tracking_id} to EmailLog ID.")
             return
        # --- End Placeholder ---

        try:
            async with self.session_maker() as session:
                stmt = update(EmailLog).where(
                    EmailLog.id == email_log_id,
                    EmailLog.opened_at == None # Only update first open
                ).values(
                    opened_at=datetime.now(timezone.utc),
                    status='opened'
                ).returning(EmailLog.client_id) # Get client_id for score update

                result = await session.execute(stmt)
                updated_client_id = result.scalar_one_or_none()
                await session.commit()

                if updated_client_id:
                    self.logger.info(f"Marked EmailLog {email_log_id} as opened.")
                    # Update client engagement score (example: small boost for open)
                    await self._update_client_engagement(updated_client_id, score_increase=0.1)
                else:
                    self.logger.debug(f"EmailLog {email_log_id} already marked as opened or not found.")

        except SQLAlchemyError as e:
            self.logger.error(f"DB Error processing email open for EmailLog ID {email_log_id}: {e}", exc_info=True)
        except Exception as e:
             self.logger.error(f"Unexpected error processing email open for EmailLog ID {email_log_id}: {e}", exc_info=True)

    async def process_email_reply(self, original_message_id: str, reply_from: str, reply_body: str):
        """Updates EmailLog and Client score upon receiving a reply."""
        self.logger.info(f"Processing email reply linked to Message-ID: {original_message_id}")
        try:
            async with self.session_maker() as session:
                # Find the original sent email log using the Message-ID
                stmt_log = select(EmailLog).where(EmailLog.message_id == original_message_id).limit(1)
                result_log = await session.execute(stmt_log)
                email_log = result_log.scalar_one_or_none()

                if not email_log:
                    self.logger.warning(f"Could not find original EmailLog for Message-ID {original_message_id}.")
                    return

                if email_log.status == 'responded':
                    self.logger.info(f"EmailLog {email_log.id} already marked as responded. Skipping.")
                    return

                # Update the original EmailLog
                email_log.status = 'responded'
                email_log.responded_at = datetime.now(timezone.utc)
                # If not already marked opened, mark it now
                if not email_log.opened_at:
                    email_log.opened_at = email_log.responded_at

                client_id = email_log.client_id
                await session.commit()
                self.logger.info(f"Marked EmailLog {email_log.id} as responded.")

                # Update client engagement score (significant boost for reply)
                if client_id:
                    await self._update_client_engagement(client_id, score_increase=1.5) # Example boost

                # Log reply content as KnowledgeFragment
                if self.kb_interface and hasattr(self.kb_interface, 'add_knowledge'):
                    await self.kb_interface.add_knowledge(
                        agent_source="EmailReplyProcessor",
                        data_type="email_reply_content",
                        content={"original_message_id": original_message_id, "from": reply_from, "body": reply_body},
                        tags=["email_reply", f"client_{client_id}"],
                        relevance_score=0.8,
                        related_client_id=client_id,
                        source_reference=f"EmailLog:{email_log.id}"
                    )

                # Trigger ThinkTool analysis of reply content
                if self.think_tool and hasattr(self.think_tool, 'execute_task'):
                    analysis_task = {
                        "action": "analyze_email_reply", # Define this action for ThinkTool
                        "reply_content": reply_body,
                        "sender": reply_from,
                        "client_id": client_id,
                        "original_subject": email_log.subject
                    }
                    # Run analysis in background
                    asyncio.create_task(self.think_tool.execute_task(analysis_task))
                    self.logger.info(f"Triggered ThinkTool analysis for reply to EmailLog {email_log.id}")

        except SQLAlchemyError as e:
            self.logger.error(f"DB Error processing email reply for Message-ID {original_message_id}: {e}", exc_info=True)
        except Exception as e:
             self.logger.error(f"Unexpected error processing email reply for Message-ID {original_message_id}: {e}", exc_info=True)

    async def _update_client_engagement(self, client_id: int, score_increase: float):
        """Helper to update client engagement score."""
        if not client_id: return
        try:
            async with self.session_maker() as session:
                # Use func.coalesce to handle potential NULL score, default to 0 before adding
                stmt = update(Client).where(Client.id == client_id).values(
                    engagement_score = func.coalesce(Client.engagement_score, 0) + score_increase,
                    last_interaction = datetime.now(timezone.utc) # Also update last interaction time
                )
                await session.execute(stmt)
                await session.commit()
                self.logger.debug(f"Updated engagement score for Client {client_id} by {score_increase}.")
        except SQLAlchemyError as e:
            self.logger.error(f"DB Error updating engagement score for Client {client_id}: {e}", exc_info=True)
        except Exception as e:
             self.logger.error(f"Unexpected error updating engagement score for Client {client_id}: {e}", exc_info=True)

    # --- IMAP Checking Logic ---
    async def _check_imap_for_replies(self):
        """Connects to IMAP, checks for unseen emails, identifies replies, and triggers processing."""
        self.logger.info("Checking Hostinger IMAP for replies...")
        host = getattr(self.config, 'HOSTINGER_IMAP_HOST', None)
        port = int(getattr(self.config, 'HOSTINGER_IMAP_PORT', 993))
        user = getattr(self.config, 'HOSTINGER_IMAP_USER', None)
        password = self._imap_password

        if not all([host, user, password]):
            self.logger.error("IMAP configuration or password missing. Cannot check for replies.")
            return

        mail = None
        processed_count = 0
        try:
            # Connect and login (synchronous library, run in thread)
            def connect_and_login():
                m = imaplib.IMAP4_SSL(host, port)
                m.login(user, password)
                m.select("inbox") # Select inbox
                return m
            mail = await asyncio.to_thread(connect_and_login)
            self.logger.info("IMAP connected successfully.")

            # Search for unseen emails
            status, messages = await asyncio.to_thread(mail.search, None, '(UNSEEN)')
            if status != 'OK':
                self.logger.error(f"IMAP search failed: {messages}")
                return

            email_ids = messages[0].split()
            if not email_ids:
                self.logger.info("No unseen emails found.")
                return

            self.logger.info(f"Found {len(email_ids)} unseen emails. Processing...")

            for email_id in reversed(email_ids): # Process newest first
                try:
                    # Fetch headers first to check for reply indicators
                    status, msg_data = await asyncio.to_thread(mail.fetch, email_id, '(BODY[HEADER.FIELDS (MESSAGE-ID FROM SUBJECT IN-REPLY-TO REFERENCES)])')
                    if status != 'OK': continue

                    headers = email.message_from_bytes(msg_data[0][1])
                    in_reply_to = headers.get('In-Reply-To')
                    references = headers.get('References')
                    subject = headers.get('Subject', '')
                    sender = headers.get('From', '')

                    original_message_id = None
                    if in_reply_to:
                        original_message_id = in_reply_to.strip('<>')
                    elif references:
                        # References header can contain multiple IDs, often the first is the original
                        ref_ids = references.split()
                        if ref_ids: original_message_id = ref_ids[0].strip('<>')

                    # Fallback: Subject line matching (less reliable)
                    if not original_message_id and subject.lower().startswith("re:"):
                         # TODO: Implement logic to find original email by subject and sender if needed
                         self.logger.debug(f"Potential reply found by subject '{subject}' from '{sender}'. Message-ID matching preferred.")
                         pass # Skip subject matching for now, focus on headers

                    if original_message_id:
                        self.logger.info(f"Potential reply identified for Message-ID: {original_message_id}. Fetching body...")
                        # Fetch full body
                        status, full_msg_data = await asyncio.to_thread(mail.fetch, email_id, '(RFC822)')
                        if status == 'OK':
                            full_msg = email.message_from_bytes(full_msg_data[0][1])
                            reply_body = ""
                            if full_msg.is_multipart():
                                for part in full_msg.walk():
                                    ctype = part.get_content_type()
                                    cdispo = str(part.get('Content-Disposition'))
                                    if ctype == 'text/plain' and 'attachment' not in cdispo:
                                        try:
                                            body_bytes = part.get_payload(decode=True)
                                            # Try common encodings
                                            for encoding in ['utf-8', 'iso-8859-1', 'windows-1252']:
                                                 try:
                                                      reply_body = body_bytes.decode(encoding)
                                                      break # Stop on success
                                                 except UnicodeDecodeError: continue
                                            else: # If no encoding worked
                                                 reply_body = body_bytes.decode('ascii', errors='ignore') # Fallback
                                            break # Use first plain text part found
                                        except Exception as dec_err:
                                             self.logger.warning(f"Error decoding email part: {dec_err}")
                            else: # Not multipart, assume plain text
                                try:
                                     body_bytes = full_msg.get_payload(decode=True)
                                     for encoding in ['utf-8', 'iso-8859-1', 'windows-1252']:
                                          try:
                                               reply_body = body_bytes.decode(encoding)
                                               break
                                          except UnicodeDecodeError: continue
                                     else: reply_body = body_bytes.decode('ascii', errors='ignore')
                                except Exception as dec_err:
                                     self.logger.warning(f"Error decoding non-multipart email body: {dec_err}")

                            if reply_body:
                                # Process the identified reply
                                await self.process_email_reply(original_message_id, sender, reply_body.strip())
                                processed_count += 1
                                # Mark as seen
                                await asyncio.to_thread(mail.store, email_id, '+FLAGS', '\\Seen')
                            else:
                                 self.logger.warning(f"Could not extract reply body for email ID {email_id.decode()}.")
                        else:
                             self.logger.warning(f"Failed to fetch full body for email ID {email_id.decode()}.")
                    # else: # No reply headers found
                    #     # Mark as seen to avoid re-processing non-replies? Optional.
                    #     # await asyncio.to_thread(mail.store, email_id, '+FLAGS', '\\Seen')
                    #     pass

                except Exception as fetch_err:
                    self.logger.error(f"Error processing email ID {email_id.decode()}: {fetch_err}", exc_info=True)
                    # Optionally mark as seen even on error?
                    # await asyncio.to_thread(mail.store, email_id, '+FLAGS', '\\Seen')

            self.logger.info(f"Finished IMAP check. Processed {processed_count} replies.")

        except imaplib.IMAP4.error as imap_err:
            self.logger.error(f"IMAP connection/login error: {imap_err}", exc_info=True)
            if "authentication failed" in str(imap_err).lower():
                 self.logger.critical("IMAP AUTHENTICATION FAILED. Check HOSTINGER_IMAP_USER/PASS.")
                 # TODO: Notify orchestrator about critical auth failure?
        except Exception as e:
            self.logger.error(f"Unexpected error during IMAP check: {e}", exc_info=True)
        finally:
            if mail:
                try:
                    await asyncio.to_thread(mail.logout)
                    self.logger.debug("IMAP connection logged out.")
                except Exception as logout_err:
                    self.logger.warning(f"Error during IMAP logout: {logout_err}")

    async def _run_imap_checker_loop(self):
        """Background loop to periodically check IMAP inbox."""
        self.logger.info("Starting background IMAP checker loop.")
        while self.status == "running": # Check agent status
            try:
                await self._check_imap_for_replies()
                await asyncio.sleep(self.internal_state['imap_check_interval_seconds'])
            except asyncio.CancelledError:
                self.logger.info("IMAP checker loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in IMAP checker loop: {e}", exc_info=True)
                # Wait longer after an error before retrying
                await asyncio.sleep(self.internal_state['imap_check_interval_seconds'] * 2)


    # --- Learning Loop ---
    async def learning_loop(self):
        """Agentic Learning Loop: Analyzes email performance to refine strategies."""
        while self.status == "running":
            try:
                await asyncio.sleep(60 * 60 * 1) # Run hourly
                self.logger.info("EmailAgent Learning Loop: Starting analysis cycle.")

                # --- Structured Thinking Step ---
                thinking_process = f"""
                Structured Thinking: Email Learning Loop
                1. Goal: Analyze recent email performance (opens/replies) linked to KB fragments (subjects, snippets) to identify effective strategies and update internal preferences.
                2. Context: Performance data from EmailLog, composition data from EmailComposition, fragment content from KnowledgeFragment.
                3. Constraints: Use database queries. Analyze fragments with sufficient send volume ({self.internal_state['min_sends_for_analysis']}). Update internal state and log learned patterns.
                4. Information Needed: Recent EmailLog statuses, EmailComposition links, KnowledgeFragment content for top performers.
                5. Plan:
                    a. Query DB for recent performance data joined with composition info.
                    b. Aggregate performance stats per fragment ID.
                    c. Calculate open/response rates for fragments meeting minimum send threshold.
                    d. Identify best performing subject fragment ID.
                    e. Fetch content for the best performing subject fragment.
                    f. If a new best subject is found, update internal state and log the learned pattern.
                    g. (Future) Extend analysis to body snippets, CTAs.
                """
                await self._internal_think(thinking_process)
                # --- End Structured Thinking Step ---

                async with self.session_maker() as session:
                    one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)
                    # Join EmailLog, EmailComposition, and filter by timestamp and status
                    stmt = select(
                        EmailLog.status,
                        EmailComposition.subject_kf_id,
                        EmailComposition.body_snippets_kf_ids # Array column
                    ).join(
                        EmailComposition, EmailLog.id == EmailComposition.email_log_id
                    ).where(
                        EmailLog.timestamp >= one_week_ago,
                        # Consider only relevant statuses for rate calculation
                        EmailLog.status.in_(['sent', 'opened', 'responded', 'failed', 'bounced'])
                    ).limit(2000) # Limit query complexity

                    results = await session.execute(stmt)
                    performance_data = results.mappings().all()

                if not performance_data:
                    self.logger.info("Learning Loop: No recent performance data found.")
                    continue

                self.logger.info(f"Learning Loop: Analyzing {len(performance_data)} recent email outcomes.")

                # Aggregate stats
                fragment_stats: Dict[int, Dict[str, int]] = {}
                for record in performance_data:
                    fragments_in_email = []
                    if record.subject_kf_id: fragments_in_email.append(record.subject_kf_id)
                    # Handle array of body snippet IDs
                    if record.body_snippets_kf_ids: fragments_in_email.extend(record.body_snippets_kf_ids)

                    for kf_id in set(fragments_in_email): # Use set to count each fragment once per email
                        if kf_id not in fragment_stats: fragment_stats[kf_id] = {'sent': 0, 'opened': 0, 'responded': 0}
                        fragment_stats[kf_id]['sent'] += 1
                        if record.status == 'opened': fragment_stats[kf_id]['opened'] += 1
                        elif record.status == 'responded':
                            fragment_stats[kf_id]['opened'] += 1 # Responded implies opened
                            fragment_stats[kf_id]['responded'] += 1

                # Analyze Subject Lines
                subject_open_rates: Dict[int, float] = {}
                min_sends = self.internal_state['min_sends_for_analysis']
                subject_ids_to_analyze = [kf_id for kf_id, stats in fragment_stats.items() if stats['sent'] >= min_sends]

                subject_contents: Dict[int, str] = {}
                if subject_ids_to_analyze:
                    async with self.session_maker() as session:
                         stmt_content = select(KnowledgeFragment.id, KnowledgeFragment.content).where(
                             KnowledgeFragment.id.in_(subject_ids_to_analyze),
                             KnowledgeFragment.data_type == 'email_subject'
                         )
                         content_results = await session.execute(stmt_content)
                         subject_contents = {row.id: row.content for row in content_results.mappings().all()}

                for kf_id in subject_ids_to_analyze:
                    if kf_id in subject_contents: # Only calculate if it's confirmed a subject and content fetched
                        stats = fragment_stats[kf_id]
                        open_rate = stats['opened'] / stats['sent']
                        subject_open_rates[kf_id] = open_rate

                best_subject_kf_id = max(subject_open_rates, key=subject_open_rates.get, default=None)

                if best_subject_kf_id and best_subject_kf_id in subject_contents:
                    new_preferred_template = subject_contents[best_subject_kf_id]
                    highest_open_rate = subject_open_rates[best_subject_kf_id]
                    current_preference = self.internal_state.get('preferred_subject_template')

                    if new_preferred_template != current_preference:
                        self.internal_state['preferred_subject_template'] = new_preferred_template
                        self.internal_state['last_learning_update_ts'] = datetime.now(timezone.utc)
                        self.logger.info(f"Learning Loop: Updated preferred subject template (Open Rate: {highest_open_rate:.2f}): '{new_preferred_template}'")
                        # Log learned pattern via KB interface or ThinkTool
                        if self.kb_interface and hasattr(self.kb_interface, 'log_learned_pattern'):
                            await self.kb_interface.log_learned_pattern(
                                pattern_description=f"Subject line '{new_preferred_template}' shows high open rate ({highest_open_rate:.2f})",
                                supporting_fragment_ids=[best_subject_kf_id],
                                confidence_score=min(0.9, 0.5 + (highest_open_rate * 0.4)),
                                implications="Prioritize using or adapting this subject structure.",
                                tags=["email_subject", "performance_optimized"]
                            )
                    else: self.logger.info("Learning Loop: Best performing subject template remains unchanged.")
                else: self.logger.info("Learning Loop: Insufficient data or unable to determine best performing subject template.")

                # TODO: Add analysis for body snippets, CTAs

            except asyncio.CancelledError:
                self.logger.info("EmailAgent learning loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error during EmailAgent learning loop: {e}", exc_info=True)
                await asyncio.sleep(60 * 30) # Wait longer after error

    async def self_critique(self) -> Dict[str, Any]:
        """Evaluates email sending performance and strategy effectiveness."""
        self.logger.info("EmailAgent: Performing self-critique.")
        critique = {"status": "ok", "feedback": "Critique pending analysis."}

        # --- Structured Thinking Step ---
        thinking_process = f"""
        Structured Thinking: Self-Critique EmailAgent
        1. Goal: Evaluate EmailAgent performance (deliverability, engagement, limits, learning effectiveness).
        2. Context: Internal state (limits, preferences), DB metrics (EmailLog stats).
        3. Constraints: Query DB for performance stats. Analyze against goals/thresholds. Output structured critique.
        4. Information Needed: Global/account send limits/counts, recent open/response/failure rates from DB. Current preferred subject template.
        5. Plan:
            a. Fetch internal state metrics (limits, counts, preferences).
            b. Query DB for 24h performance rates (open, response, failure).
            c. Analyze metrics against thresholds/goals. Check for high failure rates, low engagement, limit proximity.
            d. Formulate critique summary and feedback points.
            e. Return critique dictionary.
        """
        await self._internal_think(thinking_process)
        # --- End Structured Thinking Step ---

        try:
            # Analyze internal state
            global_limit = self.internal_state.get('global_daily_limit', 0)
            global_sent = self.internal_state.get('global_sent_today', 0)
            critique['global_send_status'] = f"Sent {global_sent}/{global_limit} globally today."
            critique['account_limits'] = self.internal_state.get('daily_limits', {})
            critique['current_preferred_subject'] = self.internal_state.get('preferred_subject_template')

            # Analyze recent DB performance
            async with self.session_maker() as session:
                one_day_ago = datetime.now(timezone.utc) - timedelta(days=1)
                stmt = select(
                    func.count(EmailLog.id).label('total_sent'),
                    func.sum(case((EmailLog.status == 'opened', 1), else_=0)).label('total_opened'),
                    func.sum(case((EmailLog.status == 'responded', 1), else_=0)).label('total_responded'),
                    func.sum(case((EmailLog.status.in_(['failed', 'bounced', 'blocked']), 1), else_=0)).label('total_failed')
                ).where(EmailLog.timestamp >= one_day_ago)
                result = await session.execute(stmt)
                perf_summary = result.mappings().first()

            feedback_points = []
            if perf_summary:
                total_sent = perf_summary.get('total_sent', 0)
                open_rate = (perf_summary.get('total_opened', 0) / total_sent) * 100 if total_sent > 0 else 0
                response_rate = (perf_summary.get('total_responded', 0) / total_sent) * 100 if total_sent > 0 else 0
                failure_rate = (perf_summary.get('total_failed', 0) / total_sent) * 100 if total_sent > 0 else 0
                critique['performance_24h'] = { "sent": total_sent, "open_rate_pct": round(open_rate, 2), "response_rate_pct": round(response_rate, 2), "failure_rate_pct": round(failure_rate, 2) }
                feedback_points.append(f"24h Perf: Open {open_rate:.1f}%, Reply {response_rate:.1f}%, Fail {failure_rate:.1f}%.")
                if failure_rate > 10: feedback_points.append("ACTION NEEDED: High failure rate (>10%) requires investigation (content/list quality/sender reputation).")
                if open_rate < 15 and total_sent > 50: feedback_points.append("WARNING: Low open rate (<15%) suggests subject line or deliverability issues.")
                if global_sent / global_limit > 0.9: feedback_points.append("INFO: Approaching global daily send limit.")
            else:
                 feedback_points.append("No performance data from last 24h.")

            critique['feedback'] = " ".join(feedback_points)

        except Exception as e:
            self.logger.error(f"Error during self-critique: {e}", exc_info=True)
            critique['status'] = 'error'
            critique['feedback'] = f"Self-critique failed: {e}"

        return critique

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Constructs context-rich prompts for LLM calls."""
        self.logger.debug(f"Generating dynamic prompt for task context: {task_context.get('task')}")

        # --- Structured Thinking Step ---
        thinking_process = f"""
        Structured Thinking: Generate Dynamic Prompt (Email Content)
        1. Goal: Create effective LLM prompt for generating personalized email subject & body.
        2. Context: Task details ({task_context.get('task')}), client info, OSINT summary, learned preferences ({task_context.get('learned_preferences')}).
        3. Constraints: Use EMAIL_AGENT_META_PROMPT. Incorporate all context. Instruct for JSON output (subject, body HTML).
        4. Information Needed: KB insights (top snippets/hooks/CTAs - Placeholder for now).
        5. Plan:
            a. Start with meta prompt.
            b. Append task context details.
            c. Append recipient profile details.
            d. Append learned preferences (subject template).
            e. Append relevant KB insights (placeholders for now).
            f. Add specific LLM instructions (personalization, tone, CTA, JSON output format).
            g. Return final prompt string.
        """
        await self._internal_think(thinking_process)
        # --- End Structured Thinking Step ---

        prompt_parts = [EMAIL_AGENT_META_PROMPT]
        prompt_parts.append("\n--- Current Task Context ---")
        prompt_parts.append(f"Task: {task_context.get('task', 'Generate email')}")
        prompt_parts.append(f"Desired Outcome: {task_context.get('desired_outcome', 'Drive engagement')}")
        if task_context.get('campaign_id'): prompt_parts.append(f"Campaign ID: {task_context['campaign_id']}")

        client_info = task_context.get('client_info', {})
        prompt_parts.append("\n--- Recipient Profile ---")
        prompt_parts.append(f"Name: {client_info.get('name', 'N/A')}")
        prompt_parts.append(f"Email: {client_info.get('email', 'N/A')}")
        prompt_parts.append(f"Country: {client_info.get('country', 'N/A')}")
        prompt_parts.append(f"Interests: {client_info.get('interests', ['general business'])}")
        prompt_parts.append(f"Engagement Score: {client_info.get('engagement_score', 0.1):.2f}")
        prompt_parts.append(f"OSINT Summary: {task_context.get('osint_summary', 'N/A')}")

        prompt_parts.append("\n--- Learned Preferences & Strategy ---")
        learned_prefs = task_context.get('learned_preferences', {})
        preferred_subject = learned_prefs.get('preferred_subject')
        if preferred_subject: prompt_parts.append(f"Preferred Subject Template (Adapt Creatively): '{preferred_subject}'")
        else: prompt_parts.append("Generate a compelling subject line.")

        prompt_parts.append("\n--- Relevant Knowledge (KB Retrieval - Placeholder) ---")
        # TODO: Implement actual KB query based on client/campaign context
        prompt_parts.append("Example Successful Snippet (Placeholder): '...found our clients achieve X% increase...'")

        prompt_parts.append("\n--- Instructions ---")
        prompt_parts.append("1. Deeply personalize using Recipient Profile and OSINT.")
        prompt_parts.append("2. Craft compelling, human-like subject, considering preferred template.")
        prompt_parts.append("3. Write engaging HTML body copy incorporating interests and successful snippets.")
        prompt_parts.append("4. Ensure tone is professional, persuasive, slightly informal.")
        prompt_parts.append("5. Include a clear Call To Action (CTA).")
        prompt_parts.append("6. **Output Format:** Respond ONLY with a valid JSON object: {\"subject\": \"string\", \"body\": \"string (HTML formatted)\"}. No extra text.")
        prompt_parts.append("```json")

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt (length: {len(final_prompt)} chars)")
        return final_prompt

    # --- Helper to log learned patterns ---
    async def log_learned_pattern(self, pattern_description: str, supporting_fragment_ids: List[int], confidence_score: float, implications: str, tags: Optional[List[str]] = None):
        """Logs a learned pattern, potentially via ThinkTool or directly to DB."""
        if self.kb_interface and hasattr(self.kb_interface, 'log_learned_pattern'):
            try:
                await self.kb_interface.log_learned_pattern(
                    pattern_description=pattern_description, supporting_fragment_ids=supporting_fragment_ids,
                    confidence_score=confidence_score, implications=implications, tags=tags
                )
            except Exception as e:
                 self.logger.error(f"Error logging learned pattern via KB interface: {e}")
        else:
            # Fallback: Log directly if KB interface is unavailable/limited
            self.logger.info(f"Learned Pattern (Not logged to KB): Desc='{pattern_description}', Conf={confidence_score:.2f}, Impl='{implications}'")
            # Optionally, queue a task for ThinkTool to log this pattern if direct DB access isn't desired here

# --- End of agents/email_agent.py ---