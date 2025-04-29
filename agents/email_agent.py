import asyncio
import logging
import random
import os
import json
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone # Ensure datetime and timezone are imported
import pytz
from collections import Counter

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, desc

# Assuming these exist and are correctly set up
from typing import Optional, Dict, Any # Added for type hints
from models import Client, EmailLog, Lead, PromptTemplate # Add others if needed
from utils.database import encrypt_data, decrypt_data # For email body if needed, though maybe not encrypting content itself
from config.settings import settings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl
import json # Already imported, ensure it stays
from typing import Optional, Dict, Any, List, Union # Ensure List, Union are added/present
import hashlib # For caching in _call_llm_with_retry
import time # For caching TTL in _call_llm_with_retry
from openai import AsyncOpenAI as AsyncLLMClient # Assuming this is how LLM client is accessed via orchestrator
# Assume Orchestrator provides access to ThinkTool, Vault, etc.
# from orchestrator import Orchestrator # Conceptual import
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pybreaker # For SMTP circuit breaker

# Import the base class and KBInterface placeholder
from .base_agent import GeniusAgentBase, KBInterface # Use relative import
from prompts.agent_meta_prompts import EMAIL_AGENT_META_PROMPT # Import meta prompt

logger = logging.getLogger(__name__)

# SMTP Circuit Breaker
smtp_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60 * 10) # 10 min timeout

class EmailAgent(GeniusAgentBase): # Renamed and inherited
    """
    Email Agent (Genius Level): Executes hyper-personalized, psychologically optimized
    email outreach campaigns. Masters human mimicry, spam evasion, engagement tracking,
    internal copywriting, and automated A/B/n testing of persuasive techniques.
    """

    # Ensure KBInterface is imported correctly at the top
    # from .base_agent import GeniusAgentBase, KBInterface

    def __init__(self, session_maker: callable, orchestrator: object, kb_interface: KBInterface, smtp_password: str, imap_password: str): # Accepts kb_interface and passwords
        """Initializes the EmailAgent.

        Args:
            session_maker: SQLAlchemy async session maker.
            orchestrator: The main Orchestrator instance.
            kb_interface: The interface for interacting with the Knowledge Base.
            smtp_password: The password for the primary SMTP account (fetched from Vault).
            imap_password: The password for the primary IMAP account (fetched from Vault).
        """
        agent_name = "EmailAgent"
        # Pass kb_interface to the base class constructor
        super().__init__(agent_name=agent_name, kb_interface=kb_interface)
        # self.config is inherited from GeniusAgentBase
        # self.kb_interface is inherited from GeniusAgentBase
        self.session_maker = session_maker # Keep DB session maker for operational data
        self.orchestrator = orchestrator # Keep orchestrator reference
        self.think_tool = orchestrator.agents.get('think') # Keep for non-core tasks initially
        self.secure_storage = orchestrator.secure_storage # Keep for secure data access

        # Store fetched passwords securely (though ideally avoid storing if possible)
        # For now, store them in internal state for use by SMTP/IMAP functions
        self._smtp_password = smtp_password
        self._imap_password = imap_password

        # --- Internal State Initialization ---
        self.internal_state['task_queue'] = asyncio.PriorityQueue()
        self.internal_state['max_concurrency'] = getattr(self.config, 'EMAIL_AGENT_MAX_CONCURRENCY', 25) # Use new setting name
        self.internal_state['active_sends'] = 0
        self.internal_state['send_semaphore'] = asyncio.Semaphore(self.internal_state['max_concurrency'])
        self.internal_state['daily_limits'] = {} # { 'sender_email': {'limit': 100, 'sent': 0, 'reset_time': ...} }
        self.internal_state['global_daily_limit'] = getattr(self.config, 'EMAIL_AGENT_MAX_PER_DAY', 1000) # Use new setting name
        self.internal_state['global_sent_today'] = 0
        self.internal_state['global_reset_time'] = self._get_next_reset_time_utc()
        # Update SMTP providers to include the fetched password
        self.internal_state['smtp_providers'] = []
        if self.config and self.config.SMTP_PROVIDERS:
            for provider_config in self.config.SMTP_PROVIDERS:
                # Assuming only one provider for now, add the password
                if provider_config.get('email') == self.config.HOSTINGER_EMAIL:
                    provider_config['pass'] = self._smtp_password
                    self.internal_state['smtp_providers'].append(provider_config)
                else:
                    # Handle multiple providers if logic is added later
                    self.logger.warning(f"Skipping SMTP provider config for {provider_config.get('email')} as password handling assumes single primary account.")

        if not self.internal_state['smtp_providers']:
             self.logger.critical("EmailAgent: SMTP_PROVIDERS configuration is missing, empty, or password could not be added!")
        self.internal_state['current_provider_index'] = 0
        self.internal_state['active_tests'] = {} # { 'test_id': {'variant_a_count': 0, 'variant_b_count': 0, ...} }
        self.internal_state['campaign_stats'] = {} # Placeholder for campaign performance tracking

        self.logger.info("Email Agent (Genius Level) initialized.")

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


    # --- Email Generation (Internal Logic) ---
    async def generate_email_content(self, client: Client, campaign_id: Optional[int] = None) -> tuple[Optional[str], Optional[str]]:
        """
        Generates subject and body using internal logic and a single LLM call guided by a dynamic prompt.
        Relies on self.generate_dynamic_prompt() to create context-rich instructions.
        """
        subject: Optional[str] = None
        body: Optional[str] = None
        subject_kf_id: Optional[int] = None
        body_kf_id: Optional[int] = None # Treating whole body as one fragment for now
        composition_ids: Dict[str, Any] = {}
        try:
            # 1. Prepare Context for Dynamic Prompt
            # Fetch recent interactions, OSINT summary if available from KB (Placeholder in generate_dynamic_prompt)
            # Placeholder - KBInterface needs implementation
            osint_summary = "No specific OSINT data available."
            # if self.kb_interface:
            #     osint_data = await self.kb_interface.get_knowledge(query=f"OSINT summary for {client.email}", type_filter='osint_summary', limit=1)
            #     if osint_data: osint_summary = osint_data[0]['content']

            task_context = {
                "client_info": {
                    "id": client.id,
                    "name": client.name,
                    "email": client.email,
                    "country": client.country,
                    "interests": client.interests.split(',') if client.interests else ['their work'],
                    "engagement_score": client.engagement_score,
                    "timezone": client.timezone,
                },
                "osint_summary": osint_summary,
                "campaign_id": campaign_id,
                "task": "Generate personalized email subject and body",
                "desired_outcome": "Compelling email driving engagement (open, click, reply)",
                # Add A/B test variant info if applicable
            }

            # 2. Generate the Dynamic Prompt using Agent's Internal Logic
            # This method will incorporate meta-prompt, context, KB insights, etc.
            comprehensive_prompt = await self.generate_dynamic_prompt(task_context)

            # 3. Make Single LLM Call for Subject & Body Generation
            # Instruct the LLM (via the prompt) to return structured output, e.g., JSON or specific delimiters
            # Example instruction within the prompt: "Generate the email. Output ONLY JSON with keys 'subject' and 'body'."
            llm_response_str = await self.think_tool._call_llm_with_retry(
                comprehensive_prompt,
                temperature=0.75, # Adjust temperature for creative generation
                max_tokens=600 # Allow sufficient length for subject and body
            ) # Using think_tool's call method as a proxy for internal LLM access for now

            if not llm_response_str:
                raise Exception("LLM call failed to return content.")

            # 4. Parse LLM Response (assuming JSON output as instructed in prompt)
            try:
                # Attempt to find JSON block if LLM adds extra text
                json_start = llm_response_str.find('{')
                json_end = llm_response_str.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    llm_response_json_str = llm_response_str[json_start:json_end]
                    llm_response = json.loads(llm_response_json_str)
                    subject = llm_response.get('subject')
                    body = llm_response.get('body')
                else:
                    # Fallback: Try simple splitting if not JSON (e.g., "Subject: ...\nBody: ...")
                    lines = llm_response_str.strip().split('\n', 1)
                    if lines[0].lower().startswith('subject:'):
                        subject = lines[0][len('subject:'):].strip()
                        body = lines[1].strip() if len(lines) > 1 else None
                    else: # Assume entire response is body if no clear structure
                         self.logger.warning(f"Could not parse LLM response structure for client {client.id}. Using full response as body.")
                         body = llm_response_str.strip()

                if not subject:
                    subject = f"Quick thought for {client.name.split()[0]}" # Fallback subject
                    self.logger.warning(f"LLM response missing subject for client {client.id}. Using fallback.")
                if not body:
                    raise Exception("LLM response parsed, but body content is missing.")

            except json.JSONDecodeError as jde:
                self.logger.error(f"Failed to parse JSON response from LLM for client {client.id}: {jde}. Response: {llm_response_str[:200]}...")
                # Fallback: Use full response as body, generate fallback subject
                subject = f"Following up with {client.name.split()[0]}"
                body = llm_response_str.strip() # Use the raw response as body
            except Exception as parse_exc:
                 self.logger.error(f"Error parsing LLM response for client {client.id}: {parse_exc}. Response: {llm_response_str[:200]}...")
                 raise # Reraise parsing error

            # 5. A/B Testing Logic Placeholder (Remains commented out)
            # ... (Keep existing commented A/B logic here for future integration) ...

            # 6. Add Tracking Pixel
            # TODO: Ensure settings.AGENCY_TRACKING_DOMAIN and uuid are correctly imported/available
            import uuid # Temporary import, move to top later
            pixel_url = f"http://localhost:8000/track/{uuid.uuid4()}.png" # Placeholder URL, use settings
            # if self.config and hasattr(self.config, 'AGENCY_TRACKING_DOMAIN'):
            #     pixel_url = f"https://{self.config.AGENCY_TRACKING_DOMAIN}/track/{uuid.uuid4()}.png"
            # else:
            #     self.logger.warning("AGENCY_TRACKING_DOMAIN not found in config. Using placeholder pixel URL.")

            final_body = body + f'<img src="{pixel_url}" width="1" height="1" alt="">'

            # Clean subject line
            subject = subject.strip().replace('"', '')

            self.logger.info(f"Generated email content internally for client {client.id}")

            # 7. Store generated content in Knowledge Base
            if self.kb_interface and subject and final_body:
                # Store subject
                subject_kf_id = await self.kb_interface.add_knowledge(
                    content=subject,
                    data_type='email_subject',
                    source_agent=self.agent_name,
                    metadata={'client_id': client.id, 'campaign_id': campaign_id}
                )
                if subject_kf_id:
                    composition_ids['subject_kf_id'] = subject_kf_id

                # Store body (as one fragment for now)
                # TODO: Implement body snippet extraction logic later
                body_kf_id = await self.kb_interface.add_knowledge(
                    content=final_body, # Store final HTML body with pixel
                    data_type='email_body_snippet', # Use generic type for now
                    source_agent=self.agent_name,
                    metadata={'client_id': client.id, 'campaign_id': campaign_id, 'snippet_type': 'full_body'}
                )
                if body_kf_id:
                    # Store as list even though it's one ID, for future compatibility
                    composition_ids['body_snippets_kf_ids'] = [body_kf_id]

            return subject, final_body, composition_ids # Return composition IDs

        except Exception as e:
            self.logger.error(f"Internal email content generation failed for client {client.id}: {e}", exc_info=True)
            return None, None, {} # Return empty dict on failure

    def _get_regional_adaptation(self, country_code):
        # Simple example, expand as needed
        if country_code == 'GB': return "Use subtle British phrasing (e.g., 'reckon', 'keen')."
        if country_code == 'AU': return "Use subtle Australian phrasing (e.g., 'no worries', 'good onya')."
        return "Use standard US English phrasing."

    # --- Standardized LLM Interaction (Copied from ThinkTool) ---
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=4, max=30), retry=retry_if_exception_type(Exception))
    async def _call_llm_with_retry(self, prompt: str, model_preference: Optional[List[str]] = None, temperature: float = 0.5, max_tokens: int = 1024, is_json_output: bool = False) -> Optional[str]:
        """
        Centralized method for calling LLMs via the Orchestrator.
        Handles client selection, retries, error reporting, and JSON formatting.
        (Adapted from ThinkTool)

        Args:
            prompt: The prompt string for the LLM.
            model_preference: Optional list of preferred model names/keys.
            temperature: The sampling temperature for the LLM.
            max_tokens: The maximum number of tokens to generate.
            is_json_output: Whether to request JSON output format from the LLM.

        Returns:
            The LLM response content as a string, or None if all retries fail.
        """
        llm_client: Optional[AsyncLLMClient] = None
        model_name: Optional[str] = None
        api_key_identifier: str = "unknown_key" # For logging/reporting issues

        try:
            # --- Caching Logic: Check Cache First ---
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            # Determine model name early for cache key consistency
            # Use a default model relevant for EmailAgent, or fetch from config
            default_model = getattr(self.config, 'OPENROUTER_MODELS', {}).get('email', "google/gemini-pro") # Fallback model
            # TODO: Refine model selection based on model_preference if provided and stable
            model_name_for_cache = default_model # Use the determined model for the cache key

            cache_key_parts = [
                "llm_call",
                prompt_hash,
                model_name_for_cache, # Include model name
                str(temperature),
                str(max_tokens),
                str(is_json_output),
            ]
            cache_key = ":".join(cache_key_parts)
            cache_ttl = 3600 # Default 1 hour TTL for LLM calls

            # Check cache first (assuming orchestrator provides caching)
            if hasattr(self.orchestrator, 'get_from_cache'):
                cached_result = self.orchestrator.get_from_cache(cache_key)
                if cached_result is not None:
                    self.logger.debug(f"LLM call cache hit for key: {cache_key[:20]}...{cache_key[-20:]}")
                    return cached_result # Return cached value
                else:
                    self.logger.debug(f"LLM call cache miss for key: {cache_key[:20]}...{cache_key[-20:]}")
            else:
                self.logger.warning("Orchestrator does not have 'get_from_cache' method. Skipping cache check.")
            # --- End Cache Check ---

            # 1. Get available clients from Orchestrator
            available_clients = await self.orchestrator.get_available_openrouter_clients()
            if not available_clients:
                self.logger.error("EmailAgent: No available LLM clients from Orchestrator.")
                return None

            # TODO: Implement smarter client/model selection
            llm_client = random.choice(available_clients)
            api_key_identifier = getattr(llm_client, 'api_key', 'unknown_key')[-6:] # Log last 6 chars

            # 2. Determine model name
            model_name = model_name_for_cache

            # 3. Prepare request arguments
            response_format = {"type": "json_object"} if is_json_output else None
            messages = [{"role": "user", "content": prompt}]

            self.logger.debug(f"EmailAgent LLM Call (Cache Miss): Model={model_name}, Temp={temperature}, MaxTokens={max_tokens}, JSON={is_json_output}, Key=...{api_key_identifier}")

            # 4. Make the API call
            response = await llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                timeout=60 # Generous timeout
            )

            content = response.choices[0].message.content.strip()

            # --- Token Tracking & Cost Estimation (Placeholder) ---
            input_tokens_est = len(prompt) // 4
            output_tokens = 0
            try:
                if response.usage and response.usage.completion_tokens:
                    output_tokens = response.usage.completion_tokens
                    if response.usage.prompt_tokens:
                        input_tokens_est = response.usage.prompt_tokens
                else:
                     output_tokens = len(content) // 4
            except AttributeError:
                 output_tokens = len(content) // 4

            total_tokens_est = input_tokens_est + output_tokens
            estimated_cost = total_tokens_est * 0.000001 # Example cost
            self.logger.debug(f"LLM Call Est. Tokens: ~{total_tokens_est}. Est. Cost: ${estimated_cost:.6f}")
            # --- End Token Tracking ---

            # --- Report Expense ---
            if estimated_cost > 0 and hasattr(self.orchestrator, 'report_expense'):
                try:
                    await self.orchestrator.report_expense(
                        agent_name=self.agent_name, # Use self.agent_name
                        amount=estimated_cost,
                        category="LLM",
                        description=f"LLM call ({model_name or 'unknown_model'}). Estimated tokens: {total_tokens_est}."
                    )
                except Exception as report_err:
                    self.logger.error(f"Failed to report LLM expense: {report_err}", exc_info=True)
            # --- End Report Expense ---

            # --- Caching Logic: Add to Cache on Success ---
            if content and hasattr(self.orchestrator, 'add_to_cache'):
                self.orchestrator.add_to_cache(cache_key, content, ttl_seconds=cache_ttl)
                self.logger.debug(f"Added LLM result to cache for key: {cache_key[:20]}...{cache_key[-20:]}")
            elif not hasattr(self.orchestrator, 'add_to_cache'):
                self.logger.warning("Orchestrator does not have 'add_to_cache' method. Skipping caching result.")
            # --- End Add to Cache ---

            return content

        except Exception as e:
            error_str = str(e).lower()
            issue_type = "llm_error"
            if "rate limit" in error_str or "quota" in error_str: issue_type = "rate_limit"
            elif "authentication" in error_str: issue_type = "auth_error"
            elif "timeout" in error_str: issue_type = "timeout_error"

            self.logger.warning(f"EmailAgent LLM call failed (attempt): Model={model_name}, Key=...{api_key_identifier}, ErrorType={issue_type}, Error={e}")

            # Report issue back to orchestrator
            if llm_client and hasattr(self.orchestrator, 'report_client_issue'):
                 await self.orchestrator.report_client_issue(api_key_identifier, issue_type)

            raise # Reraise exception for tenacity retry logic

        self.logger.error(f"EmailAgent LLM call failed after all retries: Model={model_name}, Key=...{api_key_identifier}")
        return None

    # --- Core Email Sending Logic ---
    async def send_email(self, to_address: str, subject: str, body: str, from_address: Optional[str] = None) -> Dict[str, Any]:
        """Sends an email using configured SMTP settings with deliverability checks."""
        self.logger.info(f"Preparing to send email to {to_address} with subject: {subject[:50]}...")
        result = {"status": "failure", "message": "Email sending initialization failed."}

        try:
            # 1. Content Analysis (LLM Call Placeholder)
            self.logger.debug("Analyzing email content for spam triggers...")
            analysis_prompt = f"Analyze the following email subject and body for potential spam triggers. Suggest improvements if needed.\nSubject: {subject}\nBody:\n{body}\n\nOutput: Brief analysis and spam score (0-10)."
            # Use the agent's own retry method
            analysis_result = await self._call_llm_with_retry(analysis_prompt, max_tokens=300)
            self.logger.info(f"Content analysis result: {analysis_result}")
            # TODO: Potentially halt sending or refine content based on analysis (e.g., if score > 7)

            # 2. Deliverability Checks (Placeholders)
            self.logger.debug("Performing deliverability checks (placeholders)...")
            # TODO: Check account warm-up status from internal state or config
            # TODO: Check sending volume limits for the day/hour (use self.internal_state['daily_limits'])
            # TODO: Check recipient against suppression list (if implemented, query DB/KB)
            can_send = True # Assume checks pass for now

            if not can_send:
                 raise ValueError("Deliverability pre-checks failed.")

            # 3. SMTP Connection & Sending
            # Fetch SMTP settings from config (as per example) - ensure config/settings.py has SMTP_SETTINGS
            smtp_config = getattr(self.config, 'SMTP_SETTINGS', {})
            host = smtp_config.get('host')
            port = smtp_config.get('port', 587) # Default TLS port
            username = smtp_config.get('username')
            # Fetch password securely - using the one passed during init for now
            # In a real scenario, fetch from secure_storage if needed per send
            password = self._smtp_password # Use password stored during init
            sender = from_address or smtp_config.get('default_from', username) # Use default if not specified

            if not all([host, port, username, password, sender]):
                self.logger.error(f"SMTP configuration incomplete. Host: {host}, Port: {port}, User: {username}, Pass: {'Set' if password else 'Not Set'}, Sender: {sender}")
                raise ValueError("SMTP configuration is incomplete.")

            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = f"{settings.SENDER_NAME} <{sender}>" # Use configured sender name
            message["To"] = to_address
            message['Date'] = smtplib.email.utils.formatdate(localtime=True)
            message['Message-ID'] = smtplib.email.utils.make_msgid()

            # Attach plain text and potentially HTML versions
            # Assuming 'body' might be HTML, add plain text version too
            # TODO: Add logic to generate plain text from HTML if body is HTML
            plain_body = body # Placeholder: Use body as plain text for now
            message.attach(MIMEText(plain_body, "plain"))
            # If body is intended as HTML:
            message.attach(MIMEText(body, "html"))

            context = ssl.create_default_context()
            self.logger.debug(f"Connecting to SMTP server: {host}:{port}")

            # Use asyncio.to_thread for synchronous smtplib operations
            def smtp_send():
                with smtplib.SMTP(host, port, timeout=45) as server:
                    server.ehlo()
                    if port == 587: # Use STARTTLS for port 587
                        server.starttls(context=context)
                        server.ehlo() # Re-identify after TLS
                    # Add logic for port 465 (SSL) if needed
                    # elif port == 465:
                    #     # Requires smtplib.SMTP_SSL() which needs separate handling
                    #     pass
                    server.login(username, password)
                    self.logger.debug(f"Sending email from {sender} to {to_address}")
                    server.sendmail(sender, to_address, message.as_string())

            await asyncio.to_thread(smtp_send)

            self.logger.info(f"Email successfully sent to {to_address}")
            result = {"status": "success", "message": "Email sent successfully."}

        except smtplib.SMTPException as smtp_err:
             self.logger.error(f"SMTP error sending email to {to_address}: {smtp_err}", exc_info=True)
             result["message"] = f"SMTP error: {smtp_err}"
             # TODO: Potentially trigger circuit breaker or account cooldown logic here
        except ValueError as ve:
             self.logger.error(f"Configuration or pre-check error sending email to {to_address}: {ve}")
             result["message"] = str(ve)
        except Exception as e:
            self.logger.error(f"Unexpected error sending email to {to_address}: {e}", exc_info=True)
            result["message"] = f"Failed to send email due to unexpected error: {e}"

        # TODO: Log the attempt (success or failure) using _log_email or similar
        # await self._log_email(client_id, to_address, subject, body, result["status"], sender)

        return result

    # --- Email Sending & Delivery ---
    async def send_email_task(self, client_id: int, campaign_id: Optional[int] = None):
        """Handles the entire process for sending one email."""
        async with self.send_semaphore:
            self.active_sends += 1
            log_status = "failed"
            log_entry = None
            subject: Optional[str] = None
            body: Optional[str] = None
            recipient: Optional[str] = None
            sender_email: Optional[str] = None
            composition_ids: Dict[str, Any] = {} # To store fragment IDs

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

                # 3. Generate Content & Get Composition IDs
                subject, body, composition_ids = await self.generate_email_content(client, campaign_id)
                if not subject or not body:
                    self.logger.error(f"Failed to generate content for {client.id}. Skipping send.")
                    # Log failure?
                    await self._log_email(client_id, recipient, "Content Gen Failed", "", "failed_generation", sender_email)
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
                    # Update client last interaction time (Verification: This existing logic correctly tracks basic interaction)
                    async with self.session_maker() as session:
                         await session.execute(update(Client).where(Client.id == client_id).values(last_interaction=datetime.now(timezone.utc)))
                         await session.commit()
                         self.logger.debug(f"Updated last_interaction time for client {client_id}")
                else:
                    log_status = "failed"
                    # Failure handled by retry/breaker, log as failed here
                    logger.warning(f"Email FAILED for {recipient} via {sender_email}.")

                # Pass composition_ids to log function
                log_entry = await self._log_email(client_id, recipient, subject, body, log_status, sender_email, composition_ids)

                # 8. A/B Testing: Log variant if applicable (Needs integration with composition)
                # variant_tag = composition_ids.get('ab_variant') # Example if variant info is added
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
        # Password should now be present in sender_config from __init__
        password = sender_config.get('pass')
        if not password:
             logger.error(f"SMTP password missing for sender {email}. Cannot send.")
             # This should ideally not happen if init logic is correct
             raise smtplib.SMTPAuthenticationError("Password not available for SMTP account.")

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


    async def _log_email(self, client_id, recipient, subject, body, status, sender_email, composition_ids: Optional[Dict[str, Any]] = None):
        """
        Logs email details to the database.
        Includes placeholder logic for linking to knowledge items used in composition.
        """
        # composition_ids might look like: {'subject': 10, 'hook': 15, 'body_snippets': [20, 25], 'cta': 30}
        try:
            async with self.session_maker() as session:
                log = EmailLog(
                    client_id=client_id,
                    recipient=recipient,
                    subject=subject,
                    content=body, # Store the HTML body (with pixel)
                    status=status,
                    timestamp=datetime.now(timezone.utc),
                    agent_version='EmailAgent_Genius_v1.0', # Updated version
                    # opened_at=None, # Updated via tracking endpoint later
                    # responded_at=None # Updated via reply processing later
                    sender_account=sender_email # Assuming EmailLog model has this field
                )
                session.add(log)
                await session.commit()
                await session.refresh(log)
                self.logger.debug(f"Logged email to {recipient}, status: {status}, EmailLog ID: {log.id}")

                # --- Link composition via KBInterface ---
                if composition_ids and self.kb_interface and log.id: # Check if KB is available and log was successful
                    try:
                        # Prepare details for add_email_composition
                        composition_details = {
                            "subject_kf_id": composition_ids.get('subject_kf_id'),
                            "hook_kf_id": composition_ids.get('hook_kf_id'), # Add if hook extraction is implemented
                            "body_snippets_kf_ids": composition_ids.get('body_snippets_kf_ids'),
                            "cta_kf_id": composition_ids.get('cta_kf_id') # Add if CTA extraction is implemented
                        }
                        # Remove None values before sending
                        cleaned_composition_details = {k: v for k, v in composition_details.items() if v is not None}

                        if cleaned_composition_details: # Only add if there's something to link
                            await self.kb_interface.add_email_composition(
                                email_log_id=log.id,
                                composition_details=cleaned_composition_details
                            )
                            self.logger.debug(f"Linked EmailLog {log.id} to composition items via KBInterface.")
                        else:
                            self.logger.debug(f"No valid composition IDs found to link for EmailLog {log.id}.")

                    except Exception as comp_e:
                        self.logger.error(f"Failed to link email composition for EmailLog {log.id} via KBInterface: {comp_e}", exc_info=True)
                # --- End Linking Composition ---

                return log
        except Exception as e:
            self.logger.error(f"Failed to log email for {recipient}: {e}", exc_info=True)
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
                await self.orchestrator.report_error("EmailAgent", f"Critical run loop error: {e}")
                await asyncio.sleep(60) # Wait after critical error

    # --- Abstract Method Implementations ---
    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Executes an email-related task, primarily sending."""
        self.status = "working"
        task_action = task_details.get('action', 'send') # Default to send

        result = {"status": "failure", "message": "Unknown email task action."}

        if task_action == 'send':
            to = task_details.get('to_address')
            subject = task_details.get('subject')
            body = task_details.get('body')
            # Optional: client_id if needed for logging/context, though not strictly required by send_email
            # client_id = task_details.get('client_id')

            if not all([to, subject, body]):
                 result["message"] = "Missing 'to_address', 'subject', or 'body' for send action."
                 self.logger.error(f"EmailAgent execute_task failed: {result['message']}")
            else:
                 # Call the new send_email method
                 result = await self.send_email(to_address=to, subject=subject, body=body)
        # TODO: Add other actions like 'generate_template', 'check_deliverability'
        elif task_action == 'queue_email': # Keep existing queue logic if needed via execute_task
             client_id = task_details.get('client_id')
             campaign_id = task_details.get('campaign_id')
             if client_id:
                 queued = await self.queue_email_task(client_id, campaign_id)
                 result = {"status": "success" if queued else "failure", "message": f"Task queued for client {client_id}." if queued else f"Failed to queue task for client {client_id}."}
             else:
                 result["message"] = "Missing 'client_id' for queue_email action."
                 self.logger.error(f"EmailAgent execute_task failed: {result['message']}")
        else:
             self.logger.warning(f"execute_task: Unknown task action '{task_action}'")
             result["message"] = f"Unknown email task action: {task_action}"


        self.status = "idle"
        return result

    async def learning_loop(self):
        """
        Prototype Learning Loop (v1 - Simulated Data).
        Periodically simulates retrieving performance data, analyzing it,
        and updating internal strategy state.
        """
        self.logger.info("Executing learning loop prototype...")

        try:
            # --- 1. Simulate Data Retrieval (Placeholder) ---
            # In reality, this would query KBInterface/DB using logic from database_schema_updates.md
            # Example: await self.kb_interface.get_performance_summary('email_subject', period='7d')
            simulated_performance_data = [
                {'type': 'email_subject', 'content': 'Quick question about [Interest]', 'open_rate': 0.35, 'sample_size': 150, 'id': 10},
                {'type': 'email_subject', 'content': 'Idea for [Company Name]', 'open_rate': 0.45, 'sample_size': 120, 'id': 11},
                {'type': 'email_subject', 'content': 'Following up', 'open_rate': 0.20, 'sample_size': 200, 'id': 12},
                # Add simulated data for hooks, CTAs etc. later
            ]
            self.logger.debug(f"Simulated performance data retrieved: {len(simulated_performance_data)} items.")

            # --- 2. Simulate Analysis (Placeholder) ---
            # Find the best performing subject line based on open rate, requiring a minimum sample size.
            best_subject_template = self.internal_state.get('preferred_subject_template', "Quick question about [Interest]") # Start with current or default
            highest_open_rate = 0.0
            min_sample_size = 50 # Require minimum samples for significance

            for item in simulated_performance_data:
                # Ensure item has necessary keys before accessing
                if item.get('type') == 'email_subject' and item.get('sample_size', 0) >= min_sample_size:
                    current_open_rate = item.get('open_rate', 0.0)
                    if current_open_rate > highest_open_rate:
                        highest_open_rate = current_open_rate
                        best_subject_template = item.get('content', best_subject_template) # Update template

            self.logger.info(f"Simulated Analysis: Best performing subject template identified: '{best_subject_template}' (Open Rate: {highest_open_rate:.2f})")

            # --- 3. Simulate Strategy Update (Update Internal State) ---
            # Store the learned preference in internal_state.
            # The generate_dynamic_prompt method can later use this state.
            if self.internal_state.get('preferred_subject_template') != best_subject_template:
                self.internal_state['preferred_subject_template'] = best_subject_template
                self.internal_state['last_learning_update_ts'] = datetime.now(timezone.utc)
                self.logger.info(f"Internal state updated with new preferred subject template: '{best_subject_template}'")
            else:
                self.logger.info("No change in preferred subject template based on simulated analysis.")

        except Exception as e:
            self.logger.error(f"Error during learning loop prototype: {e}", exc_info=True)

        # --- Loop Delay ---
        # Keep running periodically
        try:
            # Use a shorter delay for testing/demonstration initially
            await asyncio.sleep(60 * 10) # Run every 10 minutes for now
        except asyncio.CancelledError:
            self.logger.info("Learning loop cancelled.")
            raise # Propagate cancellation

    async def self_critique(self) -> Dict[str, Any]:
        """Method for the agent to evaluate its own performance and strategy."""
        self.logger.info("self_critique: Placeholder - Not yet implemented.")
        # TODO: Implement logic to analyze internal_state['campaign_stats'], compare against goals, identify issues.
        # Potentially use LLM call with specific critique prompt.
        return {"status": "ok", "feedback": "Self-critique not implemented."}

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """
        Constructs context-rich prompts for LLM calls, incorporating internal state and KB insights.
        Prototype v1: Uses internal state for learned preferences, simulates KB retrieval.
        """
        self.logger.debug(f"Generating dynamic prompt for task context: {task_context.get('task')}")

        # 1. Start with the base meta-prompt defining the agent's persona and goals
        prompt_parts = [EMAIL_AGENT_META_PROMPT_UPDATED] # Use updated prompt string

        # 2. Add specific task context
        prompt_parts.append("\n--- Current Task Context ---")
        prompt_parts.append(f"Task: {task_context.get('task', 'Generate email')}")
        prompt_parts.append(f"Desired Outcome: {task_context.get('desired_outcome', 'Drive engagement')}")
        if task_context.get('campaign_id'):
            prompt_parts.append(f"Campaign ID: {task_context['campaign_id']}")

        client_info = task_context.get('client_info', {})
        prompt_parts.append("\n--- Recipient Profile ---")
        prompt_parts.append(f"Name: {client_info.get('name', 'N/A')}")
        prompt_parts.append(f"Email: {client_info.get('email', 'N/A')}")
        prompt_parts.append(f"Country: {client_info.get('country', 'N/A')}")
        prompt_parts.append(f"Interests: {client_info.get('interests', ['general business'])}")
        prompt_parts.append(f"Engagement Score: {client_info.get('engagement_score', 0.1):.2f}")
        prompt_parts.append(f"OSINT Summary: {task_context.get('osint_summary', 'N/A')}")

        # 3. Incorporate Learned Preferences from Internal State
        prompt_parts.append("\n--- Learned Preferences (Internal State) ---")
        preferred_subject = self.internal_state.get('preferred_subject_template', None)
        if preferred_subject:
            prompt_parts.append(f"Preferred Subject Template (Based on recent performance): '{preferred_subject}' - Adapt this template creatively.")
        else:
            prompt_parts.append("No specific subject preference learned yet. Generate a compelling subject.")

        # Add placeholders for other learned elements (hooks, CTAs) as they get implemented
        # preferred_hook = self.internal_state.get('preferred_hook_style', 'direct')
        # prompt_parts.append(f"Preferred Hook Style: {preferred_hook}")

        # 4. Retrieve Relevant Knowledge from KB
        prompt_parts.append("\n--- Relevant Knowledge (KB Retrieval) ---")
        if self.kb_interface:
            try:
                # Query for best performing subject lines for this type of client/campaign (simplified query)
                # TODO: Add more context to query (e.g., client industry, campaign goal)
                subject_perf_query = await self.kb_interface.get_knowledge(
                    type_filter='email_subject',
                    # tag_filter=[f"audience:{client_info.get('industry', 'general')}"], # Example tag filter
                    limit=3 # Get top 3 examples
                    # TODO: Need sorting by performance in get_knowledge or a dedicated method
                )
                if subject_perf_query:
                    prompt_parts.append("Top Performing Subject Examples (Consider adapting):")
                    for item in subject_perf_query:
                        # TODO: Include performance metric (e.g., open rate) once available
                        prompt_parts.append(f"- '{item.get('content')}'")
                else:
                    prompt_parts.append("No specific subject performance data found in KB.")

                # TODO: Query for best hooks, CTAs, body snippets similarly
                # hook_perf_query = await self.kb_interface.get_knowledge(...)
                # cta_perf_query = await self.kb_interface.get_knowledge(...)

            except Exception as kb_err:
                self.logger.error(f"Error retrieving knowledge from KB for dynamic prompt: {kb_err}", exc_info=True)
                prompt_parts.append("KB Error: Could not retrieve relevant knowledge.")
        else:
            prompt_parts.append("KB Interface not available.")

        # 5. Add Specific Instructions for the LLM
        prompt_parts.append("\n--- Instructions ---")
        prompt_parts.append("1. Deeply personalize the email using the recipient profile and OSINT summary.")
        prompt_parts.append("2. Craft a compelling, human-like subject line, strongly considering the preferred template if provided.")
        prompt_parts.append("3. Write engaging body copy incorporating relevant knowledge and interests.")
        prompt_parts.append("4. Ensure the tone is appropriate (professional, persuasive, slightly informal).")
        prompt_parts.append("5. Include a clear Call To Action (CTA), considering the successful example provided.")
        prompt_parts.append("6. **Output Format:** Respond ONLY with a valid JSON object containing two keys: 'subject' (string) and 'body' (string - HTML formatted). Do not include any other text, preamble, or explanation outside the JSON object.")
        prompt_parts.append("```json") # Hint for the LLM

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt (length: {len(final_prompt)} chars)")
        # self.logger.debug(f"Prompt Preview:\n{final_prompt[:500]}...") # Optional: Log prompt preview for debugging
        return final_prompt
