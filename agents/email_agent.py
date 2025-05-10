# Filename: agents/email_agent.py
# Description: Genius Agentic Email Agent - Handles hyper-personalized outreach,
#              IMAP opt-out processing, humanization, compliance, and learning.
# Version: 3.3 (Strategic Content, Adaptive Timing, Proactive Deliverability, Full Code)

import asyncio
import logging
import random
import os
import json
import smtplib
import imaplib
import email
import re
import uuid
import hashlib
import time
from email.message import EmailMessage
from email.header import decode_header
from datetime import datetime, timedelta, timezone
import pytz
from collections import Counter
import pybreaker
import aiohttp

# --- Core Framework Imports ---
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy import update, desc, func, case, String, cast
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import SQLAlchemyError

# --- Project Imports ---
try:
    from .base_agent import GeniusAgentBase
except ImportError:
    logging.warning("Production base agent not found, using GeniusAgentBase. Ensure base_agent_prod.py is used.")
    try:
        from agents.base_agent import GeniusAgentBase
    except ImportError:
        logging.critical("Failed to import GeniusAgentBase from both relative and absolute paths.")
        raise

from models import Client, EmailLog, PromptTemplate, KnowledgeFragment, EmailComposition, LearnedPattern, EmailStyles, KVStore, ConversationState
from config.settings import settings
from utils.database import encrypt_data, decrypt_data
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional, List, Union, Tuple, Type

try:
    from mailersend import emails as MailerSendEmails
    MAILERSEND_AVAILABLE = True
except ImportError:
    MAILERSEND_AVAILABLE = False
    logging.warning("MailerSend SDK not found. Email sending will rely on SMTP fallback (if configured). Install with 'pip install mailersend'")

logger = logging.getLogger(__name__)
op_logger = logging.getLogger('OperationalLog')

EMAIL_AGENT_META_PROMPT = """
You are the EmailAgent (Level 30+ Transmuted) within the Nolli AI Sales System.
Your Core Mandate: Execute hyper-personalized, psychologically potent, and compliant email outreach campaigns to maximize profitable conversions ($10k+/day goal), adapting to real-time feedback and deliverability signals.
Key Responsibilities:
1.  **Hyper-Contextual Content Alchemy:** Generate human-like, irresistible email subjects and bodies. Fuse deep context from ThinkTool (Client data, OSINT, KB insights including enriched Clay data like job title, company details, recent news) with proven `EmailStyles`. Employ LLM self-critique for advanced humanization and pattern interruption in subject lines.
2.  **Dynamic CTA & Value Proposition:** Tailor Calls To Action based on client profile and campaign goal. If micro-assets (demos, case snippets) are available (indicated by ThinkTool), use them in CTAs.
3.  **Compliant & Deliverable Outreach:** Strict adherence to CAN-SPAM/GDPR/CASL. Validate campaigns via LegalAgent. Proactively verify email validity (MailerCheck). Perform internal LLM-based spam trigger pre-checks and request rephrasing if risky. Utilize primary sending service (MailerSend) with robust error handling. Monitor bounce/spam rates via Orchestrator/ThinkTool feedback from webhooks.
4.  **Advanced Engagement Tracking & IMAP Processing:** Embed tracking pixels. Process IMAP replies to identify "STOP" requests for immediate opt-out and to capture positive/neutral replies for ThinkTool analysis.
5.  **Rich Performance Logging & KB Integration:** Meticulously log all email activities to `EmailLog`. Link sent emails to `EmailComposition` (KB fragments, styles used).
6.  **Strategic Learning Loop:** Provide detailed performance data (engagement per segment/style/enriched data point, deliverability issues) to ThinkTool for strategic analysis and adaptation of email strategies, styles, and send-time optimizations.
7.  **Collaboration:** Execute tasks from Orchestrator (originating from ThinkTool). Request validation from LegalAgent. Report outcomes, errors, and critical deliverability insights.
**Goal:** Be the agency's master email alchemist, transmuting data into high-value conversations and conversions through intelligent, adaptive, and compliant email marketing, driven by deep personalization and AI-native insights.
"""

mailer_api_breaker = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=60 * 10, name="EmailAgentMailerAPIs_L30")

class EmailAgent(GeniusAgentBase):
    AGENT_NAME = "EmailAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any, smtp_password: Optional[str] = None, imap_password: Optional[str] = None):
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, session_maker=session_maker, config=getattr(orchestrator, 'config', settings))
        self.meta_prompt = EMAIL_AGENT_META_PROMPT
        self.think_tool = getattr(orchestrator, 'agents', {}).get('think')

        self._smtp_password = smtp_password
        self._imap_password = imap_password

        self.internal_state = getattr(self, 'internal_state', {})
        self.internal_state.setdefault('task_queue', asyncio.Queue())
        self.internal_state['max_concurrency'] = int(self.config.get("EMAIL_AGENT_MAX_CONCURRENCY", 15))
        self.internal_state['send_semaphore'] = asyncio.Semaphore(self.internal_state['max_concurrency'])
        self.internal_state['global_daily_limit'] = int(self.config.get("EMAIL_AGENT_MAX_PER_DAY", 750))
        self.internal_state['global_sent_today'] = 0
        self.internal_state['global_reset_time'] = self._get_next_reset_time_utc()
        self.internal_state['imap_check_interval_seconds'] = int(self.config.get("IMAP_CHECK_INTERVAL_S", 240))
        self.internal_state['tracking_pixel_cache'] = {}
        self.internal_state['mailercheck_cache'] = {}
        self.internal_state['mailercheck_cache_ttl'] = 60 * 60 * 24 * 5
        self.internal_state['active_sends'] = 0
        self.internal_state['spam_check_threshold'] = float(self.config.get("EMAIL_SPAM_CHECK_THRESHOLD", 0.75))
        self.internal_state['learning_interval_seconds'] = int(self.config.get("EMAIL_LEARNING_INTERVAL_S", 3600 * 2))
        self.internal_state['last_learning_run_ts'] = time.time() - self.internal_state['learning_interval_seconds']

        self.mailersend_api_key = self.config.get_secret("MAILERSEND_API_KEY")
        self.mailersend_client = None
        if MAILERSEND_AVAILABLE and self.mailersend_api_key:
            try:
                self.mailersend_client = MailerSendEmails.NewEmail(self.mailersend_api_key)
                self.logger.info("MailerSend client initialized successfully.")
            except Exception as e: self.logger.error(f"Failed to initialize MailerSend client: {e}")
        else: self.logger.warning("MailerSend not available or not configured. Email sending will be heavily impacted.")

        self.mailercheck_api_key = self.config.get_secret("MAILERCHECK_API_KEY")
        if not self.mailercheck_api_key: self.logger.warning("MAILERCHECK_API_KEY not configured. Email verification disabled.")

        self.logger.info(f"{self.AGENT_NAME} v3.3 (L30+ Transmutation) initialized.")

    async def log_operation(self, level: str, message: str):
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err: logger.error(f"Failed to write to OP log from {self.agent_name}: {log_err}")

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        action = task_details.get('action')
        self.logger.info(f"{self.AGENT_NAME} received task: {action}")
        self._status = self.STATUS_EXECUTING
        result = {"status": "failure", "message": f"Unsupported action: {action}"}
        try:
            if action == 'initiate_outreach' or action == 'send_email':
                client_id = task_details.get('client_id')
                target_identifier = task_details.get('content', {}).get('target_identifier')
                campaign_id = task_details.get('content', {}).get('campaign_id')
                enriched_data = task_details.get('content', {}).get('enriched_data')
                goal = task_details.get('content', {}).get('goal', 'initial_engagement')
                email_style_hint = task_details.get('content', {}).get('email_style_hint')
                specific_cta_suggestion = task_details.get('content', {}).get('cta_suggestion')

                if not client_id and not target_identifier:
                     result["message"] = "Missing client_id or target_identifier for outreach task."
                else:
                    asyncio.create_task(self.send_email_task(
                        client_id=client_id, target_identifier=target_identifier,
                        campaign_id=campaign_id, enriched_data=enriched_data, goal=goal,
                        email_style_hint=email_style_hint, specific_cta_suggestion=specific_cta_suggestion
                    ))
                    result = {"status": "success", "message": "Email outreach task initiated."}
            elif action == 'process_imap_replies':
                await self._check_imap_for_replies()
                result = {"status": "success", "message": "IMAP check completed."}
            elif action == 'process_open_tracking':
                tracking_id = task_details.get('tracking_id')
                if tracking_id: await self.process_email_open(tracking_id); result = {"status": "success", "message": "Open tracking processed."}
                else: result["message"] = "Missing tracking_id for open tracking."
            else: self.logger.warning(f"Unsupported action '{action}' for EmailAgent.")
        except Exception as e:
            self.logger.error(f"Error executing EmailAgent task '{action}': {e}", exc_info=True)
            result = {"status": "error", "message": f"Unexpected error: {e}"}
            await self._report_error(f"Task '{action}' failed: {e}")
        finally: self._status = self.STATUS_IDLE
        return result

    async def send_email_task(self,
                              client_id: Optional[int],
                              target_identifier: Optional[str],
                              campaign_id: Optional[int] = None,
                              enriched_data: Optional[Dict[str, Any]] = None,
                              goal: str = 'initial_engagement',
                              email_style_hint: Optional[str] = None,
                              specific_cta_suggestion: Optional[str] = None):
        async with self.internal_state['send_semaphore']:
            self.internal_state['active_sends'] = self.internal_state.get('active_sends', 0) + 1
            log_status = "failed_preparation"; message_id_from_provider = None; subject: Optional[str] = None
            body: Optional[str] = None; recipient: Optional[str] = None; actual_client_id: Optional[int] = client_id
            sender_email_used: Optional[str] = None; composition_ids: Dict[str, Any] = {}; client_obj: Optional[Client] = None
            try:
                if not self._check_global_limit():
                    self.logger.warning(f"Global daily email limit reached. Task deferred for client {actual_client_id or target_identifier}.")
                    if hasattr(self.orchestrator, 'delegate_task'):
                        await self.orchestrator.delegate_task(self.AGENT_NAME, {'action': 'initiate_outreach', 'content': {'client_id': actual_client_id, 'target_identifier': target_identifier, 'campaign_id': campaign_id, 'enriched_data': enriched_data, 'goal': goal, 'email_style_hint': email_style_hint, 'specific_cta_suggestion': specific_cta_suggestion}, 'priority': 9.0})
                    return

                async with self.session_maker() as session:
                    if actual_client_id: client_obj = await session.get(Client, actual_client_id)
                    if not client_obj and target_identifier and '@' in target_identifier:
                         stmt = select(Client).where(Client.email == target_identifier).limit(1)
                         client_obj = (await session.execute(stmt)).scalar_one_or_none()
                         if client_obj: actual_client_id = client_obj.id
                    recipient = client_obj.email if client_obj else target_identifier
                    if not recipient: self.logger.error(f"No recipient email for {actual_client_id or target_identifier}."); return
                    if client_obj and (not client_obj.opt_in or not client_obj.is_deliverable): self.logger.warning(f"Client {actual_client_id} invalid state."); return

                verification_status = await self._verify_email_address(recipient)
                if verification_status in ["syntax_error", "typo", "mailbox_not_found", "disposable", "blocked", "error"]:
                    await self._mark_client_undeliverable(recipient); await self._log_email(actual_client_id, recipient, "Verification Failed", f"Skipped: {verification_status}", "failed_verification", None, {}, None); return
                
                sender_email_used = self.config.get("HOSTINGER_EMAIL")
                if not sender_email_used: self.logger.error("HOSTINGER_EMAIL not configured."); return

                subject, body, composition_ids = await self._generate_email_content_internal(
                    client_obj, campaign_id, enriched_data, goal, email_style_hint, specific_cta_suggestion
                )
                if not subject or not body: raise ValueError("Failed to generate email content.")

                spam_check_result = await self._internal_spam_check(subject, body)
                if spam_check_result.get("is_risky"):
                    self.logger.warning(f"Email to {recipient} flagged as potentially spammy (Score: {spam_check_result.get('score')}). Reason: {spam_check_result.get('issues')}. Logging and proceeding with caution.")
                    await self._log_email(actual_client_id, recipient, subject, f"SPAM RISK: {spam_check_result.get('score')}\n{body}", "flagged_spam_risk", sender_email_used, composition_ids, None)

                compliance_context = f"Email Campaign Send: To={recipient}, Subject='{subject[:50]}...', Client Country: {client_obj.country if client_obj else 'Unknown'}"
                validation_result = await self.orchestrator.delegate_task("LegalAgent", {"action": "validate_operation", "operation_description": compliance_context})
                if not validation_result or validation_result.get('status') != 'success' or not validation_result.get('findings', {}).get('is_compliant'):
                    reason = validation_result.get('findings', {}).get('compliance_issues', ['Validation Failed'])[0] if validation_result else 'Validation Error'
                    self.logger.warning(f"Compliance check failed for {recipient}: {reason}. Skipping.")
                    await self._log_email(actual_client_id, recipient, subject, body, "blocked_compliance", sender_email_used, composition_ids, None); return

                company_address = self.config.SENDER_COMPANY_ADDRESS or "[Configure SENDER_COMPANY_ADDRESS]"
                footer_text = f"\n\n---\n{self.config.SENDER_NAME}\n{self.config.SENDER_TITLE}\n{company_address}\nReply 'STOP' to unsubscribe."
                tracking_pixel_uuid = uuid.uuid4()
                pixel_base_url = str(self.config.AGENCY_BASE_URL).rstrip('/')
                pixel_url = f"{pixel_base_url}/track/{tracking_pixel_uuid}.png"
                
                body += f'<img src="{pixel_url}" width="1" height="1" alt="" style="display:none;"/>'
                footer_html_compatible = footer_text.replace('\n', '<br>')
                body += f"<p style='font-size:10px; color:#888;'>{footer_html_compatible}</p>"

                if client_obj: await self._wait_for_optimal_send_time(client_obj)
                await self._internal_think(f"Pre-Send Checklist: To={recipient}, Subject='{subject[:30]}...', Compliance=OK, Verification={verification_status}, Sender={sender_email_used}.")
                await self._apply_throttling()

                send_result = await self._send_email_mailersend(recipient, subject, body, sender_email_used)
                send_success = send_result.get("status") == "success"
                message_id_from_provider = send_result.get("message_id")

                if send_success:
                    log_status = "sent"; self.internal_state['global_sent_today'] += 1
                    self.logger.info(f"Email SENT to {recipient} via MailerSend. X-Message-Id: {message_id_from_provider}")
                    if actual_client_id:
                        async with self.session_maker() as session:
                            await session.execute(update(Client).where(Client.id == actual_client_id).values(last_contacted_at=datetime.now(timezone.utc)))
                            await session.commit()
                else: log_status = "failed_send"; self.logger.warning(f"Email FAILED for {recipient}. Reason: {send_result.get('message')}")

                email_log_entry = await self._log_email(actual_client_id, recipient, subject, body, log_status, "MailerSend", composition_ids, message_id_from_provider)
                if email_log_entry: self.internal_state['tracking_pixel_cache'][str(tracking_pixel_uuid)] = email_log_entry.id
            except Exception as e:
                self.logger.error(f"Unhandled error in send_email_task for {actual_client_id or target_identifier}: {e}", exc_info=True)
                log_status = "error_internal"
                await self._log_email(actual_client_id, recipient or str(target_identifier or "Unknown Recipient"), subject or "ERROR", str(e), log_status, "MailerSend", None, None)
                await self._report_error(f"Send email task failed for {actual_client_id or target_identifier}: {e}")
            finally: self.internal_state['active_sends'] = max(0, self.internal_state.get('active_sends', 1) - 1)

    @mailer_api_breaker
    async def _verify_email_address(self, email_address: str) -> str:
        if not self.mailercheck_api_key: return "skipped_no_key"
        cached = self.internal_state['mailercheck_cache'].get(email_address)
        if cached and time.time() < cached['timestamp'] + self.internal_state['mailercheck_cache_ttl']:
            return cached['status']
        api_url = "https://app.mailercheck.com/api/check/single"
        headers = {'Authorization': f'Bearer {self.mailercheck_api_key}', 'Content-Type': 'application/json', 'Accept': 'application/json'}
        payload = {"email": email_address}
        await self._internal_think(f"Verifying email via MailerCheck: {email_address}")
        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                async with session.post(api_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json(); status = result.get('status', 'error')
                        self.internal_state['mailercheck_cache'][email_address] = {'status': status, 'timestamp': time.time()}
                        return status
                    elif response.status == 429: self.logger.warning(f"MailerCheck rate limit for {email_address}."); return "unknown"
                    else: self.logger.error(f"MailerCheck API error ({response.status}) for {email_address}: {await response.text()}"); return "error"
        except asyncio.TimeoutError: self.logger.error(f"Timeout calling MailerCheck API for {email_address}"); return "error"
        except aiohttp.ClientError as e: self.logger.error(f"Network error calling MailerCheck API for {email_address}: {e}"); return "error"
        except Exception as e: self.logger.error(f"Unexpected error during MailerCheck for {email_address}: {e}", exc_info=True); return "error"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=10, max=60), retry=retry_if_exception_type(Exception))
    @mailer_api_breaker
    async def _send_email_mailersend(self, recipient: str, subject: str, body_html: str, sender_email: str) -> Dict[str, Any]:
        if not self.mailersend_client: return {"status": "failure", "message": "MailerSend client not initialized."}
        result = {"status": "failure", "message": "MailerSend send initialization failed."}; message_id = None
        try:
            mail_body = {}
            mail_from = {"name": self.config.SENDER_NAME, "email": sender_email}
            recipients = [{"email": recipient}]; reply_to = {"email": sender_email}
            self.mailersend_client.set_mail_from(mail_from, mail_body)
            self.mailersend_client.set_mail_to(recipients, mail_body)
            self.mailersend_client.set_subject(subject, mail_body)
            self.mailersend_client.set_html_content(body_html, mail_body)
            self.mailersend_client.set_plaintext_content(self._html_to_plain_text(body_html), mail_body)
            self.mailersend_client.set_reply_to(reply_to, mail_body)
            await self._internal_think(f"Sending email via MailerSend to {recipient}")
            response = await asyncio.to_thread(self.mailersend_client.send, mail_body)
            if 200 <= response.status_code < 300:
                msg_id_key = next((k for k in response.headers if k.lower() == 'x-message-id'), None)
                if msg_id_key: message_id = response.headers[msg_id_key]
                result = {"status": "success", "message": "Email queued successfully via MailerSend.", "message_id": message_id}
            else:
                error_details = response.body if hasattr(response, 'body') else await asyncio.to_thread(getattr, response, 'text', 'No details')
                result["message"] = f"MailerSend API Error ({response.status_code}): {str(error_details)[:200]}"
                if response.status_code >= 500 or response.status_code == 429: raise Exception(f"MailerSend server/rate limit error: {response.status_code}")
        except Exception as e: self.logger.error(f"Error sending email via MailerSend to {recipient}: {e}", exc_info=True); result["message"] = f"MailerSend error: {e}"; raise
        return result

    async def _internal_spam_check(self, subject: str, body_html: str) -> Dict[str, Any]:
        if not self.orchestrator or not hasattr(self.orchestrator, 'call_llm'):
            return {"is_risky": False, "score": 0.0, "issues": ["Spam check LLM unavailable."]}
        
        plain_text_body = self._html_to_plain_text(body_html)
        spam_check_context = {
            "task": "Analyze email for spam triggers",
            "email_subject": subject,
            "email_body_plain_text_snippet": plain_text_body[:1500],
            "desired_output_format": "JSON: {\"spam_risk_score\": float (0.0 to 1.0, where 1.0 is high risk), \"potential_spam_trigger_keywords\": [str], \"overall_assessment\": \"<brief textual assessment of risk>\"}"
        }
        prompt = await self.generate_dynamic_prompt(spam_check_context)
        llm_model_pref = settings.OPENROUTER_MODELS.get('think_validate', 'mistralai/mistral-7b-instruct')

        try:
            response_data = await self.orchestrator.call_llm(
                agent_name=self.AGENT_NAME, prompt=prompt, model_preference=[llm_model_pref],
                temperature=0.2, max_tokens=300, is_json_output=True
            )
            response_str = response_data.get('content') if isinstance(response_data, dict) else None
            if response_str:
                parsed_response = self._parse_llm_json(response_str)
                if parsed_response and "spam_risk_score" in parsed_response:
                    score = float(parsed_response.get("spam_risk_score", 0.0))
                    return {
                        "is_risky": score >= self.internal_state['spam_check_threshold'],
                        "score": score,
                        "issues": parsed_response.get("potential_spam_trigger_keywords", []),
                        "assessment": parsed_response.get("overall_assessment", "")
                    }
        except Exception as e: self.logger.warning(f"Internal spam check LLM call failed: {e}")
        return {"is_risky": False, "score": 0.0, "issues": ["Spam check execution error."]}

    async def _generate_email_content_internal(self,
                                               client: Optional[Client],
                                               campaign_id: Optional[int],
                                               enriched_data: Optional[Dict[str, Any]] = None,
                                               goal: str = 'initial_engagement',
                                               email_style_hint: Optional[str] = None,
                                               specific_cta_suggestion: Optional[str] = None
                                               ) -> tuple[Optional[str], Optional[str], Dict[str, Any]]:
        subject: Optional[str] = None; body: Optional[str] = None; composition_ids: Dict[str, Any] = {}
        client_info_dict = {}; client_id_for_log = None; recipient_email = None
        if client:
             client_info_dict = { "id": client.id, "name": client.name, "email": client.email, "country": client.country, "interests": client.interests, "engagement_score": client.engagement_score, "timezone": client.timezone, "company": client.company, "job_title": client.job_title }
             client_id_for_log = client.id; recipient_email = client.email
        elif enriched_data and enriched_data.get('verified_email'):
             recipient_email = enriched_data.get('verified_email')
             client_info_dict = { "name": enriched_data.get('full_name', 'Valued Prospect'), "email": recipient_email, "job_title": enriched_data.get('job_title'), "company_name": enriched_data.get('company_name')}
        else: self.logger.error("Cannot generate email: No client or enriched data with email."); return None, None, {}
        if not recipient_email: self.logger.error("Cannot generate email: Recipient email undetermined."); return None, None, {}

        try:
            osint_summary = "No specific OSINT data available."; successful_styles_content = []
            if client_id_for_log and self.think_tool:
                osint_result = await self.orchestrator.delegate_task("ThinkTool", {"action": "fetch_osint_summary", "client_id": client_id_for_log})
                if osint_result and osint_result.get('status') == 'success': osint_summary = osint_result.get('summary', osint_summary)

            if self.think_tool:
                style_query_tags = ["email_style", "success_template"]
                if email_style_hint: style_query_tags.append(email_style_hint.lower().replace(" ","_"))
                style_fragments = await self.think_tool.query_knowledge_base(data_types=["EmailStyleExemplar"], tags=style_query_tags, limit=3, min_relevance=0.7)
                for frag in style_fragments:
                    try: successful_styles_content.append(json.loads(frag.content))
                    except: self.logger.warning(f"Could not parse EmailStyleExemplar KF ID {frag.id}")

            task_context = {
                "client_info": client_info_dict, "osint_summary": osint_summary, "campaign_id": campaign_id,
                "task": "Generate Hyper-Personalized Email (Subject & HTML Body)", "goal": goal,
                "successful_style_exemplars": successful_styles_content,
                "cta_suggestion_from_thinktool": specific_cta_suggestion,
                "desired_output_format": "JSON: {\"subject\": \"<Compelling, pattern-interrupt subject line incorporating urgency/curiosity/personalization>\", \"body\": \"<Engaging HTML email body. Weave in client's inferred pain points/aspirations. Use social proof if applicable. Ensure a clear, dynamic Call-To-Action. Maintain a human-like, persuasive tone. Reference enriched data subtly. Consider a slightly unconventional hook if appropriate for the brand voice and goal.>\", \"confidence_score\": float (0.0-1.0 for content quality and alignment with instructions), \"suggested_micro_asset_to_link\": \"<If applicable, suggest a specific micro-asset (e.g., 'short_demo_video_X') that ThinkTool might have, to include in CTA>\"}"
            }
            if enriched_data:
                filtered_enriched = { k: v for k, v in enriched_data.items() if k in ['verified_email', 'job_title', 'company_name', 'full_name', 'linkedin_url', 'industry', 'company_size', 'location', 'recent_activity', 'company_news', 'tech_stack_snippet'] and v }
                if filtered_enriched: task_context['enriched_data_available'] = filtered_enriched
            
            comprehensive_prompt = await self.generate_dynamic_prompt(task_context)
            llm_model_pref = settings.OPENROUTER_MODELS.get('email_draft')
            llm_response_str = await self._call_llm_with_retry(comprehensive_prompt, model=llm_model_pref, temperature=0.6, max_tokens=2500, is_json_output=True)
            if not llm_response_str: raise Exception("LLM call failed to return content for email generation.")
            
            parsed_json = self._parse_llm_json(llm_response_str)
            if not parsed_json: raise ValueError("Failed to parse LLM response for email content.")
            subject = parsed_json.get('subject'); body = parsed_json.get('body')
            if not subject or not body: raise ValueError("LLM response missing subject or body.")

            humanization_prompt = f"Review this email for natural human tone. Suggest minor tweaks if it sounds too formal or AI-like. Output the improved subject and body in JSON if changes are made, otherwise output original.\n\nSubject: {subject}\n\nBody:\n{body}\n\nOutput JSON: {{\"subject\": \"string\", \"body\": \"string (HTML formatted)\", \"changes_made\": bool}}"
            llm_humanize_model = settings.OPENROUTER_MODELS.get('email_humanize')
            verdict_json_str = await self._call_llm_with_retry(humanization_prompt, model=llm_humanize_model, temperature=0.2, max_tokens=1800, is_json_output=True)
            if verdict_json_str:
                verdict_data = self._parse_llm_json(verdict_json_str)
                if verdict_data and verdict_data.get("changes_made"):
                    subject = verdict_data.get('subject', subject); body = verdict_data.get('body', body)
                    self.logger.info(f"Email for {recipient_email} polished for humanization.")
            
            subject = subject.strip().replace('"', ''); self.logger.info(f"Generated email content for {recipient_email}")
            return subject, body, composition_ids
        except Exception as e: self.logger.error(f"Internal email content generation failed for {recipient_email}: {e}", exc_info=True); return None, None, {}

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        self.logger.debug(f"Generating dynamic prompt for EmailAgent task: {task_context.get('task')}")
        prompt_parts = [self.meta_prompt]
        prompt_parts.append("\n--- Current Task Context ---")
        priority_keys = ['task', 'goal', 'client_info', 'enriched_data_available', 'osint_summary', 'successful_style_exemplars', 'campaign_id', 'cta_suggestion_from_thinktool', 'email_subject', 'email_body_plain_text_snippet']
        for key in priority_keys:
            if key in task_context and task_context[key] is not None:
                value = task_context[key]; value_str = ""; max_len = 2000
                if key in ['osint_summary', 'successful_style_exemplars', 'enriched_data_available', 'email_body_plain_text_snippet']: max_len = 4000
                if isinstance(value, str): value_str = value[:max_len] + ("..." if len(value) > max_len else "")
                elif isinstance(value, (int, float, bool)): value_str = str(value)
                elif isinstance(value, dict):
                    if key == 'enriched_data_available': value_str = ", ".join([f"{k}: {v}" for k, v in value.items() if v])[:max_len] + "..."
                    else:
                         try: value_str = json.dumps(value, default=str, indent=2); value_str = value_str[:max_len] + ("..." if len(value_str) > max_len else "")
                         except TypeError: value_str = str(value)[:max_len] + "..."
                elif isinstance(value, list):
                    if key == 'successful_style_exemplars' and value:
                        summaries = []
                        for item in value[:3]:
                            if isinstance(item, dict): summaries.append(f"- Subject Hint: {item.get('subject_template','')[:50]}... Body Hint: {item.get('body_template','')[:100]}...")
                            else: summaries.append(str(item)[:150]+"...")
                        value_str = "\n".join(summaries)
                    else: value_str = json.dumps(value, default=str)[:max_len] + "..."
                else: value_str = str(value)[:max_len] + "..."
                prompt_key = "Enriched Prospect Data" if key == 'enriched_data_available' else key.replace('_', ' ').title()
                prompt_parts.append(f"**{prompt_key}**: {value_str}")
        
        other_params = {k: v for k, v in task_context.items() if k not in priority_keys and k not in ['desired_output_format']}
        if other_params:
            prompt_parts.append("\n**Other Parameters:**")
            try: prompt_parts.append(f"```json\n{json.dumps(other_params, default=str, indent=2)}\n```")
            except TypeError: prompt_parts.append(str(other_params)[:500] + "...")
        
        prompt_parts.append("\n--- Instructions ---")
        task_type = task_context.get('task')
        if task_type == 'Generate Hyper-Personalized Email (Subject & HTML Body)':
            prompt_parts.append("1. **Deep Personalization & Psychological Hooks:** Analyze ALL provided context (Client Info, Enriched Data, OSINT, Style Exemplars). Identify core pain points, aspirations, and recent activities. Craft a subject line that acts as a powerful pattern interrupt (e.g., highly specific question, surprising insight, reference to a very recent event/post from OSINT).")
            prompt_parts.append("2. **Value-Driven Body (HTML):** Write an engaging HTML body. Directly address the inferred pain/aspiration. Weave in enriched data points *naturally* to build rapport and demonstrate research. If 'Successful Style Exemplars' are provided, fuse their effective elements (tone, structure, phrasing) with the current context. If 'CTA Suggestion from ThinkTool' is present, build towards it. If it suggests a micro-asset, frame the CTA around accessing that asset.")
            prompt_parts.append("3. **Tone & Voice:** Adapt tone based on client profile, industry, and successful exemplars. Options: empathetic problem-solver, insightful challenger, peer-to-peer expert, slightly unconventional but valuable. Avoid generic sales speak.")
            prompt_parts.append(f"4. **Dynamic Call To Action (CTA):** Based on the 'Goal' ('{task_context.get('goal', 'initial_engagement')}') and any 'CTA Suggestion from ThinkTool', formulate a clear, low-friction CTA. This might be booking a brief call, viewing a targeted demo/resource, or simply gauging interest with a specific question.")
            prompt_parts.append("5. **Grey Area Consideration (Subtlety):** If a 'Grey Area Angle' is provided by ThinkTool, incorporate its essence subtly to maximize intrigue without triggering spam filters or appearing unprofessional. This is about clever framing, not overt rule-breaking.")
        elif task_type == 'Analyze email for spam triggers':
             prompt_parts.append("Analyze the Subject and Body for potential spam triggers (e.g., excessive caps, spammy words like 'free', 'guarantee', '$', misleading claims, excessive links, poor formatting, urgency without clear value, image-to-text ratio if applicable). Assign a spam risk score (0.0=safe, 1.0=high_risk). List specific trigger keywords/phrases found and provide a brief overall assessment.")
        elif task_type == 'Critique email human-likeness':
             prompt_parts.append("Analyze the provided email draft. Does it sound like a natural human wrote it, or does it sound robotic/AI-generated? Focus on tone, flow, word choice, and common AI patterns.")
             prompt_parts.append(f"**Output Format:** Respond ONLY with 'Human-like' or 'Robotic'.")
        elif task_type == 'Rewrite email for humanization':
             prompt_parts.append("Rewrite the provided email to sound significantly more human and natural. Improve flow, vary sentence structure, use less formal language where appropriate, but retain the core message and call to action.")
        else: prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")
        if task_context.get('desired_output_format'): prompt_parts.append(f"\n**Output Format:** {task_context['desired_output_format']}")
        if "JSON" in task_context.get('desired_output_format', ''): prompt_parts.append("\nRespond ONLY with valid JSON.\n```json")
        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for EmailAgent (length: {len(final_prompt)} chars)")
        return final_prompt

    def _get_next_reset_time_utc(self):
        now_utc = datetime.now(timezone.utc)
        return (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    def _check_global_limit(self) -> bool:
        if datetime.now(timezone.utc) > self.internal_state['global_reset_time']:
            self.logger.info(f"Resetting global daily email limit. Previous count: {self.internal_state['global_sent_today']}")
            self.internal_state['global_sent_today'] = 0
            self.internal_state['global_reset_time'] = self._get_next_reset_time_utc()
        return self.internal_state['global_sent_today'] < self.internal_state['global_daily_limit']

    async def _wait_for_optimal_send_time(self, client: Client):
        optimal_hour_local = 10
        client_tz_str = client.timezone or "America/New_York"
        try: client_tz = pytz.timezone(client_tz_str)
        except pytz.UnknownTimeZoneError: client_tz = pytz.timezone("America/New_York")

        if self.think_tool and hasattr(self.orchestrator, 'delegate_task'):
            thinktool_directive = {
                "action": "get_optimal_send_time_pattern",
                "content": { "client_industry": client.industry, "client_country": client.country, "client_job_title_category": client.job_title, "current_day_of_week_local": datetime.now(client_tz).weekday() }
            }
            try:
                pattern_result = await self.orchestrator.delegate_task("ThinkTool", thinktool_directive)
                if pattern_result and pattern_result.get("status") == "success" and pattern_result.get("findings", {}).get("optimal_hour_local"):
                    optimal_hour_local = int(pattern_result["findings"]["optimal_hour_local"])
                    self.logger.info(f"ThinkTool suggested optimal send hour {optimal_hour_local} {client_tz_str} for client {client.id}")
                else:
                    if self.session_maker:
                        async with self.session_maker() as session:
                             ninety_days_ago = datetime.now(timezone.utc) - timedelta(days=90)
                             stmt = select(EmailLog.opened_at).where(EmailLog.client_id == client.id, EmailLog.opened_at.isnot(None), EmailLog.timestamp >= ninety_days_ago).order_by(desc(EmailLog.opened_at)).limit(20)
                             db_result = await session.execute(stmt)
                             open_times_utc = [row.opened_at for row in db_result.mappings().all() if row.opened_at]
                             if len(open_times_utc) >= 3:
                                 open_hours_local = [t.astimezone(client_tz).hour for t in open_times_utc]
                                 hour_counts = Counter(h for h in open_hours_local if 8 <= h <= 17)
                                 if hour_counts: optimal_hour_local_tuple = hour_counts.most_common(1); optimal_hour_local = optimal_hour_local_tuple if optimal_hour_local_tuple else 10
            except Exception as e_pattern: self.logger.warning(f"Error getting optimal send pattern from ThinkTool: {e_pattern}. Using default/client history.")
        
        now_local = datetime.now(client_tz)
        send_time_local = now_local.replace(hour=optimal_hour_local, minute=random.randint(0, 59), second=random.randint(0,59), microsecond=0)
        if send_time_local <= now_local: send_time_local += timedelta(days=1)
        delay_seconds = (send_time_local.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds()
        if delay_seconds > 0:
            delay_seconds = min(delay_seconds, 60 * 60 * 23)
            if delay_seconds > 60: self.logger.info(f"Optimal Send Time: Waiting {delay_seconds/3600:.1f}h to send to client {client.id} at {send_time_local.strftime('%Y-%m-%d %H:%M')} {client_tz_str}")
            await asyncio.sleep(delay_seconds)

    async def _log_email(self, client_id: Optional[int], recipient: str, subject: str, body: str, status: str, sender_service: Optional[str], composition_ids: Optional[Dict[str, Any]] = None, message_id: Optional[str] = None) -> Optional[EmailLog]:
        if not self.session_maker: return None
        try:
            preview = self._html_to_plain_text(body)[:250] if body else None
            async with self.session_maker() as session:
                async with session.begin():
                    log = EmailLog(client_id=client_id, recipient=recipient, subject=subject, content_preview=preview, status=status, timestamp=datetime.now(timezone.utc), agent_version=f"{self.AGENT_NAME}_v3.3", sender_account=sender_service, message_id=message_id)
                    session.add(log)
                await session.refresh(log)
                return log
        except Exception as e: self.logger.error(f"Error logging email for {recipient}: {e}", exc_info=True); return None

    async def process_email_open(self, tracking_id: str):
        self.logger.info(f"Processing email open for tracking ID: {tracking_id}")
        email_log_id = self.internal_state['tracking_pixel_cache'].pop(tracking_id, None)
        if not email_log_id: self.logger.warning(f"Could not map tracking ID {tracking_id} to EmailLog ID."); return
        try:
            async with self.session_maker() as session:
                async with session.begin():
                    stmt = update(EmailLog).where(EmailLog.id == email_log_id, EmailLog.opened_at == None).values(opened_at=datetime.now(timezone.utc), status='opened').returning(EmailLog.client_id)
                    result = await session.execute(stmt)
                    updated_client_id = result.scalar_one_or_none()
                if updated_client_id: await self._update_client_engagement(updated_client_id, score_increase=0.15)
        except Exception as e: self.logger.error(f"Error processing email open for EmailLog ID {email_log_id}: {e}", exc_info=True)

    async def process_email_reply(self, original_message_id: str, reply_from: str, reply_body: str):
        self.logger.info(f"Processing email reply linked to Message-ID: {original_message_id}")
        try:
            async with self.session_maker() as session:
                async with session.begin():
                    stmt_log = select(EmailLog).where(EmailLog.message_id == original_message_id).limit(1).with_for_update()
                    email_log = (await session.execute(stmt_log)).scalar_one_or_none()
                    if not email_log:
                        stmt_log_fallback = select(EmailLog).where(EmailLog.recipient == reply_from).order_by(desc(EmailLog.timestamp)).limit(1).with_for_update()
                        email_log = (await session.execute(stmt_log_fallback)).scalar_one_or_none()
                        if not email_log: self.logger.warning(f"No matching EmailLog for reply from {reply_from}."); return
                    if email_log.status == 'responded': self.logger.info(f"EmailLog {email_log.id} already responded."); return
                    now_ts = datetime.now(timezone.utc)
                    email_log.status = 'responded'; email_log.responded_at = now_ts
                    if not email_log.opened_at: email_log.opened_at = now_ts
                    client_id = email_log.client_id
                if client_id: await self._update_client_engagement(client_id, score_increase=2.0, interaction_time=now_ts)
                if self.think_tool:
                    log_task = {"action": "log_knowledge_fragment", "fragment_data": {"agent_source": "EmailReplyProcessor", "data_type": "email_reply_content", "content": {"original_message_id": original_message_id, "from": reply_from, "body": reply_body}, "tags": ["email_reply", f"client_{client_id}" if client_id else "unknown_client"], "relevance_score": 0.85, "related_client_id": client_id, "source_reference": f"EmailLog:{email_log.id}"}}
                    await self.orchestrator.delegate_task("ThinkTool", log_task)
                    analysis_task = {"action": "analyze_email_reply", "reply_content": reply_body, "sender": reply_from, "client_id": client_id, "original_subject": email_log.subject, "email_log_id": email_log.id}
                    await self.orchestrator.delegate_task("ThinkTool", analysis_task)
        except Exception as e: self.logger.error(f"Error processing email reply for Message-ID {original_message_id}: {e}", exc_info=True)

    async def _update_client_engagement(self, client_id: int, score_increase: float, interaction_time: Optional[datetime] = None):
        if not client_id: return
        interaction_time = interaction_time or datetime.now(timezone.utc)
        try:
            async with self.session_maker() as session:
                async with session.begin():
                    stmt = update(Client).where(Client.id == client_id).values(engagement_score = func.coalesce(Client.engagement_score, 0) + score_increase, last_interaction = interaction_time)
                    await session.execute(stmt)
        except Exception as e: self.logger.error(f"Error updating engagement for Client {client_id}: {e}", exc_info=True)

    async def _check_imap_for_replies(self):
        if not self._imap_password: self.logger.warning("IMAP password not configured. Skipping reply check."); return
        self.logger.info("Checking Hostinger IMAP for replies/opt-outs...")
        host = self.config.HOSTINGER_IMAP_HOST; port = int(self.config.HOSTINGER_IMAP_PORT)
        user = self.config.HOSTINGER_IMAP_USER; password = self._imap_password
        if not all([host, user, password]): self.logger.error("IMAP config missing."); return
        mail = None; processed_replies = 0; processed_optouts = 0
        try:
            loop = asyncio.get_running_loop()
            def connect_and_login():
                m = imaplib.IMAP4_SSL(host, port); m.login(user, password); m.select("inbox"); return m
            mail = await loop.run_in_executor(None, connect_and_login)
            status, messages_bytes_list_outer = await loop.run_in_executor(None, mail.search, None, '(UNSEEN)')
            if status != 'OK': self.logger.error(f"IMAP search failed: {messages_bytes_list_outer}"); return
            
            if not messages_bytes_list_outer or not messages_bytes_list_outer:
                self.logger.info("No unseen emails found."); return
            
            email_ids_bytes_str = messages_bytes_list_outer
            email_ids_bytes_list = email_ids_bytes_str.split() # Corrected: split the string of IDs

            if not email_ids_bytes_list: self.logger.info("No unseen email IDs after split."); return
            self.logger.info(f"Found {len(email_ids_bytes_list)} unseen emails. Processing...")

            for email_id_bytes_single in reversed(email_ids_bytes_list):
                try:
                    status, msg_data_list = await loop.run_in_executor(None, mail.fetch, email_id_bytes_single, '(RFC822)')
                    if status != 'OK' or not msg_data_list or not isinstance(msg_data_list, tuple) or not isinstance(msg_data_list, bytes):
                        self.logger.warning(f"Failed to fetch or invalid data for email ID {email_id_bytes_single.decode()}")
                        continue
                    
                    full_msg_bytes = msg_data_list
                    full_msg = email.message_from_bytes(full_msg_bytes)
                    
                    original_message_id = None; in_reply_to = full_msg.get('In-Reply-To'); references = full_msg.get('References')
                    subject = self._decode_header(full_msg.get('Subject', '')); sender_tuple = email.utils.parseaddr(full_msg.get('From', ''))
                    sender_email_addr = sender_tuple if sender_tuple and len(sender_tuple) > 1 and sender_tuple else "unknown@example.com"

                    if in_reply_to: original_message_id = in_reply_to.strip('<>')
                    elif references:
                        ref_ids = references.split(); potential_ids = [ref.strip('<>') for ref in ref_ids if '@' in ref]
                        if potential_ids: original_message_id = potential_ids
                    
                    reply_body = self._get_email_body(full_msg)
                    if self._is_stop_request(subject, reply_body):
                        await self._mark_client_undeliverable(sender_email_addr, opt_out=True); processed_optouts += 1
                    elif original_message_id:
                        await self.process_email_reply(original_message_id, sender_email_addr, reply_body); processed_replies += 1
                    
                    await loop.run_in_executor(None, mail.store, email_id_bytes_single, '+FLAGS', '\\Seen')
                except Exception as fetch_err: self.logger.error(f"Error processing email ID {email_id_bytes_single.decode()}: {fetch_err}", exc_info=True)
            self.logger.info(f"IMAP check: Replies: {processed_replies}, Opt-Outs: {processed_optouts}.")
        except imaplib.IMAP4.error as imap_err: self.logger.error(f"IMAP error: {imap_err}", exc_info=True)
        except Exception as e: self.logger.error(f"Unexpected IMAP error: {e}", exc_info=True)
        finally:
            if mail: 
                try: 
                    await loop.run_in_executor(None, mail.logout)
                except Exception: # Silently pass on logout errors, as connection might be dead
                    pass


    def _decode_header(self, header_value: Optional[str]) -> str:
        if not header_value: return ""
        try:
            decoded_parts = decode_header(header_value); header_str = ""
            for part, encoding in decoded_parts:
                if isinstance(part, bytes): header_str += part.decode(encoding or 'utf-8', errors='replace')
                elif isinstance(part, str): header_str += part
            return header_str
        except Exception: return str(header_value)

    def _get_email_body(self, msg: email.message.Message) -> str:
        body = "";
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type(); cdispo = str(part.get('Content-Disposition'))
                if ctype == 'text/plain' and 'attachment' not in cdispo:
                    try: payload = part.get_payload(decode=True); charset = part.get_content_charset() or 'utf-8'; body = payload.decode(charset, errors='replace'); break
                    except: pass
            if not body:
                 for part in msg.walk():
                     ctype = part.get_content_type(); cdispo = str(part.get('Content-Disposition'))
                     if ctype == 'text/html' and 'attachment' not in cdispo:
                          try: payload = part.get_payload(decode=True); charset = part.get_content_charset() or 'utf-8'; html_body = payload.decode(charset, errors='replace'); body = self._html_to_plain_text(html_body); break
                          except: pass
        else:
            try:
                payload = msg.get_payload(decode=True); charset = msg.get_content_charset() or 'utf-8'
                if msg.get_content_type() == 'text/plain': body = payload.decode(charset, errors='replace')
                elif msg.get_content_type() == 'text/html': body = self._html_to_plain_text(payload.decode(charset, errors='replace'))
            except: pass
        return body.strip()

    def _is_stop_request(self, subject: str, body: str) -> bool:
        stop_keywords = ['stop', 'unsubscribe', 'remove me', 'opt out', 'opt-out', 'no more emails']
        if any(keyword in subject.lower() for keyword in stop_keywords): return True
        body_preview = body[:500]
        if any(re.search(rf'\b{re.escape(keyword)}\b', body_preview, re.IGNORECASE) for keyword in stop_keywords): return True
        return False

    async def _mark_client_undeliverable(self, email_address: str, opt_out: bool = False):
        if not email_address: return
        update_values = {'opt_in': False} if opt_out else {'is_deliverable': False}
        log_msg = f"Marking client {email_address} as {'opted-out' if opt_out else 'undeliverable'}."
        self.logger.warning(log_msg); await self.log_operation('warning', log_msg)
        try:
            async with self.session_maker() as session:
                async with session.begin():
                    stmt = update(Client).where(Client.email == email_address).values(**update_values)
                    await session.execute(stmt)
        except Exception as e: self.logger.error(f"Failed to mark client {email_address} status: {e}", exc_info=True)

    def _html_to_plain_text(self, html_content: str) -> str:
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for script_or_style in soup(["script", "style"]): script_or_style.decompose()
            text = soup.get_text(separator='\n', strip=True)
            return re.sub(r'\n\s*\n+', '\n\n', text).strip()
        except Exception: return re.sub('<[^>]+>', ' ', html_content).strip()

    async def _apply_throttling(self):
         min_delay = int(self.config.get("EMAIL_THROTTLE_MIN_S", 5))
         max_delay = int(self.config.get("EMAIL_THROTTLE_MAX_S", 25))
         throttle_delay = random.uniform(min_delay, max_delay)
         self.logger.debug(f"Throttling send for {throttle_delay:.2f}s")
         await asyncio.sleep(throttle_delay)

    async def run(self):
        if self._status == self.STATUS_RUNNING: self.logger.warning("EmailAgent run() called while already running."); return
        self.logger.info("EmailAgent starting run loop...")
        self._status = self.STATUS_RUNNING

        imap_task = asyncio.create_task(self._run_imap_checker_loop(), name=f"{self.AGENT_NAME}_IMAPChecker")
        self._background_tasks.add(imap_task)
        
        if 'task_queue' not in self.internal_state or not isinstance(self.internal_state['task_queue'], asyncio.Queue):
            self.internal_state['task_queue'] = asyncio.Queue()
        queue_processor_task = asyncio.create_task(self._process_task_queue(self.internal_state['task_queue']), name=f"{self.AGENT_NAME}_QueueProcessor")
        self._background_tasks.add(queue_processor_task)
        
        learning_task = asyncio.create_task(self._learning_loop_wrapper(), name=f"{self.AGENT_NAME}_LearningLoop")
        self._background_tasks.add(learning_task)

        while not self._stop_event.is_set():
            await asyncio.sleep(1)
        self.logger.info("EmailAgent run loop received stop signal.")

    async def _run_imap_checker_loop(self):
        self.logger.info("Starting background IMAP checker loop.")
        while not self._stop_event.is_set():
            try:
                await self._check_imap_for_replies()
                await asyncio.sleep(self.internal_state['imap_check_interval_seconds'])
            except asyncio.CancelledError: self.logger.info("IMAP checker loop cancelled."); break
            except Exception as e: self.logger.error(f"Error in IMAP checker loop: {e}", exc_info=True); await asyncio.sleep(self.internal_state['imap_check_interval_seconds'] * 2)

    async def plan_task(self, task_details: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]: return None
    async def execute_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]: return {"status": "failure", "message": "EmailAgent does not use planned steps."}
    
    async def learning_loop(self):
        self.logger.info(f"{self.AGENT_NAME} L30+ learning loop: Analyzing email performance for ThinkTool.")
        while not self._stop_event.is_set():
            try:
                learn_interval = self.internal_state.get('learning_interval_seconds', 3600 * 2)
                await asyncio.sleep(learn_interval)
                if self._stop_event.is_set(): break

                await self._internal_think("EmailAgent Learning Cycle: Gathering performance data for ThinkTool analysis.")
                if not self.think_tool or not self.session_maker:
                    self.logger.warning("ThinkTool or session_maker unavailable for EmailAgent learning.")
                    continue

                async with self.session_maker() as session:
                    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
                    stmt = select(
                        EmailLog.status, EmailLog.recipient, EmailLog.subject,
                        EmailLog.sender_account, EmailLog.timestamp, EmailLog.opened_at,
                        EmailLog.responded_at, Client.industry.label("client_industry"),
                        Client.engagement_score.label("client_engagement_score")
                    ).join(Client, EmailLog.client_id == Client.id, isouter=True)\
                     .where(EmailLog.timestamp >= seven_days_ago)\
                     .limit(1000)
                    results = await session.execute(stmt)
                    performance_data = [dict(row) for row in results.mappings().all()]

                if not performance_data:
                    self.logger.info("EmailAgent Learning: No significant email performance data in the last 7 days.")
                    continue

                feedback_payload = {
                    "agent_name": self.AGENT_NAME, "analysis_type": "email_performance_deep_dive",
                    "period_days": 7, "raw_performance_logs_sample": performance_data[:100],
                    "metrics_summary": await self._calculate_performance_metrics(performance_data),
                    "request_for_thinktool": (
                        "Analyze this email performance data. Identify: "
                        "1. Patterns in successful (opened, responded) vs. unsuccessful emails (subject lines, content snippets - if available via KB, timing, recipient industry). "
                        "2. Effectiveness of different sender accounts/services (if varied). "
                        "3. Correlations between client engagement scores/industry and email success. "
                        "4. Suggest A/B test ideas for subject lines or CTAs. "
                        "5. Recommend adjustments to optimal send time logic based on observed open/response patterns. "
                        "6. Flag any concerning trends in deliverability (high bounce/failure rates)."
                    )}
                await self.think_tool.execute_task({"action": "process_feedback", "feedback_data": {self.AGENT_NAME: feedback_payload}})
                self.logger.info(f"Sent detailed email performance data ({len(performance_data)} logs) to ThinkTool for strategic analysis.")
                self.internal_state['last_learning_run_ts'] = time.time()

            except asyncio.CancelledError: self.logger.info(f"{self.AGENT_NAME} learning loop cancelled."); break
            except Exception as e:
                self.logger.error(f"Error in {self.AGENT_NAME} learning loop: {e}", exc_info=True)
                await self._report_error(f"Learning loop error: {e}")
                await asyncio.sleep(60 * 30)

    async def _calculate_performance_metrics(self, performance_data: List[Dict]) -> Dict:
        total_attempts = len(performance_data)
        sent_count = sum(1 for d in performance_data if d.get('status') == 'sent')
        opened_count = sum(1 for d in performance_data if d.get('opened_at') is not None)
        responded_count = sum(1 for d in performance_data if d.get('responded_at') is not None)
        failed_verification = sum(1 for d in performance_data if d.get('status') == 'failed_verification')
        failed_send = sum(1 for d in performance_data if d.get('status') == 'failed_send')
        bounced = sum(1 for d in performance_data if d.get('status') == 'bounced')
        valid_attempts_for_delivery_rates = sent_count + failed_send + bounced
        return {
            "total_processed_logs": total_attempts, "sent_successfully_count": sent_count,
            "opened_count": opened_count, "responded_count": responded_count,
            "failed_verification_count": failed_verification, "failed_send_count": failed_send, "bounced_count": bounced,
            "open_rate_on_sent_pct": round((opened_count / sent_count * 100) if sent_count > 0 else 0, 2),
            "response_rate_on_sent_pct": round((responded_count / sent_count * 100) if sent_count > 0 else 0, 2),
            "delivery_success_rate_pct": round((sent_count / valid_attempts_for_delivery_rates * 100) if valid_attempts_for_delivery_rates > 0 else 0, 2),
        }

    async def self_critique(self) -> Dict[str, Any]:
        self.logger.info(f"{self.AGENT_NAME} (L30+): Performing strategic self-critique.")
        critique = {"status": "ok", "feedback": "Critique pending comprehensive analysis."}
        recent_performance_metrics = {}
        if self.session_maker:
            async with self.session_maker() as session:
                one_day_ago = datetime.now(timezone.utc) - timedelta(days=1)
                stmt = select(EmailLog.status, func.count(EmailLog.id).label('count')).where(EmailLog.timestamp >= one_day_ago).group_by(EmailLog.status)
                results = await session.execute(stmt)
                status_counts = {row.status: row.count for row in results.mappings().all()}
                perf_data_for_calc = []
                for s, c in status_counts.items():
                    for _ in range(c):
                        item = {"status": s, "opened_at": None, "responded_at": None}
                        if s == "opened": item["opened_at"] = datetime.now(timezone.utc)
                        if s == "responded": item["responded_at"] = datetime.now(timezone.utc); item["opened_at"] = datetime.now(timezone.utc)
                        perf_data_for_calc.append(item)
                recent_performance_metrics = await self._calculate_performance_metrics(perf_data_for_calc)
        
        active_directives_summary = "N/A"
        if self.think_tool and hasattr(self.think_tool, 'get_active_directives'):
            directives = await self.think_tool.get_active_directives(target_agent=self.AGENT_NAME, limit=3)
            active_directives_summary = [{"type": d.directive_type, "content_preview": d.content[:70], "priority": d.priority} for d in directives]

        critique_context = {
            "task": "Strategic Self-Critique of EmailAgent Performance",
            "current_operational_metrics_24h": recent_performance_metrics,
            "active_directives_for_emailagent": active_directives_summary,
            "meta_prompt_core_mandate": "Maximize profitable conversions ($10k+/day goal) through hyper-personalized, compliant, and adaptive email outreach.",
            "key_responsibilities_to_assess": ["Hyper-Personalization Effectiveness", "Deliverability & Anti-Spam Success", "Engagement Tracking & Opt-Out Processing Efficiency", "Learning Loop Data Quality for ThinkTool"],
            "desired_output_format": "JSON: { \"overall_effectiveness_rating\": str ('Excellent'|'Good'|'Needs Improvement'|'Poor'), \"alignment_with_mandate_assessment\": str, \"key_strengths_observed\": [str], \"critical_weaknesses_identified\": [str], \"data_quality_for_thinktool_rating\": str ('High'|'Medium'|'Low'), \"top_3_actionable_recommendations_for_thinktool_or_self\": [\"Specific, actionable item (e.g., 'ThinkTool: Request A/B test for subject lines targeting [Industry X]', 'Self: Improve parsing of complex STOP requests in IMAP', 'ThinkTool: Investigate high bounce rate with Sender Y')\"] }"
        }
        prompt = await self.generate_dynamic_prompt(critique_context)
        llm_model_pref = self.config.get("OPENROUTER_MODELS", {}).get('think_critique')
        critique_json = await self._call_llm_with_retry(prompt, model=llm_model_pref, temperature=0.3, max_tokens=1500, is_json_output=True)
        if critique_json:
             try:
                 critique_result = self._parse_llm_json(critique_json)
                 if not critique_result: raise ValueError("Parsed critique is None")
                 critique['feedback'] = critique_result.get('overall_effectiveness_rating', 'Critique generated.')
                 critique['details'] = critique_result
                 if self.think_tool:
                     await self.think_tool.execute_task({"action": "log_knowledge_fragment", "fragment_data":{"agent_source": self.AGENT_NAME, "data_type": "self_critique_summary_L30", "content": critique_result, "tags": ["critique", "email_agent", "L30"], "relevance_score": 0.9 }})
                     if critique_result.get("top_3_actionable_recommendations_for_thinktool_or_self"):
                         for rec in critique_result["top_3_actionable_recommendations_for_thinktool_or_self"]:
                             if "ThinkTool:" in rec and hasattr(self.think_tool, 'execute_task'):
                                 await self.think_tool.execute_task({"action": "create_directive_from_suggestion", "content": {"source_agent": self.AGENT_NAME, "suggestion": rec.replace("ThinkTool:","").strip(), "priority": 5}})
             except Exception as e_parse: self.logger.error(f"Failed to parse L30+ self-critique LLM response: {e_parse}"); critique['feedback'] += " Failed to parse LLM critique."; critique['status'] = 'error'
        else: critique['feedback'] += " LLM critique call failed."; critique['status'] = 'error'
        return critique

    async def collect_insights(self) -> Dict[str, Any]:
        insights = {"agent_name": self.AGENT_NAME, "status": self.status, "timestamp": datetime.now(timezone.utc).isoformat(), "active_sends": self.internal_state.get('active_sends',0), "global_sent_today": self.internal_state.get('global_sent_today',0), "global_daily_limit": self.internal_state.get('global_daily_limit',0), "key_observations": []}
        try:
            async with self.session_maker() as session:
                one_day_ago = datetime.now(timezone.utc) - timedelta(days=1)
                stmt = select(EmailLog.status, func.count(EmailLog.id).label('count')).where(EmailLog.timestamp >= one_day_ago).group_by(EmailLog.status)
                results = await session.execute(stmt); insights["performance_24h"] = {row.status: row.count for row in results.mappings().all()}
            insights["key_observations"].append("Included 24h performance summary.")
        except Exception as e: insights["key_observations"].append(f"Failed to retrieve 24h performance: {e}")
        return insights

    async def stop(self, timeout: float = 30.0):
        self.logger.info(f"{self.AGENT_NAME} received stop signal.")
        self._stop_event.set()
        all_tasks_to_cancel = list(self._background_tasks)
        # Ensure specific tasks are included if they are managed as separate attributes
        if hasattr(self, '_task_queue_processor_task') and self._task_queue_processor_task and self._task_queue_processor_task not in all_tasks_to_cancel: # Corrected attribute name
            all_tasks_to_cancel.append(self._task_queue_processor_task)
        if hasattr(self, '_imap_checker_task') and self._imap_checker_task and self._imap_checker_task not in all_tasks_to_cancel: # Corrected attribute name
            all_tasks_to_cancel.append(self._imap_checker_task)
        if hasattr(self, '_learning_loop_task') and self._learning_loop_task and self._learning_loop_task not in all_tasks_to_cancel: # Corrected attribute name
            all_tasks_to_cancel.append(self._learning_loop_task)

        unique_tasks = []
        for task in all_tasks_to_cancel:
            if task and task not in unique_tasks: unique_tasks.append(task)
        for task in unique_tasks:
            if task and not task.done(): task.cancel()
        if unique_tasks:
            self.logger.info(f"Waiting for {len(unique_tasks)} EmailAgent background tasks to cancel...")
            await asyncio.gather(*unique_tasks, return_exceptions=True)
            self.logger.info("EmailAgent background tasks cancellation complete.")
        await super().stop(timeout) # Call base class stop for any generic cleanup

    def _parse_llm_json(self, json_string: str, expect_type: Type = dict) -> Union[Dict, List, None]:
        if not json_string: return None
        try:
            match = None; start_char, end_char = '{', '}'
            if expect_type == list: start_char, end_char = '[', ']'
            match = re.search(rf'(?:```(?:json)?\s*)?(\{start_char}.*\{end_char})\s*(?:```)?', json_string, re.DOTALL)
            parsed_json = None
            if match:
                potential_json = match.group(1)
                try: parsed_json = json.loads(potential_json)
                except json.JSONDecodeError:
                    cleaned_json = re.sub(r',\s*([\}\]])', r'\1', potential_json)
                    cleaned_json = re.sub(r'^\s*|\s*$', '', cleaned_json)
                    try: parsed_json = json.loads(cleaned_json)
                    except json.JSONDecodeError: self.logger.error(f"JSON cleaning failed, unable to parse: {potential_json[:200]}..."); return None
            elif json_string.strip().startswith(start_char) and json_string.strip().endswith(end_char):
                 try: parsed_json = json.loads(json_string)
                 except json.JSONDecodeError as e: self.logger.error(f"Direct JSON parsing failed ({e}): {json_string[:200]}..."); return None
            else: self.logger.warning(f"Could not find expected JSON structure ({expect_type}) in LLM output: {json_string[:200]}..."); return None

            if isinstance(parsed_json, expect_type): return parsed_json
            else: self.logger.error(f"Parsed JSON type mismatch. Expected {expect_type}, got {type(parsed_json)}"); return None
        except Exception as e: self.logger.error(f"Unexpected error during JSON parsing in EmailAgent: {e}", exc_info=True); return None

    async def _learning_loop_wrapper(self): # Added wrapper from base agent pattern
        """Wraps the learning loop to handle errors and stop signals, as in base."""
        self.logger.info(f"{self.AGENT_NAME} learning loop wrapper started.")
        try:
            while not self._stop_event.is_set():
                if self._status == self.STATUS_ERROR:
                    self.logger.warning(f"{self.AGENT_NAME} in error state, pausing learning loop.")
                    await asyncio.sleep(60)
                    continue
                original_status = self._status
                self._status = self.STATUS_LEARNING
                await self.learning_loop()
                if self._status == self.STATUS_LEARNING: self._status = original_status
        except asyncio.CancelledError: self.logger.info(f"{self.AGENT_NAME} learning loop cancelled.")
        except Exception as e:
            self.logger.error(f"Error in {self.AGENT_NAME} learning loop: {e}", exc_info=True)
            self._status = self.STATUS_ERROR
            await self._report_error(f"Error in learning loop: {e}")
        finally:
            self.logger.info(f"{self.AGENT_NAME} learning loop wrapper finished.")
            if self._status == self.STATUS_LEARNING: self._status = self.STATUS_IDLE

    async def _process_task_queue(self, task_queue: asyncio.Queue): # Added from base agent pattern
        """Continuously processes tasks from the internal queue."""
        self.logger.info(f"{self.AGENT_NAME} internal task queue processor started.")
        while not self._stop_event.is_set():
            try:
                task_data = await asyncio.wait_for(task_queue.get(), timeout=5.0)
                if self._stop_event.is_set():
                    await task_queue.put(task_data); break
                task_id = task_data.get('id', 'N/A')
                self.logger.info(f"Dequeued internal task {task_id}: {task_data.get('action', 'Unknown Action')}")
                asyncio.create_task(self.execute_task(task_data), name=f"{self.AGENT_NAME}_Task_{task_id}")
            except asyncio.TimeoutError: continue
            except asyncio.CancelledError: self.logger.info(f"{self.AGENT_NAME} task queue processor cancelled."); break
            except Exception as e:
                 self.logger.error(f"Error in {self.AGENT_NAME} task queue processor: {e}", exc_info=True)
                 self._status = self.STATUS_ERROR
                 await self._report_error(f"Error in task queue processor: {e}")
                 await asyncio.sleep(5)
        self.logger.info(f"{self.AGENT_NAME} internal task queue processor stopped.")

# --- End of agents/email_agent.py ---