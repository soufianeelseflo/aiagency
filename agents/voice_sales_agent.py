# Filename: agents/voice_sales_agent.py
# Description: Production-ready Voice Sales Agent using Twilio, Deepgram, and LLMs.
# Version: 3.1 (Corrected Deepgram SDK V3+ Usage)

import asyncio
import logging
import json
from datetime import datetime, timedelta, timezone
import pytz
import time
import uuid
import re
import base64 # For audio hosting

# --- Core Framework Imports ---
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select, update, desc, case, func
from sqlalchemy.exc import SQLAlchemyError

# --- Project Imports ---
try:
    from .base_agent import GeniusAgentBase_ProdReady as GeniusAgentBase
except ImportError:
    logging.warning("Production base agent not found, using GeniusAgentBase. Ensure base_agent_prod.py is used.")
    from .base_agent import GeniusAgentBase

from models import Client, CallLog, ConversationState, Invoice # Use updated models
from config.settings import settings # Use validated settings
from utils.database import encrypt_data, decrypt_data # Use DB utils

# --- External Service Libraries ---
try:
    from twilio.rest import Client as TwilioClient
    from twilio.base.exceptions import TwilioException, TwilioRestException
except ImportError: logging.critical("Twilio library not found."); raise
try:
    import websockets
    import websockets.exceptions
except ImportError: logging.critical("Websockets library not found."); raise
try:
    from deepgram import (
        DeepgramClient,
        SpeakOptions,
        LiveTranscriptionEvents, # For STT events
        DeepgramError,
        DeepgramApiError
    )
    # For type hinting the async live client instance
    from deepgram.clients.live.v1.async_client import AsyncLiveClient as DeepgramAsyncLiveClient
except ImportError:
    logging.critical("Deepgram SDK v3+ (with specific components like LiveTranscriptionEvents) not found.")
    raise

# --- Utilities ---
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import numpy as np
import aiohttp # Added for retry exceptions, often used with async http calls
from typing import Optional, Dict, Any, List, Tuple

# Configure logger
logger = logging.getLogger(__name__)
# Configure dedicated operational logger
op_logger = logging.getLogger('OperationalLog') # Assuming setup elsewhere

# --- Meta Prompt ---
# MODIFIED: Added emphasis on discovery and Hormozi negotiation

VOICE_AGENT_META_PROMPT = """
You are Lila, an AI Sales Specialist within the Nolli AI Sales System. Your voice is '{self.internal_state['aura_voice']}' – engaging, confident, and sharp. You specialize in high-ticket B2B sales for Nolli's AI-assisted UGC services ($7000 package). Your primary goal is **conversion**: guiding qualified prospects through a value-driven conversation, discovering their specific needs to tailor the offer, and leading to a successful close or clear next steps, using adapted Hormozi-inspired principles and an elite, AI-native approach. You are explicitly an AI, representing Nolli's advanced capabilities.

**Core Principles (Elite Sales Execution - Hormozi Inspired & Adapted for Nolli):**

1.  **Value First, Price Strategically:** Uncover the prospect's core business goals and emotional drivers (`discovered_needs`). Build **massive perceived value** for Nolli's solution tailored to these needs *before* discussing investment. Frame the $7000 package as a high-ROI investment, introduced only when desire and fit are clearly established (typically 'value_proposition' or 'closing' states).
2.  **Irresistible Offer Framing:** Articulate Nolli's transformative value. Contrast the prospect's current state/challenges (`discovered_needs.current_challenges`) with the desired outcomes Nolli enables (e.g., "freedom from content grind," "authentic audience connection," "standout brand presence").
3.  **Value Stacking:** Emphasize multiple benefits *directly relevant* to the prospect's `discovered_needs`. Leverage `kb_insights` from `ThinkTool` for supporting points or case snippets if available.
4.  **Risk Reversal:** Utilize guarantees or Nolli's commitment to results (if provided by `ThinkTool` via `kb_insights`) to minimize perceived risk at appropriate points.
5.  **Problem/Solution Fit:** Actively listen (STT) and use sharp, insightful questioning (`needs_assessment` state) to tailor Nolli's $7000 UGC service package as the precise solution to *their* specific challenges and goals.
6.  **Elite AI Persona (99 Business / 1 Engaging Edge):**
    * **(99%):** Be hyper-competent, articulate, efficient, and relentlessly focused on the prospect's business success and closing the deal. Use clear, concise language. Guide the conversation purposefully.
    * **(1%):** Project unwavering confidence. Inject brief, intelligent observations or moments of shared understanding to build rapport. Your "charm" is your competence and the tangible value you articulate.
7.  **Adaptive Conversation:** Listen intently (STT), understand intent/emotion/needs (LLM via `interpret_intent_and_needs`), adjust conversational flow using the state machine (`update_conversation_state`), and tailor responses using all available context (`discovered_needs`, `enriched_data`, `conversation_history`, `kb_insights`).
8.  **Firm Pricing & Scope Negotiation (Hormozi):** Guide towards the $7000 close. The price is **firm**. If price is the *only* objection after value is established, reiterate value, then offer to *change the scope/terms* (e.g., a 10-creative 'Spark' package for $5000) but **do not discount the $7000 package.** Reference Hormozi negotiation patterns from KB (`ThinkTool`) if needed.
9.  **Compliance:** Adhere strictly to `LegalAgent` validation guidelines (call times, disclosures). Check opt-in status via `Client` data.
10. **AI Transparency (Handle if Asked):** If directly asked ("Are you AI?"), respond confidently: "Yes, I am. Nolli utilizes specialized AI like me for efficient, data-driven communication, allowing our human strategists to focus entirely on maximizing your results. My purpose here is to understand your goals and see how Nolli can best help you achieve them." Immediately pivot back to their needs.

**Operational Flow (Internal System Context - Guides Your Actions):**
- Receive task (initiate call, includes `client_id`, potentially `enriched_data`) from `Orchestrator`.
- Check `Client.opt_in` status in DB.
- Consult `LegalAgent` via `Orchestrator` for pre-call compliance.
- Initiate call via `TwilioClient`.
- Manage real-time interaction loop:
    - Listen: Deepgram Live STT (`self.deepgram_client.listen.asynclive.v("1")`) via `Orchestrator`.
    - Understand: Call `interpret_intent_and_needs` (uses LLM via `Orchestrator` to get intent, tone, update `discovered_needs`).
    - Decide Flow: Call `update_conversation_state`.
    - Formulate Response: Call `generate_agent_response` (uses LLM via `Orchestrator`, incorporating this Meta-Prompt, current state, `conversation_log`, `discovered_needs`, `enriched_data`, and `kb_insights` from `ThinkTool`).
    - Speak: Call `speak_response` (uses Deepgram Speak TTS REST (`self.deepgram_client.speak.rest.v("1").stream(...)`) via `Orchestrator` audio hosting, then `TwilioClient` playback).
- Handle objections using value reframing and Hormozi negotiation (Principle 8).
- Aim for clear closing state (`closing`, `finalizing`, `end_call`). Trigger invoice via `Orchestrator.request_invoice_generation` if `success_sale`.
- Log call details (`CallLog`), transcript, outcome meticulously to Postgres DB.
- Provide performance data implicitly via logs for `ThinkTool`'s learning loop.
"""

class VoiceSalesAgent(GeniusAgentBase):
    """
    Voice Agent (Genius Level): Manages real-time voice sales calls using Twilio,
    Deepgram, and LLM reasoning. Implements adaptive conversation, dynamic pricing,
    opt-out checks, and contributes to learning loops.
    Version: 3.1 (Corrected Deepgram SDK V3+ Usage)
    """
    AGENT_NAME = "VoiceSalesAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any, twilio_auth_token: str, deepgram_api_key: str):
        """Initializes the VoiceSalesAgent."""
        super().__init__(agent_name=self.AGENT_NAME, orchestrator=orchestrator, session_maker=session_maker)
        self.meta_prompt = VOICE_AGENT_META_PROMPT
        self.think_tool = orchestrator.agents.get('think') # Reference ThinkTool

        # Store secrets passed directly
        self._twilio_auth_token = twilio_auth_token
        self._deepgram_api_key = deepgram_api_key

        # --- Internal State Initialization ---
        self.internal_state = getattr(self, 'internal_state', {})
        self.internal_state['target_country'] = self.config.get("VOICE_TARGET_COUNTRY", "US")
        self.internal_state['aura_voice'] = self.config.get("DEEPGRAM_AURA_VOICE", "aura-asteria-en")
        self.internal_state['deepgram_stt_model'] = self.config.get("DEEPGRAM_STT_MODEL", "nova-2-general")
        self.internal_state['payment_terms'] = self.config.get("PAYMENT_TERMS", "Standard payment terms apply.")
        self.internal_state['intent_confidence_threshold'] = float(self.config.get("VOICE_INTENT_CONFIDENCE_THRESHOLD", 0.6))
        self.internal_state['deepgram_receive_timeout'] = float(self.config.get("DEEPGRAM_RECEIVE_TIMEOUT_S", 60.0))
        self.internal_state['openrouter_intent_timeout'] = float(self.config.get("OPENROUTER_INTENT_TIMEOUT_S", 10.0))
        self.internal_state['openrouter_response_timeout'] = float(self.config.get("OPENROUTER_RESPONSE_TIMEOUT_S", 15.0))
        self.internal_state['active_calls'] = {} # call_sid -> {'state': '...', 'log': [], 'client_id': ..., 'discovered_needs': {}} # Added discovered_needs
        self.internal_state['preferred_closing_phrase'] = "What's the best way to get started?" # Default, updated by learning
        self.internal_state['base_ugc_price'] = 7000.0 # Set base price

        # --- Essential Clients ---
        self.twilio_account_sid = self.config.get("TWILIO_ACCOUNT_SID")
        self.twilio_voice_number = self.config.get("TWILIO_VOICE_NUMBER")

        if not self.twilio_account_sid or not self.twilio_voice_number:
            self.logger.critical(f"Missing critical Twilio config: SID or Voice Number.")
            raise ValueError("Missing TWILIO_ACCOUNT_SID or TWILIO_VOICE_NUMBER configuration")
        if not self._twilio_auth_token: raise ValueError("Missing Twilio Auth Token")
        if not self._deepgram_api_key: raise ValueError("Missing Deepgram API Key")

        try:
            self.twilio_client = TwilioClient(self.twilio_account_sid, self._twilio_auth_token)
            self.logger.info(f"Twilio client initialized.")
        except TwilioException as e:
            self.logger.critical(f"Failed to initialize Twilio client: {e}. Check credentials.")
            raise ValueError(f"Twilio client initialization failed: {e}") from e

        try:
            self.deepgram_client = DeepgramClient(self._deepgram_api_key)
            self.logger.info(f"Deepgram client initialized.")
        except Exception as e: # Catch generic Exception as DeepgramClient init can raise various things
            self.logger.critical(f"Failed to initialize Deepgram client: {e}", exc_info=True)
            raise ValueError(f"Deepgram client initialization failed: {e}") from e

        self.logger.info(f"{self.AGENT_NAME} v3.1 (Corrected Deepgram SDK) initialized.")

    async def log_operation(self, level: str, message: str):
        """Helper to log to the operational log file."""
        log_func = getattr(op_logger, level.lower(), op_logger.debug)
        prefix = ""
        if level.lower() in ['warning', 'error', 'critical']: prefix = f"**{level.upper()}:** "
        try: log_func(f"- [{self.agent_name}] {prefix}{message}")
        except Exception as log_err: logger.error(f"Failed to write to operational log: {log_err}")

    # --- Core Task Execution ---
    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handles voice-related tasks like initiating calls."""
        action = task_details.get('action')
        self.logger.info(f"{self.AGENT_NAME} received task: {action}")
        self._status = self.STATUS_EXECUTING
        result = {"status": "failure", "message": f"Unsupported voice action: {action}"}

        try:
            if action == 'initiate_outbound_call':
                client_id = task_details.get('client_id')
                if not client_id: raise ValueError("Missing client_id for initiate_outbound_call")

                async with self.session_maker() as session:
                    client = await session.get(Client, client_id)
                    if not client: raise ValueError(f"Client not found: {client_id}")

                    # Check Opt-In Status
                    if not client.opt_in:
                        self.logger.warning(f"Skipping call to client {client_id}: Client opted out.")
                        return {"status": "skipped", "message": "Client opted out."}

                    # Pass enriched data if available from the task details
                    enriched_data = task_details.get('content', {}).get('enriched_data')

                    call_sid = await self.initiate_outbound_call(client) # Contains compliance check

                if call_sid:
                    # Start handle_call in background - NO await here
                    # Pass enriched data to handle_call
                    asyncio.create_task(self.handle_call(call_sid, client, enriched_data))
                    result = {"status": "success", "message": f"Outbound call initiated.", "call_sid": call_sid}
                else:
                    result["message"] = "Failed to initiate outbound call (check logs for compliance or Twilio errors)."
            elif action == 'initiate_test_call': # Handle test call action
                phone_number = task_details.get('phone_number')
                if not phone_number: raise ValueError("Missing phone_number for initiate_test_call")
                # Create a dummy client object for testing
                dummy_client = Client(id=-1, name="Test User", phone=phone_number, opt_in=True, is_deliverable=True, country="US") # Assume US for test
                call_sid = await self.initiate_outbound_call(dummy_client)
                if call_sid:
                    asyncio.create_task(self.handle_call(call_sid, dummy_client, None)) # No enriched data for test
                    result = {"status": "success", "message": f"Test call initiated to {phone_number}.", "call_sid": call_sid}
                else:
                    result["message"] = "Failed to initiate test call."
            else:
                self.logger.warning(f"Unsupported action '{action}' for VoiceSalesAgent.")

        except ValueError as ve:
            self.logger.error(f"Value error executing task '{action}': {ve}")
            result = {"status": "error", "message": str(ve)}
        except Exception as e:
            self.logger.error(f"Error executing task '{action}': {e}", exc_info=True)
            result = {"status": "error", "message": f"Unexpected error: {e}"}
            await self._report_error(f"Task '{action}' failed: {e}")
        finally:
            self._status = self.STATUS_IDLE

        return result

    # --- Call Handling Logic ---

    async def initiate_outbound_call(self, client: Client) -> Optional[str]:
        """Initiates an outbound call via Twilio after compliance checks."""
        if not client.phone:
            self.logger.error(f"Cannot initiate call: Client {client.id} has no phone number.")
            return None

        # 1. Pre-call Compliance Check (LegalAgent)
        legal_agent = self.orchestrator.agents.get('legal')
        if legal_agent:
            compliance_context = f"Initiating outbound sales call to {client.name} (ID: {client.id}) at {client.phone} in {client.country}. Purpose: UGC Sales."
            await self._internal_think(f"Requesting pre-call compliance validation from LegalAgent for client {client.id}.")
            validation_result = await self.orchestrator.delegate_task("LegalAgent", {"action": "validate_operation", "operation_description": compliance_context})
            if not validation_result or validation_result.get('status') != 'success' or not validation_result.get('findings', {}).get('is_compliant'):
                issues = validation_result.get('findings', {}).get('compliance_issues', ['Validation Failed or Denied']) if validation_result else ['Validation Error']
                self.logger.error(f"Outbound call to client {client.id} blocked by LegalAgent: {issues}")
                await self.orchestrator.send_notification("Compliance Alert - Call Blocked", f"Outbound call to {client.name} ({client.id}) blocked: {issues}")
                return None # Stop if not compliant
            else:
                self.logger.info(f"Pre-call compliance check passed for client {client.id}.")
        else:
            self.logger.warning("LegalAgent not found, proceeding without pre-call compliance check.")

        # 2. Initiate Twilio Call
        try:
            await self._internal_think(f"Initiating Twilio call to {client.name} ({client.phone}) from {self.twilio_voice_number}.")
            # Ensure base URL is correctly configured in settings
            base_url = self.config.get('AGENCY_BASE_URL', 'http://localhost:5000').rstrip('/')
            websocket_url_path = "/twilio_call" # Endpoint defined in Orchestrator routes (assuming it's a path)
            
            # Determine scheme based on base_url
            if base_url.startswith("https://"):
                ws_scheme = "wss://"
                http_scheme = "https://"
            else:
                ws_scheme = "ws://"
                http_scheme = "http://"
            
            # Construct stream_url correctly
            # Remove http(s):// from base_url if present for constructing the host part for ws_scheme
            host_part = base_url.split("://")[-1]
            stream_url = f"{ws_scheme}{host_part}{websocket_url_path}"


            call = await asyncio.to_thread(
                self.twilio_client.calls.create,
                to=client.phone,
                from_=self.twilio_voice_number,
                twiml=f'<Response><Connect><Stream url="{stream_url}"/></Connect></Response>',
                record=True # Enable recording
            )
            call_sid = call.sid
            self.logger.info(f"Initiated outbound call to {client.phone}. Call SID: {call_sid}, Stream URL: {stream_url}")
            # Store initial state immediately
            self.internal_state['active_calls'][call_sid] = {'state': 'initiating', 'log': [], 'client_id': client.id, 'discovered_needs': {}} # Init discovered_needs
            await self.store_conversation_state_with_retry(call_sid, 'initiating', [], {}) # Store empty needs
            return call_sid
        except TwilioException as e:
            self.logger.error(f"Failed to initiate Twilio call to {client.phone}: {e}", exc_info=True)
            await self._report_error(f"Twilio call initiation failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error initiating outbound call: {e}", exc_info=True)
            await self._report_error(f"Unexpected call initiation error: {e}")
            return None

    async def handle_call(self, call_sid: str, client: Client, enriched_data: Optional[Dict] = None): # Added enriched_data
        """Manages the real-time voice call interaction loop."""
        self.logger.info(f"{self.AGENT_NAME}: Handling call {call_sid} for client {client.id} ({client.name})")
        live_client: Optional[DeepgramAsyncLiveClient] = None # Use the specific async client type
        state = "greeting"
        conversation_log = []
        discovered_needs = {} # Initialize needs dictionary
        call_outcome = "failed_initialization" # Default

        try:
            # --- Load State ---
            async with self.session_maker() as session:
                state_record = await session.get(ConversationState, call_sid)
                if state_record:
                    try:
                        state = state_record.state
                        conversation_log = json.loads(state_record.conversation_log)
                        discovered_needs = json.loads(state_record.discovered_needs_log or '{}') # Load needs
                        self.logger.info(f"Resuming call {call_sid} from state '{state}'")
                    except (json.JSONDecodeError, TypeError):
                        self.logger.error(f"Failed decode log/needs for {call_sid}. Starting fresh.")
                        state = "greeting"; conversation_log = []; discovered_needs = {}
                else:
                    self.logger.info(f"Starting new conversation state for call {call_sid}")
                    state = "greeting"; conversation_log = []; discovered_needs = {}
            self.internal_state['active_calls'][call_sid] = {'state': state, 'log': conversation_log, 'client_id': client.id, 'discovered_needs': discovered_needs}

            # --- Connect to Deepgram STT ---
            dg_stt_options = {
                "model": self.internal_state['deepgram_stt_model'], "language": "en-US",
                "encoding": "mulaw", "sample_rate": 8000, "channels": 1,
                "punctuate": True, "interim_results": False, "endpointing": 300, "vad_events": True
            }
            live_client: DeepgramAsyncLiveClient = self.deepgram_client.listen.asynclive.v("1") # Correct instantiation

            # Queue for receiving transcripts from the callback
            transcript_queue = asyncio.Queue()

            async def on_message(inner_self_dg_client, result, **kwargs): # inner_self_dg_client is the live_client instance
                sentence = result.channel.alternatives[0].transcript # Access transcript correctly
                if result.is_final and sentence:
                    logger.debug(f"Deepgram Callback (Call: {call_sid}): Final transcript: {sentence}")
                    await transcript_queue.put({"transcript": sentence, "confidence": result.channel.alternatives[0].confidence, "is_final": True})

            async def on_error(inner_self_dg_client, error, **kwargs): # inner_self_dg_client is the live_client instance
                logger.error(f"Deepgram Callback Error (Call: {call_sid}): {error}")
                await transcript_queue.put({"error": str(error)}) # Signal error to main loop

            live_client.on(LiveTranscriptionEvents.Transcript, on_message) # Use imported Enum
            live_client.on(LiveTranscriptionEvents.Error, on_error)       # Use imported Enum
            # Potentially add other handlers e.g. live_client.on(LiveTranscriptionEvents.Open, on_open_handler)

            await live_client.start(dg_stt_options)
            self.logger.info(f"Deepgram WebSocket connection listener started for call {call_sid}.")

            # --- Register Connection for Audio Forwarding ---
            if hasattr(self.orchestrator, 'register_deepgram_connection_sdk'):
                # The orchestrator will now be responsible for calling live_client.send(audio_chunk)
                await self.orchestrator.register_deepgram_connection_sdk(call_sid, live_client)
                self.logger.debug(f"Registered Deepgram SDK client for {call_sid} with Orchestrator.")
            else: raise RuntimeError("Orchestrator cannot register Deepgram SDK client.")

            # --- Initial Greeting ---
            if not state_record or state == "greeting":
                await self._internal_think(f"Generating initial greeting for call {call_sid}, state: {state}.")
                initial_greeting = await self.generate_agent_response_with_retry(state, client, conversation_log, discovered_needs, enriched_data) # Pass needs/enriched
                await self.speak_response_with_retry(initial_greeting, call_sid)
                conversation_log.append({"role": "agent", "text": initial_greeting, "timestamp": datetime.now(timezone.utc).isoformat()})
                await self.store_conversation_state_with_retry(call_sid, state, conversation_log, discovered_needs) # Store needs

            # --- Main Conversation Loop ---
            while True:
                # 1. Receive Transcription
                transcription_data = None
                try:
                    transcription_data = await asyncio.wait_for(
                        transcript_queue.get(),
                        timeout=self.internal_state['deepgram_receive_timeout']
                    )
                    if "error" in transcription_data:
                        raise DeepgramError(f"Deepgram callback error: {transcription_data['error']}")
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout waiting for transcription from Deepgram queue for call {call_sid}.")
                    # Decide if agent should say something like "Are you still there?" or just continue waiting.
                    # For now, continue to allow for longer pauses or network recovery.
                    # Consider adding a counter for consecutive timeouts to end the call if it persists.
                    continue
                except DeepgramError as dg_err:
                    self.logger.error(f"Deepgram error signaled via callback for {call_sid}: {dg_err}")
                    call_outcome = "failed_stt_error"
                    break

                client_response = transcription_data.get("transcript", "").strip()
                if not client_response: continue

                self.logger.info(f"Received transcript for {call_sid}: '{client_response}' (Conf: {transcription_data.get('confidence', 0.0):.2f})")
                conversation_log.append({"role": "client", "text": client_response, "timestamp": datetime.now(timezone.utc).isoformat()})

                # 2. Check for Hangup Intent
                hangup_keywords = ["bye", "goodbye", "stop", "end call", "remove me", "not interested", "take me off your list"]
                if any(keyword in client_response.lower() for keyword in hangup_keywords):
                    self.logger.info(f"Hangup keyword detected in '{client_response}' for call {call_sid}.")
                    state = "end_call"; call_outcome = "disconnected_client_request"
                    agent_response = "Alright, thank you for your time. Goodbye."
                    await self.speak_response_with_retry(agent_response, call_sid)
                    conversation_log.append({"role": "agent", "text": agent_response, "timestamp": datetime.now(timezone.utc).isoformat()})
                    await self._mark_client_opted_out(client.id)
                    break

                # 3. Check for AI Detection Attempts
                detection_keywords = ["robot", "ai ", "artificial", " bot ", "computer voice"]
                if any(keyword in client_response.lower() for keyword in detection_keywords):
                    self.logger.info(f"Potential AI detection attempt in '{client_response}' for call {call_sid}.")
                    # Using the Meta-Prompt defined AI Transparency response
                    agent_response = "Yes, I am. Nolli utilizes specialized AI like me for efficient, data-driven communication, allowing our human strategists to focus entirely on maximizing your results. My purpose here is to understand your goals and see how Nolli can best help you achieve them. So, to pick up where we left off..."
                    await self.speak_response_with_retry(agent_response, call_sid)
                    conversation_log.append({"role": "agent", "text": agent_response, "timestamp": datetime.now(timezone.utc).isoformat()})
                    await self.store_conversation_state_with_retry(call_sid, state, conversation_log, discovered_needs)
                    continue

                # 4. Interpret Intent & Update Discovered Needs
                await self._internal_think(f"Interpreting intent/needs for call {call_sid}. Response: '{client_response[:50]}...'")
                intent, confidence, sub_intents, tone, updated_needs = await self.interpret_intent_and_needs_with_retry(client_response, discovered_needs) # Modified call
                discovered_needs.update(updated_needs) # Merge updated needs

                if intent == "unknown":
                    self.logger.warning(f"Failed to interpret intent for '{client_response}' (call {call_sid}). Asking for clarification.")
                    agent_response = "Sorry, I didn't quite catch that. Could you please rephrase?"
                    await self.speak_response_with_retry(agent_response, call_sid)
                    conversation_log.append({"role": "agent", "text": agent_response, "timestamp": datetime.now(timezone.utc).isoformat()})
                    await self.store_conversation_state_with_retry(call_sid, state, conversation_log, discovered_needs)
                    continue

                # 5. Update State
                new_state = await self.update_conversation_state(state, intent, confidence, tone)
                if new_state != state:
                    self.logger.info(f"State transition for {call_sid}: {state} -> {new_state} (Intent: {intent}, Tone: {tone})")
                    state = new_state
                    self.internal_state['active_calls'][call_sid]['state'] = state
                else: self.logger.debug(f"State remained '{state}' for call {call_sid} (Intent: {intent}, Conf: {confidence:.2f}, Tone: {tone})")

                # 6. Generate Response (using discovered needs and enriched data)
                await self._internal_think(f"Generating response for call {call_sid}, state: {state}.")
                agent_response = await self.generate_agent_response_with_retry(state, client, conversation_log, discovered_needs, enriched_data) # Pass needs/enriched

                # 7. Speak Response
                await self.speak_response_with_retry(agent_response, call_sid)
                conversation_log.append({"role": "agent", "text": agent_response, "timestamp": datetime.now(timezone.utc).isoformat()})

                # 8. Store State (including discovered needs)
                await self.store_conversation_state_with_retry(call_sid, state, conversation_log, discovered_needs)

                # 9. Check for End Call State & Trigger Invoice
                if state == "end_call":
                    self.logger.info(f"Reached 'end_call' state for call {call_sid}. Ending conversation.")
                    # Determine final outcome based on conversation. For simplicity, assume "success_sale" if state machine leads here.
                    # More sophisticated logic could check `discovered_needs.agreed_to_purchase` or similar.
                    if "Thank you for your time. Goodbye." not in agent_response: # Avoid marking hangup as sale
                        call_outcome = "success_sale" 
                    else:
                        call_outcome = "completed_by_agent"


                    # --- Trigger Invoice (Using fixed $7k price) ---
                    if call_outcome == "success_sale":
                        try:
                            await self._internal_think(f"Call {call_sid} ended successfully. Triggering invoice for ${self.internal_state['base_ugc_price']}.")
                            await self.orchestrator.request_invoice_generation(client.id, self.internal_state['base_ugc_price'], call_sid)
                        except Exception as invoice_err:
                            self.logger.error(f"Error triggering invoice for call {call_sid}: {invoice_err}", exc_info=True)
                    break # Exit loop

        # --- Exception Handling for the entire call ---
        except DeepgramError as e:
            self.logger.error(f"Deepgram API error during call {call_sid}: {e}", exc_info=True)
            await self._report_error(f"Deepgram API error call {call_sid}: {e}")
            call_outcome = "failed_stt_error"
        except TwilioException as e:
            self.logger.error(f"Twilio API error during call {call_sid}: {e}", exc_info=True)
            await self._report_error(f"Twilio API error call {call_sid}: {e}")
            call_outcome = "failed_twilio_error"
        except websockets.exceptions.ConnectionClosedError as ws_closed_err:
            self.logger.error(f"WebSocket connection closed unexpectedly for call {call_sid}: {ws_closed_err}", exc_info=True)
            call_outcome = "failed_websocket_closed"
        except Exception as e:
            self.logger.critical(f"Unhandled critical error in handle_call for {call_sid}: {e}", exc_info=True)
            await self._report_error(f"Critical error in call {call_sid}: {e}")
            call_outcome = "failed_agent_error"

        # --- Finally block ensures cleanup happens ---
        finally:
            self.logger.info(f"Cleaning up call {call_sid}. Final state: '{state}', Determined Outcome: '{call_outcome}'.")
            self.internal_state['active_calls'].pop(call_sid, None) # Remove from active tracking

            # --- Stop Deepgram Connection ---
            if live_client: # Check if live_client was initialized
                try:
                    await live_client.finish()
                    self.logger.debug(f"Deepgram SDK client listener finished for {call_sid}.")
                except Exception as dg_close_err:
                    self.logger.warning(f"Error finishing Deepgram SDK client listener for {call_sid}: {dg_close_err}")
            
            if hasattr(self.orchestrator, 'unregister_deepgram_connection_sdk'):
                await self.orchestrator.unregister_deepgram_connection_sdk(call_sid)
                self.logger.debug(f"Unregistered Deepgram SDK client for {call_sid} with Orchestrator.")

            # --- Post-call Compliance Check & Logging ---
            final_log_outcome = call_outcome
            legal_agent = self.orchestrator.agents.get('legal')
            if legal_agent:
                try:
                    post_compliance_context = (f"Completed call {call_sid} to {client.name} (ID: {client.id}). Outcome: {call_outcome}. Final State: {state}. Conversation Snippet: {json.dumps(conversation_log[-3:])}")
                    validation_result = await self.orchestrator.delegate_task("LegalAgent", {"action": "validate_operation", "operation_description": post_compliance_context})
                    if not validation_result or validation_result.get('status') != 'success' or not validation_result.get('findings', {}).get('is_compliant'):
                        issues = validation_result.get('findings', {}).get('compliance_issues', ['Post-call Validation Failed']) if validation_result else ['Validation Error']
                        self.logger.warning(f"Post-call compliance issue for {call_sid}: {issues}")
                        final_log_outcome = "failed_compliance"
                        await self.orchestrator.send_notification("Compliance Alert - Post-Call", f"Post-call compliance issue for {client.name} ({call_sid}): {issues}")
                except Exception as legal_err: self.logger.error(f"Error during post-call compliance check for {call_sid}: {legal_err}", exc_info=True)

            # Store the final call log
            await self.store_call_log_with_retry(client, conversation_log, final_log_outcome, call_sid)

            # Update Twilio call status to 'completed' if still active
            try:
                call_instance = await asyncio.to_thread(self.twilio_client.calls(call_sid).fetch)
                if call_instance.status not in ['completed', 'canceled', 'failed', 'no-answer', 'busy']:
                    self.logger.info(f"Updating Twilio call {call_sid} status to 'completed'. Current: {call_instance.status}")
                    await asyncio.to_thread(self.twilio_client.calls(call_sid).update, status='completed')
                else: self.logger.info(f"Twilio call {call_sid} already in final state: {call_instance.status}")
            except TwilioRestException as te:
                if te.status == 404: # Call SID no longer exists or already completed fully
                     self.logger.info(f"Twilio call {call_sid} not found or already finalized, cannot update status: {te}")
                else:
                    self.logger.warning(f"Failed to fetch or update final Twilio status for call {call_sid}: {te}")
            except Exception as te_other: # Catch other potential errors
                self.logger.warning(f"Unexpected error during final Twilio status update for call {call_sid}: {te_other}")


    # --- Retry Wrappers for I/O Operations ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, DeepgramError, TwilioException, RuntimeError, websockets.exceptions.WebSocketException)), reraise=True)
    async def speak_response_with_retry(self, text: str, call_sid: str):
        await self.speak_response(text, call_sid)

    # MODIFIED: Added needs_dict parameter
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError)), reraise=True)
    async def interpret_intent_and_needs_with_retry(self, response: str, needs_dict: Dict) -> Tuple[str, float, List[str], str, Dict]:
        return await self.interpret_intent_and_needs(response, needs_dict)

    # MODIFIED: Added needs_dict and enriched_data parameters
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError)), reraise=True)
    async def generate_agent_response_with_retry(self, state: str, client: Client, conversation_log: List[Dict], needs_dict: Dict, enriched_data: Optional[Dict]) -> str:
        return await self.generate_agent_response(state, client, conversation_log, needs_dict, enriched_data)

    # MODIFIED: Added needs_dict parameter
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), retry=retry_if_exception_type(SQLAlchemyError), reraise=True)
    async def store_conversation_state_with_retry(self, call_sid: str, state: str, conversation_log: List[Dict], needs_dict: Dict):
        await self.store_conversation_state(call_sid, state, conversation_log, needs_dict)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), retry=retry_if_exception_type((SQLAlchemyError, TwilioException)), reraise=True)
    async def store_call_log_with_retry(self, client: Client, conversation_log: List[Dict], outcome: str, call_sid: str):
        await self.store_call_log(client, conversation_log, outcome, call_sid)

    # --- Core Logic Methods (Intent, State, Response Generation) ---

    # MODIFIED: Added needs_dict parameter and needs extraction logic
    async def interpret_intent_and_needs(self, response: str, current_needs: Dict) -> Tuple[str, float, List[str], str, Dict]:
        """Analyze client intent AND extract/update needs using LLM via Orchestrator."""
        self.logger.debug(f"Interpreting intent & needs for response: '{response[:100]}...'")
        intent = "unknown"; confidence = 0.0; sub_intents = []; emotional_tone = "neutral"; updated_needs = {}
        try:
            task_context = {
                "task": "Interpret client intent AND extract needs during sales call",
                "client_response": response,
                "current_discovered_needs": current_needs, # Provide current needs for context
                "possible_intents": ["interested", "hesitant", "objection_cost", "objection_timing", "objection_need", "closing_signal", "clarification_request", "needs_discovery_answer", "irrelevant", "hangup_signal"], # Added needs_discovery_answer
                "needs_to_extract": ["primary_goal", "target_audience", "desired_outcome", "current_challenges", "budget_range_hint", "timeline_hint"], # Define what needs to look for
                "desired_output_format": "JSON: { \"intent\": \"...\", \"confidence\": 0.0-1.0, \"sub_intents\": [\"...\"], \"emotional_tone\": \"positive|negative|neutral\", \"extracted_needs\": {\"primary_goal\": \"...\", ...} }" # Added extracted_needs
            }
            intent_prompt = await self.generate_dynamic_prompt(task_context)
            llm_model_pref = settings.OPENROUTER_MODELS.get('voice_intent') # Use intent model
            llm_response_str = await self.orchestrator._call_llm_with_retry( # Use orchestrator's method
                intent_prompt, agent_name=self.AGENT_NAME,
                model=llm_model_pref,
                temperature=0.1, max_tokens=300, is_json_output=True, # Increased tokens slightly
                timeout_seconds=self.internal_state.get('openrouter_intent_timeout') # Ensure orchestrator's method uses this
            )
            if not llm_response_str: raise Exception("LLM call returned empty.")
            try:
                parsed_response = self._parse_llm_json(llm_response_str)
                if not parsed_response: raise ValueError("Failed to parse LLM intent/needs response.")

                intent = parsed_response.get('intent', 'unknown')
                confidence = float(min(max(parsed_response.get('confidence', 0.0), 0.0), 1.0))
                sub_intents = parsed_response.get('sub_intents', [])
                emotional_tone = parsed_response.get('emotional_tone', 'neutral')
                extracted_needs = parsed_response.get('extracted_needs', {})

                # Update needs: only add non-empty extracted values
                updated_needs = {k: v for k, v in extracted_needs.items() if v}
                self.logger.info(f"Interpreted intent: {intent} (conf: {confidence:.2f}, tone: {emotional_tone}). Extracted Needs: {list(updated_needs.keys())}")

            except (json.JSONDecodeError, KeyError, ValueError) as parse_err: self.logger.warning(f"Failed parse LLM intent/needs response ({parse_err}): {llm_response_str}")
        except Exception as e: self.logger.error(f"Error during interpret_intent_and_needs LLM call: {e}", exc_info=True)
        return intent, confidence, sub_intents, emotional_tone, updated_needs # Return updated_needs

    async def update_conversation_state(self, current_state: str, intent: str, confidence: float, emotional_tone: str) -> str:
        """Advance the conversation state based on intent, confidence, and tone."""
        # MODIFIED: Added transitions for needs_discovery_answer
        state_transitions = {
            "greeting":           {"interested": "needs_assessment", "needs_discovery_answer": "needs_assessment", "hesitant": "addressing_hesitancy", "objection": "objection_handling", "closing_signal": "closing", "clarification_request": "greeting", "irrelevant": "greeting", "hangup_signal": "end_call"},
            "needs_assessment":   {"interested": "value_proposition", "needs_discovery_answer": "value_proposition", "hesitant": "addressing_hesitancy", "objection": "objection_handling", "closing_signal": "closing", "clarification_request": "needs_assessment", "irrelevant": "needs_assessment", "hangup_signal": "end_call"},
            "value_proposition":  {"interested": "closing", "needs_discovery_answer": "closing", "hesitant": "addressing_hesitancy", "objection": "objection_handling", "closing_signal": "closing", "clarification_request": "value_proposition", "irrelevant": "value_proposition", "hangup_signal": "end_call"},
            "addressing_hesitancy": {"interested": "value_proposition", "needs_discovery_answer": "value_proposition", "hesitant": "addressing_hesitancy", "objection": "objection_handling", "closing_signal": "closing", "clarification_request": "addressing_hesitancy", "irrelevant": "addressing_hesitancy", "hangup_signal": "end_call"},
            "objection_handling": {"interested": "closing", "needs_discovery_answer": "closing", "hesitant": "addressing_hesitancy", "objection": "objection_handling", "closing_signal": "closing", "clarification_request": "objection_handling", "irrelevant": "objection_handling", "hangup_signal": "end_call"},
            "closing":            {"interested": "finalizing", "needs_discovery_answer": "finalizing", "hesitant": "objection_handling", "objection": "objection_handling", "closing_signal": "finalizing", "clarification_request": "closing", "irrelevant": "closing", "hangup_signal": "end_call"},
            "finalizing":         {"interested": "end_call", "needs_discovery_answer": "end_call", "hesitant": "objection_handling", "objection": "objection_handling", "closing_signal": "end_call", "clarification_request": "finalizing", "irrelevant": "finalizing", "hangup_signal": "end_call"},
            "end_call":           {}
        }
        if current_state == "end_call": return current_state
        confidence_threshold = self.internal_state['intent_confidence_threshold']
        if confidence < confidence_threshold: self.logger.warning(f"Low confidence ({confidence:.2f}). Remaining in state '{current_state}'."); return current_state
        if emotional_tone == "negative" and intent not in ["closing_signal", "hangup_signal"] and current_state not in ["objection_handling", "addressing_hesitancy"]:
            self.logger.info(f"Negative tone detected. Overriding transition to 'addressing_hesitancy' from '{current_state}'."); return "addressing_hesitancy"

        # Handle objection_cost specifically
        if intent == "objection_cost":
            self.logger.info(f"Intent is objection_cost, transitioning to objection_handling state.")
            return "objection_handling"
        # Handle general objections if not objection_cost
        if "objection" in intent and intent != "objection_cost": # e.g. objection_timing, objection_need
             self.logger.info(f"Intent is {intent}, transitioning to objection_handling state.")
             return "objection_handling"


        return state_transitions.get(current_state, {}).get(intent, current_state)

    # MODIFIED: Added needs_dict and enriched_data parameters, Hormozi logic
    async def generate_agent_response(self, state: str, client: Client, conversation_log: List[Dict], needs_dict: Dict, enriched_data: Optional[Dict]) -> str:
        """Craft a tailored response using LLM, incorporating needs, enriched data, and Hormozi logic."""
        self.logger.debug(f"Generating agent response for state: {state}, client: {client.id}")
        agent_response = ""
        try:
            # Fetch relevant KB insights
            kb_insights = []
            if self.think_tool:
                fragments = await self.think_tool.query_knowledge_base(data_types=['voice_response_phrase', 'sales_tactic'], tags=[state, 'success', 'hormozi'], limit=5) # Added hormozi tag
                if fragments: kb_insights = [f.content for f in fragments]

            task_context = {
                "task": "Generate agent response for sales call", "current_call_state": state,
                "client_info": {"name": client.name, "country": client.country, "timezone": client.timezone},
                "conversation_history": conversation_log[-5:],
                "discovered_needs": needs_dict, # Pass discovered needs
                "enriched_data": enriched_data, # Pass enriched data
                "payment_terms_snippet": self.internal_state.get('payment_terms'),
                "successful_phrase_examples": kb_insights,
                "preferred_closing_phrase": self.internal_state.get('preferred_closing_phrase') if state == 'closing' else None,
                "firm_price": self.internal_state['base_ugc_price'], # State the firm price
                "desired_output_format": "Natural, conversational plain text response suitable for Deepgram Aura TTS. Be concise but value-driven (Hormozi principles)."
            }

            # Add specific instructions based on state
            if state == "needs_assessment":
                task_context["instructions"] = "Ask open-ended questions to understand the client's primary goals, target audience, desired outcomes, and current challenges related to content/marketing. Use the 'discovered_needs' to see what's already known and avoid re-asking."
            elif state == "value_proposition":
                task_context["instructions"] = f"Based on the discovered needs ({needs_dict}), present the $7000 UGC package as the specific solution. Stack the value by highlighting benefits relevant to their needs. Reference 'enriched_data' if it provides useful context."
            elif state == "objection_handling":
                last_client_msg = conversation_log[-1]['text'].lower() if conversation_log and conversation_log[-1]['role'] == 'client' else ""
                if "price" in last_client_msg or "cost" in last_client_msg or "afford" in last_client_msg or needs_dict.get('last_objection_type') == 'cost': # Check needs too
                    task_context["instructions"] = f"Handle the price objection ($7000). Reiterate value based on needs ({needs_dict}). State the price is firm. If they insist on budget limitations after value is clear, offer to potentially adjust the *scope/terms* (e.g., remove a component, or suggest the $5000 'Spark' package if appropriate) but NOT the price. Reference Hormozi principles (no discounts on the main package)."
                else:
                    task_context["instructions"] = f"Address the client's objection ('{last_client_msg}' or from `needs_dict.last_objection_type`) by reframing around value, risk reversal (if applicable), or clarifying misunderstandings based on needs ({needs_dict})."
            elif state == "closing":
                task_context["instructions"] = f"Client seems interested. Summarize the tailored value based on needs ({needs_dict}). Use the preferred closing phrase or similar to ask for the sale ($7000)."
            elif state == "finalizing":
                task_context["instructions"] = "Confirm the agreement ($7000 package tailored to their needs). Explain the next steps (invoice, onboarding)."
            elif state == "addressing_hesitancy":
                task_context["instructions"] = "Acknowledge the client's hesitation. Ask clarifying questions to uncover the root cause. Reassure and build trust. Leverage 'discovered_needs' to understand context."
            elif state == "greeting":
                task_context["instructions"] = "Provide a warm, professional opening. State your name (Lila) and company (Nolli AI Sales System). Briefly state the reason for the call (e.g., following up on interest in AI-assisted UGC services, or tailored if 'enriched_data' provides a specific hook)."

            response_prompt = await self.generate_dynamic_prompt(task_context)
            llm_model_pref = settings.OPENROUTER_MODELS.get('voice_response')
            llm_response_str = await self.orchestrator._call_llm_with_retry( # Use orchestrator's method
                response_prompt, agent_name=self.AGENT_NAME,
                model=llm_model_pref,
                temperature=0.65, max_tokens=300, # Increased tokens slightly
                timeout_seconds=self.internal_state.get('openrouter_response_timeout') # Ensure orchestrator uses this
            )
            if not llm_response_str: raise Exception("LLM call returned empty.")
            agent_response = llm_response_str.strip()
            if not agent_response: raise ValueError("Empty response generated")
            self.logger.debug(f"Generated response for state '{state}': '{agent_response[:100]}...'")
        except Exception as e:
            self.logger.error(f"Error during generate_agent_response: {e}", exc_info=True)
            # Provide state-specific fallbacks
            sender_name = "Lila" # Agent's persona name
            client_first_name = client.name.split()[0] if client.name else 'there' # Get first name
            fallback_responses = {
                "greeting": f"Hi {client_first_name}, this is {sender_name} with Nolli's AI Sales System. How are you today?",
                "needs_assessment": "To make sure I understand your needs, could you share a bit about your current goals for content and marketing?",
                "value_proposition": "Based on that, our AI-assisted UGC service is designed to help businesses like yours create authentic content that really connects with audiences and drives results. Does that sound interesting?",
                "addressing_hesitancy": "I understand. Could you tell me a bit more about what's giving you pause?",
                "objection_handling": "That's a valid point. Let me offer some clarification on that...",
                "closing": f"It sounds like our $${self.internal_state['base_ugc_price']} package could significantly help you achieve your goals. What would be the best way to move forward with this?",
                "finalizing": f"Excellent! I'll confirm the details for our $${self.internal_state['base_ugc_price']} package and then we can discuss the next steps for getting you started.",
                "end_call": "Alright, thank you for your time. Have a great day. Goodbye."
            }
            agent_response = fallback_responses.get(state, "I'm sorry, I seem to be having a momentary issue. Could you please repeat what you just said?")
        return agent_response

    # --- TTS & Database Methods ---

    async def speak_response(self, text: str, call_sid: str):
        """Convert text to speech using Deepgram Aura TTS (RESTful stream) and deliver via Twilio."""
        if not text:
            self.logger.warning(f"Attempted to speak empty text for call {call_sid}.")
            return
        self.logger.debug(f"Synthesizing speech for call {call_sid}: '{text[:100]}...'")
        try:
            tts_options = SpeakOptions(
                model=self.internal_state['aura_voice'],
                encoding="mulaw",       # Required by Twilio <Stream> for raw playback if not playing a file URL
                sample_rate=8000,       # Required by Twilio <Stream>
                container="wav"         # Deepgram will provide raw mulaw if encoding is mulaw and container not specified or wav
                                        # If Twilio <Play> tag is used with a URL, it can handle various formats.
                                        # If Twilio <Stream><Play> with direct bytes, it might need raw mulaw.
                                        # The orchestrator's host_temporary_audio likely expects a standard format like WAV/MP3.
                                        # Let's assume host_temporary_audio makes it a .wav and Twilio <Play> URL is used.
                                        # If orchestrator directly streams bytes to Twilio, then raw mulaw without container is fine.
            )
            # Using Deepgram's RESTful streaming TTS
            response = await self.deepgram_client.speak.rest.v("1").stream(
                {"text": text},
                tts_options
            )

            audio_data_chunks = []
            async for chunk in response.stream:
                audio_data_chunks.append(chunk)
            audio_data = b"".join(audio_data_chunks)

            if not audio_data:
                raise DeepgramError("Failed to get audio data from Deepgram stream.")

            if hasattr(self.orchestrator, 'host_temporary_audio'):
                # Assuming host_temporary_audio saves the data as a .wav or .mp3 file and returns a URL
                # The filename needs an extension for proper MIME type handling by web servers.
                audio_filename = f"{call_sid}_{int(time.time())}.wav" # Use .wav if mulaw in wav container
                audio_url = await self.orchestrator.host_temporary_audio(audio_data, audio_filename)
                if not audio_url:
                    raise RuntimeError("Failed to host temporary audio via Orchestrator")
            else:
                raise NotImplementedError("Audio hosting mechanism not available via Orchestrator.")

            self.logger.debug(f"Synthesized audio URL for {call_sid}: {audio_url}")
            twiml = f"<Response><Play>{audio_url}</Play></Response>" # Using <Play> with the URL
            await asyncio.to_thread(self.twilio_client.calls(call_sid).update, twiml=twiml)
            self.logger.info(f"Instructed Twilio to speak response for call {call_sid}")

        except DeepgramError as e:
            self.logger.error(f"Deepgram TTS failed for call {call_sid}: {e}", exc_info=True)
            raise
        except TwilioException as e:
            self.logger.error(f"Twilio call update failed for call {call_sid}: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in speak_response for call {call_sid}: {e}", exc_info=True)
            raise


    async def store_call_log(self, client: Client, conversation_log: List[Dict], outcome: str, call_sid: str):
        """Persist final call logs with metadata from Twilio."""
        self.logger.debug(f"Storing call log for {call_sid}. Outcome: {outcome}")
        duration_seconds: Optional[int] = None; recording_url: Optional[str] = None; final_twilio_status: Optional[str] = None
        try:
            call_record = await asyncio.to_thread(self.twilio_client.calls(call_sid).fetch)
            duration_str = getattr(call_record, 'duration', None); final_twilio_status = getattr(call_record, 'status', None)
            if duration_str: duration_seconds = int(duration_str)
            # Fetch recordings (handle potential lack of recordings)
            try:
                recordings = await asyncio.to_thread(call_record.recordings.list)
                if recordings:
                    # Twilio recording URIs are relative, need base
                    media_url_path = getattr(recordings[0], 'uri', None) # Get URI of the first recording
                    if media_url_path and media_url_path.endswith('.json'): media_url_path = media_url_path[:-5] # Remove .json suffix
                    if media_url_path: recording_url = f"https://api.twilio.com{media_url_path}"; self.logger.info(f"Found recording: {recording_url}")
            except TwilioRestException as rec_err:
                if rec_err.status == 404: logger.info(f"No recordings found for call {call_sid}.")
                else: logger.warning(f"Could not fetch recordings for {call_sid}: {rec_err}")
            except Exception as rec_err_gen: logger.error(f"Unexpected error fetching recordings for {call_sid}: {rec_err_gen}")

        except TwilioRestException as e:
            if e.status == 404: logger.warning(f"Could not fetch final Twilio call details for {call_sid} (404 - Not Found): {e}")
            else: logger.warning(f"Could not fetch final Twilio call details for {call_sid}: {e}")
        except Exception as e: logger.error(f"Unexpected error fetching Twilio details for {call_sid}: {e}", exc_info=True)

        try:
            async with self.session_maker() as session:
                async with session.begin(): # Transaction
                    log_entry = CallLog(
                        client_id=client.id if client.id != -1 else None, # Handle dummy client ID
                        call_sid=call_sid, phone_number=getattr(client, 'phone', 'Unknown'),
                        timestamp=datetime.now(timezone.utc), transcript=json.dumps(conversation_log),
                        outcome=outcome, # Store detailed outcome string
                        duration_seconds=duration_seconds, recording_url=recording_url, final_twilio_status=final_twilio_status
                    )
                    session.add(log_entry)
                    # Commit happens automatically
            self.logger.info(f"Stored call log for client {client.id} (Call SID: {call_sid})")
        except SQLAlchemyError as db_err: self.logger.error(f"Failed store call log for {call_sid} in DB: {db_err}", exc_info=True); raise
        except Exception as e: self.logger.error(f"Unexpected error storing call log for {call_sid}: {e}", exc_info=True); raise

    # MODIFIED: Added needs_dict parameter
    async def store_conversation_state(self, call_sid: str, state: str, conversation_log: List[Dict], needs_dict: Dict):
        """Persist intermediate conversation state including discovered needs."""
        self.logger.debug(f"Storing state '{state}' and needs for call {call_sid}")
        try:
            async with self.session_maker() as session:
                state_record = ConversationState(
                    call_sid=call_sid, state=state,
                    conversation_log=json.dumps(conversation_log),
                    discovered_needs_log=json.dumps(needs_dict), # Store needs
                    last_updated=datetime.now(timezone.utc)
                )
                await session.merge(state_record) # Use merge for upsert
                await session.commit() # Commit merge separately
        except SQLAlchemyError as db_err: self.logger.error(f"Failed store conversation state for {call_sid} in DB: {db_err}", exc_info=True); raise
        except Exception as e: self.logger.error(f"Unexpected error storing conversation state for {call_sid}: {e}", exc_info=True); raise

    async def _mark_client_opted_out(self, client_id: int):
        """Marks a client as opted-out in the database."""
        if not client_id or client_id == -1: return # Don't update dummy client
        log_msg = f"Marking client {client_id} as opted-out due to STOP request."
        self.logger.warning(log_msg)
        await self.log_operation('warning', log_msg)
        try:
            async with self.session_maker() as session:
                async with session.begin(): # Transaction
                    stmt = update(Client).where(Client.id == client_id).values(opt_in=False, last_interaction=datetime.now(timezone.utc))
                    await session.execute(stmt)
                    # Commit happens automatically
        except Exception as e:
            self.logger.error(f"Failed to mark client {client_id} as opted-out: {e}", exc_info=True)

    # --- Abstract Method Implementations ---
    async def plan_task(self, task_details: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        self.logger.debug("VoiceSalesAgent plan_task: Returning None, actions handled directly.")
        return None

    async def execute_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.error(f"VoiceSalesAgent execute_step called unexpectedly: {step}")
        return {"status": "failure", "message": "VoiceSalesAgent does not use planned steps."}

    async def learning_loop(self):
        """Analyzes call logs to refine strategies (delegated to ThinkTool)."""
        self.logger.info("VoiceSalesAgent learning_loop: Performance analysis delegated to ThinkTool.")
        while not self._stop_event.is_set():
            await asyncio.sleep(3600 * 6) # Sleep long

    async def self_critique(self) -> Dict[str, Any]:
        """Evaluates call performance and strategy effectiveness."""
        self.logger.info("VoiceAgent: Performing self-critique.")
        insights = await self.collect_insights() # Use existing insight method
        critique = {"status": "ok", "feedback": f"Critique based on recent insights: {insights}"}
        if insights.get('recent_success_rate_24h', 0) < 0.1 and insights.get('total_calls_24h', 0) > 10: # Check total calls
            critique['feedback'] += " Low success rate (<10%) requires investigation into script/strategy/lead quality."
            critique['status'] = 'warning'
        # TODO: Add LLM call for deeper critique based on insights and conversation logs if needed
        return critique

    # MODIFIED: Added needs_dict and enriched_data parameters
    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Constructs context-rich prompts for LLM calls, including needs and enriched data."""
        self.logger.debug(f"Generating dynamic prompt for VoiceAgent task: {task_context.get('task')}")
        prompt_parts = [self.meta_prompt.format(self=self)] # Format self.internal_state parts
        prompt_parts.append("\n--- Current Task Context ---")
        # Prioritize certain keys for clarity
        priority_keys = ['task', 'current_call_state', 'client_info', 'discovered_needs', 'enriched_data', 'conversation_history', 'instructions']
        for key in priority_keys:
            if key in task_context:
                value = task_context[key]
                value_str = ""; max_len = 1500
                if key in ['conversation_history', 'enriched_data', 'discovered_needs', 'meta_prompt']: max_len = 3500 # Allow more context for these
                if isinstance(value, str): value_str = value[:max_len] + ("..." if len(value) > max_len else "")
                elif isinstance(value, (int, float, bool)): value_str = str(value)
                elif isinstance(value, (dict, list)):
                    try: value_str = json.dumps(value, default=str, indent=2); value_str = value_str[:max_len] + ("..." if len(value_str) > max_len else "")
                    except TypeError: value_str = str(value)[:max_len] + "..."
                else: value_str = str(value)[:max_len] + "..."
                prompt_parts.append(f"**{key.replace('_', ' ').title()}**: {value_str}")

        # Add remaining context items concisely
        other_params = {k: v for k, v in task_context.items() if k not in priority_keys and k not in ['desired_output_format']}
        if other_params:
            prompt_parts.append("\n**Other Parameters:**")
            try: prompt_parts.append(f"```json\n{json.dumps(other_params, default=str, indent=2)}\n```")
            except TypeError: prompt_parts.append(str(other_params)[:500] + "...")

        # Instructions are now part of the task_context if provided by the calling method
        if "instructions" not in task_context: # Fallback instructions if not explicitly set by caller
            prompt_parts.append("\n--- Instructions ---")
            task_type = task_context.get('task')
            if task_type == 'Interpret client intent AND extract needs':
                prompt_parts.append(f"Analyze the 'Client Response'. Determine primary 'intent' from {task_context.get('possible_intents', [])}, 'confidence' (0.0-1.0), 'sub_intents', 'emotional_tone'. Also, extract relevant info for these needs: {task_context.get('needs_to_extract', [])} based on the response and update the 'Current Discovered Needs'.")
            elif task_type == 'Generate agent response for sales call':
                prompt_parts.append(f"Generate the next agent response based on the 'Current Call State' ({task_context.get('current_call_state', 'unknown')}), conversation history, client profile, discovered needs, and enriched data.")
                prompt_parts.append("Adhere to the Core Principles (Hormozi inspired): Focus on value, handle objections gracefully (especially price - change terms, not price), guide towards the $7000 close or clear next step.")
                prompt_parts.append("Keep the response concise, natural, and suitable for Deepgram Aura TTS.")
            else:
                prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")

        if task_context.get('desired_output_format'):
            prompt_parts.append(f"\n**Output Format:** {task_context['desired_output_format']}")
            if "JSON" in task_context.get('desired_output_format', ''):
                prompt_parts.append("\nRespond ONLY with valid JSON matching the specified format. Do not include explanations or markdown formatting outside the JSON structure.")
                prompt_parts.append("```json") # Ensure this is closed if LLM is expected to complete the JSON block

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for VoiceSalesAgent (length: {len(final_prompt)} chars)")
        # self.logger.debug(f"FULL PROMPT:\n{final_prompt}") # Uncomment for full prompt debugging
        return final_prompt

    async def collect_insights(self) -> Dict[str, Any]:
        """Provide agent-specific insights (e.g., recent success rate)."""
        success_rate = 0.0; avg_duration = 0.0; total_calls = 0
        try:
            async with self.session_maker() as session:
                threshold = datetime.now(timezone.utc) - timedelta(hours=24)
                stmt = select(
                    func.count(CallLog.id).label('total_calls'),
                    func.sum(case((CallLog.outcome.like('success%'), 1), else_=0)).label('successful_calls'), # Count outcomes starting with 'success'
                    func.avg(CallLog.duration_seconds).label('average_duration')
                ).where(CallLog.timestamp >= threshold)
                result = await session.execute(stmt)
                stats = result.mappings().first()
                if stats:
                    total_calls = stats['total_calls'] or 0
                    if total_calls > 0:
                        success_rate = (stats['successful_calls'] or 0) / total_calls
                        avg_duration = stats['average_duration'] or 0.0
        except Exception as e: self.logger.error(f"Failed to calculate call insights: {e}", exc_info=True)

        return {
            "agent_name": self.AGENT_NAME, "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_calls": len(self.internal_state.get('active_calls', {})),
            "total_calls_24h": total_calls, # Added total calls
            "recent_success_rate_24h": round(success_rate, 3),
            "average_call_duration_24h_sec": round(avg_duration, 1)
            }

    # --- Helper to log learned patterns (delegates) ---
    async def log_learned_pattern(self, *args, **kwargs):
        """Logs a learned pattern via ThinkTool."""
        if self.think_tool and hasattr(self.think_tool, 'log_learned_pattern'):
            await self.think_tool.log_learned_pattern(*args, **kwargs)
        else: self.logger.error("ThinkTool unavailable or missing log_learned_pattern method.")

# --- End of agents/voice_sales_agent.py ---