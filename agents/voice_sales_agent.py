# Filename: agents/voice_sales_agent.py
# Description: Production-ready Voice Sales Agent using Twilio, Deepgram, and LLMs.
# Version: 3.1 (Added UGC Discovery & Hormozi Logic)

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
    from deepgram import DeepgramClient, SpeakOptions
    from deepgram.clients.live.v1.client import LiveClient as DeepgramLiveClient # Alias for clarity
    from deepgram.clients.speak.v1.client import SpeakClient as DeepgramSpeakClient # Alias for clarity
    from deepgram.errors import DeepgramError, DeepgramApiError
except ImportError: logging.critical("Deepgram SDK v3+ not found."); raise

# --- Utilities ---
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

# Configure logger
logger = logging.getLogger(__name__)
# Configure dedicated operational logger
op_logger = logging.getLogger('OperationalLog') # Assuming setup elsewhere

# --- Meta Prompt ---
# MODIFIED: Added emphasis on discovery and Hormozi negotiation
VOICE_AGENT_META_PROMPT = """
You are a world-class AI Voice Sales Agent within the Synapse AI Sales System, specializing in high-ticket B2B sales for UGC services ($7000 package). Your primary goal is to guide prospects through a value-driven conversation, **discover their specific needs to tailor the offer**, and lead to a successful close or clear next steps, using Hormozi-inspired principles.
**Core Principles (Hormozi Inspired):**
1.  **Irresistible Offer Focus:** Clearly articulate massive value of the tailored UGC package. Frame around desired outcome vs. current state.
2.  **Value Stacking:** Emphasize multiple benefits relevant to the *discovered needs*.
3.  **Risk Reversal:** Use guarantees (if provided by ThinkTool) to minimize perceived risk.
4.  **Urgency & Scarcity (Ethical):** Create genuine reasons to act sooner (if context provided).
5.  **Problem/Solution Fit:** **Actively discover** prospect's challenges (from context/conversation) and tailor the $7000 UGC service package as the precise solution.
6.  **Clarity & Conciseness:** Simple language, avoid jargon. Natural, conversational tone (Deepgram Aura).
7.  **Adaptive Conversation:** Listen intently (STT), understand intent/emotion (LLM), adjust flow using state machine (including needs discovery).
8.  **Profit Maximization & Firm Pricing:** Guide towards the $7000 close. **Do not offer discounts.** If price is the *only* objection after value is established, offer to *change the terms/scope* (e.g., remove a component) but maintain the price for the defined package. Reference Alex Hormozi's negotiation principles from KB if needed.
9.  **Compliance:** Adhere strictly to LegalAgent guidelines (call times, disclosures). Check opt-in status.
**Operational Flow:**
- Receive task (initiate call) from Orchestrator.
- Check Client opt-in status in DB.
- Consult LegalAgent for pre-call compliance.
- Initiate call via Twilio.
- Manage real-time interaction: Deepgram STT -> LLM Intent/Emotion Analysis -> State Update (includes **Needs Discovery**) -> LLM Response Generation (using KB insights, **discovered needs**, learned phrases) -> Deepgram TTS (Aura) -> Twilio Playback.
- Handle objections by reframing around value/risk reversal, **using Hormozi negotiation for price objections**.
- Aim for clear closing state (deal won at $7000 -> trigger invoice, follow-up scheduled, or disqualification).
- Log call details, transcript, outcome meticulously to Postgres.
- Provide performance data for ThinkTool's learning loop.
"""

class VoiceSalesAgent(GeniusAgentBase):
    """
    Voice Agent (Genius Level): Manages real-time voice sales calls using Twilio,
    Deepgram, and LLM reasoning. Implements adaptive conversation, dynamic pricing,
    opt-out checks, and contributes to learning loops.
    Version: 3.1 (Added UGC Discovery & Hormozi Logic)
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
        except Exception as e:
            self.logger.critical(f"Failed to initialize Deepgram client: {e}", exc_info=True)
            raise ValueError(f"Deepgram client initialization failed: {e}") from e

        self.logger.info(f"{self.AGENT_NAME} v3.1 initialized.")

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
            websocket_url = f"{base_url}/twilio_call" # Endpoint defined in Orchestrator routes
            # Use wss:// if AGENCY_BASE_URL uses https://
            ws_scheme = "wss://" if websocket_url.startswith("https://") else "ws://"
            ws_host_path = websocket_url.split("://")[-1]
            stream_url = f"{ws_scheme}{ws_host_path}"

            call = await asyncio.to_thread(
                self.twilio_client.calls.create,
                to=client.phone,
                from_=self.twilio_voice_number,
                twiml=f'<Response><Connect><Stream url="{stream_url}"/></Connect></Response>',
                record=True # Enable recording
            )
            call_sid = call.sid
            self.logger.info(f"Initiated outbound call to {client.phone}. Call SID: {call_sid}")
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
        deepgram_ws: Optional[websockets.client.WebSocketClientProtocol] = None
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
            dg_config = {
                "model": self.internal_state['deepgram_stt_model'], "language": "en-US",
                "encoding": "mulaw", "sample_rate": 8000, "channels": 1,
                "punctuate": True, "interim_results": False, "endpointing": 300, "vad_events": True
            }
            live_client: DeepgramLiveClient = self.deepgram_client.listen.asynclive.v("1")

            # Queue for receiving transcripts from the callback
            transcript_queue = asyncio.Queue()

            async def on_message(inner_self, result, **kwargs): # Use inner_self to avoid conflict
                sentence = result.channel.alternatives[0].transcript # Access transcript correctly
                if result.is_final and sentence:
                    logger.debug(f"Deepgram Callback (Call: {call_sid}): Final transcript: {sentence}")
                    await transcript_queue.put({"transcript": sentence, "confidence": result.channel.alternatives[0].confidence, "is_final": True})

            async def on_error(inner_self, error, **kwargs): # Use inner_self
                logger.error(f"Deepgram Callback Error (Call: {call_sid}): {error}")
                await transcript_queue.put({"error": str(error)}) # Signal error to main loop

            live_client.on(DeepgramLiveClient.Events.Transcript, on_message)
            live_client.on(DeepgramLiveClient.Events.Error, on_error)

            await live_client.start(dg_config)
            self.logger.info(f"Deepgram WebSocket connection started for call {call_sid}.")

            # --- Register Connection for Audio Forwarding ---
            if hasattr(self.orchestrator, 'register_deepgram_connection_sdk'):
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
                    agent_response = "I’m part of a cutting-edge team using advanced technology to deliver top-tier UGC services efficiently. We focus on results – how can I help you achieve yours?"
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
                    call_outcome = "success_sale" # Assume sale if ended via state machine

                    # --- Trigger Invoice (Using fixed $7k price) ---
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
        except Exception as e:
            self.logger.critical(f"Unhandled critical error in handle_call for {call_sid}: {e}", exc_info=True)
            await self._report_error(f"Critical error in call {call_sid}: {e}")
            call_outcome = "failed_agent_error"

        # --- Finally block ensures cleanup happens ---
        finally:
            self.logger.info(f"Cleaning up call {call_sid}. Final state: '{state}', Determined Outcome: '{call_outcome}'.")
            self.internal_state['active_calls'].pop(call_sid, None) # Remove from active tracking

            # --- Stop Deepgram Connection ---
            if 'live_client' in locals() and live_client:
                try: await live_client.finish(); self.logger.debug(f"Deepgram SDK client finished for {call_sid}.")
                except Exception as dg_close_err: self.logger.warning(f"Error finishing Deepgram SDK client for {call_sid}: {dg_close_err}")
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
            except TwilioException as te: self.logger.warning(f"Failed to fetch or update final Twilio status for call {call_sid}: {te}")


    # --- Retry Wrappers for I/O Operations ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, DeepgramError, TwilioException, RuntimeError)), reraise=True)
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
                timeout=self.internal_state.get('openrouter_intent_timeout')
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
                task_context["instructions"] = "Ask open-ended questions to understand the client's primary goals, target audience, desired outcomes, and current challenges related to content/marketing."
            elif state == "value_proposition":
                task_context["instructions"] = f"Based on the discovered needs ({needs_dict}), present the $7000 UGC package as the specific solution. Stack the value by highlighting benefits relevant to their needs."
            elif state == "objection_handling":
                last_client_msg = conversation_log[-1]['text'].lower() if conversation_log and conversation_log[-1]['role'] == 'client' else ""
                if "price" in last_client_msg or "cost" in last_client_msg or "afford" in last_client_msg:
                    task_context["instructions"] = f"Handle the price objection ($7000). Reiterate value based on needs ({needs_dict}). State the price is firm. If they insist on budget limitations after value is clear, offer to potentially adjust the *scope/terms* (e.g., remove a component) but NOT the price. Reference Hormozi principles (no discounts)."
                else:
                    task_context["instructions"] = f"Address the client's objection ('{last_client_msg}') by reframing around value, risk reversal (if applicable), or clarifying misunderstandings based on needs ({needs_dict})."
            elif state == "closing":
                task_context["instructions"] = f"Client seems interested. Summarize the tailored value based on needs ({needs_dict}). Use the preferred closing phrase or similar to ask for the sale ($7000)."
            elif state == "finalizing":
                task_context["instructions"] = "Confirm the agreement ($7000 package tailored to their needs). Explain the next steps (invoice, onboarding)."
            elif state == "addressing_hesitancy":
                task_context["instructions"] = "Acknowledge the client's hesitation. Ask clarifying questions to uncover the root cause. Reassure and build trust."
            elif state == "greeting":
                task_context["instructions"] = "Provide a warm, professional opening. State your name and company. Briefly state the reason for the call (tailored if possible using enriched data)."

            response_prompt = await self.generate_dynamic_prompt(task_context)
            llm_model_pref = settings.OPENROUTER_MODELS.get('voice_response')
            llm_response_str = await self.orchestrator._call_llm_with_retry( # Use orchestrator's method
                response_prompt, agent_name=self.AGENT_NAME,
                model=llm_model_pref,
                temperature=0.65, max_tokens=300, # Increased tokens slightly
                timeout=self.internal_state.get('openrouter_response_timeout')
            )
            if not llm_response_str: raise Exception("LLM call returned empty.")
            agent_response = llm_response_str.strip()
            if not agent_response: raise ValueError("Empty response generated")
            self.logger.debug(f"Generated response for state '{state}': '{agent_response[:100]}...'")
        except Exception as e:
            self.logger.error(f"Error during generate_agent_response: {e}", exc_info=True)
            # Provide state-specific fallbacks (same as previous version)
            sender_name = self.config.get('SENDER_NAME', 'Alex')
            client_first_name = client.name.split()[0] if client.name else 'there' # Get first name
            fallback_responses = {
                "greeting": f"Hi {client_first_name}, this is {sender_name} calling from the UGC team. How are you?",
                "needs_assessment": "To make sure I respect your time, could you tell me a bit about your current content goals?",
                "value_proposition": "Got it. Our UGC service helps businesses like yours boost trust and conversions with authentic content. How does that sound?",
                "addressing_hesitancy": "I understand. What are your main concerns right now?",
                "objection_handling": "That's a fair point. Let me clarify how we address that...",
                "closing": "Based on what you've shared, it sounds like our $7000 package could be a great fit. Are you ready to move forward?",
                "finalizing": "Excellent. I'll confirm the details for the $7000 package and outline the next steps.",
                "end_call": "Okay, thank you for your time. Have a great day. Goodbye."
            }
            agent_response = fallback_responses.get(state, "Apologies, I'm having a slight technical issue. Could you please repeat that?")
        return agent_response

    # --- TTS & Database Methods ---

    async def speak_response(self, text: str, call_sid: str):
        """Convert text to speech using Deepgram Aura TTS and deliver via Twilio."""
        if not text: self.logger.warning(f"Attempted to speak empty text for call {call_sid}."); return
        self.logger.debug(f"Synthesizing speech for call {call_sid}: '{text[:100]}...'")
        try:
            options = SpeakOptions(model=self.internal_state['aura_voice'], encoding="mulaw", container="wav", sample_rate=8000)
            speak_client: DeepgramSpeakClient = self.deepgram_client.speak
            async with speak_client.stream({"text": text}, options) as streamer:
                audio_data = await streamer.read()
                if not audio_data: raise DeepgramError("Failed to get audio data from Deepgram stream.")

            if hasattr(self.orchestrator, 'host_temporary_audio'):
                audio_url = await self.orchestrator.host_temporary_audio(audio_data, f"{call_sid}_{int(time.time())}.wav")
                if not audio_url: raise RuntimeError("Failed to host temporary audio via Orchestrator")
            else: raise NotImplementedError("Audio hosting mechanism not available via Orchestrator.")

            self.logger.debug(f"Synthesized audio URL for {call_sid}: {audio_url}")
            twiml = f"<Response><Play>{audio_url}</Play></Response>"
            await asyncio.to_thread(self.twilio_client.calls(call_sid).update, twiml=twiml)
            self.logger.info(f"Instructed Twilio to speak response for call {call_sid}")
        except DeepgramError as e: self.logger.error(f"Deepgram TTS failed for call {call_sid}: {e}", exc_info=True); raise
        except TwilioException as e: self.logger.error(f"Twilio call update failed for call {call_sid}: {e}", exc_info=True); raise
        except Exception as e: self.logger.error(f"Unexpected error in speak_response for call {call_sid}: {e}", exc_info=True); raise

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
                    if media_url_path and media_url_path.endswith('.json'): media_url_path = media_url_path[:-5]
                    if media_url_path: recording_url = f"https://api.twilio.com{media_url_path}"; self.logger.info(f"Found recording: {recording_url}")
            except TwilioRestException as rec_err:
                if rec_err.status == 404: logger.info(f"No recordings found for call {call_sid}.")
                else: logger.warning(f"Could not fetch recordings for {call_sid}: {rec_err}")
            except Exception as rec_err_gen: logger.error(f"Unexpected error fetching recordings for {call_sid}: {rec_err_gen}")

        except TwilioException as e: logger.warning(f"Could not fetch final Twilio call details for {call_sid}: {e}")
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
        prompt_parts = [self.meta_prompt]
        prompt_parts.append("\n--- Current Task Context ---")
        # Prioritize certain keys for clarity
        priority_keys = ['task', 'current_call_state', 'client_info', 'discovered_needs', 'enriched_data', 'conversation_history', 'instructions']
        for key in priority_keys:
            if key in task_context:
                value = task_context[key]
                value_str = ""; max_len = 1500
                if key in ['conversation_history', 'enriched_data', 'discovered_needs']: max_len = 2500 # Allow more context
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
        if "instructions" not in task_context:
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
                prompt_parts.append("```json")

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for VoiceSalesAgent (length: {len(final_prompt)} chars)")
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