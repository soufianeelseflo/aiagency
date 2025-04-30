# Filename: agents/voice_sales_agent.py
# Description: Production-ready Voice Sales Agent using Twilio, Deepgram, and LLMs.
# Version: 2.0 (Full Implementation)

import asyncio
import logging
import json
from datetime import datetime, timedelta, timezone
import pytz
import time
import uuid # Added for unique identifiers if needed
import re # Added for potential text cleaning

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker # Ensure async_sessionmaker is imported
from sqlalchemy import text as sql_text, select, update, desc, case, func # Import necessary SQLAlchemy components
import sqlalchemy # For error handling

# --- External Service Libraries ---
try:
    from twilio.rest import Client as TwilioClient
    from twilio.base.exceptions import TwilioException, TwilioRestException
except ImportError:
    logging.critical("Twilio library not found. Please install: pip install twilio")
    raise ImportError("Twilio library is required for VoiceSalesAgent")

try:
    import websockets
    import websockets.exceptions
except ImportError:
    logging.critical("Websockets library not found. Please install: pip install websockets")
    raise ImportError("Websockets library is required for VoiceSalesAgent")

try:
    from deepgram import DeepgramClient, SpeakOptions # Use updated SDK structure if necessary
    from deepgram.clients.live.v1.client import LiveClient # For STT
    # from deepgram.clients.prerecorded.v1.client import PrerecordedClient # If needed
    from deepgram.clients.speak.v1.client import SpeakClient # For TTS
    from deepgram.errors import DeepgramError, DeepgramApiError # Updated error handling
except ImportError:
    logging.critical("Deepgram SDK not found. Please install: pip install deepgram-sdk>=3.0.0")
    raise ImportError("Deepgram SDK is required for VoiceSalesAgent")

# --- LLM Client ---
# Assuming OpenAI compatible client provided by orchestrator
from openai import AsyncOpenAI as AsyncLLMClient

# --- Utilities ---
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import numpy as np # Keep for potential calculations
from typing import Optional, Dict, Any, List, Tuple # Ensure all needed types are here

# --- Project Imports ---
try:
    from .base_agent import GeniusAgentBase, KBInterface # Use relative import
except ImportError:
    from base_agent import GeniusAgentBase, KBInterface # Fallback

from models import Client, CallLog, ConversationState # Import necessary models
# from config.settings import settings # Config accessed via self.config

# Define the meta prompt string incorporating Hormozi techniques
VOICE_AGENT_META_PROMPT = """
You are a world-class AI Voice Sales Agent, specializing in high-ticket B2B sales for UGC (User-Generated Content) services. Your primary goal is to guide prospects through a value-driven conversation, leading to a successful close or clear next steps.

**Core Principles (Hormozi Inspired):**
1.  **Irresistible Offer Focus:** Clearly articulate the massive value proposition. Frame the conversation around the prospect's desired outcome (e.g., increased conversions, brand trust, engagement) versus their current state.
2.  **Value Stacking:** Emphasize the multiple components and benefits included in the service to build perceived value far exceeding the price.
3.  **Risk Reversal:** Utilize strong guarantees (if applicable and defined in context) to minimize the prospect's perceived risk.
4.  **Urgency & Scarcity (Use Ethically):** If applicable based on context (e.g., limited onboarding slots, introductory pricing), create genuine reasons for the prospect to act sooner rather than later.
5.  **Problem/Solution Fit:** Deeply understand the prospect's specific content challenges and tailor the presentation of the UGC service as the precise solution.
6.  **Clarity & Conciseness:** Communicate complex ideas simply. Avoid jargon.
7.  **Adaptive Conversation:** Listen intently, understand the prospect's intent and emotional state, and adjust the conversation flow accordingly using the defined state machine.
8.  **Profit Maximization:** While providing immense value, guide the conversation towards a profitable outcome for the business.
9.  **Compliance:** Adhere strictly to all legal and ethical guidelines provided.

**Operational Flow:**
- Initiate or receive calls via Twilio.
- Use Deepgram for real-time STT and TTS (Aura voice).
- Interpret client intent using LLM analysis.
- Update conversation state based on intent, confidence, and tone.
- Generate contextually relevant, value-focused responses using LLM, guided by these principles and the current state.
- Handle objections by reframing around value and risk reversal.
- Aim for a clear closing state (deal won, follow-up scheduled, or clear disqualification).
- Log all interactions meticulously.
- Continuously learn from call outcomes to refine strategy (handled by learning loop).
"""

# Configure logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class VoiceSalesAgent(GeniusAgentBase):
    """
    Voice Agent (Genius Level): Manages real-time voice sales calls using Twilio,
    Deepgram (STT/TTS), and internal LLM reasoning. Focuses on adaptive conversation,
    deep learning from interactions, and maximizing profitable outcomes.
    Version: 2.0
    """
    AGENT_NAME = "VoiceSalesAgent"

    def __init__(self, session_maker: async_sessionmaker[AsyncSession], orchestrator: Any, twilio_auth_token: str, deepgram_api_key: str):
        """Initializes the VoiceSalesAgent."""
        # Get dependencies from orchestrator
        config = getattr(orchestrator, 'config', None)
        kb_interface = getattr(orchestrator, 'kb_interface', None)
        super().__init__(agent_name=self.AGENT_NAME, kb_interface=kb_interface, orchestrator=orchestrator, config=config)

        self.session_maker = session_maker
        self.think_tool = orchestrator.agents.get('think')

        # Store passed-in secrets
        self._twilio_auth_token = twilio_auth_token
        self._deepgram_api_key = deepgram_api_key

        # --- Internal State Initialization ---
        self.internal_state = getattr(self, 'internal_state', {})
        self.internal_state['target_country'] = self.config.get("VOICE_TARGET_COUNTRY", "US") # Use US standard
        self.internal_state['aura_voice'] = self.config.get("DEEPGRAM_AURA_VOICE", "aura-asteria-en")
        self.internal_state['deepgram_stt_model'] = self.config.get("DEEPGRAM_STT_MODEL", "nova-2-general")
        self.internal_state['payment_terms'] = self.config.get("PAYMENT_TERMS", "Standard payment terms apply.")
        self.internal_state['intent_confidence_threshold'] = float(self.config.get("VOICE_INTENT_CONFIDENCE_THRESHOLD", 0.6))
        self.internal_state['openrouter_intent_model'] = self.config.get("OPENROUTER_MODELS", {}).get('voice_intent_llm', "google/gemini-flash-1.5")
        self.internal_state['openrouter_response_model'] = self.config.get("OPENROUTER_MODELS", {}).get('voice_response_llm', "google/gemini-flash-1.5")
        self.internal_state['deepgram_receive_timeout'] = float(self.config.get("DEEPGRAM_RECEIVE_TIMEOUT_S", 60.0))
        self.internal_state['openrouter_intent_timeout'] = float(self.config.get("OPENROUTER_INTENT_TIMEOUT_S", 10.0))
        self.internal_state['openrouter_response_timeout'] = float(self.config.get("OPENROUTER_RESPONSE_TIMEOUT_S", 15.0))
        self.internal_state['active_calls'] = {} # Track state per call_sid: {'state': '...', 'log': [], 'client_id': ...}

        # --- Essential Clients ---
        self.twilio_account_sid = self.config.get("TWILIO_ACCOUNT_SID")
        self.twilio_voice_number = self.config.get("TWILIO_VOICE_NUMBER") # Outbound number

        if not self.twilio_account_sid or not self.twilio_voice_number:
             self.logger.critical(f"{self.AGENT_NAME}: Missing critical configuration: TWILIO_ACCOUNT_SID or TWILIO_VOICE_NUMBER.")
             raise ValueError("Missing TWILIO_ACCOUNT_SID or TWILIO_VOICE_NUMBER configuration")
        if not self._twilio_auth_token:
             self.logger.critical(f"{self.AGENT_NAME}: Twilio Auth Token was not provided during initialization.")
             raise ValueError("Missing Twilio Auth Token")
        if not self._deepgram_api_key:
             self.logger.critical(f"{self.AGENT_NAME}: Deepgram API Key was not provided during initialization.")
             raise ValueError("Missing Deepgram API Key")

        try:
            self.twilio_client = TwilioClient(self.twilio_account_sid, self._twilio_auth_token)
            self.logger.info(f"{self.AGENT_NAME}: Twilio client initialized.")
        except TwilioException as e:
             self.logger.critical(f"{self.AGENT_NAME}: Failed to initialize Twilio client: {e}. Check credentials.")
             raise ValueError(f"Twilio client initialization failed: {e}") from e

        try:
            # Initialize Deepgram client using key from argument
            # Assuming DeepgramClient is the correct entry point for the SDK version
            self.deepgram_client = DeepgramClient(self._deepgram_api_key)
            self.logger.info(f"{self.AGENT_NAME}: Deepgram client initialized.")
        except Exception as e:
            self.logger.critical(f"{self.AGENT_NAME}: Failed to initialize Deepgram client: {e}", exc_info=True)
            raise ValueError(f"Deepgram client initialization failed: {e}") from e

        self.meta_prompt = VOICE_AGENT_META_PROMPT

        self.logger.info(f"{self.AGENT_NAME} (Genius Level) v2.0 initialized. Target: {self.internal_state['target_country']}, Voice: {self.internal_state['aura_voice']}")

    async def predict_optimal_call_time(self, client: Client) -> datetime:
        """Predict optimal call times using historical success data and client timezone."""
        optimal_time_utc: Optional[datetime] = None
        try:
            async with self.session_maker() as session:
                stmt = select(CallLog.timestamp).where(
                    CallLog.client_id == client.id, CallLog.outcome == 'success'
                ).order_by(desc(CallLog.timestamp)).limit(100)
                result = await session.execute(stmt)
                success_timestamps = [row.timestamp for row in result.mappings().all()] # Use mappings

            client_tz_str = client.timezone or "UTC"
            try: client_tz = pytz.timezone(client_tz_str)
            except pytz.UnknownTimeZoneError:
                self.logger.warning(f"Unknown timezone '{client_tz_str}' for client {client.id}. Defaulting to UTC.")
                client_tz = pytz.utc

            now_client_tz = datetime.now(client_tz)

            if success_timestamps:
                success_hours = [ts.astimezone(client_tz).hour for ts in success_timestamps]
                # Predict based on mode of successful hours within business hours (9-17)
                business_hours = [h for h in success_hours if 9 <= h <= 17]
                if business_hours:
                    from collections import Counter
                    hour_counts = Counter(business_hours)
                    optimal_hour_local = hour_counts.most_common(1)[0][0]
                else:
                    optimal_hour_local = 10 # Fallback if no success in business hours
            else:
                optimal_hour_local = 10 # Default 10 AM

            optimal_time_client_tz = now_client_tz.replace(hour=optimal_hour_local, minute=random.randint(0, 15), second=0, microsecond=0) # Add jitter
            if optimal_time_client_tz <= now_client_tz:
                optimal_time_client_tz += timedelta(days=1)

            optimal_time_utc = optimal_time_client_tz.astimezone(pytz.utc)
            self.logger.info(f"Predicted optimal call time for client {client.id} ({client.name}): {optimal_time_utc} UTC ({optimal_time_client_tz} {client_tz_str})")

        except sqlalchemy.exc.SQLAlchemyError as db_err:
             self.logger.error(f"DB error predicting optimal call time for client {client.id}: {db_err}", exc_info=True)
        except Exception as e:
             self.logger.error(f"Error predicting optimal call time for client {client.id}: {e}", exc_info=True)

        if optimal_time_utc is None:
             optimal_time_utc = datetime.now(timezone.utc) + timedelta(minutes=random.randint(5, 60)) # Fallback: 5-60 mins from now
             self.logger.warning(f"Failed to predict optimal time for client {client.id}, using fallback: {optimal_time_utc} UTC")

        return optimal_time_utc

    async def initiate_outbound_call(self, client: Client) -> Optional[str]:
        """Initiates an outbound call via Twilio."""
        if not client.phone:
            self.logger.error(f"Cannot initiate call: Client {client.id} has no phone number.")
            return None

        # Predict optimal time (optional, can be done before queuing task)
        # optimal_time = await self.predict_optimal_call_time(client)
        # await asyncio.sleep((optimal_time - datetime.now(timezone.utc)).total_seconds())

        # Pre-call compliance check
        legal_agent = self.orchestrator.agents.get('legal') # Assuming 'legal' is the key
        if legal_agent:
            compliance_context = f"Initiating outbound sales call to {client.name} (ID: {client.id}) at {client.phone} in {client.country}. Purpose: UGC Sales."
            compliance_check = await legal_agent.validate_operation(compliance_context)
            if not compliance_check.get('is_compliant', False):
                issues = compliance_check.get('compliance_issues', ['Unknown compliance issue'])
                self.logger.error(f"Outbound call to client {client.id} blocked by LegalAgent: {issues}")
                await self.orchestrator.send_notification("Compliance Alert - Call Blocked", f"Outbound call to {client.name} ({client.id}) blocked: {issues}")
                return None
        else:
            self.logger.warning("LegalAgent not found, proceeding without pre-call compliance check.")

        try:
            await self._internal_think(f"Initiating outbound call to {client.name} ({client.phone}) from {self.twilio_voice_number}.")
            # URL for Twilio to connect to your WebSocket endpoint for audio streaming
            # This endpoint needs to be publicly accessible and handle the Twilio stream protocol.
            websocket_url = f"{self.config.get('AGENCY_BASE_URL', 'http://localhost:5000')}/twilio_call" # Example endpoint

            call = await asyncio.to_thread(
                self.twilio_client.calls.create,
                to=client.phone,
                from_=self.twilio_voice_number,
                twiml=f'<Response><Connect><Stream url="wss://{websocket_url.split("://")[-1]}"/></Connect></Response>', # Use wss for secure websocket
                # Add recording parameters if needed: record=True, recording_status_callback=...
                record=True
            )
            call_sid = call.sid
            self.logger.info(f"Initiated outbound call to {client.phone}. Call SID: {call_sid}")
            # Store initial state immediately
            self.internal_state['active_calls'][call_sid] = {'state': 'initiating', 'log': [], 'client_id': client.id}
            await self.store_conversation_state_with_retry(call_sid, 'initiating', [])
            return call_sid
        except TwilioException as e:
            self.logger.error(f"Failed to initiate Twilio call to {client.phone}: {e}", exc_info=True)
            await self.orchestrator.report_error(self.AGENT_NAME, f"Twilio call initiation failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error initiating outbound call: {e}", exc_info=True)
            await self.orchestrator.report_error(self.AGENT_NAME, f"Unexpected call initiation error: {e}")
            return None

    async def handle_call(self, call_sid: str, client: Client):
        """
        Manage a real-time voice call: STT -> NLU -> Response -> TTS.
        Assumes Twilio is streaming audio to a WebSocket endpoint managed by Orchestrator.
        """
        self.logger.info(f"{self.AGENT_NAME}: Handling call {call_sid} for client {client.id} ({client.name})")
        deepgram_ws: Optional[websockets.client.WebSocketClientProtocol] = None
        state = "greeting"
        conversation_log = []
        call_outcome = "failed_initialization" # Default outcome

        try:
            # --- Load State ---
            async with self.session_maker() as session:
                state_record = await session.get(ConversationState, call_sid)
                if state_record:
                    try:
                        state = state_record.state
                        conversation_log = json.loads(state_record.conversation_log)
                        self.logger.info(f"Resuming call {call_sid} from state '{state}'")
                    except json.JSONDecodeError:
                        self.logger.error(f"Failed to decode conversation log for {call_sid}. Starting fresh.")
                        state = "greeting"; conversation_log = []
                else:
                    self.logger.info(f"Starting new conversation state for call {call_sid}")
                    state = "greeting"; conversation_log = []
            # Update internal state tracking
            self.internal_state['active_calls'][call_sid] = {'state': state, 'log': conversation_log, 'client_id': client.id}

            # --- Connect to Deepgram STT ---
            dg_config: LiveClient.Options = { # Use correct type hint if available
                "model": self.internal_state['deepgram_stt_model'],
                "language": "en-US", # Make configurable if needed
                "encoding": "mulaw", # Twilio uses mulaw
                "sample_rate": 8000, # Twilio uses 8kHz
                "channels": 1,
                "punctuate": True,
                "interim_results": False, # Keep false for simplicity
                "endpointing": 300, # Milliseconds of silence to detect end of utterance
                "vad_events": True # Get VAD events if needed
            }
            deepgram_ws_url = f"wss://api.deepgram.com/v1/listen?{ '&'.join([f'{k}={v}' for k,v in dg_config.items()]) }"

            await self._internal_think(f"Connecting to Deepgram STT for call {call_sid}. URL: wss://api.deepgram.com/v1/listen?model={dg_config['model']}")
            deepgram_ws = await websockets.connect(
                deepgram_ws_url,
                extra_headers={"Authorization": f"Token {self._deepgram_api_key}"},
                ping_interval=10, ping_timeout=5
            )
            self.logger.info(f"Deepgram WebSocket connected for call {call_sid}.")

            # --- Register Connection for Audio Forwarding ---
            if hasattr(self.orchestrator, 'register_deepgram_connection'):
                await self.orchestrator.register_deepgram_connection(call_sid, deepgram_ws)
                self.logger.debug(f"Registered Deepgram WS for {call_sid} with Orchestrator.")
            else:
                raise RuntimeError("Orchestrator cannot register Deepgram connection.")

            # --- Initial Greeting (if not resuming) ---
            if not state_record:
                 await self._internal_think(f"Generating initial greeting for call {call_sid}, state: {state}.")
                 initial_greeting = await self.generate_agent_response_with_retry(state, client, conversation_log)
                 await self.speak_response_with_retry(initial_greeting, call_sid)
                 conversation_log.append({"role": "agent", "text": initial_greeting, "timestamp": datetime.now(timezone.utc).isoformat()})
                 await self.store_conversation_state_with_retry(call_sid, state, conversation_log)

            # --- Main Conversation Loop ---
            while True:
                # 1. Receive Transcription from Deepgram
                transcription_data = None
                try:
                    transcription_data = await asyncio.wait_for(
                        self._receive_deepgram_transcription(deepgram_ws),
                        timeout=self.internal_state['deepgram_receive_timeout']
                    )
                except asyncio.TimeoutError:
                     self.logger.warning(f"Timeout waiting for transcription from Deepgram for call {call_sid}.")
                     # Send a prompt if timeout occurs?
                     # await self.speak_response_with_retry("Are you still there?", call_sid)
                     continue
                except websockets.exceptions.ConnectionClosed:
                     self.logger.info(f"Deepgram WebSocket closed for {call_sid} while receiving.")
                     call_outcome = "disconnected_websocket"
                     break

                if transcription_data is None:
                    self.logger.info(f"No transcription received or WS closed for {call_sid}. Ending loop.")
                    # Determine if closure was expected or error
                    if deepgram_ws.closed: call_outcome = "disconnected_websocket"
                    break

                client_response = transcription_data.get("transcript", "").strip()
                if not client_response: continue # Ignore empty transcripts

                self.logger.info(f"Received transcript for {call_sid}: '{client_response}' (Conf: {transcription_data.get('confidence', 0.0):.2f})")
                conversation_log.append({"role": "client", "text": client_response, "timestamp": datetime.now(timezone.utc).isoformat()})

                # 2. Check for Hangup Intent
                hangup_keywords = ["bye", "goodbye", "stop", "end call", "remove me", "not interested"]
                if any(keyword in client_response.lower() for keyword in hangup_keywords):
                     self.logger.info(f"Hangup keyword detected in '{client_response}' for call {call_sid}.")
                     state = "end_call"; call_outcome = "disconnected_client"
                     agent_response = "Alright, thank you for your time. Goodbye."
                     await self.speak_response_with_retry(agent_response, call_sid)
                     conversation_log.append({"role": "agent", "text": agent_response, "timestamp": datetime.now(timezone.utc).isoformat()})
                     break

                # 3. Check for AI Detection Attempts
                detection_keywords = ["robot", "ai", "artificial", "bot", "computer voice"]
                if any(keyword in client_response.lower() for keyword in detection_keywords):
                    self.logger.info(f"Potential AI detection attempt in '{client_response}' for call {call_sid}.")
                    agent_response = "I’m part of a cutting-edge team using advanced technology to deliver top-tier UGC services efficiently. We focus on results – how can I help you achieve yours?"
                    await self.speak_response_with_retry(agent_response, call_sid)
                    conversation_log.append({"role": "agent", "text": agent_response, "timestamp": datetime.now(timezone.utc).isoformat()})
                    await self.store_conversation_state_with_retry(call_sid, state, conversation_log)
                    continue

                # 4. Interpret Intent
                await self._internal_think(f"Interpreting intent for call {call_sid}. Response: '{client_response[:50]}...'")
                intent, confidence, sub_intents, tone = await self.interpret_intent_with_retry(client_response)
                if intent == "unknown":
                     self.logger.warning(f"Failed to interpret intent for '{client_response}' (call {call_sid}). Asking for clarification.")
                     agent_response = "Sorry, I didn't quite catch that. Could you please rephrase?"
                     await self.speak_response_with_retry(agent_response, call_sid)
                     conversation_log.append({"role": "agent", "text": agent_response, "timestamp": datetime.now(timezone.utc).isoformat()})
                     await self.store_conversation_state_with_retry(call_sid, state, conversation_log)
                     continue

                # 5. Update State
                new_state = await self.update_conversation_state(state, intent, confidence, tone)
                if new_state != state:
                     self.logger.info(f"State transition for {call_sid}: {state} -> {new_state} (Intent: {intent}, Tone: {tone})")
                     state = new_state
                     self.internal_state['active_calls'][call_sid]['state'] = state # Update internal tracker
                else:
                     self.logger.debug(f"State remained '{state}' for call {call_sid} (Intent: {intent}, Confidence: {confidence:.2f}, Tone: {tone})")

                # 6. Generate Response
                await self._internal_think(f"Generating response for call {call_sid}, state: {state}.")
                agent_response = await self.generate_agent_response_with_retry(state, client, conversation_log)

                # 7. Speak Response
                await self.speak_response_with_retry(agent_response, call_sid)
                conversation_log.append({"role": "agent", "text": agent_response, "timestamp": datetime.now(timezone.utc).isoformat()})

                # 8. Store State
                await self.store_conversation_state_with_retry(call_sid, state, conversation_log)

                # 9. Check for End Call State
                if state == "end_call":
                    self.logger.info(f"Reached 'end_call' state for call {call_sid}. Ending conversation.")
                    call_outcome = "success" # Assume success if reached via state machine

                    # --- Dynamic Pricing / Invoicing Trigger ---
                    try:
                         await self._internal_think(f"Call {call_sid} ended successfully. Calculating price and triggering invoice.")
                         # Use ScoringAgent for score, ThinkTool for pricing logic?
                         scoring_agent = self.orchestrator.agents.get('scoring')
                         client_score = 0.0
                         if scoring_agent: client_score = await scoring_agent.score_client(client.id)

                         base_price = float(self.config.get("BASE_UGC_PRICE", 5000.0))
                         # Simplified pricing logic - enhance with ThinkTool/KB later
                         pricing = min(max(base_price * (1 + client_score / 200), 3000), 10000) # Score adjusts price +/- 50% around base

                         self.logger.info(f"Calculated price for client {client.id}: ${pricing:.2f} (Score: {client_score:.2f})")

                         if hasattr(self.orchestrator, 'request_invoice_generation'):
                             await self.orchestrator.request_invoice_generation(client.id, pricing, call_sid)
                         else: self.logger.warning("Orchestrator cannot request invoice generation.")

                    except Exception as price_err:
                         self.logger.error(f"Error during dynamic pricing/invoice trigger for call {call_sid}: {price_err}", exc_info=True)
                    # --- End Pricing Example ---
                    break # Exit loop

        # --- Exception Handling for the entire call ---
        except websockets.exceptions.ConnectionClosedOK:
            self.logger.info(f"Deepgram WebSocket connection closed normally for call {call_sid}.")
            call_outcome = "disconnected_websocket"
        except websockets.exceptions.ConnectionClosedError as e:
            self.logger.error(f"Deepgram WebSocket connection closed with error for {call_sid}: {e}", exc_info=True)
            call_outcome = "disconnected_websocket_error"
        except DeepgramError as e: # Catch specific Deepgram errors
            self.logger.error(f"Deepgram API error during call {call_sid}: {e}", exc_info=True)
            if hasattr(self.orchestrator, 'report_error'): await self.orchestrator.report_error(self.AGENT_NAME, f"Deepgram API error call {call_sid}: {e}")
            call_outcome = "failed_stt_error"
        except TwilioException as e:
             self.logger.error(f"Twilio API error during call {call_sid}: {e}", exc_info=True)
             if hasattr(self.orchestrator, 'report_error'): await self.orchestrator.report_error(self.AGENT_NAME, f"Twilio API error call {call_sid}: {e}")
             call_outcome = "failed_twilio_error"
        except Exception as e:
            self.logger.critical(f"Unhandled critical error in handle_call for {call_sid}: {e}", exc_info=True)
            if hasattr(self.orchestrator, 'report_error'): await self.orchestrator.report_error(self.AGENT_NAME, f"Critical error in call {call_sid}: {e}")
            call_outcome = "failed_agent_error"

        # --- Finally block ensures cleanup happens ---
        finally:
            self.logger.info(f"Cleaning up call {call_sid}. Final state: '{state}', Determined Outcome: '{call_outcome}'.")
            # Remove call from active tracking
            self.internal_state['active_calls'].pop(call_sid, None)

            # --- Unregister Deepgram Connection ---
            if deepgram_ws and hasattr(self.orchestrator, 'unregister_deepgram_connection'):
                await self.orchestrator.unregister_deepgram_connection(call_sid)
                self.logger.debug(f"Unregistered Deepgram WS for {call_sid} with Orchestrator.")
            if deepgram_ws and not deepgram_ws.closed:
                try: await deepgram_ws.close(); self.logger.debug(f"Closed Deepgram WS connection for {call_sid}.")
                except Exception as close_err: self.logger.warning(f"Error closing Deepgram WS for {call_sid}: {close_err}")

            # --- Post-call Compliance Check & Logging ---
            final_log_outcome = call_outcome
            legal_agent = self.orchestrator.agents.get('legal')
            if legal_agent:
                try:
                    post_compliance_context = (f"Completed call {call_sid} to {client.name} (ID: {client.id}). Outcome: {call_outcome}. Final State: {state}. Conversation Snippet: {json.dumps(conversation_log[-3:])}")
                    post_compliance = await legal_agent.validate_operation(post_compliance_context)
                    if not post_compliance.get('is_compliant', True):
                         issues = post_compliance.get('compliance_issues', 'Post-call compliance issue')
                         self.logger.warning(f"Post-call compliance issue for {call_sid}: {issues}")
                         final_log_outcome = "failed_compliance"
                         if hasattr(self.orchestrator, 'send_notification'): await self.orchestrator.send_notification("Compliance Alert - Post-Call", f"Post-call compliance issue for {client.name} ({call_sid}): {issues}")
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError)), reraise=True)
    async def interpret_intent_with_retry(self, response: str) -> Tuple[str, float, List[str], str]:
         return await self.interpret_intent(response)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError)), reraise=True)
    async def generate_agent_response_with_retry(self, state: str, client: Client, conversation_log: List[Dict]) -> str:
        return await self.generate_agent_response(state, client, conversation_log)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), retry=retry_if_exception_type(sqlalchemy.exc.SQLAlchemyError), reraise=True)
    async def store_conversation_state_with_retry(self, call_sid: str, state: str, conversation_log: List[Dict]):
        await self.store_conversation_state(call_sid, state, conversation_log)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), retry=retry_if_exception_type((sqlalchemy.exc.SQLAlchemyError, TwilioException)), reraise=True)
    async def store_call_log_with_retry(self, client: Client, conversation_log: List[Dict], outcome: str, call_sid: str):
         await self.store_call_log(client, conversation_log, outcome, call_sid)

    # --- Core Logic Methods ---

    async def interpret_intent(self, response: str) -> Tuple[str, float, List[str], str]:
        """Analyze client intent using internal logic and a single LLM call."""
        self.logger.debug(f"Interpreting intent for response: '{response[:100]}...'")
        intent = "unknown"; confidence = 0.0; sub_intents = []; emotional_tone = "neutral"

        try:
            # Use self.internal_state to get current call state if needed
            current_call_state = "unknown" # Default if not tracked or call_sid missing
            # Need a way to get call_sid here if not passed directly
            # For now, assume context is sufficient without explicit call_sid state

            task_context = {
                "task": "Interpret client intent during sales call",
                "client_response": response,
                "current_call_state": current_call_state, # Add if available
                "possible_intents": ["interested", "hesitant", "objection_cost", "objection_timing", "objection_need", "closing_signal", "clarification_request", "irrelevant", "hangup_signal"],
                "desired_output_format": "JSON: { \"intent\": \"...\", \"confidence\": 0.0-1.0, \"sub_intents\": [\"...\"], \"emotional_tone\": \"positive|negative|neutral\" }"
            }
            intent_prompt = await self.generate_dynamic_prompt(task_context)

            # Use orchestrator's LLM call method
            llm_response_str = await self.orchestrator._call_llm_with_retry(
                intent_prompt, agent_name=self.AGENT_NAME,
                model=self.internal_state.get('openrouter_intent_model'),
                temperature=0.1, max_tokens=150, is_json_output=True,
                timeout=self.internal_state.get('openrouter_intent_timeout')
            )

            if not llm_response_str: raise Exception("LLM call for intent interpretation returned empty response.")

            try:
                json_match = json.loads(llm_response_str[llm_response_str.find('{'):llm_response_str.rfind('}')+1])
                intent = json_match.get('intent', 'unknown')
                confidence = float(min(max(json_match.get('confidence', 0.0), 0.0), 1.0))
                sub_intents = json_match.get('sub_intents', [])
                emotional_tone = json_match.get('emotional_tone', 'neutral')
                self.logger.info(f"Interpreted intent: {intent} (conf: {confidence:.2f}, tone: {emotional_tone})")
            except (json.JSONDecodeError, KeyError, ValueError) as parse_err:
                 self.logger.warning(f"Failed to parse LLM intent response ({parse_err}): {llm_response_str}")

        except Exception as e:
            self.logger.error(f"Error during interpret_intent LLM call: {e}", exc_info=True)

        return intent, confidence, sub_intents, emotional_tone

    async def update_conversation_state(self, current_state: str, intent: str, confidence: float, emotional_tone: str) -> str:
        """Advance the conversation state based on intent, confidence, and tone."""
        # Refined state machine
        state_transitions = {
            "greeting":           {"interested": "needs_assessment", "hesitant": "addressing_hesitancy", "objection": "objection_handling", "closing_signal": "closing", "clarification_request": "greeting", "irrelevant": "greeting", "hangup_signal": "end_call"},
            "needs_assessment":   {"interested": "value_proposition", "hesitant": "addressing_hesitancy", "objection": "objection_handling", "closing_signal": "closing", "clarification_request": "needs_assessment", "irrelevant": "needs_assessment", "hangup_signal": "end_call"},
            "value_proposition":  {"interested": "closing", "hesitant": "addressing_hesitancy", "objection": "objection_handling", "closing_signal": "closing", "clarification_request": "value_proposition", "irrelevant": "value_proposition", "hangup_signal": "end_call"},
            "addressing_hesitancy": {"interested": "value_proposition", "hesitant": "addressing_hesitancy", "objection": "objection_handling", "closing_signal": "closing", "clarification_request": "addressing_hesitancy", "irrelevant": "addressing_hesitancy", "hangup_signal": "end_call"}, # New state
            "objection_handling": {"interested": "closing", "hesitant": "addressing_hesitancy", "objection": "objection_handling", "closing_signal": "closing", "clarification_request": "objection_handling", "irrelevant": "objection_handling", "hangup_signal": "end_call"},
            "closing":            {"interested": "finalizing", "hesitant": "objection_handling", "objection": "objection_handling", "closing_signal": "finalizing", "clarification_request": "closing", "irrelevant": "closing", "hangup_signal": "end_call"},
            "finalizing":         {"interested": "end_call", "hesitant": "objection_handling", "objection": "objection_handling", "closing_signal": "end_call", "clarification_request": "finalizing", "irrelevant": "finalizing", "hangup_signal": "end_call"}, # New state
            "end_call":           {} # Terminal state
        }

        next_state = current_state
        if current_state == "end_call": return current_state

        confidence_threshold = self.internal_state['intent_confidence_threshold']
        if confidence < confidence_threshold:
            self.logger.warning(f"Low confidence ({confidence:.2f} < {confidence_threshold}) for intent '{intent}'. Remaining in state '{current_state}'.")
            return current_state # Stay in state, maybe ask clarification in response generation

        # Negative tone can push towards addressing concerns
        if emotional_tone == "negative" and intent not in ["closing_signal", "hangup_signal"] and current_state not in ["objection_handling", "addressing_hesitancy"]:
            self.logger.info(f"Negative tone detected. Overriding transition to 'addressing_hesitancy' from '{current_state}'.")
            return "addressing_hesitancy"

        next_state = state_transitions.get(current_state, {}).get(intent, current_state)
        return next_state


    async def generate_agent_response(self, state: str, client: Client, conversation_log: List[Dict]) -> str:
        """Craft a tailored response using internal logic and a single LLM call."""
        self.logger.debug(f"Generating agent response for state: {state}, client: {client.id}")
        agent_response = ""

        try:
            # Fetch relevant KB fragments (e.g., successful phrases for this state)
            kb_insights = []
            if self.kb_interface:
                 # Example: Get successful phrases for the current state
                 # fragments = await self.kb_interface.get_knowledge(
                 #     data_types=['voice_response_phrase'], tags=[state, 'success'], limit=3
                 # )
                 # if fragments: kb_insights = [f.content for f in fragments]
                 pass # Placeholder for actual KB query

            task_context = {
                "task": "Generate agent response for sales call",
                "current_call_state": state,
                "client_info": {"name": client.name, "country": client.country, "timezone": client.timezone},
                "conversation_history": conversation_log[-5:], # Recent history
                "payment_terms_snippet": self.internal_state.get('payment_terms'),
                "successful_phrase_examples": kb_insights, # Add KB insights
                "desired_output_format": "Natural, conversational plain text response suitable for Text-to-Speech. Be concise but value-driven."
            }
            response_prompt = await self.generate_dynamic_prompt(task_context)

            # Use orchestrator's LLM call method
            llm_response_str = await self.orchestrator._call_llm_with_retry(
                response_prompt, agent_name=self.AGENT_NAME,
                model=self.internal_state.get('openrouter_response_model'),
                temperature=0.65, max_tokens=250,
                timeout=self.internal_state.get('openrouter_response_timeout')
            )

            if not llm_response_str: raise Exception("LLM call for agent response returned empty.")

            agent_response = llm_response_str.strip()
            if not agent_response: raise ValueError("Empty response generated")

            self.logger.debug(f"Generated response for state '{state}': '{agent_response[:100]}...'")

        except Exception as e:
            self.logger.error(f"Error during generate_agent_response: {e}", exc_info=True)
            # Provide state-specific fallbacks
            sender_name = getattr(self.config, 'SENDER_NAME', 'Alex')
            client_first_name = client.name.split()[0] if client.name else 'there'
            fallback_responses = {
                "greeting": f"Hi {client_first_name}, this is {sender_name} calling from UGC Genius. How's your day going?",
                "needs_assessment": "To make sure I'm not wasting your time, could you share a bit about how you're currently handling content creation?",
                "value_proposition": "Based on that, our service could really help by providing [Specific Benefit 1] and [Specific Benefit 2]. How does that sound?",
                "addressing_hesitancy": "I hear some hesitation. Could you tell me what's holding you back at the moment?",
                "objection_handling": "I understand that concern. Many clients felt that way initially, but found that [Counter/Value Prop]. Does that help clarify?",
                "closing": "It sounds like this could be a great fit. Are you open to discussing the next steps to get started?",
                "finalizing": "Excellent. To get started, I just need to confirm the details and then I can send over the invoice and agreement.",
                "end_call": "Okay, thank you for your time today. Have a great day. Goodbye."
            }
            agent_response = fallback_responses.get(state, "I apologize, I seem to be having a slight technical difficulty. Could you repeat that?")

        return agent_response


    async def speak_response(self, text: str, call_sid: str):
        """Convert text to speech using Deepgram Aura TTS and deliver via Twilio."""
        if not text:
             self.logger.warning(f"Attempted to speak empty text for call {call_sid}.")
             return

        self.logger.debug(f"Synthesizing speech for call {call_sid}: '{text[:100]}...'")
        try:
            # 1. Synthesize speech with Deepgram Aura
            # Use SpeakOptions for clarity if using newer SDK versions
            options = SpeakOptions(
                 model=self.internal_state['aura_voice'],
                 encoding="mulaw", # For Twilio PSTN
                 container="wav", # Standard container
                 sample_rate=8000 # For Twilio PSTN
                 )
            # Use speak_stream to get bytes directly
            # Ensure deepgram_client is initialized correctly
            speak_client: SpeakClient = self.deepgram_client.speak # Get speak client instance
            # The speak_stream method returns an async context manager
            async with speak_client.stream({"text": text}, options) as streamer:
                 # Read all data from the stream
                 audio_data = await streamer.read()
                 if not audio_data:
                      raise DeepgramError("Failed to get audio data from Deepgram stream.")

            # 2. Host audio data temporarily via Orchestrator
            if hasattr(self.orchestrator, 'host_temporary_audio'):
                 audio_url = await self.orchestrator.host_temporary_audio(audio_data, f"{call_sid}_{int(time.time())}.wav")
                 if not audio_url: raise RuntimeError("Failed to host temporary audio for Twilio via Orchestrator")
            else:
                 raise NotImplementedError("Audio hosting mechanism not available via Orchestrator.")

            self.logger.debug(f"Synthesized audio URL for {call_sid}: {audio_url}")

            # 3. Update Twilio call to play the audio
            twiml = f"<Response><Play>{audio_url}</Play></Response>"
            await asyncio.to_thread(self.twilio_client.calls(call_sid).update, twiml=twiml)
            self.logger.info(f"Instructed Twilio to speak response for call {call_sid}")

        except DeepgramError as e:
            self.logger.error(f"Deepgram TTS synthesis failed for call {call_sid}: {e}", exc_info=True)
            raise e
        except TwilioException as e:
            self.logger.error(f"Twilio call update failed for call {call_sid}: {e}", exc_info=True)
            raise e
        except Exception as e:
            self.logger.error(f"Unexpected error in speak_response for call {call_sid}: {e}", exc_info=True)
            raise e


    async def _receive_deepgram_transcription(self, ws: websockets.client.WebSocketClientProtocol) -> Optional[Dict]:
        """Receives and processes a single message from the Deepgram WebSocket."""
        try:
            message_json = await ws.recv()
            message_data = json.loads(message_json)
            msg_type = message_data.get('type')

            if msg_type == 'Results':
                if (message_data.get('channel') and message_data['channel'].get('alternatives')):
                    alt = message_data['channel']['alternatives'][0]
                    transcript = alt.get('transcript', '').strip()
                    confidence = alt.get('confidence', 0.0)
                    is_final = message_data.get('is_final', False) # Check if final

                    # Only return final, non-empty transcripts
                    if transcript and is_final:
                        return {"transcript": transcript, "is_final": is_final, "confidence": confidence}
                    # else: logger.debug(f"Discarding non-final or empty transcript: '{transcript}'")

            elif msg_type == 'Metadata': logger.info(f"Received Deepgram metadata: {message_data}")
            elif msg_type == 'SpeechStarted': logger.debug("Deepgram detected speech start.")
            elif msg_type == 'UtteranceEnd': logger.debug("Deepgram detected utterance end.")
            # Handle other types like Error if needed

        except websockets.exceptions.ConnectionClosedOK: logger.info("Deepgram WebSocket connection closed normally during recv."); return None
        except websockets.exceptions.ConnectionClosedError as e: logger.error(f"Deepgram WebSocket connection closed with error during recv: {e}"); return None
        except json.JSONDecodeError as e: logger.error(f"Failed to decode JSON from Deepgram: {message_json}. Error: {e}")
        except Exception as e: logger.error(f"Error receiving or processing Deepgram WebSocket message: {e}", exc_info=True); return None

        return None # Return None if no valid final transcript received


    async def store_call_log(self, client: Client, conversation_log: List[Dict], outcome: str, call_sid: str):
        """Persist final call logs with metadata from Twilio."""
        self.logger.debug(f"Storing call log for {call_sid}. Outcome: {outcome}")
        duration_seconds: Optional[int] = None
        recording_url: Optional[str] = None
        final_twilio_status: Optional[str] = None

        try:
            call_record = await asyncio.to_thread(self.twilio_client.calls(call_sid).fetch)
            duration_str = getattr(call_record, 'duration', None)
            if duration_str: duration_seconds = int(duration_str)
            final_twilio_status = getattr(call_record, 'status', None)

            recordings = await asyncio.to_thread(call_record.recordings.list)
            if recordings:
                 media_url_path = getattr(recordings[0], 'uri', None)
                 if media_url_path and media_url_path.endswith('.json'): media_url_path = media_url_path[:-5]
                 if media_url_path: recording_url = f"https://api.twilio.com{media_url_path}"
                 self.logger.info(f"Found recording for call {call_sid}: {recording_url}")

        except TwilioException as e: logger.warning(f"Could not fetch final Twilio call details/recording for {call_sid}: {e}")
        except Exception as e: logger.error(f"Unexpected error fetching Twilio call details for {call_sid}: {e}", exc_info=True)

        try:
            async with self.session_maker() as session:
                log_entry = CallLog(
                    client_id=client.id, call_sid=call_sid,
                    phone_number=getattr(client, 'phone', 'Unknown'),
                    timestamp=datetime.now(timezone.utc),
                    conversation=json.dumps(conversation_log), outcome=outcome,
                    duration_seconds=duration_seconds, recording_url=recording_url,
                    # final_twilio_status=final_twilio_status # Add if column exists
                )
                session.add(log_entry)
                await session.commit()
                self.logger.info(f"Stored call log for {client.id} (Call SID: {call_sid})")
        except sqlalchemy.exc.SQLAlchemyError as db_err:
             self.logger.error(f"Failed to store call log for {call_sid} in DB: {db_err}", exc_info=True)
             raise db_err
        except Exception as e:
             self.logger.error(f"Unexpected error storing call log for {call_sid}: {e}", exc_info=True)
             raise e


    async def store_conversation_state(self, call_sid: str, state: str, conversation_log: List[Dict]):
        """Persist intermediate conversation state for resilience."""
        self.logger.debug(f"Storing state '{state}' for call {call_sid}")
        try:
            async with self.session_maker() as session:
                state_record = ConversationState(
                    call_sid=call_sid, state=state,
                    conversation_log=json.dumps(conversation_log),
                    last_updated=datetime.now(timezone.utc)
                )
                await session.merge(state_record) # Use merge for upsert
                await session.commit()
        except sqlalchemy.exc.SQLAlchemyError as db_err:
             self.logger.error(f"Failed to store conversation state for {call_sid} in DB: {db_err}", exc_info=True)
             raise db_err
        except Exception as e:
             self.logger.error(f"Unexpected error storing conversation state for {call_sid}: {e}", exc_info=True)
             raise e

    async def get_insights(self) -> Dict[str, Any]:
        """Provide agent-specific insights (e.g., recent success rate)."""
        success_rate = 0.0
        avg_duration = 0.0
        try:
            async with self.session_maker() as session:
                threshold = datetime.now(timezone.utc) - timedelta(hours=24)
                stmt = select(
                    func.count(CallLog.id).label('total_calls'),
                    func.sum(case((CallLog.outcome == 'success', 1), else_=0)).label('successful_calls'),
                    func.avg(CallLog.duration_seconds).label('average_duration')
                ).where(CallLog.timestamp >= threshold)
                result = await session.execute(stmt)
                stats = result.mappings().first()
                if stats and stats['total_calls'] > 0:
                    success_rate = (stats['successful_calls'] / stats['total_calls'])
                    avg_duration = stats['average_duration'] or 0.0
        except Exception as e:
             self.logger.error(f"Failed to calculate call insights: {e}", exc_info=True)

        return {
            "agent_name": self.AGENT_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_calls": len(self.internal_state.get('active_calls', {})),
            "recent_success_rate_24h": round(success_rate, 3),
            "average_call_duration_24h_sec": round(avg_duration, 1)
            }

    # --- Abstract Method Implementations ---

    async def execute_task(self, task_details: Dict[str, Any]) -> Any:
        """Handles tasks like initiating or handling calls."""
        self.logger.info(f"VoiceSalesAgent execute_task received task: {task_details}")
        self.status = "working"
        result = {"status": "failure", "message": "Unknown voice task action."}
        task_type = task_details.get('type')

        try:
            if task_type == 'initiate_outbound_call':
                client_id = task_details.get('client_id')
                if not client_id: raise ValueError("Missing client_id for initiate_outbound_call")
                async with self.session_maker() as session:
                    client = await session.get(Client, client_id)
                if not client: raise ValueError(f"Client not found: {client_id}")
                call_sid = await self.initiate_outbound_call(client)
                if call_sid:
                    # Start handle_call in background
                    asyncio.create_task(self.handle_call(call_sid, client))
                    result = {"status": "success", "message": f"Outbound call initiated.", "call_sid": call_sid}
                else:
                    result["message"] = "Failed to initiate outbound call."
            elif task_type == 'handle_incoming_call': # Assuming webhook triggers this
                call_sid = task_details.get('call_sid')
                # Need to look up client based on incoming number or other data
                # client = await self._find_client_for_call(call_sid) # Needs implementation
                client = None # Placeholder
                if call_sid and client:
                     asyncio.create_task(self.handle_call(call_sid, client))
                     result = {"status": "success", "message": "Handling incoming call."}
                else:
                     result["message"] = "Could not handle incoming call (missing SID or client lookup failed)."
            else:
                self.logger.warning(f"Unsupported task type for VoiceSalesAgent: {task_type}")
                result["message"] = f"Unsupported task type: {task_type}"

        except Exception as e:
             self.logger.error(f"Error in execute_task for action '{task_type}': {e}", exc_info=True)
             result["message"] = f"Unexpected error: {e}"
             result["status"] = "error"

        self.status = "idle"
        return result

    async def learning_loop(self):
        """Analyzes call logs to refine strategies (e.g., best closing phrases)."""
        while True:
            try:
                await asyncio.sleep(3600 * 2) # Run every 2 hours
                self.logger.info("Executing VoiceAgent learning loop...")

                # --- Analyze successful closing phrases ---
                async with self.session_maker() as session:
                    threshold = datetime.now(timezone.utc) - timedelta(days=7) # Analyze last 7 days
                    stmt = select(CallLog.conversation).where(
                        CallLog.timestamp >= threshold,
                        CallLog.outcome == 'success', # Focus on successful calls
                        CallLog.conversation.isnot(None)
                    ).limit(500) # Limit data processed
                    results = await session.execute(stmt)
                    call_logs_json = [row.conversation for row in results.mappings().all()]

                successful_closing_phrases = Counter()
                for log_json in call_logs_json:
                    try:
                        log_list = json.loads(log_json)
                        # Find last agent utterance before outcome=success (might need refinement)
                        agent_turns = [turn['text'] for turn in log_list if turn.get('role') == 'agent']
                        if agent_turns: successful_closing_phrases[agent_turns[-1]] += 1
                    except (json.JSONDecodeError, TypeError): continue

                best_closing_phrase = self.internal_state.get('preferred_closing_phrase', "Suggest next steps.")
                if successful_closing_phrases:
                    most_common = successful_closing_phrases.most_common(1)
                    if most_common and most_common[0][1] > 2: # Require at least 3 occurrences
                        best_closing_phrase = most_common[0][0]
                        self.logger.info(f"Learning Loop: Most common successful closing phrase: '{best_closing_phrase}' (Count: {most_common[0][1]})")

                # --- Update Internal State ---
                if self.internal_state.get('preferred_closing_phrase') != best_closing_phrase:
                    self.internal_state['preferred_closing_phrase'] = best_closing_phrase
                    self.internal_state['last_learning_update_ts'] = datetime.now(timezone.utc)
                    self.logger.info(f"Internal state updated with new preferred closing phrase: '{best_closing_phrase}'")
                    # Log pattern
                    await self.log_learned_pattern(
                        pattern_description=f"Closing phrase '{best_closing_phrase}' correlates with successful calls.",
                        supporting_fragment_ids=[], # TODO: Link to CallLog IDs if possible
                        confidence_score=0.7, # Moderate confidence based on frequency
                        implications="Prioritize using this closing phrase.",
                        tags=["voice_sales", "closing_phrase", "performance_optimized"]
                    )
                else: self.logger.info("No change in preferred closing phrase.")

                # TODO: Add analysis for objection handling, opening lines, etc.

            except asyncio.CancelledError: self.logger.info("VoiceAgent learning loop cancelled."); break
            except Exception as e: self.logger.error(f"Error during VoiceAgent learning loop: {e}", exc_info=True)

    async def self_critique(self) -> Dict[str, Any]:
        """Evaluates call performance and strategy effectiveness."""
        self.logger.info("VoiceAgent: Performing self-critique.")
        insights = await self.get_insights() # Use existing insight method
        critique = {"status": "ok", "feedback": f"Critique based on recent insights: {insights}"}
        # Add more detailed critique based on thresholds or specific patterns
        if insights.get('recent_success_rate_24h', 0) < 0.1: # Example threshold
            critique['feedback'] += " Low success rate requires investigation into script/strategy/lead quality."
        return critique

    async def generate_dynamic_prompt(self, task_context: Dict[str, Any]) -> str:
        """Constructs context-rich prompts for LLM calls."""
        self.logger.debug(f"Generating dynamic prompt for VoiceAgent task: {task_context.get('task')}")
        prompt_parts = [self.meta_prompt] # Start with meta prompt

        prompt_parts.append("\n--- Current Task Context ---")
        prompt_parts.append(f"Task: {task_context.get('task')}")
        if task_context.get('current_call_state'): prompt_parts.append(f"Current Call State: {task_context['current_call_state']}")
        if task_context.get('client_response'): prompt_parts.append(f"Last Client Response: {task_context['client_response']}")

        client_info = task_context.get('client_info', {})
        if client_info:
            prompt_parts.append("\n--- Client Profile ---")
            prompt_parts.append(f"Name: {client_info.get('name', 'N/A')}")
            prompt_parts.append(f"Country: {client_info.get('country', 'N/A')}")
            # Add other relevant client details if available

        history = task_context.get('conversation_history', [])
        if history:
            prompt_parts.append("\n--- Recent Conversation History ---")
            for turn in history:
                prompt_parts.append(f"{turn.get('role', 'unknown').capitalize()}: {turn.get('text', '')}")

        # Add learned preferences or KB insights
        prompt_parts.append("\n--- Strategic Guidance ---")
        if task_context.get('task') == 'Generate agent response' and self.internal_state.get('preferred_closing_phrase') and task_context.get('current_call_state') == 'closing':
             prompt_parts.append(f"Consider using this effective closing phrase: '{self.internal_state['preferred_closing_phrase']}'")
        # Add more KB insights based on state/context
        # prompt_parts.append(f"KB Insight Example: Clients in {client_info.get('country')} often respond well to [Specific Approach].")

        prompt_parts.append("\n--- Instructions ---")
        if task_context.get('task') == 'Interpret client intent':
            prompt_parts.append(f"Analyze the 'Last Client Response'. Determine the primary 'intent' from {task_context.get('possible_intents', [])}, estimate 'confidence' (0.0-1.0), list any 'sub_intents', and assess 'emotional_tone' (positive/negative/neutral).")
            prompt_parts.append(f"Output ONLY JSON: {task_context.get('desired_output_format')}")
        elif task_context.get('task') == 'Generate agent response':
            prompt_parts.append(f"Generate the next agent response based on the 'Current Call State' ({task_context.get('current_call_state', 'unknown')}), conversation history, and client profile.")
            prompt_parts.append("Adhere to the Core Principles (Hormozi inspired): Focus on value, handle objections gracefully, guide towards a close or clear next step.")
            prompt_parts.append("Keep the response concise, natural, and suitable for the Aura TTS voice.")
            prompt_parts.append(f"Output ONLY the response text: {task_context.get('desired_output_format')}")
        else:
            prompt_parts.append("Analyze the provided context and generate the required output based on the task description.")

        final_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated dynamic prompt for VoiceSalesAgent (length: {len(final_prompt)} chars)")
        return final_prompt

    # --- Helper to log learned patterns (if KBInterface is not directly used) ---
    async def log_learned_pattern(self, pattern_description: str, supporting_fragment_ids: List[int], confidence_score: float, implications: str, tags: Optional[List[str]] = None):
        """Logs a learned pattern, potentially via ThinkTool or directly to DB."""
        if self.kb_interface and hasattr(self.kb_interface, 'log_learned_pattern'):
            await self.kb_interface.log_learned_pattern(
                pattern_description=pattern_description, supporting_fragment_ids=supporting_fragment_ids,
                confidence_score=confidence_score, implications=implications, tags=tags
            )
        else:
            self.logger.info(f"Learned Pattern (Not logged to KB): Desc='{pattern_description}', Conf={confidence_score:.2f}, Impl='{implications}'")


