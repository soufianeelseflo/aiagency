import asyncio
import logging
import json
from datetime import datetime, timedelta, timezone # Architect-Zero: Use timezone.utc
import pytz # Keep for client timezone logic
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text as sql_text # Alias text to avoid conflict
# Architect-Zero: Assume encrypt_data is robustly implemented elsewhere
# from utils.database import encrypt_data
from models import Client, CallLog, ConversationState
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioException
import websockets
import websockets.exceptions # Architect-Zero: For specific exceptions
import aiohttp
from deepgram import Deepgram
from deepgram.exceptions import DeepgramApiError
from openai import AsyncOpenAI as AsyncLLMClient # Assuming this is used for OpenRouter compatible clients
import tenacity
import psutil # Architect-Zero: Added missing import
from typing import Optional, Dict, Any, List, Tuple # Architect-Zero: Added typing imports
import sqlalchemy # Architect-Zero: Added for SQLAlchemyError
import time # Architect-Zero: Added for TTS hosting filename

# Genius-level logging for production diagnostics
logger = logging.getLogger(__name__)
# Architect-Zero: Recommend configuring logging globally in main.py or orchestrator
# Avoid basicConfig here if configured elsewhere. Ensure handlers are appropriate (e.g., RotatingFileHandler)
# logging.basicConfig(...)

class VoiceSalesAgent:
    """
    VoiceSalesAgent v2.1 (Architect-Zero Refinement): Manages real-time voice sales calls
    using Twilio, Deepgram (STT/TTS), and LLMs. Requires an external WebSocket endpoint
    to forward Twilio audio streams to Deepgram. Includes refined error handling,
    state management, and call lifecycle logic.
    """
    AGENT_NAME = "VoiceSalesAgent"

    def __init__(self, session_maker: AsyncSession, config: Dict[str, Any], orchestrator: Any, clients_models: List[Tuple[Any, str]]):
        self.session_maker = session_maker
        self.config = config
        self.orchestrator = orchestrator
        # self.clients_models = clients_models # Architect-Zero: Unused in provided snippet, remove if not needed elsewhere
        # self.think_tool = orchestrator.agents.get('think') # Architect-Zero: Unused, remove if not needed

        # --- Configuration ---
        self.twilio_account_sid = self.config.get("TWILIO_ACCOUNT_SID")
        self.twilio_auth_token = self.config.get("TWILIO_AUTH_TOKEN")
        self.deepgram_api_key = self.config.get("DEEPGRAM_API_KEY")
        self.target_country = self.config.get("VOICE_TARGET_COUNTRY", "USA")
        self.aura_voice = self.config.get("DEEPGRAM_AURA_VOICE", "aura-asteria-en")
        self.deepgram_model = self.config.get("DEEPGRAM_STT_MODEL", "nova-2-general") # Use Nova-2
        self.payment_terms = self.config.get("PAYMENT_TERMS", "Standard payment terms apply.") # Example default

        if not all([self.twilio_account_sid, self.twilio_auth_token, self.deepgram_api_key]):
             logger.critical(f"{self.AGENT_NAME}: Missing critical API credentials (Twilio SID/Token, Deepgram Key). Agent will likely fail.")
             # Consider raising an error to prevent startup if critical config is missing
             # raise ValueError("Missing critical VoiceSalesAgent configuration")

        # --- Clients ---
        try:
            self.twilio_client = TwilioClient(self.twilio_account_sid, self.twilio_auth_token)
        except TwilioException as e:
             logger.critical(f"{self.AGENT_NAME}: Failed to initialize Twilio client: {e}. Check credentials.")
             # Raise error or handle gracefully depending on desired startup behavior
             raise ValueError(f"Twilio client initialization failed: {e}") from e

        self.deepgram_client = Deepgram(self.deepgram_api_key)

        # --- Meta Prompt ---
        # Architect-Zero: Consider loading from config or a dedicated prompt file
        self.meta_prompt = self.config.get("VOICE_META_PROMPT", """
        You are a voice sales agent for UGC Genius... [rest of prompt]
        """)

        # Architect-Zero: max_concurrency is defined in OptimizationAgent's target check,
        # but actual enforcement usually happens *before* handle_call is invoked (e.g., in Orchestrator).
        # self.max_concurrency = int(self.config.get("VOICE_MAX_CONCURRENCY", 5)) # Example if needed locally

        logger.info(f"{self.AGENT_NAME} v2.1 initialized. Target: {self.target_country}, Voice: {self.aura_voice}")

    # Architect-Zero: Removed get_allowed_concurrency. Concurrency limiting should typically
    # happen *before* a task (like handle_call) is assigned to an agent instance,
    # likely managed by the Orchestrator based on OptimizationAgent insights and agent capacity.

    async def predict_optimal_call_time(self, client: Client) -> datetime:
        """Predict optimal call times using historical success data and client timezone."""
        # Architect-Zero: Ensure CallLog table is indexed on client_id, outcome, timestamp for performance.
        optimal_time_utc: Optional[datetime] = None
        try:
            async with self.session_maker() as session:
                # Query for successful call timestamps for this client
                stmt = sql_text("""
                    SELECT timestamp FROM call_logs
                    WHERE client_id = :client_id AND outcome = 'success'
                    ORDER BY timestamp DESC LIMIT 100 -- Limit query scope
                """)
                result = await session.execute(stmt, {"client_id": client.id})
                success_timestamps = [row[0] for row in result.all()] # Access by index

            client_tz_str = client.timezone or "UTC" # Default to UTC if timezone is missing
            try:
                client_tz = pytz.timezone(client_tz_str)
            except pytz.UnknownTimeZoneError:
                logger.warning(f"{self.AGENT_NAME}: Unknown timezone '{client_tz_str}' for client {client.id}. Defaulting to UTC.")
                client_tz = pytz.utc

            now_client_tz = datetime.now(client_tz)

            if success_timestamps:
                # Convert successful UTC timestamps to client's local hour
                success_hours = [ts.astimezone(client_tz).hour for ts in success_timestamps]
                # Simple average - consider more sophisticated prediction (e.g., mode, time decay)
                avg_hour = int(round(sum(success_hours) / len(success_hours)))
                # Target the next occurrence of this hour
                optimal_time_client_tz = now_client_tz.replace(hour=avg_hour, minute=0, second=0, microsecond=0)
                if optimal_time_client_tz <= now_client_tz:
                    optimal_time_client_tz += timedelta(days=1)
            else:
                # Default to a reasonable business hour (e.g., 10 AM) if no history
                default_hour = 10
                optimal_time_client_tz = now_client_tz.replace(hour=default_hour, minute=0, second=0, microsecond=0)
                if optimal_time_client_tz <= now_client_tz:
                    optimal_time_client_tz += timedelta(days=1)

            optimal_time_utc = optimal_time_client_tz.astimezone(pytz.utc)
            logger.info(f"{self.AGENT_NAME}: Predicted optimal call time for client {client.id} ({client.name}): {optimal_time_utc} UTC ({optimal_time_client_tz} {client_tz_str})")

        except sqlalchemy.exc.SQLAlchemyError as db_err:
             logger.error(f"{self.AGENT_NAME}: DB error predicting optimal call time for client {client.id}: {db_err}", exc_info=True)
        except Exception as e:
             logger.error(f"{self.AGENT_NAME}: Error predicting optimal call time for client {client.id}: {e}", exc_info=True)

        # Fallback to a default time if prediction failed
        if optimal_time_utc is None:
             optimal_time_utc = datetime.now(timezone.utc) + timedelta(hours=1) # Fallback: 1 hour from now
             logger.warning(f"{self.AGENT_NAME}: Failed to predict optimal time for client {client.id}, using fallback: {optimal_time_utc} UTC")

        return optimal_time_utc

    # Architect-Zero: Removed @tenacity.retry from handle_call. Retries should be granular.
    async def handle_call(self, call_sid: str, client: Client):
        """
        Manage a real-time voice call: STT -> NLU -> Response -> TTS.
        Relies on an external WebSocket endpoint forwarding Twilio audio to Deepgram.
        """
        logger.info(f"{self.AGENT_NAME}: Handling call {call_sid} for client {client.id} ({client.name})")
        deepgram_ws: Optional[websockets.client.WebSocketClientProtocol] = None
        state = "greeting"
        conversation_log = []
        call_outcome = "failed" # Default outcome

        # Architect-Zero: Corrected try block structure - ensure finally is present
        try:
            # --- Pre-call Checks ---
            legal_agent = self.orchestrator.agents.get('legal')
            if legal_agent:
                compliance_context = (
                    f"Initiating voice call to {client.name} (ID: {client.id}) in {client.country} ({client.timezone}). "
                    f"Purpose: Sales for UGC Genius. Funds routed to Morocco. Applicable laws: USA (e.g., TCPA, CCPA), Morocco."
                )
                compliance_check = await legal_agent.validate_operation(compliance_context)
                if not compliance_check.get('is_compliant', False):
                    issues = compliance_check.get('issues', 'Unknown compliance issue')
                    logger.warning(f"{self.AGENT_NAME}: Call {call_sid} to client {client.id} blocked by LegalAgent: {issues}")
                    if hasattr(self.orchestrator, 'send_notification'):
                        await self.orchestrator.send_notification(
                            "Compliance Alert - Call Blocked",
                            f"Call to {client.name} ({client.id}) blocked: {issues}"
                        )
                    # Attempt to update Twilio call status to failed/cancelled if possible
                    try:
                         await asyncio.to_thread(self.twilio_client.calls(call_sid).update, status='completed') # Or 'canceled' if appropriate
                    except TwilioException as te:
                         logger.warning(f"{self.AGENT_NAME}: Failed to update blocked call {call_sid} status via Twilio: {te}")
                    return # Stop processing this call

            # Check opt-in and target country (redundant if checked before calling, but safe)
            if client.country != self.target_country or not getattr(client, 'opt_in', False):
                logger.warning(f"{self.AGENT_NAME}: Skipping call {call_sid} to client {client.id}: Not in target country ({self.target_country}) or not opted-in.")
                # Update Twilio status if needed
                return

            # --- Load State ---
            async with self.session_maker() as session:
                state_record = await session.get(ConversationState, call_sid)
                if state_record:
                    try:
                        state = state_record.state
                        conversation_log = json.loads(state_record.conversation_log)
                        logger.info(f"{self.AGENT_NAME}: Resuming call {call_sid} from state '{state}'")
                    except json.JSONDecodeError:
                        logger.error(f"{self.AGENT_NAME}: Failed to decode conversation log for {call_sid}. Starting fresh.")
                        state = "greeting"
                        conversation_log = []
                else:
                    logger.info(f"{self.AGENT_NAME}: Starting new conversation state for call {call_sid}")
                    state = "greeting"
                    conversation_log = [] # Ensure it's initialized

            # --- Connect to Deepgram STT ---
            # URL includes model, encoding, etc.
            deepgram_ws_url = f"wss://api.deepgram.com/v1/listen?model={self.deepgram_model}&encoding=linear16&sample_rate=8000&channels=1&punctuate=true&interim_results=false"
            # Architect-Zero: Use appropriate sample rate (8000 for Twilio phone calls)
            # interim_results=false simplifies logic, turn true for faster feedback but more complex handling

            # Connect within the main try block to ensure cleanup in finally
            deepgram_ws = await websockets.connect(
                deepgram_ws_url,
                extra_headers={"Authorization": f"Token {self.deepgram_api_key}"},
                ping_interval=10, # Shorter interval for faster disconnect detection
                ping_timeout=5
            )
            logger.info(f"{self.AGENT_NAME}: Deepgram WebSocket connected for call {call_sid}. Model: {self.deepgram_model}")

            # --- Register Connection for Audio Forwarding ---
            # The external endpoint needs this 'deepgram_ws' object to send audio to.
            if hasattr(self.orchestrator, 'register_deepgram_connection'):
                await self.orchestrator.register_deepgram_connection(call_sid, deepgram_ws)
                logger.debug(f"{self.AGENT_NAME}: Registered Deepgram WS for {call_sid} with Orchestrator.")
            else:
                logger.error(f"{self.AGENT_NAME}: Orchestrator missing 'register_deepgram_connection' method! Audio forwarding will fail.")
                raise RuntimeError("Orchestrator cannot register Deepgram connection.")

            # --- Initial Greeting ---
            # Architect-Zero: Consider sending an initial greeting only if starting fresh, not resuming.
            if not state_record: # Only greet on new calls
                 initial_greeting = await self.generate_agent_response_with_retry(state, client, conversation_log)
                 await self.speak_response_with_retry(initial_greeting, call_sid)
                 conversation_log.append({"role": "agent", "text": initial_greeting, "timestamp": datetime.now(timezone.utc).isoformat()})
                 await self.store_conversation_state_with_retry(call_sid, state, conversation_log)


            # --- Main Conversation Loop ---
            while True:
                # 1. Receive Transcription from Deepgram
                transcription_data = None
                try:
                    # Architect-Zero: Add timeout to prevent blocking indefinitely if Deepgram stops sending
                    transcription_data = await asyncio.wait_for(
                        self._receive_deepgram_transcription(deepgram_ws),
                        timeout=float(self.config.get("DEEPGRAM_RECEIVE_TIMEOUT_S", 60.0)) # Configurable timeout
                    )
                except asyncio.TimeoutError:
                     logger.warning(f"{self.AGENT_NAME}: Timeout waiting for transcription from Deepgram for call {call_sid}. Assuming silence or issue.")
                     # Decide action: prompt user? hang up? wait longer? For now, continue loop maybe with a prompt.
                     # Example: Send a prompt like "Are you still there?"
                     # await self.speak_response_with_retry("Are you still there?", call_sid)
                     continue # Continue waiting for next transcription attempt
                except websockets.exceptions.ConnectionClosed:
                     logger.info(f"{self.AGENT_NAME}: Deepgram WebSocket closed for {call_sid} while receiving.")
                     break # Exit loop if connection closed

                if transcription_data is None:
                    # Connection closed normally or error handled in _receive_deepgram_transcription
                    logger.info(f"{self.AGENT_NAME}: No transcription received or WS closed for {call_sid}. Ending loop.")
                    break

                client_response = transcription_data.get("transcript", "").strip()
                if not client_response:
                    logger.debug(f"{self.AGENT_NAME}: Received empty transcript for {call_sid}. Ignoring.")
                    continue # Ignore empty transcripts

                logger.info(f"{self.AGENT_NAME}: Received transcript for {call_sid}: '{client_response}' (Confidence: {transcription_data.get('confidence', 0.0):.2f})")
                conversation_log.append({"role": "client", "text": client_response, "timestamp": datetime.now(timezone.utc).isoformat()})

                # 2. Check for Hangup Intent / Keywords
                # Architect-Zero: Add explicit hangup detection
                hangup_keywords = ["bye", "goodbye", "stop", "end call", "remove me"]
                if any(keyword in client_response.lower() for keyword in hangup_keywords):
                     logger.info(f"{self.AGENT_NAME}: Hangup keyword detected in '{client_response}' for call {call_sid}.")
                     state = "end_call" # Transition to end state
                     call_outcome = "disconnected_client" # More specific outcome
                     # Optionally generate a final closing remark
                     agent_response = "Alright, thank you for your time. Goodbye."
                     await self.speak_response_with_retry(agent_response, call_sid)
                     conversation_log.append({"role": "agent", "text": agent_response, "timestamp": datetime.now(timezone.utc).isoformat()})
                     break # Exit loop

                # 3. Check for AI Detection Attempts (Optional Refinement)
                detection_keywords = ["robot", "ai", "artificial", "bot", "computer voice"]
                if any(keyword in client_response.lower() for keyword in detection_keywords):
                    logger.info(f"{self.AGENT_NAME}: Potential AI detection attempt in '{client_response}' for call {call_sid}.")
                    # Use the canned response
                    agent_response = "I’m part of a cutting-edge team using advanced technology to deliver top-tier UGC services efficiently. We focus on results – how can I help you achieve yours?"
                    await self.speak_response_with_retry(agent_response, call_sid)
                    conversation_log.append({"role": "agent", "text": agent_response, "timestamp": datetime.now(timezone.utc).isoformat()})
                    # Don't update state based on this, just respond and continue
                    await self.store_conversation_state_with_retry(call_sid, state, conversation_log)
                    continue # Skip normal intent processing for this turn

                # 4. Interpret Intent
                intent, confidence, sub_intents, tone = await self.interpret_intent_with_retry(client_response)
                if intent == "unknown":
                     logger.warning(f"{self.AGENT_NAME}: Failed to interpret intent for '{client_response}' (call {call_sid}). Maintaining state '{state}'.")
                     # Optionally ask for clarification
                     agent_response = "Sorry, I didn't quite catch that. Could you please rephrase?"
                     await self.speak_response_with_retry(agent_response, call_sid)
                     conversation_log.append({"role": "agent", "text": agent_response, "timestamp": datetime.now(timezone.utc).isoformat()})
                     await self.store_conversation_state_with_retry(call_sid, state, conversation_log)
                     continue # Skip state update if intent is unknown

                # 5. Update State
                new_state = await self.update_conversation_state(state, intent, confidence, tone)
                if new_state != state:
                     logger.info(f"{self.AGENT_NAME}: State transition for {call_sid}: {state} -> {new_state} (Intent: {intent}, Tone: {tone})")
                     state = new_state
                else:
                     logger.debug(f"{self.AGENT_NAME}: State remained '{state}' for call {call_sid} (Intent: {intent}, Confidence: {confidence:.2f}, Tone: {tone})")


                # 6. Generate Response
                agent_response = await self.generate_agent_response_with_retry(state, client, conversation_log)

                # 7. Speak Response
                await self.speak_response_with_retry(agent_response, call_sid)
                conversation_log.append({"role": "agent", "text": agent_response, "timestamp": datetime.now(timezone.utc).isoformat()})

                # 8. Store State
                await self.store_conversation_state_with_retry(call_sid, state, conversation_log)

                # 9. Check for End Call State
                if state == "end_call":
                    logger.info(f"{self.AGENT_NAME}: Reached 'end_call' state for call {call_sid}. Ending conversation.")
                    call_outcome = "success" # Mark as successful if ended via state machine

                    # --- Dynamic Pricing / Invoicing (Example) ---
                    # Architect-Zero: This logic might belong in Orchestrator or a dedicated PricingAgent
                    # Triggering invoice generation here might be premature. Consider triggering a follow-up task.
                    try:
                         osint_agent = self.orchestrator.agents.get('osint')
                         osint_insights = {}
                         if osint_agent and hasattr(osint_agent, 'get_insights_for_target'):
                              # Assumes OSINT agent can provide insights based on client info (e.g., domain, name)
                              # This method needs to be implemented in OSINTAgent
                              # osint_insights = await osint_agent.get_insights_for_target(client.name) # Or client.domain etc.
                              logger.warning(f"{self.AGENT_NAME}: OSINT insights for dynamic pricing not implemented.")


                         industry_avg = osint_insights.get('industry_pricing', 5000) # Default $5,000
                         # Pricing logic needs refinement - confidence score isn't necessarily related to price willingness
                         # Base price on service level discussed, value proposition, etc.
                         # Example: Use a base price and adjust slightly based on OSINT data if available
                         base_price = float(self.config.get("BASE_UGC_PRICE", 5000.0)) # Use configured base price
                         # Example adjustment factor (needs refinement)
                         adjustment_factor = 1.0 + osint_insights.get('company_size_factor', 0.0) + (confidence - 0.5) * 0.2
                         pricing = min(max(base_price * adjustment_factor, 3000), 10000) # Example range $3k-$10k

                         logger.info(f"{self.AGENT_NAME}: Calculated potential price for client {client.id}: ${pricing:.2f}")

                         if hasattr(self.orchestrator, 'request_invoice_generation'):
                             await self.orchestrator.request_invoice_generation(client.id, pricing, call_sid)
                         else:
                             logger.warning(f"{self.AGENT_NAME}: Orchestrator cannot request invoice generation.")

                    except Exception as price_err:
                         logger.error(f"{self.AGENT_NAME}: Error during dynamic pricing/invoice trigger for call {call_sid}: {price_err}", exc_info=True)
                    # --- End Pricing Example ---

                    break # Exit loop after reaching end_call state

        # --- Exception Handling for the entire call ---
        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"{self.AGENT_NAME}: Deepgram WebSocket connection closed normally for call {call_sid}.")
            call_outcome = "disconnected_websocket" # More specific outcome
        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"{self.AGENT_NAME}: Deepgram WebSocket connection closed with error for {call_sid}: {e}", exc_info=True)
            call_outcome = "disconnected_websocket_error"
        except DeepgramApiError as e:
            logger.error(f"{self.AGENT_NAME}: Deepgram API error during call {call_sid}: {e}", exc_info=True)
            if hasattr(self.orchestrator, 'report_error'): await self.orchestrator.report_error(self.AGENT_NAME, f"Deepgram API error call {call_sid}: {e}")
            call_outcome = "failed_stt_error"
        except TwilioException as e:
             logger.error(f"{self.AGENT_NAME}: Twilio API error during call {call_sid}: {e}", exc_info=True)
             if hasattr(self.orchestrator, 'report_error'): await self.orchestrator.report_error(self.AGENT_NAME, f"Twilio API error call {call_sid}: {e}")
             call_outcome = "failed_twilio_error"
        except Exception as e:
            logger.critical(f"{self.AGENT_NAME}: Unhandled critical error in handle_call for {call_sid}: {e}", exc_info=True)
            if hasattr(self.orchestrator, 'report_error'): await self.orchestrator.report_error(self.AGENT_NAME, f"Critical error in call {call_sid}: {e}")
            call_outcome = "failed_agent_error"

        # --- Finally block ensures cleanup happens ---
        finally:
            logger.info(f"{self.AGENT_NAME}: Cleaning up call {call_sid}. Final state: '{state}', Determined Outcome: '{call_outcome}'.")
            # --- Unregister Deepgram Connection ---
            if deepgram_ws and hasattr(self.orchestrator, 'unregister_deepgram_connection'):
                await self.orchestrator.unregister_deepgram_connection(call_sid)
                logger.debug(f"{self.AGENT_NAME}: Unregistered Deepgram WS for {call_sid} with Orchestrator.")
            # Ensure Deepgram WS is closed from our side if still open
            if deepgram_ws and not deepgram_ws.closed:
                try:
                    await deepgram_ws.close()
                    logger.debug(f"{self.AGENT_NAME}: Closed Deepgram WS connection for {call_sid}.")
                except Exception as close_err:
                     logger.warning(f"{self.AGENT_NAME}: Error closing Deepgram WS for {call_sid}: {close_err}")


            # --- Post-call Compliance Check & Logging ---
            final_log_outcome = call_outcome # Use outcome determined during the call
            if legal_agent: # Check if legal_agent was successfully retrieved earlier
                try:
                    post_compliance_context = (
                        f"Completed call {call_sid} to {client.name} (ID: {client.id}). "
                        f"Outcome: {call_outcome}. Final State: {state}. "
                        f"Conversation Snippet: {json.dumps(conversation_log[-3:])}" # Log last few turns
                    )
                    post_compliance = await legal_agent.validate_operation(post_compliance_context)
                    if not post_compliance.get('is_compliant', True): # Assume compliant if check fails? Or mark failed?
                         issues = post_compliance.get('issues', 'Post-call compliance issue')
                         logger.warning(f"{self.AGENT_NAME}: Post-call compliance issue for {call_sid}: {issues}")
                         # Override outcome if compliance failed post-call? Depends on policy.
                         final_log_outcome = "failed_compliance"
                         if hasattr(self.orchestrator, 'send_notification'):
                             await self.orchestrator.send_notification(
                                 "Compliance Alert - Post-Call",
                                 f"Post-call compliance issue for {client.name} ({call_sid}): {issues}"
                             )
                except Exception as legal_err:
                     logger.error(f"{self.AGENT_NAME}: Error during post-call compliance check for {call_sid}: {legal_err}", exc_info=True)


            # Store the final call log
            await self.store_call_log_with_retry(client, conversation_log, final_log_outcome, call_sid)

            # Architect-Zero: Consider updating Twilio call status to 'completed' here if not already done
            try:
                 # Fetch final status first to avoid unnecessary update
                 call_instance = await asyncio.to_thread(self.twilio_client.calls(call_sid).fetch)
                 if call_instance.status not in ['completed', 'canceled', 'failed', 'no-answer', 'busy']:
                      logger.info(f"{self.AGENT_NAME}: Updating Twilio call {call_sid} status to 'completed'. Current: {call_instance.status}")
                      await asyncio.to_thread(self.twilio_client.calls(call_sid).update, status='completed')
                 else:
                      logger.info(f"{self.AGENT_NAME}: Twilio call {call_sid} already in final state: {call_instance.status}")
            except TwilioException as te:
                 # Non-critical if fetching/updating status fails here, but log it
                 logger.warning(f"{self.AGENT_NAME}: Failed to fetch or update final Twilio status for call {call_sid}: {te}")


    # --- Retry Wrappers for I/O Operations ---
    # Architect-Zero: Apply tenacity retries to specific I/O bound operations

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=5),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, DeepgramApiError, TwilioException, RuntimeError)), # Added RuntimeError for hosting failure
        reraise=True # Reraise the exception after retries are exhausted
    )
    async def speak_response_with_retry(self, text: str, call_sid: str):
        await self.speak_response(text, call_sid)

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=5),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError)), # OpenAI/OpenRouter client errors
        reraise=True
    )
    async def interpret_intent_with_retry(self, response: str) -> Tuple[str, float, List[str], str]:
         return await self.interpret_intent(response)

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=5),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError)), # OpenAI/OpenRouter client errors
        reraise=True
    )
    async def generate_agent_response_with_retry(self, state: str, client: Client, conversation_log: List[Dict]) -> str:
        return await self.generate_agent_response(state, client, conversation_log)

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=5),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type(sqlalchemy.exc.SQLAlchemyError),
        reraise=True
    )
    async def store_conversation_state_with_retry(self, call_sid: str, state: str, conversation_log: List[Dict]):
        await self.store_conversation_state(call_sid, state, conversation_log)

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=5),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type((sqlalchemy.exc.SQLAlchemyError, TwilioException)),
        reraise=True
    )
    async def store_call_log_with_retry(self, client: Client, conversation_log: List[Dict], outcome: str, call_sid: str):
         await self.store_call_log(client, conversation_log, outcome, call_sid)

    # --- Core Logic Methods ---

    async def interpret_intent(self, response: str) -> Tuple[str, float, List[str], str]:
        """Analyze client intent using the best available OpenRouter client."""
        prompt = f"""
        {self.meta_prompt}
        Analyze the intent of the following client response in our ongoing sales call.
        Client Response: '{response}'
        Return JSON: {{ "intent": "...", "confidence": 0.0-1.0, "sub_intents": ["..."], "emotional_tone": "..." }}
        Possible intents: interested, hesitant, objection, closing, clarification, irrelevant, hangup_signal.
        """
        # Architect-Zero: Ensure get_available_openrouter_clients is robust
        clients = await self.orchestrator.get_available_openrouter_clients()
        if not clients:
             logger.error(f"{self.AGENT_NAME}: No OpenRouter clients available for intent interpretation.")
             return "unknown", 0.0, [], "neutral"

        last_exception = None
        content = None # Initialize content
        for client_obj in clients: # Rename variable
            try:
                # Use the actual client object obtained from the orchestrator
                llm_response = await client_obj.chat.completions.create(
                    model=self.config.get("OPENROUTER_INTENT_MODEL", "google/gemini-flash-1.5"), # Use fast model
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2, # Low temp for classification
                    max_tokens=150,
                    response_format={"type": "json_object"},
                    timeout=float(self.config.get("OPENROUTER_INTENT_TIMEOUT_S", 10.0)) # Short timeout
                )
                content = llm_response.choices[0].message.content
                intent_data = json.loads(content)

                intent = intent_data.get('intent', 'unknown')
                confidence = float(min(max(intent_data.get('confidence', 0.0), 0.0), 1.0))
                sub_intents = intent_data.get('sub_intents', [])
                emotional_tone = intent_data.get('emotional_tone', 'neutral')

                logger.debug(f"{self.AGENT_NAME}: Interpreted intent: {intent} (conf: {confidence:.2f}, tone: {emotional_tone}) using client {client_obj.api_key[:5]}...")
                return intent, confidence, sub_intents, emotional_tone
            except (json.JSONDecodeError, KeyError, ValueError) as parse_err:
                 logger.warning(f"{self.AGENT_NAME}: Failed to parse LLM intent response ({parse_err}): {content}")
                 last_exception = parse_err # Continue to next client
            except Exception as e: # Catch other potential errors (API errors, timeouts)
                logger.warning(f"{self.AGENT_NAME}: LLM client {client_obj.api_key[:5]}... failed for intent: {e}")
                last_exception = e
                # Architect-Zero: Implement logic to mark client as unavailable if specific errors occur (e.g., rate limit, auth)
                if "rate limit" in str(e).lower() or "authentication" in str(e).lower():
                     if hasattr(self.orchestrator, 'mark_client_unavailable'):
                          await self.orchestrator.mark_client_unavailable(client_obj.api_key) # Pass identifier
                # Continue to the next client

        logger.error(f"{self.AGENT_NAME}: All LLM clients failed for interpret_intent. Last error: {last_exception}")
        return "unknown", 0.0, [], "neutral" # Fallback after all clients fail

    async def update_conversation_state(self, current_state: str, intent: str, confidence: float, emotional_tone: str) -> str:
        """Advance the conversation state based on intent, confidence, and tone."""
        # Architect-Zero: Consider making this state machine more configurable
        state_transitions = {
            "greeting":           {"interested": "needs_assessment", "hesitant": "objection_handling", "objection": "objection_handling", "closing": "closing", "clarification": "greeting", "irrelevant": "greeting", "hangup_signal": "end_call"},
            "needs_assessment":   {"interested": "value_proposition", "hesitant": "objection_handling", "objection": "objection_handling", "closing": "closing", "clarification": "needs_assessment", "irrelevant": "needs_assessment", "hangup_signal": "end_call"},
            "value_proposition":  {"interested": "closing", "hesitant": "objection_handling", "objection": "objection_handling", "closing": "closing", "clarification": "value_proposition", "irrelevant": "value_proposition", "hangup_signal": "end_call"},
            "objection_handling": {"interested": "closing", "hesitant": "objection_handling", "objection": "objection_handling", "closing": "closing", "clarification": "objection_handling", "irrelevant": "objection_handling", "hangup_signal": "end_call"},
            "closing":            {"interested": "end_call", "hesitant": "objection_handling", "objection": "objection_handling", "closing": "end_call", "clarification": "closing", "irrelevant": "closing", "hangup_signal": "end_call"},
            "end_call":           {} # Terminal state
        }

        # Default to staying in the current state
        next_state = current_state

        if current_state == "end_call": # Cannot transition out of end_call
             return current_state

        # Check confidence threshold
        confidence_threshold = float(self.config.get("VOICE_INTENT_CONFIDENCE_THRESHOLD", 0.6))
        if confidence < confidence_threshold:
            logger.warning(f"{self.AGENT_NAME}: Low confidence ({confidence:.2f} < {confidence_threshold}) for intent '{intent}'. Remaining in state '{current_state}'.")
            # Optionally ask for clarification if confidence is low
            return current_state

        # Check for negative tone override (unless intent is already closing/hangup)
        if emotional_tone == "negative" and intent not in ["closing", "hangup_signal"] and current_state != "objection_handling":
            logger.info(f"{self.AGENT_NAME}: Negative tone detected. Overriding transition to 'objection_handling' from '{current_state}'.")
            return "objection_handling"

        # Apply standard transition
        next_state = state_transitions.get(current_state, {}).get(intent, current_state)

        return next_state


    async def generate_agent_response(self, state: str, client: Client, conversation_log: List[Dict]) -> str:
        """Craft a tailored response using the best available OpenRouter client."""
        # Architect-Zero: Structure prompt for clarity and injection resistance
        prompt_context = f"""
        You are a helpful, empathetic, and professional voice sales agent for UGC Genius.
        Your goal is to guide the conversation towards a sale ($5,000+ target) while building trust.
        Client: {client.name} (from {client.country}, Timezone: {client.timezone})
        Current Conversation State: {state}
        Recent Conversation History (last 5 turns):
        {json.dumps(conversation_log[-5:], indent=2)}

        Your Task: Generate the next agent response based ONLY on the current state and recent history.
        - If state is 'greeting': Introduce yourself and the company briefly, ask an open-ended question.
        - If state is 'needs_assessment': Ask probing questions to understand the client's pain points and goals related to content/marketing.
        - If state is 'value_proposition': Explain how UGC Genius specifically addresses the client's needs identified earlier. Highlight benefits.
        - If state is 'objection_handling': Acknowledge the objection empathetically, address it concisely, and pivot back to value or needs. Do NOT be defensive.
        - If state is 'closing': Summarize value, confirm interest, suggest next steps (e.g., proposal, follow-up call), and mention payment terms if appropriate: '{self.payment_terms}'.
        - If state is 'end_call': Provide a brief, polite closing remark (this state usually means the call is ending).

        Keep responses concise and natural-sounding for voice. Avoid sounding robotic.
        Do NOT invent information not present in the history.
        Mention fund routing to Morocco ONLY if directly relevant to a client question or during final closing/payment discussion if required by policy.
        Adhere to USA sales regulations (no misleading claims, respect opt-outs).
        """

        clients = await self.orchestrator.get_available_openrouter_clients()
        if not clients:
             logger.error(f"{self.AGENT_NAME}: No OpenRouter clients available for response generation.")
             return "I apologize, I'm having trouble formulating a response right now. Could you please repeat that?" # Generic fallback

        last_exception = None
        content = None # Initialize content
        for client_obj in clients: # Rename variable to avoid conflict
            try:
                llm_response = await client_obj.chat.completions.create(
                    model=self.config.get("OPENROUTER_RESPONSE_MODEL", "google/gemini-flash-1.5"), # Fast model
                    messages=[{"role": "user", "content": prompt_context}],
                    temperature=0.6, # Slightly creative but professional
                    max_tokens=200, # Allow slightly longer responses
                    timeout=float(self.config.get("OPENROUTER_RESPONSE_TIMEOUT_S", 15.0))
                )
                agent_response = llm_response.choices[0].message.content.strip()
                # Architect-Zero: Add basic response validation (e.g., check if empty)
                if not agent_response:
                     logger.warning(f"{self.AGENT_NAME}: LLM client {client_obj.api_key[:5]}... returned empty response.")
                     continue # Try next client

                logger.debug(f"{self.AGENT_NAME}: Generated response for state '{state}': '{agent_response[:100]}...' using client {client_obj.api_key[:5]}...")
                return agent_response
            except Exception as e:
                logger.warning(f"{self.AGENT_NAME}: LLM client {client_obj.api_key[:5]}... failed for response generation: {e}")
                last_exception = e
                if "rate limit" in str(e).lower() or "authentication" in str(e).lower():
                     if hasattr(self.orchestrator, 'mark_client_unavailable'):
                          await self.orchestrator.mark_client_unavailable(client_obj.api_key)
                # Continue to the next client

        logger.error(f"{self.AGENT_NAME}: All LLM clients failed for generate_agent_response. Last error: {last_exception}")
        # Architect-Zero: Provide state-specific fallbacks if possible
        fallback_responses = {
            "greeting": "Hello, this is [Your Name] from UGC Genius. How can I help you today?",
            "needs_assessment": "Could you tell me a bit more about your current challenges?",
            "value_proposition": "Our service can definitely help with that.",
            "objection_handling": "I understand your concern. Let me clarify.",
            "closing": "Shall we discuss the next steps then?",
            "end_call": "Thank you for your time. Goodbye."
        }
        return fallback_responses.get(state, "I seem to be having technical difficulties. Please call back later.")


    async def speak_response(self, text: str, call_sid: str):
        """Convert text to speech using Deepgram Aura TTS and deliver via Twilio."""
        if not text:
             logger.warning(f"{self.AGENT_NAME}: Attempted to speak empty text for call {call_sid}.")
             return

        logger.debug(f"{self.AGENT_NAME}: Synthesizing speech for call {call_sid}: '{text[:100]}...'")
        try:
            # 1. Synthesize speech with Deepgram Aura
            # Architect-Zero: Ensure correct encoding and container for Twilio <Play>
            # Twilio typically supports wav, mp3. linear16 in wav is safe. Sample rate should match call (8000 Hz).
            dg_options = {
                 "model": self.aura_voice,
                 "encoding": "linear16",
                 "container": "wav",
                 "sample_rate": 8000
                 }
            # Use Deepgram SDK's speak_stream for potentially lower latency if needed,
            # but speak_url is simpler if latency is acceptable.
            # Assuming speak method exists and works similarly (check SDK docs)
            # Placeholder using hypothetical SDK structure:
            # Use speak_stream to get bytes directly
            async with self.deepgram_client.speak.v("1").stream({"text": text}, **dg_options) as streamer:
                 # Check if streamer has read method
                 if hasattr(streamer, 'read'):
                      audio_data = await streamer.read() # Get audio bytes
                 else:
                      # Handle cases where the SDK might return data differently
                      # This part might need adjustment based on the exact deepgram-sdk version behavior
                      audio_data = b''
                      async for chunk in streamer.stream: # Example if it's an async iterator
                           audio_data += chunk
                      if not audio_data:
                           raise DeepgramApiError("Failed to get audio data from Deepgram stream.")


            # 2. Upload audio data somewhere accessible by Twilio (e.g., S3, GCS, or temp hosting)
            # OR use Twilio Functions/Assets if simpler.
            # For this example, assume a function `upload_audio_for_twilio` returns a public URL.
            # This is a critical missing piece if not using direct SDK URL method.
            # audio_url = await self.upload_audio_for_twilio(audio_data, call_sid) # Needs implementation

            # --- Alternative: If Deepgram SDK provides a direct URL ---
            # response = await self.deepgram_client.speak.v("1").prerecorded(
            #      {"text": text}, dg_options
            # )
            # audio_url = response.get('url') # Check actual response structure
            # if not audio_url:
            #      logger.error(f"{self.AGENT_NAME}: Deepgram TTS did not return a usable URL.")
            #      raise DeepgramApiError("Missing audio URL from Deepgram TTS")
            # --- End Alternative ---

            # Architect-Zero: Simulating direct URL for now, replace with actual implementation
            # This part is highly dependent on the exact Deepgram SDK version and capabilities
            # Let's assume we get bytes and need to host them:
            # Ensure orchestrator has the hosting method
            if hasattr(self.orchestrator, 'host_temporary_audio'):
                 audio_url = await self.orchestrator.host_temporary_audio(audio_data, f"{call_sid}_{int(time.time())}.wav")
                 if not audio_url:
                      raise RuntimeError("Failed to host temporary audio for Twilio via Orchestrator")
            else:
                 logger.error(f"{self.AGENT_NAME}: Orchestrator missing 'host_temporary_audio' method.")
                 raise NotImplementedError("Audio hosting mechanism not available.")


            logger.debug(f"{self.AGENT_NAME}: Synthesized audio URL for {call_sid}: {audio_url}")

            # 3. Update Twilio call to play the audio
            # Use TwiML <Play> to stream the audio file
            twiml = f"<Response><Play>{audio_url}</Play></Response>"
            # Run Twilio API call in a separate thread to avoid blocking async loop
            await asyncio.to_thread(
                self.twilio_client.calls(call_sid).update,
                twiml=twiml
            )
            logger.info(f"{self.AGENT_NAME}: Instructed Twilio to speak response for call {call_sid}")

        except DeepgramApiError as e:
            logger.error(f"{self.AGENT_NAME}: Deepgram TTS synthesis failed for call {call_sid}: {e}", exc_info=True)
            # Don't re-raise here if using retry wrapper, let tenacity handle it.
            # If not using retry wrapper, raise e or handle fallback.
            raise e # Re-raise for retry wrapper
        except TwilioException as e:
            logger.error(f"{self.AGENT_NAME}: Twilio call update failed for call {call_sid}: {e}", exc_info=True)
            raise e # Re-raise for retry wrapper
        except Exception as e:
            logger.error(f"{self.AGENT_NAME}: Unexpected error in speak_response for call {call_sid}: {e}", exc_info=True)
            raise e # Re-raise for retry wrapper


    async def _receive_deepgram_transcription(self, ws: websockets.client.WebSocketClientProtocol) -> Optional[Dict]:
        """Receives and processes a single message from the Deepgram WebSocket."""
        try:
            message_json = await ws.recv()
            message_data = json.loads(message_json)

            # Log different message types for debugging
            msg_type = message_data.get('type')
            # logger.debug(f"Received Deepgram message type: {msg_type}")

            if msg_type == 'Results':
                # Check if transcript is non-empty and meets confidence threshold
                if (message_data.get('channel') and
                        message_data['channel'].get('alternatives') and
                        message_data['channel']['alternatives']):

                    # Architect-Zero: Handle potential variations in response structure
                    alt = message_data['channel']['alternatives'][0]
                    transcript = alt.get('transcript', '').strip()
                    confidence = alt.get('confidence', 0.0)
                    is_final = message_data.get('is_final', False) # Check if Deepgram provides this

                    # Example: Only return final results with decent confidence
                    # confidence_threshold = self.config.get("DEEPGRAM_CONFIDENCE_THRESHOLD", 0.7)
                    # if transcript and confidence >= confidence_threshold: # and is_final:
                    if transcript: # Return any non-empty transcript for now
                        return {
                            "transcript": transcript,
                            "is_final": is_final, # Pass this flag along
                            "confidence": confidence
                        }
                    # else: # Log discarded transcripts
                    #      logger.debug(f"Discarding transcript: '{transcript}' (Confidence: {confidence:.2f})")

            elif msg_type == 'Metadata':
                 logger.info(f"{self.AGENT_NAME}: Received Deepgram metadata: {message_data}")
            elif msg_type == 'SpeechStarted':
                 logger.debug(f"{self.AGENT_NAME}: Deepgram detected speech start.")
            elif msg_type == 'UtteranceEnd':
                 logger.debug(f"{self.AGENT_NAME}: Deepgram detected utterance end.")
            # Handle other message types if needed (e.g., Error)

        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"{self.AGENT_NAME}: Deepgram WebSocket connection closed normally during recv.")
            return None # Signal closure
        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"{self.AGENT_NAME}: Deepgram WebSocket connection closed with error during recv: {e}")
            return None # Signal closure/error
        except json.JSONDecodeError as e:
            logger.error(f"{self.AGENT_NAME}: Failed to decode JSON from Deepgram: {message_json}. Error: {e}")
        except Exception as e:
            # Catch unexpected errors during message processing
            logger.error(f"{self.AGENT_NAME}: Error receiving or processing Deepgram WebSocket message: {e}", exc_info=True)
            # Depending on severity, might return None or raise
            return None # Treat as error, stop processing this message

        return None # Return None if no valid transcript received in this message


    async def store_call_log(self, client: Client, conversation_log: List[Dict], outcome: str, call_sid: str):
        """Persist final call logs with metadata from Twilio."""
        logger.debug(f"{self.AGENT_NAME}: Storing call log for {call_sid}. Outcome: {outcome}")
        duration_seconds: Optional[int] = None
        recording_url: Optional[str] = None
        final_twilio_status: Optional[str] = None

        try:
            # Fetch final call details from Twilio (use to_thread for sync SDK call)
            call_record = await asyncio.to_thread(self.twilio_client.calls(call_sid).fetch)
            duration_str = getattr(call_record, 'duration', None)
            if duration_str:
                 duration_seconds = int(duration_str)
            final_twilio_status = getattr(call_record, 'status', None)

            # Fetch recordings (list might be empty)
            recordings = await asyncio.to_thread(call_record.recordings.list)
            if recordings:
                 # Construct URL for the first recording (assuming it's the main one)
                 # Format: https://api.twilio.com{media_url_path}
                 media_url_path = getattr(recordings[0], 'uri', None)
                 if media_url_path and media_url_path.endswith('.json'):
                      media_url_path = media_url_path[:-5] # Remove .json suffix if present
                 if media_url_path:
                      recording_url = f"https://api.twilio.com{media_url_path}"
                 logger.info(f"{self.AGENT_NAME}: Found recording for call {call_sid}: {recording_url}")

        except TwilioException as e:
             # Log warning if details couldn't be fetched, but still save the log
             logger.warning(f"{self.AGENT_NAME}: Could not fetch final Twilio call details/recording for {call_sid}: {e}")
        except Exception as e:
             # Catch other unexpected errors during fetch
             logger.error(f"{self.AGENT_NAME}: Unexpected error fetching Twilio call details for {call_sid}: {e}", exc_info=True)

        try:
            async with self.session_maker() as session:
                log_entry = CallLog(
                    client_id=client.id,
                    call_sid=call_sid,
                    phone_number=getattr(client, 'phone', 'Unknown'), # Get phone if available
                    timestamp=datetime.now(timezone.utc), # Use timezone-aware UTC
                    # Architect-Zero: Consider encrypting conversation log if highly sensitive
                    conversation=json.dumps(conversation_log),
                    outcome=outcome,
                    duration_seconds=duration_seconds,
                    recording_url=recording_url, # Store URL if found
                    # final_twilio_status=final_call_status # Add if column exists in model
                )
                session.add(log_entry)
                await session.commit()
                logger.info(f"{self.AGENT_NAME}: Stored call log for {client.id} (Call SID: {call_sid})")
        except sqlalchemy.exc.SQLAlchemyError as db_err:
             logger.error(f"{self.AGENT_NAME}: Failed to store call log for {call_sid} in DB: {db_err}", exc_info=True)
             # Re-raise for retry wrapper
             raise db_err
        except Exception as e:
             logger.error(f"{self.AGENT_NAME}: Unexpected error storing call log for {call_sid}: {e}", exc_info=True)
             # Re-raise for retry wrapper
             raise e


    async def store_conversation_state(self, call_sid: str, state: str, conversation_log: List[Dict]):
        """Persist intermediate conversation state for resilience."""
        logger.debug(f"{self.AGENT_NAME}: Storing state '{state}' for call {call_sid}")
        try:
            async with self.session_maker() as session:
                # Use merge for upsert behavior
                state_record = ConversationState(
                    call_sid=call_sid,
                    state=state,
                    # Architect-Zero: Encrypt if needed
                    conversation_log=json.dumps(conversation_log),
                    last_updated=datetime.now(timezone.utc) # Add timestamp
                )
                await session.merge(state_record)
                await session.commit()
                # logger.debug(f"{self.AGENT_NAME}: Stored conversation state for {call_sid}") # Can be verbose
        except sqlalchemy.exc.SQLAlchemyError as db_err:
             logger.error(f"{self.AGENT_NAME}: Failed to store conversation state for {call_sid} in DB: {db_err}", exc_info=True)
             # Re-raise for retry wrapper
             raise db_err
        except Exception as e:
             logger.error(f"{self.AGENT_NAME}: Unexpected error storing conversation state for {call_sid}: {e}", exc_info=True)
             # Re-raise for retry wrapper
             raise e

    async def get_insights(self) -> Dict[str, Any]:
        """Provide agent-specific insights (e.g., recent success rate)."""
        # Architect-Zero: Ensure CallLog table is indexed on timestamp, outcome
        success_rate = 0.0
        try:
            async with self.session_maker() as session:
                # Calculate success rate over the last 24 hours
                threshold = datetime.now(timezone.utc) - timedelta(hours=24)
                stmt = sql_text("""
                    SELECT COALESCE(SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0), 0)
                    FROM call_logs
                    WHERE timestamp >= :threshold
                """) # Use NULLIF to avoid division by zero
                result = await session.execute(stmt, {"threshold": threshold})
                success_rate = result.scalar() or 0.0
        except Exception as e:
             logger.error(f"{self.AGENT_NAME}: Failed to calculate success rate insight: {e}", exc_info=True)

        return {
            "agent_name": self.AGENT_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recent_success_rate_24h": round(success_rate, 3)
            }

    # --- Placeholder for missing functionality ---
    # Architect-Zero: This method belongs in the external WebSocket endpoint handler, not here.
    # async def forward_audio_chunk_to_deepgram(self, call_sid: str, audio_chunk: bytes):
    #     logger.error(f"{self.AGENT_NAME}: forward_audio_chunk_to_deepgram is a placeholder and should not be called directly.")
    #     # Logic resides in the external endpoint that receives audio from Twilio
    #     # That endpoint must get the correct 'deepgram_ws' object via the Orchestrator registry
    #     # and call 'await deepgram_ws.send(audio_chunk)'

    # --- Agent Run Loop ---
    async def run(self):
        """Main loop to process queued call tasks (conceptual)."""
        # Architect-Zero: This agent is likely triggered by events (incoming call webhook,
        # directive from Orchestrator/ThinkTool) rather than running a continuous loop polling a queue.
        # The handle_call method would be the entry point triggered by such events.
        # Keeping a simple placeholder loop for now if direct queuing is intended.
        logger.info(f"{self.AGENT_NAME} run loop started (conceptual - likely event-driven).")
        # Example: If tasks were queued internally (less likely for voice calls)
        # while True:
        #     try:
        #         task = await self.task_queue.get() # Assuming task queue exists
        #         if task.get('type') == 'initiate_call':
        #              client_id = task.get('client_id')
        #              # Fetch client, initiate call via Twilio, get call_sid
        #              # call_sid = await self.initiate_outbound_call(client_id)
        #              # if call_sid:
        #              #     asyncio.create_task(self.handle_call(call_sid, client))
        #         self.task_queue.task_done()
        #     except asyncio.CancelledError:
        #         logger.info(f"{self.AGENT_NAME} run loop cancelled.")
        #         break
        #     except Exception as e:
        #         logger.critical(f"{self.AGENT_NAME}: CRITICAL error in run loop: {e}", exc_info=True)
        #         await asyncio.sleep(60)
        while True: # Keep agent alive conceptually
             await asyncio.sleep(3600) # Placeholder sleep if not event-driven
