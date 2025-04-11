import asyncio
import logging
import json
from datetime import datetime, timedelta
import pytz
from sqlalchemy.ext.asyncio import AsyncSession
from utils.database import encrypt_data
from models import Client, CallLog, ConversationState
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioException
import websockets
import aiohttp
from deepgram import Deepgram
from deepgram.exceptions import DeepgramApiError
from openai import AsyncOpenAI as AsyncDeepSeekClient
import tenacity

# Genius-level logging for production diagnostics
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("voice_sales_agent.log", mode="a")]
)

class VoiceSalesAgent:
    def __init__(self, session_maker, config, orchestrator, clients_models):
        self.session_maker = session_maker
        self.config = config
        self.orchestrator = orchestrator
        self.clients_models = clients_models  # List of (client, model) tuples
        self.think_tool = orchestrator.agents['think']

        # Twilio client for call management
        self.twilio_client = TwilioClient(
            self.config.get("TWILIO_ACCOUNT_SID"),
            self.config.get("TWILIO_AUTH_TOKEN")
        )

        # Deepgram client for transcription and TTS (Aura)
        self.deepgram_client = Deepgram(self.config.get("DEEPGRAM_API_KEY"))

        # Genius-level meta-prompt for precision and compliance
        self.meta_prompt = """
        You are a voice sales agent for UGC Genius, an AI-powered agency helping USA businesses scale content, with funds routed to Morocco.
        Your goal is to close deals starting at $5,000. Use consultative selling, empathy, and strategic questioning to build trust.
        Never sound pushy or robotic. If asked about AI or credentials, say: 'I’m part of a cutting-edge team using AI to deliver top-tier UGC services.
        We’re all about results—check out our portfolio or let’s hop on a quick call to discuss your needs.'
        Ensure compliance with USA and Moroccan laws.
        """

        self.target_country = "USA"
        self.aura_voice = "aura-asteria-en"  # Female voice for confidence and reliability

    async def simulate_call(self, client_id):
        """Simulate a sales call for testing purposes."""
        async with self.session_maker() as session:
            client = await session.get(Client, client_id)
            if not client:
                client = Client(id=client_id, name="Test Client", country="USA", timezone="America/New_York")
                session.add(client)
                await session.commit()

            conversation_log = [
                {"role": "agent", "text": "Hello, this is your AI sales assistant. How can I assist you today?"},
                {"role": "client", "text": "I’m interested in your services."},
                {"role": "agent", "text": "Great! Our solutions can boost your business, with funds securely routed to Morocco. What are your goals?"}
            ]
            log = CallLog(
                client_id=client.id,
                timestamp=datetime.now(pytz.UTC),
                conversation=json.dumps(conversation_log),
                outcome="success"
            )
            session.add(log)
            await session.commit()
            logger.info(f"Simulated call completed for client {client_id}")
            return "Simulated call completed successfully."

    def get_allowed_concurrency(self):
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        concurrency_factor = 1 - (cpu_usage / 100) * 0.5 - (memory_usage / 100) * 0.5
        allowed = max(1, int(self.max_concurrency * concurrency_factor))
        return allowed

    async def predict_optimal_call_time(self, client):
        """Predict optimal call times using historical success data and client behavior."""
        async with self.session_maker() as session:
            past_calls = await session.execute(
                "SELECT timestamp FROM call_logs WHERE client_id = :client_id AND outcome = 'success'",
                {"client_id": client.id}
            )
            success_times = [row.timestamp for row in past_calls]

            client_tz = pytz.timezone(client.timezone)
            if success_times:
                success_hours = [t.astimezone(client_tz).hour for t in success_times]
                avg_hour = int(sum(success_hours) / len(success_hours))
                optimal_time = datetime.now(client_tz).replace(hour=avg_hour, minute=0, second=0)
                if optimal_time < datetime.now(client_tz):
                    optimal_time += timedelta(days=1)
            else:
                optimal_time = datetime.now(client_tz).replace(hour=10, minute=0, second=0)
                if optimal_time < datetime.now(client_tz):
                    optimal_time += timedelta(days=1)

            optimal_time = optimal_time.astimezone(pytz.utc)
            logger.info(f"Optimal call time for {client.name}: {optimal_time}")
            return optimal_time

    @tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
    stop=tenacity.stop_after_attempt(3),
    reraise=True
)
    async def handle_call(self, call_sid, client):
        """Manage real-time calls with legal compliance and adaptability."""
        # Pre-call compliance check
        compliance_context = (
            f"Voice call to {client.name} (ID: {client.id}) in {client.country}. "
            f"Funds routed to Morocco. USA CCPA and Moroccan financial laws apply."
        )
        compliance_check = await self.orchestrator.agents['legal'].validate_operation(compliance_context)
        if not compliance_check['is_compliant']:
            logger.warning(f"Call to {client.name} blocked: {compliance_check['issues']}")
            await self.orchestrator.send_notification(
                "Compliance Alert",
                f"Call to {client.name} blocked: {compliance_check['issues']}"
            )
            return

        if client.country != self.target_country or not client.opt_in:
            logger.info(f"Skipping {client.name}: Not in {self.target_country} or not opted in.")
            return

        async with self.session_maker() as session:
            state_record = await session.get(ConversationState, call_sid)
            if state_record:
                state = state_record.state
                conversation_log = json.loads(state_record.conversation_log)
            else:
                state = "greeting"
                conversation_log = []

        # Deepgram WebSocket for real-time transcription
        deepgram_ws_url = "wss://api.deepgram.com/v1/listen?model=general-enhanced"
        async with websockets.connect(
            deepgram_ws_url,
            extra_headers={"Authorization": f"Token {self.config.get('DEEPGRAM_API_KEY')}"},
            ping_interval=20,
            ping_timeout=10
        ) as ws:
            stream_task = asyncio.create_task(self.stream_audio_to_deepgram(ws, call_sid))
            stream_task.set_name(f"stream_{call_sid}")

            try:
                while True:
                    client_response = await asyncio.wait_for(self.get_client_response(ws), timeout=30)
                    if not client_response:
                        logger.info(f"Call {call_sid} ended by client.")
                        break

                    # Check for AI detection attempts
                    detection_keywords = ["robot", "ai", "artificial", "bot"]
                    if any(keyword in client_response.lower() for keyword in detection_keywords):
                        agent_response = "I’m part of a cutting-edge team using AI to deliver top-tier UGC services. We’re all about results—check out our portfolio or let’s hop on a quick call to discuss your needs."
                        await self.speak_response(agent_response, call_sid)
                        conversation_log.append({"role": "agent", "text": agent_response})
                        continue

                    conversation_log.append({"role": "client", "text": client_response})

                    intent, confidence, sub_intents, tone = await self.interpret_intent(client_response)
                    state = await self.update_conversation_state(state, intent, confidence, tone)

                    agent_response = await self.generate_agent_response(state, client, conversation_log)
                    await self.speak_response(agent_response, call_sid)
                    conversation_log.append({"role": "agent", "text": agent_response})

                    await self.store_conversation_state(call_sid, state, conversation_log)

                    if state == "end_call":
                        logger.info(f"Call {call_sid} completed successfully.")
                        # Dynamic pricing based on OSINT and conversation
                        osint_insights = await self.orchestrator.agents['osint'].get_insights()
                        industry_avg = osint_insights.get('industry_pricing', 5000)  # Default $5,000
                        pricing = min(max(industry_avg * (1 + confidence - 0.5), 3000), 7000)  # $3,000–$7,000 range
                        await self.orchestrator.generate_invoice(client.id, pricing)
                        break
            except asyncio.TimeoutError:
                logger.error(f"Timeout in call {call_sid}.")
                await self.orchestrator.report_error("VoiceSalesAgent", f"Timeout in call {call_sid}")
            except DeepgramApiError as e:
                logger.error(f"Deepgram error in call {call_sid}: {e}")
                await self.orchestrator.report_error("VoiceSalesAgent", str(e))
            finally:
                # Post-call compliance check
                post_compliance = await self.orchestrator.agents['legal'].validate_operation(
                    f"Completed call to {client.name} (ID: {client.id}). Conversation: {json.dumps(conversation_log)}"
                )
                outcome = "success" if post_compliance['is_compliant'] else "failed"
                await self.store_call_log(client, conversation_log, outcome)
                stream_task.cancel()
    

    async def interpret_intent(self, response):
        """Analyze client intent with dynamically fetched OpenRouter clients."""
        prompt = f"""
        {self.meta_prompt}
        You are a sales intent classifier for a USA-based voice sales system directing funds to Morocco.
        Interpret the intent of this client response: '{response}'.
        Consider:
        - Conversational context from prior interactions
        - Sales psychology principles (e.g., urgency, trust-building, objection signals)
        - Emotional undertones (e.g., excitement, frustration, skepticism)
        Return a JSON object with:
        - 'intent' (string): Primary intent (e.g., 'interested', 'hesitant', 'objection', 'closing')
        - 'confidence' (float): Confidence score between 0.0 and 1.0
        - 'sub_intents' (list): Secondary intents if detected (e.g., ['curious', 'budget_concern'])
        - 'emotional_tone' (string): Detected emotional tone (e.g., 'positive', 'neutral', 'negative')
        """
        clients = await self.orchestrator.get_available_openrouter_clients()
        for client in clients:
            try:
                intent_response = await client.chat.completions.create(
                    model="google/gemini-2.0-flash-exp:free",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=100,
                    response_format={"type": "json_object"}
                )
                intent_data = json.loads(intent_response.choices[0].message.content)
                intent = intent_data.get('intent', 'unknown')
                confidence = min(max(intent_data.get('confidence', 0.0), 0.0), 1.0)
                sub_intents = intent_data.get('sub_intents', [])
                emotional_tone = intent_data.get('emotional_tone', 'neutral')
                logger.info(f"Interpreted intent: {intent} (confidence: {confidence}, sub-intents: {sub_intents}, tone: {emotional_tone})")
                return intent, confidence, sub_intents, emotional_tone
            except Exception as e:
                if "rate limit" in str(e).lower():
                    async with self.session_maker() as session:
                        await session.execute(
                            "UPDATE accounts SET is_available = FALSE WHERE api_key = :api_key",
                            {"api_key": client.api_key}
                        )
                        await session.commit()
                    continue
                logger.warning(f"Failed to use client {client.api_key}: {e}")
        logger.error("All clients failed for interpret_intent")
        return "unknown", 0.0, [], "neutral"

    async def update_conversation_state(self, current_state, intent, confidence, emotional_tone):
        """Advance the conversation state with a sophisticated state machine."""
        state_transitions = {
            "greeting": {
                "interested": "needs_assessment",
                "hesitant": "objection_handling",
                "objection": "objection_handling",
                "closing": "closing",
                "unknown": "greeting"
            },
            "needs_assessment": {
                "interested": "value_proposition",
                "hesitant": "objection_handling",
                "objection": "objection_handling",
                "closing": "closing",
                "unknown": "needs_assessment"
            },
            "value_proposition": {
                "interested": "closing",
                "hesitant": "objection_handling",
                "objection": "objection_handling",
                "closing": "closing",
                "unknown": "value_proposition"
            },
            "objection_handling": {
                "interested": "closing",
                "hesitant": "objection_handling",
                "objection": "objection_handling",
                "closing": "closing",
                "unknown": "objection_handling"
            },
            "closing": {
                "interested": "end_call",
                "hesitant": "objection_handling",
                "objection": "objection_handling",
                "closing": "end_call",
                "unknown": "closing"
            }
        }
        
        next_state = state_transitions.get(current_state, {}).get(intent, current_state)
        if confidence < 0.7:
            next_state = current_state
            logger.warning(f"Low confidence ({confidence}) for intent '{intent}'—remaining in {current_state}")
        
        if emotional_tone == "negative" and intent != "closing":
            next_state = "objection_handling"
            logger.info(f"Negative tone detected—escalating to objection_handling")

        if next_state != current_state:
            logger.info(f"Transitioning from {current_state} to {next_state} (intent: {intent}, tone: {emotional_tone})")
        return next_state

    async def generate_agent_response(self, state, client, conversation_log):
        """Craft a tailored response with dynamically fetched OpenRouter clients."""
        payment_terms = self.config.PAYMENT_TERMS
        prompt = f"""
        You are a voice sales agent for a USA-based system directing funds to Morocco.
        Current state: {state}
        Client: {client.name} (USA)
        Conversation history: {json.dumps(conversation_log, indent=2)}
        Generate a professional, adaptive response that:
        - Aligns with USA sales psychology (urgency, trust, value)
        - Mentions Morocco as the fund destination where relevant
        - Ensures compliance with USA sales regulations
        """
        if state == "closing":
            prompt += f"\n- Includes the payment terms: '{payment_terms}'"
        else:
            prompt += "\n- Does not mention payment terms yet"

        clients = await self.orchestrator.get_available_openrouter_clients()
        for client in clients:
            try:
                response = await client.chat.completions.create(
                    model="google/gemini-2.0-flash-exp:free",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=150
                )
                agent_response = response.choices[0].message.content.strip()
                logger.info(f"Generated response for {client.name}: {agent_response}")
                return agent_response
            except Exception as e:
                if "rate limit" in str(e).lower():
                    async with self.session_maker() as session:
                        await session.execute(
                            "UPDATE accounts SET is_available = FALSE WHERE api_key = :api_key",
                            {"api_key": client.api_key}
                        )
                        await session.commit()
                    continue
                logger.warning(f"Failed to use client {client.api_key}: {e}")
        logger.error("All clients failed for generate_agent_response")
        return "I’m here to assist—let’s talk about how we can direct your investment to Morocco."

    async def speak_response(self, text, call_sid):
        """Convert text to speech using Deepgram Aura TTS and deliver via Twilio."""
        try:
            # Synthesize speech with Aura TTS (female, confident, reliable)
            synthesis_response = await self.deepgram_client.speak(
                text=text,
                model="aura-asteria-en",  # Female voice
                encoding="linear16",
                container="wav"
            )
            audio_url = synthesis_response['url']
            
            # Update Twilio call with the audio
            await self.twilio_client.calls(call_sid).update(
                twiml=f"<Response><Play>{audio_url}</Play></Response>"
            )
            logger.info(f"Spoke response to {call_sid}: {text}")
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            await self.orchestrator.report_error("VoiceSalesAgent", f"Speech synthesis failed: {e}")

    async def store_call_log(self, client, conversation_log):
        """Persist conversation logs for compliance and analytics."""
        async with self.session_maker() as session:
            log = CallLog(
                client_id=client.id,
                timestamp=datetime.now(pytz.UTC),
                conversation=json.dumps(conversation_log),
                outcome="pending"  # Updated post-call
            )
            session.add(log)
            await session.commit()
            logger.info(f"Stored call log for {client.name}")

    async def store_conversation_state(self, call_sid, state, conversation_log):
        """Persist conversation state for resilience."""
        async with self.session_maker() as session:
            state_record = await session.get(ConversationState, call_sid)
            if state_record:
                state_record.state = state
                state_record.conversation_log = json.dumps(conversation_log)
            else:
                state_record = ConversationState(
                    call_sid=call_sid,
                    state=state,
                    conversation_log=json.dumps(conversation_log)
                )
                session.add(state_record)
            await session.commit()
            logger.debug(f"Stored conversation state for {call_sid}: {state}")

    async def get_insights(self):
        async with self.session_maker() as session:
            success_rate = await session.execute(
                "SELECT COUNT(*) FILTER (WHERE outcome = 'success') / COUNT(*)::float FROM call_logs WHERE timestamp > :threshold",
                {"threshold": datetime.now(pytz.UTC) - timedelta(hours=24)}
            )
            return {"success_rate": success_rate.scalar() or 0.0}