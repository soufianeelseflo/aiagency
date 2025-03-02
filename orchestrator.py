# orchestrator.py
import os
import json
import time
import requests
import psycopg2
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from agents.email_manager import EmailManager
from agents.executor import AcquisitionEngine
from agents.voice_agent import VoiceSalesAgent
from agents.legal_compliance_agent import LegalComplianceAgent
from integrations.deepseek_r1 import DeepSeekOrchestrator
from utils.budget_manager import BudgetManager
from utils.proxy_rotator import ProxyRotator
from agents.argil_automation_agent import ArgilAutomationAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

class Orchestrator:
    def __init__(self):
        # Budget manager with dynamic configuration
        self.budget_manager = BudgetManager(
            total_budget=float(os.getenv("TOTAL_BUDGET", 20.0)),
            input_cost_per_million=float(os.getenv("INPUT_COST_PER_M", 0.80)),
            output_cost_per_million=float(os.getenv("OUTPUT_COST_PER_M", 2.40))
        )
        
        # Proxy rotator for SmartProxy
        self.proxy_rotator = ProxyRotator()
        
        # Service configuration with dynamic endpoints
        self.services = {
            "argil_ugc": {
                "endpoint": None,  # Handled via web automation
                "cost_per_unit": 0.0,  # Free via trials
                "fallback": "open_router_ugc",
                "rate_limit": (int(os.getenv("ARGIL_RATE_LIMIT", 50)), 60)
            },
            "open_router_ugc": {
                "endpoint": os.getenv("OPEN_ROUTER_ENDPOINT", "https://openrouter.ai/api/v1/ugc"),
                "cost_per_unit": float(os.getenv("OPEN_ROUTER_COST", 0.10)),
                "rate_limit": (int(os.getenv("OPEN_ROUTER_RATE_LIMIT", 100)), 60)
            }
        }

        # Initialize core systems
        self.db_pool = self._init_db_pool()
        self.email_manager = EmailManager()
        self.acquisition_engine = AcquisitionEngine()
        self.legal_agent = LegalComplianceAgent()
        self.deepseek_r1 = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)
        self.argil_agent = ArgilAutomationAgent()

        # Twilio WhatsApp and Voice setup
        self.twilio_client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))
        self.my_whatsapp_number = os.getenv("WHATSAPP_NUMBER")
        if not self.my_whatsapp_number:
            raise ValueError("WHATSAPP_NUMBER must be set in environment variables.")
        self.twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        self.twiml_bin_url = os.getenv("TWIML_BIN_URL")
        if not self.twiml_bin_url:
            raise ValueError("TWIML_BIN_URL must be set in environment variables.")
        self.twilio_voice_number = os.getenv("TWILIO_VOICE_NUMBER")
        if not self.twilio_voice_number:
            raise ValueError("TWILIO_VOICE_NUMBER must be set in environment variables.")

        # State management
        self.service_usage = {service: {'count': 0, 'errors': 0} for service in self.services}
        self.last_optimization = datetime.utcnow()
        self.first_video_approved = False  # For initialization approval

        # Initialize database
        self.initialize_database()

    def _init_db_pool(self):
        """Initialize PostgreSQL connection pool."""
        try:
            pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=int(os.getenv("DB_MIN_CONN", 5)),
                maxconn=int(os.getenv("DB_MAX_CONN", 20)),
                dbname=os.getenv('POSTGRES_DB', 'smma_db'),
                user=os.getenv('POSTGRES_USER', 'postgres'),
                password=os.getenv('POSTGRES_PASSWORD'),
                host=os.getenv('POSTGRES_HOST', 'postgres')
            )
            logging.info("Database pool initialized successfully.")
            return pool
        except psycopg2.Error as e:
            logging.error(f"Failed to initialize DB pool: {str(e)}")
            raise

    def initialize_database(self):
        """Ensure necessary tables exist in PostgreSQL."""
        create_tables_query = """
        CREATE TABLE IF NOT EXISTS api_usage (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255),
            tokens_used INT,
            cost FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS critical_alerts (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message TEXT,
            resolved BOOLEAN DEFAULT FALSE
        );
        CREATE TABLE IF NOT EXISTS optimizations (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_path VARCHAR(255),
            new_code TEXT,
            reason TEXT
        );
        CREATE TABLE IF NOT EXISTS client_interactions (
            id SERIAL PRIMARY KEY,
            client_id VARCHAR(255),
            interaction_type VARCHAR(50),
            details JSONB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_tables_query)
                conn.commit()
                self.db_pool.putconn(conn)
                logging.info("Database tables initialized.")
        except psycopg2.Error as e:
            logging.error(f"Database initialization failed: {str(e)}")
            raise

    def get_status(self):
        """Expose current status of the orchestrator."""
        return {
            "service_usage": self.service_usage,
            "last_optimization": self.last_optimization.isoformat(),
            "total_costs": self._calculate_costs(),
            "remaining_budget": self.budget_manager.get_remaining_budget(),
            "first_video_approved": self.first_video_approved
        }

    def update_parameters(self, updates):
        """Update parameters safely."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logging.info(f"Updated parameter {key} to {value}")

    def route_ugc_request(self, client_request: dict) -> dict:
        """Handle end-to-end UGC request lifecycle."""
        routing_prompt = f"""
        Analyze UGC request from client {client_request['client_id']}:
        Type: {client_request['content_type']}
        Tone: {client_request['tone']}
        Platforms: {client_request['platforms']}
        Budget: ${client_request['budget']}
        
        Available services: {list(self.services.keys())}
        Current service status: {json.dumps(self.service_usage)}
        
        Recommend optimal service considering:
        1. Cost efficiency (prioritize free Argil.ai trials)
        2. Content quality requirements
        3. Current system load
        4. Client priority ({client_request.get('priority', 'standard')})
        """
        try:
            if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=500):
                self._critical_alert("Budget too low to process UGC request. Acquire a client to replenish.")
                raise ValueError("Insufficient budget for UGC generation.")
            
            service_decision = self.deepseek_r1.query(routing_prompt, max_tokens=500)
            selected_service = json.loads(service_decision['choices'][0]['message']['content'])['service']
            
            if selected_service == "argil_ugc":
                content = self._generate_argil_ugc(client_request)
            elif selected_service == "open_router_ugc":
                content = self._generate_openrouter_ugc(client_request)
            else:
                raise ValueError(f"Unknown service selected: {selected_service}")
            
            if not self._quality_check(content):
                raise ValueError("Content failed quality check.")
            
            self._deliver_to_client(client_request, content)
            return content
        except Exception as e:
            self._handle_failure(client_request, str(e))
            return self.route_ugc_request(client_request)  # Retry with fallback

    def _generate_argil_ugc(self, request: dict) -> dict:
        """Generate UGC via ArgilAutomationAgent."""
        logging.info("Triggering Argil UGC generation.")
        result = self.argil_agent.run()
        if result.get("status") == "success":
            self.service_usage["argil_ugc"]["count"] += 1
            return {"url": result['url'], "content": "Generated via Argil.ai trial"}
        self.service_usage["argil_ugc"]["errors"] += 1
        raise Exception("Argil automation failed.")

    def _generate_openrouter_ugc(self, request: dict) -> dict:
        """Generate UGC content using OpenRouter API."""
        prompt = f"Generate UGC content: {json.dumps(request)}"
        try:
            if not self.budget_manager.can_afford(input_tokens=1000, output_tokens=2000):
                raise ValueError("Insufficient budget for OpenRouter UGC.")
            response = self.deepseek_r1.query(prompt, max_tokens=2000)
            self.service_usage['open_router_ugc']['count'] += 1
            return json.loads(response['choices'][0]['message']['content'])
        except Exception as e:
            self.service_usage['open_router_ugc']['errors'] += 1
            raise Exception(f"OpenRouter UGC generation failed: {str(e)}")

    def _quality_check(self, content: dict) -> bool:
        """Perform deep quality assurance check using DeepSeek R1."""
        validation_prompt = f"""
        Validate UGC content against requirements:
        - Technical specs: {content.get('specifications', 'N/A')}
        - Brand guidelines: {content.get('brand_rules', 'N/A')}
        - Platform requirements: {content.get('platform_specs', 'N/A')}
        
        Content to validate: {content.get('content', 'N/A')[:1000]}
        """
        try:
            if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
                logging.warning("Budget too low for quality check.")
                return False
            validation = self.deepseek_r1.query(validation_prompt, max_tokens=500)
            return json.loads(validation['choices'][0]['message']['content'])['approval']
        except Exception as e:
            logging.error(f"Quality check failed: {str(e)}")
            return False

    def _deliver_to_client(self, client_request: dict, content: dict) -> None:
        """Deliver generated content via WhatsApp, with initial video approval."""
        client_message = f"@Orchestrator Your UGC Content is Ready\nHere is your requested content: {content.get('url', 'N/A')}"
        owner_message = f"@Orchestrator New UGC Video for {client_request['company']}\nVideo Link: {content.get('url', 'N/A')} (Open in Chrome)"
        try:
            if not self.first_video_approved:
                self.twilio_client.messages.create(
                    body=f"@Orchestrator First Video Ready: {content.get('url', 'N/A')}\nReply '@videoagent yes' to approve.",
                    from_=self.twilio_whatsapp_number,
                    to=self.my_whatsapp_number
                )
                logging.info("Waiting for first video approval.")
            else:
                self.twilio_client.messages.create(
                    body=client_message,
                    from_=self.twilio_whatsapp_number,
                    to=self.my_whatsapp_number
                )
                logging.info(f"Content delivered to client {client_request['client_id']}.")
                self.twilio_client.messages.create(
                    body=owner_message,
                    from_=self.twilio_whatsapp_number,
                    to=self.my_whatsapp_number
                )
                logging.info("Video link sent to owner.")
        except Exception as e:
            logging.error(f"Failed to send WhatsApp message: {str(e)}")

    def _handle_failure(self, client_request: dict, error: str) -> None:
        """Handle failures with fallback and alerting."""
        logging.error(f"Request failed for client {client_request['client_id']}: {error}")
        self._critical_alert(f"Request failure: {error}")

    def _critical_alert(self, message: str) -> None:
        """Send emergency alerts via WhatsApp and log to PostgreSQL."""
        try:
            self.twilio_client.messages.create(
                body=f"@Orchestrator CRITICAL: {message}",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
            remaining_budget = self.budget_manager.get_remaining_budget()
            if remaining_budget < self.budget_manager.total_budget * 0.2:
                self.twilio_client.messages.create(
                    body=f"@Orchestrator Budget Alert: Only ${remaining_budget:.2f} left!",
                    from_=self.twilio_whatsapp_number,
                    to=self.my_whatsapp_number
                )
        except TwilioRestException as e:
            logging.error(f"Failed to send WhatsApp alert: {str(e)}")

        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO critical_alerts (timestamp, message, resolved) VALUES (%s, %s, %s)",
                        (datetime.utcnow(), message, False)
                    )
                conn.commit()
                self.db_pool.putconn(conn)
        except psycopg2.Error as e:
            logging.error(f"Failed to log alert to database: {str(e)}")

    def _calculate_costs(self) -> float:
        """Calculate real-time costs for services."""
        total = 0.0
        for service, data in self.service_usage.items():
            total += data['count'] * self.services[service]['cost_per_unit']
        return total

    def _send_status_update(self) -> None:
        """Send daily WhatsApp update on clients acquired and budgets."""
        clients_acquired = self._count_clients_24h()
        budgets = {
            "openrouter": self.budget_manager.get_remaining_budget(),
            "smartproxy": float(os.getenv("SMARTPROXY_BUDGET", 10.0)) - 10.0,  # Assume spent
            "elevenlabs": 0.0  # Free tier assumed
        }
        message = (f"@Orchestrator Status Update ({datetime.now().strftime('%b %d, %Y')}):\n"
                   f"Clients Acquired in 24h: {clients_acquired}\n"
                   f"Budgets Remaining: OpenRouter=${budgets['openrouter']:.2f}, "
                   f"SmartProxy=${budgets['smartproxy']:.2f}, ElevenLabs=${budgets['elevenlabs']:.2f}")
        try:
            self.twilio_client.messages.create(
                body=message,
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
            logging.info("Sent daily status update via WhatsApp.")
        except TwilioRestException as e:
            logging.error(f"Failed to send status update: {str(e)}")

    def _count_clients_24h(self) -> int:
        """Count clients acquired in the last 24 hours from the database."""
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT COUNT(*) FROM api_usage 
                        WHERE timestamp > NOW() - INTERVAL '24 hours'
                        AND model_name LIKE '%client_acquisition%'
                    """)
                    count = cursor.fetchone()[0]
                self.db_pool.putconn(conn)
            return count
        except psycopg2.Error as e:
            logging.error(f"Failed to count clients: {str(e)}")
            return 0

    def _extract_phone_number(self, whatsapp_number: str) -> str:
        """Extract E.164 phone number from WhatsApp number format."""
        if whatsapp_number.startswith("whatsapp:"):
            return whatsapp_number[len("whatsapp:"):]
        return whatsapp_number

    def _initiate_voice_call(self):
        """Initiate a voice call to the owner's personal number."""
        try:
            to_number = self._extract_phone_number(self.my_whatsapp_number)
            call = self.twilio_client.calls.create(
                url=self.twiml_bin_url,
                to=to_number,
                from_=self.twilio_voice_number
            )
            logging.info(f"Initiated voice call to {to_number}: SID {call.sid}")
        except TwilioRestException as e:
            logging.error(f"Failed to initiate voice call: {str(e)}")
            self._critical_alert(f"Voice call failed: {e}")

    def _poll_whatsapp_messages(self):
        """Poll Twilio for new WhatsApp messages."""
        try:
            messages = self.twilio_client.messages.list(
                to=self.twilio_whatsapp_number,
                date_sent=datetime.utcnow().date(),
                limit=10
            )
            return [(msg.body, msg.from_) for msg in messages if msg.body.startswith("@")]
        except TwilioRestException as e:
            logging.error(f"Failed to poll WhatsApp messages: {str(e)}")
            return []

    def _handle_whatsapp_command(self, message: str, from_number: str) -> None:
        """Handle WhatsApp commands, restricting sensitive actions to the owner."""
        if from_number != self.my_whatsapp_number:
            logging.warning(f"Unauthorized command attempt from {from_number}")
            self.twilio_client.messages.create(
                body="Unauthorized access. This command is restricted to the owner.",
                from_=self.twilio_whatsapp_number,
                to=from_number
            )
            return

        if message.startswith("@videoagent"):
            command = message.split(" ", 1)[1].strip().lower() if len(message.split(" ")) > 1 else ""
            if command == "yes" and not self.first_video_approved:
                self.first_video_approved = True
                self._record_video_feedback("First video approved by owner.")
                self.twilio_client.messages.create(
                    body="@Orchestrator First video approved! Full operations starting.",
                    from_=self.twilio_whatsapp_number,
                    to=self.my_whatsapp_number
                )
            elif command == "yes this is exactly what i want":
                self._record_video_feedback("This is exactly the type of ads to create.")
                self.twilio_client.messages.create(
                    body="@Orchestrator Video feedback recorded. Ads will match your preference.",
                    from_=self.twilio_whatsapp_number,
                    to=self.my_whatsapp_number
                )
            else:
                self.twilio_client.messages.create(
                    body="@Orchestrator Unknown command. Use '@videoagent yes' or '@videoagent yes this is exactly what i want'.",
                    from_=self.twilio_whatsapp_number,
                    to=self.my_whatsapp_number
                )
        elif message == "@orchestrator call_me":
            self._initiate_voice_call()
            self.twilio_client.messages.create(
                body="@Orchestrator Initiating voice call to your personal number.",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
        else:
            self.twilio_client.messages.create(
                body="@Orchestrator Unknown command.",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
        logging.info(f"Processed WhatsApp command: '{message}' from {from_number}")

    def _record_video_feedback(self, feedback: str) -> None:
        """Record owner feedback to refine video creation."""
        try:
            with self.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO optimizations (timestamp, file_path, new_code, reason)
                        VALUES (%s, %s, %s, %s)
                    """, (datetime.utcnow(), "agents/argil_automation_agent.py", "", feedback))
                conn.commit()
                self.db_pool.putconn(conn)
            logging.info(f"Recorded video feedback: {feedback}")
        except psycopg2.Error as e:
            logging.error(f"Failed to record video feedback: {str(e)}")

    def run(self) -> None:
        """Main execution loop with daily status updates and WhatsApp command handling."""
        while True:
            try:
                if self.first_video_approved:
                    lead = {"company": "TechCo", "phone": "+12025550123", "industry": "Tech", "pains": "Low engagement", "email": "lead@techco.com", "decision_maker_email": "dm@techco.com", "news": "Recent funding"}
                    if not self.acquisition_engine.is_european(self.acquisition_engine.get_country_from_phone(lead["phone"])):
                        outreach_result = self.acquisition_engine.cold_outreach(lead)
                        if outreach_result["status"] == "prepared_for_email":
                            self.email_manager.send_campaign([lead["decision_maker_email"]], outreach_result["email_payload"]["body"])
                        voice_agent = VoiceSalesAgent()
                        voice_result = voice_agent.handle_lead(lead, {"response": "Interested"})
                        logging.info(f"Voice call result: {json.dumps(voice_result, indent=2)}")
                
                time.sleep(300)  # 5-minute cycle
                logging.info("Orchestrator cycle completed.")
                if datetime.utcnow().hour == 0:  # Send update at midnight UTC
                    self._send_status_update()

                # Check for new WhatsApp messages
                messages = self._poll_whatsapp_messages()
                for message_body, from_number in messages:
                    self._handle_whatsapp_command(message_body, from_number)

            except Exception as e:
                self._critical_alert(f"Main loop failure: {str(e)}")
                time.sleep(60)

# Backend API endpoints
@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    return jsonify(orchestrator.get_status())

@app.route('/api/update-agent', methods=['POST'])
def update_agent():
    data = request.get_json()
    updates = data.get("updates", {})
    orchestrator.update_parameters(updates)
    return jsonify({"message": "Orchestrator updated successfully."})

if __name__ == "__main__":
    orchestrator = Orchestrator()
    app.run(host=os.getenv("WEB_UI_HOST", "0.0.0.0"), port=int(os.getenv("WEB_UI_PORT", 5000)))