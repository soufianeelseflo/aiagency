import os
import json
import time
import requests
import psycopg2
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from twilio.rest import Client  # Ref: https://www.twilio.com/docs/libraries/python
from twilio.base.exceptions import TwilioRestException  # For error handling
from agents.email_manager import EmailManager
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
        # Budget manager for OpenRouter ($20 limit, DeepSeek R1 pricing)
        self.budget_manager = BudgetManager(total_budget=20.0, input_cost_per_million=0.80, output_cost_per_million=2.40)
        
        # Proxy rotator for SmartProxy
        self.proxy_rotator = ProxyRotator()
        
        # Service configuration (Argil.ai uses web automation initially)
        self.services = {
            "argil_ugc": {
                "endpoint": None,  # Handled via web automation
                "cost_per_unit": 0.0,  # Free via trials
                "fallback": "open_router_ugc",
                "rate_limit": (50, 60)
            },
            "open_router_ugc": {
                "endpoint": "https://openrouter.ai/api/v1/ugc",
                "cost_per_unit": 0.10,
                "rate_limit": (100, 60)
            }
        }

        # Initialize core systems
        self.db_pool = self._init_db_pool()
        self.email_manager = EmailManager()
        self.deepseek_r1 = DeepSeekOrchestrator(self.budget_manager, proxy_rotator=self.proxy_rotator)
        self.argil_agent = ArgilAutomationAgent()

        # Twilio WhatsApp and Voice setup
        self.twilio_client = Client(
            os.getenv("TWILIO_SID"),  # From environment variables
            os.getenv("TWILIO_TOKEN")
        )
        self.my_whatsapp_number = os.getenv("WHATSAPP_NUMBER")  # e.g., "whatsapp:+YOUR_PHONE_NUMBER"
        if not self.my_whatsapp_number:
            raise ValueError("WHATSAPP_NUMBER must be set in environment variables.")
        self.twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")  # Sandbox number
        self.twiml_bin_url = os.getenv("TWIML_BIN_URL")  # TwiML Bin URL for voice call
        if not self.twiml_bin_url:
            raise ValueError("TWIML_BIN_URL must be set in environment variables.")
        self.twilio_voice_number = os.getenv("TWILIO_VOICE_NUMBER")  # Twilio trial phone number for voice
        if not self.twilio_voice_number:
            raise ValueError("TWILIO_VOICE_NUMBER must be set in environment variables.")

        # State management
        self.service_usage = {service: {'count': 0, 'errors': 0} for service in self.services}
        self.last_optimization = datetime.utcnow()

        # Initialize database
        self.initialize_database()

    def _init_db_pool(self):
        """Initialize PostgreSQL connection pool (ref: https://www.psycopg.org/docs/pool.html)."""
        try:
            pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=5,
                maxconn=20,
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
        """Ensure necessary tables exist."""
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
            "remaining_budget": self.budget_manager.get_remaining_budget()
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
                raise ValueError("Insufficient OpenRouter budget.")
            
            service_decision = self.deepseek_r1.query(routing_prompt, max_tokens=500)
            selected_service = json.loads(service_decision['choices'][0]['message']['content'])['service']
            
            if selected_service == "argil_ugc":
                content = self._generate_argil_ugc(client_request)
            elif selected_service == "open_router_ugc":
                content = self._generate_openrouter_ugc(client_request)
            else:
                raise ValueError(f"Unknown service: {selected_service}")
            
            if not self._quality_check(content):
                raise ValueError("Quality check failed")
            
            self._deliver_to_client(client_request, content)
            return content
        except Exception as e:
            self._handle_failure(client_request, str(e))
            return self.route_ugc_request(client_request)  # Retry with fallback

    def _generate_argil_ugc(self, request: dict) -> dict:
        """Generate UGC via ArgilAutomationAgent."""
        logging.info("Argil UGC generation via automation triggered.")
        result = self.argil_agent.run()
        if result.get("status") == "success":
            return {"url": result['url'], "content": "Generated via Argil.ai trial"}
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
        """Deep quality assurance check using DeepSeek R1."""
        validation_prompt = f"""
        Validate UGC content against requirements:
        - Technical specs: {content.get('specifications', 'N/A')}
        - Brand guidelines: {content.get('brand_rules', 'N/A')}
        - Platform requirements: {content.get('platform_specs', 'N/A')}
        
        Content to validate: {content.get('content', 'N/A')[:1000]}
        """
        try:
            if not self.budget_manager.can_afford(input_tokens=500, output_tokens=500):
                return False
            validation = self.deepseek_r1.query(validation_prompt, max_tokens=500)
            return json.loads(validation['choices'][0]['message']['content'])['approval']
        except Exception as e:
            logging.error(f"Quality check failed: {str(e)}")
            return False

    def _deliver_to_client(self, client_request: dict, content: dict) -> None:
        """Deliver generated content to the client via WhatsApp and owner via link."""
        client_message = f"@Orchestrator Your UGC Content is Ready\nHere is your requested content: {content.get('url', 'N/A')}"
        owner_message = f"@Orchestrator New UGC Video for {client_request['company']}\nVideo Link: {content.get('url', 'N/A')} (Open in Chrome)"
        try:
            # Send to client (your WhatsApp number for testing)
            self.twilio_client.messages.create(
                body=client_message,
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
            logging.info(f"Content delivered to client {client_request['client_id']}.")

            # Send to you (owner) with Chrome link
            self.twilio_client.messages.create(
                body=owner_message,
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
            logging.info(f"Video link sent to owner for review.")
        except Exception as e:
            logging.error(f"Failed to send WhatsApp message: {str(e)}")

    def _handle_failure(self, client_request: dict, error: str) -> None:
        """Handle failures with fallback and alerting."""
        logging.error(f"Request failed: {error}")
        self._critical_alert(f"Request for client {client_request['client_id']} failed: {error}")

    def _critical_alert(self, message: str) -> None:
        """Emergency alerting system via WhatsApp and PostgreSQL."""
        try:
            self.twilio_client.messages.create(
                body=f"@Orchestrator {message}",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
            remaining_budget = self.budget_manager.get_remaining_budget()
            if remaining_budget < 4.0:  # 20% of $20
                self.twilio_client.messages.create(
                    body=f"@Orchestrator Budget Warning: Only ${remaining_budget:.2f} left! Acquire a client soon!",
                    from_=self.twilio_whatsapp_number,
                    to=self.my_whatsapp_number
                )
        except Exception as e:
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
        """Real-time cost calculation for services (excluding OpenRouter)."""
        total = 0.0
        for service, data in self.service_usage.items():
            total += data['count'] * self.services[service]['cost_per_unit']
        return total

    def _send_status_update(self) -> None:
        """Send daily WhatsApp update on clients acquired and budgets."""
        clients_acquired = self._count_clients_24h()
        budgets = {
            "openrouter": self.budget_manager.get_remaining_budget(),
            "smartproxy": 10.0 - 10.0,  # Assume $10 spent initially, remaining $0
            "elevenlabs": 0.0  # Free tier, no cost
        }
        message = (f"@Orchestrator Status Update (Feb 27, 2025):\n"
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
        except Exception as e:
            logging.error(f"Failed to send WhatsApp status update: {str(e)}")

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
        """Extract E.164 phone number from WhatsApp number format for voice calls."""
        if whatsapp_number.startswith("whatsapp:"):
            return whatsapp_number[len("whatsapp:"):]
        return whatsapp_number

    def _initiate_voice_call(self):
        """Initiate a voice call to the owner's personal number using Twilio Voice API."""
        try:
            to_number = self._extract_phone_number(self.my_whatsapp_number)
            call = self.twilio_client.calls.create(
                url=self.twiml_bin_url,  # TwiML Bin URL defining call behavior
                to=to_number,  # Your personal phone number in E.164 format
                from_=self.twilio_voice_number  # Twilio trial phone number
            )
            logging.info(f"Initiated voice call to {to_number}: SID {call.sid}")
        except TwilioRestException as e:
            logging.error(f"Failed to initiate voice call: {e}")
            self._critical_alert(f"Voice call failed: {e}")

    def _poll_whatsapp_messages(self):
        """Poll Twilio for new WhatsApp messages (returns list of tuples: message body, sender number)."""
        try:
            messages = self.twilio_client.messages.list(
                to=self.twilio_whatsapp_number,  # Sandbox number receiving messages
                date_sent=datetime.utcnow().date(),  # Messages from today
                limit=10  # Limit to recent messages
            )
            # Return list of (message body, sender number) for messages starting with "@"
            return [(msg.body, msg.from_) for msg in messages if msg.body.startswith("@")]
        except TwilioRestException as e:
            logging.error(f"Failed to poll WhatsApp messages: {e}")
            return []

    def _handle_whatsapp_command(self, message: str, from_number: str) -> None:
        """Handle WhatsApp commands, restricting sensitive actions to the owner's number."""
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
            if command == "yes this is exactly what i want":
                self._record_video_feedback("This is exactly the type of ads to create")
                self.twilio_client.messages.create(
                    body="@Orchestrator Video feedback recorded. Ads will now match your preference.",
                    from_=self.twilio_whatsapp_number,
                    to=self.my_whatsapp_number
                )
            else:
                self.twilio_client.messages.create(
                    body="@Orchestrator Unknown command. Use '@videoagent yes this is exactly what i want' to set ad preferences.",
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
        """Record owner feedback to refine ArgilAutomationAgent video creation."""
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
            logging.error(f"Failed to record video feedback: {e}")

    def run(self) -> None:
        """Main execution loop with daily status updates and WhatsApp command handling."""
        while True:
            try:
                time.sleep(300)  # 5-minute cycle
                logging.info("Orchestrator cycle completed.")
                if datetime.utcnow().hour == 0:  # Send update at midnight UTC
                    self._send_status_update()

                # Check for new WhatsApp messages
                messages = self._poll_whatsapp_messages()
                for message_body, from_number in messages:
                    self._handle_whatsapp_command(message_body, from_number)

            except Exception as e:
                self._critical_alert(f"Main loop failure: {e}")
                time.sleep(60)

# Backend API endpoints
@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    """Fetch real-time status (ref: https://flask.palletsprojects.com/en/2.3.x/api/)."""
    return jsonify(orchestrator.get_status())

@app.route('/api/update-agent', methods=['POST'])
def update_agent():
    """Update orchestrator parameters via web interface."""
    data = request.get_json()
    updates = data.get("updates", {})
    orchestrator.update_parameters(updates)
    return jsonify({"message": "Orchestrator updated successfully."})

if __name__ == "__main__":
    orchestrator = Orchestrator()
    app.run(host=os.getenv("WEB_UI_HOST", "0.0.0.0"), port=int(os.getenv("WEB_UI_PORT", 5000)))