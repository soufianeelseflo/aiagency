import os
import json
import time
import requests
import psycopg2
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from twilio.rest import Client  # Import Twilio client <button class="citation-flag" data-index="1">
from utils.db_logger import DBLogger
from agents.email_manager import EmailManager
from integrations.deepseek_r1 import DeepSeekOrchestrator

# Initialize Flask app
app = Flask(__name__)

class Orchestrator:
    def __init__(self):
        # Service configuration with real-time cost tracking
        self.services = {
            "argil_ugc": {
                "endpoint": "https://api.argil.ai/v1/ugc/generate",
                "cost_per_unit": 0.15,
                "fallback": "open_router_ugc",
                "rate_limit": (50, 60)  # 50 requests/minute
            },
            "open_router_ugc": {
                "endpoint": "https://openrouter.ai/api/v1/ugc",
                "cost_per_unit": 0.10,
                "rate_limit": (100, 60)
            }
        }

        # Initialize core systems
        self.db_pool = self._init_db_pool()
        self.db_logger = DBLogger()
        self.email_manager = EmailManager()
        self.deepseek_r1 = DeepSeekOrchestrator()

        # Twilio WhatsApp setup
        self.twilio_client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )
        self.my_whatsapp_number = os.getenv("MY_WHATSAPP_NUMBER")  # Your WhatsApp number
        self.twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER")  # Twilio's WhatsApp number

        # Ensure database tables exist
        self.initialize_database()

        # State management
        self.service_usage = {service: {'count': 0, 'errors': 0} 
                              for service in self.services}
        self.last_optimization = datetime.utcnow()

    def _init_db_pool(self):
        """Initialize PostgreSQL connection pool."""
        return psycopg2.pool.ThreadedConnectionPool(
            minconn=5,
            maxconn=20,
            dbname=os.getenv('POSTGRES_DB'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host=os.getenv('POSTGRES_HOST')
        )

    def initialize_database(self):
        """Ensure all necessary tables exist."""
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
            resolved BOOLEAN
        );

        CREATE TABLE IF NOT EXISTS optimizations (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_path VARCHAR(255),
            new_code TEXT,
            reason TEXT
        );
        """
        with self.db_pool.getconn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_tables_query)
            conn.commit()
            self.db_pool.putconn(conn)

    def get_status(self):
        """Expose current status of the orchestrator."""
        return {
            "service_usage": self.service_usage,
            "last_optimization": self.last_optimization.isoformat(),
            "total_costs": self._calculate_costs()
        }

    def update_parameters(self, updates):
        """Update parameters safely."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def route_ugc_request(self, client_request: dict) -> dict:
        """Handle end-to-end UGC request lifecycle."""
        # Phase 1: Service selection
        routing_prompt = f"""
        Analyze UGC request from client {client_request['client_id']}:
        Type: {client_request['content_type']}
        Tone: {client_request['tone']}
        Platforms: {client_request['platforms']}
        Budget: ${client_request['budget']}
        
        Available services: {list(self.services.keys())}
        Current service status: {json.dumps(self.service_usage)}
        
        Recommend optimal service considering:
        1. Cost efficiency
        2. Content quality requirements
        3. Current system load
        4. Client priority ({client_request.get('priority', 'standard')})
        """

        service_decision = self.deepseek_r1.query(routing_prompt)
        selected_service = json.loads(service_decision)['service']
        
        # Phase 2: Content generation
        try:
            if selected_service == "argil_ugc":
                content = self._generate_argil_ugc(client_request)
            elif selected_service == "open_router_ugc":
                content = self._generate_openrouter_ugc(client_request)
            
            # Phase 3: Quality assurance
            if not self._quality_check(content):
                raise ValueError("Quality check failed")
            
            # Phase 4: Client delivery
            self._deliver_to_client(client_request, content)
            
            return content
        
        except Exception as e:
            self._handle_failure(client_request, str(e))
            return self.route_ugc_request(client_request)  # Retry with fallback

    def _generate_argil_ugc(self, request: dict) -> dict:
        """Generate UGC content using Argil.ai's API."""
        headers = {"Authorization": f"Bearer {os.getenv('ARGIL_API_KEY')}"}
        payload = {
            "template_id": request['template_id'],
            "variables": request['variables'],
            "output_format": request.get('format', 'mp4')
        }
        
        response = requests.post(
            self.services['argil_ugc']['endpoint'],
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def _quality_check(self, content: dict) -> bool:
        """Deep quality assurance check using multiple criteria."""
        validation_prompt = f"""
        Validate UGC content against requirements:
        - Technical specs: {content['specifications']}
        - Brand guidelines: {content['brand_rules']}
        - Platform requirements: {content['platform_specs']}
        
        Content to validate: {content['content'][:1000]}
        """
        
        validation = self.deepseek_r1.query(validation_prompt)
        return json.loads(validation)['approval']

    def _deliver_to_client(self, client_request: dict, content: dict) -> None:
        """Deliver generated content to the client via WhatsApp."""
        message_body = f"@Orchestrator Your UGC Content is Ready\nHere is your requested content: {content['url']}"
        try:
            self.twilio_client.messages.create(
                body=message_body,
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
        except Exception as e:
            logging.error(f"Failed to send WhatsApp message: {str(e)}")

    def optimize_workflows(self) -> None:
        """Autonomous system optimization every 6 hours."""
        if datetime.utcnow() - self.last_optimization < timedelta(hours=6):
            return

        analysis_prompt = f"""
        Analyze system performance data:
        {json.dumps(self.service_usage)}
        Current costs: {self._calculate_costs()}
        
        Recommend optimizations considering:
        1. Cost reduction opportunities
        2. Service reliability improvements
        3. Load balancing
        4. New features/services to implement
        """
        
        optimizations = self.deepseek_r1.query(analysis_prompt)
        self._apply_optimizations(json.loads(optimizations))
        self.last_optimization = datetime.utcnow()

    def _apply_optimizations(self, plan: dict) -> None:
        """Execute optimization plan through self-modifier agent."""
        from agents.self_modifier import CodeRefactorer
        
        modifier = CodeRefactorer()
        for change in plan['changes']:
            try:
                modifier.safe_apply_update({
                    'file_path': change['file_path'],
                    'new_content': change['new_code'],
                    'commit_message': f"Optimization: {change['reason']}"
                })
                self.db_logger.log_optimization(change)
            except Exception as e:
                self._critical_alert(f"Optimization failed: {str(e)}")

    def _critical_alert(self, message: str) -> None:
        """Emergency alerting system via WhatsApp."""
        try:
            self.twilio_client.messages.create(
                body=f"@Orchestrator {message}",
                from_=self.twilio_whatsapp_number,
                to=self.my_whatsapp_number
            )
        except Exception as e:
            logging.error(f"Failed to send WhatsApp alert: {str(e)}")

        # Log to PostgreSQL
        with self.db_pool.getconn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO critical_alerts (timestamp, message, resolved)
                    VALUES (%s, %s, %s)
                """, (datetime.utcnow(), message, False))
            conn.commit()
            self.db_pool.putconn(conn)

    def _calculate_costs(self) -> float:
        """Real-time cost calculation."""
        total = 0.0
        for service, data in self.service_usage.items():
            total += data['count'] * self.services[service]['cost_per_unit']
        return total

    def run(self) -> None:
        """Main execution loop."""
        while True:
            try:
                # Monitor system health
                self.optimize_workflows()
                
                # Auto-scale with Coolify
                self._scale_resources()
                
                # Cleanup old tasks
                self._cleanup_old_tasks()
                
                time.sleep(300)  # 5-minute cycle
                
            except Exception as e:
                self._critical_alert(f"Main loop failure: {str(e)}")
                time.sleep(60)

# Backend API endpoints
@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    """Fetch real-time status of the orchestrator."""
    return jsonify(orchestrator.get_status())

@app.route('/api/update-agent', methods=['POST'])
def update_agent():
    """Update orchestrator parameters via web interface."""
    data = request.json
    updates = data.get("updates", {})
    orchestrator.update_parameters(updates)
    return jsonify({"message": "Orchestrator updated successfully."}) 

if __name__ == "__main__":
    orchestrator = Orchestrator()
    app.run(host="0.0.0.0", port=5000)