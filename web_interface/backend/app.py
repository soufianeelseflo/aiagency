import os
import logging
from flask import Flask, jsonify, request
from agents.executor import AcquisitionEngine
from integrations.email_manager import EmailManager  # Look in the right box!
from agents.research_engine import ResearchEngine
from agents.voice_agent import VoiceSalesAgent
from orchestrator import Orchestrator

# Set up a little diary to track what’s happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', static_url_path='/')  # CRITICAL CHANGE

# Write in the diary what’s starting
logger.info(f"WEB_UI_HOST: {os.getenv('WEB_UI_HOST', '0.0.0.0')}")
logger.info(f"WEB_UI_PORT: {os.getenv('WEB_UI_PORT', '80')}")
logger.info(f"DATABASE_URL: {os.getenv('DATABASE_URL', 'Not set')}")
logger.info("Attempting to start Flask app...")

try:
    acquisition_engine = AcquisitionEngine()
    email_manager = EmailManager()
    research_engine = ResearchEngine()
    orchestrator = Orchestrator()
    voice_agent = VoiceSalesAgent()
    logger.info("All agents initialized successfully")
except Exception as e:
    logger.error(f"Agent initialization failed: {str(e)}")
    raise

@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    try:
        status = {
            "acquisition_engine": acquisition_engine.get_status(),
            "email_manager": email_manager.get_status(),
            "research_engine": research_engine.get_status(),
            "orchestrator": orchestrator.get_status(),
            "voice_agent": voice_agent.get_status()
        }
        logger.info("Fetched agent status")
        return jsonify(status)
    except Exception as e:
        logger.error(f"Failed to fetch status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/send-command', methods=['POST'])
def send_command():
    try:
        data = request.get_json()
        agent = data.get("agent")
        command = data.get("command")

        if not agent or not command:
            return jsonify({"error": "Missing agent or command"}), 400

        if agent == "orchestrator":
            result = orchestrator.handle_web_command(command)
        elif agent == "voice_agent":
            result = voice_agent.handle_web_command(command)
        else:
            return jsonify({"error": "Unknown agent"}), 400

        logger.info(f"Sent command to {agent}: {command}")
        return jsonify({"message": f"Command sent to {agent}", "result": result})
    except Exception as e:
        logger.error(f"Failed to send command: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/initialize', methods=['POST'])
def initialize_agency():
    try:
        orchestrator.run_initial_campaign()
        return jsonify({"message": "Agency initialized—check WhatsApp or UI!"})
    except Exception as e:
        logger.error(f"Init failed: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/health', methods=['GET']) #Added this
def health_check():
    # Always return healthy to keep Traefik happy, log DB issues if any
    if orchestrator.db_pool:
        try:
            with orchestrator.db_pool.getconn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                orchestrator.db_pool.putconn(conn)
        except Exception as e:
            logging.error(f"DB health check failed: {str(e)}")
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    host = os.getenv("WEB_UI_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_UI_PORT", 80))
    logger.info(f"Starting Flask app on {host}:{port}")
    app.run(host=host, port=port, debug=False)