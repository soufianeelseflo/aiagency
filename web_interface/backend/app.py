# web_interface/backend/app.py (Updated)
import os
import logging
from flask import Flask, jsonify, request
from agents.executor import AcquisitionEngine
from agents.email_manager import EmailManager
from agents.research_engine import ResearchEngine
from orchestrator import Orchestrator  # Assuming this is in the root dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

acquisition_engine = AcquisitionEngine()
email_manager = EmailManager()
research_engine = ResearchEngine()
orchestrator = Orchestrator()

@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    try:
        status = {
            "acquisition_engine": acquisition_engine.get_status(),
            "email_manager": email_manager.get_status(),
            "research_engine": research_engine.get_status(),
            "orchestrator": orchestrator.get_status()
        }
        logging.info("Fetched agent status.")
        return jsonify(status)
    except Exception as e:
        logging.error(f"Failed to fetch status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/update-agent', methods=['POST'])
def update_agent():
    try:
        data = request.get_json()
        agent_name = data.get("agent")
        updates = data.get("updates", {})
        if agent_name == "orchestrator":
            orchestrator.update_parameters(updates)
        elif agent_name == "email_manager":
            email_manager.update_parameters(updates)
        elif agent_name == "research_engine":
            research_engine.update_parameters(updates)
        else:
            return jsonify({"error": "Unknown agent"}), 400
        logging.info(f"Updated {agent_name} with {updates}")
        return jsonify({"message": f"{agent_name} updated successfully."})
    except Exception as e:
        logging.error(f"Failed to update agent: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/initialize', methods=['POST'])
def initialize_agency():
    try:
        client_request = {
            "client_id": "user_init",
            "content_type": "video",
            "tone": "casual",
            "platforms": ["social"],
            "budget": 5000,
            "company": "MyFirstClient"
        }
        content = orchestrator.route_ugc_request(client_request)
        logging.info("Agency initialized, first video sent via WhatsApp.")
        return jsonify({"message": "Agency initialized, check WhatsApp for first video link"})
    except Exception as e:
        logging.error(f"Initialization failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host=os.getenv("WEB_UI_HOST", "0.0.0.0"), port=int(os.getenv("WEB_UI_PORT", 5000)), debug=False)