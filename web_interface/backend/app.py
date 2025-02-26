import os
import logging
from flask import Flask, jsonify, request  # Ref: https://flask.palletsprojects.com/en/2.3.x/api/
from agents.executor import AcquisitionEngine
from agents.email_manager import EmailManager
from agents.research_engine import ResearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Initialize agents
acquisition_engine = AcquisitionEngine()
email_manager = EmailManager()
research_engine = ResearchEngine()

@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    """Fetch real-time status of all agents."""
    try:
        status = {
            "acquisition_engine": acquisition_engine.get_status(),
            "email_manager": email_manager.get_status(),
            "research_engine": research_engine.get_status(),
        }
        logging.info("Fetched agent status.")
        return jsonify(status)
    except Exception as e:
        logging.error(f"Failed to fetch agent status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/update-agent', methods=['POST'])
def update_agent():
    """Update agent parameters via web interface."""
    try:
        data = request.get_json()
        agent_name = data.get("agent")
        updates = data.get("updates", {})
        
        if agent_name == "acquisition_engine":
            acquisition_engine.update_parameters(updates)
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

if __name__ == "__main__":
    app.run(
        host=os.getenv("WEB_UI_HOST", "0.0.0.0"),
        port=int(os.getenv("WEB_UI_PORT", 5000)),  # Matches your env
        debug=False
    )