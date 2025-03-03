# web_interface/backend/app.py
import os
import logging
from flask import Flask, jsonify, request
from agents.executor import AcquisitionEngine
from agents.email_manager import EmailManager
from agents.research_engine import ResearchEngine
from agents.voice_agent import VoiceSalesAgent  # Added Voice Agent
from orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

acquisition_engine = AcquisitionEngine()
email_manager = EmailManager()
research_engine = ResearchEngine()
orchestrator = Orchestrator()
voice_agent = VoiceSalesAgent()  # Added Voice Agent instance

@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    try:
        status = {
            "acquisition_engine": acquisition_engine.get_status(),
            "email_manager": email_manager.get_status(),
            "research_engine": research_engine.get_status(),
            "orchestrator": orchestrator.get_status(),
            "voice_agent": voice_agent.get_status()  # Added Voice Agent status
        }
        logging.info("Fetched agent status.")
        return jsonify(status)
    except Exception as e:
        logging.error(f"Failed to fetch status: {str(e)}")
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
        logging.info(f"Sent command to {agent}: {command}")
        return jsonify({"message": f"Command sent to {agent}", "result": result})
    except Exception as e:
        logging.error(f"Failed to send command: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/initialize', methods=['POST'])
def initialize_agency():
    try:
        orchestrator.run_initial_campaign()
        return jsonify({"message": "Agency initialized—check WhatsApp or UI!"})
    except Exception as e:
        logging.error(f"Init failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host=os.getenv("WEB_UI_HOST", "0.0.0.0"), port=int(os.getenv("WEB_UI_PORT", 5000)), debug=False)