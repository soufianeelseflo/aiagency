import os
from flask import Flask, jsonify, request
from orchestrator import Orchestrator
from research_engine import ResearchEngine
from email_manager import EmailManager

# Initialize Flask app
app = Flask(__name__)

# Initialize agents
orchestrator = Orchestrator()
research_engine = ResearchEngine()
email_manager = EmailManager()

@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    """Fetch real-time status of all agents."""
    status = {
        "orchestrator": orchestrator.get_status(),
        "research_engine": research_engine.get_status(),
        "email_manager": email_manager.get_status(),
    }
    return jsonify(status)

@app.route('/api/update-agent', methods=['POST'])
def update_agent():
    """Update agent parameters via web interface."""
    data = request.json
    agent_name = data.get("agent")
    updates = data.get("updates", {})
    
    if agent_name == "orchestrator":
        orchestrator.update_parameters(updates)
    elif agent_name == "research_engine":
        research_engine.update_parameters(updates)
    elif agent_name == "email_manager":
        email_manager.update_parameters(updates)
    
    return jsonify({"message": f"{agent_name} updated successfully."})

if __name__ == "__main__":
    app.run(
        host=os.getenv("WEB_UI_HOST", "0.0.0.0"),
        port=int(os.getenv("WEB_UI_PORT", 443)),
        ssl_context='adhoc'  # Use HTTPS in production
    )