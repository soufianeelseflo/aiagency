from quart import Quart, render_template, request, jsonify
import asyncio
import logging
from agents.orchestrator import Orchestrator

app = Quart(__name__, template_folder='templates')
logger = logging.getLogger(__name__)
orchestrator = Orchestrator()

@app.route('/')
async def index():
    return await render_template('index.html')

@app.route('/api/start_simulated_call', methods=['POST'])
async def start_simulated_call():
    try:
        data = await request.get_json()
        client_id = data.get('client_id', 'test_client')
        result = await orchestrator.agents['voice_sales'].simulate_call(client_id)
        return jsonify({"status": "call started", "result": result})
    except Exception as e:
        logger.error(f"Simulated call error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_test_videos', methods=['GET'])
async def get_test_videos():
    try:
        industries = ["tech", "finance", "healthcare"]
        videos = await orchestrator.agents['browsing'].generate_test_videos(industries)
        return jsonify({"videos": videos})
    except Exception as e:
        logger.error(f"Video generation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/approve_agency', methods=['POST'])
async def approve_agency():
    try:
        orchestrator.approved = True
        await orchestrator.send_notification("Agency Approved", "Agency is now fully operational.")
        return jsonify({"status": "approved"})
    except Exception as e:
        logger.error(f"Approval error: {e}")
        return jsonify({"error": str(e)}), 500

# Moved from Orchestrator.setup_routes
@app.route('/start_testing', methods=['POST'])
async def start_testing():
    try:
        await orchestrator.start_testing_phase()
        return jsonify({"status": "testing started"})
    except Exception as e:
        logger.error(f"Start testing failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/status', methods=['GET'])
async def status():
    try:
        return jsonify({"approved": orchestrator.approved, "concurrency_limit": orchestrator.concurrency_limit})
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/suggest', methods=['POST'])
async def suggest():
    try:
        data = await request.get_json()
        suggestion = data.get('suggestion')
        if not suggestion:
            return jsonify({"error": "No suggestion provided"}), 400
        validation = await orchestrator.agents['think'].evaluate_suggestion(suggestion)
        if validation['approved']:
            await orchestrator.agents['think'].implement_suggestion(suggestion)
            logger.info(f"User suggestion applied: {suggestion}")
            return jsonify({"status": "suggestion accepted"})
        logger.warning(f"Suggestion rejected: {suggestion}")
        return jsonify({"status": "suggestion rejected", "reason": validation['reason']})
    except Exception as e:
        logger.error(f"Suggestion processing failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    asyncio.run(app.run(host='0.0.0.0', port=5000))
