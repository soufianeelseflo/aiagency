# Filename: ui/app.py
# Description: Quart backend for the AI Agency Dashboard, handling API requests.
# Version: 2.2 (Verified Logic & Error Handling)

import os
import asyncio
import logging
import json # For potential data export
import csv # For potential data export
import io # For streaming CSV data
import re # For phone number validation

from quart import Quart, render_template, request, jsonify, send_file, Response
from datetime import datetime, timedelta, timezone

# --- Model Imports (needed for queries/export) ---
# Ensure models are accessible from the root directory or adjust path as needed
try:
    from models import EmailLog, CallLog, Invoice, Client, Base # Import Base for table mapping
    from sqlalchemy import select, func, case
    from sqlalchemy.ext.asyncio import AsyncSession
except ImportError as e:
    logging.critical(f"Failed to import models in ui/app.py: {e}. Check PYTHONPATH.")
    # Define dummy classes if import fails to allow basic app structure loading
    class EmailLog: pass
    class CallLog: pass
    class Invoice: pass
    class Client: pass
    class Base: pass
    AsyncSession = None # type: ignore

# --- Settings Import ---
# Assuming settings are validated and loaded correctly by main.py
try:
    from config.settings import settings
except ImportError as e:
    logging.critical(f"Failed to import settings in ui/app.py: {e}. Check PYTHONPATH.")
    # Define a dummy settings object if import fails
    class DummySettings:
        DOWNLOAD_PASSWORD = "dummy_password"
        def get(self, key, default=None): return default
        def get_secret(self, key): return None
    settings = DummySettings() # type: ignore

# --- Orchestrator Access ---
# We rely on the single instance created in main.py
# This function assumes 'main.py' defines 'orchestrator_instance' globally after creation.
# A more robust pattern might involve a dedicated application context or factory.
def get_orchestrator_instance_from_main():
    """Attempts to access the global orchestrator instance from main.py"""
    try:
        # Attempt to import the instance directly from main
        # This relies on main.py having run and defined the global variable
        from main import orchestrator_instance
        if orchestrator_instance:
            return orchestrator_instance
        else:
            # This should ideally not happen if main.py runs correctly
            logger.error("Orchestrator instance is None in main.py.")
            return None
    except ImportError:
        logger.error("Could not import orchestrator_instance from main. Circular dependency or main not run?")
        return None
    except Exception as e:
        logger.error(f"Error accessing orchestrator instance from main: {e}", exc_info=True)
        return None

# --- Application Setup ---
# The Quart app object is now created and managed by the Orchestrator instance in main.py.
# We will retrieve it via the orchestrator instance when needed for route definition,
# but we need a way to register routes *before* the server starts.
# Solution: Define routes using a Blueprint or attach them directly in Orchestrator.__init__.
# For simplicity here, we'll assume routes are attached to the app instance obtained from the orchestrator later.
# This means this file primarily contains route *handler* functions
# that are registered elsewhere (e.g., in Orchestrator or main.py).

logger = logging.getLogger(__name__)

# --- Route Handler Functions ---

async def index_route_handler():
    """Serves the main dashboard HTML."""
    # Path relative to the project root where templates are expected
    try:
        # Assuming 'ui/templates' exists relative to project root
        template_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
        if os.path.exists(template_path):
             # Manual reading for now if render_template setup is complex:
             with open(template_path, 'r', encoding='utf-8') as f:
                 return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
        else:
             logger.error(f"UI Template not found at {template_path}")
             return jsonify({"error": "UI Template not found."}), 404
    except Exception as e:
         logger.error(f"Error serving index.html: {e}", exc_info=True)
         return jsonify({"error": "Internal Server Error rendering UI."}), 500

async def get_status_and_kpi_handler():
    """Fetches current status and key performance indicators from the database."""
    orchestrator = get_orchestrator_instance_from_main()
    if not orchestrator or not orchestrator.session_maker:
        logger.error("API Error: Orchestrator or session maker unavailable for status/KPI.")
        # Return 503 Service Unavailable
        return jsonify({"error": "Orchestrator or session maker unavailable"}), 503

    kpis = {
        "emails_sent_24h": 0,
        "emails_opened_24h": 0,
        "emails_responded_24h": 0,
        "calls_made_24h": 0,
        "calls_success_24h": 0,
        "total_profit": 0.0,
        "avg_client_score": 0.0,
        "active_clients": 0,
        "leads_contacted_24h": 0,
    }
    agent_statuses = {}
    llm_status = {}

    try:
        # Get agent statuses from the running orchestrator
        if hasattr(orchestrator, 'agents') and isinstance(orchestrator.agents, dict):
            agent_statuses = {name: agent.get_status_summary() for name, agent in orchestrator.agents.items() if hasattr(agent, 'get_status_summary') and callable(agent.get_status_summary)}
        # Get LLM status from the running orchestrator
        if hasattr(orchestrator, '_llm_client_status') and isinstance(orchestrator._llm_client_status, dict):
            # Adapt based on actual structure in Orchestrator v3.2
            llm_status = {"primary": orchestrator._llm_client_status.get("status", "unknown")}

        # Query Database for KPIs using the orchestrator's session maker
        async with orchestrator.session_maker() as session:
            now_utc = datetime.now(timezone.utc)
            one_day_ago = now_utc - timedelta(days=1)

            # Email KPIs
            email_stmt = select(
                func.count(EmailLog.id).label("total"),
                # Count distinct opens/responses if needed, otherwise simple status check
                func.sum(case((EmailLog.status.in_(['opened', 'responded']), 1), else_=0)).label("opened"),
                func.sum(case((EmailLog.status == 'responded', 1), else_=0)).label("responded")
            ).where(EmailLog.timestamp >= one_day_ago)
            email_res = await session.execute(email_stmt)
            email_stats = email_res.mappings().first()
            if email_stats:
                kpis["emails_sent_24h"] = email_stats.get("total", 0)
                kpis["emails_opened_24h"] = email_stats.get("opened", 0)
                kpis["emails_responded_24h"] = email_stats.get("responded", 0)

            # Call KPIs
            call_stmt = select(
                func.count(CallLog.id).label("total"),
                func.sum(case((CallLog.outcome.like('success%'), 1), else_=0)).label("success")
            ).where(CallLog.timestamp >= one_day_ago)
            call_res = await session.execute(call_stmt)
            call_stats = call_res.mappings().first()
            if call_stats:
                kpis["calls_made_24h"] = call_stats.get("total", 0)
                kpis["calls_success_24h"] = call_stats.get("success", 0)

            # Total Profit (from paid invoices)
            profit_stmt = select(func.sum(Invoice.amount)).where(Invoice.status == 'paid')
            profit_res = await session.execute(profit_stmt)
            kpis["total_profit"] = round(float(profit_res.scalar() or 0.0), 2)

            # Client KPIs
            client_stmt = select(
                func.count(Client.id).label("active_clients"),
                func.avg(Client.engagement_score).label("avg_score"),
                func.count(case((Client.last_contacted_at >= one_day_ago, 1))).label("contacted_24h")
            ).where(Client.opt_in == True) # Count only active, opted-in clients
            client_res = await session.execute(client_stmt)
            client_stats = client_res.mappings().first()
            if client_stats:
                kpis["active_clients"] = client_stats.get("active_clients", 0)
                kpis["avg_client_score"] = round(float(client_stats.get("avg_score") or 0.0), 3)
                kpis["leads_contacted_24h"] = client_stats.get("contacted_24h", 0)

        return jsonify({
            "status": "success",
            "orchestrator_status": orchestrator.status if orchestrator and hasattr(orchestrator, 'status') else "unknown",
            "approved_for_operation": orchestrator.approved if orchestrator and hasattr(orchestrator, 'approved') else False,
            "kpis": kpis,
            "agent_statuses": agent_statuses,
            "llm_client_status": llm_status
        })

    except Exception as e:
        logger.error(f"Error fetching status/KPIs: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch status: {e}"}), 500

async def approve_agency_handler():
    """Handles the request to approve the agency for full operation."""
    orchestrator = get_orchestrator_instance_from_main()
    if not orchestrator: return jsonify({"error": "Orchestrator unavailable"}), 503
    try:
        if orchestrator.approved:
             return jsonify({"status": "already_approved", "message": "Agency already approved."}), 200

        orchestrator.approved = True
        logger.info("!!! AGENCY APPROVED FOR FULL OPERATION via API !!!")
        # Use orchestrator's notification method
        if hasattr(orchestrator, 'send_notification') and callable(orchestrator.send_notification):
            # Run notification in background to avoid blocking response
            asyncio.create_task(orchestrator.send_notification("Agency Approved", "Agency is now fully operational via UI."))
        else:
            logger.warning("Orchestrator has no send_notification method.")
        return jsonify({"status": "approved"})
    except Exception as e:
        logger.error(f"Approval error: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error during approval: {str(e)}"}), 500

async def export_data_handler():
    """Handles secure data export requests (e.g., Client list to CSV)."""
    orchestrator = get_orchestrator_instance_from_main()
    if not orchestrator or not orchestrator.session_maker:
        return jsonify({"error": "Orchestrator or session maker unavailable"}), 503

    try:
        data = await request.get_json()
        if not data: return jsonify({"error": "Missing request body"}), 400
        password = data.get('password')
        export_type = data.get('type', 'clients') # Default to exporting clients

        # Use password from settings for authentication
        if password != settings.DOWNLOAD_PASSWORD:
            logger.warning("Unauthorized attempt to download data.")
            return jsonify({"error": "Unauthorized"}), 401

        if export_type == 'clients':
            # Export Client data to CSV
            output = io.StringIO()
            writer = csv.writer(output)

            async with orchestrator.session_maker() as session:
                stmt = select(Client) # Select all columns for now, refine if needed
                result = await session.execute(stmt)
                clients = result.scalars().all()

                if not clients:
                    return jsonify({"message": "No client data found to export."}), 404

                # Write header row (using model attributes)
                header = [c.name for c in Client.__table__.columns]
                writer.writerow(header)

                # Write data rows
                for client in clients:
                    writer.writerow([getattr(client, col) for col in header])

            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment;filename=clients_export.csv"}
            )
        # Add other export types here (e.g., 'logs', 'kpis_json')
        # elif export_type == 'logs': ... query logs ... return JSON/CSV ...
        else:
            return jsonify({"error": f"Unsupported export type: {export_type}"}), 400

    except Exception as e:
        logger.error(f"Data export failed: {e}", exc_info=True)
        return jsonify({"error": f"Data export failed: {str(e)}"}), 500

async def submit_feedback_handler():
    """Handles feedback submitted from the UI."""
    orchestrator = get_orchestrator_instance_from_main()
    if not orchestrator: return jsonify({"error": "Orchestrator unavailable"}), 503

    try:
        data = await request.get_json()
        if not data: return jsonify({"error": "Missing request body"}), 400
        feedback_text = data.get('feedback')

        if not feedback_text or not isinstance(feedback_text, str):
            return jsonify({"error": "Invalid or missing 'feedback' text."}), 400

        logger.info(f"Received feedback from UI: {feedback_text[:100]}...")

        # Delegate feedback processing to ThinkTool
        feedback_task = {
            "action": "process_external_feedback", # Define this action for ThinkTool
            "feedback_data": {
                "source": "UI_Dashboard",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": feedback_text
            },
            "description": "Process feedback submitted via UI Dashboard"
        }
        # Run task delegation in background
        if hasattr(orchestrator, 'delegate_task') and callable(orchestrator.delegate_task):
            asyncio.create_task(orchestrator.delegate_task("ThinkTool", feedback_task))
            return jsonify({"status": "success", "message": "Feedback submitted for processing."}), 202 # Accepted
        else:
            logger.error("Orchestrator cannot delegate task for feedback.")
            return jsonify({"error": "Internal error: Cannot delegate feedback task."}), 500

    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error submitting feedback: {str(e)}"}), 500

async def test_voice_call_handler():
    """Handles request to initiate a test voice call."""
    orchestrator = get_orchestrator_instance_from_main()
    if not orchestrator: return jsonify({"error": "Orchestrator unavailable"}), 503

    try:
        data = await request.get_json()
        if not data: return jsonify({"error": "Missing request body"}), 400
        phone_number = data.get('phone_number')

        if not phone_number or not isinstance(phone_number, str):
            return jsonify({"error": "Invalid or missing 'phone_number'."}), 400

        # Basic phone number validation (improve as needed)
        if not re.match(r"^\+?[1-9]\d{1,14}$", phone_number):
             return jsonify({"error": "Invalid phone number format (should be E.164 like +15551234567)."}), 400

        logger.info(f"Received request to initiate test call to: {phone_number}")

        # Create a specific task for VoiceAgent like 'initiate_test_call'
        test_call_task = {
            "action": "initiate_test_call", # Define this action for VoiceSalesAgent
            "phone_number": phone_number,
            "description": f"Initiate test call to {phone_number} from UI"
        }
        # Run task delegation in background
        if hasattr(orchestrator, 'delegate_task') and callable(orchestrator.delegate_task):
            asyncio.create_task(orchestrator.delegate_task("VoiceSalesAgent", test_call_task))
            return jsonify({"status": "success", "message": f"Test call initiation requested for {phone_number}."}), 202 # Accepted
        else:
            logger.error("Orchestrator cannot delegate task for test call.")
            return jsonify({"error": "Internal error: Cannot delegate test call task."}), 500

    except Exception as e:
        logger.error(f"Error initiating test call: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error initiating test call: {str(e)}"}), 500

async def generate_videos_handler():
    """Handles request to trigger the complex video generation workflow."""
    orchestrator = get_orchestrator_instance_from_main()
    if not orchestrator: return jsonify({"error": "Orchestrator unavailable"}), 503

    try:
        # Get parameters from request if needed (e.g., topic, style)
        # data = await request.get_json()
        # video_topic = data.get('topic', 'Default UGC Video')
        # num_videos = data.get('count', 3)

        logger.info(f"Received request to generate videos.")

        # Delegate the high-level task to ThinkTool for planning and execution
        video_workflow_task = {
            "action": "initiate_video_generation_workflow", # Define this action for ThinkTool
            "params": { # Pass any relevant parameters from the UI
                "count": 3, # Example: generate 3 videos
                "goal": "Generate sample UGC videos and email to operator"
            },
            "description": "Initiate video generation workflow requested via UI"
        }
        # Run task delegation in background
        if hasattr(orchestrator, 'delegate_task') and callable(orchestrator.delegate_task):
            asyncio.create_task(orchestrator.delegate_task("ThinkTool", video_workflow_task))
            return jsonify({"status": "success", "message": "Video generation workflow initiated."}), 202 # Accepted
        else:
            logger.error("Orchestrator cannot delegate task for video generation.")
            return jsonify({"error": "Internal error: Cannot delegate video generation task."}), 500

    except Exception as e:
        logger.error(f"Error initiating video generation: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error initiating video generation: {str(e)}"}), 500


# --- Helper to register routes with the app instance ---
# This should be called from Orchestrator.__init__ or main.py
def register_ui_routes(app: Quart):
    """Registers all UI route handlers with the Quart app instance."""
    app.route('/')(index_route_handler)
    app.route('/api/status_kpi', methods=['GET'])(get_status_and_kpi_handler)
    app.route('/api/approve_agency', methods=['POST'])(approve_agency_handler)
    app.route('/api/export_data', methods=['POST'])(export_data_handler) # Changed name
    app.route('/api/submit_feedback', methods=['POST'])(submit_feedback_handler)
    app.route('/api/test_voice_call', methods=['POST'])(test_voice_call_handler)
    app.route('/api/generate_videos', methods=['POST'])(generate_videos_handler)
    # Add registration for /hosted_audio/<path:filename> if not handled by Orchestrator directly
    # Example: app.route('/hosted_audio/<path:filename>')(serve_hosted_audio_handler) # Assuming handler is defined
    logger.info("UI API routes registered.")

# Note: This file itself doesn't run the app. It defines handlers
# that are registered and run by the app instance created in main.py.

# --- End of ui/app.py ---