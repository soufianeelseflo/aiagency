# Filename: ui/app.py
# Description: Quart backend for the AI Agency Dashboard.
# Version: 2.0 (Postgres KPI Integration)

import os
import asyncio
import logging
from quart import Quart, render_template, request, jsonify, send_file
from datetime import datetime, timedelta, timezone

# --- Assuming Orchestrator is initialized in main.py and accessible ---
# This is tricky. A better pattern is dependency injection or a shared context.
# For simplicity now, we might need to import and instantiate it here too,
# OR pass the session_maker from main.py. Let's assume we can get the
# session_maker from a global or passed-in orchestrator instance.
# This needs refinement based on how main.py manages the orchestrator.

# --- Placeholder for getting Orchestrator/SessionMaker ---
# This function needs to be implemented based on your main.py structure
# to get access to the initialized orchestrator or its session_maker.
async def get_orchestrator_instance():
    # In a real app, this might involve looking up a global instance
    # or using a proper application context/factory pattern.
    # For now, we'll try importing from main (might cause circular issues)
    # or assume it's passed somehow. THIS IS A LIKELY POINT OF FAILURE.
    try:
        # This direct import is generally bad practice, shows dependency issue
        from main import orchestrator_instance
        if orchestrator_instance:
            return orchestrator_instance
        else:
            # Attempt re-initialization if not running from main.py context
            # This is also problematic as it creates a second instance.
            logger.warning("Orchestrator instance not found from main. Trying secondary init (NOT RECOMMENDED).")
            from agents.orchestrator import Orchestrator
            return Orchestrator() # Creates a new instance - state will be separate!
    except ImportError:
        logger.error("Cannot import orchestrator_instance from main. UI cannot access agents/DB.")
        return None
    except Exception as e:
         logger.error(f"Error getting orchestrator instance: {e}")
         return None

# --- Model Imports (needed for queries) ---
from models import EmailLog, CallLog, Invoice, Client
from sqlalchemy import select, func, case

app = Quart(__name__, template_folder='templates')
logger = logging.getLogger(__name__)

@app.route('/')
async def index():
    """Serves the main dashboard HTML."""
    # ### Phase 5 Plan Ref: 13.2 (Update index.html serving)
    try:
        # Path relative to this file's location
        template_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
        if os.path.exists(template_path):
             with open(template_path, 'r') as f:
                  return f.read(), 200, {'Content-Type': 'text/html'}
        else:
             logger.error(f"UI Template not found at {template_path}")
             return "UI Template not found.", 404
    except Exception as e:
         logger.error(f"Error serving index.html: {e}", exc_info=True)
         return "Internal Server Error", 500

@app.route('/api/status_kpi', methods=['GET'])
async def get_status_and_kpi():
    """Fetches current status and key performance indicators from the database."""
    # ### Phase 5 Plan Ref: 13.1 (Implement KPI endpoint)
    orchestrator = await get_orchestrator_instance()
    if not orchestrator or not orchestrator.session_maker:
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
        # Get agent statuses
        if hasattr(orchestrator, 'agents'):
             agent_statuses = {name: agent.get_status_summary() for name, agent in orchestrator.agents.items() if hasattr(agent, 'get_status_summary')}
        # Get LLM status
        if hasattr(orchestrator, '_llm_client_status'):
             llm_status = {key: info['status'] for key, info in orchestrator._llm_client_status.items()}

        # Query Database for KPIs
        async with orchestrator.session_maker() as session:
            now = datetime.now(timezone.utc)
            one_day_ago = now - timedelta(days=1)

            # Email KPIs
            email_stmt = select(
                func.count(EmailLog.id).label("total"),
                func.sum(case((EmailLog.status == 'opened', 1), else_=0)).label("opened"),
                func.sum(case((EmailLog.status == 'responded', 1), else_=0)).label("responded")
            ).where(EmailLog.timestamp >= one_day_ago)
            email_res = await session.execute(email_stmt)
            email_stats = email_res.mappings().first()
            if email_stats:
                kpis["emails_sent_24h"] = email_stats.get("total", 0)
                # Note: Responded implies opened in EmailAgent logic
                kpis["emails_opened_24h"] = email_stats.get("opened", 0)
                kpis["emails_responded_24h"] = email_stats.get("responded", 0)

            # Call KPIs
            call_stmt = select(
                func.count(CallLog.id).label("total"),
                func.sum(case((CallLog.outcome.like('success%'), 1), else_=0)).label("success") # Count outcomes starting with 'success'
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
                func.count(case((Client.last_contacted_at >= one_day_ago, 1))).label("contacted_24h") # Count clients contacted recently
            ).where(Client.opt_in == True) # Count only active, opted-in clients
            client_res = await session.execute(client_stmt)
            client_stats = client_res.mappings().first()
            if client_stats:
                kpis["active_clients"] = client_stats.get("active_clients", 0)
                kpis["avg_client_score"] = round(float(client_stats.get("avg_score") or 0.0), 3)
                kpis["leads_contacted_24h"] = client_stats.get("contacted_24h", 0)

        return jsonify({
            "status": "success",
            "orchestrator_status": orchestrator.status if orchestrator else "unknown",
            "approved_for_operation": orchestrator.approved if orchestrator else False,
            "kpis": kpis,
            "agent_statuses": agent_statuses,
            "llm_client_status": llm_status
        })

    except Exception as e:
        logger.error(f"Error fetching status/KPIs: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch status: {e}"}), 500

# Add other API endpoints from the original ui/app.py if they are still needed
# (e.g., /api/approve_agency, /api/download_data) - Ensure they use the orchestrator instance correctly.

@app.route('/api/approve_agency', methods=['POST'])
async def approve_agency():
    orchestrator = await get_orchestrator_instance()
    if not orchestrator: return jsonify({"error": "Orchestrator unavailable"}), 503
    try:
        orchestrator.approved = True
        # Use orchestrator's notification method
        if hasattr(orchestrator, 'send_notification'):
            await orchestrator.send_notification("Agency Approved", "Agency is now fully operational.")
        else:
            logger.warning("Orchestrator has no send_notification method.")
        return jsonify({"status": "approved"})
    except Exception as e:
        logger.error(f"Approval error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/download_data', methods=['POST'])
async def download_data():
    # WARNING: This endpoint provides direct DB access. Secure appropriately.
    orchestrator = await get_orchestrator_instance()
    if not orchestrator: return jsonify({"error": "Orchestrator unavailable"}), 503

    data = await request.get_json()
    password = data.get('password')
    # Use password from settings
    if password != settings.DOWNLOAD_PASSWORD:
        logger.warning("Unauthorized attempt to download data.")
        return jsonify({"error": "Unauthorized"}), 401
    try:
        # Determine DB path/connection details from settings
        # For Postgres, we can't just send the file. We'd need to dump it.
        # This is complex and potentially insecure. Returning placeholder for now.
        # A better approach is specific API endpoints to export *data*, not the DB file.
        logger.warning("Direct database file download requested. This is insecure for Postgres. Endpoint needs redesign for data export.")
        # Example: Dump specific tables to CSV/JSON if needed
        return jsonify({"message": "Direct DB download not supported for Postgres. Use specific data export APIs."}), 501
        # db_path = '/app/database.db' # This is for SQLite, not Postgres
        # if not os.path.exists(db_path): raise FileNotFoundError("Database file not found")
        # return await send_file(db_path, as_attachment=True, download_name="agency_data.db")
    except Exception as e:
        logger.error(f"Data download failed: {e}")
        return jsonify({"error": str(e)}), 500

# Note: Removed test/debug endpoints like start_simulated_call, get_test_videos, test_osint_scrape
# Add them back if needed for specific testing purposes.

# Running the app is handled by main.py now
# if __name__ == "__main__":
#     # This won't run when executed via main.py
#     pass