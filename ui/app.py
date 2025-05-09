# Filename: ui/app.py
# Description: Quart backend for the AI Agency Dashboard, handling API requests.
# Version: 3.1 (Production Ready - Aligned with Orchestrator v4.2)

import os
import asyncio
import logging
import json
import csv
import io
import re
from quart import Quart, render_template, request, jsonify, send_file, Response, current_app
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import TYPE_CHECKING, Callable, Any, Coroutine, Dict, Optional, List, Tuple

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator # For type hinting
    from sqlalchemy.ext.asyncio import AsyncSession # For type hinting

# --- Model Imports ---
try:
    from models import (
        EmailLog, CallLog, Invoice, Client, Task,
        AccountCredentials, StrategicDirective, KnowledgeFragment,
        FinancialTransaction, Base, PaymentStatus, InteractionType
    )
    from sqlalchemy import select, func, case, desc, text, and_
    MODELS_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).critical(f"CRITICAL: Failed to import SQLAlchemy models in ui/app.py: {e}. KPIs and data export will be severely limited.")
    class EmailLog: pass; class CallLog: pass; class Invoice: pass; class Client: pass; class Task: pass;
    class AccountCredentials: pass; class StrategicDirective: pass; class KnowledgeFragment: pass;
    class FinancialTransaction: pass; class Base: pass; class PaymentStatus(enum.Enum): COMPLETED="completed"; PAID="paid"; class InteractionType: pass # type: ignore
    select, func, case, desc, text, and_ = (None,)*6 # type: ignore
    MODELS_AVAILABLE = False

# --- Settings and Database Utilities Import ---
try:
    from config.settings import settings
    from utils.database import get_session # Assuming get_session yields an AsyncSession from orchestrator's session_maker
    SETTINGS_DB_UTILS_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).critical(f"CRITICAL: Failed to import settings or DB utils in ui/app.py: {e}. App may not function correctly.")
    class DummySettings: # type: ignore
        DOWNLOAD_PASSWORD = "change_this_in_env_critical_fallback"
        DEBUG = False
        def get(self, key, default=None): return default
        def get_secret(self, key): return None
    settings = DummySettings() # type: ignore
    async def get_session(): # type: ignore
        logger.error("Dummy get_session called; DB utils likely failed to import.")
        yield None
    SETTINGS_DB_UTILS_AVAILABLE = False


logger = logging.getLogger("ui_app")
op_logger = logging.getLogger('OperationalLog')

# --- Helper Decorator for Orchestrator Access ---
def ensure_orchestrator(f: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Decorator to ensure the orchestrator instance is available via current_app."""
    @wraps(f)
    async def decorated_function(*args: Any, **kwargs: Any) -> Any:
        # Access orchestrator instance attached to the Quart app object by Orchestrator itself
        orchestrator: Optional['Orchestrator'] = getattr(current_app, 'orchestrator_instance', None)
        if orchestrator is None:
            logger.error("Orchestrator instance not found in current_app. UI actions will fail.")
            op_logger.error("UI Endpoint called but Orchestrator not available.")
            return jsonify({"error": "Orchestrator service unavailable. Please check system logs."}), 503
        return await f(orchestrator, *args, **kwargs)
    return decorated_function


async def render_template_safe(template_name: str, **context: Any) -> Tuple[str, int, Dict[str, str]]:
    """Safely renders a template, handling potential errors."""
    try:
        # Orchestrator v4.2 sets template_folder on its app instance.
        # render_template should find it if called within app context.
        return await render_template(template_name, **context), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error rendering template {template_name}: {e}", exc_info=True)
        op_logger.error(f"Template rendering error for {template_name}: {e}")
        error_html = f"<html><head><title>Error</title></head><body><h1>Internal Server Error</h1><p>Sorry, an error occurred while rendering the page: {template_name}. Please check server logs.</p></body></html>"
        return error_html, 500, {'Content-Type': 'text/html; charset=utf-8'}

# --- Route Handler Functions ---

async def index_route_handler() -> Tuple[str, int, Dict[str, str]]:
    """Serves the main dashboard HTML. This handler is imported by Orchestrator."""
    logger.debug("Index route '/' requested via ui.app handler.")
    return await render_template_safe('index.html')

@ensure_orchestrator
async def get_status_and_kpi_handler(orchestrator: 'Orchestrator') -> Response:
    """Fetches current status and key performance indicators."""
    logger.debug("API /api/status_kpi requested.")
    response_data: Dict[str, Any] = {
        "approved_for_operation": orchestrator.approved, # Direct attribute access from Orchestrator v4.2
        "orchestrator_status": orchestrator.status, # Direct attribute access
        "llm_client_status": {},
        "database_status": "UNKNOWN",
        "proxy_provider_status": "UNKNOWN",
        "agent_statuses": {},
        "kpis": {},
    }

    # LLM Status from Orchestrator
    if hasattr(orchestrator, '_llm_client_status'): # Orchestrator v4.2 uses _llm_client_status
        response_data["llm_client_status"] = orchestrator._llm_client_status
    else:
        response_data["llm_client_status"] = {"status": "unavailable", "reason": "Status method not found on orchestrator"}

    # Agent Statuses from Orchestrator
    if hasattr(orchestrator, 'agents') and isinstance(orchestrator.agents, dict):
        agent_statuses_raw = {name: agent.get_status_summary() for name, agent in orchestrator.agents.items() if hasattr(agent, 'get_status_summary')}
        response_data["agent_statuses"] = agent_statuses_raw
    else:
        response_data["agent_statuses"] = {"error": "Agent status unavailable"}
    
    # Proxy Provider Status from Orchestrator
    if hasattr(orchestrator, '_proxy_list'): # Orchestrator v4.2 uses _proxy_list
        if orchestrator._proxy_list:
             # Simplified status: if list exists and has items, assume 'available' for UI. Deeper health is internal.
            response_data["proxy_provider_status"] = "available" if any(p.get("status") != "banned" for p in orchestrator._proxy_list) else "all_banned_or_empty"
        else:
            response_data["proxy_provider_status"] = "not_configured_or_empty"
    else:
        response_data["proxy_provider_status"] = "status_method_unavailable"


    if not MODELS_AVAILABLE or not SETTINGS_DB_UTILS_AVAILABLE or not orchestrator.session_maker:
        response_data["database_status"] = "UNAVAILABLE_DEPENDENCIES"
        response_data["kpis"] = {"error": "Database dependent KPIs cannot be calculated due to missing models, settings, or session_maker."}
        logger.warning("KPI calculation skipped due to unavailable models, settings, or session_maker.")
        return jsonify(response_data)

    try:
        async with orchestrator.session_maker() as session: # Use orchestrator's session_maker
            if session is None: # Should not happen if session_maker is valid
                raise Exception("Failed to acquire database session from orchestrator.")

            response_data["database_status"] = "CONNECTED"
            now_utc = datetime.now(timezone.utc)
            one_day_ago = now_utc - timedelta(days=1)

            # Total Profit
            total_profit_stmt = select(func.sum(FinancialTransaction.amount)).where(
                FinancialTransaction.status == PaymentStatus.COMPLETED,
                FinancialTransaction.type == 'payment'
            )
            response_data["kpis"]["total_profit"] = (await session.execute(total_profit_stmt)).scalar_one_or_none() or 0.0

            # Emails Sent (24h)
            emails_sent_stmt = select(func.count(EmailLog.id)).where(EmailLog.timestamp >= one_day_ago)
            response_data["kpis"]["emails_sent_24h"] = (await session.execute(emails_sent_stmt)).scalar_one() or 0

            # Emails Opened (24h)
            emails_opened_stmt = select(func.count(EmailLog.id)).where(
                EmailLog.timestamp >= one_day_ago, EmailLog.status == 'opened'
            )
            response_data["kpis"]["emails_opened_24h"] = (await session.execute(emails_opened_stmt)).scalar_one() or 0

            # Emails Responded (24h)
            emails_responded_stmt = select(func.count(EmailLog.id)).where(
                EmailLog.timestamp >= one_day_ago, EmailLog.status == 'responded'
            )
            response_data["kpis"]["emails_responded_24h"] = (await session.execute(emails_responded_stmt)).scalar_one() or 0

            # Calls Made (24h)
            calls_made_stmt = select(func.count(CallLog.id)).where(CallLog.timestamp >= one_day_ago)
            response_data["kpis"]["calls_made_24h"] = (await session.execute(calls_made_stmt)).scalar_one() or 0

            # Calls Successful (24h)
            calls_success_stmt = select(func.count(CallLog.id)).where(
                CallLog.timestamp >= one_day_ago, CallLog.outcome.like('success%') # Match outcomes starting with 'success'
            )
            response_data["kpis"]["calls_success_24h"] = (await session.execute(calls_success_stmt)).scalar_one() or 0

            # Active Clients
            active_clients_stmt = select(func.count(Client.id)).where(Client.client_status == ClientStatus.ACTIVE_SERVICE) # Example: using ClientStatus enum
            response_data["kpis"]["active_clients"] = (await session.execute(active_clients_stmt)).scalar_one() or 0
            
            # Average Client Score
            avg_client_score_stmt = select(func.avg(Client.client_score)).where(Client.client_score != None)
            response_data["kpis"]["avg_client_score"] = round(float((await session.execute(avg_client_score_stmt)).scalar_one_or_none() or 0.0), 3)

            # Leads Contacted (24h) - Example: count clients whose last_interaction was in last 24h
            leads_contacted_stmt = select(func.count(Client.id)).where(Client.last_interaction >= one_day_ago)
            response_data["kpis"]["leads_contacted_24h"] = (await session.execute(leads_contacted_stmt)).scalar_one() or 0


    except Exception as e:
        logger.error(f"Error fetching KPIs or DB status: {e}", exc_info=True)
        op_logger.error(f"KPI/DB Status Error: {e}")
        response_data["database_status"] = f"ERROR: {str(e)[:100]}"
        response_data["kpis"] = {"error": f"DB error prevented KPI calculation: {str(e)[:100]}"}

    return jsonify(response_data)

@ensure_orchestrator
async def approve_agency_handler(orchestrator: 'Orchestrator') -> Response:
    """Handles the request to approve the agency for full operation."""
    logger.info("API /api/approve_agency POST request received.")
    try:
        if orchestrator.approved:
             return jsonify({"status": "already_approved", "message": "Agency already approved."}), 200

        await orchestrator.approve_for_operation() # Call the method
        op_logger.info("Agency operation manually APPROVED via API.")
        return jsonify({"status": "approved", "message": "Agency operation approved successfully."})
    except Exception as e:
        logger.error(f"Error approving agency operation: {e}", exc_info=True)
        op_logger.error(f"Agency Approval Error: {e}")
        return jsonify({"status": "error", "message": f"Failed to approve agency: {str(e)}"}), 500

@ensure_orchestrator
async def export_data_handler(orchestrator: 'Orchestrator') -> Response:
    """Handles secure data export requests (e.g., Client list to CSV)."""
    logger.info("API /api/export_data POST request received.")
    if not SETTINGS_DB_UTILS_AVAILABLE or not MODELS_AVAILABLE or not orchestrator.session_maker:
         return jsonify({"error": "Data export unavailable due to missing core components or session_maker."}), 503

    try:
        data = await request.get_json()
        if not data: return jsonify({"error": "Missing JSON payload"}), 400

        password = data.get('password')
        export_type = data.get('type', 'clients')

        expected_password = settings.DOWNLOAD_PASSWORD
        if not expected_password or expected_password == "changeme123": # Add check for default password
            logger.error("DOWNLOAD_PASSWORD is not set or is default. Export aborted for security.")
            op_logger.error("Data export attempted but DOWNLOAD_PASSWORD not set/default.")
            return jsonify({"error": "Server configuration error: download password not securely set."}), 500

        if password != expected_password:
            logger.warning(f"Failed export attempt for type '{export_type}' due to incorrect password.")
            return jsonify({"error": "Invalid password"}), 403

        op_logger.info(f"Data export approved for type: {export_type}")

        output = io.StringIO()
        writer = csv.writer(output)
        filename = f"{export_type}_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
        
        headers: List[str] = []
        rows: List[List[Any]] = []

        async with orchestrator.session_maker() as session:
            if session is None: return jsonify({"error": "Database connection failed for export."}), 503

            if export_type == 'clients':
                headers = [c.name for c in Client.__table__.columns] if hasattr(Client, '__table__') else []
                if not headers: return jsonify({"error": "Client model structure not found."}), 500
                stmt = select(Client)
                result = await session.execute(stmt)
                for client_obj in result.scalars().all():
                    rows.append([getattr(client_obj, col, None) for col in headers])
            elif export_type == 'tasks':
                headers = [c.name for c in Task.__table__.columns] if hasattr(Task, '__table__') else []
                if not headers: return jsonify({"error": "Task model structure not found."}), 500
                stmt = select(Task).order_by(desc(Task.created_at)).limit(5000)
                result = await session.execute(stmt)
                for task_obj in result.scalars().all():
                     rows.append([getattr(task_obj, col, None) for col in headers])
            elif export_type == 'financials':
                headers = [c.name for c in FinancialTransaction.__table__.columns] if hasattr(FinancialTransaction, '__table__') else []
                if not headers: return jsonify({"error": "FinancialTransaction model structure not found."}), 500
                stmt = select(FinancialTransaction).order_by(desc(FinancialTransaction.timestamp)).limit(5000)
                result = await session.execute(stmt)
                for ft_obj in result.scalars().all():
                    rows.append([getattr(ft_obj, col, None) for col in headers])
            else:
                return jsonify({"error": "Invalid export type specified."}), 400
        
        if not rows and headers: # Headers exist but no data
            writer.writerow(headers) # Write header even if no data
        elif rows:
            writer.writerow(headers)
            writer.writerows(rows)
        else: # No headers and no rows (should be caught by model structure check)
            return jsonify({"error": f"No data or structure found for export type '{export_type}'."}), 404

        output.seek(0)
        return await send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv', as_attachment=True, attachment_filename=filename
        )
    except Exception as e:
        logger.error(f"Error exporting data: {e}", exc_info=True)
        op_logger.error(f"Data Export Error: {e}")
        return jsonify({"error": f"An error occurred during export: {str(e)}"}), 500

@ensure_orchestrator
async def submit_feedback_handler(orchestrator: 'Orchestrator') -> Response:
    """Handles feedback submitted from the UI."""
    logger.info("API /api/submit_feedback POST request received.")
    try:
        data = await request.get_json()
        if not data or 'feedback' not in data:
            return jsonify({"error": "Missing 'feedback' in JSON payload"}), 400
        
        feedback_text = str(data['feedback']).strip()
        if not feedback_text:
            return jsonify({"error": "Feedback content cannot be empty"}), 400

        feedback_task = {
            "action": "process_external_feedback",
            "content": { # Ensure content key exists for ThinkTool
                "feedback_data": {
                    "source": "UI_Dashboard_v3.1",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "content": feedback_text
                }
            },
            "priority": 5 # Example priority
        }
        asyncio.create_task(orchestrator.delegate_task("ThinkTool", feedback_task))
        op_logger.info(f"Feedback/Directive submitted via UI: '{feedback_text[:100]}...'")
        return jsonify({"status": "success", "message": "Feedback submitted for processing."}), 202
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        op_logger.error(f"UI Feedback Submission Error: {e}")
        return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500

@ensure_orchestrator
async def test_voice_call_handler(orchestrator: 'Orchestrator') -> Response:
    """Handles request to initiate a test voice call."""
    logger.info("API /api/test_voice_call POST request received.")
    try:
        data = await request.get_json()
        if not data or 'phone_number' not in data:
            return jsonify({"error": "Missing 'phone_number' in JSON payload"}), 400
        
        phone_number = str(data['phone_number']).strip()
        if not re.match(r"^\+[1-9]\d{1,14}$", phone_number):
             return jsonify({"error": "Invalid phone number format. Use E.164 (e.g., +12223334444)."}), 400

        test_call_task = {
            "action": "initiate_test_call",
            "phone_number": phone_number # VoiceSalesAgent expects this directly
        }
        asyncio.create_task(orchestrator.delegate_task("VoiceSalesAgent", test_call_task))
        op_logger.info(f"Test voice call initiated via UI to: {phone_number}")
        return jsonify({"status": "success", "message": f"Test call initiation requested for {phone_number}."}), 202
    except Exception as e:
        logger.error(f"Error initiating test voice call: {e}", exc_info=True)
        op_logger.error(f"UI Test Call Error: {e}")
        return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500

@ensure_orchestrator
async def generate_videos_handler(orchestrator: 'Orchestrator') -> Response:
    """Handles request to trigger the complex video generation workflow."""
    logger.info("API /api/generate_videos POST request received.")
    try:
        data = await request.get_json()
        if not data: return jsonify({"error": "Missing JSON payload"}), 400
        
        topic_keywords = str(data.get('topic_keywords', '')).strip()
        video_count = int(data.get('video_count', 1))

        if not topic_keywords: return jsonify({"error": "Missing 'topic_keywords'."}), 400
        if not (1 <= video_count <= 5): return jsonify({"error": "Video count must be between 1 and 5."}), 400

        user_email_for_videos = settings.get('USER_EMAIL')
        if not user_email_for_videos:
            logger.error("USER_EMAIL not configured. Cannot determine where to send generated videos.")
            op_logger.error("Video generation requested, but USER_EMAIL for delivery is missing.")
            return jsonify({"error": "Operator email (USER_EMAIL) not configured for video delivery."}), 500

        video_workflow_task = {
            "action": "initiate_video_generation_workflow",
            "params": {
                "topic_keywords": topic_keywords,
                "count": video_count,
                "goal": f"Generate {video_count} sample UGC videos on '{topic_keywords}' and email to operator ({user_email_for_videos})."
                # ThinkTool will add recipient_email to the final notification step in its plan
            }
        }
        asyncio.create_task(orchestrator.delegate_task("ThinkTool", video_workflow_task))
        op_logger.info(f"Sample video generation requested via UI: '{topic_keywords}', count: {video_count} for {user_email_for_videos}")
        return jsonify({"status": "success", "message": "Video generation workflow initiated."}), 202
    except ValueError:
        return jsonify({"error": "Invalid 'video_count'. Must be an integer."}), 400
    except Exception as e:
        logger.error(f"Error in generate_sample_videos: {e}", exc_info=True)
        op_logger.error(f"UI Video Generation Error: {e}")
        return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500

# --- Function to register all routes with the Quart app ---
def register_ui_routes(app: Quart, orchestrator_instance_ref: 'Orchestrator'):
    """Registers all UI route handlers with the Quart app instance."""
    # Attach orchestrator to app context for handlers to use via @ensure_orchestrator
    app.orchestrator_instance = orchestrator_instance_ref # type: ignore

    # Note: Orchestrator v4.2 already registers '/' to index_route_handler.
    # If this file's index_route_handler is preferred, Orchestrator's setup_routes should use it.
    # Assuming Orchestrator v4.2 imports and uses this index_route_handler:
    # app.route('/')(index_route_handler) # This is handled by Orchestrator's import

    app.route('/api/status_kpi', methods=['GET'])(get_status_and_kpi_handler)
    app.route('/api/approve_agency', methods=['POST'])(approve_agency_handler)
    app.route('/api/export_data', methods=['POST'])(export_data_handler)
    app.route('/api/submit_feedback', methods=['POST'])(submit_feedback_handler)
    app.route('/api/test_voice_call', methods=['POST'])(test_voice_call_handler)
    app.route('/api/generate_videos', methods=['POST'])(generate_videos_handler)
    
    logger.info("UI API routes registered by ui.app.register_ui_routes.")

# --- End of ui/app.py ---