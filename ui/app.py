# Filename: ui/app.py
# Description: Quart route handlers for the AI Agency Dashboard.
# Version: 3.0 (IGNIS Transmutation - Complete Rebuild, Robust Endpoints, No Placeholders)

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
    from agents.orchestrator import Orchestrator
    from sqlalchemy.ext.asyncio import AsyncSession

# --- Model Imports ---
# These are critical for database interactions
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
    # Define dummy classes if import fails to prevent full crash but functionality will be degraded
    class EmailLog: pass; class CallLog: pass; class Invoice: pass; class Client: pass; class Task: pass;
    class AccountCredentials: pass; class StrategicDirective: pass; class KnowledgeFragment: pass;
    class FinancialTransaction: pass; class Base: pass; class PaymentStatus: pass; class InteractionType: pass
    select, func, case, desc, text, and_ = (None,)*6 # type: ignore
    MODELS_AVAILABLE = False

# --- Settings and Database Utilities Import ---
try:
    from config.settings import settings
    from utils.database import get_session
    SETTINGS_DB_UTILS_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).critical(f"CRITICAL: Failed to import settings or DB utils in ui/app.py: {e}. App may not function correctly.")
    class DummySettings: # type: ignore
        DOWNLOAD_PASSWORD = "change_this_in_env"
        DEBUG = False
        def get(self, key, default=None): return default
        def get_secret(self, key): return None
    settings = DummySettings() # type: ignore
    async def get_session(): # type: ignore
        yield None # Dummy session
    SETTINGS_DB_UTILS_AVAILABLE = False


logger = logging.getLogger("ui_app")
op_logger = logging.getLogger('OperationalLog')

# --- Helper Decorator for Orchestrator Access ---
def ensure_orchestrator(f: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Decorator to ensure the orchestrator instance is available."""
    @wraps(f)
    async def decorated_function(*args: Any, **kwargs: Any) -> Any:
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
        # Assuming templates are in a 'templates' subdirectory relative to this file's location
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        return await render_template(template_name, template_folder=template_dir, **context), 200, {'Content-Type': 'text/html'}
    except Exception as e:
        logger.error(f"Error rendering template {template_name}: {e}", exc_info=True)
        op_logger.error(f"Template rendering error for {template_name}: {e}")
        # Fallback to a very basic HTML error page
        error_html = f"<html><head><title>Error</title></head><body><h1>Internal Server Error</h1><p>Sorry, an error occurred while rendering the page: {template_name}. Please check server logs.</p></body></html>"
        return error_html, 500, {'Content-Type': 'text/html'}

# --- Route Definitions ---
def add_ui_routes(app: Quart, orchestrator_instance_ref: 'Orchestrator') -> None:
    """
    Adds all UI-related routes to the Quart application.
    `orchestrator_instance_ref` is stored on the app context for access by routes.
    """
    app.orchestrator_instance = orchestrator_instance_ref # type: ignore
    logger.info("UI routes initialized and orchestrator instance attached to app context.")

    @app.route('/')
    async def index() -> Tuple[str, int, Dict[str, str]]:
        logger.debug("Index route '/' requested.")
        return await render_template_safe('index.html')

    @app.route('/api/status_kpi', methods=['GET'])
    @ensure_orchestrator
    async def get_status_and_kpis(orchestrator: 'Orchestrator') -> Response:
        logger.debug("API /api/status_kpi requested.")
        response_data: Dict[str, Any] = {
            "approved_for_operation": orchestrator.is_approved_for_operation(),
            "orchestrator_status": orchestrator.get_status().get('overall_status', 'UNKNOWN'),
            "llm_client_status": {},
            "database_status": "UNKNOWN",
            "proxy_provider_status": "UNKNOWN",
            "agent_statuses": {},
            "kpis": {},
        }

        # LLM Status
        if hasattr(orchestrator, 'llm_clients_status_summary'):
            response_data["llm_client_status"] = await orchestrator.llm_clients_status_summary()
        else:
            response_data["llm_client_status"] = {"error": "Status unavailable"}

        # Agent Statuses
        agent_statuses_raw = orchestrator.get_all_agent_statuses()
        for agent_name, status_info in agent_statuses_raw.items():
            response_data["agent_statuses"][agent_name] = status_info # Already well-structured

        # Proxy Provider Status
        if hasattr(orchestrator, 'proxy_manager') and orchestrator.proxy_manager:
            response_data["proxy_provider_status"] = orchestrator.proxy_manager.get_status().get('status', 'NOT_CONFIGURED')
        elif settings.get('PROXY_PROVIDER_ENABLED', False):
             response_data["proxy_provider_status"] = "CONFIGURED_BUT_INACTIVE"
        else:
             response_data["proxy_provider_status"] = "DISABLED"


        # Database Status and KPIs
        if not MODELS_AVAILABLE or not SETTINGS_DB_UTILS_AVAILABLE:
            response_data["database_status"] = "UNAVAILABLE_MODELS_OR_SETTINGS"
            response_data["kpis"] = {"error": "Database dependent KPIs cannot be calculated."}
            logger.warning("KPI calculation skipped due to unavailable models or settings.")
            return jsonify(response_data)

        try:
            async with get_session() as db_session: # type: ignore
                if db_session is None:
                    raise Exception("Failed to acquire database session.")

                response_data["database_status"] = "CONNECTED"
                now_utc = datetime.now(timezone.utc)
                one_day_ago = now_utc - timedelta(days=1)

                # Total Profit (from successful FinancialTransactions related to Invoices)
                total_profit_stmt = select(func.sum(FinancialTransaction.amount)).where(
                    FinancialTransaction.status == PaymentStatus.COMPLETED, # type: ignore
                    FinancialTransaction.type == 'payment' # type: ignore
                )
                total_profit_result = await db_session.execute(total_profit_stmt)
                response_data["kpis"]["total_profit"] = total_profit_result.scalar_one_or_none() or 0.0

                # Emails Sent (24h)
                emails_sent_stmt = select(func.count(EmailLog.id)).where(EmailLog.timestamp >= one_day_ago) # type: ignore
                emails_sent_result = await db_session.execute(emails_sent_stmt)
                response_data["kpis"]["emails_sent_24h"] = emails_sent_result.scalar_one()

                # Emails Opened (24h) - Assuming a boolean 'opened' field or similar tracking
                emails_opened_stmt = select(func.count(EmailLog.id)).where( # type: ignore
                    EmailLog.timestamp >= one_day_ago, # type: ignore
                    EmailLog.status == 'opened' # type: ignore # Or some other 'opened' indicator
                )
                emails_opened_result = await db_session.execute(emails_opened_stmt)
                response_data["kpis"]["emails_opened_24h"] = emails_opened_result.scalar_one()

                # Emails Responded (24h) - Assuming 'responded' status
                emails_responded_stmt = select(func.count(EmailLog.id)).where( # type: ignore
                    EmailLog.timestamp >= one_day_ago, # type: ignore
                    EmailLog.status == 'responded' # type: ignore
                )
                emails_responded_result = await db_session.execute(emails_responded_stmt)
                response_data["kpis"]["emails_responded_24h"] = emails_responded_result.scalar_one()

                # Calls Made (24h)
                calls_made_stmt = select(func.count(CallLog.id)).where(CallLog.timestamp >= one_day_ago) # type: ignore
                calls_made_result = await db_session.execute(calls_made_stmt)
                response_data["kpis"]["calls_made_24h"] = calls_made_result.scalar_one()

                # Calls Successful (24h) - Assuming 'completed' or 'answered' status
                calls_success_stmt = select(func.count(CallLog.id)).where( # type: ignore
                    CallLog.timestamp >= one_day_ago, # type: ignore
                    CallLog.status.in_(['completed', 'answered', 'successful_pitch']) # type: ignore
                )
                calls_success_result = await db_session.execute(calls_success_stmt)
                response_data["kpis"]["calls_success_24h"] = calls_success_result.scalar_one()

                # Active Clients (e.g., opt_in_status = 'active')
                active_clients_stmt = select(func.count(Client.id)).where(Client.opt_in_status == 'active') # type: ignore
                active_clients_result = await db_session.execute(active_clients_stmt)
                response_data["kpis"]["active_clients"] = active_clients_result.scalar_one()
                
                # Average Client Score (assuming 'client_score' field on Client model)
                avg_client_score_stmt = select(func.avg(Client.client_score)).where(Client.client_score != None) # type: ignore
                avg_client_score_result = await db_session.execute(avg_client_score_stmt)
                response_data["kpis"]["avg_client_score"] = avg_client_score_result.scalar_one_or_none() or 0.0

        except Exception as e:
            logger.error(f"Error fetching KPIs or DB status: {e}", exc_info=True)
            op_logger.error(f"KPI/DB Status Error: {e}")
            response_data["database_status"] = f"ERROR: {str(e)[:100]}"
            response_data["kpis"] = {"error": f"DB error prevented KPI calculation: {str(e)[:100]}"}

        return jsonify(response_data)

    @app.route('/api/approve_agency', methods=['POST'])
    @ensure_orchestrator
    async def approve_agency_operation(orchestrator: 'Orchestrator') -> Response:
        logger.info("API /api/approve_agency POST request received.")
        try:
            await orchestrator.approve_for_operation()
            op_logger.info("Agency operation manually APPROVED via API.")
            return jsonify({"status": "approved", "message": "Agency operation approved successfully."})
        except Exception as e:
            logger.error(f"Error approving agency operation: {e}", exc_info=True)
            op_logger.error(f"Agency Approval Error: {e}")
            return jsonify({"status": "error", "message": f"Failed to approve agency: {e}"}), 500

    @app.route('/api/export_data', methods=['POST'])
    @ensure_orchestrator # Orchestrator might not be strictly needed but good for consistency
    async def export_data(orchestrator: 'Orchestrator') -> Response:
        logger.info("API /api/export_data POST request received.")
        if not SETTINGS_DB_UTILS_AVAILABLE or not MODELS_AVAILABLE:
             return jsonify({"error": "Data export unavailable due to missing core components."}), 503

        try:
            data = await request.get_json()
            if not data:
                return jsonify({"error": "Missing JSON payload"}), 400

            password = data.get('password')
            export_type = data.get('type', 'clients') # Default to 'clients'

            expected_password = settings.DOWNLOAD_PASSWORD
            if not expected_password:
                logger.error("DOWNLOAD_PASSWORD is not set in environment. Export aborted.")
                op_logger.error("Data export attempted but DOWNLOAD_PASSWORD not set.")
                return jsonify({"error": "Server configuration error: download password not set."}), 500

            if password != expected_password:
                logger.warning(f"Failed export attempt for type '{export_type}' due to incorrect password.")
                return jsonify({"error": "Invalid password"}), 403

            op_logger.info(f"Data export approved for type: {export_type}")

            output = io.StringIO()
            writer = csv.writer(output)
            filename = f"{export_type}_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
            
            headers: List[str] = []
            rows: List[List[Any]] = []

            async with get_session() as db_session: # type: ignore
                if db_session is None:
                    return jsonify({"error": "Database connection failed for export."}), 503

                if export_type == 'clients':
                    headers = Client.__table__.columns.keys() if hasattr(Client, '__table__') else ["id", "company_name", "contact_email", "status"] # fallback
                    stmt = select(Client) # type: ignore
                    result = await db_session.execute(stmt)
                    for client_obj in result.scalars().all():
                        rows.append([getattr(client_obj, col, None) for col in headers])
                elif export_type == 'tasks':
                    headers = Task.__table__.columns.keys() if hasattr(Task, '__table__') else ["id", "agent_name", "status", "created_at"] # fallback
                    stmt = select(Task).order_by(desc(Task.created_at)).limit(5000) # type: ignore
                    result = await db_session.execute(stmt)
                    for task_obj in result.scalars().all():
                         rows.append([getattr(task_obj, col, None) for col in headers])
                elif export_type == 'financials':
                    headers = FinancialTransaction.__table__.columns.keys() if hasattr(FinancialTransaction, '__table__') else ["id", "amount", "status", "timestamp"] # fallback
                    stmt = select(FinancialTransaction).order_by(desc(FinancialTransaction.timestamp)).limit(5000) # type: ignore
                    result = await db_session.execute(stmt)
                    for ft_obj in result.scalars().all():
                        rows.append([getattr(ft_obj, col, None) for col in headers])
                else:
                    return jsonify({"error": "Invalid export type specified."}), 400
            
            if not headers and not rows: # If somehow models were dummies
                 return jsonify({"error": f"No data structure available for export type '{export_type}'. Models might be missing."}), 500


            writer.writerow(headers)
            writer.writerows(rows)
            output.seek(0)
            
            # Use send_file for proper handling of file downloads
            return await send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                attachment_filename=filename
            )

        except Exception as e:
            logger.error(f"Error exporting data: {e}", exc_info=True)
            op_logger.error(f"Data Export Error: {e}")
            return jsonify({"error": f"An error occurred during export: {str(e)}"}), 500


    @app.route('/api/submit_feedback', methods=['POST'])
    @ensure_orchestrator
    async def submit_feedback(orchestrator: 'Orchestrator') -> Response:
        logger.info("API /api/submit_feedback POST request received.")
        try:
            data = await request.get_json()
            if not data or 'feedback' not in data:
                return jsonify({"error": "Missing 'feedback' in JSON payload"}), 400
            
            feedback_text = str(data['feedback']).strip()
            if not feedback_text:
                return jsonify({"error": "Feedback content cannot be empty"}), 400

            # Assuming orchestrator has a method to handle feedback/directives
            if hasattr(orchestrator, 'submit_directive_idea'):
                success, message = await orchestrator.submit_directive_idea(feedback_text, source="manual_ui_input")
                if success:
                    op_logger.info(f"Feedback/Directive submitted via UI: '{feedback_text[:100]}...'")
                    return jsonify({"status": "success", "message": message or "Feedback submitted successfully."})
                else:
                    op_logger.warning(f"Feedback/Directive submission failed via UI: {message}")
                    return jsonify({"status": "error", "message": message or "Failed to process feedback."}), 500
            else:
                logger.warning("Orchestrator does not have 'submit_directive_idea' method. Feedback not processed.")
                return jsonify({"status": "error", "message": "Feedback processing not implemented in orchestrator."}), 501
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}", exc_info=True)
            op_logger.error(f"UI Feedback Submission Error: {e}")
            return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500


    @app.route('/api/test_voice_call', methods=['POST'])
    @ensure_orchestrator
    async def test_voice_call(orchestrator: 'Orchestrator') -> Response:
        logger.info("API /api/test_voice_call POST request received.")
        try:
            data = await request.get_json()
            if not data or 'phone_number' not in data:
                return jsonify({"error": "Missing 'phone_number' in JSON payload"}), 400
            
            phone_number = str(data['phone_number']).strip()
            # Basic E.164 validation (very simplified, real validation is complex)
            if not re.match(r"^\+[1-9]\d{1,14}$", phone_number):
                return jsonify({"error": "Invalid phone number format. Use E.164 (e.g., +12223334444)."}), 400

            if hasattr(orchestrator, 'initiate_test_call'):
                success, message = await orchestrator.initiate_test_call(phone_number)
                if success:
                    op_logger.info(f"Test voice call initiated via UI to: {phone_number}")
                    return jsonify({"status": "success", "message": message or f"Test call to {phone_number} requested."})
                else:
                    op_logger.warning(f"Test voice call failed for {phone_number}: {message}")
                    return jsonify({"status": "error", "message": message or "Failed to initiate test call."}), 500
            else:
                logger.warning("Orchestrator does not have 'initiate_test_call' method.")
                return jsonify({"status": "error", "message": "Test call functionality not implemented."}), 501
        except Exception as e:
            logger.error(f"Error initiating test voice call: {e}", exc_info=True)
            op_logger.error(f"UI Test Call Error: {e}")
            return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500

    @app.route('/api/generate_videos', methods=['POST'])
    @ensure_orchestrator
    async def generate_sample_videos(orchestrator: 'Orchestrator') -> Response:
        logger.info("API /api/generate_videos POST request received.")
        try:
            data = await request.get_json()
            if not data:
                return jsonify({"error": "Missing JSON payload"}), 400
            
            topic_keywords = str(data.get('topic_keywords', '')).strip()
            video_count = int(data.get('video_count', 1))

            if not topic_keywords:
                return jsonify({"error": "Missing 'topic_keywords'."}), 400
            if not (1 <= video_count <= 5): # Max 5 videos for a sample run
                return jsonify({"error": "Video count must be between 1 and 5."}), 400

            user_email_for_videos = settings.get('USER_EMAIL')
            if not user_email_for_videos:
                logger.error("USER_EMAIL not configured. Cannot determine where to send generated videos.")
                op_logger.error("Video generation requested, but USER_EMAIL for delivery is missing.")
                return jsonify({"error": "Operator email (USER_EMAIL) not configured for video delivery."}), 500

            if hasattr(orchestrator, 'request_sample_video_generation'):
                # This method would enqueue a task for the VideoCreationAgent (or similar)
                # and likely use the notification system to email the results.
                success, message = await orchestrator.request_sample_video_generation(
                    topic_keywords=topic_keywords,
                    num_videos=video_count,
                    recipient_email=user_email_for_videos
                )
                if success:
                    op_logger.info(f"Sample video generation requested via UI: '{topic_keywords}', count: {video_count} for {user_email_for_videos}")
                    return jsonify({"status": "success", "message": message or "Sample video generation process initiated."})
                else:
                    op_logger.warning(f"Sample video generation request failed: {message}")
                    return jsonify({"status": "error", "message": message or "Failed to initiate video generation."}), 500
            else:
                logger.warning("Orchestrator does not have 'request_sample_video_generation' method.")
                return jsonify({"status": "error", "message": "Video generation functionality not implemented."}), 501

        except ValueError:
            return jsonify({"error": "Invalid 'video_count'. Must be an integer."}), 400
        except Exception as e:
            logger.error(f"Error in generate_sample_videos: {e}", exc_info=True)
            op_logger.error(f"UI Video Generation Error: {e}")
            return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500

    logger.info("All UI routes added to Quart app.")
# --- End of ui/app.py ---