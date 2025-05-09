# Filename: app.py
# Description: AI Agency Main Entry Point - Handles initialization and graceful shutdown.
# Version: 3.2.1 (Transmuted by IGNIS - Renamed from main.py, app instance renamed to 'app')

import asyncio
import logging
import os
import sys
import traceback # For detailed error logging
from dotenv import load_dotenv
from typing import Optional, Any

# --- Environment Loading (Must be first) ---
project_root = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(project_root, '.env.local')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"[INFO] app.py: Loaded environment variables from: {dotenv_path}")
else:
    print(f"[INFO] app.py: .env.local file not found. Relying on system environment variables.")

# --- Settings Import & Validation (Must be early) ---
try:
    from config.settings import settings
    # Basic logging setup using settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    log_file = settings.LOG_FILE_PATH
    op_log_file = settings.OPERATIONAL_LOG_FILE_PATH
    log_handlers: List[Any] = [logging.StreamHandler(sys.stdout)] # Ensure type hint for list
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_handlers.append(logging.FileHandler(log_file, mode='a'))
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', handlers=log_handlers, datefmt='%Y-%m-%d %H:%M:%S')
    if op_log_file:
        os.makedirs(os.path.dirname(op_log_file), exist_ok=True)
        op_logger = logging.getLogger('OperationalLog')
        op_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        op_handler = logging.FileHandler(op_log_file, mode='a')
        op_handler.setFormatter(op_formatter)
        op_logger.addHandler(op_handler)
        op_logger.setLevel(logging.INFO)
        op_logger.propagate = False
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.INFO)
    logging.getLogger("playwright").setLevel(logging.INFO)
    logger = logging.getLogger(__name__) # This will be '__main__' if run directly, or 'app' if imported.
    logger.info("-------------------- Application Starting (app.py) --------------------")
    logger.info("Logging configured based on settings.")
    logger.info("Configuration settings loaded and validated.")
    try:
        db_url_to_check = str(settings.DATABASE_URL) if settings.DATABASE_URL else "DATABASE_URL NOT SET IN SETTINGS"
        print(f"DEBUG_DB_URL_CHECK: DATABASE_URL as read by Settings = '{db_url_to_check}'", file=sys.stderr)
    except Exception as e_diag: # Renamed e to avoid conflict
        print(f"DEBUG_DB_URL_CHECK: Error accessing settings.DATABASE_URL: {e_diag}", file=sys.stderr)
except (ImportError, ValueError, SystemExit) as e:
    print(f"CRITICAL STARTUP ERROR: Failed during settings import or validation: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL STARTUP ERROR: Unexpected error during initial setup: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# --- Core Imports (After Settings & Logging) ---
try:
    from agents.orchestrator import Orchestrator
    from quart import Quart # Keep Quart import
    logger.info("Core components (Orchestrator, Quart) imported successfully.")
except ImportError as e:
    logger.critical(f"Fatal Error: Failed to import core components: {e}. Check project structure and PYTHONPATH.", exc_info=True)
    sys.exit(1)

# --- Global Orchestrator and Quart App Instances ---
orchestrator_instance: Optional[Orchestrator] = None
app: Optional[Quart] = None # Transmuted: Renamed from app_instance to app

async def startup():
    """Initializes the orchestrator and starts its background tasks."""
    global orchestrator_instance, app # Transmuted: Ensure 'app' is used here if it were assigned differently
    logger.info("Application startup sequence initiated.")
    try:
        # Orchestrator is already initialized before Quart's before_serving if __name__ == "__main__" runs.
        # This startup function is mainly for tasks that must run *after* Quart's loop starts
        # but *before* it starts serving, or for re-initialization logic if needed.
        # The current Orchestrator initialization in __main__ is fine.
        if orchestrator_instance is None: # Should not happen with current structure
            logger.error("Orchestrator instance is None at Quart startup. This is unexpected.")
            # Attempt re-initialization (though this indicates a structural issue if hit)
            orchestrator_instance = Orchestrator(schema='public')
            if app is None: # Should also not happen
                 # If app was not set, it means orchestrator_instance.app wasn't assigned to global 'app'
                 # This indicates a logic error in the main block if __name__ == "__main__"
                 logger.critical("Quart 'app' instance is None. Orchestrator's app not correctly assigned globally.")
                 raise RuntimeError("Quart app instance not available.")
        
        # Start the orchestrator's background tasks (agent loops, periodic tasks)
        # The orchestrator.run() method handles this internally.
        # We need to ensure orchestrator.run() is launched as an asyncio task.
        if orchestrator_instance and not orchestrator_instance.running: # Check if orchestrator logic is already running
            asyncio.create_task(run_agency_logic(orchestrator_instance), name="AgencyLogicTask")
            logger.info("Orchestrator initialized/confirmed and background tasks started/confirmed.")
        else:
            logger.info("Orchestrator already running or startup sequence adjusted.")

    except Exception as e:
        logger.critical(f"Fatal Error during Quart startup event: {e}", exc_info=True)
        await shutdown() # Attempt graceful shutdown
        sys.exit(1)

async def run_agency_logic(orchestrator: Orchestrator):
    """Runs the main orchestrator loop."""
    logger.info("Starting Orchestrator main execution loop...")
    try:
        await orchestrator.run() # This is orchestrator's main async processing loop
        logger.info("Orchestrator run loop finished normally.")
    except Exception as e:
        logger.critical(f"Fatal Error: Critical error during agency execution: {e}", exc_info=True)
        # Potentially trigger a more system-wide shutdown if Orchestrator crashes
        # For now, it will be caught by the task handler
        raise

async def shutdown():
    """Handles graceful shutdown."""
    global orchestrator_instance
    logger.info("Initiating graceful shutdown sequence...")
    if orchestrator_instance and hasattr(orchestrator_instance, 'stop'):
        if orchestrator_instance.status not in [orchestrator_instance.STATUS_STOPPING, orchestrator_instance.STATUS_STOPPED]:
            logger.info("Calling orchestrator stop method...")
            try:
                await orchestrator_instance.stop(timeout=25.0)
            except Exception as stop_err:
                logger.error(f"Error during orchestrator stop: {stop_err}", exc_info=True)
        else:
            logger.info(f"Orchestrator already in state: {orchestrator_instance.status}. Skipping stop call.")
    else:
        logger.warning("Orchestrator instance not available for final stop call during shutdown.")
    logger.info("-------------------- Application Stopping (app.py) --------------------")
    logging.shutdown()
    print("[INFO] app.py: Process stopped.")

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # Initialize Orchestrator and get the Quart app instance
        # This needs to happen before Quart tries to run `app`
        orchestrator_instance = Orchestrator(schema='public')
        app = orchestrator_instance.app # Transmuted: Assign to 'app'

        if app is None:
            logger.critical("CRITICAL: Quart application 'app' was not created by Orchestrator.")
            sys.exit("Quart app not initialized.")

        # Register Quart's startup and shutdown events
        @app.before_serving
        async def before_serving_handler(): # Renamed to avoid conflict
            await startup()

        @app.after_serving
        async def after_serving_handler(): # Renamed to avoid conflict
            await shutdown()

        logger.info("Setup complete. Handing off to 'quart run' command (expects 'app:app' or similar).")
        # To run this directly (e.g., python app.py), you would add:
        # import uvicorn # or hypercorn
        # uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
        # However, the deployment uses `quart run`, so this direct run block is not strictly necessary here.
        # The `quart run` command should be configured to point to `app:app` (filename:variable_name)
        # For default `quart run` behavior, it expects the Quart app object to be named `app` in `app.py`
        
        # The Nixpacks or Docker CMD should be:
        # quart run --host 0.0.0.0 --port ${PORT:-8080}
        # This command, by default, will look for 'app.py' and an object named 'app' within it.

    except Exception as e:
        print(f"CRITICAL ERROR during pre-run initialization in app.py: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

# --- End of app.py ---