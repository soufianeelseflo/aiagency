# Filename: main.py
# Description: AI Agency Main Entry Point - Handles initialization and graceful shutdown.
# Version: 3.3 (Ensures app_instance is module-level for Quart CLI discovery)

import asyncio
import logging
import os
import sys
import traceback # For detailed error logging
from dotenv import load_dotenv
from typing import Optional

# --- Environment Loading (Must be first) ---
project_root = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(project_root, '.env.local')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"[INFO] main.py: Loaded environment variables from: {dotenv_path}")
else:
    print(f"[INFO] main.py: .env.local file not found. Relying on system environment variables.")

# --- Settings Import & Validation (Must be early) ---
# This block needs to execute successfully for app_instance to be created.
# If settings fail, the app can't start.
try:
    from config.settings import settings
    # Basic logging setup using settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    log_file = settings.LOG_FILE_PATH
    op_log_file = settings.OPERATIONAL_LOG_FILE_PATH
    log_handlers = [logging.StreamHandler(sys.stdout)] # Ensure console output for container logs
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_handlers.append(logging.FileHandler(log_file, mode='a'))

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=log_handlers,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if op_log_file:
        os.makedirs(os.path.dirname(op_log_file), exist_ok=True)
        op_logger = logging.getLogger('OperationalLog')
        op_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        op_handler = logging.FileHandler(op_log_file, mode='a')
        op_handler.setFormatter(op_formatter)
        op_logger.addHandler(op_handler)
        op_logger.setLevel(logging.INFO) # Ensure operational logs are captured
        op_logger.propagate = False # Avoid duplicate logging to root

    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.INFO)
    logging.getLogger("playwright").setLevel(logging.INFO)
    # SQLAlchemy loggers can be noisy, set to WARNING unless debugging DB issues
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)


    logger = logging.getLogger(__name__) # Get logger after basicConfig
    logger.info("-------------------- Application Starting (main.py v3.3) --------------------")
    logger.info("Logging configured based on settings.")
    logger.info("Configuration settings loaded and validated.")

    # --- TEMPORARY DIAGNOSTIC PRINT ---
    try:
        db_url_to_check = str(settings.DATABASE_URL) if settings.DATABASE_URL else "DATABASE_URL NOT SET IN SETTINGS"
        logger.debug(f"DEBUG_DB_URL_CHECK (main.py): DATABASE_URL as read by Settings = '{db_url_to_check}'")
    except Exception as e_diag:
        logger.error(f"DEBUG_DB_URL_CHECK (main.py): Error accessing settings.DATABASE_URL: {e_diag}")
    # --- END TEMPORARY DIAGNOSTIC PRINT ---

except (ImportError, ValueError, SystemExit) as e_settings:
    # Use print as logging might not be fully configured if settings fail
    print(f"CRITICAL STARTUP ERROR (main.py): Failed during settings import or validation: {e_settings}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1) # Essential step for deployment failure
except Exception as e_initial_setup:
    print(f"CRITICAL STARTUP ERROR (main.py): Unexpected error during initial setup: {e_initial_setup}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1) # Essential step for deployment failure

# --- Core Imports (After Settings & Logging) ---
try:
    from agents.orchestrator import Orchestrator
    from quart import Quart # Keep Quart import
    logger.info("Core components (Orchestrator, Quart) imported successfully.")
except ImportError as e_core_import:
    logger.critical(f"Fatal Error (main.py): Failed to import core components: {e_core_import}. Check project structure and PYTHONPATH.", exc_info=True)
    sys.exit(1) # Essential step for deployment failure

# --- Global Orchestrator and App Instances ---
# These need to be at the module level for `quart run main:app_instance` to find them.
orchestrator_instance: Optional[Orchestrator] = None
app_instance: Optional[Quart] = None

try:
    orchestrator_instance = Orchestrator(schema='public') # Initialize Orchestrator
    app_instance = orchestrator_instance.app # Get the Quart app from the Orchestrator
    logger.info("Orchestrator and Quart app_instance created at module level.")
except Exception as e_orch_init:
    logger.critical(f"Fatal Error (main.py): Failed to initialize Orchestrator or app_instance at module level: {e_orch_init}", exc_info=True)
    sys.exit(1) # Essential step for deployment failure

# --- Startup and Shutdown Logic ---
async def startup():
    """Initializes the orchestrator and starts its background tasks."""
    # Orchestrator instance is already created. Here we focus on its async setup if any.
    # The Orchestrator's main run logic (agent loops) is now started by its own run() method,
    # which should be called by a background task managed by Quart's lifecycle.
    global orchestrator_instance
    if not orchestrator_instance:
        logger.critical("Orchestrator instance not found during startup sequence. This should not happen.")
        return

    logger.info("Application startup sequence initiated by Quart.")
    try:
        # Ensure Orchestrator's main loop is started as a background task
        # The orchestrator.run() method itself handles agent loops and periodic tasks.
        # We create a task for it so Quart's before_serving doesn't block.
        if not hasattr(orchestrator_instance, '_main_loop_task') or orchestrator_instance._main_loop_task is None or orchestrator_instance._main_loop_task.done():
            orchestrator_instance._main_loop_task = asyncio.create_task(orchestrator_instance.run(), name="OrchestratorMainLoop")
            logger.info("Orchestrator main execution loop (run()) tasked by Quart startup.")
        else:
            logger.info("Orchestrator main execution loop already tasked or running.")

    except Exception as e_startup:
        logger.critical(f"Fatal Error during Quart startup lifecycle: {e_startup}", exc_info=True)
        await shutdown() # Attempt graceful shutdown
        sys.exit(1) # Exit after attempting shutdown

async def shutdown():
    """Handles graceful shutdown."""
    global orchestrator_instance
    logger.info("Initiating graceful shutdown sequence via Quart.")

    if orchestrator_instance and hasattr(orchestrator_instance, 'stop'):
        current_orch_status = getattr(orchestrator_instance, 'status', 'unknown')
        orch_status_stopping = getattr(orchestrator_instance, 'STATUS_STOPPING', 'stopping')
        orch_status_stopped = getattr(orchestrator_instance, 'STATUS_STOPPED', 'stopped')

        if current_orch_status not in [orch_status_stopping, orch_status_stopped]:
            logger.info(f"Calling orchestrator stop method (current status: {current_orch_status})...")
            try:
                await orchestrator_instance.stop(timeout=25.0) # Orchestrator handles agent stops
            except Exception as stop_err:
                logger.error(f"Error during orchestrator stop: {stop_err}", exc_info=True)
        else:
            logger.info(f"Orchestrator already in state: {current_orch_status}. Skipping explicit stop call.")
    else:
        logger.warning("Orchestrator instance not available or has no stop method for final shutdown.")

    # Cancel any remaining asyncio tasks created directly by this main module (if any)
    # Typically, Orchestrator should manage its own tasks.
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        logger.info(f"Cancelling {len(tasks)} outstanding asyncio tasks...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Outstanding asyncio tasks cancellation complete.")

    logger.info("-------------------- Application Stopping (main.py v3.3) --------------------")
    # logging.shutdown() is called automatically on process exit by Python 3.8+ atexit.
    # Explicit call can be made if needed for older versions or specific handlers.
    print("[INFO] main.py: Process stop sequence completed.")


# --- Quart Event Registration ---
# This ensures that Quart manages the startup and shutdown of our application logic.
if app_instance: # Only register if app_instance was successfully created
    @app_instance.before_serving
    async def _before_serving():
        logger.info("Quart 'before_serving' event triggered.")
        await startup()

    @app_instance.after_serving
    async def _after_serving():
        logger.info("Quart 'after_serving' event triggered.")
        await shutdown()

    logger.info("Quart startup and shutdown events registered for app_instance.")
else:
    logger.critical("Quart app_instance is None. Cannot register lifecycle events. Application will not start correctly.")
    sys.exit(1) # Prevent running if app_instance could not be made.

# --- Main Execution Block (for `python main.py`) ---
# This block is primarily for local development if you run `python main.py`.
# For production deployment using `quart run main:app_instance`,
# Quart's CLI will import this file and use `app_instance`.
if __name__ == "__main__":
    if not app_instance:
        logger.critical("app_instance is None in __main__ block. Critical initialization failure.")
        sys.exit(1)

    logger.info("Running application directly using 'python main.py' (for local development).")
    logger.info("Production should use 'python -m quart run main:app_instance'.")

    # Get host and port from settings, with defaults for local dev
    host = settings.get("QUART_HOST", "127.0.0.1")
    port = int(settings.get("QUART_PORT", 5000)) # Ensure port is int
    debug_mode = settings.get("DEBUG", False)

    try:
        # This will run the Quart app using its development server.
        # The before_serving and after_serving events handle our startup/shutdown.
        app_instance.run(host=host, port=port, debug=debug_mode, use_reloader=False) # Reloader off for stability with async tasks
    except Exception as e_run:
        logger.critical(f"Error running Quart application directly: {e_run}", exc_info=True)
        # Attempt a final shutdown if run fails catastrophically
        asyncio.run(shutdown())
        sys.exit(1)

# --- End of main.py ---