# Filename: main.py
# Description: AI Agency Main Entry Point - Handles initialization and graceful shutdown.
# Version: 3.2 (Simplified - Uses Quart run, removed Hypercorn)

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
try:
    from config.settings import settings
    # Basic logging setup using settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    log_file = settings.LOG_FILE_PATH
    op_log_file = settings.OPERATIONAL_LOG_FILE_PATH
    log_handlers = [logging.StreamHandler(sys.stdout)]
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
    logger = logging.getLogger(__name__)
    logger.info("-------------------- Application Starting --------------------")
    logger.info("Logging configured based on settings.")
    logger.info("Configuration settings loaded and validated.")
    # --- TEMPORARY DIAGNOSTIC PRINT ---
    try:
        db_url_to_check = str(settings.DATABASE_URL) if settings.DATABASE_URL else "DATABASE_URL NOT SET IN SETTINGS"
        print(f"DEBUG_DB_URL_CHECK: DATABASE_URL as read by Settings = '{db_url_to_check}'", file=sys.stderr)
    except Exception as e:
        print(f"DEBUG_DB_URL_CHECK: Error accessing settings.DATABASE_URL: {e}", file=sys.stderr)
    # --- END TEMPORARY DIAGNOSTIC PRINT ---
except (ImportError, ValueError, SystemExit) as e:
    print(f"CRITICAL STARTUP ERROR: Failed during settings import or validation: {e}", file=sys.stderr)
    traceback.print_exc(); sys.exit(1)
except Exception as e:
    print(f"CRITICAL STARTUP ERROR: Unexpected error during initial setup: {e}", file=sys.stderr)
    traceback.print_exc(); sys.exit(1)

# --- Core Imports (After Settings & Logging) ---
try:
    from agents.orchestrator import Orchestrator
    from quart import Quart # Keep Quart import
    logger.info("Core components (Orchestrator, Quart) imported successfully.")
except ImportError as e:
    logger.critical(f"Fatal Error: Failed to import core components: {e}. Check project structure and PYTHONPATH.", exc_info=True)
    sys.exit(1)

# --- Global Orchestrator Instance ---
orchestrator_instance: Optional[Orchestrator] = None
app_instance: Optional[Quart] = None # Keep track of the app instance

async def run_agency_logic(orchestrator: Orchestrator):
    """Runs the main orchestrator loop."""
    logger.info("Starting Orchestrator main execution loop...")
    try:
        await orchestrator.run()
        logger.info("Orchestrator run loop finished normally.")
    except Exception as e:
        logger.critical(f"Fatal Error: Critical error during agency execution: {e}", exc_info=True)
        raise

async def startup():
    """Initializes the orchestrator and starts its background tasks."""
    global orchestrator_instance, app_instance
    logger.info("Application startup sequence initiated.")
    try:
        orchestrator_instance = Orchestrator(schema='public')
        app_instance = orchestrator_instance.app # Get the app from orchestrator
        # Start the orchestrator's background tasks (agent loops, periodic tasks)
        # The orchestrator.run() method now handles this internally
        asyncio.create_task(run_agency_logic(orchestrator_instance), name="AgencyLogicTask")
        logger.info("Orchestrator initialized and background tasks started.")
    except Exception as e:
        logger.critical(f"Fatal Error during startup: {e}", exc_info=True)
        # Attempt graceful shutdown even if startup fails
        await shutdown()
        sys.exit(1) # Exit after attempting shutdown

async def shutdown():
    """Handles graceful shutdown."""
    global orchestrator_instance
    logger.info("Initiating graceful shutdown sequence...")
    # Stop the orchestrator (which should handle agent stops and task cancellations)
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
        logger.warning("Orchestrator instance not available for final stop call.")
    logger.info("-------------------- Application Stopping --------------------")
    logging.shutdown()
    print("[INFO] main.py: Process stopped.")

# --- Main Execution Block ---
# This block is now simpler as 'quart run' handles the server lifecycle.
# We use Quart's startup/shutdown events to manage the orchestrator.
if __name__ == "__main__":
    # We need the app instance defined *before* Quart tries to run it.
    # Initialize orchestrator (and thus the app) synchronously here for simplicity,
    # although ideally, async setup would be preferred if Quart supported async factories easily.
    # This is a slight simplification for the 'quart run' command.
    try:
        orchestrator_instance = Orchestrator(schema='public')
        app_instance = orchestrator_instance.app

        # Register startup and shutdown events
        @app_instance.before_serving
        async def before_serving():
            await startup()

        @app_instance.after_serving
        async def after_serving():
            await shutdown()

        # The 'quart run' command specified in nixpacks.toml will now
        # find this 'app_instance' and run it using its development server.
        # We don't need asyncio.run(main()) or hypercorn here.
        logger.info("Setup complete. Handing off to 'quart run' command.")

    except Exception as e:
        # Catch initialization errors before Quart even starts
        print(f"CRITICAL ERROR during pre-run initialization: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

# --- End of main.py ---