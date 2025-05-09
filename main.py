# Filename: main.py
# Description: AI Agency Main Entry Point - Handles initialization and graceful shutdown.
# Version: 3.3 (Level 30+ Transmutation - Standardized logging, robust error handling)

import asyncio
import logging
import os
import sys
import traceback
from dotenv import load_dotenv
from typing import Optional

# --- Environment Loading (Must be first) ---
project_root = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(project_root, '.env.local') # Prioritize .env.local
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True) # Override system vars if .env.local exists
    print(f"[INFO] main.py: Loaded environment variables from: {dotenv_path}")
elif os.path.exists(os.path.join(project_root, '.env')): # Fallback to .env
    load_dotenv(dotenv_path=os.path.join(project_root, '.env'), override=True)
    print(f"[INFO] main.py: Loaded environment variables from: .env")
else:
    print(f"[INFO] main.py: No .env or .env.local file found. Relying solely on system environment variables.")

# --- Settings Import & Validation (Must be early) ---
# This structure ensures logging is configured AFTER settings are loaded but BEFORE other imports.
try:
    from config.settings import settings # Attempt to import
except (ImportError, ValueError, SystemExit) as e: # Catch Pydantic's validation errors (ValueError/SystemExit) too
    # Basic print for critical early failure if logging isn't even setup
    print(f"CRITICAL STARTUP ERROR: Failed during settings import or validation: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1) # Hard exit if settings fail

# --- Standardized Logging Configuration (AFTER settings are loaded) ---
log_level_name = getattr(settings, 'LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_name, logging.INFO)
log_file_main = getattr(settings, 'LOG_FILE_PATH', None) # Default to None if not set
log_file_op = getattr(settings, 'OPERATIONAL_LOG_FILE_PATH', None) # Default to None

log_handlers = [logging.StreamHandler(sys.stdout)] # Always log to stdout
if log_file_main and log_file_main.strip():
    os.makedirs(os.path.dirname(os.path.abspath(log_file_main)), exist_ok=True)
    log_handlers.append(logging.FileHandler(log_file_main, mode='a'))

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=log_handlers,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configure Operational Logger (Human-readable, less verbose)
op_logger = logging.getLogger('OperationalLog')
op_logger.setLevel(logging.INFO) # Operational log should generally be INFO level
op_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
op_logger.addHandler(logging.StreamHandler(sys.stdout)) # Also to stdout
if log_file_op and log_file_op.strip():
    os.makedirs(os.path.dirname(os.path.abspath(log_file_op)), exist_ok=True)
    op_file_handler = logging.FileHandler(log_file_op, mode='a')
    op_file_handler.setFormatter(op_formatter)
    op_logger.addHandler(op_file_handler)
op_logger.propagate = False # Avoid double-logging to root logger's handlers

# Reduce noise from common libraries
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING) # Quieter unless debugging websockets
logging.getLogger("playwright").setLevel(logging.INFO) # Playwright can be noisy on DEBUG
logging.getLogger("httpx").setLevel(logging.WARNING) # For OpenAI client

logger = logging.getLogger(__name__) # Main application logger
logger.info("-------------------- Application Starting --------------------")
logger.info(f"Logging configured. Level: {log_level_name}. Main Log: {log_file_main or 'stdout only'}. Op Log: {log_file_op or 'stdout only'}.")
logger.info(f"Configuration settings loaded for {settings.APP_NAME} v{settings.APP_VERSION}.")
logger.info(f"Agency Purpose: {settings.AGENCY_PURPOSE}")

# --- Core Imports (After Settings & Logging) ---
try:
    from agents.orchestrator import Orchestrator
    from quart import Quart
    logger.info("Core components (Orchestrator, Quart) imported successfully.")
except ImportError as e:
    logger.critical(f"Fatal Error: Failed to import core components: {e}. Check project structure and PYTHONPATH.", exc_info=True)
    sys.exit(1) # Hard exit
except Exception as e: # Catch any other unexpected import errors
    logger.critical(f"Fatal Error: Unexpected error importing core components: {e}", exc_info=True)
    sys.exit(1)

# --- Global Orchestrator Instance ---
orchestrator_instance: Optional[Orchestrator] = None
app_instance: Optional[Quart] = None

async def run_agency_logic_async(orchestrator: Orchestrator):
    """Runs the main orchestrator background tasks."""
    logger.info("Starting Orchestrator main background logic...")
    try:
        # Orchestrator.run() is now more about starting its internal loops and periodic tasks
        # rather than being a single blocking call.
        await orchestrator.run() # This should set up periodic tasks and agent loops
        logger.info("Orchestrator background logic tasks are running.")
    except Exception as e:
        logger.critical(f"Fatal Error: Critical error during agency background logic execution: {e}", exc_info=True)
        # Trigger a more graceful shutdown if possible
        if orchestrator_instance:
            await orchestrator_instance.stop(timeout=10.0) # Attempt to stop agents
        sys.exit(1) # Indicate failure

async def startup_event_handler():
    """Initializes the orchestrator and starts its background tasks for Quart."""
    global orchestrator_instance, app_instance
    logger.info("Application startup event handler initiated.")
    try:
        if orchestrator_instance is None: # Ensure it's only initialized once
            # Orchestrator now creates its own app instance.
            orchestrator_instance = Orchestrator(schema='public') # Or your desired schema
            app_instance = orchestrator_instance.app # Get the app from orchestrator
        
        # Orchestrator.run() now starts its internal periodic tasks and agent loops.
        # It doesn't block, so we create a task for it.
        asyncio.create_task(run_agency_logic_async(orchestrator_instance), name="AgencyBackgroundLogicTask")
        logger.info("Orchestrator initialized and background logic tasks started.")

    except Exception as e:
        logger.critical(f"Fatal Error during application startup event: {e}", exc_info=True)
        await shutdown_event_handler() # Attempt graceful shutdown
        sys.exit(1)

async def shutdown_event_handler():
    """Handles graceful shutdown for Quart."""
    global orchestrator_instance
    logger.info("Application shutdown event handler initiated...")
    if orchestrator_instance and hasattr(orchestrator_instance, 'stop'):
        current_status = getattr(orchestrator_instance, 'status', 'unknown')
        stopping_statuses = [
            getattr(orchestrator_instance, 'STATUS_STOPPING', 'stopping'),
            getattr(orchestrator_instance, 'STATUS_STOPPED', 'stopped')
        ]
        if current_status not in stopping_statuses:
            logger.info("Calling orchestrator stop method...")
            try:
                await orchestrator_instance.stop(timeout=25.0) # Extended timeout
            except Exception as stop_err:
                logger.error(f"Error during orchestrator stop: {stop_err}", exc_info=True)
        else:
            logger.info(f"Orchestrator already in state: {current_status}. Skipping redundant stop call.")
    else:
        logger.warning("Orchestrator instance not available or no stop method for final shutdown.")

    logger.info("-------------------- Application Stopping --------------------")
    logging.shutdown() # Flushes and closes all logging handlers
    print("[INFO] main.py: Process stopped.")

# --- Main Execution Block (for Quart) ---
if __name__ == "__main__":
    try:
        # The Quart app instance needs to be available globally for `quart run`.
        # Orchestrator's __init__ creates the Quart app instance and assigns it to self.app.
        # We initialize Orchestrator here to make its app instance available.
        orchestrator_instance = Orchestrator(schema='public') # Or your default schema
        app_instance = orchestrator_instance.app # Get the app created by the orchestrator

        # Register Quart's startup and shutdown event handlers
        @app_instance.before_serving
        async def before_serving_wrapper():
            await startup_event_handler()

        @app_instance.after_serving
        async def after_serving_wrapper():
            await shutdown_event_handler()

        logger.info("Quart application setup complete. Handing off to 'quart run' command (or direct run if not using CLI).")
        
        # If you intend to run this file directly (e.g., `python main.py`)
        # without `quart run`, you'd need to start Quart's server here.
        # For Docker/Nixpacks, `quart run` in the start command is preferred.
        # Example for direct run (less common for production deployment):
        # from quart.app import Lifespan
        # from hypercorn.asyncio import serve
        # from hypercorn.config import Config as HypercornConfig
        #
        # async def main_direct_run():
        #     hypercorn_cfg = HypercornConfig()
        #     hypercorn_cfg.bind = [f"0.0.0.0:{os.environ.get('PORT', 5000)}"]
        #     # hypercorn_cfg.startup_timeout = 120 # Increase if startup is slow
        #     # hypercorn_cfg.shutdown_timeout = 60
        #
        #     # Wrap Quart app with Lifespan for Hypercorn
        #     lifespan_app = Lifespan(app_instance)
        #     await serve(lifespan_app, hypercorn_cfg)
        #
        # if os.environ.get("RUN_DIRECTLY_WITH_HYPERCORN"): # Example flag to enable direct run
        #      asyncio.run(main_direct_run())
        # else:
        #      logger.info("To run the server, use: quart run --host 0.0.0.0 --port <your_port>")
        #      # For `quart run` to work, `app_instance` needs to be top-level or discoverable.
        #      # If this script is the entry point for `quart run`, it will find `app_instance`.

    except Exception as e:
        # Catch initialization errors before Quart even starts
        logger.critical(f"CRITICAL ERROR during pre-run initialization: {e}", exc_info=True)
        sys.exit(1) # Hard exit

# --- End of main.py ---