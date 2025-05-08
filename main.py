# Filename: main.py
# Description: AI Agency Main Entry Point - Handles initialization and graceful shutdown.
# Version: 3.3 (Docker/Coolify Deployment Ready)

import asyncio
import logging
import os
import sys
import traceback # For detailed error logging
# Removed dotenv load - Rely on Coolify environment variables
# from dotenv import load_dotenv
from typing import Optional

# --- Environment Loading Removed ---
# project_root = os.path.dirname(os.path.abspath(__file__))
# dotenv_path = os.path.join(project_root, '.env') # Use standard .env now
# if os.path.exists(dotenv_path):
#     load_dotenv(dotenv_path=dotenv_path)
#     print(f"[INFO] main.py: Loaded environment variables from: {dotenv_path}")
# else:
#     print(f"[INFO] main.py: .env file not found. Relying on system environment variables provided by Coolify/Docker.")

# --- Settings Import & Validation (Must be early) ---
try:
    # Settings will now read directly from environment variables
    from config.settings import settings
    # Basic logging setup using settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    # Prefer stdout/stderr for Docker logging unless explicitly configured otherwise
    log_handlers = [logging.StreamHandler(sys.stdout)] # Default to stdout
    if settings.LOG_FILE_PATH: # Only add file handler if path is explicitly set
        log_file = settings.LOG_FILE_PATH
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            log_handlers.append(logging.FileHandler(log_file, mode='a'))
            print(f"[INFO] Logging also directed to file: {log_file}")
        except Exception as log_dir_err:
            print(f"[ERROR] Failed to create log directory {os.path.dirname(log_file)}: {log_dir_err}. Logging to stdout only.")

    # Configure root logger
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', handlers=log_handlers, datefmt='%Y-%m-%d %H:%M:%S')

    # Configure operational logger (potentially still useful for structured event logs)
    if settings.OPERATIONAL_LOG_FILE_PATH:
        op_log_file = settings.OPERATIONAL_LOG_FILE_PATH
        try:
            os.makedirs(os.path.dirname(op_log_file), exist_ok=True)
            op_logger = logging.getLogger('OperationalLog')
            op_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            op_handler = logging.FileHandler(op_log_file, mode='a')
            op_handler.setFormatter(op_formatter)
            op_logger.addHandler(op_handler)
            op_logger.setLevel(logging.INFO)
            op_logger.propagate = False # Don't duplicate to root logger
            print(f"[INFO] Operational logging directed to file: {op_log_file}")
        except Exception as op_log_dir_err:
            print(f"[ERROR] Failed to create operational log directory {os.path.dirname(op_log_file)}: {op_log_dir_err}.")

    # Reduce noise from libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.INFO)
    logging.getLogger("playwright").setLevel(logging.INFO)
    logging.getLogger("multipart").setLevel(logging.WARNING) # Added for Quart/uploads

    logger = logging.getLogger(__name__) # Get logger for main.py
    logger.info("-------------------- Application Starting (Docker Mode) --------------------")
    logger.info("Logging configured. Relying on environment variables for settings.")
    # Basic check if critical env var is loaded by settings
    db_url_status = "OK" if settings.DATABASE_URL else "MISSING!"
    logger.info(f"DATABASE_URL loaded status: {db_url_status}")
    agency_url_status = "OK" if settings.AGENCY_BASE_URL else "MISSING!"
    logger.info(f"AGENCY_BASE_URL loaded status: {agency_url_status}")

except (ImportError, ValueError, SystemExit) as e:
    # Use print as logger might not be configured yet
    print(f"CRITICAL STARTUP ERROR: Failed during settings import or validation: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL STARTUP ERROR: Unexpected error during initial setup: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# --- Core Imports (After Settings & Logging) ---
try:
    from agents.orchestrator import Orchestrator, shutdown_event # Import shutdown_event
    from quart import Quart # Keep Quart import
    logger.info("Core components (Orchestrator, Quart) imported successfully.")
except ImportError as e:
    logger.critical(f"Fatal Error: Failed to import core components: {e}. Check project structure and PYTHONPATH.", exc_info=True)
    sys.exit(1)

# --- Global Orchestrator Instance ---
# We MUST define these globally for Quart's run command and event decorators to work correctly.
orchestrator_instance: Optional[Orchestrator] = None
app_instance: Optional[Quart] = None

async def startup_event():
    """Initializes the orchestrator and starts its background tasks. Runs before first request."""
    global orchestrator_instance, app_instance
    logger.info("Quart startup event triggered. Initializing Orchestrator...")
    if orchestrator_instance is None:
        logger.critical("Orchestrator instance is unexpectedly None during startup.")
        # Attempting re-init as a fallback, but this indicates a potential issue
        try:
             orchestrator_instance = Orchestrator() # Assuming schema='public' default or handled internally
             app_instance = orchestrator_instance.app # Get app from re-initialized orchestrator
        except Exception as reinit_err:
             logger.critical(f"Failed to re-initialize Orchestrator during startup: {reinit_err}")
             # Decide how to handle - potentially exit? For now, log critical.
             return # Prevent further execution if re-init fails

    if not orchestrator_instance.running:
        # Trigger the orchestrator's main run loop in the background
        # This handles DB init, client init, agent init, and agent loops
        logger.info("Starting Orchestrator main run loop in background...")
        # Use create_task to run the orchestrator's main loop concurrently
        asyncio.create_task(orchestrator_instance.run(), name="OrchestratorRunLoop")
    else:
        logger.info("Orchestrator already running.")


async def shutdown_event_handler():
    """Handles graceful shutdown. Runs after server stops."""
    global orchestrator_instance
    logger.info("Quart shutdown event triggered. Initiating graceful shutdown...")
    shutdown_event.set() # Signal shutdown to loops/tasks
    if orchestrator_instance and hasattr(orchestrator_instance, 'stop'):
        logger.info("Calling orchestrator stop method...")
        try:
            # The stop method now handles cancelling tasks and stopping agents
            await asyncio.wait_for(orchestrator_instance.stop(timeout=25.0), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error("Orchestrator stop method timed out during shutdown.")
        except Exception as stop_err:
            logger.error(f"Error during orchestrator stop: {stop_err}", exc_info=True)
    else:
        logger.warning("Orchestrator instance not available or has no stop method for final shutdown.")

    logger.info("-------------------- Application Stopping (via Quart Shutdown) --------------------")
    logging.shutdown() # Flush and close logging handlers


# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # Initialize Orchestrator HERE so 'app_instance' is available globally
        # Orchestrator __init__ creates the Quart app instance.
        orchestrator_instance = Orchestrator() # Default schema='public' or set as needed
        app_instance = orchestrator_instance.app

        # Register Quart startup and shutdown events
        app_instance.before_serving(startup_event)
        app_instance.after_serving(shutdown_event_handler)

        # Get HOST and PORT from environment variables, falling back to defaults
        # These are used by the `quart run` command if invoked manually,
        # but Coolify typically overrides PORT via its own mechanisms.
        # Binding to 0.0.0.0 is crucial for Docker.
        host = os.environ.get('HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', 5000))

        logger.info(f"Quart app setup complete. Ready to run on {host}:{port}")
        # The actual server start is handled by the CMD in the Dockerfile:
        # ["quart", "run", "--host", "0.0.0.0", "--port", "5000", "--no-reload"]
        # This __main__ block primarily ensures the app and event handlers are registered.
        # If you were running this script directly (python main.py), you would add:
        # app_instance.run(host=host, port=port, debug=settings.DEBUG)
        # However, for Docker/Coolify using the CMD is standard practice.

    except Exception as e:
        # Catch initialization errors before Quart even starts
        logger.critical(f"CRITICAL ERROR during main block initialization: {e}", exc_info=True)
        print(f"CRITICAL ERROR during main block initialization: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

# --- End of main.py ---