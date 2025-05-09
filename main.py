# Filename: main.py
# Description: AI Agency Main Entry Point - Handles initialization and graceful shutdown.
# Version: 3.4 (Hardened for production `quart run` and clarified dev run)

import asyncio
import logging
import os
import sys
import traceback # For detailed error logging
from dotenv import load_dotenv
from typing import Optional

# --- Environment Loading (Must be first) ---
# Ensure this runs to make environment variables available for settings.
project_root = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(project_root, '.env.local') # Standard .env.local for local overrides
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True) # Override system env vars if .env.local exists
    print(f"[INFO] main.py: Loaded environment variables from: {dotenv_path}")
else:
    print(f"[INFO] main.py: .env.local file not found. Relying on system/deployment environment variables.")

# --- Settings Import & Validation (Must be early and robust) ---
# This section is critical. If settings fail to load, the app cannot proceed.
try:
    from config.settings import settings # This now has access to .env variables

    # Basic logging setup using settings
    log_level_str = settings.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_str, logging.INFO) # Default to INFO if invalid
    
    log_handlers = [logging.StreamHandler(sys.stdout)] # stdout is crucial for container logs

    # File logging (optional, based on settings)
    log_file_path = settings.LOG_FILE_PATH
    if log_file_path: # Check if path is configured
        try:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            log_handlers.append(logging.FileHandler(log_file_path, mode='a'))
        except Exception as e_lf:
            print(f"[ERROR] main.py: Could not create log file directory for '{log_file_path}': {e_lf}", file=sys.stderr)


    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=log_handlers,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger_main = logging.getLogger(__name__) # Get logger for main.py itself

    # Operational Log (human-readable, separate file if configured)
    op_log_file_path = settings.OPERATIONAL_LOG_FILE_PATH
    if op_log_file_path:
        try:
            os.makedirs(os.path.dirname(op_log_file_path), exist_ok=True)
            op_logger = logging.getLogger('OperationalLog')
            op_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            op_file_handler = logging.FileHandler(op_log_file_path, mode='a')
            op_file_handler.setFormatter(op_formatter)
            op_logger.addHandler(op_file_handler)
            op_logger.setLevel(logging.INFO) # Operational logs should generally be INFO and above
            op_logger.propagate = False # Prevent duplication to root logger if handlers are different
        except Exception as e_olf:
            logger_main.error(f"Could not create operational log file directory for '{op_log_file_path}': {e_olf}")


    # Quieten overly verbose libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.INFO) # Keep INFO for connection status
    logging.getLogger("playwright").setLevel(logging.WARNING) # Reduce Playwright noise unless debugging it
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)

    logger_main.info(f"-------------------- Application Starting (main.py v3.4) --------------------")
    logger_main.info(f"Logging configured. Level: {log_level_str}. Main Log File: {log_file_path or 'Console only'}. Op Log File: {op_log_file_path or 'Disabled'}.")
    logger_main.info("Configuration settings object 'settings' loaded and validated.")
    logger_main.debug(f"AGENCY_BASE_URL from settings: {settings.AGENCY_BASE_URL}")
    logger_main.debug(f"DATABASE_URL host from settings: {settings.DATABASE_URL.host if settings.DATABASE_URL else 'N/A'}")

except ImportError as e_imp:
    # This means config.settings or a dependency of it is missing/misplaced
    print(f"CRITICAL IMPORT ERROR (main.py): Failed to import 'config.settings': {e_imp}. Check PYTHONPATH and file existence.", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
except ValueError as e_val:
    # This means a Pydantic validation error occurred in settings.py
    print(f"CRITICAL VALIDATION ERROR (main.py): Settings validation failed: {e_val}. Check .env variables and settings.py definitions.", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
except SystemExit as e_sys_exit:
    # If settings.py itself calls sys.exit due to critical missing env vars
    print(f"CRITICAL SYSTEM EXIT (main.py): Exiting due to error during settings initialization: {e_sys_exit}", file=sys.stderr)
    sys.exit(e_sys_exit.code or 1) # Propagate exit code
except Exception as e_unexpected_setup:
    print(f"CRITICAL UNEXPECTED ERROR (main.py): During initial settings and logging setup: {e_unexpected_setup}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)


# --- Core Application Imports (after settings and logging are safely configured) ---
try:
    from agents.orchestrator import Orchestrator
    from quart import Quart
    logger_main.info("Core application components (Orchestrator, Quart) imported successfully.")
except ImportError as e_core_components:
    logger_main.critical(f"Fatal Error (main.py): Failed to import core application components (Orchestrator or Quart): {e_core_components}. This usually indicates a problem with the project structure or PYTHONPATH.", exc_info=True)
    sys.exit(1)


# --- Global Orchestrator and Quart App Instance Creation ---
# These MUST be defined at the module level for `quart run main:app_instance`
orchestrator_instance: Optional[Orchestrator] = None
app_instance: Optional[Quart] = None # This is what `quart run` will look for

try:
    # Initialize Orchestrator, which in turn creates its own Quart app instance
    orchestrator_instance = Orchestrator(schema='public') # schema can be from settings if needed
    if orchestrator_instance and hasattr(orchestrator_instance, 'app'):
        app_instance = orchestrator_instance.app
        logger_main.info("Orchestrator initialized and Quart app_instance obtained at module level.")
    else:
        raise RuntimeError("Orchestrator initialization failed or did not provide a Quart app instance.")
except Exception as e_app_init:
    logger_main.critical(f"Fatal Error (main.py): Failed to initialize Orchestrator or obtain app_instance: {e_app_init}", exc_info=True)
    sys.exit(1)


# --- Application Lifecycle Functions (Startup & Shutdown) ---
async def application_startup():
    """Async operations to perform before the Quart server starts serving requests."""
    global orchestrator_instance
    if not orchestrator_instance:
        logger_main.critical("Orchestrator instance is None during application_startup. Cannot proceed.")
        sys.exit(1) # Critical failure

    logger_main.info("Quart 'before_serving': Initiating application startup sequence...")
    try:
        # The Orchestrator's run() method should handle its full lifecycle,
        # including initializing DB, clients, and starting agent loops.
        # It should be designed to be called once.
        if not hasattr(orchestrator_instance, '_main_loop_task') or orchestrator_instance._main_loop_task is None or orchestrator_instance._main_loop_task.done():
             # Ensure orchestrator.run() is idempotent or guarded if called multiple times
            orchestrator_instance._main_loop_task = asyncio.create_task(orchestrator_instance.run(), name="OrchestratorMainLoop")
            logger_main.info("Orchestrator's main run() method has been scheduled by Quart startup.")
        else:
            logger_main.info("Orchestrator's main run() method appears to be already tasked or running.")

    except Exception as e_startup_logic:
        logger_main.critical(f"Fatal Error during application_startup logic: {e_startup_logic}", exc_info=True)
        await application_shutdown() # Attempt graceful shutdown even if startup fails partway
        sys.exit(1)

async def application_shutdown():
    """Async operations to perform after the Quart server has stopped."""
    global orchestrator_instance
    logger_main.info("Quart 'after_serving': Initiating application shutdown sequence...")

    if orchestrator_instance and hasattr(orchestrator_instance, 'stop'):
        current_status = getattr(orchestrator_instance, 'status', 'unknown')
        status_stopping = getattr(orchestrator_instance, 'STATUS_STOPPING', 'stopping') # Get const from instance
        status_stopped = getattr(orchestrator_instance, 'STATUS_STOPPED', 'stopped')   # Get const from instance

        if current_status not in [status_stopping, status_stopped]:
            logger_main.info(f"Calling Orchestrator's stop() method (current status: {current_status}).")
            try:
                await orchestrator_instance.stop(timeout=25.0) # Orchestrator handles its agents
            except Exception as e_orch_stop:
                logger_main.error(f"Error during Orchestrator stop(): {e_orch_stop}", exc_info=True)
        else:
            logger_main.info(f"Orchestrator already in stopping/stopped state: {current_status}. Skipping explicit stop.")
    else:
        logger_main.warning("Orchestrator instance not available or has no stop() method for shutdown.")

    # Additional general asyncio task cleanup (optional, good practice)
    # This attempts to cancel tasks not managed by the orchestrator's stop method.
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        logger_main.info(f"Cancelling {len(tasks)} potentially outstanding asyncio tasks...")
        for task in tasks:
            task.cancel()
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger_main.info("Outstanding asyncio tasks cancellation process complete.")
        except asyncio.CancelledError:
            logger_main.info("Gather was cancelled during task cleanup, expected on shutdown.")
        except Exception as e_task_gather:
            logger_main.error(f"Error during gathering of cancelled tasks: {e_task_gather}")


    logger_main.info(f"-------------------- Application Stopping (main.py v3.4) --------------------")
    # logging.shutdown() is typically called automatically by Python's atexit.
    print("[INFO] main.py: Application shutdown sequence completed.")


# --- Register Quart Lifecycle Events ---
# This ensures Quart calls our startup and shutdown functions.
if app_instance:
    app_instance.before_serving(application_startup)
    app_instance.after_serving(application_shutdown)
    logger_main.info("Quart lifecycle events (before_serving, after_serving) registered for app_instance.")
else:
    # This case should have been caught earlier, but as a final safeguard:
    logger_main.critical("app_instance is None. Cannot register Quart lifecycle events. Application will not start via 'quart run'.")
    sys.exit(1)


# --- Main Execution Block (for `python main.py` local development) ---
# This block is primarily for convenience when running locally.
# Production deployments should use `python -m quart run main:app_instance ...`
if __name__ == "__main__":
    if not app_instance:
        logger_main.critical("Fatal: app_instance is None in __main__ block. This indicates a critical failure during module-level initialization.")
        sys.exit(1)

    logger_main.info("Running application directly using 'python main.py' (intended for local development).")
    logger_main.info("For production, the command should be: 'python -m quart run main:app_instance --host 0.0.0.0 --port <PORT>'")

    host = settings.get("QUART_HOST", "127.0.0.1") # Default for local dev
    port = int(settings.get("QUART_PORT", 5000))   # Default for local dev, ensure it's an int
    debug_mode = settings.get("DEBUG", True)      # Enable debug for local dev by default
    use_reloader = settings.get("QUART_USE_RELOADER", True) # Enable reloader for local dev

    logger_main.info(f"Starting Quart development server on http://{host}:{port} (Debug: {debug_mode}, Reloader: {use_reloader})")

    try:
        # The lifecycle events (before_serving, after_serving) will handle
        # the application_startup and application_shutdown logic.
        app_instance.run(
            host=host,
            port=port,
            debug=debug_mode,
            use_reloader=use_reloader,
            # Optionally, pass other dev server config here
        )
    except Exception as e_dev_run:
        logger_main.critical(f"Error running Quart development server: {e_dev_run}", exc_info=True)
        # Attempt a final shutdown if run fails catastrophically even in dev
        asyncio.run(application_shutdown()) # Run shutdown in a new loop if necessary
        sys.exit(1)

# --- End of main.py ---