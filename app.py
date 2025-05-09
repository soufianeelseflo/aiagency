# Filename: app.py
# Description: AI Agency Main Entry Point - Handles initialization and graceful shutdown.
# Version: 3.2.2 (Transmuted by IGNIS - Removed .env.local loading, fixed NameError for List)

import asyncio
import logging
import os
import sys
import traceback # For detailed error logging
from typing import List, Any, Optional # Transmuted: Added List, Any, Optional

# --- Environment Variables are expected to be set in the deployment environment (e.g., Coolify) ---
# The .env.local loading mechanism has been removed as it's not used in production
# and was causing informational log clutter.

# --- Settings Import & Validation (Must be early) ---
try:
    from config.settings import settings
    # Basic logging setup using settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    log_file = settings.LOG_FILE_PATH
    op_log_file = settings.OPERATIONAL_LOG_FILE_PATH
    # Transmuted: Corrected type hint usage by importing List and Any
    log_handlers: List[Any] = [logging.StreamHandler(sys.stdout)]
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
        op_logger.setLevel(logging.INFO)
        op_logger.propagate = False # Prevent operational logs from going to root logger

    # Reduce verbosity of common libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.INFO) # Keep websockets a bit more verbose if needed for debugging connections
    logging.getLogger("playwright").setLevel(logging.INFO) # Playwright can be verbose, adjust as needed

    logger = logging.getLogger(__name__) # This will be 'app' as Quart runs it as a module
    logger.info("-------------------- Application Starting (app.py) --------------------")
    logger.info("Logging configured based on settings. System environment variables are used.")
    logger.info("Configuration settings loaded and validated.")

    # Diagnostic print for DATABASE_URL to help with debugging if needed
    try:
        db_url_to_check = str(settings.DATABASE_URL) if settings.DATABASE_URL else "DATABASE_URL NOT SET IN SETTINGS"
        # Use logger.debug for less critical startup info, or logger.info if it's important for every start.
        logger.debug(f"DEBUG_DB_URL_CHECK: DATABASE_URL as read by Settings = '{db_url_to_check}'")
    except Exception as e_diag:
        logger.error(f"DEBUG_DB_URL_CHECK: Error accessing settings.DATABASE_URL: {e_diag}")

except (ImportError, ValueError, SystemExit) as e: # Catch SystemExit if settings validation fails critically
    # Use a basic print here as logging might not be fully set up if settings fail early
    print(f"CRITICAL STARTUP ERROR: Failed during settings import or validation: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1) # Exit if essential settings can't be loaded/validated
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
app: Optional[Quart] = None # Renamed from app_instance to app for Quart CLI compatibility

async def startup():
    """Initializes the orchestrator and starts its background tasks."""
    global orchestrator_instance # 'app' is already globally defined and assigned in __main__
    logger.info("Application startup sequence initiated (Quart before_serving).")
    try:
        # Orchestrator is initialized in the __main__ block before Quart starts serving.
        # This function is for tasks that should run within the Quart event loop context at startup.
        if orchestrator_instance is None:
            logger.error("Orchestrator instance is None at Quart startup. This is unexpected and indicates a logic flaw.")
            # Attempt re-initialization (though this indicates a structural issue if hit)
            # This path should ideally not be taken if __main__ block executes correctly.
            orchestrator_instance = Orchestrator(schema='public') # Assuming 'public' schema
            if app is None: # This check is redundant if orchestrator_instance.app is assigned to global 'app'
                 logger.critical("Quart 'app' instance is None. Orchestrator's app not correctly assigned globally.")
                 raise RuntimeError("Quart app instance not available.")
        
        # The orchestrator's main run loop is started as an asyncio task in the __main__ block.
        # This startup function can be used for other async initializations if needed.
        # For now, we just confirm the orchestrator is running.
        if orchestrator_instance and orchestrator_instance.running:
            logger.info("Orchestrator confirmed to be running its background tasks.")
        elif orchestrator_instance:
            logger.warning("Orchestrator instance exists but is not marked as running. The main task might not have started correctly.")
            # Optionally, try to start it here if it wasn't started, though this might indicate an issue in __main__
            # asyncio.create_task(run_agency_logic(orchestrator_instance), name="AgencyLogicTask_StartupFallback")
        else:
            logger.error("Orchestrator instance is still None in startup. Cannot confirm background tasks.")


    except Exception as e:
        logger.critical(f"Fatal Error during Quart startup event: {e}", exc_info=True)
        await shutdown() # Attempt graceful shutdown
        # Consider exiting if startup fails critically, as the app might be in an inconsistent state.
        # Depending on the deployment, the process manager might restart it.
        # For now, we let Quart handle the exit if it's configured to do so on unhandled startup errors.
        # sys.exit(1) # Uncomment if a hard exit is preferred on startup failure

async def run_agency_logic(orchestrator: Orchestrator):
    """Runs the main orchestrator loop. This is created as a task."""
    logger.info("Starting Orchestrator main execution loop via run_agency_logic task...")
    try:
        await orchestrator.run() # This is orchestrator's main async processing loop
        logger.info("Orchestrator run loop finished normally.")
    except asyncio.CancelledError:
        logger.info("Orchestrator run_agency_logic task was cancelled.")
    except Exception as e:
        logger.critical(f"Fatal Error: Critical error during agency execution (run_agency_logic): {e}", exc_info=True)
        # This exception will be caught by the asyncio task handler.
        # Depending on supervisor setup, it might lead to app restart.
        raise # Re-raise to ensure the task is seen as failed

async def shutdown():
    """Handles graceful shutdown of the application."""
    global orchestrator_instance
    logger.info("Initiating graceful shutdown sequence (Quart after_serving)...")
    if orchestrator_instance and hasattr(orchestrator_instance, 'stop'):
        # Check status to prevent multiple stop calls if shutdown is triggered from multiple places
        if orchestrator_instance.status not in [Orchestrator.STATUS_STOPPING, Orchestrator.STATUS_STOPPED]:
            logger.info("Calling orchestrator stop method...")
            try:
                await orchestrator_instance.stop(timeout=25.0) # timeout for graceful agent shutdown
            except Exception as stop_err:
                logger.error(f"Error during orchestrator stop: {stop_err}", exc_info=True)
        else:
            logger.info(f"Orchestrator already in state: {orchestrator_instance.status}. Skipping redundant stop call.")
    else:
        logger.warning("Orchestrator instance not available or no stop method for final call during shutdown.")
    
    logger.info("Flushing logging handlers...")
    logging.shutdown() # Flushes and closes all handlers
    print("[INFO] app.py: Process stopped, logging shut down.") # Final print to stdout

# --- Main Execution Block ---
# This block is executed when you run `python app.py` or when `quart run` imports `app.py`.
# The `quart run` command will look for an object named `app` of type Quart.
if __name__ == "__main__":
    # This part of the conditional (__name__ == "__main__") is primarily for direct execution (`python app.py`).
    # However, `quart run` also executes the module to find the `app` object.
    # So, initialization here is generally fine, but be mindful of what `quart run` expects.
    try:
        logger.info("Initializing Orchestrator and Quart app instance in __main__ block...")
        orchestrator_instance = Orchestrator(schema='public') # Default schema
        app = orchestrator_instance.app # Assign the Quart app from Orchestrator to the global 'app' variable

        if app is None:
            logger.critical("CRITICAL: Quart application 'app' was not created by Orchestrator.")
            sys.exit("Quart app not initialized. Orchestrator did not provide it.")

        # Register Quart's startup and shutdown events
        # These ensure that our startup/shutdown logic runs within Quart's lifecycle.
        @app.before_serving
        async def before_serving_handler():
            logger.info("Quart 'before_serving' event triggered.")
            await startup()

        @app.after_serving
        async def after_serving_handler():
            logger.info("Quart 'after_serving' event triggered.")
            await shutdown()
        
        # Create and start the main agency logic task
        # This ensures the orchestrator's core processing loop runs in the background.
        # It's important this is created here so it uses the same event loop as Quart.
        if orchestrator_instance:
            logger.info("Creating asyncio task for run_agency_logic.")
            # loop = asyncio.get_event_loop() # Not needed if app.run is used or within Quart's context
            asyncio.create_task(run_agency_logic(orchestrator_instance), name="AgencyLogicTask_Main")
        else:
            logger.critical("Orchestrator instance not created, cannot start agency logic task.")
            sys.exit("Failed to initialize orchestrator for agency logic.")

        logger.info("Setup complete in __main__. Quart app object 'app' is ready.")
        logger.info("If using 'quart run', it will now take over and run the app.")
        
        # For direct execution (python app.py), you would typically add:
        # app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False, use_reloader=False)
        # However, since the deployment uses `quart run`, this direct `app.run` call is not needed
        # and might conflict if `quart run` is also used.
        # The `quart run` command handles serving the `app` object.

    except Exception as e:
        # This catches errors during the initial setup within the `if __name__ == "__main__":` block
        print(f"CRITICAL ERROR during pre-run initialization in app.py (__main__): {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1) # Exit if critical setup fails

# --- End of app.py ---
```

Please replace the content of your `app.py` with the code above. This should resolve the `NameError` and stop the messages about `.env.local` not being found (as it will no longer attempt to load it).

Once you've updated this file and redeployed, please let me know the outcome. We can then proceed with the Decodo.com integration and discuss the environment variabl