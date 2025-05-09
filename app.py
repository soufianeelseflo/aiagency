# Filename: app.py
# Description: AI Agency Main Entry Point - Handles initialization and graceful shutdown.
# Version: 3.2.3 (Transmuted by IGNIS - Removed extraneous text causing SyntaxError)

import asyncio
import logging
import os
import sys
import traceback # For detailed error logging
from typing import List, Any, Optional

# --- Environment Variables are expected to be set in the deployment environment (e.g., Coolify) ---

# --- Settings Import & Validation (Must be early) ---
try:
    from config.settings import settings
    # Basic logging setup using settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    log_file = settings.LOG_FILE_PATH
    op_log_file = settings.OPERATIONAL_LOG_FILE_PATH
    
    log_handlers: List[Any] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir: 
            os.makedirs(log_dir, exist_ok=True)
        log_handlers.append(logging.FileHandler(log_file, mode='a'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=log_handlers,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if op_log_file:
        op_log_dir = os.path.dirname(op_log_file)
        if op_log_dir:
             os.makedirs(op_log_dir, exist_ok=True)
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
    logger.info("-------------------- Application Starting (app.py) --------------------")
    logger.info("Logging configured based on settings. System environment variables are used.")
    logger.info("Configuration settings loaded and validated.")

    try:
        db_url_to_check = str(settings.DATABASE_URL) if settings.DATABASE_URL else "DATABASE_URL NOT SET IN SETTINGS"
        logger.debug(f"DEBUG_DB_URL_CHECK: DATABASE_URL as read by Settings = '{db_url_to_check}'")
    except Exception as e_diag:
        logger.error(f"DEBUG_DB_URL_CHECK: Error accessing settings.DATABASE_URL: {e_diag}")

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
    from quart import Quart 
    logger.info("Core components (Orchestrator, Quart) imported successfully.")
except ImportError as e:
    logger.critical(f"Fatal Error: Failed to import core components: {e}. Check project structure and PYTHONPATH.", exc_info=True)
    sys.exit(1)

# --- Global Orchestrator and Quart App Instances ---
orchestrator_instance: Optional[Orchestrator] = None
app: Optional[Quart] = None 

async def startup():
    """Initializes the orchestrator and starts its background tasks."""
    global orchestrator_instance 
    logger.info("Application startup sequence initiated (Quart before_serving).")
    try:
        if orchestrator_instance is None:
            logger.error("Orchestrator instance is None at Quart startup. This is unexpected and indicates a logic flaw.")
            # This path should ideally not be taken if __main__ block executes correctly.
            # For robustness, one might re-initialize, but it's better to ensure __main__ sets it.
            # orchestrator_instance = Orchestrator(schema='public') 
            # if app is None: # This check is redundant if orchestrator_instance.app is assigned to global 'app'
            #      logger.critical("Quart 'app' instance is None. Orchestrator's app not correctly assigned globally.")
            #      raise RuntimeError("Quart app instance not available.")
            raise RuntimeError("Orchestrator not initialized before Quart startup. Critical error in __main__.")

        if orchestrator_instance and orchestrator_instance.running:
            logger.info("Orchestrator confirmed to be running its background tasks.")
        elif orchestrator_instance:
            logger.warning("Orchestrator instance exists but is not marked as running. The main task might not have started correctly.")
        else: # This case should be caught by the RuntimeError above if orchestrator_instance is None
            logger.error("Orchestrator instance is still None in startup. Cannot confirm background tasks.")

    except Exception as e:
        logger.critical(f"Fatal Error during Quart startup event: {e}", exc_info=True)
        await shutdown() 

async def run_agency_logic(orchestrator: Orchestrator):
    """Runs the main orchestrator loop. This is created as a task."""
    logger.info("Starting Orchestrator main execution loop via run_agency_logic task...")
    try:
        await orchestrator.run() 
        logger.info("Orchestrator run loop finished normally.")
    except asyncio.CancelledError:
        logger.info("Orchestrator run_agency_logic task was cancelled.")
    except Exception as e:
        logger.critical(f"Fatal Error: Critical error during agency execution (run_agency_logic): {e}", exc_info=True)
        raise 

async def shutdown():
    """Handles graceful shutdown of the application."""
    global orchestrator_instance
    logger.info("Initiating graceful shutdown sequence (Quart after_serving)...")
    if orchestrator_instance and hasattr(orchestrator_instance, 'stop'):
        # Check status to prevent multiple stop calls if shutdown is triggered from multiple places
        # Assuming Orchestrator.STATUS_STOPPING and Orchestrator.STATUS_STOPPED are defined
        if orchestrator_instance.status not in [getattr(Orchestrator, "STATUS_STOPPING", "stopping"), getattr(Orchestrator, "STATUS_STOPPED", "stopped")]:
            logger.info("Calling orchestrator stop method...")
            try:
                await orchestrator_instance.stop(timeout=25.0) 
            except Exception as stop_err:
                logger.error(f"Error during orchestrator stop: {stop_err}", exc_info=True)
        else:
            logger.info(f"Orchestrator already in state: {orchestrator_instance.status}. Skipping redundant stop call.")
    else:
        logger.warning("Orchestrator instance not available or no stop method for final call during shutdown.")
    
    logger.info("Flushing logging handlers...")
    logging.shutdown() 
    print("[INFO] app.py: Process stopped, logging shut down.") 

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        logger.info("Initializing Orchestrator and Quart app instance in __main__ block...")
        # Ensure Orchestrator is initialized here so 'app' can be found by Quart runner
        orchestrator_instance = Orchestrator(schema='public') 
        app = orchestrator_instance.app 

        if app is None:
            logger.critical("CRITICAL: Quart application 'app' was not created by Orchestrator.")
            sys.exit("Quart app not initialized. Orchestrator did not provide it.")

        @app.before_serving
        async def before_serving_handler():
            logger.info("Quart 'before_serving' event triggered.")
            await startup()

        @app.after_serving
        async def after_serving_handler():
            logger.info("Quart 'after_serving' event triggered.")
            await shutdown()
        
        if orchestrator_instance:
            logger.info("Creating asyncio task for run_agency_logic.")
            asyncio.create_task(run_agency_logic(orchestrator_instance), name="AgencyLogicTask_Main")
        else: # Should not be reached if app is not None
            logger.critical("Orchestrator instance not created, cannot start agency logic task.")
            sys.exit("Failed to initialize orchestrator for agency logic.")

        logger.info("Setup complete in __main__. Quart app object 'app' is ready.")
        logger.info("If using 'quart run', it will now take over and run the app.")
        
    except Exception as e:
        print(f"CRITICAL ERROR during pre-run initialization in app.py (__main__): {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1) 

# --- End of app.py ---
