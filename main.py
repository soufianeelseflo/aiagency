# Filename: main.py
# Description: AI Agency Main Entry Point - Handles initialization, web server, and graceful shutdown.
# Version: 3.1 (Genius Agentic - Refactored Orchestrator Handling)

import asyncio
import logging
import os
import sys
import traceback # For detailed error logging
from dotenv import load_dotenv
from typing import Optional

# --- Environment Loading (Must be first) ---
# Determine the root directory of the project (assuming main.py is at the root)
project_root = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(project_root, '.env.local')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"[INFO] main.py: Loaded environment variables from: {dotenv_path}") # Use print before logging
else:
    print(f"[INFO] main.py: .env.local file not found at {dotenv_path}. Relying on system environment variables.")

# --- Settings Import & Validation (Must be early) ---
# This will also trigger settings validation and potentially exit if critical vars are missing
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

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=log_handlers,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup separate operational logger if path is provided
    if op_log_file:
        os.makedirs(os.path.dirname(op_log_file), exist_ok=True)
        op_logger = logging.getLogger('OperationalLog')
        op_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        op_handler = logging.FileHandler(op_log_file, mode='a')
        op_handler.setFormatter(op_formatter)
        op_logger.addHandler(op_handler)
        op_logger.setLevel(logging.INFO) # Operational log usually INFO level
        op_logger.propagate = False # Don't send operational logs to root logger

    # Silence noisy libraries if needed (after basicConfig)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.INFO) # Can be noisy on DEBUG
    # Playwright can be very verbose, adjust level as needed
    logging.getLogger("playwright").setLevel(logging.INFO)

    logger = logging.getLogger(__name__) # Get logger after setup
    logger.info("-------------------- Application Starting --------------------")
    logger.info("Logging configured based on settings.")
    logger.info("Configuration settings loaded and validated.")

except (ImportError, ValueError, SystemExit) as e:
    # Catch errors during settings import/validation
    # Use print because logging might not be fully configured if settings failed
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
    from quart import Quart # Import Quart here
    from hypercorn.config import Config as HypercornConfig
    from hypercorn.asyncio import serve as hypercorn_serve
    logger.info("Core components (Orchestrator, Quart, Hypercorn) imported successfully.")
except ImportError as e:
    logger.critical(f"Fatal Error: Failed to import core components: {e}. Check project structure and PYTHONPATH.", exc_info=True)
    sys.exit(1)

# --- Global Orchestrator Instance ---
# This single instance will be used by both the agency logic and the web server
orchestrator_instance: Optional[Orchestrator] = None

async def run_agency_logic(orchestrator: Orchestrator):
    """Runs the main orchestrator loop."""
    logger.info("Starting Orchestrator main execution loop...")
    try:
        # The run method handles internal initialization (DB, clients, agents)
        # and the main async loop for the agency's background tasks
        await orchestrator.run()
        logger.info("Orchestrator run loop finished normally.")
    except Exception as e:
        logger.critical(f"Fatal Error: Critical error during agency execution: {e}", exc_info=True)
        # Signal shutdown if a critical error occurs in the core loop
        raise # Re-raise to be caught by main()

async def run_web_server(app: Quart, config: HypercornConfig):
    """Runs the Hypercorn web server."""
    logger.info(f"Starting web server on {config.bind}...")
    try:
        await hypercorn_serve(app, config)
        logger.info("Web server finished normally.")
    except Exception as e:
        logger.critical(f"Fatal Error: Critical error during web server execution: {e}", exc_info=True)
        # Signal shutdown if a critical error occurs in the web server
        raise # Re-raise to be caught by main()

async def main():
    """Main entry point: Initializes orchestrator, starts agency & web server, handles shutdown."""
    global orchestrator_instance
    agency_task = None
    server_task = None

    try:
        # 1. Initialize the SINGLE Orchestrator instance
        logger.info("Initializing AI Agency Orchestrator...")
        orchestrator_instance = Orchestrator(schema='public') # Assuming default schema
        app = orchestrator_instance.app # Get the Quart app instance from the orchestrator
        logger.info("Orchestrator instance created successfully.")

        # 2. Configure Hypercorn
        hypercorn_config = HypercornConfig()
        # Bind to 0.0.0.0 to be accessible within Docker/Coolify
        hypercorn_config.bind = [f"0.0.0.0:{os.getenv('PORT', '5000')}"] # Use PORT env var if set by Coolify/platform
        hypercorn_config.accesslog = '-' if settings.DEBUG else None # Log access only in debug
        hypercorn_config.errorlog = '-' # Log errors to stderr (captured by Docker logs)
        hypercorn_config.loglevel = settings.LOG_LEVEL.lower() # Sync Hypercorn log level

        # 3. Create and run tasks concurrently
        logger.info("Creating agency logic and web server tasks...")
        agency_task = asyncio.create_task(run_agency_logic(orchestrator_instance), name="AgencyLogicTask")
        server_task = asyncio.create_task(run_web_server(app, hypercorn_config), name="WebServerTask")

        # Wait for either task to complete or fail
        done, pending = await asyncio.wait(
            [agency_task, server_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # 4. Handle Task Completion/Failure
        for task in done:
            task_name = task.get_name()
            try:
                task.result() # Raise exception if task failed
                logger.warning(f"Task {task_name} completed unexpectedly (should normally run forever). Initiating shutdown.")
            except asyncio.CancelledError:
                logger.info(f"Task {task_name} was cancelled.")
            except Exception as e:
                logger.critical(f"Task {task_name} failed critically: {e}", exc_info=True)
                # If one task fails, cancel the other to initiate shutdown
                for p_task in pending:
                    if not p_task.done():
                        logger.info(f"Cancelling pending task {p_task.get_name()} due to failure in {task_name}.")
                        p_task.cancel()

    except ValueError as e:
        logger.critical(f"Fatal Error: Orchestrator initialization failed: {e}", exc_info=True)
    except RuntimeError as e:
        logger.critical(f"Fatal Error: Orchestrator internal initialization failed: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"Fatal Error: Unhandled exception in main setup: {e}", exc_info=True)

    finally:
        # 5. Graceful Shutdown
        logger.info("Initiating graceful shutdown sequence...")

        # Cancel any remaining pending tasks (agency or server)
        all_tasks = [t for t in [agency_task, server_task] if t and not t.done()]
        if all_tasks:
            logger.info(f"Cancelling {len(all_tasks)} running tasks...")
            for task in all_tasks:
                task.cancel()
            # Wait for cancellations to complete
            await asyncio.gather(*all_tasks, return_exceptions=True)
            logger.info("Running tasks cancellation complete.")

        # Explicitly stop the orchestrator (handles agent shutdown)
        if orchestrator_instance and hasattr(orchestrator_instance, 'stop'):
             # Check if orchestrator is already stopping/stopped to avoid redundant calls
            if orchestrator_instance.status not in [orchestrator_instance.STATUS_STOPPING, orchestrator_instance.STATUS_STOPPED]:
                logger.info("Calling orchestrator stop method...")
                try:
                    await orchestrator_instance.stop(timeout=25.0) # Give agents time to stop
                except Exception as stop_err:
                    logger.error(f"Error during orchestrator stop: {stop_err}", exc_info=True)
            else:
                 logger.info(f"Orchestrator already in state: {orchestrator_instance.status}. Skipping stop call.")
        else:
             logger.warning("Orchestrator instance not available for final stop call.")

        logger.info("-------------------- Application Stopping --------------------")
        logging.shutdown() # Ensure all logs are flushed before exiting
        print("[INFO] main.py: Process stopped.") # Final print statement

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested via KeyboardInterrupt (Ctrl+C).")
    except SystemExit as e:
         logger.info(f"System exit requested: {e}")
    except Exception as e:
        # Catch any unexpected errors that might occur outside the main async loop
        print(f"CRITICAL UNHANDLED ERROR at top level: {e}", file=sys.stderr)
        traceback.print_exc()
        logging.shutdown() # Attempt to flush logs
        sys.exit(1) # Ensure non-zero exit code on critical failure

# --- End of main.py ---
