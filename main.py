# Filename: main.py
# Description: AI Agency Main Entry Point.
# Version: 3.0 (Genius Agentic - Production Ready)

import asyncio
import logging
import os
import sys
import traceback # For detailed error logging
from dotenv import load_dotenv

# --- Environment Loading (Must be first) ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env.local')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"[INFO] main.py: Loaded environment variables from: {dotenv_path}") # Use print before logging
else:
    print(f"[INFO] main.py: .env.local file not found at {dotenv_path}. Relying on system environment variables.")

# --- Basic Logging Setup (Configure early) ---
log_file_path = "agency.log"
logging.basicConfig(
    level=logging.INFO, # Consider using os.getenv("LOG_LEVEL", "INFO").upper()
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),
        logging.StreamHandler(sys.stdout)
    ],
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Silence noisy libraries if needed
# logging.getLogger("asyncio").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)
# logging.getLogger("playwright").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info("-------------------- Application Starting --------------------")
logger.info("Basic logging configured.")

# --- Core Imports & Initialization ---
# Import necessary components *after* env vars are loaded and logging is basic configured
try:
    # Settings validation runs automatically on import
    from config.settings import settings
    logger.info("Configuration settings loaded and validated.")
    # Import the main orchestrator class
    from agents.orchestrator import Orchestrator
    logger.info("Orchestrator class imported successfully.")
except ImportError as e:
     logger.critical(f"Fatal Error: Failed to import core components: {e}. Check project structure and PYTHONPATH.", exc_info=True)
     sys.exit(1)
except ValueError as e:
     # Catches errors from settings validation
     logger.critical(f"Fatal Error: Configuration validation failed: {e}. Check environment variables.", exc_info=True)
     sys.exit(1)
except Exception as e:
     logger.critical(f"Fatal Error: Unexpected error during initial imports: {e}", exc_info=True)
     sys.exit(1)

# Global Orchestrator instance
orchestrator_instance: Optional[Orchestrator] = None

async def start_agency():
    """Initializes and runs the AI Agency Orchestrator."""
    global orchestrator_instance
    logger.info("Initializing AI Agency Orchestrator...")
    try:
        # Pass the validated settings object to the orchestrator if needed,
        # but current design reads from imported settings object directly.
        orchestrator_instance = Orchestrator(schema='public') # Assuming default schema
        logger.info("Orchestrator instance created. Starting main execution loop...")

        # The run method contains the main async loop for the agency's background tasks
        # It handles internal initialization (DB, clients, agents)
        await orchestrator_instance.run()

        # This line might only be reached if orchestrator.run() finishes normally
        logger.info("Orchestrator run loop finished normally.")

    except ValueError as e:
         logger.critical(f"Fatal Error: Orchestrator initialization failed: {e}", exc_info=True)
         sys.exit(1)
    except RuntimeError as e:
         logger.critical(f"Fatal Error: Orchestrator internal initialization failed: {e}", exc_info=True)
         sys.exit(1)
    except Exception as e:
        logger.critical(f"Fatal Error: Critical error during agency execution: {e}", exc_info=True)
        # Attempt graceful shutdown if orchestrator was partially initialized
        if orchestrator_instance and hasattr(orchestrator_instance, 'stop'):
            logger.info("Attempting emergency shutdown...")
            await orchestrator_instance.stop()
        sys.exit(1) # Exit after critical error

async def main():
    """Main entry point with graceful shutdown handling."""
    global orchestrator_instance
    main_task = None
    # Get the Quart app instance from the orchestrator
    # We need to initialize orchestrator first to get the app
    temp_orchestrator = None
    try:
        temp_orchestrator = Orchestrator(schema='public')
        app = temp_orchestrator.app # Get the Quart app instance
    except Exception as init_err:
        logger.critical(f"Fatal: Could not initialize Orchestrator to get Quart app: {init_err}", exc_info=True)
        sys.exit(1)

    # Configure Quart server task
    from hypercorn.config import Config as HypercornConfig
    from hypercorn.asyncio import serve as hypercorn_serve

    hypercorn_config = HypercornConfig()
    hypercorn_config.bind = ["0.0.0.0:5000"] # Bind to all interfaces on port 5000
    hypercorn_config.accesslog = '-' # Log access to stdout
    hypercorn_config.errorlog = '-' # Log errors to stderr

    # Create tasks for the agency logic and the web server
    agency_task = asyncio.create_task(start_agency(), name="AgencyMainTask")
    server_task = asyncio.create_task(hypercorn_serve(app, hypercorn_config), name="WebServerTask")

    # Wait for either task to complete (or fail)
    done, pending = await asyncio.wait(
        [agency_task, server_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Handle completion/failure
    for task in done:
        try:
            task.result() # Raise exception if task failed
            logger.info(f"Task {task.get_name()} completed normally.")
        except asyncio.CancelledError:
             logger.info(f"Task {task.get_name()} was cancelled.")
        except Exception as e:
            logger.critical(f"Task {task.get_name()} failed critically: {e}", exc_info=True)
            # If one task fails, we likely want to stop the other
            for p_task in pending:
                if not p_task.done():
                    p_task.cancel()

    # Ensure pending tasks are cancelled if loop exits
    for task in pending:
        if not task.done():
            task.cancel()
            try:
                await task # Allow cancellation to process
            except asyncio.CancelledError:
                logger.info(f"Pending task {task.get_name()} cancelled during shutdown.")
            except Exception as e:
                 logger.error(f"Error during cancellation of pending task {task.get_name()}: {e}")

    # Final orchestrator stop if needed (might be redundant if agency_task completed/failed)
    if orchestrator_instance and orchestrator_instance.running:
        logger.info("Ensuring orchestrator stop is called...")
        await orchestrator_instance.stop()

if __name__ == "__main__":
    logger.info("Starting AI Agency...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agency shutdown requested via KeyboardInterrupt (Ctrl+C).")
    except Exception as e:
        logger.critical(f"Fatal Error: Unhandled exception at top level: {e}", exc_info=True)
        traceback.print_exc() # Print traceback directly for critical errors
    finally:
        logger.info("-------------------- Application Stopping --------------------")
        logging.shutdown() # Ensure all logs are flushed before exiting
        print("[INFO] main.py: Process stopped.") # Final print statement

# --- End of main.py ---