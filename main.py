# Filename: main.py
# Description: AI Agency Main Entry Point.
# Version: 2.1 (Production Ready - Robust Init & Shutdown)

import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# --- Environment Loading (Must be first) ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env.local')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    # Use print for immediate feedback before logging is fully configured
    print(f"[INFO] main.py: Loaded environment variables from: {dotenv_path}")
else:
    print(f"[INFO] main.py: .env.local file not found at {dotenv_path}. Relying on system environment variables.")

# --- Basic Logging Setup (Configure early) ---
# Ensure logs go to stdout/stderr for container visibility and to a file
log_file_path = "agency.log"
logging.basicConfig(
    level=logging.INFO, # Default level, consider changing via ENV VAR for debug
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'), # Append to log file
        logging.StreamHandler(sys.stdout) # Log to console
    ],
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Optionally set higher level for noisy libraries
# logging.getLogger("aiohttp").setLevel(logging.WARNING)
# logging.getLogger("asyncio").setLevel(logging.WARNING)

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
     sys.exit(1) # Exit if core components cannot be imported
except ValueError as e:
     logger.critical(f"Fatal Error: Configuration validation failed: {e}. Check environment variables defined in config/settings.py and your .env.local file.", exc_info=True)
     sys.exit(1) # Exit if configuration is invalid
except Exception as e:
     logger.critical(f"Fatal Error: Unexpected error during initial imports: {e}", exc_info=True)
     sys.exit(1)

# Global Orchestrator instance (consider dependency injection for larger apps)
orchestrator_instance: Optional[Orchestrator] = None

async def start_agency():
    """Initializes and runs the AI Agency Orchestrator."""
    global orchestrator_instance
    logger.info("Initializing AI Agency Orchestrator...")
    try:
        # Schema can be default 'public' or configured if needed via settings
        orchestrator_instance = Orchestrator(schema='public') # Settings accessed via self.config inside
        logger.info("Orchestrator instance created. Starting main execution loop...")

        # The run method contains the main async loop for the agency's background tasks
        # It handles internal initialization (DB, clients, agents)
        await orchestrator_instance.run()

        # This line might only be reached if orchestrator.run() finishes normally (e.g., receives a stop signal internally)
        logger.info("Orchestrator run loop finished normally.")

    except ValueError as e:
         # Catch potential ValueErrors during Orchestrator init if settings validation missed something
         logger.critical(f"Fatal Error: Orchestrator initialization failed: {e}", exc_info=True)
         # No orchestrator instance to stop, just exit
         sys.exit(1)
    except RuntimeError as e:
         # Catch RuntimeErrors raised during orchestrator's internal init (e.g., DB/Agent init failure)
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
    try:
        logger.info("Starting AI Agency main task...")
        main_task = asyncio.create_task(start_agency(), name="AgencyMainTask")
        await main_task
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Initiating graceful shutdown...")
        if main_task and not main_task.done():
            main_task.cancel() # Cancel the main agency task
            try:
                await main_task # Allow cancellation to propagate
            except asyncio.CancelledError:
                logger.info("Main agency task successfully cancelled.")
        # Ensure orchestrator stop is called if instance exists
        if orchestrator_instance and hasattr(orchestrator_instance, 'stop'):
             logger.info("Calling orchestrator stop...")
             await orchestrator_instance.stop() # Trigger agent/task shutdown
        else:
             logger.warning("Orchestrator instance not available for stop call during KeyboardInterrupt.")
    except Exception as e:
        # Catch any unexpected errors during asyncio execution itself
        logger.critical(f"Fatal Error: Unhandled exception at main level: {e}", exc_info=True)
    finally:
        logger.info("-------------------- Application Stopping --------------------")
        # Ensure all logs are flushed before exiting
        logging.shutdown()

if __name__ == "__main__":
    # Check Python version if needed
    # if sys.version_info < (3, 8):
    #     sys.exit("Python 3.8 or higher is required.")

    # Run the main async function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This handles Ctrl+C if it happens *very* early before the main() loop starts
        print("\n[INFO] main.py: Shutdown requested via KeyboardInterrupt (early).")
    except Exception as e:
        # Catch errors during asyncio.run() setup itself
        print(f"\n[CRITICAL] main.py: Fatal error during asyncio setup: {e}")
        traceback.print_exc()
    finally:
        print("[INFO] main.py: Process stopped.")

# --- End of main.py ---