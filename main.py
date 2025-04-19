# main.py - AI Agency Entry Point
import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# --- Environment Loading ---
# Load .env file first to ensure settings are available
dotenv_path = os.path.join(os.path.dirname(__file__), '.env.local')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    # Use print for immediate feedback before logging is fully configured
    print(f"[INFO] Loaded environment variables from: {dotenv_path}")
else:
    print(f"[INFO] .env.local file not found at {dotenv_path}. Relying on system environment variables.")

# --- Basic Logging Setup ---
# Configure logging early, can be refined by specific modules/settings later
logging.basicConfig(
    level=logging.INFO, # Default level, can be overridden by env var if needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agency.log", mode='a'), # Append to log file
        logging.StreamHandler(sys.stdout) # Also log to console
    ]
)
logger = logging.getLogger(__name__)
logger.info("Basic logging configured.")

# --- Core Imports & Initialization ---
# Import necessary components *after* env vars are loaded
try:
    # Settings validation runs automatically on import
    from config.settings import settings
    logger.info("Configuration settings loaded and validated.")
    # Import the main orchestrator class
    from agents.orchestrator import Orchestrator
    logger.info("Orchestrator class imported successfully.")
except ImportError as e:
     # Log critical import errors and exit
     logger.critical(f"Fatal Error: Failed to import core components: {e}. Check project structure and PYTHONPATH.", exc_info=True)
     sys.exit(1) # Exit if core components cannot be imported
except ValueError as e:
     # Log critical configuration errors (e.g., missing required env vars from settings validation)
     logger.critical(f"Fatal Error: Configuration validation failed: {e}. Check environment variables defined in config/settings.py and your .env.local file.", exc_info=True)
     sys.exit(1) # Exit if configuration is invalid
except Exception as e:
     # Catch any other unexpected errors during import/initial setup
     logger.critical(f"Fatal Error: Unexpected error during initial setup: {e}", exc_info=True)
     sys.exit(1)


async def start_agency():
    """Initializes and runs the AI Agency Orchestrator."""
    logger.info("Initializing AI Agency Orchestrator...")
    orchestrator = None # Initialize to None for finally block
    try:
        # Schema can be default 'public' or configured if needed via settings
        # Pass the validated settings object to the orchestrator
        orchestrator = Orchestrator(schema='public') # Assuming default schema, settings are accessed via self.config inside Orchestrator
        logger.info("Orchestrator initialized. Starting main execution loop...")
        # The run method contains the main async loop for the agency
        await orchestrator.run()
        logger.info("Orchestrator run loop finished.") # Should ideally not be reached unless stopped gracefully
    except ValueError as e:
         # Catch potential ValueErrors during Orchestrator init if settings validation missed something
         logger.critical(f"Fatal Error: Orchestrator initialization failed due to configuration issue: {e}", exc_info=True)
    except KeyboardInterrupt:
         logger.info("KeyboardInterrupt received during agency execution. Attempting graceful shutdown...")
         # Add any specific shutdown logic here if needed (e.g., orchestrator.stop())
    except Exception as e:
        logger.critical(f"Fatal Error: Critical error during agency execution: {e}", exc_info=True)
        # Consider more specific error handling or shutdown procedures
    finally:
        logger.info("Agency shutdown sequence initiated.")
        # Add any final cleanup tasks here, regardless of success or failure
        # e.g., closing database connections if not handled by context managers

if __name__ == "__main__":
    logger.info("Starting AI Agency...")
    try:
        # Run the main async function
        asyncio.run(start_agency())
    except KeyboardInterrupt:
        # This handles Ctrl+C at the top level before asyncio loop starts or after it exits
        logger.info("Agency shutdown requested via KeyboardInterrupt (Ctrl+C).")
    except Exception as e:
        # Catch any unexpected errors during asyncio.run itself
        logger.critical(f"Fatal Error: Unhandled exception at top level: {e}", exc_info=True)
    finally:
        logger.info("AI Agency process stopped.")
        logging.shutdown() # Ensure all logs are flushed