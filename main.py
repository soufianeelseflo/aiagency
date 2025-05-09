# Filename: main.py
# Description: AI Agency Main Entry Point - Handles initialization and graceful shutdown.
# Version: 3.5 (Added pre-settings-import diagnostic logging for env vars)

import asyncio
import logging # Keep this initial basic logging import separate
import os
import sys
import traceback
from dotenv import load_dotenv
from typing import Optional

# --- Step 1: Environment Loading (Absolute First) ---
# This ensures that if a .env.local file is present (e.g., for local dev),
# its values are loaded into the environment. Pydantic will then read from this combined environment.
project_root = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(project_root, '.env.local')

# Load .env.local if it exists. python-dotenv does not overwrite existing system env vars by default.
# If `override=True` is used, .env.local vars would take precedence over system env vars.
# For Coolify, where env vars are set in the UI, we typically want Coolify's vars to be authoritative if there's any conflict.
# Thus, default behavior of `load_dotenv()` (no override) or `override=False` is often preferred when combining with platform-set envs.
# However, if `.env.local` is *only* for local dev and *not* deployed, this is less of an issue.
# The current code has `override=True` in a comment, let's stick to that for now assuming `.env.local` is not part of deployment.
if os.path.exists(dotenv_path):
    loaded_vars = load_dotenv(dotenv_path=dotenv_path, override=True, verbose=True)
    print(f"[INFO] main.py (v3.5): Attempted to load environment variables from: {dotenv_path}. Loaded: {loaded_vars}")
else:
    print(f"[INFO] main.py (v3.5): .env.local file not found at {dotenv_path}. Relying on system/deployment environment variables.")

# --- Step 2: Pre-Settings Import Environment Variable Diagnostics ---
# Log critical environment variables as seen by os.getenv() BEFORE Pydantic tries to load them.
# This helps diagnose if Coolify is correctly injecting them.
print("[DIAGNOSTIC] main.py (v3.5): Checking critical environment variables before settings import:")
critical_env_vars_to_check = [
    "DATABASE_URL", "DATABASE_ENCRYPTION_KEY", "AGENCY_BASE_URL", "OPENROUTER_API_KEY",
    "HOSTINGER_EMAIL", "SENDER_COMPANY_ADDRESS", "TWILIO_ACCOUNT_SID", "TWILIO_VOICE_NUMBER", "DEEPGRAM_API_KEY"
]
for var_name in critical_env_vars_to_check:
    var_value = os.getenv(var_name)
    if var_value:
        # For secrets, just log presence and a few chars if it's long
        display_value = f"Present (value starts with: '{var_value[:4]}...')" if len(var_value) > 8 else f"Present (value: '{var_value}')"
        if var_name in ["DATABASE_ENCRYPTION_KEY", "OPENROUTER_API_KEY", "TWILIO_AUTH_TOKEN", "DEEPGRAM_API_KEY", "HOSTINGER_IMAP_PASS"]: # Add other sensitive keys
             display_value = f"Present (secret)"
        print(f"[DIAGNOSTIC]   '{var_name}': {display_value}")
    else:
        print(f"[DIAGNOSTIC]   '{var_name}': NOT FOUND in environment!")
print("[DIAGNOSTIC] main.py (v3.5): End of pre-settings environment variable check.")

# --- Step 3: Settings Import & Logging Configuration (Reliant on Environment) ---
try:
    from config.settings import settings # This is where Pydantic will try to load from the environment

    log_level_str = settings.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_handlers = [logging.StreamHandler(sys.stdout)]
    log_file_path = settings.LOG_FILE_PATH
    if log_file_path:
        try:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            log_handlers.append(logging.FileHandler(log_file_path, mode='a'))
        except Exception as e_lf:
            print(f"[ERROR] main.py (v3.5): Could not create log file directory for '{log_file_path}': {e_lf}", file=sys.stderr)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=log_handlers,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger_main = logging.getLogger(__name__)

    op_log_file_path = settings.OPERATIONAL_LOG_FILE_PATH
    if op_log_file_path:
        try:
            os.makedirs(os.path.dirname(op_log_file_path), exist_ok=True)
            op_logger = logging.getLogger('OperationalLog')
            op_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            op_file_handler = logging.FileHandler(op_log_file_path, mode='a')
            op_file_handler.setFormatter(op_formatter)
            op_logger.addHandler(op_file_handler)
            op_logger.setLevel(logging.INFO)
            op_logger.propagate = False
        except Exception as e_olf:
            logger_main.error(f"Could not create operational log file directory for '{op_log_file_path}': {e_olf}")

    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.INFO)
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)

    logger_main.info(f"-------------------- Application Starting (main.py v3.5) --------------------")
    logger_main.info(f"Logging configured. Level: {log_level_str}. Main Log File: {log_file_path or 'Console only'}. Op Log File: {op_log_file_path or 'Disabled'}.")
    logger_main.info("Configuration settings object 'settings' loaded successfully.")
    logger_main.debug(f"AGENCY_BASE_URL from settings: {settings.AGENCY_BASE_URL}")
    logger_main.debug(f"DATABASE_URL host from settings: {settings.DATABASE_URL.host if settings.DATABASE_URL else 'N/A'}")

except ImportError as e_imp:
    print(f"CRITICAL IMPORT ERROR (main.py v3.5): Failed to import 'config.settings': {e_imp}. This typically means the file is missing, or there's an issue in settings.py itself or its imports.", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
except Exception as e_settings_init: # Catches Pydantic ValidationErrors from settings.py or other init issues
    print(f"CRITICAL SETTINGS ERROR (main.py v3.5): Failed during settings object initialization or logging setup: {e_settings_init}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# --- Step 4: Core Application Imports (Reliant on Successful Settings) ---
try:
    from agents.orchestrator import Orchestrator
    from quart import Quart
    logger_main.info("Core application components (Orchestrator, Quart) imported successfully.")
except ImportError as e_core_components:
    logger_main.critical(f"Fatal Error (main.py v3.5): Failed to import core application components: {e_core_components}. Check structure/PYTHONPATH.", exc_info=True)
    sys.exit(1)

# --- Step 5: Global Orchestrator and Quart App Instance Creation ---
orchestrator_instance: Optional[Orchestrator] = None
app_instance: Optional[Quart] = None

try:
    orchestrator_instance = Orchestrator(schema='public')
    if orchestrator_instance and hasattr(orchestrator_instance, 'app'):
        app_instance = orchestrator_instance.app
        logger_main.info("Orchestrator initialized and Quart app_instance obtained at module level.")
    else:
        raise RuntimeError("Orchestrator initialization failed or did not provide a Quart app instance.")
except Exception as e_app_init:
    logger_main.critical(f"Fatal Error (main.py v3.5): Failed to initialize Orchestrator or obtain app_instance: {e_app_init}", exc_info=True)
    sys.exit(1)

# --- Application Lifecycle Functions ---
async def application_startup():
    global orchestrator_instance
    if not orchestrator_instance:
        logger_main.critical("Orchestrator instance is None during application_startup. Critical failure.")
        sys.exit(1)
    logger_main.info("Quart 'before_serving': Initiating application startup sequence...")
    try:
        if not hasattr(orchestrator_instance, '_main_loop_task') or orchestrator_instance._main_loop_task is None or orchestrator_instance._main_loop_task.done():
            orchestrator_instance._main_loop_task = asyncio.create_task(orchestrator_instance.run(), name="OrchestratorMainLoop")
            logger_main.info("Orchestrator's main run() method scheduled by Quart startup.")
        else:
            logger_main.info("Orchestrator's main run() method appears to be already tasked or running.")
    except Exception as e_startup_logic:
        logger_main.critical(f"Fatal Error during application_startup logic: {e_startup_logic}", exc_info=True)
        await application_shutdown()
        sys.exit(1)

async def application_shutdown():
    global orchestrator_instance
    logger_main.info("Quart 'after_serving': Initiating application shutdown sequence...")
    if orchestrator_instance and hasattr(orchestrator_instance, 'stop'):
        current_status = getattr(orchestrator_instance, 'status', 'unknown')
        status_stopping = getattr(orchestrator_instance, 'STATUS_STOPPING', 'stopping')
        status_stopped = getattr(orchestrator_instance, 'STATUS_STOPPED', 'stopped')
        if current_status not in [status_stopping, status_stopped]:
            logger_main.info(f"Calling Orchestrator's stop() method (current status: {current_status}).")
            try:
                await orchestrator_instance.stop(timeout=25.0)
            except Exception as e_orch_stop:
                logger_main.error(f"Error during Orchestrator stop(): {e_orch_stop}", exc_info=True)
        else:
            logger_main.info(f"Orchestrator already in stopping/stopped state: {current_status}.")
    else:
        logger_main.warning("Orchestrator instance not available or has no stop() method for shutdown.")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        logger_main.info(f"Cancelling {len(tasks)} potentially outstanding asyncio tasks...")
        for task in tasks: task.cancel()
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger_main.info("Outstanding asyncio tasks cancellation process complete.")
        except asyncio.CancelledError:
            logger_main.info("Gather was cancelled during task cleanup, expected on shutdown.")
        except Exception as e_task_gather:
            logger_main.error(f"Error during gathering of cancelled tasks: {e_task_gather}")
    logger_main.info(f"-------------------- Application Stopping (main.py v3.5) --------------------")
    print("[INFO] main.py (v3.5): Application shutdown sequence completed.")

# --- Register Quart Lifecycle Events ---
if app_instance:
    app_instance.before_serving(application_startup)
    app_instance.after_serving(application_shutdown)
    logger_main.info("Quart lifecycle events registered for app_instance.")
else:
    logger_main.critical("app_instance is None. Cannot register Quart lifecycle events. App will fail to start.")
    sys.exit(1)

# --- Main Execution Block (for `python main.py` local development) ---
if __name__ == "__main__":
    if not app_instance:
        logger_main.critical("Fatal: app_instance is None in __main__ block. Check module-level initialization.")
        sys.exit(1)
    logger_main.info("Running application directly using 'python main.py' (local development).")
    logger_main.info("Production should use: 'python -m quart run main:app_instance --host 0.0.0.0 --port <PORT>'")
    host = settings.get("QUART_HOST", "127.0.0.1")
    port = int(settings.get("QUART_PORT", 5000))
    debug_mode = settings.get("DEBUG", True)
    use_reloader_dev = settings.get("QUART_USE_RELOADER_DEV", True) # Specific for dev run
    logger_main.info(f"Starting Quart development server on http://{host}:{port} (Debug: {debug_mode}, Reloader: {use_reloader_dev})")
    try:
        app_instance.run(host=host, port=port, debug=debug_mode, use_reloader=use_reloader_dev)
    except Exception as e_dev_run:
        logger_main.critical(f"Error running Quart development server: {e_dev_run}", exc_info=True)
        asyncio.run(application_shutdown())
        sys.exit(1)

# --- End of main.py ---