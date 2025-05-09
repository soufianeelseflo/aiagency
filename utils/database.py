# Filename: utils/database.py
# Description: Lean Database Utilities for Postgres with Secure Encryption.
# Version: 3.2 (IGNIS Transmutation - Enhanced Logging, Error Specificity, Async Session Maker Robustness)

import os
import base64
import logging
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag
from typing import Optional, AsyncGenerator, Any
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.orm import declarative_base

# Import settings AFTER it's defined and validated, or handle its absence
try:
    from config.settings import settings
    SETTINGS_AVAILABLE = True
except ImportError:
    # This is a fallback if settings.py itself is not found or has errors.
    # Code using `settings` will need to handle its potential absence or use defaults.
    class DummySettings:
        DATABASE_ENCRYPTION_KEY = None
        DATABASE_URL = None
        DEBUG = False
        def get_secret(self, key): return None # type: ignore
        def get(self, key, default=None): return default # type: ignore
    settings = DummySettings() # type: ignore
    SETTINGS_AVAILABLE = False
    logging.getLogger(__name__).critical("CRITICAL: config.settings not found or failed to import in utils/database.py. Database functionality will be severely impaired.")


# Configure logging
logger = logging.getLogger(__name__)
op_logger = logging.getLogger('OperationalLog')

# --- Custom SQLAlchemy Base ---
# Allows models to be defined independently but use this common base.
Base = declarative_base()


# --- Encryption Configuration ---
NONCE_BYTES = 12
KEY_BYTES = 32 # AES-256
SALT_BYTES = 16
PBKDF2_ITERATIONS = 600000 # NIST recommended minimum for new applications

class EncryptionError(Exception):
    """Custom exception for encryption/decryption errors."""
    pass

class DatabaseConfigurationError(Exception):
    """Custom exception for database configuration issues."""
    pass

# --- Core Key Derivation ---
_MASTER_KEY_CACHE: Optional[bytes] = None

def _get_master_key() -> bytes:
    """Retrieves and caches the database master key from settings."""
    global _MASTER_KEY_CACHE
    if _MASTER_KEY_CACHE is not None:
        return _MASTER_KEY_CACHE

    if not SETTINGS_AVAILABLE: # Should not happen if settings is imported above, but as a safeguard
        error_msg = "CRITICAL: Settings module unavailable. Cannot retrieve DATABASE_ENCRYPTION_KEY."
        logger.critical(error_msg)
        op_logger.critical(error_msg)
        raise DatabaseConfigurationError(error_msg)

    master_key_str = settings.get_secret("DATABASE_ENCRYPTION_KEY") # Use get_secret for clarity
    if not master_key_str:
        error_msg = "CRITICAL: DATABASE_ENCRYPTION_KEY not found in settings. Data encryption/decryption will fail."
        logger.critical(error_msg)
        op_logger.critical(error_msg)
        raise DatabaseConfigurationError(error_msg)

    if len(master_key_str) < 32: # Basic check, actual entropy is more important
        error_msg = (f"CRITICAL: DATABASE_ENCRYPTION_KEY is too short (must be >= 32 chars for robust security). "
                     f"Current length: {len(master_key_str)}. Please generate a strong, random key.")
        logger.critical(error_msg)
        op_logger.critical(error_msg)
        raise DatabaseConfigurationError(error_msg)

    _MASTER_KEY_CACHE = master_key_str.encode('utf-8') # Ensure it's bytes
    logger.info("Database master encryption key loaded and cached for use.")
    return _MASTER_KEY_CACHE

def _derive_key(salt: bytes) -> bytes:
    """Derives the encryption key using PBKDF2HMAC based on the provided salt."""
    if not isinstance(salt, bytes) or len(salt) != SALT_BYTES:
        logger.error(f"Invalid salt provided for key derivation. Expected {SALT_BYTES} bytes, got {len(salt)} bytes of type {type(salt)}.")
        raise ValueError(f"Salt must be {SALT_BYTES} bytes of type bytes.")

    master_key = _get_master_key() # This will raise DatabaseConfigurationError if key is bad/missing
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_BYTES,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
        backend=default_backend()
    )
    derived_key = kdf.derive(master_key)
    return derived_key

# --- Per-Value Salt Encryption/Decryption (AES-GCM) ---

def encrypt_data(data: Optional[str]) -> Optional[str]:
    """
    Encrypts string data using AES-GCM with a unique salt and nonce per operation.
    Returns Base64 encoded string: salt + nonce + ciphertext_and_tag.
    Returns None if input data is None.
    Raises EncryptionError on failure, DatabaseConfigurationError if key is misconfigured.
    """
    if data is None:
        return None
    if not isinstance(data, str):
        logger.error(f"Invalid data type for encryption: {type(data)}. Expected str.")
        raise TypeError("Data to encrypt must be a string.")

    try:
        salt = os.urandom(SALT_BYTES)
        derived_key = _derive_key(salt) # Can raise DatabaseConfigurationError
        aesgcm = AESGCM(derived_key)
        nonce = os.urandom(NONCE_BYTES)
        data_bytes = data.encode('utf-8')
        encrypted_bytes_with_tag = aesgcm.encrypt(nonce, data_bytes, None)
        encrypted_payload = salt + nonce + encrypted_bytes_with_tag
        encoded_payload = base64.urlsafe_b64encode(encrypted_payload).decode('utf-8')
        logger.debug(f"Encryption successful. Original length: {len(data)}, Encoded length: {len(encoded_payload)}")
        return encoded_payload
    except DatabaseConfigurationError: # Propagate config errors clearly
        raise
    except Exception as e:
        logger.exception(f"Encryption failed unexpectedly: {e}")
        raise EncryptionError(f"Encryption process failed: {e}") from e

def decrypt_data(encrypted_data_b64: Optional[str]) -> Optional[str]:
    """
    Decrypts data encrypted by the encrypt_data function.
    Expects Base64 encoded input: salt + nonce + ciphertext_and_tag.
    Returns the decrypted string.
    Returns None if input is None.
    Returns None and logs error if decryption fails for other reasons (e.g. tampered, wrong key, corrupted).
    Can raise DatabaseConfigurationError if master key is misconfigured during _derive_key.
    """
    if encrypted_data_b64 is None:
        return None
    if not isinstance(encrypted_data_b64, str):
        logger.error(f"Invalid data type for decryption: {type(encrypted_data_b64)}. Expected str.")
        return None # Or raise TypeError based on strictness needed

    try:
        encrypted_payload = base64.urlsafe_b64decode(encrypted_data_b64.encode('utf-8'))
        min_len = SALT_BYTES + NONCE_BYTES + 16 # AES-GCM tag is 16 bytes
        if len(encrypted_payload) < min_len:
            logger.error(f"Decryption failed: Payload too short ({len(encrypted_payload)} bytes). Expected format: salt+nonce+ciphertext+tag. Input: '{encrypted_data_b64[:30]}...'")
            return None

        salt = encrypted_payload[:SALT_BYTES]
        nonce = encrypted_payload[SALT_BYTES : SALT_BYTES + NONCE_BYTES]
        ciphertext_with_tag = encrypted_payload[SALT_BYTES + NONCE_BYTES :]

        derived_key = _derive_key(salt) # Can raise DatabaseConfigurationError
        aesgcm = AESGCM(derived_key)

        decrypted_bytes = aesgcm.decrypt(nonce, ciphertext_with_tag, None)
        decrypted_string = decrypted_bytes.decode('utf-8')
        logger.debug("Decryption successful.")
        return decrypted_string
    except InvalidTag:
        logger.error("Decryption failed: Invalid authentication tag (data tampered, wrong key, or corrupted).")
        return None
    except DatabaseConfigurationError: # Propagate critical config errors
        logger.critical("Decryption impossible due to database encryption key configuration error during key derivation.")
        raise # Re-raise as this is a configuration issue stopping the process.
    except (ValueError, TypeError, IndexError) as e: # Catch Base64, slicing, or decode errors
        logger.error(f"Decryption failed due to data format or decoding error: {e}. Input: '{encrypted_data_b64[:30]}...'")
        return None
    except Exception as e:
        logger.exception(f"Decryption failed unexpectedly: {e}")
        return None

# --- Database Session Utility ---
_SESSION_MAKER_SINGLETON: Optional[async_sessionmaker[AsyncSession]] = None

def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Creates and returns a singleton async session maker."""
    global _SESSION_MAKER_SINGLETON
    if _SESSION_MAKER_SINGLETON is None:
        if not SETTINGS_AVAILABLE:
            err_msg = "CRITICAL: Settings module unavailable. Cannot retrieve DATABASE_URL for session maker."
            logger.critical(err_msg)
            op_logger.critical(err_msg)
            raise DatabaseConfigurationError(err_msg)

        db_url = settings.get("DATABASE_URL")
        if not db_url:
            err_msg = "CRITICAL: DATABASE_URL not configured in settings. Cannot create session maker."
            logger.critical(err_msg)
            op_logger.critical(err_msg)
            raise DatabaseConfigurationError(err_msg)
        try:
            engine = create_async_engine(
                str(db_url),
                echo=settings.get_bool("DEBUG", False),
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_timeout=30,
                connect_args={
                    "server_settings": {"application_name": "NolliAIAgency"},
                    # Consider adding SSL args if connecting to a managed DB that requires it
                    # "ssl": "prefer" # or "require" depending on DB setup
                }
            )
            _SESSION_MAKER_SINGLETON = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            logger.info("Async session maker created successfully for database.")
            op_logger.info("Database session manager initialized.")
        except Exception as e: # Catch broader exceptions during engine/session maker creation
            logger.critical(f"CRITICAL: Failed to create database engine or session maker for URL '{db_url}': {e}", exc_info=True)
            op_logger.critical(f"Database session maker creation FAILED: {e}")
            raise DatabaseConfigurationError(f"Failed to create DB engine/session maker: {e}") from e
    return _SESSION_MAKER_SINGLETON

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Provides an async database session via context manager."""
    try:
        session_maker = get_session_maker() # Can raise DatabaseConfigurationError
    except DatabaseConfigurationError:
        logger.critical("Failed to get session maker due to configuration error. Database operations will fail.")
        # Decide if to raise or yield None. Raising is usually better to halt operations.
        raise # Re-raise to make it clear that DB is unusable.

    async with session_maker() as session:
        try:
            yield session
            # Implicit commit is not handled here; caller should commit.
        except OperationalError as oe: # Specific type of SQLAlchemyError for connection issues
            logger.error(f"Database operational error occurred: {oe}", exc_info=True)
            op_logger.error(f"DB Transaction Operational ERROR: {oe}")
            try:
                await session.rollback()
                logger.info("Session rolled back due to OperationalError.")
            except Exception as rb_err:
                logger.error(f"Failed to rollback session after OperationalError: {rb_err}", exc_info=True)
            raise # Re-raise the original OperationalError
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error occurred within database session: {e}", exc_info=True)
            op_logger.error(f"DB Transaction SQLAlchemy ERROR: {e}")
            try:
                await session.rollback()
                logger.info("Session rolled back due to SQLAlchemyError.")
            except Exception as rb_err:
                logger.error(f"Failed to rollback session after SQLAlchemyError: {rb_err}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Generic exception occurred within database session: {e}", exc_info=True)
            op_logger.error(f"DB Session ERROR (Non-SQLAlchemy): {e}")
            try:
                await session.rollback()
                logger.info("Session rolled back due to generic exception.")
            except Exception as rb_err:
                logger.error(f"Failed to rollback session after generic exception: {rb_err}", exc_info=True)
            raise
        # Session is automatically closed by the async context manager

# --- End of utils/database.py ---