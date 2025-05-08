# Filename: utils/database.py
# Description: Lean Database Utilities for Postgres with Secure Encryption.
# Version: 3.1 (Level 50+ Transmutation - Added type hints, minor logging improvements)

import os
import base64
import logging
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag
from typing import Optional, AsyncGenerator, Union
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.engine import URL

# Import settings AFTER it's defined and validated
from config.settings import settings, SecretStr # Import SecretStr

# Configure logging
logger = logging.getLogger(__name__)

# --- Encryption Configuration ---
NONCE_BYTES = 12 # Standard for AES-GCM
KEY_BYTES = 32 # AES-256
SALT_BYTES = 16 # Per-value salt size
PBKDF2_ITERATIONS = 600000 # NIST recommended minimum

class EncryptionError(Exception):
    """Custom exception for encryption/decryption errors."""
    pass

# --- Core Key Derivation ---
_MASTER_KEY_CACHE: bytes | None = None

def _get_master_key() -> bytes:
    """Retrieves and caches the database master key from settings."""
    global _MASTER_KEY_CACHE
    if _MASTER_KEY_CACHE is not None:
        return _MASTER_KEY_CACHE

    master_key_secret: Optional[SecretStr] = settings.DATABASE_ENCRYPTION_KEY
    if not master_key_secret:
        error_msg = "CRITICAL: DATABASE_ENCRYPTION_KEY not found in settings."
        logger.critical(error_msg)
        raise EncryptionError(error_msg)

    _MASTER_KEY_CACHE = master_key_secret.get_secret_value().encode('utf-8')
    logger.info("Database master encryption key loaded.")
    return _MASTER_KEY_CACHE

def _derive_key(salt: bytes) -> bytes:
    """Derives the encryption key using PBKDF2HMAC based on the provided salt."""
    if not isinstance(salt, bytes) or len(salt) != SALT_BYTES:
        raise ValueError(f"Salt must be {SALT_BYTES} bytes.")

    master_key = _get_master_key()
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
    Returns Base64 encoded string: salt(16) + nonce(12) + ciphertext+tag.
    Returns None if input data is None.
    Raises EncryptionError on failure.
    """
    if data is None:
        return None
    if not isinstance(data, str):
        logger.error(f"Invalid data type for encryption: {type(data)}")
        raise TypeError("Data to encrypt must be a string.")

    try:
        salt = os.urandom(SALT_BYTES)
        derived_key = _derive_key(salt)
        aesgcm = AESGCM(derived_key)
        nonce = os.urandom(NONCE_BYTES)
        data_bytes = data.encode('utf-8')
        encrypted_bytes_with_tag = aesgcm.encrypt(nonce, data_bytes, None)
        encrypted_payload = salt + nonce + encrypted_bytes_with_tag
        encoded_payload = base64.urlsafe_b64encode(encrypted_payload).decode('utf-8')
        # logger.debug(f"Encryption successful. Payload length: {len(encoded_payload)}") # Reduce verbosity
        return encoded_payload
    except EncryptionError as ee:
        logger.exception(f"Encryption failed due to key derivation error: {ee}")
        raise
    except Exception as e:
        logger.exception(f"Encryption failed unexpectedly: {e}")
        raise EncryptionError(f"Encryption failed: {e}") from e

def decrypt_data(encrypted_data_b64: Optional[str]) -> Optional[str]:
    """
    Decrypts data encrypted by the encrypt_data function (with per-value salt).
    Expects Base64 encoded input: salt(16) + nonce(12) + ciphertext+tag.
    Returns the decrypted string, or None if input is None or decryption fails.
    """
    if encrypted_data_b64 is None:
        return None
    if not isinstance(encrypted_data_b64, str):
        logger.error(f"Invalid data type for decryption: {type(encrypted_data_b64)}")
        return None

    try:
        encrypted_payload = base64.urlsafe_b64decode(encrypted_data_b64)
        min_len = SALT_BYTES + NONCE_BYTES + 16
        if len(encrypted_payload) < min_len:
            logger.error(f"Decryption failed: Payload too short ({len(encrypted_payload)} bytes). Format error or truncation suspected.")
            return None

        salt = encrypted_payload[:SALT_BYTES]
        nonce = encrypted_payload[SALT_BYTES : SALT_BYTES + NONCE_BYTES]
        ciphertext_with_tag = encrypted_payload[SALT_BYTES + NONCE_BYTES :]

        derived_key = _derive_key(salt)
        aesgcm = AESGCM(derived_key)

        decrypted_bytes = aesgcm.decrypt(nonce, ciphertext_with_tag, None)
        decrypted_string = decrypted_bytes.decode('utf-8')
        # logger.debug("Decryption successful.") # Reduce verbosity
        return decrypted_string

    except InvalidTag:
        logger.error("Decryption failed: Invalid authentication tag (data tampered or wrong key/salt/nonce).")
        return None
    except EncryptionError as ee:
        logger.error(f"Decryption failed due to key derivation error: {ee}")
        return None
    except (ValueError, TypeError, IndexError) as e:
        logger.error(f"Decryption failed due to data format or decoding error: {e}. Input snippet: {encrypted_data_b64[:20]}...")
        return None
    except Exception as e:
        logger.exception(f"Decryption failed unexpectedly: {e}")
        return None

# --- Database Session Utility ---
_SESSION_MAKER: Optional[async_sessionmaker[AsyncSession]] = None
_DB_ENGINE = None

def get_db_engine(db_url: Union[str, URL]):
    """Creates and returns a database engine instance."""
    global _DB_ENGINE
    if _DB_ENGINE is None:
        try:
            _DB_ENGINE = create_async_engine(
                db_url,
                echo=settings.DEBUG, # Echo SQL in debug mode only
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=10, # Adjust pool size as needed
                max_overflow=5
            )
            logger.info("Database engine created.")
        except Exception as e:
            logger.critical(f"Failed to create database engine: {e}", exc_info=True)
            raise
    return _DB_ENGINE

def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Creates and returns a singleton async session maker."""
    global _SESSION_MAKER
    if _SESSION_MAKER is None:
        db_url = settings.DATABASE_URL
        if not db_url:
            logger.critical("DATABASE_URL not configured in settings. Cannot create session maker.")
            raise ValueError("Database URL is not configured.")
        try:
            engine = get_db_engine(db_url) # Get or create engine
            _SESSION_MAKER = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            logger.info("Async session maker created successfully.")
        except Exception as e:
            logger.critical(f"Failed to create session maker: {e}", exc_info=True)
            raise
    return _SESSION_MAKER

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Provides an async database session via context manager using the singleton session_maker."""
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
            # Caller handles commit within the 'async with session.begin():' block
        except Exception as e:
            logger.error(f"Exception occurred within database session, rolling back: {e}", exc_info=True)
            await session.rollback()
            raise # Re-raise the exception after rollback
        # Session is automatically closed by the context manager

# --- End of utils/database.py ---