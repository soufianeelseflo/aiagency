# Filename: utils/database.py
# Description: Secure Data Handling & Postgres Session Management.
# Version: 2.0 (Genius Agentic - Postgres Focused, Per-Value Salts)

import os
import base64
import logging
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag
from typing import Optional, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Import settings AFTER it's defined and validated
from config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

# --- Encryption Configuration ---
NONCE_BYTES = 12
KEY_BYTES = 32
SALT_BYTES = 16 # Used for new per-value salts
PBKDF2_ITERATIONS = 600000 # Increased iterations for better security

class EncryptionError(Exception):
    """Custom exception for encryption/decryption errors."""
    pass

# --- Core Key Derivation ---
# Cache the master key bytes to avoid reading env var repeatedly
_MASTER_KEY_CACHE: bytes | None = None

def _get_master_key() -> bytes:
    """Retrieves and caches the database master key."""
    global _MASTER_KEY_CACHE
    if _MASTER_KEY_CACHE is not None:
        return _MASTER_KEY_CACHE

    # ### Phase 1 Plan Ref: 2.1 (Use settings for key)
    master_key_str = settings.DATABASE_ENCRYPTION_KEY
    if not master_key_str:
        error_msg = "CRITICAL: DATABASE_ENCRYPTION_KEY environment variable not set in settings."
        logger.critical(error_msg)
        raise EncryptionError(error_msg) # Fail hard if key is missing

    _MASTER_KEY_CACHE = master_key_str.encode('utf-8')
    return _MASTER_KEY_CACHE

def _derive_key(salt: bytes) -> bytes:
    """Derives the encryption key using PBKDF2HMAC based on the provided salt."""
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

# --- Per-Value Salt Encryption/Decryption ---

def encrypt_data(data: Optional[str]) -> Optional[str]:
    """
    Encrypts string data using AES-GCM with a unique salt per operation.
    Returns Base64 encoded string: salt(16) + nonce(12) + ciphertext+tag.
    Returns None if input data is None.
    ### Phase 1 Plan Ref: 2.2 (Verify encryption) - This is the verified implementation.
    """
    if data is None:
        return None
    if not isinstance(data, str):
        logger.error(f"Invalid data type for encryption: {type(data)}")
        raise TypeError("Data to encrypt must be a string.")

    try:
        salt = os.urandom(SALT_BYTES) # Generate unique salt per encryption
        derived_key = _derive_key(salt) # Derive key using this unique salt
        aesgcm = AESGCM(derived_key)
        nonce = os.urandom(NONCE_BYTES) # Unique nonce per encryption
        data_bytes = data.encode('utf-8')
        encrypted_bytes = aesgcm.encrypt(nonce, data_bytes, None) # AAD is None
        # Prepend salt and nonce to the ciphertext
        encrypted_payload = salt + nonce + encrypted_bytes
        encoded_payload = base64.urlsafe_b64encode(encrypted_payload).decode('utf-8')
        return encoded_payload
    except EncryptionError: raise # Re-raise critical key errors
    except Exception as e:
        logger.exception(f"Encryption failed unexpectedly: {e}")
        raise EncryptionError(f"Encryption failed: {e}")

def decrypt_data(encrypted_data_b64: Optional[str]) -> Optional[str]:
    """
    Decrypts data encrypted by encrypt_data (with per-value salt).
    Expects Base64 encoded input: salt(16) + nonce(12) + ciphertext+tag.
    Returns None if input is None or decryption fails.
    ### Phase 1 Plan Ref: 2.2 (Verify decryption) - This is the verified implementation.
    """
    if encrypted_data_b64 is None:
        return None
    if not isinstance(encrypted_data_b64, str):
        logger.error(f"Invalid data type for decryption: {type(encrypted_data_b64)}")
        raise TypeError("Encrypted data must be a base64 encoded string.")

    try:
        encrypted_payload = base64.urlsafe_b64decode(encrypted_data_b64)
        # Minimum length check: Salt + Nonce + 16 bytes for AES-GCM tag
        min_len = SALT_BYTES + NONCE_BYTES + 16
        if len(encrypted_payload) < min_len:
             # This usually means the data wasn't encrypted with this method (or is corrupted)
             logger.error(f"Decryption failed: Payload too short ({len(encrypted_payload)} bytes) for salt+nonce+tag. Data might be unencrypted or use old format.")
             return None # Cannot decrypt

        # Extract components
        salt = encrypted_payload[:SALT_BYTES]
        nonce = encrypted_payload[SALT_BYTES : SALT_BYTES + NONCE_BYTES]
        ciphertext_with_tag = encrypted_payload[SALT_BYTES + NONCE_BYTES :]

        # Derive key using the extracted salt
        derived_key = _derive_key(salt)
        aesgcm = AESGCM(derived_key)

        # Decrypt
        decrypted_bytes = aesgcm.decrypt(nonce, ciphertext_with_tag, None) # No AAD
        decrypted_string = decrypted_bytes.decode('utf-8')
        return decrypted_string

    except InvalidTag:
        logger.error("Decryption failed: Invalid authentication tag (data tampered or wrong key).")
        return None
    except EncryptionError as ee: # Catch error from _derive_key
        logger.error(f"Decryption failed due to key derivation error: {ee}")
        return None
    except (ValueError, TypeError, IndexError) as e:
        # Catches potential base64 decoding errors, slicing errors, or utf-8 decoding errors
        logger.error(f"Decryption failed due to data format or decoding error: {e}")
        return None
    except Exception as e:
        # Catch-all for any other unexpected errors during decryption
        logger.exception(f"Decryption failed unexpectedly: {e}")
        return None

# --- Database Session Management ---
# ### Phase 1 Plan Ref: 2.1 (Postgres only connection)
# ### Phase 1 Plan Ref: 2.4 (Remove KB helpers) - No DB interaction logic here beyond session.

# Create the engine and session maker once when the module is imported
try:
    if not settings.DATABASE_URL:
        raise ValueError("DATABASE_URL is not configured in settings.")
    # Use pool_recycle to prevent stale connections, pre-ping to check connection validity
    async_engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False, # Set to True for debugging SQL
        pool_pre_ping=True,
        pool_recycle=3600 # Recycle connections older than 1 hour
    )
    # Session maker configured for async use
    AsyncSessionMaker = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False # Important for async usage
    )
    logger.info("Successfully created SQLAlchemy async engine and session maker for Postgres.")
except Exception as e:
    logger.critical(f"Failed to create database engine or session maker: {e}", exc_info=True)
    # This is critical, re-raise to prevent application startup without DB access
    raise RuntimeError(f"Database initialization failed: {e}") from e

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Provides an async database session via context manager using the pre-configured maker."""
    # ### Phase 1 Plan Ref: 2.4 (Keep get_session)
    async with AsyncSessionMaker() as session:
        try:
            yield session
            # Commit is typically handled within the agent logic using the session
            # await session.commit() # Removed - let caller manage commit
        except Exception as e:
            logger.error("Exception occurred within database session, rolling back.", exc_info=True)
            await session.rollback()
            raise # Re-raise the exception after rollback
        # Session is automatically closed by the context manager

# --- DELETED ---
# ### Phase 1 Plan Ref: 2.3 (Delete old migration function)
# Function decrypt_data_fixed_salt_migration removed.
# ### Phase 1 Plan Ref: 2.4 (Delete KB helpers)
# Functions like log_knowledge_fragment, query_knowledge_base etc. removed.

# --- End of utils/database.py ---