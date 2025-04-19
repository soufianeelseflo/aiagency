# utils/database.py
# Genius-Level Implementation v1.0 - Secure Data Handling

import os
import base64
import logging
import hashlib # Added missing import
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag
from typing import Optional, AsyncGenerator # Import AsyncGenerator for type hinting
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker # Import async_sessionmaker

# Configure logging
logger = logging.getLogger(__name__)
# Assume logging configured globally

# --- Encryption Configuration ---
NONCE_BYTES = 12
KEY_BYTES = 32
SALT_BYTES = 16
PBKDF2_ITERATIONS = 600000 # Increased iterations for better security
# Generate a secure, fixed salt ONCE per deployment and store it securely (e.g., in Vault or env var)
# DO NOT commit a hardcoded salt like this in real production code unless it's truly static per deployment.
# For this example, we use a hardcoded one derived from an env var if possible, or a default.
FIXED_SALT_STR = os.getenv("DATABASE_FIXED_SALT", "default_insecure_salt_replace_me") # MUST BE REPLACED IN ENV
FIXED_SALT = hashlib.sha256(FIXED_SALT_STR.encode()).digest()[:SALT_BYTES] # Derive a fixed-size salt

# --- Global Key Cache ---
_DERIVED_KEY_CACHE: bytes | None = None

class EncryptionError(Exception):
    """Custom exception for encryption/decryption errors."""
    pass

def _get_master_key() -> bytes:
    """Retrieves the database master key from environment variables."""
    master_key_str = os.getenv("DATABASE_ENCRYPTION_KEY")
    if not master_key_str:
        error_msg = "CRITICAL: DATABASE_ENCRYPTION_KEY environment variable not set."
        logger.critical(error_msg)
        raise EncryptionError(error_msg)
    return master_key_str.encode('utf-8')

def _derive_key(salt: bytes) -> bytes:
    """Derives the encryption key using PBKDF2HMAC."""
    global _DERIVED_KEY_CACHE
    # Cache only if using the globally defined FIXED_SALT
    if salt == FIXED_SALT and _DERIVED_KEY_CACHE is not None:
        # logger.debug("Using cached encryption key.")
        return _DERIVED_KEY_CACHE

    master_key = _get_master_key()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_BYTES,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
        backend=default_backend()
    )
    derived_key = kdf.derive(master_key)

    if salt == FIXED_SALT:
        _DERIVED_KEY_CACHE = derived_key
        logger.info("Derived and cached encryption key using fixed salt.")
    else:
        # This path shouldn't be hit if only FIXED_SALT is used by encrypt_data
        logger.warning(f"Derived encryption key using non-fixed salt: {salt.hex()}")

    return derived_key

# --- Genius-Level Encryption/Decryption ---

def encrypt_data(data: Optional[str]) -> Optional[str]:
    """
    Encrypts string data using AES-GCM with a fixed salt and unique nonce.
    Returns Base64 encoded string: nonce(12) + ciphertext+tag.
    Returns None if input data is None.
    """
    if data is None:
        return None
    if not isinstance(data, str):
        logger.error(f"Invalid data type for encryption: {type(data)}")
        raise TypeError("Data to encrypt must be a string.")

    try:
        salt = FIXED_SALT # Use the fixed salt
        derived_key = _derive_key(salt)
        aesgcm = AESGCM(derived_key)
        nonce = os.urandom(NONCE_BYTES) # Unique nonce per encryption
        data_bytes = data.encode('utf-8')
        encrypted_bytes = aesgcm.encrypt(nonce, data_bytes, None) # AAD is None
        encrypted_payload = nonce + encrypted_bytes
        encoded_payload = base64.urlsafe_b64encode(encrypted_payload).decode('utf-8')
        return encoded_payload
    except EncryptionError: raise
    except Exception as e:
        logger.exception(f"Encryption failed unexpectedly: {e}")
        raise EncryptionError(f"Encryption failed: {e}")

def decrypt_data(encrypted_data_b64: Optional[str]) -> Optional[str]:
    """
    Decrypts data encrypted by encrypt_data (using fixed salt).
    Returns None if input is None or decryption fails.
    """
    if encrypted_data_b64 is None:
        return None
    if not isinstance(encrypted_data_b64, str):
        logger.error(f"Invalid data type for decryption: {type(encrypted_data_b64)}")
        raise TypeError("Encrypted data must be a base64 encoded string.")

    try:
        encrypted_payload = base64.urlsafe_b64decode(encrypted_data_b64)
        min_len = NONCE_BYTES + 16 # 16 bytes for AES-GCM tag
        if len(encrypted_payload) <= min_len:
             logger.error(f"Decryption failed: Payload too short ({len(encrypted_payload)} bytes).")
             raise EncryptionError("Invalid encrypted data format: too short.")

        nonce = encrypted_payload[:NONCE_BYTES]
        ciphertext_with_tag = encrypted_payload[NONCE_BYTES:]
        salt = FIXED_SALT # Use the fixed salt for key derivation
        derived_key = _derive_key(salt)
        aesgcm = AESGCM(derived_key)
        decrypted_bytes = aesgcm.decrypt(nonce, ciphertext_with_tag, None) # No AAD
        decrypted_string = decrypted_bytes.decode('utf-8')
        return decrypted_string

    except InvalidTag:
        logger.error("Decryption failed: Invalid authentication tag (data tampered or wrong key).")
        # Return None instead of raising to allow graceful handling in agents
        return None
    except EncryptionError as ee:
        logger.error(f"Decryption failed due to specific encryption error: {ee}")
        return None # Return None on key/format errors
    except (ValueError, TypeError, IndexError) as e:
        logger.error(f"Decryption failed due to data format or decoding error: {e}")
        return None # Return None on format errors
    except Exception as e:
        logger.exception(f"Decryption failed unexpectedly: {e}")
        return None # Return None on other errors

# --- Database Session Utility ---
# This structure is primarily for FastAPI dependency injection.
# Agents will typically use the session_maker directly via 'async with'.
async def get_session(session_maker: async_sessionmaker[AsyncSession]) -> AsyncGenerator[AsyncSession, None]:
    """Provides an async database session via context manager."""
    async with session_maker() as session:
        try:
            yield session
            # Let the caller handle commit/rollback within the 'async with' block
        except Exception:
            await session.rollback() # Ensure rollback on exception within yield
            logger.error("Exception occurred within database session, rolling back.", exc_info=True)
            raise # Re-raise the exception

# --- End of utils/database.py ---