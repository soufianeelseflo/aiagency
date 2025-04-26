# utils/database.py
# Genius-Level Implementation v1.1 - Secure Data Handling with Per-Value Salts & Migration Support

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
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# Configure logging
logger = logging.getLogger(__name__)

# --- Encryption Configuration ---
NONCE_BYTES = 12
KEY_BYTES = 32
SALT_BYTES = 16 # Used for new per-value salts
PBKDF2_ITERATIONS = 600000

class EncryptionError(Exception):
    """Custom exception for encryption/decryption errors."""
    pass

# --- Core Key Derivation ---

def _get_master_key() -> bytes:
    """Retrieves the database master key from environment variables."""
    master_key_str = os.getenv("DATABASE_ENCRYPTION_KEY")
    if not master_key_str:
        error_msg = "CRITICAL: DATABASE_ENCRYPTION_KEY environment variable not set."
        logger.critical(error_msg)
        raise EncryptionError(error_msg)
    return master_key_str.encode('utf-8')

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

# --- NEW Per-Value Salt Encryption/Decryption (Default Usage) ---

def encrypt_data(data: Optional[str]) -> Optional[str]:
    """
    Encrypts string data using AES-GCM with a unique salt and nonce per operation.
    Returns Base64 encoded string: salt(16) + nonce(12) + ciphertext+tag.
    Returns None if input data is None.
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
        encrypted_payload = salt + nonce + encrypted_bytes # Prepend unique salt
        encoded_payload = base64.urlsafe_b64encode(encrypted_payload).decode('utf-8')
        return encoded_payload
    except EncryptionError: raise
    except Exception as e:
        logger.exception(f"Encryption failed unexpectedly: {e}")
        raise EncryptionError(f"Encryption failed: {e}")

def decrypt_data(encrypted_data_b64: Optional[str]) -> Optional[str]:
    """
    Decrypts data encrypted by the updated encrypt_data function (with per-value salt).
    Expects Base64 encoded input: salt(16) + nonce(12) + ciphertext+tag.
    Returns None if input is None or decryption fails.
    """
    if encrypted_data_b64 is None:
        return None
    if not isinstance(encrypted_data_b64, str):
        logger.error(f"Invalid data type for decryption: {type(encrypted_data_b64)}")
        # Return None instead of raising TypeError for potentially old data during transition?
        # For now, keep raising TypeError for invalid input type.
        raise TypeError("Encrypted data must be a base64 encoded string.")

    try:
        encrypted_payload = base64.urlsafe_b64decode(encrypted_data_b64)
        min_len = SALT_BYTES + NONCE_BYTES + 16 # salt + nonce + 16 bytes for AES-GCM tag
        if len(encrypted_payload) < min_len:
             logger.warning(f"Decryption failed: Payload too short ({len(encrypted_payload)} bytes) for salt+nonce+tag. Might be old format data.")
             # Return None for short payloads, could be old format without salt.
             return None

        salt = encrypted_payload[:SALT_BYTES] # Extract the salt from the beginning
        nonce = encrypted_payload[SALT_BYTES : SALT_BYTES + NONCE_BYTES] # Extract the nonce
        ciphertext_with_tag = encrypted_payload[SALT_BYTES + NONCE_BYTES :] # Extract the rest

        derived_key = _derive_key(salt) # Derive key using the extracted salt
        aesgcm = AESGCM(derived_key)
        decrypted_bytes = aesgcm.decrypt(nonce, ciphertext_with_tag, None) # No AAD
        decrypted_string = decrypted_bytes.decode('utf-8')
        return decrypted_string

    except InvalidTag:
        logger.error("Decryption failed: Invalid authentication tag (data tampered or wrong key).")
        return None
    except EncryptionError as ee:
        logger.error(f"Decryption failed due to specific encryption error: {ee}")
        return None
    except (ValueError, TypeError, IndexError) as e:
        logger.error(f"Decryption failed due to data format or decoding error: {e}")
        return None
    except Exception as e:
        logger.exception(f"Decryption failed unexpectedly: {e}")
        return None

# --- OLD Fixed-Salt Decryption Logic (For Migration Only) ---

_OLD_FIXED_SALT: bytes | None = None
_OLD_DERIVED_KEY_CACHE: bytes | None = None

def _get_old_fixed_salt() -> bytes:
    """Gets the OLD fixed salt used before per-value salting. For migration."""
    global _OLD_FIXED_SALT
    if _OLD_FIXED_SALT is not None:
        return _OLD_FIXED_SALT

    old_salt_str = os.getenv("DATABASE_FIXED_SALT", "default_insecure_salt_replace_me") # MUST MATCH OLD VALUE
    if old_salt_str == "default_insecure_salt_replace_me":
         logger.warning("Using default insecure fixed salt for migration decryption. Ensure DATABASE_FIXED_SALT env var is set if a custom one was used previously.")
    try:
        _OLD_FIXED_SALT = hashlib.sha256(old_salt_str.encode()).digest()[:SALT_BYTES] # Use SALT_BYTES here too for consistency
        return _OLD_FIXED_SALT
    except Exception as e:
        logger.error(f"Failed to derive old fixed salt for migration: {e}")
        raise EncryptionError(f"Cannot derive old fixed salt needed for migration: {e}")

def _derive_key_old_fixed_salt() -> bytes:
    """Derives key using the OLD fixed salt. For migration."""
    global _OLD_DERIVED_KEY_CACHE
    old_salt = _get_old_fixed_salt()

    if _OLD_DERIVED_KEY_CACHE is not None:
        return _OLD_DERIVED_KEY_CACHE

    # Derive using the fixed salt
    derived_key = _derive_key(old_salt) # Use the core _derive_key function
    _OLD_DERIVED_KEY_CACHE = derived_key
    return derived_key

def decrypt_data_fixed_salt_migration(encrypted_data_b64: Optional[str]) -> Optional[str]:
    """
    Decrypts data encrypted using the OLD fixed salt method. FOR MIGRATION ONLY.
    Expects Base64 encoded input: nonce(12) + ciphertext+tag.
    """
    if encrypted_data_b64 is None: return None
    if not isinstance(encrypted_data_b64, str):
        logger.error(f"[Migration Decrypt] Invalid data type: {type(encrypted_data_b64)}")
        return None # Fail gracefully for migration

    try:
        encrypted_payload = base64.urlsafe_b64decode(encrypted_data_b64)
        min_len = NONCE_BYTES + 16 # OLD format: nonce + ciphertext + tag
        if len(encrypted_payload) < min_len:
             logger.error(f"[Migration Decrypt] Payload too short ({len(encrypted_payload)} bytes). Value: {encrypted_data_b64[:50]}...")
             return None

        nonce = encrypted_payload[:NONCE_BYTES]
        ciphertext_with_tag = encrypted_payload[NONCE_BYTES:]

        derived_key = _derive_key_old_fixed_salt() # Use OLD fixed salt key derivation
        aesgcm = AESGCM(derived_key)
        decrypted_bytes = aesgcm.decrypt(nonce, ciphertext_with_tag, None)
        return decrypted_bytes.decode('utf-8')

    except InvalidTag:
        logger.error(f"[Migration Decrypt] Invalid authentication tag. Value: {encrypted_data_b64[:50]}...")
        return None
    except EncryptionError as ee: # Catch error from _derive_key_old_fixed_salt
        logger.error(f"[Migration Decrypt] Failed due to encryption error: {ee}")
        return None
    except Exception as e:
        logger.error(f"[Migration Decrypt] Failed unexpectedly for value '{encrypted_data_b64[:50]}...': {e}")
        return None

# --- Database Session Utility (Unchanged) ---
async def get_session(session_maker: async_sessionmaker[AsyncSession]) -> AsyncGenerator[AsyncSession, None]:
    """Provides an async database session via context manager."""
    async with session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            logger.error("Exception occurred within database session, rolling back.", exc_info=True)
            raise

# --- End of utils/database.py ---