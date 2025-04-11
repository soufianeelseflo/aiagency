import os
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag
from sqlalchemy.ext.asyncio import AsyncSession
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Encryption Configuration ---
# AES-GCM uses a 12-byte (96-bit) nonce, which is recommended.
NONCE_BYTES = 12
# We derive a 32-byte (256-bit) key for AES-256-GCM
KEY_BYTES = 32
# Use a 16-byte (128-bit) salt for PBKDF2
SALT_BYTES = 16
# PBKDF2 iterations - OWASP recommends at least 100,000, 600,000 is safer (adjust based on performance needs)
PBKDF2_ITERATIONS = 600000

class EncryptionError(Exception):
    """Custom exception for encryption/decryption errors."""
    pass

def _get_master_key() -> bytes:
    """
    Retrieves the database master key from environment variables.
    This key is used as input for the Key Derivation Function (PBKDF2).

    Raises:
        EncryptionError: If the master key is not set.

    Returns:
        bytes: The master key.
    """
    master_key_str = os.getenv("DATABASE_MASTER_KEY")
    if not master_key_str:
        logger.error("DATABASE_MASTER_KEY environment variable not set.")
        raise EncryptionError("Database master key is not configured.")
    # Encode the master key string to bytes using UTF-8
    return master_key_str.encode('utf-8')

def _derive_key(salt: bytes, master_key: bytes) -> bytes:
    """
    Derives an encryption key from the master key and salt using PBKDF2HMAC.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_BYTES,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
        backend=default_backend()
    )
    return kdf.derive(master_key)

def encrypt_data(data: str) -> str:
    """
    Encrypts string data using AES-GCM with a key derived via PBKDF2.

    Args:
        data: The string data to encrypt.

    Returns:
        str: A base64 encoded string containing salt + nonce + ciphertext + tag.

    Raises:
        EncryptionError: If encryption fails or master key is invalid/missing.
    """
    if not isinstance(data, str):
        raise TypeError("Data to encrypt must be a string.")

    try:
        master_key = _get_master_key()
        salt = os.urandom(SALT_BYTES)
        derived_key = _derive_key(salt, master_key)

        aesgcm = AESGCM(derived_key)
        nonce = os.urandom(NONCE_BYTES)
        data_bytes = data.encode('utf-8')
        encrypted_bytes = aesgcm.encrypt(nonce, data_bytes, None) # No associated data

        # Combine salt, nonce, and encrypted data (which includes the tag)
        encrypted_payload = salt + nonce + encrypted_bytes

        # Base64 encode for safe storage
        return base64.urlsafe_b64encode(encrypted_payload).decode('utf-8')
    except EncryptionError:
        # Re-raise specific key errors
        raise
    except Exception as e:
        logger.exception(f"Encryption failed: {e}")
        raise EncryptionError(f"Encryption failed: {e}")

def decrypt_data(encrypted_data_b64: str) -> str:
    """
    Decrypts data encrypted with encrypt_data using AES-GCM and PBKDF2 derived key.

    Args:
        encrypted_data_b64: The base64 encoded string (salt + nonce + ciphertext + tag).

    Returns:
        str: The original decrypted string.

    Raises:
        EncryptionError: If decryption fails (e.g., invalid tag, wrong key, corrupted data).
        TypeError: If input is not a string.
    """
    if not isinstance(encrypted_data_b64, str):
        raise TypeError("Encrypted data must be a string.")

    try:
        master_key = _get_master_key()
        encrypted_payload = base64.urlsafe_b64decode(encrypted_data_b64)

        # Check minimum length (salt + nonce + at least 1 byte ciphertext + 16 byte tag)
        if len(encrypted_payload) <= SALT_BYTES + NONCE_BYTES + 16:
             raise EncryptionError("Invalid encrypted data format: too short.")

        # Extract salt, nonce, and ciphertext (which includes the tag)
        salt = encrypted_payload[:SALT_BYTES]
        nonce = encrypted_payload[SALT_BYTES:SALT_BYTES + NONCE_BYTES]
        ciphertext_with_tag = encrypted_payload[SALT_BYTES + NONCE_BYTES:]

        # Derive the *same* key using the extracted salt
        derived_key = _derive_key(salt, master_key)

        aesgcm = AESGCM(derived_key)
        decrypted_bytes = aesgcm.decrypt(nonce, ciphertext_with_tag, None) # No associated data

        return decrypted_bytes.decode('utf-8')
    except InvalidTag:
        logger.error("Decryption failed: Invalid authentication tag. Data may be tampered or master key is incorrect.")
        raise EncryptionError("Decryption failed: Data integrity check failed.")
    except EncryptionError:
        # Re-raise specific key errors
        raise
    except (ValueError, TypeError, IndexError) as e:
        logger.error(f"Decryption failed due to data format or decoding error: {e}")
        raise EncryptionError(f"Decryption failed: Invalid format or decoding error ({e}).")
    except Exception as e:
        logger.exception(f"Decryption failed: {e}")
        raise EncryptionError(f"Decryption failed: {e}")


# --- Existing Database Session Utility ---

async def get_session(session_maker):
    """Provides an async database session."""
    async with session_maker() as session:
        yield session