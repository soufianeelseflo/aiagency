# migrations/migrate_encryption_v1_to_v2.py
# Data migration script to update ExpenseLog.description encryption from fixed-salt to per-value salt.
# WARNING: Run this script OFFLINE (application stopped) after backing up the database.

import asyncio
import os
import sys
import logging
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, select, update, Column, Integer, Text # Import necessary SQLAlchemy components
from sqlalchemy.orm import declarative_base

# --- Configuration (Adjust as needed) ---
DATABASE_URL = os.getenv("DATABASE_URL") # Ensure this is set in the environment where the script runs
if not DATABASE_URL:
    print("CRITICAL: DATABASE_URL environment variable not set. Exiting.")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants for Encryption Logic ---
# These MUST match the values used in the OLD and NEW utils/database.py versions
NONCE_BYTES = 12
KEY_BYTES = 32
SALT_BYTES = 16 # For the NEW per-value salt
PBKDF2_ITERATIONS = 600000

# --- OLD Fixed-Salt Decryption Logic (Copied & adapted from old utils/database.py) ---
# Retrieve OLD fixed salt - CRITICAL: Ensure this matches the salt used for existing data
OLD_FIXED_SALT_STR = os.getenv("DATABASE_FIXED_SALT", "default_insecure_salt_replace_me") # MUST MATCH OLD VALUE
if OLD_FIXED_SALT_STR == "default_insecure_salt_replace_me":
     logger.warning("Using default insecure fixed salt for decryption. Ensure DATABASE_FIXED_SALT is set correctly if a custom one was used.")
try:
    # Use hashlib directly as it might not be available in the old context
    import hashlib
    OLD_FIXED_SALT = hashlib.sha256(OLD_FIXED_SALT_STR.encode()).digest()[:SALT_BYTES]
except ImportError:
    logger.error("hashlib module not found. Cannot derive old fixed salt.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error deriving old fixed salt: {e}")
    sys.exit(1)

_OLD_DERIVED_KEY_CACHE: bytes | None = None

def _get_master_key_old() -> bytes:
    master_key_str = os.getenv("DATABASE_ENCRYPTION_KEY")
    if not master_key_str:
        raise ValueError("DATABASE_ENCRYPTION_KEY environment variable not set.")
    return master_key_str.encode('utf-8')

def _derive_key_old(salt: bytes) -> bytes:
    global _OLD_DERIVED_KEY_CACHE
    if salt == OLD_FIXED_SALT and _OLD_DERIVED_KEY_CACHE is not None:
        return _OLD_DERIVED_KEY_CACHE

    master_key = _get_master_key_old()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_BYTES,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
        backend=default_backend()
    )
    derived_key = kdf.derive(master_key)
    if salt == OLD_FIXED_SALT:
        _OLD_DERIVED_KEY_CACHE = derived_key
    return derived_key

def decrypt_data_old(encrypted_data_b64: str | None) -> str | None:
    if encrypted_data_b64 is None: return None
    try:
        encrypted_payload = base64.urlsafe_b64decode(encrypted_data_b64)
        min_len = NONCE_BYTES + 16 # OLD format: nonce + ciphertext + tag
        if len(encrypted_payload) <= min_len:
             logger.error(f"OLD Decryption failed: Payload too short ({len(encrypted_payload)} bytes). Value: {encrypted_data_b64[:50]}...")
             return None # Cannot decrypt

        nonce = encrypted_payload[:NONCE_BYTES]
        ciphertext_with_tag = encrypted_payload[NONCE_BYTES:]
        salt = OLD_FIXED_SALT # Use the fixed salt
        derived_key = _derive_key_old(salt)
        aesgcm = AESGCM(derived_key)
        decrypted_bytes = aesgcm.decrypt(nonce, ciphertext_with_tag, None)
        return decrypted_bytes.decode('utf-8')
    except InvalidTag:
        logger.error(f"OLD Decryption failed: Invalid authentication tag. Value: {encrypted_data_b64[:50]}...")
        return None
    except Exception as e:
        logger.error(f"OLD Decryption failed unexpectedly for value '{encrypted_data_b64[:50]}...': {e}")
        return None

# --- NEW Per-Value Salt Encryption Logic (Copied & adapted from new utils/database.py) ---

def _derive_key_new(salt: bytes) -> bytes: # No caching needed
    master_key = _get_master_key_old() # Master key remains the same
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_BYTES,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
        backend=default_backend()
    )
    return kdf.derive(master_key)

def encrypt_data_new(data: str | None) -> str | None:
    if data is None: return None
    try:
        salt = os.urandom(SALT_BYTES) # Generate unique salt
        derived_key = _derive_key_new(salt)
        aesgcm = AESGCM(derived_key)
        nonce = os.urandom(NONCE_BYTES)
        data_bytes = data.encode('utf-8')
        encrypted_bytes = aesgcm.encrypt(nonce, data_bytes, None)
        encrypted_payload = salt + nonce + encrypted_bytes # Prepend salt
        return base64.urlsafe_b64encode(encrypted_payload).decode('utf-8')
    except Exception as e:
        logger.exception(f"NEW Encryption failed unexpectedly: {e}")
        raise # Re-raise to stop migration on encryption error

# --- Database Model Definition (Minimal required for migration) ---
Base = declarative_base()

class ExpenseLog(Base):
    __tablename__ = 'expense_logs'
    id = Column(Integer, primary_key=True)
    description = Column(Text, nullable=False)
    # Add other columns if needed for context, but not strictly required for migration

# --- Migration Logic ---
async def migrate_data():
    logger.info(f"Starting encryption migration for 'expense_logs.description' in database: {DATABASE_URL.split('@')[-1]}") # Mask credentials in log
    engine = create_async_engine(DATABASE_URL, echo=False) # Turn off SQL echo for cleaner logs
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    total_processed = 0
    total_updated = 0
    total_failed_decryption = 0
    batch_size = 100 # Process in batches

    async with async_session() as session:
        offset = 0
        while True:
            logger.info(f"Processing batch starting from offset {offset}...")
            # Select batch of records
            stmt = select(ExpenseLog.id, ExpenseLog.description).order_by(ExpenseLog.id).offset(offset).limit(batch_size)
            result = await session.execute(stmt)
            records = result.fetchall()

            if not records:
                logger.info("No more records found. Migration processing complete.")
                break

            updates_to_commit = []
            for record_id, old_encrypted_desc in records:
                total_processed += 1
                if not old_encrypted_desc:
                    logger.debug(f"Skipping record ID {record_id}: description is null.")
                    continue

                # 1. Decrypt using OLD logic
                decrypted_desc = decrypt_data_old(old_encrypted_desc)

                if decrypted_desc is None:
                    logger.error(f"Failed to decrypt description for ExpenseLog ID {record_id}. Skipping update for this record.")
                    total_failed_decryption += 1
                    continue # Skip this record if decryption failed

                # 2. Re-encrypt using NEW logic
                try:
                    new_encrypted_desc = encrypt_data_new(decrypted_desc)
                    if new_encrypted_desc is None: # Should not happen if input is not None
                         logger.error(f"NEW encryption returned None for ExpenseLog ID {record_id}. This is unexpected. Skipping.")
                         continue
                except Exception as enc_err:
                    logger.error(f"Failed during NEW encryption for ExpenseLog ID {record_id}: {enc_err}. Skipping update.")
                    continue # Skip if new encryption fails

                # 3. Check if re-encryption actually changed the value (it should have due to salt/nonce)
                if new_encrypted_desc != old_encrypted_desc:
                    updates_to_commit.append({'id': record_id, 'description': new_encrypted_desc})
                    total_updated += 1
                else:
                     logger.warning(f"New encrypted value is identical to old for ExpenseLog ID {record_id}. This might indicate an issue. Skipping update.")


            # 4. Perform batch update if there are changes
            if updates_to_commit:
                try:
                    # Use bulk update for efficiency
                    update_stmt = update(ExpenseLog).where(ExpenseLog.id == text(':id')).values(description=text(':description'))
                    await session.execute(update_stmt, updates_to_commit)
                    await session.commit()
                    logger.info(f"Committed updates for {len(updates_to_commit)} records in batch (offset {offset}).")
                except Exception as commit_err:
                    logger.error(f"Failed to commit batch update (offset {offset}): {commit_err}")
                    await session.rollback()
                    logger.info("Rolling back transaction for the failed batch.")
                    # Decide whether to stop or continue with the next batch
                    logger.error("Stopping migration due to commit error.")
                    break # Stop on commit error

            offset += len(records) # Move to the next batch

    logger.info("--- Migration Summary ---")
    logger.info(f"Total records processed: {total_processed}")
    logger.info(f"Total records successfully updated: {total_updated}")
    logger.info(f"Total records failed decryption (skipped): {total_failed_decryption}")
    logger.info("-------------------------")

    if total_failed_decryption > 0:
        logger.warning("Some records failed decryption and were not updated. Review logs for details.")

    await engine.dispose() # Clean up connections

if __name__ == "__main__":
    print("--- ExpenseLog Encryption Migration Script ---")
    print("WARNING: Ensure the application is STOPPED and the database is BACKED UP before running.")
    print(f"Target Database: {DATABASE_URL.split('@')[-1]}")
    print(f"Using OLD Fixed Salt derived from: '{OLD_FIXED_SALT_STR}'")
    print(f"Using Master Key from DATABASE_ENCRYPTION_KEY environment variable.")
    confirm = input("Proceed with migration? (yes/no): ")
    if confirm.lower() == 'yes':
        asyncio.run(migrate_data())
    else:
        print("Migration cancelled.")