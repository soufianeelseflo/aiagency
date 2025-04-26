# Evolution Plan: utils/database.py (v1.0)

## 1. Objective
Address the critical security vulnerability identified in `utils/database.py` related to the use of a fixed salt for AES-GCM encryption, aligning it with cryptographic best practices and the Genius Agentic Standard's requirement for robust security.

## 2. Identified Gap (from Baseline_utils_database.py.md & Evolution_Plan_Overall.md)
- **CRITICAL SECURITY FLAW:** The current implementation uses a single, fixed salt (`FIXED_SALT`), derived from an environment variable or a default string, for *all* encryption operations via `encrypt_data`.
- **Impact:** Reusing the same salt means the same key (derived via PBKDF2HMAC from the master key and the fixed salt) is used for encrypting multiple different pieces of data. While AES-GCM uses unique nonces per encryption which prevents identical plaintexts from producing identical ciphertexts, reusing the *key* across many encryptions significantly weakens the overall security guarantees of AES-GCM and makes certain cryptographic attacks (like key recovery attacks if other vulnerabilities exist) much easier if the master key or ciphertexts are compromised. This violates the principle of key separation and is not standard practice.

## 3. Proposed Superior Solution
- **Implement Per-Value Salting:** Modify the encryption process to generate a unique, random salt for *each individual piece of data* being encrypted.
- **Store Salt with Ciphertext:** Store this unique salt alongside the nonce and the resulting ciphertext (and authentication tag). The typical format becomes: `salt + nonce + ciphertext + tag`.
- **Key Derivation During Decryption:** Modify the decryption process to extract the salt from the stored payload first, then use *that specific salt* along with the master key to re-derive the correct decryption key for that specific piece of data.
- **Remove Fixed Salt:** Eliminate the concept of `FIXED_SALT` and `DATABASE_FIXED_SALT` environment variable entirely.
- **Remove Key Caching:** The global key cache (`_DERIVED_KEY_CACHE`) becomes unnecessary and should be removed, as each decryption will derive a unique key based on the stored salt.

## 4. Detailed Implementation Steps

1.  **Modify `encrypt_data` function (`utils/database.py`):**
    *   Remove the line `salt = FIXED_SALT`.
    *   Generate a unique salt for each call: `salt = os.urandom(SALT_BYTES)`.
    *   Derive the key using this unique salt: `derived_key = _derive_key(salt)`. (Note: `_derive_key` needs modification - see step 3).
    *   Construct the final payload by prepending the unique salt: `encrypted_payload = salt + nonce + encrypted_bytes`.
    *   Base64 encode `encrypted_payload` as before.
2.  **Modify `decrypt_data` function (`utils/database.py`):**
    *   After Base64 decoding (`encrypted_payload = base64.urlsafe_b64decode(...)`), extract the salt from the beginning: `salt = encrypted_payload[:SALT_BYTES]`.
    *   Extract the nonce: `nonce = encrypted_payload[SALT_BYTES : SALT_BYTES + NONCE_BYTES]`.
    *   Extract the ciphertext + tag: `ciphertext_with_tag = encrypted_payload[SALT_BYTES + NONCE_BYTES :]`.
    *   Adjust the minimum length check: `min_len = SALT_BYTES + NONCE_BYTES + 16`.
    *   Derive the key using the *extracted* salt: `derived_key = _derive_key(salt)`. (Note: `_derive_key` needs modification - see step 3).
    *   Proceed with AESGCM decryption using the derived key, extracted nonce, and `ciphertext_with_tag`.
3.  **Modify `_derive_key` function (`utils/database.py`):**
    *   Remove the global `_DERIVED_KEY_CACHE` variable and all logic related to checking or setting it (lines 49-53, 65-71). The function should simply derive the key based on the provided salt and master key every time.
4.  **Remove Fixed Salt Configuration (`utils/database.py` & `config/settings.py`):**
    *   Remove the `FIXED_SALT_STR` and `FIXED_SALT` definitions from `utils/database.py` (lines 28-29).
    *   Remove the `DATABASE_FIXED_SALT` environment variable check/usage if present elsewhere (though it seems confined to `utils/database.py`).
5.  **Data Migration (CRITICAL):**
    *   **Problem:** Existing data in the database was encrypted using the old method (fixed salt). The updated `decrypt_data` function will *fail* to decrypt this old data because it expects the salt to be prepended, which it isn't.
    *   **Solution:** A data migration script is required *before* deploying the updated code or immediately after. This script must:
        a.  Connect to the database.
        b.  Identify all columns containing encrypted data (requires knowledge of the database schema - `models.py` needs analysis).
        c.  For each encrypted value:
            i.  Decrypt it using the *old* `decrypt_data` logic (temporarily preserved or run before the code update).
            ii. Re-encrypt it using the *new* `encrypt_data` logic (with per-value salt).
            iii. Update the database record with the newly encrypted value.
    *   **Action:** Plan and create this migration script (`migrations/migrate_encryption_v1_to_v2.py`). This is a high-priority prerequisite for deploying the code changes.
6.  **Testing:**
    *   Add unit tests specifically for the updated `encrypt_data` and `decrypt_data` functions, ensuring data encrypted by the new function can be decrypted, and that decryption fails gracefully with invalid inputs or tampered data.
    *   Perform integration testing after the data migration to ensure the application functions correctly with the re-encrypted data.

## 5. Next Steps
- Proceed to Phase 3 (Strategic Planning & Simulation) for this specific plan, focusing on the data migration strategy and potential downtime implications.
- Create the migration script (`migrations/migrate_encryption_v1_to_v2.py`).
- Implement the code changes in `utils/database.py`.