# Roo-Code Task: Implement/Verify Robust Database Encryption

This file tracks the progress of implementing/verifying encryption functions in `utils/database.py`.

**Sub-Tasks:**

*   [ ] 1. Add/Verify Imports: `os`, `base64`, `logging`, `cryptography.hazmat.primitives.ciphers.aead.AESGCM`, `cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC`, `cryptography.hazmat.primitives.hashes`, `cryptography.hazmat.backends.default_backend`, `cryptography.exceptions.InvalidTag`.
*   [ ] 2. Implement/Verify `get_encryption_key()`: Define/check function. Retrieves `DATABASE_ENCRYPTION_KEY` env var. Derives 32-byte key using PBKDF2HMAC (SHA256, salt=`b'fixed_salt_for_agency_db'`, iterations>=480000). Caches key globally. Raises `ValueError` if env var missing.
*   [ ] 3. Implement/Verify `encrypt_data(data: str)`: Define/check function. Handles `None` input. Gets key. Generates 12-byte nonce. Encrypts with AESGCM. Returns Base64(nonce + ciphertext). Handles encryption errors (logs, raises `ValueError`).
*   [ ] 4. Implement/Verify `decrypt_data(encrypted_data: str)`: Define/check function. Handles `None` input. Base64 decodes. Separates nonce (12 bytes) and ciphertext. Gets key. Decrypts with AESGCM. Handles `InvalidTag` (logs error, returns `None`). Handles other exceptions (logs error, returns `None`). Returns decoded UTF-8 string.