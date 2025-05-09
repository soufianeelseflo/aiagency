# Filename: Dockerfile
# Description: Dockerfile for deploying the Nolli AI Sales System (IGNIS Transmutation)
# Version: 2.0 (Clean Room Build - Standard Pip, Explicit PATH, Python Invocation)

# --- Base Image ---
# Use python:3.11-bookworm (non-slim) for a more complete environment
FROM python:3.11-bookworm

# --- Environment Variables ---
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers \
    PORT=5000 \
    HOST=0.0.0.0 \
    # Add standard user binary path to system PATH
    # This is where pip often installs command-line scripts like 'playwright' and 'quart'
    PATH=/root/.local/bin:$PATH

# --- Working Directory ---
WORKDIR /app

# --- System Dependencies ---
# Install system libs including curl for HEALTHCHECK and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev \
    build-essential \
    curl wget gnupg \
    # Playwright deps
    libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2 libexpat1 libxcb1 \
    libxkbcommon0 libx11-6 libxext6 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# --- Application Dependencies ---
# Copy requirements first
COPY requirements.txt .

# Install Python dependencies using the standard method
# They will go into the default site-packages for the Python in the image.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Install Playwright Browsers ---
# Use 'python -m playwright' to ensure the correct playwright executable is found via Python's module system
# The PATH update above should also help, but 'python -m' is more direct.
RUN python -m playwright install chromium --with-deps

# --- Application Code ---
# Copy application code AFTER dependencies are installed
COPY . .

# --- Create necessary directories ---
# These paths should match what's in your config/settings.py
RUN mkdir -p /app/logs /app/temp_audio /app/temp_downloads /app/learning_for_AI && \
    chmod -R 777 /app/logs /app/temp_audio /app/temp_downloads /app/learning_for_AI
    # Granting broad permissions for simplicity. For stricter production,
    # create a non-root user and chown these dirs to that user.

# --- Expose Port ---
EXPOSE 5000

# --- Healthcheck ---
# Checks if the Quart server (started by CMD) is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl --fail --silent --show-error http://localhost:5000/ || exit 1

# --- Start Command ---
# Use 'python -m quart' which explicitly uses the Python interpreter to find and run the quart module.
CMD ["python", "-m", "quart", "run", "--host", "0.0.0.0", "--port", "5000", "--no-reload"]