# Filename: Dockerfile
# Description: Dockerfile for deploying the Nolli AI Sales System (Level 50+ ready)
# Version: 1.2 (PYTHONPATH strategy for module resolution)

# --- Base Image ---
FROM python:3.11-slim

# --- Environment Variables ---
ENV PYTHONUNBUFFERED=1 \
  DEBIAN_FRONTEND=noninteractive \
  PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers \
  PORT=5000 \
  HOST=0.0.0.0 \
  # Explicitly add the application's dependency directory to Python's search path
  PYTHONPATH=/app/deps

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

# Install dependencies into the dedicated /app/deps directory
# Using --target ensures they are placed here, separate from system site-packages
RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir --target=/app/deps -r requirements.txt

# --- Install Playwright Browsers ---
# This should still work fine, installs browsers globally based on PLAYWRIGHT_BROWSERS_PATH
RUN playwright install chromium --with-deps

# --- Application Code ---
# Copy application code AFTER dependencies are installed
COPY . .

# --- Create necessary directories ---
# Ensure directories exist and are writable by the container process
RUN mkdir -p /app/logs /app/temp_audio /app/temp_downloads /app/learning_for_AI && \
  chmod -R 777 /app/logs /app/temp_audio /app/temp_downloads
# Note: 777 is permissive, consider a non-root user and specific ownership for production hardening later.

# --- Expose Port ---
EXPOSE 5000

# --- Healthcheck ---
# This should work once the application starts correctly
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl --fail --silent --show-error http://localhost:5000/ || exit 1

# --- Start Command ---
# Use the absolute path AND rely on PYTHONPATH set via ENV to find modules in /app/deps
CMD ["/usr/local/bin/python", "-m", "quart", "run", "--host", "0.0.0.0", "--port", "5000", "--no-reload"]