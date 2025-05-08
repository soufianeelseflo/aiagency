# Filename: Dockerfile
# Description: Dockerfile for deploying the Nolli AI Sales System (Level 50+ ready)
# Version: 1.1 (Corrected CMD path, added HEALTHCHECK)

# --- Base Image ---
# Use an official Python slim image for smaller size
FROM python:3.11-slim

# --- Environment Variables ---
ENV PYTHONUNBUFFERED=1 \
  # Set non-interactive frontend for package installs
  DEBIAN_FRONTEND=noninteractive \
  # Playwright specific environment variables to skip browser downloads during pip install
  PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers \
  # Default port, can be overridden by Coolify
  PORT=5000 \
  # Default host, ensures listening on all interfaces within the container
  HOST=0.0.0.0

# --- Working Directory ---
WORKDIR /app

# --- System Dependencies ---
# Update apt cache, install necessary system libs (including curl for HEALTHCHECK)
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  # For psycopg2 (binary needs libpq)
  libpq-dev \
  # For Playwright Browsers & dependencies
  build-essential \
  curl \
  wget \
  gnupg \
  # Playwright browser dependencies
  libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
  libcups2 libdrm2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
  libgbm1 libpango-1.0-0 libcairo2 libasound2 libexpat1 libxcb1 \
  libxkbcommon0 libx11-6 libxext6 libfontconfig1 \
  # Clean up apt cache
  && rm -rf /var/lib/apt/lists/*

# --- Application Dependencies ---
# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

# --- Install Playwright Browsers ---
# Installs browsers into the path specified by PLAYWRIGHT_BROWSERS_PATH
RUN playwright install chromium --with-deps

# --- Application Code ---
# Copy the rest of the application code
COPY . .

# --- Create necessary directories and set permissions (if needed) ---
# Ensure directories exist and are writable by the container process
RUN mkdir -p /app/logs /app/temp_audio /app/temp_downloads /app/learning_for_AI && \
  chmod -R 777 /app/logs /app/temp_audio /app/temp_downloads
# Note: 777 is permissive, consider a non-root user and specific ownership for production hardening later.

# --- Expose Port ---
EXPOSE 5000

# --- Healthcheck ---
# Check if the Quart server is responding on port 5000 inside the container
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl --fail --silent --show-error http://localhost:5000/ || exit 1

# --- Start Command ---
# Use the absolute path to python to ensure the correct environment and installed modules are found
CMD ["/usr/local/bin/python", "-m", "quart", "run", "--host", "0.0.0.0", "--port", "5000", "--no-reload"]