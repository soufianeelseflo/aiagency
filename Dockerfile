# Filename: Dockerfile
# Description: Dockerfile for deploying the Nolli AI Sales System (Level 50+ ready)
# Version: 1.0

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
# Update apt cache, install necessary system libs (including those for psycopg2, playwright browsers), clean up cache
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  # For psycopg2 (binary needs libpq)
  libpq-dev \
  # For Playwright Browsers & dependencies
  build-essential \
  curl \
  wget \
  gnupg \
  # Playwright browser dependencies (check latest Playwright docs if needed)
  libnss3 \
  libnspr4 \
  libdbus-1-3 \
  libatk1.0-0 \
  libatk-bridge2.0-0 \
  libcups2 \
  libdrm2 \
  libxcomposite1 \
  libxdamage1 \
  libxfixes3 \
  libxrandr2 \
  libgbm1 \
  libpango-1.0-0 \
  libcairo2 \
  libasound2 \
  libexpat1 \
  libxcb1 \
  libxkbcommon0 \
  libx11-6 \
  libxext6 \
  libfontconfig1 \
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
# Installing only chromium as it's commonly used and reduces image size. Add others if needed.
RUN playwright install chromium --with-deps

# --- Application Code ---
# Copy the rest of the application code
COPY . .

# --- Create necessary directories and set permissions (if needed) ---
# Assuming logs and temp files go into subdirs within /app
# Adjust if your application needs different paths or permissions
RUN mkdir -p /app/logs /app/temp_audio /app/temp_downloads /app/learning_for_AI && \
  # Set permissions if running as non-root (good practice, but requires USER instruction)
  # For simplicity now, assuming root execution (default in many base images)
  # If running as non-root, add user creation and chown commands here.
  chmod -R 777 /app/logs /app/temp_audio /app/temp_downloads
# Note: 777 is permissive, adjust as needed for security.

# --- Expose Port ---
# Expose the port the application listens on (matches ENV PORT default)
EXPOSE 5000

# --- Healthcheck ---
# Check if the Quart server is responding on port 5000 inside the container
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl --fail --silent --show-error http://localhost:5000/ || exit 1

# --- Start Command ---
# Use Quart's run command, binding to the configured HOST and PORT
# --- Start Command ---
# Use python -m to ensure the correct Python environment runs Quart
CMD ["python", "-m", "quart", "run", "--host", "0.0.0.0", "--port", "5000", "--no-reload"]