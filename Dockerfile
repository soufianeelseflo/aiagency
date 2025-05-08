# Filename: Dockerfile
# Description: Dockerfile for deploying the Nolli AI Sales System (Level 50+ ready)
# Version: 1.3 (Ultimate Debug CMD for Python Env/Module Check)

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
RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir --target=/app/deps -r requirements.txt

# --- Install Playwright Browsers ---
RUN playwright install chromium --with-deps

# --- Application Code ---
COPY . .

# --- Create necessary directories ---
RUN mkdir -p /app/logs /app/temp_audio /app/temp_downloads /app/learning_for_AI && \
  chmod -R 777 /app/logs /app/temp_audio /app/temp_downloads

# --- Expose Port ---
EXPOSE 5000

# --- Healthcheck ---
# Still useful to see if the container stays up long enough for the debug CMD
# Modify to just check if the container is running initially
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD exit 0

# --- Start Command (ULTIMATE DEBUG) ---
# This command checks PYTHONPATH, sys.path, site-packages, and tries to find the 'quart' module spec directly.
CMD ["/usr/local/bin/python", "-c", "import os, sys, site, importlib.util; print('---ENV PYTHONPATH---'); print(os.environ.get('PYTHONPATH')); print('---SYS EXECUTABLE---'); print(sys.executable); print('---SYS PATH---'); print(sys.path); print('---SITE PACKAGES---'); print(site.getsitepackages()); print('---FINDING QUART SPEC---'); spec = importlib.util.find_spec('quart'); print(f'Quart Spec Found: {spec is not None}'); print(f'Quart Spec Origin: {spec.origin if spec else None}'); exit(0 if spec else 1)"]