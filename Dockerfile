# Filename: Dockerfile
# Description: Dockerfile for deploying the Nolli AI Sales System (IGNIS Definitive Transmutation)
# Version: 2.1 (Standard Pip Install, Explicit PATH, Robust Python Invocation - RE-AFFIRMED)

# --- Base Image ---
# Use python:3.11-bookworm (non-slim) for a more complete and standard environment
FROM python:3.11-bookworm

# --- Environment Variables ---
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers \
    PORT=5000 \
    HOST=0.0.0.0 \
    # Add standard user binary path to system PATH for pip-installed CLIs
    # For root user (default in Docker), this is /root/.local/bin
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
    # Playwright deps (ensure this list is comprehensive for -bookworm image)
    libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2 libexpat1 libxcb1 \
    libxkbcommon0 libx11-6 libxext6 libfontconfig1 \
    # Clean up apt cache
    && rm -rf /var/lib/apt/lists/*

# --- Application Dependencies ---
# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies using the standard method into the system site-packages
# This ensures executables like 'playwright' and 'quart' are typically placed in a PATH-accessible location
# (e.g., /usr/local/bin or /root/.local/bin which we added to PATH)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Install Playwright Browsers ---
# Use 'python -m playwright' which directly uses the installed Playwright module's CLI component.
# This is more robust than relying on the shell finding 'playwright' directly in PATH immediately after pip install.
RUN python -m playwright install chromium --with-deps

# --- Application Code ---
# Copy the rest of the application code
COPY . .

# --- Create necessary directories ---
# These paths must match your application's expectations (e.g., from config/settings.py)
# This includes the learning_for_AI directory.
RUN mkdir -p /app/logs /app/temp_audio /app/temp_downloads /app/learning_for_AI && \
    # Grant write permissions. For production, consider a non-root user and more specific ownership.
    chmod -R 777 /app/logs /app/temp_audio /app/temp_downloads /app/learning_for_AI

# --- Expose Port ---
# This is the port your Quart application will listen on INSIDE the container
EXPOSE 5000

# --- Healthcheck ---
# Checks if the Quart server is responding on port 5000 inside the container
# Increased start-period to give the app more time to initialize fully.
HEALTHCHECK --interval=45s --timeout=10s --start-period=30s --retries=3 \
  CMD curl --fail --silent --show-error http://localhost:5000/ || exit 1

# --- Start Command ---
# Use 'python -m quart' which explicitly uses the Python interpreter to find and run the quart module.
# This is the most reliable way to start the Quart application.
CMD ["python", "-m", "quart", "run", "--host", "0.0.0.0", "--port", "5000", "--no-reload"]