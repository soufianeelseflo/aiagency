# Filename: Dockerfile
# Description: Docker build instructions for the Genius AI Sales System.
# Version: 4.1 (Postgres, Cleaned Dependencies, Corrected Healthcheck)

# Use a specific Python 3.10 slim image for reproducibility
FROM python:3.10.13-slim-bookworm

WORKDIR /app

# Install system dependencies
# Added: libpq-dev (for asyncpg), git, curl
# Kept: Playwright dependencies
# Using non-interactive frontend to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    # Playwright dependencies
    libnss3 \
    libatk-bridge2.0-0 \
    libxkbcommon0 \
    libgbm1 \
    libasound2 \
    libatk1.0-0 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libexpat1 \
    libfontconfig1 \
    libgcc1 \
    libgconf-2-4 \
    libgdk-pixbuf2.0-0 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    ca-certificates \
    fonts-liberation \
    libappindicator1 \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libc6 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libexpat1 \
    libfontconfig1 \
    libgbm1 \
    libgcc1 \
    libgconf-2-4 \
    libgdk-pixbuf2.0-0 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    lsb-release \
    wget \
    xdg-utils \
    # Other dependencies
    git \
    libpq-dev \
    build-essential \
    # Clean up APT cache and lists
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from the requirements.txt
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
# Consider using a virtual environment
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (needed by BrowsingAgent)
# Run as non-root user potentially? For now, run as root.
RUN python -m playwright install --with-deps chromium

# Copy application code into the container
COPY . .

# Expose the port the UI/API server runs on (default 5000 for Quart/Hypercorn)
# Make port configurable via PORT env var, default to 5000
ENV PORT 5000
EXPOSE ${PORT}

# Healthcheck (Checks if the web server is responding)
# Updated to use the correct API endpoint from ui/app.py
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:${PORT}/api/status_kpi || exit 1

# Set the default command to run the application using main.py
# Use exec form to properly handle signals
CMD ["python", "main.py"]
