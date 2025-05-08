# Filename: Dockerfile
# Description: Docker build instructions for the Nolli AI Sales System.
# Version: 4.2 (Added reportlab dependencies, explicit deepgram upgrade, invoice dir)

# Use a specific Python 3.10 slim image for reproducibility
FROM python:3.10.13-slim-bookworm

WORKDIR /app

# Install system dependencies
# Added: libpq-dev, git, curl, reportlab dependencies (libjpeg, zlib, freetype, libxml2, libxslt1)
# Kept: Playwright dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  git \
  libpq-dev \
  build-essential \
  # Reportlab dependencies (adjust if specific errors occur during PDF generation)
  libjpeg-dev \
  zlib1g-dev \
  libfreetype6-dev \
  libxml2-dev \
  libxslt1-dev \
  # Playwright dependencies (ensure no unnecessary duplication)
  libnss3 libatk-bridge2.0-0 libxkbcommon0 libgbm1 libasound2 libatk1.0-0 \
  libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libgcc1 \
  libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 \
  libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 \
  libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 \
  libxi6 libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates \
  fonts-liberation libappindicator1 lsb-release wget xdg-utils \
  # Clean up APT cache and lists
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from the requirements.txt
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

# Explicitly install/upgrade deepgram-sdk to ensure correct version/submodules
RUN pip install --no-cache-dir --upgrade "deepgram-sdk>=3.4.0,<3.8.0"

# Install Playwright browsers (needed by BrowsingAgent)
RUN python -m playwright install --with-deps chromium

# Copy application code into the container
COPY . .

# Create directory for invoices
RUN mkdir -p /app/invoices && chown -R nobody:nogroup /app/invoices # Create and set permissions if running as non-root later

# Expose the port the UI/API server runs on (default 5000 for Quart/Hypercorn)
# Make port configurable via PORT env var, default to 5000
ENV PORT 5000
EXPOSE ${PORT}

# Healthcheck (Checks if the web server is responding)
# Uses curl installed above. Should work once dependency issues are fixed.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:${PORT}/api/status_kpi || exit 1

# Set the default command to run the application using main.py
# Use exec form to properly handle signals
CMD ["python", "main.py"]