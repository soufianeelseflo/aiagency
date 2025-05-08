# Filename: Dockerfile
# Description: Docker build instructions for the Nolli AI Sales System.
# Version: 4.3 (Simplified pip install, relies on requirements.txt for deepgram)

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
  # Reportlab dependencies
  libjpeg-dev \
  zlib1g-dev \
  libfreetype6-dev \
  libxml2-dev \
  libxslt1-dev \
  # Playwright dependencies
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
# REMOVED: Redundant deepgram-sdk install line was here

# Install Playwright browsers (needed by BrowseAgent)
# This ensures playwright downloads browsers correctly after pip packages are installed
RUN python -m playwright install --with-deps chromium

# Copy application code into the container
COPY . .

# Create directory for invoices
RUN mkdir -p /app/invoices && chown -R nobody:nogroup /app/invoices

# Expose the port the UI/API server runs on.
# Your main.py uses os.getenv('PORT', '5000'), so it will adapt.
# Coolify usually provides a PORT env var.
ENV PORT 5000 
EXPOSE ${PORT} 

# Healthcheck (Checks if the web server is responding)
# Ensure your app has a /api/status_kpi endpoint for this to work
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:${PORT}/api/status_kpi || exit 1

# Set the default command to run the application using main.py which uses Hypercorn
# This will be used if Coolify's "Start Command" is empty.
CMD ["python", "main.py"]