# Filename: Dockerfile
# Description: Docker build instructions for the Genius AI Sales System.
# Version: 3.0 (Postgres, Cleaned Dependencies)

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# Added: libpq-dev (for asyncpg), git (already there), curl (already there)
# Kept: Playwright dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libnss3 \
    libatk-bridge2.0-0 \
    libxkbcommon0 \
    libgbm1 \
    libasound2 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libpango-1.0-0 \
    libcairo2 \
    libatk1.0-0 \
    libgtk-3-0 \
    git \
    libpq-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers and Spacy model (if needed)
# Keep spacy download only if NLP features are actively used by an agent
RUN python -m playwright install --with-deps # Installs browsers needed by Playwright
# RUN python -m spacy download en_core_web_sm # Uncomment if spacy is used

# Copy application code
COPY . .

# Expose the port the UI/API server runs on (default 5000 for Quart)
EXPOSE 5000

# Healthcheck (Optional but recommended) - Checks if the web server is responding
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:5000/ || exit 1

# Command to run the application
CMD ["python", "main.py"]