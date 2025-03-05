# Use Python 3.11 slim as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for psycopg2, playwright, and other libraries
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    libx11-6 libxkbcommon-x11-0 libglib2.0-0 libnss3 libatk1.0-0 \
    libatk-bridge2.0-0 libcups2 libdrm2 libxcomposite1 libxdamage1 \
    libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (optimization for caching)
COPY requirements.txt .

# Install Python dependencies, including Playwright
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium

# Add openrouter explicitly since it’s missing from your current requirements.txt
RUN pip install openrouter==0.1.0

# Copy the entire codebase
COPY . .

# Set environment variable to ensure Python output is unbuffered
ENV PYTHONUNBUFFERED=1

# Expose port 5000 (as per coolify.yml)
EXPOSE 5000

# Run orchestrator.py
CMD ["python", "orchestrator.py"]