================================================
File: Dockerfile
================================================
# Use Python 3.11 slim as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for psycopg2, Playwright, and other libraries
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    libx11-6 libxkbcommon-x11-0 libglib2.0-0 libnss3 libatk1.0-0 \
    libatk-bridge2.0-0 libcups2 libdrm2 libxcomposite1 libxdamage1 \
    libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file from root
COPY requirements.txt .

# Install Python dependencies, including Playwright
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium

# Install openrouter explicitly
RUN pip install openrouter==1.0.0

# Copy the Flask app from web_interface/backend/
COPY web_interface/backend/app.py .

# Set environment variable for unbuffered output
ENV PYTHONUNBUFFERED=1

# Expose port 80 (Coolify default)
EXPOSE 80

# Run the Flask app
CMD ["python", "app.py"]