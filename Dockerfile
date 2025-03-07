# Step 1: Build the robot’s face (frontend) with Node.js
FROM node:18 as frontend
WORKDIR /app/web_interface/frontend
COPY web_interface/frontend /app/web_interface/frontend
ENV REACT_APP_API_BASE_URL=/api
RUN npm install && npm run build

# Step 2: Build the robot’s brain (backend) with Python
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    libx11-6 libxkbcommon-x11-0 libglib2.0-0 libnss3 libatk1.0-0 \
    libatk-bridge2.0-0 libcups2 libdrm2 libxcomposite1 libxdamage1 \
    libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium

# Copy the built frontend into the ROOT of the app directory
COPY --from=frontend /app/web_interface/frontend/build .

# Copy the backend code
COPY agents/ ./agents/
COPY integrations/ ./integrations/
COPY utils/ ./utils/
COPY orchestrator.py .
COPY web_interface/backend/ ./

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 80

# Start the app
CMD ["python", "app.py"]