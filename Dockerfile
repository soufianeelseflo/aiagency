# Step 1: Build the control panel (frontend) with Node.js
FROM node:18 AS frontend
WORKDIR /app
COPY web_interface/frontend /app
ENV REACT_APP_API_BASE_URL=/api
RUN npm install && npm run build

# Step 2: Build the robot’s brain (backend) with Python
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    libx11-6 libxkbcommon-x11-0 libglib2.0-0 libnss3 libatk1.0-0 \
    libatk-bridge2.0-0 libcups2 libdrm2 libxcomposite1 libxdamage1 \
    libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install openrouter==1.0.0 \
    && playwright install --with-deps chromium
COPY --from=frontend /app/build ./build
COPY agents/ ./agents/
COPY integrations/ ./integrations/
COPY utils/ ./utils/
COPY orchestrator.py .
COPY web_interface/backend/ ./
ENV PYTHONUNBUFFERED=1
EXPOSE 80
CMD ["python", "app.py"]