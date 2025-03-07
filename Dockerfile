# Stage 1: Build the React frontend
FROM node:18 AS frontend
WORKDIR /app/web_interface/frontend
COPY web_interface/frontend/package.json ./
COPY web_interface/frontend/src/ ./src/
COPY web_interface/frontend/public/ ./public/
RUN npm install
RUN npm run build

# Stage 2: Build the Python backend
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
 libpq-dev gcc \
 libx11-6 libxkbcommon-x11-0 libglib2.0-0 libnss3 libatk1.0-0 \
 libatk-bridge2.0-0 libcups2 libdrm2 libxcomposite1 libxdamage1 \
 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 ffmpeg \
 curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && playwright install --with-deps chromium

# Copy frontend build
COPY --from=frontend /app/web_interface/frontend/build ./static

# Copy backend and orchestrator
COPY web_interface/backend/ ./web_interface/backend/
COPY orchestrator.py .
COPY agents/ ./agents/ 2>/dev/null || :
COPY integrations/ ./integrations/ 2>/dev/null || :
COPY utils/ ./utils/ 2>/dev/null || :

ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=web_interface/backend/app.py
ENV FLASK_ENV=production

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s \
 CMD curl -f http://localhost:80/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:80", "web_interface.backend.app:app"]