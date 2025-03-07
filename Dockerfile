
# Step 1: Build the robot’s face (frontend) with Node.js
FROM node:18 as frontend
WORKDIR /app/web_interface/frontend  # Changed!
COPY web_interface/frontend /app/web_interface/frontend  # Changed!
ENV REACT_APP_API_BASE_URL=/api
RUN npm install && npm run build


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