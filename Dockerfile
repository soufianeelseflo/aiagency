

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    libx11-6 libxkbcommon-x11-0 libglib2.0-0 libnss3 libatk1.0-0 \
    libatk-bridge2.0-0 libcups2 libdrm2 libxcomposite1 libxdamage1 \
    libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium

RUN pip install openrouter==1.0.0

COPY web_interface/backend/app.py .

ENV PYTHONUNBUFFERED=1

EXPOSE 80

CMD ["python", "app.py"]