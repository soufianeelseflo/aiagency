FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    curl libnss3 libatk-bridge2.0-0 libxkbcommon0 libgbm1 libasound2 libatspi2.0-0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libpango-1.0-0 libcairo2 \
    libatk1.0-0 libgtk-3-0 git && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/lanmaster53/recon-ng.git /opt/recon-ng && \
    cd /opt/recon-ng && pip install --no-cache-dir -r REQUIREMENTS
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m playwright install --with-deps && python -m spacy download en_core_web_sm
COPY . .
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:5000 || exit 1
CMD ["python", "main.py"]