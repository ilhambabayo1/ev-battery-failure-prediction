FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install dependencies first (better layer caching).
COPY requirements.txt .
# CPU-only torch wheel keeps the image ~10x smaller than the default CUDA one.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code and trained artifacts.
COPY app ./app
COPY static ./static
COPY models ./models

EXPOSE 8000

# ${PORT} is honored so it also works on platforms that inject one (Heroku, Render, Cloud Run).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
