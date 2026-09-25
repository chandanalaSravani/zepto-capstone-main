FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False \
    HF_HOME=/app/.hf_cache \
    MOCK_LLM=1

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .

# Download the embedding model and build the ChromaDB index at build time,
# so the running container needs no network access at all in mock mode.
RUN python ingest.py
# Model is now cached in the image: never contact the Hugging Face Hub at runtime.
ENV HF_HUB_OFFLINE=1

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
