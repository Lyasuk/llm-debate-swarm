# Minimal image for the FastAPI serving layer. Pinned base (no :latest) so a
# rollback target exists.
FROM python:3.12-slim

WORKDIR /app

# Install deps first (better layer caching), then the package.
COPY pyproject.toml README.md ./
COPY src ./src
COPY config.yaml ./
RUN pip install --no-cache-dir -e ".[graph,otlp,serve]"

EXPOSE 8000

# Liveness is /health; orchestrators can probe it.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "llm_debate_swarm.service:app", "--host", "0.0.0.0", "--port", "8000"]
