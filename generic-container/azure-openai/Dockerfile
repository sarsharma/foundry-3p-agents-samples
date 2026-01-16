FROM python:3.12-slim
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

COPY requirements.txt requirements.txt

# Install system dependencies needed for git-based pip installs
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies using pip
RUN pip install -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

EXPOSE 8080

# Copy entire project (respecting .dockerignore)
COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "langgraph_agent:flask_app"]
