# --- DEVELOPMENT STAGE (optional) ---
    FROM python:3.11-slim AS development

    WORKDIR /app
    
    RUN apt-get update && apt-get install -y \
        curl \
        git \
        build-essential \
        && apt-get clean
    
    RUN curl -sSL https://install.python-poetry.org | python3 -
    ENV PATH="/root/.local/bin:$PATH"
    
    COPY . /app
    RUN poetry install --no-interaction --no-ansi
    
    # --- BUILDER STAGE ---
    FROM ankane/pgvector:v0.5.1 AS builder
    
    RUN apt-get update && apt-get install -y \
        python3 \
        python3-venv \
        python3-pip \
        python3-full \
        build-essential \
        libpq-dev \
        python3-dev \
        && rm -rf /var/lib/apt/lists/*
    
    # === ENV SETUP ===
    ARG LETTA_ENVIRONMENT=PRODUCTION
    ARG LETTA_VERSION
    
    ENV LETTA_ENVIRONMENT=${LETTA_ENVIRONMENT} \
        LETTA_VERSION=${LETTA_VERSION} \
        POETRY_NO_INTERACTION=1 \
        POETRY_CACHE_DIR=/tmp/poetry_cache \
        VIRTUAL_ENV="/opt/venv" \
        PATH="/opt/venv/bin:$PATH"
    
    WORKDIR /app
    
    # Create venv and install Poetry
    RUN python3 -m venv $VIRTUAL_ENV
    RUN pip install --no-cache-dir poetry==2.1.3
    
    # Install Python dependencies and the CLI package
    COPY pyproject.toml poetry.lock ./
    COPY . .
    
    RUN poetry lock && \
        poetry install --all-extras && \
        poetry run pip install . && \
        poetry run pip install e2b e2b_code_interpreter orjson && \
        rm -rf $POETRY_CACHE_DIR
    
    # --- RUNTIME STAGE ---
    FROM ankane/pgvector:v0.5.1 AS runtime
    
    ARG NODE_VERSION=22
    ARG LETTA_ENVIRONMENT=PRODUCTION
    ARG LETTA_VERSION
    
    ENV LETTA_ENVIRONMENT=${LETTA_ENVIRONMENT} \
        LETTA_VERSION=${LETTA_VERSION} \
        VIRTUAL_ENV="/opt/venv" \
        PATH="/opt/venv/bin:$PATH" \
        POSTGRES_USER=letta \
        POSTGRES_PASSWORD=letta \
        POSTGRES_DB=letta \
        COMPOSIO_DISABLE_VERSION_CHECK=true
    
    WORKDIR /app
    
    RUN apt-get update && \
        apt-get install -y curl python3 python3-venv && \
        curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - && \
        apt-get install -y nodejs && \
        curl -L https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.96.0/otelcol-contrib_0.96.0_linux_amd64.tar.gz -o /tmp/otel-collector.tar.gz && \
        tar xzf /tmp/otel-collector.tar.gz -C /usr/local/bin && \
        rm /tmp/otel-collector.tar.gz && \
        mkdir -p /etc/otel && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*
    
    COPY otel/otel-collector-config-file.yaml /etc/otel/config-file.yaml
    COPY otel/otel-collector-config-clickhouse.yaml /etc/otel/config-clickhouse.yaml
    
    COPY --from=builder /app /app
    COPY --from=builder /opt/venv /opt/venv
    
    COPY init.sql /docker-entrypoint-initdb.d/
    RUN chmod +x ./letta/server/startup.sh
    
    EXPOSE 8283 5432 4317 4318
    
    ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
    CMD ["./letta/server/startup.sh"]
    