FROM python:3.11-slim

WORKDIR /app

# Build-time only: some wheels still need a compiler on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before the rest of the source so that edits to
# application code do not invalidate the dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY *.py ./
RUN pip install --no-cache-dir .

COPY . .

RUN mkdir -p data output

ENV PYTHONUNBUFFERED=1

CMD ["notioniq"]
