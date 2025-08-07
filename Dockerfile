# Multi-stage build for NotionIQ
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 notioniq && \
    mkdir -p /app/data /app/output && \
    chown -R notioniq:notioniq /app

# Copy dependencies from builder
COPY --from=builder /root/.local /home/notioniq/.local

# Copy application code
COPY --chown=notioniq:notioniq . .

# Switch to non-root user
USER notioniq

# Add local bin to PATH
ENV PATH=/home/notioniq/.local/bin:$PATH

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production

# Create volume mount points
VOLUME ["/app/data", "/app/output"]

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
ENTRYPOINT ["python", "notion_organizer.py"]
CMD ["--help"]