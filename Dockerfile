FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data and output directories
RUN mkdir -p data output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "notion_organizer.py"]