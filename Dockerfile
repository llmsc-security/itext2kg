# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create non-root user for security
RUN groupadd -r itext2kg && \
    useradd -r -g itext2kg -u 1000 -m -d /home/itext2kg itext2kg && \
    chown -R itext2kg:itext2kg /app

# Switch to non-root user
USER itext2kg

# Set working directory
WORKDIR /app

# Expose the port from port_mapping_50_gap10.json
EXPOSE 11380

# Default command - run the main module
CMD ["python", "-m", "itext2kg"]
