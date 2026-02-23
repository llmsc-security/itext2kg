#!/bin/bash
# Entrypoint script for iText2KG Docker container
# Starts the iText2KG FastAPI application

set -e

echo "Starting iText2KG FastAPI application..."

# Set environment variables
export PYTHONUNBUFFERED=1

# Change to app directory
cd /app

# Run the FastAPI application directly
exec python main.py "$@"
