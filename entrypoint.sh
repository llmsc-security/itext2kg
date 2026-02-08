#!/bin/bash
# Entrypoint script for iText2KG Docker container
# Starts the iText2KG application

set -e

echo "Starting iText2KG application..."

# Set environment variables
export PYTHONUNBUFFERED=1

# Change to app directory
cd /app

# Run the application - can be overridden with arguments
exec python -m itext2kg "$@"
