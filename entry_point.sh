#!/bin/bash
set -e

# Start the itext2KG FastAPI application on port 11380
echo "Starting iText2KG FastAPI application..."
exec python main.py
