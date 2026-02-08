#!/bin/bash
# Build and run script for iText2KG Docker container
# Maps host port 11380 to container port 11380

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building iText2KG Docker image..."
docker build -t itext2kg:latest .

echo ""
echo "Starting iText2KG container..."
echo "Container will be available at http://localhost:11380"
echo ""

docker run \
    --rm \
    -p 11380:11380 \
    --name itext2kg \
    -v "$(pwd):/app:ro" \
    --env-file .env \
    itext2kg:latest
