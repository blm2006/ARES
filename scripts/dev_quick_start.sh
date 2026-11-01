#!/bin/bash
# Quick start for development: Redis + DB + Phase 1

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================"
echo "yolo-al QUICK START (Dev)"
echo "========================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Redis
echo -e "${YELLOW}Step 1: Starting Redis...${NC}"
docker compose up -d
sleep 2

# Step 2: Database
echo -e "${YELLOW}Step 2: Initializing Database...${NC}"
python -m scripts.init_db

# Step 3: Phase 1 - Realtime (no collector)
echo -e "${YELLOW}Step 3: Phase 1 - Realtime Detection${NC}"
echo -e "${GREEN}Dashboard: http://127.0.0.1:8000${NC}"
echo -e "${GREEN}Running 69 tests to validate setup...${NC}"
pytest -q

echo -e "${YELLOW}Starting realtime pipeline (no collector)...${NC}"
echo "To enable collector: python -m realtime"
echo "Press Ctrl+C to stop"

python -m realtime --no-collector

echo -e "${GREEN}All done!${NC}"
