#!/bin/bash
# NeuroShard Release Script
# Usage: ./scripts/release.sh [patch|minor|major]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VERSION_FILE="$PROJECT_DIR/src/neuroshard/version.py"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}NeuroShard Release Script${NC}"
echo "================================"

# Get current version
CURRENT_VERSION=$(grep -oP '__version__ = "\K[^"]+' "$VERSION_FILE")
echo -e "Current version: ${YELLOW}$CURRENT_VERSION${NC}"

# Parse version components
IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR="${VERSION_PARTS[0]}"
MINOR="${VERSION_PARTS[1]}"
PATCH="${VERSION_PARTS[2]}"

# Determine bump type
BUMP_TYPE="${1:-patch}"

case "$BUMP_TYPE" in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
    *)
        echo -e "${RED}Invalid bump type: $BUMP_TYPE${NC}"
        echo "Usage: $0 [patch|minor|major]"
        exit 1
        ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo -e "New version: ${GREEN}$NEW_VERSION${NC} (${BUMP_TYPE} bump)"
echo ""

# Confirm
read -p "Proceed with release? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Update version file
echo -e "\n${YELLOW}Updating version...${NC}"
sed -i "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" "$VERSION_FILE"
echo "Updated $VERSION_FILE"

# Clean old builds
echo -e "\n${YELLOW}Cleaning old builds...${NC}"
rm -rf "$PROJECT_DIR/dist/"*
rm -rf "$PROJECT_DIR/build/"
rm -rf "$PROJECT_DIR"/*.egg-info
rm -rf "$PROJECT_DIR/src"/*.egg-info

# Regenerate proto files (inside neuroshard package)
echo -e "\n${YELLOW}Regenerating proto files...${NC}"
cd "$PROJECT_DIR"
python -m grpc_tools.protoc \
    -I./src/neuroshard/protos \
    --python_out=./src/neuroshard/protos \
    --grpc_python_out=./src/neuroshard/protos \
    ./src/neuroshard/protos/neuroshard.proto
# Fix the import in grpc file (protoc generates bare import, we need package import)
sed -i 's/^import neuroshard_pb2 as/from neuroshard.protos import neuroshard_pb2 as/' src/neuroshard/protos/neuroshard_pb2_grpc.py
echo -e "${GREEN}Proto files regenerated${NC}"

# Build
echo -e "\n${YELLOW}Building package...${NC}"
cd "$PROJECT_DIR"
python -m build

# Verify no whitepaper
echo -e "\n${YELLOW}Verifying package contents...${NC}"
if unzip -l dist/*.whl | grep -qi "whitepaper\|\.tex\|\.pdf"; then
    echo -e "${RED}ERROR: Whitepaper files found in package!${NC}"
    exit 1
fi
echo -e "${GREEN}OK - No whitepaper files in package${NC}"

# Show package contents summary
echo -e "\n${YELLOW}Package contents:${NC}"
unzip -l dist/*.whl | tail -5

# Upload
echo -e "\n${YELLOW}Uploading to PyPI...${NC}"
twine upload dist/*

echo -e "\n${GREEN}Released neuroshard-ai v$NEW_VERSION${NC}"
echo -e "View at: ${YELLOW}https://pypi.org/project/neuroshard-ai/$NEW_VERSION/${NC}"
