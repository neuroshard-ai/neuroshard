#!/bin/bash

# Ensure we are using the project root as context
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load GitHub token from .env file if available
if [ -f "$PROJECT_ROOT/website/.env" ]; then
    export $(grep -E "^GITHUB_TOKEN=" "$PROJECT_ROOT/website/.env" | xargs)
elif [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -E "^GITHUB_TOKEN=" "$PROJECT_ROOT/.env" | xargs)
fi

# Read the version from the version file by setting PYTHONPATH to project root
VERSION=$(PYTHONPATH="$PROJECT_ROOT" python3 -c "from neuroshard.version import __version__; print(__version__)")

if [ -z "$VERSION" ]; then
    echo "❌ Error: Could not read version from neuroshard/version.py"
    exit 1
fi

echo "🚀 Tagging release $VERSION"

# Prepare authenticated remote URL if token is available
AUTH_REMOTE=""
if [ -n "$GITHUB_TOKEN" ]; then
    ORIGINAL_URL=$(git remote get-url origin)
    # Extract repo path (everything after github.com/)
    REPO_PATH=$(echo "$ORIGINAL_URL" | sed -E 's|https?://[^/]+/(.+)$|\1|')
    AUTH_REMOTE="https://${GITHUB_TOKEN}@github.com/${REPO_PATH}"
fi

# Check if tag already exists
if git rev-parse "$VERSION" >/dev/null 2>&1; then
    echo "⚠️  Tag $VERSION already exists locally."
    read -p "Do you want to delete and re-tag it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    git tag -d "$VERSION"
    # Check if tag exists on remote before trying to delete
    REMOTE_TAG_EXISTS=false
    if [ -n "$AUTH_REMOTE" ]; then
        if git ls-remote --tags "$AUTH_REMOTE" "refs/tags/$VERSION" >/dev/null 2>&1; then
            REMOTE_TAG_EXISTS=true
        fi
    else
        if git ls-remote --tags origin "refs/tags/$VERSION" >/dev/null 2>&1; then
            REMOTE_TAG_EXISTS=true
        fi
    fi
    
    # Delete remote tag only if it exists
    if [ "$REMOTE_TAG_EXISTS" = true ]; then
        if [ -n "$AUTH_REMOTE" ]; then
            git push "$AUTH_REMOTE" --delete "$VERSION"
        else
            git push origin --delete "$VERSION"
        fi
    else
        echo "ℹ️  Tag $VERSION does not exist on remote, skipping deletion."
    fi
fi

# Create git tag
git tag -a "$VERSION" -m "Release version $VERSION"

# Push to remote with authentication
if [ -n "$AUTH_REMOTE" ]; then
    git push "$AUTH_REMOTE" "$VERSION"
else
    # Fallback to regular push (will prompt for credentials)
    git push origin "$VERSION"
fi

echo "✅ Pushed tag $VERSION to GitHub. The Actions workflow should start building now."
