#!/usr/bin/env bash
#
# release.sh — Bump version, build, and publish neuroshard-ai to PyPI
#
# Usage:
#   ./release.sh           # bump patch  (0.2.58 -> 0.2.59)
#   ./release.sh minor     # bump minor  (0.2.58 -> 0.3.0)
#   ./release.sh major     # bump major  (0.2.58 -> 1.0.0)
#   ./release.sh 1.2.3     # set exact version
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_FILE="$SCRIPT_DIR/src/neuroshard/version.py"
VENV="$SCRIPT_DIR/venv_build"

# --------------- activate venv_build ---------------
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  echo "Using venv: $VENV"
else
  echo "Error: venv_build not found at $VENV"
  echo "Create it with: python -m venv venv_build && venv_build/bin/pip install build twine"
  exit 1
fi

# --------------- read current version ---------------
CURRENT=$(grep -oP '(?<=__version__ = ")[^"]+' "$VERSION_FILE")
echo "Current version: $CURRENT"

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

# --------------- compute next version ---------------
BUMP="${1:-patch}"

case "$BUMP" in
  patch)
    PATCH=$((PATCH + 1))
    NEXT="$MAJOR.$MINOR.$PATCH"
    ;;
  minor)
    MINOR=$((MINOR + 1))
    NEXT="$MAJOR.$MINOR.0"
    ;;
  major)
    MAJOR=$((MAJOR + 1))
    NEXT="$MAJOR.0.0"
    ;;
  *)
    # treat argument as an explicit version string
    if [[ "$BUMP" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      NEXT="$BUMP"
    else
      echo "Error: invalid argument '$BUMP'"
      echo "Usage: $0 [patch|minor|major|X.Y.Z]"
      exit 1
    fi
    ;;
esac

echo "Next version:    $NEXT"
echo ""

# --------------- confirm ---------------
read -rp "Proceed with release $NEXT? [y/N] " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 0
fi

# --------------- update version file ---------------
echo "__version__ = \"$NEXT\"" > "$VERSION_FILE"
echo "✓ Updated $VERSION_FILE"

# --------------- clean old artifacts ---------------
rm -rf "$SCRIPT_DIR/dist" "$SCRIPT_DIR/build" "$SCRIPT_DIR"/src/*.egg-info
echo "✓ Cleaned build artifacts"

# --------------- build ---------------
echo ""
echo "Building neuroshard-ai $NEXT ..."
python -m build "$SCRIPT_DIR"
echo ""
echo "✓ Build complete"

# --------------- upload ---------------
echo ""
echo "Uploading to PyPI ..."
python -m twine upload "$SCRIPT_DIR/dist/"*
echo ""
echo "✓ Published neuroshard-ai $NEXT to PyPI"
echo "  https://pypi.org/project/neuroshard-ai/$NEXT/"
