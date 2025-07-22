#!/bin/bash

set -e  # Exit on error

# Path to submodule
SUBMODULE_PATH="data"
SUBMODULE_BRANCH="main"

echo ">>> Updating submodule: $SUBMODULE_PATH"

# Go into the submodule
cd "$SUBMODULE_PATH"

# Fetch and checkout the latest version of the main branch
echo ">>> Fetching latest changes from origin/$SUBMODULE_BRANCH"
git fetch origin
git checkout $SUBMODULE_BRANCH
git pull origin $SUBMODULE_BRANCH

# Go back to the main repo
cd ..

# Stage the updated submodule reference
echo ">>> Staging submodule update"
git add "$SUBMODULE_PATH"

# Commit the update
echo ">>> Committing updated submodule reference"
git commit -m "Update submodule '$SUBMODULE_PATH' to latest '$SUBMODULE_BRANCH'"

echo "✅ Submodule '$SUBMODULE_PATH' updated successfully."
