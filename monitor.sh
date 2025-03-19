#!/bin/bash

# Markdown Git Auto-Commit Script
# Watches for new or modified .md files and automatically commits and pushes them

# Configuration
WATCH_DIR=${1:-.}
REMOTE=${REMOTE:-"origin"}
BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
DEBOUNCE_TIME=${DEBOUNCE_TIME:-2}  # Time in seconds to wait before committing

# Change to the watch directory
cd "$WATCH_DIR" || { echo "Error: Cannot change to directory $WATCH_DIR"; exit 1; }
WATCH_DIR=$(pwd)  # Get absolute path

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    echo "Error: $WATCH_DIR is not a git repository."
    exit 1
fi

echo "Starting Markdown Auto-Commit script..."
echo "Watching directory: $WATCH_DIR"
echo "Target branch: $REMOTE/$BRANCH"
echo "========================================"

# Function to commit and push changes
commit_and_push() {
    echo "Processing changes..."
    
    # Get list of modified markdown files
    CHANGED_FILES=$(git status --porcelain | grep '\.md$' | sed 's/^...//g')
    
    if [ -z "$CHANGED_FILES" ]; then
        echo "No markdown files to commit."
        return
    fi
    
    # Add all markdown files
    git add "*.md"
    
    # Check if there are changes to commit
    if ! git diff --cached --quiet; then
        # Count the number of files changed
        FILE_COUNT=$(echo "$CHANGED_FILES" | wc -l | tr -d ' ')
        
        # Construct commit message
        if [ "$FILE_COUNT" -eq 1 ]; then
            FILENAME=$(basename "$(echo "$CHANGED_FILES" | tr -d ' ')")
            COMMIT_MSG="Add markdown: $FILENAME"
        else
            COMMIT_MSG="Add $FILE_COUNT markdown files"
        fi
        
        # Commit changes
        if git commit -m "$COMMIT_MSG"; then
            echo "✓ Successfully committed with message: $COMMIT_MSG"
            
            # Push changes
            echo "Pushing to $REMOTE/$BRANCH..."
            if git push $REMOTE $BRANCH; then
                echo "✓ Successfully pushed changes."
            else
                echo "✗ Error: Failed to push changes. Will try again on next change."
            fi
        else
            echo "✗ Error: Failed to commit changes."
        fi
    else
        echo "No changes to commit."
    fi
}

# Try to detect the available file monitoring tool
if command -v inotifywait &> /dev/null; then
    echo "Using inotifywait for file monitoring..."
    
    # Monitor directory with inotifywait (Linux)
    inotifywait -m -r -e create,modify,moved_to --format "%w%f" "$WATCH_DIR" | 
    while read FILE; do
        if [[ "$FILE" == *.md ]]; then
            echo "📝 New/modified: $FILE"
            
            # Wait a bit to allow file operations to complete
            sleep "$DEBOUNCE_TIME"
            commit_and_push
        fi
    done

elif command -v fswatch &> /dev/null; then
    echo "Using fswatch for file monitoring..."
    
    # Monitor directory with fswatch (macOS)
    fswatch -0 -e ".*" -i "\.md$" -r "$WATCH_DIR" | 
    while read -d "" FILE; do
        echo "📝 New/modified: $FILE"
        
        # Wait a bit to allow file operations to complete
        sleep "$DEBOUNCE_TIME"
        commit_and_push
    done

else
    echo "Warning: Neither inotifywait nor fswatch is installed."
    echo "Using a simple polling method instead."
    echo ""
    echo "For better performance, consider installing:"
    echo "  - On Debian/Ubuntu: sudo apt-get install inotify-tools"
    echo "  - On Fedora: sudo dnf install inotify-tools"
    echo "  - On macOS with Homebrew: brew install fswatch"
    
    # Simple polling method as fallback
    while true; do
        # Find recently modified markdown files (in the last 10 seconds)
        CHANGED=$(git status --porcelain | grep '\.md$')
        
        if [ -n "$CHANGED" ]; then
            echo "📝 Changes detected in markdown files"
            commit_and_push
        fi
        
        sleep 5
    done
fi