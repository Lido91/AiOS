#!/bin/bash
# Remove all files that are NOT person_0 from the specified directory
# Usage: bash cleanup_non_person0.sh <directory>

DIR="${1:?Usage: bash cleanup_non_person0.sh <directory>}"

if [ ! -d "$DIR" ]; then
    echo "Error: '$DIR' is not a directory"
    exit 1
fi

# Count files to be removed
count=$(find "$DIR" -maxdepth 1 -type f ! -name '*_person_0.*' | wc -l)
echo "Found $count files that are NOT person_0. Deleting..."

# Delete all files that don't match *_person_0.*
find "$DIR" -maxdepth 1 -type f ! -name '*_person_0.*' -delete

echo "Done. Remaining files:"
ls "$DIR" | head -20
