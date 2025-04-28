#!/bin/bash

# Check if user passed directory
if [ -z "$1" ]; then
  echo "Usage: bash download.sh /path/to/your/local-dir"
  exit 1
fi

TARGET_DIR="$1"

# List of files to download
FILES=(
00155.tar
00156.tar
00157.tar
00158.tar
00159.tar
00160.tar
00161.tar
00162.tar
00163.tar
00164.tar
00165.tar
00166.tar
00167.tar
00168.tar
00169.tar
00170.tar
00171.tar
00172.tar
00173.tar
00174.tar
00175.tar
00176.tar
)

TOTAL=${#FILES[@]}
COUNT=1

echo "Starting download to: $TARGET_DIR"
echo "Total files: $TOTAL"
echo "----------------------------------------"

for FILE in "${FILES[@]}"; do
  echo "[${COUNT}/${TOTAL}] Downloading $FILE..."
  huggingface-cli download "Spawning/pd12m-full" "$FILE" --repo-type dataset --local-dir "$TARGET_DIR"
  
  if [ $? -ne 0 ]; then
    echo "Failed to download $FILE"
  else
    echo "Successfully downloaded $FILE"
  fi
  
  echo "----------------------------------------"
  COUNT=$((COUNT + 1))
done

echo "All downloads completed!"
