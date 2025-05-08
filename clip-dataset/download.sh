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
00177.tar
00178.tar
00179.tar
00180.tar
00181.tar
00182.tar
00183.tar
00184.tar
00185.tar
00186.tar
00187.tar
00188.tar
00189.tar
00190.tar
00191.tar
00192.tar
00193.tar
00194.tar
00195.tar
00196.tar
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