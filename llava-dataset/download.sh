#!/bin/bash

# Check if user passed directory
if [ -z "$1" ]; then
  echo "Usage: bash download.sh /path/to/your/local-dir"
  exit 1
fi

TARGET_DIR="$1"

# List of files to download
FILES=(
blip_laion_cc_sbu_558k.json
blip_laion_cc_sbu_558k_meta.json
images.zip
)

TOTAL=${#FILES[@]}
COUNT=1

echo "Starting download to: $TARGET_DIR"
echo "Total files: $TOTAL"
echo "----------------------------------------"

for FILE in "${FILES[@]}"; do
  echo "[${COUNT}/${TOTAL}] Downloading $FILE..."
  huggingface-cli download "liuhaotian/LLaVA-Pretrain" "$FILE" --repo-type dataset --local-dir "$TARGET_DIR"
  
  if [ $? -ne 0 ]; then
    echo "Failed to download $FILE"
  else
    echo "Successfully downloaded $FILE"
  fi
  
  echo "----------------------------------------"
  COUNT=$((COUNT + 1))
done

# Unzip the images.zip file if it exists
if [ -f "$TARGET_DIR/images.zip" ]; then
  echo "Unzipping images.zip to $TARGET_DIR/images..."
  mkdir -p "$TARGET_DIR/images"
  unzip "$TARGET_DIR/images.zip" -d "$TARGET_DIR/images"
  
  if [ $? -ne 0 ]; then
    echo "Failed to unzip images.zip"
  else
    echo "Successfully unzipped images.zip"
    
    # Remove the zip file after extraction
    echo "Removing images.zip file..."
    rm "$TARGET_DIR/images.zip"
    
    if [ $? -ne 0 ]; then
      echo "Failed to remove images.zip"
    else
      echo "Successfully removed images.zip"
    fi
  fi
  echo "----------------------------------------"
fi

echo "All downloads completed!"