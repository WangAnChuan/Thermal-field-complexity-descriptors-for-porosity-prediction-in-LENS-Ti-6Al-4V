#!/usr/bin/env bash
# Download the Zamiela et al. (2023) thermal-porosity dataset from Harvard Dataverse.
# The dataset is not redistributed with this repository.
set -euo pipefail
cd "$(dirname "$0")"

ZIP_ID=7172366
XLSX_ID=7172365
BASE=https://dataverse.harvard.edu/api/access/datafile

if [ ! -f thermal-porosity-table.xlsx ]; then
  echo "Downloading label table..."
  curl -L --fail -o thermal-porosity-table.xlsx "$BASE/$XLSX_ID"
fi

if [ ! -f cropped_thinwall_CSV_pyrometer.zip ]; then
  echo "Downloading pyrometer frames (52 MB)..."
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x16 -s16 -k1M --continue=true -o cropped_thinwall_CSV_pyrometer.zip "$BASE/$ZIP_ID"
  else
    curl -L --fail -o cropped_thinwall_CSV_pyrometer.zip "$BASE/$ZIP_ID"
  fi
fi

if [ ! -d cropped_thinwall_CSV_pyrometer ]; then
  unzip -q cropped_thinwall_CSV_pyrometer.zip
fi

echo "Done. Now run: python experiments/01_build_cache.py"
echo "Source: https://doi.org/10.7910/DVN/BWHYEH"
