#!/bin/bash -e
# Simple setup script for MMseqs2 mini debug database

# Folder where the database will be downloaded
WORKDIR="$(dirname "$0")/mini_mmseqs_db"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Check if mmseqs is installed
if ! command -v mmseqs >/dev/null 2>&1; then
    echo "Error: mmseqs command not found in PATH. Please install MMseqs2."
    exit 1
fi

# Check for a download tool
STRATEGY=""
if command -v aria2c >/dev/null 2>&1; then STRATEGY="ARIA"; fi
if command -v curl >/dev/null 2>&1; then STRATEGY="$STRATEGY CURL"; fi
if command -v wget >/dev/null 2>&1; then STRATEGY="$STRATEGY WGET"; fi
if [ -z "$STRATEGY" ]; then
    echo "Error: No download tool found. Install aria2c, curl, or wget."
    exit 1
fi

# Function to download files
downloadFile() {
    URL="$1"
    OUTPUT="$2"
    set +e
    for i in $STRATEGY; do
        case "$i" in
            ARIA)
                aria2c --max-connection-per-server=8 --allow-overwrite=true -o "$(basename $OUTPUT)" -d "$(dirname $OUTPUT)" "$URL" && set -e && return 0
                ;;
            CURL)
                curl -L -o "$OUTPUT" "$URL" && set -e && return 0
                ;;
            WGET)
                wget -O "$OUTPUT" "$URL" && set -e && return 0
                ;;
        esac
    done
    set -e
    echo "Error: Could not download $URL"
    exit 1
}

# Download mini SwissProt
downloadFile "https://opendata.mmseqs.org/colabfold/mini_swissprot2503.tar.gz" "mini_swissprot2503.tar.gz"

# Extract and rename to standard MMseqs2 database names
tar -xzvf "mini_swissprot2503.tar.gz"

# Move extracted files into MMseqs2 database structure
mmseqs mvdb sprot2503_h uniref30_mini_db_h
mmseqs mvdb sprot2503 uniref30_mini_db
mmseqs mvdb sprot2503_aln uniref30_mini_db_aln
mmseqs mvdb sprot2503_seq_h uniref30_mini_db_seq_h
mmseqs mvdb sprot2503_seq uniref30_mini_db_seq
mv -f sprot2503_taxonomy uniref30_mini_db_taxonomy
mv -f sprot2503_mapping uniref30_mini_db_mapping

echo "Mini debug database setup complete in $WORKDIR"
