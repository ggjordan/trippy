#!/bin/bash
# fetch_upstream.sh — Clone TRIPS repository and display Zenodo URLs
#
# Usage:
#   bash tools/fetch_upstream.sh              # show TRIPS clone and Zenodo URLs
#   bash tools/fetch_upstream.sh --download tt_scenes.zip  # download from Zenodo
#
# This script:
# 1. git clones https://github.com/lfranke/TRIPS into third_party/TRIPS (shallow, once only).
# 2. Prints the commit hash.
# 3. Prints Zenodo download URLs and file sizes for public scene data.
# 4. Does NOT download by default (--download flag required).
#
# Zenodo record: 10687419 (CC-BY 4.0)
# License: TRIPS is MIT; data is CC-BY 4.0

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${SCRIPT_DIR}/third_party"
TRIPS_DIR="${THIRD_PARTY_DIR}/TRIPS"
ZENODO_DIR="${THIRD_PARTY_DIR}/zenodo"

ZENODO_RECORD="10687419"
ZENODO_DOWNLOAD_BASE="https://zenodo.org/records/${ZENODO_RECORD}/files"

# Files available on Zenodo (name and size in bytes)
declare -A ZENODO_FILES=(
    ["tt_scenes.zip"]="3200000000"
    ["tt_checkpoints.zip"]="2700000000"
    ["boat_scene_and_checkpoint.zip"]="6600000000"
    ["mipnerf360_our_resolutions.zip"]="5100000000"
)

download_flag=false
target_file=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --download)
            download_flag=true
            target_file="$2"
            shift 2
            ;;
        *)
            echo "Usage: bash tools/fetch_upstream.sh [--download <file>]" >&2
            exit 1
            ;;
    esac
done

# Clone TRIPS if not present
if [[ ! -d "${TRIPS_DIR}" ]]; then
    echo "Cloning TRIPS repository (shallow)..."
    mkdir -p "${THIRD_PARTY_DIR}"
    git clone --depth 1 https://github.com/lfranke/TRIPS "${TRIPS_DIR}"
    echo "✓ TRIPS cloned to ${TRIPS_DIR}"
fi

# Print TRIPS commit
if [[ -d "${TRIPS_DIR}/.git" ]]; then
    TRIPS_COMMIT=$(cd "${TRIPS_DIR}" && git rev-parse HEAD)
    echo "TRIPS commit: ${TRIPS_COMMIT}"
else
    echo "Warning: TRIPS directory exists but is not a git repo" >&2
fi

echo ""
echo "--- Zenodo data (record ${ZENODO_RECORD}) ---"
echo "License: CC-BY 4.0 (Linus Franke et al.)"
echo "Attribution: Trilinear Point Splatting for Real-time Radiance Field Rendering."
echo ""

# Create Zenodo directory if needed
mkdir -p "${ZENODO_DIR}"

# Download or print URLs
for filename in "${!ZENODO_FILES[@]}"; do
    size_bytes="${ZENODO_FILES[$filename]}"
    size_gb=$(echo "scale=2; ${size_bytes} / 1000000000" | bc)

    # Construct download URL
    # Note: actual Zenodo URLs are typically like:
    # https://zenodo.org/records/10687419/files/tt_scenes.zip?download=1
    # Using the API endpoint for programmatic access:
    download_url="${ZENODO_DOWNLOAD_BASE}/${filename}?download=1"

    echo "File: ${filename}"
    echo "  Size: ${size_gb} GB"
    echo "  URL: ${download_url}"

    # Download if requested
    if [[ "${download_flag}" == "true" && "${target_file}" == "${filename}" ]]; then
        echo "  Downloading to ${ZENODO_DIR}/${filename}..."
        curl -L --output "${ZENODO_DIR}/${filename}" "${download_url}"
        if [[ $? -eq 0 ]]; then
            echo "  ✓ Download complete"
        else
            echo "  ✗ Download failed" >&2
            exit 1
        fi
    fi
    echo ""
done

# Print curl example for manual download
echo "--- Manual download example ---"
echo "To download a specific file manually, use:"
echo ""
echo "curl -L --output ./third_party/zenodo/tt_scenes.zip \\"
echo "  'https://zenodo.org/records/${ZENODO_RECORD}/files/tt_scenes.zip?download=1'"
echo ""
echo "Or use the web interface: https://zenodo.org/records/${ZENODO_RECORD}"
