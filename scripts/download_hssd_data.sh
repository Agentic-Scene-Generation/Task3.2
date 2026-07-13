#!/usr/bin/env bash
#
# Download HSSD preprocessed data required for asset retrieval.
# This script downloads:
# 1. CLIP indices and embeddings for semantic search (~60MB)
# 2. Pre-validated support surfaces from HSM (~2GB)
#
# Based on HSM's setup.sh but simplified for our needs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${SCENEEXPERT_ENV_FILE:-$PROJECT_ROOT/.env}"

source_env_file() {
    local env_path="$1"
    local tmp_env
    tmp_env="$(mktemp)"
    sed 's/\r$//' "$env_path" > "$tmp_env"
    # shellcheck disable=SC1090
    source "$tmp_env"
    rm -f "$tmp_env"
}

if [ -f "$ENV_FILE" ]; then
    source_env_file "$ENV_FILE"
    echo "Loaded config: $ENV_FILE"
fi

DATA_DIR="${SCENEEXPERT_HSSD_DATA_DIR:-${SCENEEXPERT_DATA_DIR:-$PROJECT_ROOT/data}}"

PREPROCESSED_DIR="$DATA_DIR/preprocessed"
HSSD_MODELS_DIR="$DATA_DIR/hssd-models"
SUPPORT_SURFACES_DIR="$HSSD_MODELS_DIR/support-surfaces"

echo "=========================================="
echo "HSSD Preprocessed Data Download"
echo "=========================================="
echo
echo "HSSD/HSM data directory: $DATA_DIR"
echo

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed. Please install it first."
        exit 1
    fi
}

extract_preprocessed() {
    local zip_path="$1"
    local tmp_dir
    tmp_dir="$(mktemp -d)"

    unzip -q "$zip_path" -d "$tmp_dir"

    if [ -d "$tmp_dir/data/preprocessed" ]; then
        mv "$tmp_dir/data/preprocessed" "$PREPROCESSED_DIR"
    elif [ -d "$tmp_dir/preprocessed" ]; then
        mv "$tmp_dir/preprocessed" "$PREPROCESSED_DIR"
    else
        echo "Error: could not find preprocessed directory inside $zip_path"
        echo "Archive contents:"
        find "$tmp_dir" -maxdepth 3 -type d | sort
        rm -rf "$tmp_dir"
        exit 1
    fi

    rm -rf "$tmp_dir"
}

check_command wget
check_command unzip

if ! mkdir -p "$DATA_DIR"; then
    echo "Error: cannot create HSSD/HSM data directory: $DATA_DIR"
    echo "Set SCENEEXPERT_HSSD_DATA_DIR to a writable staging path, or ask the data"
    echo "administrator to prepare this read-only shared directory."
    exit 1
fi

if [ ! -w "$DATA_DIR" ]; then
    echo "Error: HSSD/HSM data directory is not writable: $DATA_DIR"
    echo "This is expected on read-only cluster shared storage. Run this script on a"
    echo "writable data-build node/path, then mount/copy the completed directory."
    exit 1
fi

echo "Downloading preprocessed data (~60MB)..."
echo

PREPROCESSED_URL="https://github.com/3dlg-hcvc/hsm/releases/latest/download/data.zip"
PREPROCESSED_ZIP="$DATA_DIR/preprocessed_data.zip"

if [ -f "$PREPROCESSED_ZIP" ]; then
    echo "Preprocessed data archive already exists, skipping download."
else
    wget --no-verbose --show-progress "$PREPROCESSED_URL" -O "$PREPROCESSED_ZIP"
fi

echo
echo "Extracting preprocessed data..."

if [ -d "$PREPROCESSED_DIR" ]; then
    echo "Warning: $PREPROCESSED_DIR already exists."
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping extraction."
    else
        rm -rf "$PREPROCESSED_DIR"
        extract_preprocessed "$PREPROCESSED_ZIP"
    fi
else
    extract_preprocessed "$PREPROCESSED_ZIP"
fi

echo
echo "Cleaning up archive..."
rm "$PREPROCESSED_ZIP"

echo
echo "Downloading pre-validated support surfaces (~2GB)..."
echo

SUPPORT_SURFACES_URL="https://github.com/3dlg-hcvc/hsm/releases/latest/download/support-surfaces.zip"
SUPPORT_SURFACES_ZIP="$DATA_DIR/support_surfaces.zip"

if [ -f "$SUPPORT_SURFACES_ZIP" ]; then
    echo "Support surfaces archive already exists, skipping download."
else
    wget --no-verbose --show-progress "$SUPPORT_SURFACES_URL" -O "$SUPPORT_SURFACES_ZIP"
fi

echo
echo "Extracting support surfaces..."

mkdir -p "$HSSD_MODELS_DIR"

if [ -d "$SUPPORT_SURFACES_DIR" ]; then
    echo "Warning: $SUPPORT_SURFACES_DIR already exists."
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping extraction."
    else
        rm -rf "$SUPPORT_SURFACES_DIR"
        unzip -q "$SUPPORT_SURFACES_ZIP" -d "$HSSD_MODELS_DIR"
    fi
else
    unzip -q "$SUPPORT_SURFACES_ZIP" -d "$HSSD_MODELS_DIR"
fi

echo
echo "Cleaning up archive..."
rm "$SUPPORT_SURFACES_ZIP"

echo
echo "=========================================="
echo "Data downloaded successfully!"
echo
echo "Preprocessed data: $PREPROCESSED_DIR"
echo "Support surfaces: $SUPPORT_SURFACES_DIR"
echo
echo "Next steps:"
echo "1. Download HSSD models (~72GB):"
echo "   cd $DATA_DIR"
echo "   git lfs install"
echo "   git clone https://huggingface.co/datasets/hssd/hssd-models"
echo
echo "2. Enable HSSD in your config:"
echo "   furniture_agent.asset_manager.general_asset_source=hssd"
echo "   manipuland_agent.asset_manager.general_asset_source=hssd"
echo "=========================================="
