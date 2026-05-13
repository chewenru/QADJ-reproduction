#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/3rdparty"
SC2PATH="$THIRD_PARTY/StarCraftII"
SC2_VERSION="4.6.2.69232"
SC2_BUILD="Base69232"
MAP_ARCHIVE_URL="https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip"
SC2_ARCHIVE_URL="http://blzdistsc2-a.akamaihd.net/Linux/SC2.${SC2_VERSION}.zip"

mkdir -p "$THIRD_PARTY"
cd "$THIRD_PARTY"

export SC2PATH
echo "SC2PATH=$SC2PATH"

if [ -d "$SC2PATH" ] && [ ! -d "$SC2PATH/Versions/$SC2_BUILD" ]; then
  CURRENT_BUILD="$(find "$SC2PATH/Versions" -maxdepth 1 -type d -name 'Base*' -printf '%f\n' | head -n 1 || true)"
  BACKUP_PATH="${THIRD_PARTY}/StarCraftII_backup_${CURRENT_BUILD:-unknown}_$(date +%Y%m%d_%H%M%S)"
  echo "Existing StarCraft II install is not ${SC2_VERSION} (${CURRENT_BUILD:-unknown})."
  echo "Moving current install to $BACKUP_PATH"
  mv "$SC2PATH" "$BACKUP_PATH"
fi

if [ ! -d "$SC2PATH/Versions/$SC2_BUILD" ]; then
  echo "Downloading StarCraft II ${SC2_VERSION} ..."
  wget -O "SC2.${SC2_VERSION}.zip" "$SC2_ARCHIVE_URL"
  unzip -oq -P iagreetotheeula "SC2.${SC2_VERSION}.zip"
  rm -f "SC2.${SC2_VERSION}.zip"
else
  echo "StarCraft II ${SC2_VERSION} already present."
fi

mkdir -p "$SC2PATH/Maps"

if [ ! -d "$SC2PATH/Maps/SMAC_Maps" ]; then
  echo "Downloading SMAC maps ..."
  wget -O SMAC_Maps.zip "$MAP_ARCHIVE_URL"
  rm -rf SMAC_Maps __MACOSX
  unzip -oq SMAC_Maps.zip
  mv SMAC_Maps "$SC2PATH/Maps/"
  rm -rf __MACOSX
  rm -f SMAC_Maps.zip
else
  echo "SMAC maps already present."
fi

echo "SC2 resources ready under $SC2PATH"
