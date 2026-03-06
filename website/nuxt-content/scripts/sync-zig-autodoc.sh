#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <autodoc-dir>"
  echo "example: $0 /path/to/generated/api"
  exit 1
fi

src_dir="$1"
out_dir="$(cd "$(dirname "$0")/.." && pwd)/public/api"

for required in index.html main.js main.wasm sources.tar; do
  if [ ! -f "$src_dir/$required" ]; then
    echo "missing required file: $src_dir/$required"
    exit 1
  fi
done

mkdir -p "$out_dir"
cp "$src_dir/index.html" "$out_dir/index.html"
cp "$src_dir/main.js" "$out_dir/main.js"
cp "$src_dir/main.wasm" "$out_dir/main.wasm"
cp "$src_dir/sources.tar" "$out_dir/sources.tar"

if [ -f "$src_dir/zg-logo.svg" ]; then
  cp "$src_dir/zg-logo.svg" "$out_dir/zg-logo.svg"
fi

echo "synced Zig autodoc bundle to $out_dir"
