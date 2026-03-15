#!/usr/bin/env bash
set -euo pipefail

# Upload autodoc artifacts to the R2 bucket.
# Requires CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID env vars,
# or being authenticated via `wrangler login`.

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
autodoc_dir="${1:-$repo_root/zig-out/autodoc}"
bucket="zigrad-autodoc"

for file in main.js main.wasm sources.tar; do
  if [ ! -f "$autodoc_dir/$file" ]; then
    echo "missing: $autodoc_dir/$file" >&2
    exit 1
  fi
done

echo "uploading autodoc artifacts to R2 bucket '$bucket'..."

for file in main.js main.wasm sources.tar index.html zg-logo.svg; do
  if [ -f "$autodoc_dir/$file" ]; then
    echo "  $file"
    npx wrangler r2 object put "$bucket/$file" --file="$autodoc_dir/$file" --remote
  fi
done

echo "done."
