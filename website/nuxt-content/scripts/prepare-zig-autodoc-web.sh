#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
autodoc_dir="${1:-$repo_root/zig-out/autodoc}"

if [ -z "${ZG_EXTERNAL_SDK_ROOT:-}" ]; then
  echo "prepare-zig-autodoc-web failed: ZG_EXTERNAL_SDK_ROOT is not set" >&2
  exit 1
fi

echo "building Zig autodocs..."
zig build -Dsdk="$ZG_EXTERNAL_SDK_ROOT" docs

echo "syncing autodoc bundle..."
"$script_dir/sync-zig-autodoc.sh" "$autodoc_dir"

echo "applying autodoc branding..."
"$script_dir/brand-zig-autodoc.sh"

echo "prepared autodoc bundle for website at $repo_root/website/nuxt-content/public/api"
