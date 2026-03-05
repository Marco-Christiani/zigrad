#!/usr/bin/env bash
set -euo pipefail

index_file="${1:-$(cd "$(dirname "$0")/.." && pwd)/public/api/index.html}"

if [ ! -f "$index_file" ]; then
  echo "autodoc branding patch failed: missing file $index_file" >&2
  exit 1
fi

logo_file="$(dirname "$index_file")/zg-logo.svg"
if [ ! -f "$logo_file" ]; then
  echo "autodoc branding patch failed: missing logo asset $logo_file" >&2
  exit 1
fi

tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT

cp "$index_file" "$tmp_file"

# inject brand logo
perl -0777 -i -pe '
  BEGIN { $n = 0 }
  $n += s{<a class="logo" href="#">\s*<svg\b.*?</svg>\s*</a>}{<a class="logo" href="#"><img src="zg-logo.svg" alt="Zigrad" class="zigrad-logo"></a>}sg;
  $n += s{<a class="logo" href="#">\s*<img src="zg-logo\.svg" alt="Zigrad" style="[^"]*">\s*</a>}{<a class="logo" href="#"><img src="zg-logo.svg" alt="Zigrad" class="zigrad-logo"></a>}sg;
  if ($n == 0 && /<a class="logo" href="#">\s*<img src="zg-logo\.svg" alt="Zigrad" class="zigrad-logo">\s*<\/a>/s) {
    $n = 1;
  }
  END {
    if ($n != 1) {
      print STDERR "autodoc branding patch failed: expected 1 logo replacement, got $n\\n";
      exit 1;
    }
  }
' "$tmp_file"

# tab title override
perl -0777 -i -pe 's{<title>Zig Documentation</title>}{<title>Zigrad API Documentation</title>}g' "$tmp_file"

mv "$tmp_file" "$index_file"

echo "patched autodoc branding in $index_file"
