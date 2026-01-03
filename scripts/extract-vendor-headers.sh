#!/usr/bin/env bash
# Extract MLIR and StableHLO C headers from bazel build
set -euo pipefail

# Portable util to strip common leading whitespace.
# Infers the prefix to strip from first line bc calculating
#   the actual shared common prefix is more complicated and
#   would need an awk script or similar.
#
# Usage:
# ```sh
# if true; then
#   dedent <<EOF
#         This is a multiline string.
#         It has a common 8-space prefix.
#         The 'dedent' function will strip it.
#         This is a list:
#           1. Item 1
#           2. Item 2
#         Now 4 spaces:
#             1. Item 1
#             2. Item 2
# EOF
# fi
# ```
#
# Redirect output:
#
# ```sh
#
# if true; then
#   # This sends the processed (dedented) string to stderr
#   dedent <<EOF >&2
#         This is a multiline string.
#         It will go to stderr
# EOF
# fi
# ```
#
# Redirect multiple statements by grouping in a block:
# ```sh
# if true; then
#   {
#     dedent <<EOF
#         This is a multiline string...
# EOF
#     echo "Another error message"
#   } >&2
# fi
# ```
dedent() {
  # read input into a var
  local input
  input=$(cat)

  # Identify the exact leading whitespace of the first non-empty line
  local indent
  indent=$(printf "%s" "$input" | sed -n '/[^[:space:]]/ {s/^\([[:space:]]*\).*/\1/p; q;}')

  # If indentation exists, strip it from every line
  if [ -n "$indent" ]; then
    printf "%s" "$input" | sed "s/^$indent//"
  else
    printf "%s" "$input"
  fi
}

ZML_DIR="reference/zml"
VENDOR_DIR="vendor"

main() {
  # Check if ZML exists
  if [[ ! -d "$ZML_DIR" ]]; then
    echo "Error: $ZML_DIR not found"
    echo "Clone ZML first: git clone https://github.com/zml/zml.git reference/zml"
    exit 1
  fi

  # Check if ZML is built
  if [[ ! -d "$ZML_DIR/bazel-zml/external/+llvm+llvm-project" ]]; then
    dedent <<EOF >&2
      Error: ZML bazel build not found. Please build ZML first.

      In zigrad root:
      uv init --bare --python 3.10 && uv sync && . .venv/bin/activate
      cd $ZML_DIR
      ./bazel.sh build --config=release --@zml//runtimes:cuda=true --@zml//runtimes:cpu=false //examples/mnist
EOF
    exit 1
  fi

  echo "Extracting vendor headers from ZML's bazel build..."

  # Extract MLIR C headers
  echo "  - MLIR C headers..."
  mkdir -p "$VENDOR_DIR/mlir-c"
  cp -r "$ZML_DIR/bazel-zml/external/+llvm+llvm-project/mlir/include/mlir-c/"* "$VENDOR_DIR/mlir-c/"

  # Extract StableHLO C headers
  echo "  - StableHLO C headers..."
  mkdir -p "$VENDOR_DIR/stablehlo-c"
  cp -r "$ZML_DIR/bazel-zml/external/+xla+stablehlo/stablehlo/integrations/c/"* "$VENDOR_DIR/stablehlo-c/"

  echo "✓ Vendor headers extracted to $VENDOR_DIR/"
  echo ""
  echo "Headers extracted:"
  echo "  $(find $VENDOR_DIR/mlir-c -name '*.h' | wc -l) MLIR C headers"
  echo "  $(find $VENDOR_DIR/stablehlo-c -name '*.h' | wc -l) StableHLO C headers"
}

main
