#!/usr/bin/env bash
# Extract MLIR and StableHLO C headers from ZML's bazel build
set -euo pipefail

ZML_DIR="reference/zml"
VENDOR_DIR="vendor"

# Check if ZML exists
if [[ ! -d "$ZML_DIR" ]]; then
    echo "Error: $ZML_DIR not found"
    echo "Clone ZML first: git clone https://github.com/zml/zml.git reference/zml"
    exit 1
fi

# Check if ZML is built
if [[ ! -d "$ZML_DIR/bazel-zml/external/+llvm+llvm-project" ]]; then
    echo "Error: ZML bazel build not found"
    echo "Build ZML first:"
    echo "  cd $ZML_DIR"
    echo "  ./bazel.sh build //..."
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
