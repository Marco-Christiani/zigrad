{
  src,
  lib,
  stdenv,
  cmake,
  ninja,
  xlaMlirStablehloCapiSdk,
  llvm,
  doCheck ? true,
}:
stdenv.mkDerivation {
  pname = "zigrad-mlir-ext";
  version = xlaMlirStablehloCapiSdk.version;

  inherit src doCheck;

  outputs = ["out" "dev"];

  nativeBuildInputs = [cmake ninja];

  dontConfigure = true;

  buildPhase = ''
    set -eo pipefail

    cmake -S "$src" -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DZG_ENABLE_DEV_TARGETS=ON \
      -DZG_SDK_INCLUDE=${xlaMlirStablehloCapiSdk.dev}/include \
      -DZG_SDK_LIB=${xlaMlirStablehloCapiSdk.out}/lib \
      -DZG_LLVM_ROOT=${llvm} \
      -DZG_INSTALL_RPATH="${xlaMlirStablehloCapiSdk.out}/lib:${llvm}/lib"

    cmake --build build \
      --parallel "$NIX_BUILD_CORES" \
      --target zigrad_dev_tools
  '';

  checkPhase = ''
    set -eo pipefail

    MLIR_OPT="${llvm}/bin/mlir-opt"
    ZG_EXT="build/libzigrad_mlir_ext.so"
    PLUGIN="--load-dialect-plugin=$ZG_EXT --load-pass-plugin=$ZG_EXT"

    "$MLIR_OPT" $PLUGIN --show-dialects 2>&1 | grep -q zigrad

    failed=0
    for f in "$src"/test/select_*.mlir; do
      echo "=== $f ==="
      "$MLIR_OPT" $PLUGIN \
        --pass-pipeline="builtin.module(func.func(zg-mirage-kernel-select))" \
        "$f" || failed=1
    done
    for f in "$src"/test/legalize*.mlir "$src"/test/expand_*.mlir; do
      [ -f "$f" ] || continue
      pass=$(head -1 "$f" | sed -n 's|^// RUN-PIPELINE: ||p')
      [ -n "$pass" ] || continue
      echo "=== $f ==="
      "$MLIR_OPT" $PLUGIN --pass-pipeline="$pass" "$f" || failed=1
    done
    [ "$failed" -eq 0 ] || { echo "MLIR pass tests failed" >&2; exit 1; }
  '';

  installPhase = ''
    set -eo pipefail

    # out: production .so composed into demanding build configurations.
    mkdir -p "$out/lib"
    cp -v build/libzigrad_mlir_ext.so* "$out/lib/"

    # dev: LSP server binary + build metadata. Named mlir-lsp-server so editor
    #  configurations using the canonical name resolve via PATH.
    mkdir -p "$dev/bin" "$dev/share/build-metadata"
    cp -v build/mlir-lsp-server "$dev/bin/mlir-lsp-server"
    cp -v build/compile_commands.json "$dev/share/build-metadata/" || true
    cp -v build/CMakeCache.txt "$dev/share/build-metadata/" || true
  '';

  meta = {
    description = "Zigrad MLIR dialect extension (C++ DSO + LSP server)";
    license = lib.licenses.asl20;
    platforms = lib.platforms.linux;
  };
}
