{
  src,
  lib,
  stdenv,
  cmake,
  ninja,
  patchelf,
  xlaMlirStablehloCapiSdk,
  llvm,
  devel ? false,
  doCheck ? true,
}:
stdenv.mkDerivation {
  pname = "zigrad-mlir-ext";
  version = xlaMlirStablehloCapiSdk.version + lib.optionalString devel "-devel";

  inherit src doCheck;

  nativeBuildInputs = [
    cmake
    ninja
    patchelf
  ];

  dontConfigure = true;
  dontBuild = false;
  phases =
    [
      "unpackPhase"
      "patchPhase"
      "buildPhase"
    ]
    ++ lib.optional doCheck "checkPhase"
    ++ [
      "installPhase"
    ];

  buildPhase = ''
    set -euo pipefail

    mkdir -p build
    cmake -S $src -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DZG_SDK_INCLUDE=${xlaMlirStablehloCapiSdk}/include \
      -DZG_SDK_LIB=${xlaMlirStablehloCapiSdk}/lib \
      -DZG_LLVM_ROOT=${llvm}

    cmake --build build --target zigrad_mlir_ext
  '';

  checkPhase = ''
    set -euo pipefail

    MLIR_OPT="${llvm}/bin/mlir-opt"
    ZG_EXT="build/libzigrad_mlir_ext.so"
    PLUGIN="--load-dialect-plugin=$ZG_EXT --load-pass-plugin=$ZG_EXT"

    # Verify plugin loads and exposes the zigrad dialect.
    $MLIR_OPT $PLUGIN --show-dialects 2>&1 | grep -q zigrad

    failed=0

    # Selection pass tests.
    for f in $src/test/select_*.mlir; do
      echo "=== $f ==="
      $MLIR_OPT $PLUGIN \
        --pass-pipeline="builtin.module(func.func(zg-mirage-kernel-select))" \
        "$f" || failed=1
    done

    # Legalize / expand tests (pipeline read from first line of each file).
    for f in $src/test/legalize*.mlir $src/test/expand_*.mlir; do
      [ -f "$f" ] || continue
      pass=$(head -1 "$f" | sed -n 's|^// RUN-PIPELINE: ||p')
      [ -n "$pass" ] || continue
      echo "=== $f ==="
      $MLIR_OPT $PLUGIN --pass-pipeline="$pass" "$f" || failed=1
    done

    if [ "$failed" -ne 0 ]; then
      echo "MLIR pass tests failed" >&2
      exit 1
    fi
  '';

  installPhase = ''
    set -euo pipefail
    mkdir -p $out/lib $out/share

    so="$(ls -1 build/libzigrad_mlir_ext.so* 2>/dev/null | head -n1 || true)"
    if [ -z "$so" ]; then
      echo "ERROR: libzigrad_mlir_ext.so not built" >&2
      find build -maxdepth 3 -type f -name "*.so*" -o -name "link.txt" -o -name "compile_commands.json" >&2 || true
      exit 1
    fi
    cp -v build/libzigrad_mlir_ext.so* $out/lib/

    # Finds SDK libs at runtime via symlinkJoin (same lib/ dir as MLIR libs).
    patchelf --set-rpath '$ORIGIN' $out/lib/libzigrad_mlir_ext.so || true

    if [ "${lib.boolToString devel}" = "true" ]; then
      mkdir -p $out/share/build-metadata
      cp -v build/compile_commands.json $out/share/build-metadata/ || true
      find build -name link.txt -print -exec cp -v {} $out/share/build-metadata/ \; || true
      cp -v build/CMakeCache.txt $out/share/build-metadata/ || true
    fi
  '';

  meta = {
    description = "Zigrad MLIR dialect extension (C++ DSO)";
    license = lib.licenses.asl20;
    platforms = lib.platforms.linux;
  };
}
