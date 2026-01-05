{
  src,
  lib,
  stdenv,
  cmake,
  ninja,
  patchelf,
  xlaMlirStablehloCapiSdk,
  devel ? false,
}:
stdenv.mkDerivation {
  pname = "zigrad-mlir-shim";
  version = xlaMlirStablehloCapiSdk.version + lib.optionalString devel "-devel";

  inherit src;

  nativeBuildInputs = [
    cmake
    ninja
    patchelf
  ];

  dontConfigure = true;
  dontBuild = false;
  phases = [
    "unpackPhase"
    "patchPhase"
    "buildPhase"
    "installPhase"
  ];

  buildPhase = ''
    set -euo pipefail

    mkdir -p build
    cmake -S $src -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DZG_SDK_INCLUDE=${xlaMlirStablehloCapiSdk}/include \
      -DZG_SDK_LIB=${xlaMlirStablehloCapiSdk}/lib

    cmake --build build --target zigrad_mlir_ext
  '';

  installPhase = ''
    set -euo pipefail
    mkdir -p $out/lib $out/share

    so="$(ls -1 build/libzigrad_mlir_ext.so* 2>/dev/null | head -n1 || true)"
    if [ -z "$so" ]; then
      echo "ERROR: shim .so not built (expected build/libzigrad_mlir_ext.so*)" >&2
      find build -maxdepth 3 -type f -name "*.so*" -o -name "link.txt" -o -name "compile_commands.json" >&2 || true
      exit 1
    fi
    cp -v build/libzigrad_mlir_ext.so* $out/lib/

    # Make sure shim finds SDK libs at runtime. It will live in the same lib/ dir as MLIR libs via symlinkJoin.
    patchelf --set-rpath '$ORIGIN' $out/lib/libzigrad_mlir_ext.so || true

    if [ "${lib.boolToString devel}" = "true" ]; then
      mkdir -p $out/share/build-metadata
      cp -v build/compile_commands.json $out/share/build-metadata/ || true
      find build -name link.txt -print -exec cp -v {} $out/share/build-metadata/ \; || true
      cp -v build/CMakeCache.txt $out/share/build-metadata/ || true
    fi
  '';

  meta = {
    description = "Zigrad MLIR/StableHLO dialect registration shim (C++ DSO)";
    license = lib.licenses.asl20;
    platforms = lib.platforms.linux;
  };
}
