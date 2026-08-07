# Builds the IREE compiler CLI against the separately packaged IREE LLVM fork.
#
# StableHLO input, CPU and VMVX targets, and local CPU drivers are enabled.
# Required submodules are explicit source inputs.
{
  lib,
  stdenv,
  cmake,
  ninja,
  python3,
  patchelf,
  zlib,
  zstd,
  libxml2,
  ncurses,
  libffi,
  lld,
  binutils,
  ireeSrc,
  ireeRevision,
  ireeStablehloSrc,
  ireeFlatccSrc,
  ireeBenchmarkSrc,
  ireeLlvm,
  # Retain debug information in a RelWithDebInfo build.
  withDebugSymbols ? false,
  withNativeTuning ? false,
  enableLto ? false,
  extraCxxFlags ? [],
  extraLdFlags ? [],
}:
stdenv.mkDerivation {
  pname = "iree-compiler";
  version = "iree-${builtins.substring 0 7 ireeRevision}";

  # Source assembly happens in `buildPhase` because each submodule is pinned.
  dontUnpack = true;
  dontConfigure = true;
  dontStrip = withDebugSymbols;
  dontPatchELF = true;

  strictDeps = true;

  # Native IREE build tools load these libraries inside the strict-deps sandbox.
  nativeBuildInputs = [
    cmake
    ninja
    python3
    patchelf
    lld
    binutils
    stdenv.cc.cc.lib
    zlib
    zstd
    libxml2
    ncurses
    libffi
  ];

  buildInputs = [
    zlib
    zstd
    libxml2
    ncurses
    libffi
  ];

  buildPhase = ''
    set -euo pipefail
    log() { echo "[iree-compiler] $*" >&2; }

    log "Copying IREE source tree"
    cp -r ${ireeSrc} iree-src
    chmod -R u+w iree-src

    # Replace upstream placeholders with the pinned submodule sources.
    log "Injecting submodules"
    rm -rf iree-src/third_party/stablehlo
    cp -r ${ireeStablehloSrc} iree-src/third_party/stablehlo
    chmod -R u+w iree-src/third_party/stablehlo

    rm -rf iree-src/third_party/flatcc
    cp -r ${ireeFlatccSrc} iree-src/third_party/flatcc
    chmod -R u+w iree-src/third_party/flatcc

    rm -rf iree-src/third_party/benchmark
    cp -r ${ireeBenchmarkSrc} iree-src/third_party/benchmark
    chmod -R u+w iree-src/third_party/benchmark

    log "Configuring IREE"
    mkdir -p iree-build

    cmake -S iree-src -B iree-build -G Ninja \
      -DCMAKE_BUILD_TYPE=${
      if withDebugSymbols
      then "RelWithDebInfo"
      else "Release"
    } \
      \
      -DIREE_BUILD_BUNDLED_LLVM=OFF \
      -DLLVM_DIR="${ireeLlvm}/lib/cmake/llvm" \
      -DMLIR_DIR="${ireeLlvm}/lib/cmake/mlir" \
      -DLLD_DIR="${ireeLlvm}/lib/cmake/lld" \
      -DClang_DIR="${ireeLlvm}/lib/cmake/clang" \
      \
      -DIREE_BUILD_COMPILER=ON \
      -DIREE_BUILD_TESTS=OFF \
      -DIREE_BUILD_DOCS=OFF \
      -DIREE_BUILD_SAMPLES=OFF \
      -DIREE_BUILD_PYTHON_BINDINGS=OFF \
      -DIREE_BUILD_BINDINGS_TFLITE=OFF \
      -DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF \
      \
      -DIREE_INPUT_STABLEHLO=ON \
      -DIREE_INPUT_TORCH=OFF \
      -DIREE_INPUT_TOSA=OFF \
      \
      -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
      -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
      -DIREE_TARGET_BACKEND_VMVX=ON \
      \
      -DIREE_HAL_DRIVER_DEFAULTS=OFF \
      -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
      -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
      \
      -DIREE_TARGET_BACKEND_CUDA=OFF \
      -DIREE_HAL_DRIVER_CUDA=OFF \
      -DIREE_HAL_DRIVER_VULKAN=OFF \
      -DIREE_HAL_DRIVER_METAL=OFF \
      \
      -DIREE_ENABLE_CPUINFO=OFF \
      -DIREE_ENABLE_LIBBACKTRACE=OFF \
      \
      -DCMAKE_INSTALL_PREFIX="$out" \
      ${lib.optionalString enableLto "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"} \
      ${let
      cxxFlags = (lib.optionals withNativeTuning ["-march=native" "-mtune=native"]) ++ extraCxxFlags;
    in
      lib.optionalString (cxxFlags != []) "-DCMAKE_CXX_FLAGS='${lib.concatStringsSep " " cxxFlags}'"} \
      ${lib.optionalString (extraLdFlags != []) "-DCMAKE_SHARED_LINKER_FLAGS='${lib.concatStringsSep " " extraLdFlags}'"}

    log "Building iree-compile"
    cmake --build iree-build \
      --parallel "$NIX_BUILD_CORES" \
      --target iree_compiler_API_SharedImpl iree-compile

    # Nix fixup hooks expect unset variables to expand without failure.
    set +u
  '';

  installPhase = ''
    set -euo pipefail
    log() { echo "[iree-compiler] $*" >&2; }

    mkdir -p "$out/bin" "$out/lib"

    log "Installing libIREECompiler.so"
    find iree-build -name "libIREECompiler*.so*" -print -exec cp -v {} "$out/lib/" \;

    log "Installing iree-compile"
    compiler_bin="$(find iree-build/tools -name iree-compile -type f 2>/dev/null | head -1)"
    if [ -z "$compiler_bin" ]; then
      log "ERROR: iree-compile was not produced"
      exit 1
    fi
    cp -v "$compiler_bin" "$out/bin/iree-compile"

    log "Patching RUNPATH"
    chmod -R u+w "$out/lib"

    lib_rpath="\$ORIGIN:${ireeLlvm}/lib:${
      lib.makeLibraryPath [
        zlib
        zstd
        libxml2
        ncurses
        libffi
        stdenv.cc.cc.lib
      ]
    }"

    bin_rpath="\$ORIGIN/../lib:${ireeLlvm}/lib:${
      lib.makeLibraryPath [
        zlib
        zstd
        libxml2
        ncurses
        libffi
        stdenv.cc.cc.lib
      ]
    }"

    for f in "$out/lib/"*.so*; do
      [ -f "$f" ] || continue
      patchelf --set-rpath "$lib_rpath" "$f" || true
    done

    patchelf --set-rpath "$bin_rpath" "$out/bin/iree-compile"

    lib_path="$out/lib/libIREECompiler.so"
    if [ -f "$lib_path" ]; then
      if nm -D "$lib_path" 2>/dev/null | grep -q "ireeCompilerGetAPIVersion"; then
        log "OK: ireeCompilerGetAPIVersion found in libIREECompiler.so"
      else
        log "WARNING: ireeCompilerGetAPIVersion not found, check symbol export configuration"
      fi
    else
      log "ERROR: libIREECompiler.so was not produced"
      exit 1
    fi

    "$out/bin/iree-compile" --version

    log "Installation complete"

    # Nix fixup hooks expect unset variables to expand without failure.
    set +u
  '';

  meta = {
    description = "IREE compiler CLI with StableHLO and CPU targets";
    license = lib.licenses.asl20;
  };
}
