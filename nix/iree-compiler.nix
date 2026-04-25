# nix/iree-compiler.nix
#
# Builds the IREE compiler using the BYO-LLVM path (-DIREE_BUILD_BUNDLED_LLVM=OFF).
# Depends on iree-llvm.nix for LLVM+Clang+LLD+MLIR.
#
# Required submodules are injected as separate flake inputs and linked into the
# source tree before the build, keeping the derivation hermetic.
#
# Output layout:
#   $out/lib/libIREECompiler.so   - stable C embedding API
#   $out/include/iree/compiler/   - embedding_api.h, loader.h, mlir_interop.h
#   $out/include/mlir-c/          - MLIR C API headers (re-exported by IREE)
#   $out/bin/iree-compile         - compiler CLI tool (when withCli=true)
#
# ## What is built
#
# - Compiler only (IREE_BUILD_COMPILER=ON).
# - StableHLO input dialect (IREE_INPUT_STABLEHLO=ON).
# - Torch + TOSA input dialects disabled (no torch-mlir submodule needed).
# - CPU backend (IREE_TARGET_BACKEND_LLVM_CPU=ON).
# - CPU HAL drivers: local-sync and local-task.
# - No CUDA backend (IREE_TARGET_BACKEND_CUDA=OFF): avoids nvidia_sdk_download.
#   CUDA codegen can be layered on once the CPU path is stable.
# - Tests, samples, Python bindings: all OFF.
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
  # Flake source inputs.
  ## Main IREE repository (without submodules checked out).
  ireeSrc,
  ## IREE's stablehlo fork (iree-org/stablehlo).
  ireeStablehloSrc,
  ## flatcc library source (dvidelabs/flatcc).
  ireeFlatccSrc,
  ## google/benchmark (needed by IREE's threading runtime install target).
  ireeBenchmarkSrc,
  # Pre-built LLVM+Clang+LLD+MLIR from iree-llvm.nix.
  ireeLlvm,
  ## When true, build & install the compiler CLI tools (iree-compile, iree-opt,
  ##  iree-run-module) into $out/bin alongside the embedding API library.
  withCli ? false,
}:
stdenv.mkDerivation {
  pname = "iree-compiler";
  version = "iree-${ireeSrc.shortRev or "unknown"}" + lib.optionalString withCli "-cli";

  # We do our own source setup: copy ireeSrc and inject submodule sources.
  dontUnpack = true;
  dontConfigure = true;
  dontStrip = true;
  dontPatchELF = true;

  strictDeps = true;

  # System libs in both lists for the same reason as iree-llvm.nix: native
  # build tools compiled during cmake (iree-tblgen, etc.) need to find their
  # shared-library deps at runtime inside the Nix sandbox under strictDeps.
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

    # -----------------------------------------------------------------------
    # Source tree setup: copy IREE + inject required submodule sources.
    # -----------------------------------------------------------------------
    log "Copying IREE source tree"
    cp -r ${ireeSrc} iree-src
    chmod -R u+w iree-src

    # Inject required submodules.  cmake expects them at third_party/<name>/.
    # Remove the empty placeholder dirs that the IREE git tree has for each
    # submodule; if they exist, cp -r puts the source *inside* them instead
    # of replacing them, and cmake's CMakeLists.txt existence check fails.
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

    # -----------------------------------------------------------------------
    # CMake configure.
    # -----------------------------------------------------------------------
    log "Configuring IREE"
    mkdir -p iree-build

    cmake -S iree-src -B iree-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
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
      -DCMAKE_INSTALL_PREFIX="$out"

    # -----------------------------------------------------------------------
    # Build the compiler shared library (and optionally tools).
    # -----------------------------------------------------------------------
    log "Building libIREECompiler.so"
    ninja -C iree-build iree_compiler_API_SharedImpl

    ${lib.optionalString withCli ''
      log "Building compiler tools"
      ninja -C iree-build iree-compile iree-opt iree-run-module
    ''}
  '';

  installPhase = ''
    set -euo pipefail
    log() { echo "[iree-compiler] $*" >&2; }

    mkdir -p "$out/lib" "$out/include"

    # -----------------------------------------------------------------------
    # Library: install libIREECompiler.so.
    # -----------------------------------------------------------------------
    log "Installing libIREECompiler.so"
    find iree-build -name "libIREECompiler*.so*" -print -exec cp -v {} "$out/lib/" \;

    # -----------------------------------------------------------------------
    # Headers: IREE C embedding API + MLIR-C headers re-exported by IREE.
    # -----------------------------------------------------------------------
    log "Installing IREE C API headers"
    mkdir -p "$out/include/iree/compiler"
    cp -v iree-src/compiler/bindings/c/iree/compiler/*.h "$out/include/iree/compiler/"

    log "Installing MLIR-C headers (re-exported by IREE)"
    if [ -d "${ireeLlvm}/include/mlir-c" ]; then
      cp -r "${ireeLlvm}/include/mlir-c" "$out/include/mlir-c"
    fi

    # -----------------------------------------------------------------------
    # Tools (only when withCli=true).
    # -----------------------------------------------------------------------
    ${lib.optionalString withCli ''
      log "Installing compiler tools"
      mkdir -p "$out/bin"
      for tool in iree-compile iree-opt iree-run-module; do
        bin="$(find iree-build/tools -name "$tool" -type f 2>/dev/null | head -1)"
        if [ -n "$bin" ]; then
          cp -v "$bin" "$out/bin/$tool"
        fi
      done
    ''}

    # -----------------------------------------------------------------------
    # Patch RUNPATH on all installed DSOs and binaries.
    # -----------------------------------------------------------------------
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

    ${lib.optionalString withCli ''
      for f in "$out/bin/"*; do
        [ -f "$f" ] && [ -x "$f" ] || continue
        patchelf --set-rpath "$bin_rpath" "$f" 2>/dev/null || true
      done
    ''}

    # -----------------------------------------------------------------------
    # Sanity check: verify the embedding API symbol is exported.
    # -----------------------------------------------------------------------
    lib_path="$out/lib/libIREECompiler.so"
    if [ -f "$lib_path" ]; then
      if nm -D "$lib_path" 2>/dev/null | grep -q "ireeCompilerGetAPIVersion"; then
        log "OK: ireeCompilerGetAPIVersion found in libIREECompiler.so"
      else
        log "WARNING: ireeCompilerGetAPIVersion not found -- check symbol export configuration"
      fi
    else
      log "ERROR: libIREECompiler.so was not produced"
      exit 1
    fi

    log "Installation complete"
  '';

  meta = {
    description = "IREE compiler shared library (libIREECompiler.so) with StableHLO + CPU backend";
    license = lib.licenses.asl20;
  };
}
