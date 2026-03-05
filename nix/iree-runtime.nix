# nix/iree-runtime.nix
#
# Builds the IREE runtime static archives using the BYO-LLVM path.
# Depends on iree-llvm.nix for LLVM+Clang+LLD+MLIR.
#
# Required submodules are injected as separate flake inputs and linked into the
# source tree before the build, keeping the derivation hermetic.
#
# Output layout:
#   $out/lib/libiree_runtime_unified.a  - unified runtime archive (base + hal + vm + local drivers)
#   $out/lib/libflatcc_*.a              - flatcc archives (IREE's FlatBuffer dependency)
#   $out/include/iree/                  - C API headers (base, hal, vm, runtime, task, io, ...)
#
# ## What is built
#
# - Runtime only (IREE_BUILD_COMPILER=OFF - no compiler, faster build).
# - CPU HAL drivers: local-sync and local-task.
# - No GPU backends (CUDA/Vulkan/Metal all OFF).
# - Tests, samples, Python bindings: all OFF.
#
# Static archives are installed directly for static linking at build time.
# The IREE compiler (libIREECompiler.so) is loaded via dlopen and is NOT
# part of this derivation.
{
  lib,
  stdenv,
  cmake,
  ninja,
  python3,
  zlib,
  zstd,
  libxml2,
  ncurses,
  libffi,
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
}:
stdenv.mkDerivation {
  pname = "iree-runtime";
  version = "iree-${ireeSrc.shortRev or "unknown"}";

  # We do our own source setup: copy ireeSrc and inject submodule sources.
  dontUnpack = true;
  dontConfigure = true;
  dontStrip = true;

  strictDeps = true;

  # Both nativeBuildInputs and buildInputs for the same reason as iree-compiler.nix:
  # native build tools compiled during cmake (internal code generators) need
  # their shared-library deps at runtime inside the Nix sandbox under strictDeps.
  nativeBuildInputs = [
    cmake
    ninja
    python3
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
    log() { echo "[iree-runtime] $*" >&2; }

    # -----------------------------------------------------------------------
    # Source tree setup: copy IREE + inject required submodule sources.
    # -----------------------------------------------------------------------
    log "Copying IREE source tree"
    cp -r ${ireeSrc} iree-src
    chmod -R u+w iree-src

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
    # CMake configure - runtime only (no compiler).
    # -----------------------------------------------------------------------
    log "Configuring IREE (runtime-only build)"
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
      -DIREE_BUILD_COMPILER=OFF \
      -DIREE_BUILD_TESTS=OFF \
      -DIREE_BUILD_DOCS=OFF \
      -DIREE_BUILD_SAMPLES=OFF \
      -DIREE_BUILD_PYTHON_BINDINGS=OFF \
      -DIREE_BUILD_BINDINGS_TFLITE=OFF \
      -DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF \
      \
      -DIREE_HAL_DRIVER_DEFAULTS=OFF \
      -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
      -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
      \
      -DIREE_HAL_DRIVER_CUDA=OFF \
      -DIREE_HAL_DRIVER_VULKAN=OFF \
      -DIREE_HAL_DRIVER_METAL=OFF \
      \
      -DIREE_ENABLE_CPUINFO=OFF \
      -DIREE_ENABLE_LIBBACKTRACE=OFF \
      \
      -DCMAKE_INSTALL_PREFIX="$out"

    # -----------------------------------------------------------------------
    # Build runtime static archives.
    # -----------------------------------------------------------------------
    log "Building IREE runtime (may take a while)"

    # Try known target names; IREE cmake naming is <module>_<target> with dots
    # replaced by underscores.  Fall back to ninja with no targets (build all
    # non-compiler targets) if the specific targets are unavailable.
    if ninja -C iree-build iree_runtime_runtime; then
      log "Built via target: iree_runtime_runtime"
    elif ninja -C iree-build iree_base_base iree_hal_hal iree_vm_vm; then
      log "Built via targets: iree_base_base iree_hal_hal iree_vm_vm"
    else
      log "WARNING: specific targets unavailable, building all non-compiler targets"
      ninja -C iree-build
    fi

    log "Build phase complete"
  '';

  installPhase = ''
        set -euo pipefail
        log() { echo "[iree-runtime] $*" >&2; }

        mkdir -p "$out/lib" "$out/include"

        # -----------------------------------------------------------------------
        # Install static archives directly.
        # -----------------------------------------------------------------------
        log "Installing static archives"

        unified=$(find iree-build -name "libiree_runtime_unified.a" | head -1)

        if [ -z "$unified" ]; then
          log "ERROR: libiree_runtime_unified.a not found - cmake build may have failed"
          exit 1
        fi

        log "Installing unified archive: $unified"
        cp "$unified" "$out/lib/libiree_runtime_unified.a"

        # flatcc archives (IREE's FlatBuffer dependency).
        find iree-build -name "libflatcc*.a" ! -name "*test*" \
          -exec cp {} "$out/lib/" \;

        # -----------------------------------------------------------------------
        # Install runtime headers from source tree.
        # -----------------------------------------------------------------------
        log "Installing runtime headers"

        # Primary location: IREE's runtime sources live under runtime/src/.
        if [ -d iree-src/runtime/src/iree ]; then
          cp -r iree-src/runtime/src/iree/. "$out/include/iree/"
        fi

        # Older or alternate layout: headers might be directly under iree-src.
        if [ -d iree-src/iree ] && [ ! -d "$out/include/iree/base" ]; then
          cp -r iree-src/iree/. "$out/include/iree/"
        fi

        # Merge cmake-generated headers (e.g. iree/base/config.h) without
        # overwriting source-tree headers.
        find iree-build \
          \( -path "*/iree/base/*.h" \
            -o -path "*/iree/hal/*.h" \
            -o -path "*/iree/vm/*.h" \
            -o -path "*/iree/runtime/*.h" \
            -o -path "*/iree/task/*.h" \
          \) \
          ! -path "*/compiler/*" \
        | while IFS= read -r src; do
            # Compute destination relative to any "src/iree" or "iree-build" prefix.
            rel="$(echo "$src" | sed 's|.*iree-build[^/]*/||;s|^runtime/src/||')"
            dest="$out/include/$rel"
            if [ ! -f "$dest" ]; then
              mkdir -p "$(dirname "$dest")"
              cp "$src" "$dest"
            fi
          done

        # -----------------------------------------------------------------------
        # Sanity checks.
        # -----------------------------------------------------------------------
        if nm "$out/lib/libiree_runtime_unified.a" 2>/dev/null | grep -q 'T iree_runtime_instance_create'; then
          log "OK: iree_runtime_instance_create found in archive"
        else
          log "WARNING: iree_runtime_instance_create not found"
        fi

        if [ -f "$out/include/iree/runtime/api.h" ]; then
          log "OK: iree/runtime/api.h installed"
        else
          log "WARNING: iree/runtime/api.h not found - @cImport will fail at build time"
        fi

        log "Installation complete"
  '';

  meta = {
    description = "IREE runtime static archives with CPU HAL drivers (local-sync + local-task)";
    license = lib.licenses.asl20;
  };
}
