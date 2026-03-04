# nix/iree-runtime.nix
#
# Builds the IREE runtime shared library (libIREERuntime.so) using the BYO-LLVM path.
# Depends on iree-llvm.nix for LLVM+Clang+LLD+MLIR.
#
# Required submodules are injected as separate flake inputs and linked into the
# source tree before the build, keeping the derivation hermetic.
#
# Output layout:
#   $out/lib/libIREERuntime.so    - combined runtime DSO (base + hal + vm + local drivers)
#   $out/include/iree/            - C API headers (base, hal, vm, runtime, task, io, ...)
#
# ## What is built
#
# - Runtime only (IREE_BUILD_COMPILER=OFF - no compiler, faster build).
# - CPU HAL drivers: local-sync and local-task.
# - No GPU backends (CUDA/Vulkan/Metal all OFF).
# - Tests, samples, Python bindings: all OFF.
#
# ## libIREERuntime.so construction
#
# IREE cmake does not produce a standard combined runtime DSO.  After building
# the static archives for the runtime, hal, vm, and base subsystems, this
# derivation links them together using --whole-archive to export all symbols.
# The rpath is set to $ORIGIN and the ireeLlvm lib directory.
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
        # Collect runtime static archives.
        # Exclude compiler, tools, test, and benchmark artifacts.
        # -----------------------------------------------------------------------
        log "Collecting runtime static archives"

        # Prefer the unified archive (libiree_runtime_unified.a) which already
        # bundles vm, hal, base, etc.  Mixing it with the individual component
        # archives causes duplicate-symbol errors under --whole-archive.
        unified=$(find iree-build -name "libiree_runtime_unified.a" | head -1)

        if [ -n "$unified" ]; then
          log "Using unified archive: $unified"
          # The unified archive plus flatcc is sufficient.
          echo "$unified" > runtime_libs.txt
          find iree-build -name "libflatcc*.a" \
            ! -name "*test*" \
            >> runtime_libs.txt
        else
          log "No unified archive found - collecting individual archives"
          find iree-build \
            \( -name "libiree_*.a" -o -name "libflatcc*.a" \) \
            ! -path "*/compiler/*" \
            ! -path "*/tools/*" \
            ! -name "*test*" \
            ! -name "*benchmark*" \
            | sort > runtime_libs.txt
        fi

        n=$(wc -l < runtime_libs.txt)
        log "Found $n static archives"

        if [ "$n" -eq 0 ]; then
          log "ERROR: No runtime static archives found - cmake build may have failed"
          exit 1
        fi

        # -----------------------------------------------------------------------
        # Create combined runtime DSO.
        # -----------------------------------------------------------------------
        log "Creating libIREERuntime.so via --whole-archive link"

        sys_rpath="${lib.makeLibraryPath [
      zlib
      zstd
      libxml2
      ncurses
      libffi
      stdenv.cc.cc.lib
    ]}"

        # Generate a version script that exports all iree_* symbols.
        # The static archives are compiled with -fvisibility=hidden so the
        # symbols default to local in the DSO.  This script overrides that.
        cat > export.map <<'VERSCRIPT'
    {
      global:
        iree_*;
      local:
        *;
    };
    VERSCRIPT

        g++ -shared -fPIC \
          -o "$out/lib/libIREERuntime.so" \
          -Wl,--whole-archive \
          $(cat runtime_libs.txt | tr '\n' ' ') \
          -Wl,--no-whole-archive \
          -lz -lzstd -lxml2 -lncurses -lffi -lstdc++ -lm \
          -Wl,--allow-shlib-undefined \
          -Wl,--version-script=export.map

        patchelf --set-rpath "\$ORIGIN:${ireeLlvm}/lib:$sys_rpath" \
          "$out/lib/libIREERuntime.so"

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
        lib_path="$out/lib/libIREERuntime.so"
        if [ -f "$lib_path" ]; then
          if nm -D "$lib_path" 2>/dev/null | grep -q "iree_runtime_instance_create"; then
            log "OK: iree_runtime_instance_create exported from libIREERuntime.so"
          else
            log "WARNING: iree_runtime_instance_create not found - check archive selection"
          fi
        else
          log "ERROR: libIREERuntime.so was not produced"
          exit 1
        fi

        if [ -f "$out/include/iree/runtime/api.h" ]; then
          log "OK: iree/runtime/api.h installed"
        else
          log "WARNING: iree/runtime/api.h not found - @cImport will fail at build time"
        fi

        log "Installation complete"
  '';

  meta = {
    description = "IREE runtime shared library (libIREERuntime.so) with CPU HAL drivers (local-sync + local-task)";
    license = lib.licenses.asl20;
  };
}
