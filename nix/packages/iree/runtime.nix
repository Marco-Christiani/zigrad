# Local CPU drivers are enabled. Required submodules are explicit source inputs,
#  and the compiler remains a separate package input.
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
  ireeSrc,
  ireeRevision,
  ireeStablehloSrc,
  ireeFlatccSrc,
  ireeBenchmarkSrc,
  cudaToolkit ? null,
  drivers ? ["local-sync" "local-task"],
  # Retain debug information in a RelWithDebInfo build.
  withDebugSymbols ? false,
  # Native tuning applies to runtime kernel dispatch and HAL submission.
  withNativeTuning ? false,
  enableLto ? false,
  extraCxxFlags ? [],
  extraLdFlags ? [],
  extraCmakeFlags ? [],
  extraNativeBuildInputs ? [],
  extraBuildInputs ? [],
  ireeHostTools ? null,
}:
assert lib.assertMsg (drivers != []) "the IREE runtime requires a HAL driver";
assert lib.assertMsg (!lib.elem "cuda" drivers || cudaToolkit != null)
"the IREE CUDA runtime requires cudaToolkit";
  stdenv.mkDerivation {
    pname = "iree-runtime-${lib.concatStringsSep "-" drivers}";
    version = "iree-${builtins.substring 0 7 ireeRevision}";

    # Source assembly happens in `buildPhase` because each submodule is pinned.
    dontUnpack = true;
    dontConfigure = true;
    dontStrip = withDebugSymbols;

    strictDeps = true;

    # Native IREE build tools load these libraries inside the strict-deps sandbox.
    nativeBuildInputs =
      [
        cmake
        ninja
        python3
        stdenv.cc.cc.lib
        zlib
        zstd
        libxml2
        ncurses
        libffi
      ]
      ++ lib.optional (cudaToolkit != null) cudaToolkit
      ++ extraNativeBuildInputs;

    buildInputs =
      [
        zlib
        zstd
        libxml2
        ncurses
        libffi
      ]
      ++ extraBuildInputs;

    buildPhase = ''
      set -euo pipefail
      log() { echo "[iree-runtime] $*" >&2; }

      ${lib.optionalString (cudaToolkit != null) "export CUDAToolkit_ROOT=${cudaToolkit}"}

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

      log "Configuring IREE (runtime-only build)"
      mkdir -p iree-build

      cmake -S iree-src -B iree-build -G Ninja \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=${stdenv.hostPlatform.parsed.cpu.name} \
        ${lib.optionalString (ireeHostTools != null) "-DIREE_HOST_BIN_DIR=${ireeHostTools}/bin"} \
        -DCMAKE_BUILD_TYPE=${
        if withDebugSymbols
        then "RelWithDebInfo"
        else "Release"
      } \
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
        ${lib.concatMapStringsSep " \\\n+        " (driver: "-DIREE_HAL_DRIVER_${lib.toUpper (lib.replaceStrings ["-"] ["_"] driver)}=ON") drivers} \
        \
        -DIREE_ENABLE_CPUINFO=OFF \
        -DIREE_ENABLE_LIBBACKTRACE=OFF \
        \
        -DCMAKE_INSTALL_PREFIX="$out" \
        ${lib.escapeShellArgs extraCmakeFlags} \
        ${lib.optionalString enableLto "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"} \
        ${let
        cxxFlags = (lib.optionals withNativeTuning ["-march=native" "-mtune=native"]) ++ extraCxxFlags;
      in
        lib.optionalString (cxxFlags != []) "-DCMAKE_CXX_FLAGS='${lib.concatStringsSep " " cxxFlags}'"} \
        ${lib.optionalString (extraLdFlags != []) "-DCMAKE_SHARED_LINKER_FLAGS='${lib.concatStringsSep " " extraLdFlags}'"}

      log "Building IREE runtime"
      build_jobs="$NIX_BUILD_CORES"
      if [ "$build_jobs" -gt 32 ]; then build_jobs=32; fi

      mkdir -p host-tools
      ${
        if ireeHostTools == null
        then ''
          cmake --build iree-build --parallel "$build_jobs" \
            --target iree-flatcc-cli iree-c-embed-data
          for tool in iree-flatcc-cli iree-c-embed-data; do
            executable=$(find iree-build -type f -name "$tool" -perm -0100 | head -1)
            test -n "$executable"
            ln -s "$PWD/$executable" "host-tools/$tool"
          done
        ''
        else ''
          ln -s ${ireeHostTools}/bin/iree-flatcc-cli host-tools/iree-flatcc-cli
          ln -s ${ireeHostTools}/bin/iree-c-embed-data host-tools/iree-c-embed-data
        ''
      }
      export PATH="$PWD/host-tools:$PATH"

      # Prefer the narrow runtime targets and fall back to the configured build.
      #
      # IREE replaces dots with underscores in `<module>_<target>` CMake names.
      if cmake --build iree-build --parallel "$build_jobs" --target iree_runtime_runtime; then
        log "Built via target: iree_runtime_runtime"
      elif cmake --build iree-build --parallel "$build_jobs" --target iree_base_base iree_hal_hal iree_vm_vm; then
        log "Built via targets: iree_base_base iree_hal_hal iree_vm_vm"
      else
        log "WARNING: specific targets unavailable, building all non-compiler targets"
        cmake --build iree-build --parallel "$build_jobs"
      fi

      log "Build phase complete"

      # Nix fixup hooks expect unset variables to expand without failure.
      set +u
    '';

    installPhase = ''
      set -euo pipefail
      log() { echo "[iree-runtime] $*" >&2; }

      mkdir -p "$out/lib" "$out/include" "$hostTools/bin"

      log "Installing static archives"

      unified=$(find iree-build -name "libiree_runtime_unified.a" | head -1)

      if [ -z "$unified" ]; then
        log "ERROR: libiree_runtime_unified.a not found - cmake build may have failed"
        exit 1
      fi

      log "Installing unified archive: $unified"
      cp "$unified" "$out/lib/libiree_runtime_unified.a"

      # Install the FlatCC archives required by the IREE runtime.
      find iree-build -name "libflatcc*.a" ! -name "*test*" \
        -exec cp {} "$out/lib/" \;

      cp -L host-tools/iree-flatcc-cli "$hostTools/bin/iree-flatcc-cli"
      cp -L host-tools/iree-c-embed-data "$hostTools/bin/iree-c-embed-data"

      log "Installing runtime headers"

      # Copy headers from the configured source layout.
      if [ -d iree-src/runtime/src/iree ]; then
        cp -r iree-src/runtime/src/iree/. "$out/include/iree/"
      fi

      # Accept the alternate upstream source layout.
      if [ -d iree-src/iree ] && [ ! -d "$out/include/iree/base" ]; then
        cp -r iree-src/iree/. "$out/include/iree/"
      fi

      # Add generated headers without replacing source headers.
      find iree-build \
        \( -path "*/iree/base/*.h" \
          -o -path "*/iree/hal/*.h" \
          -o -path "*/iree/vm/*.h" \
          -o -path "*/iree/runtime/*.h" \
          -o -path "*/iree/task/*.h" \
        \) \
        ! -path "*/compiler/*" \
      | while IFS= read -r src; do
          # Preserve each generated header's path below the IREE root.
          rel="$(echo "$src" | sed 's|.*iree-build[^/]*/||;s|^runtime/src/||')"
          dest="$out/include/$rel"
          if [ ! -f "$dest" ]; then
            mkdir -p "$(dirname "$dest")"
            cp "$src" "$dest"
          fi
        done

      if nm "$out/lib/libiree_runtime_unified.a" 2>/dev/null | grep -q 'T iree_runtime_instance_create'; then
        log "OK: iree_runtime_instance_create found in archive"
      else
        log "WARNING: iree_runtime_instance_create not found"
      fi

      if [ -f "$out/include/iree/runtime/api.h" ]; then
        log "OK: iree/runtime/api.h installed"
      else
        log "WARNING: iree/runtime/api.h not found, Zig compilation will fail"
      fi

      log "Installation complete"

      # Nix fixup hooks expect unset variables to expand without failure.
      set +u
    '';

    meta = {
      description = "IREE runtime static archives for ${lib.concatStringsSep ", " drivers}";
      license = lib.licenses.asl20;
    };

    outputs = ["out" "hostTools"];
  }
