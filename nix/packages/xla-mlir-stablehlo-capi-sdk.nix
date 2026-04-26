# nix/xla-mlir-stablehlo-capi-sdk.nix
#
# Builds the StableHLO C API against a pre-built LLVM/MLIR (from llvm.nix).
# Sources are provided as flake inputs (xlaSrc, stablehloSrc).
# XLA patches for StableHLO are applied from xlaSrc/third_party/.
{
  lib,
  stdenv,
  runCommand,
  cmake,
  ninja,
  python3,
  perl,
  patch,
  patchelf,
  zlib,
  zstd,
  libxml2,
  ncurses,
  libedit,
  libffi,
  lld,
  binutils,
  # Flake source inputs.
  xlaSrc,
  stablehloSrc,
  # Pre-built LLVM/MLIR from llvm.nix (shared with TVM).
  llvm,
  # When true: RelWithDebInfo, retain DWARF, don't strip.
  # When false (default, production): Release, NDEBUG, stripped.
  withDebugSymbols ? false,
  # Native CPU codegen. StablehloCAPI is mostly compiler infra; native tuning
  #  has small impact. Threaded for completeness.
  withNativeTuning ? false,
  enableLto ? false,
  extraCxxFlags ? [],
  extraLdFlags ? [],
}: let
  stablehloPatches = ["temporary.patch"];

  patchedStablehloSrc = runCommand "stablehlo-src-patched" {nativeBuildInputs = [patch perl];} ''
    set -euo pipefail
    cp -r ${stablehloSrc} "$out"
    chmod -R u+w "$out"
    cd "$out"

    echo "[stablehlo] Verifying patch set"
    expected_patches="${lib.concatStringsSep " " stablehloPatches}"
    actual_patches="$(cd ${xlaSrc}/third_party/stablehlo && ls *.patch | tr '\n' ' ')"
    for p in $actual_patches; do
      case " $expected_patches " in
        *" $p "*) ;;
        *) echo "ERROR: New or unexpected StableHLO patch detected: $p" >&2; exit 1;;
      esac
    done
    for p in $expected_patches; do
      echo "[stablehlo] Applying $p"
      patch -p1 < "${xlaSrc}/third_party/stablehlo/$p"
    done

    # Force ONLY StablehloCAPI to be SHARED without turning on BUILD_SHARED_LIBS globally for StableHLO.
    # This relies on MLIR's add_mlir_public_c_api_library forwarding args into add_mlir_library which supports SHARED.
    if [ -f stablehlo/integrations/c/CMakeLists.txt ]; then
      perl -pi -e 's/add_mlir_public_c_api_library\(StablehloCAPI/add_mlir_public_c_api_library(StablehloCAPI SHARED/' \
        stablehlo/integrations/c/CMakeLists.txt
    fi

    # Sanity check
    grep -n "add_mlir_public_c_api_library(StablehloCAPI" -n stablehlo/integrations/c/CMakeLists.txt || true
    grep -n "StablehloCAPI SHARED" stablehlo/integrations/c/CMakeLists.txt || true

    echo "[stablehlo] post patch stablehlo/integrations/c/CMakeLists.txt:"
    cat stablehlo/integrations/c/CMakeLists.txt
  '';
in
  stdenv.mkDerivation {
    pname = "xla-mlir-stablehlo-capi-sdk";
    version = "xla-${xlaSrc.shortRev or "unknown"}";

    # out: headers + minimal (DT_NEEDED) lib closure for runtime use.
    # dev: full lib closure (all StableHLO + LLVM/MLIR libs) for header
    #      navigation, linking against extra MLIR libs, debugger symbol
    #      resolution. Strict superset of `out`'s lib set.
    outputs = ["out" "dev"];

    strictDeps = true;
    dontUnpack = true;
    dontConfigure = true;
    dontStrip = withDebugSymbols;

    nativeBuildInputs = [
      cmake
      ninja
      python3
      patchelf
      lld
      binutils

      # host-tool runtime deps (strictDeps)
      stdenv.cc.cc.lib
      zlib
      zstd
    ];

    buildInputs = [
      zlib
      zstd
      libxml2
      ncurses
      libedit
      libffi
    ];

    # Only StableHLO is built here; LLVM/MLIR comes pre-built from llvm.nix.
    buildPhase = ''
      set -euo pipefail

      mkdir -p stablehlo-build
      cxxFlags="${lib.concatStringsSep " " ((lib.optionals withNativeTuning ["-march=native" "-mtune=native"]) ++ extraCxxFlags)}"
      ldFlags="${lib.concatStringsSep " " extraLdFlags}"
      cmake -S ${patchedStablehloSrc} -B stablehlo-build -G Ninja \
        -DCMAKE_BUILD_TYPE=${if withDebugSymbols then "RelWithDebInfo" else "Release"} \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
        -DMLIR_DIR="${llvm}/lib/cmake/mlir" \
        -DLLVM_DIR="${llvm}/lib/cmake/llvm" \
        ${lib.optionalString enableLto "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"} \
        ''${cxxFlags:+-DCMAKE_CXX_FLAGS="$cxxFlags"} \
        ''${ldFlags:+-DCMAKE_SHARED_LINKER_FLAGS="$ldFlags"}

      ninja -C stablehlo-build StablehloCAPI

      # Restore -u to default so fixupPhase's strip-hook doesn't trip on its
      #  own unset variable references when stripping kicks in.
      set +u
    '';

    installPhase = ''
      set -euo pipefail
      log() { echo "[xla-mlir-stablehlo-capi-sdk] $*" >&2; }

      mkdir -p "$out/include" "$out/lib"

      # --- Headers ---
      # MLIR C API and internal headers from pre-built LLVM
      cp -r "${llvm}/include/mlir-c" "$out/include/mlir-c"
      if [ -d "${llvm}/include/mlir" ]; then
        cp -r "${llvm}/include/mlir" "$out/include/mlir"
      fi

      # StableHLO C API headers
      mkdir -p "$out/include/stablehlo/integrations/c"
      cp -r "${patchedStablehloSrc}/stablehlo/integrations/c/"* "$out/include/stablehlo/integrations/c/"

      # XLA PJRT C API headers
      mkdir -p "$out/include/xla/pjrt/c"
      cp -v "${xlaSrc}/xla/pjrt/c/"*.h "$out/include/xla/pjrt/c/"

      # XLA FFI C API headers (for typed FFI custom call handlers)
      mkdir -p "$out/include/xla/ffi/api"
      cp -v "${xlaSrc}/xla/ffi/api/"*.h "$out/include/xla/ffi/api/"

      # --- Libraries (out): minimal DT_NEEDED closure for runtime ---
      log "Copying MLIR-C + StablehloCAPI and DT_NEEDED closure to out"

      # Copy the two root DSOs
      cp -v "${llvm}/lib/libMLIR-C.so"* "$out/lib/" || true
      cp -v stablehlo-build/lib/libStablehloCAPI.so* "$out/lib/" || true

      # Copy transitive deps from pre-built LLVM and StableHLO build tree.
      copy_needed_closure() {
        local search_dirs=("${llvm}/lib" "$PWD/stablehlo-build/lib")
        local -A seen
        local queue=("$@")

        while [ "''${#queue[@]}" -gt 0 ]; do
          local libpath="''${queue[0]}"
          queue=("''${queue[@]:1}")

          [ -f "$libpath" ] || continue
          local base="$(basename "$libpath")"
          if [ -n "''${seen[$base]:-}" ]; then
            continue
          fi
          seen["$base"]=1

          # Ensure it's present in output
          if [ ! -f "$out/lib/$base" ]; then
            cp -v "$libpath" "$out/lib/"
          fi

          # Enqueue internal deps that we can find in search_dirs
          while IFS= read -r need; do
            local found=""
            for d in "''${search_dirs[@]}"; do
              if [ -f "$d/$need" ]; then
                found="$d/$need"
                break
              fi
            done
            if [ -n "$found" ]; then
              queue+=("$found")
            fi
          done < <(patchelf --print-needed "$libpath" || true)
        done
      }

      mlir_root="$(ls -1 "${llvm}/lib/libMLIR-C.so."* 2>/dev/null | head -n1 || true)"
      stablehlo_root="$(ls -1 stablehlo-build/lib/libStablehloCAPI.so.* 2>/dev/null | head -n1 || true)"
      copy_needed_closure "$mlir_root" "$stablehlo_root"

      # --- Libraries (dev): full closure for header navigation / extra linking ---
      log "Copying ALL StableHLO + LLVM/MLIR build artifacts to dev"
      mkdir -p "$dev/lib"
      find stablehlo-build -type f \( -name "*.a" -o -name "*.so*" \) -exec cp -v {} "$dev/lib/" \;
      for f in "${llvm}/lib/"*.so* "${llvm}/lib/"*.a; do
        [ -f "$f" ] || continue
        base="$(basename "$f")"
        [ -f "$dev/lib/$base" ] || cp -v "$f" "$dev/lib/"
      done

      # Files copied from the nix store are read-only; make writable for patchelf.
      chmod -R u+w "$out/lib" "$dev/lib"

      # Ensure unversioned linker names exist for the two link-entry DSOs in both outputs.
      for libdir in "$out/lib" "$dev/lib"; do
        for name in libMLIR-C libStablehloCAPI; do
          so="$(ls -1 "$libdir/$name.so."* 2>/dev/null | head -n1 || true)"
          if [ -n "$so" ]; then
            ln -sfn "$(basename "$so")" "$libdir/$name.so"
          fi
        done
      done

      # Patch RUNPATH for all shipped DSOs in both outputs.
      rpath="\$ORIGIN:\$ORIGIN/../runtime/sys/lib:${
        lib.makeLibraryPath [
          zlib
          zstd
          libxml2
          ncurses
          libedit
          libffi
          stdenv.cc.cc.lib
        ]
      }"

      patch_rpath_in() {
        local libdir="$1"
        for f in "$libdir/"*.so*; do
          [ -f "$f" ] || continue
          if ! patchelf --set-rpath "$rpath" "$f"; then
            case "$(basename "$f")" in
              libMLIR-C.so*|libStablehloCAPI.so*)
                log "ERROR: patchelf failed for critical DSO: $f"
                exit 1
                ;;
              *)
                log "WARNING: patchelf failed for non-critical DSO: $f"
                ;;
            esac
          fi
        done
      }
      patch_rpath_in "$out/lib"
      patch_rpath_in "$dev/lib"

      # Sanity check... C API dialect handle function should be present when StablehloCAPI is built properly.
      if [ -f "$out/lib/libStablehloCAPI.so" ]; then
        if ! nm -D "$out/lib/libStablehloCAPI.so" 2>/dev/null | grep -q "mlirGetDialectHandle__stablehlo__"; then
          log "WARNING: mlirGetDialectHandle__stablehlo__ not found in libStablehloCAPI.so (symbol export may differ by version/config)"
        fi
      fi

      log "SDK installation complete"

      # Restore -u to default so fixupPhase's strip-hook doesn't trip.
      set +u
    '';

    meta = {
      description = "MLIR + StableHLO + PJRT C API SDK";
      license = lib.licenses.asl20;
    };
  }
