# nix/xla-mlir-stablehlo-capi-sdk.nix
{ lib
, stdenv
, fetchurl
, runCommand
, cmake
, ninja
, python3
, perl
, unzip
, patch
, zlib
, zstd
, libxml2
, ncurses
, libedit
, libffi
, lockFile
}:

let
  lock = builtins.fromJSON (builtins.readFile lockFile);
  pins = lock.pins;

  xla = pins.xla;
  llvm = pins.llvm;
  stablehlo = pins.stablehlo;

  xlaTar = fetchurl {
    url  = xla.tarball_url;
    hash = xla.hash_sri;
  };

  xlaSrc = runCommand "xla-src-${builtins.substring 0 12 xla.commit}" {} ''
    mkdir -p $out
    tar -xzf ${xlaTar} -C $out --strip-components=1
  '';

  llvmTar = fetchurl {
    urls = llvm.urls;
    hash = llvm.hash_sri;
  };

  llvmSrc = runCommand "llvm-src-${builtins.substring 0 12 llvm.commit}"
    { nativeBuildInputs = [ patch ]; }
    ''
      mkdir -p $out
      tar -xzf ${llvmTar} -C $out --strip-components=1
      cd $out
      patch -p1 < ${xlaSrc}/third_party/llvm/build.patch
      patch -p1 < ${xlaSrc}/third_party/llvm/mathextras.patch
      patch -p1 < ${xlaSrc}/third_party/llvm/toolchains.patch
      patch -p1 < ${xlaSrc}/third_party/llvm/zstd.patch
      patch -p1 < ${xlaSrc}/third_party/llvm/lit_test.patch
    '';

  stablehloZip = fetchurl {
    urls = stablehlo.urls;
    hash = stablehlo.hash_sri;
  };

  stablehloSrc = runCommand "stablehlo-src-${builtins.substring 0 12 stablehlo.commit}"
    { nativeBuildInputs = [ unzip patch ]; }
    ''
      mkdir -p $out
      unzip -q ${stablehloZip} -d $out
      mv $out/*/* $out/
      cd $out
      patch -p1 < ${xlaSrc}/third_party/stablehlo/temporary.patch
    '';
in

stdenv.mkDerivation {
  pname = "xla-mlir-stablehlo-capi-sdk";
  version = "xla-${builtins.substring 0 12 xla.commit}";

  strictDeps = true;
  dontUnpack = true;
  dontConfigure = true;
  dontStrip = true;

  nativeBuildInputs = [ cmake ninja python3 perl ];
  buildInputs = [ zlib zstd libxml2 ncurses libedit libffi ];

  # buildPhase = ''
  #   mkdir -p llvm-build
  #   cmake -S ${llvmSrc}/llvm -B llvm-build -G Ninja \
  #     -DCMAKE_BUILD_TYPE=Release \
  #     -DLLVM_ENABLE_PROJECTS=mlir \
  #     -DLLVM_TARGETS_TO_BUILD=host \
  #     -DLLVM_INCLUDE_TESTS=OFF \
  #     -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
  #
  #   # Ensure tablegen + pdll exist (StableHLO needs these).
  #   cmake --build llvm-build --target llvm-tblgen mlir-tblgen mlir-pdll
  #
  #   cmake --build llvm-build --target \
  #     MLIRCAPIIR MLIRCAPIArith MLIRCAPIMath MLIRCAPISCF \
  #     MLIRCAPITransforms MLIRCAPIFunc MLIRCAPITensor
  #
  #   mkdir -p stablehlo-build
  #   cmake -S ${stablehloSrc} -B stablehlo-build -G Ninja \
  #     -DMLIR_DIR="$PWD/llvm-build/lib/cmake/mlir" \
  #     -DLLVM_DIR="$PWD/llvm-build/lib/cmake/llvm" \
  #     -DSTABLEHLO_ENABLE_TESTS=OFF
  #
  #   if ninja -C stablehlo-build -t targets all | grep -q '^StablehloCAPI:'; then
  #     ninja -C stablehlo-build StablehloCAPI
  #   else
  #     echo "ERROR: StablehloCAPI target not found in StableHLO build." >&2
  #     echo "DEBUG: available stablehlo-build targets (first 200):" >&2
  #     ninja -C stablehlo-build -t targets all | head -n 200 >&2
  #     exit 1
  #   fi
  # '';

  buildPhase = ''
    set -euo pipefail
    export HOME="$TMPDIR/home"
    mkdir -p "$HOME"

    mkdir -p llvm-build
    cmake -S ${llvmSrc}/llvm -B llvm-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_TARGETS_TO_BUILD=host \
      -DLLVM_INCLUDE_TESTS=OFF \
      -DLLVM_INCLUDE_EXAMPLES=OFF \
      -DLLVM_INCLUDE_DOCS=OFF \
      -DMLIR_INCLUDE_TESTS=OFF \
      -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
      -DMLIR_BUILD_MLIR_C_DYLIB=ON

    # Tools StableHLO headers generation might expect (safe even if unused later, tbd whats needed still)
    cmake --build llvm-build --target llvm-tblgen mlir-tblgen mlir-pdll
    cmake --build llvm-build --target MLIR-C

    mkdir -p stablehlo-build
    cmake -S ${stablehloSrc} -B stablehlo-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DMLIR_DIR="$PWD/llvm-build/lib/cmake/mlir" \
      -DLLVM_DIR="$PWD/llvm-build/lib/cmake/llvm"

    ninja -C stablehlo-build StablehloCAPI

    # Build MLIR C API libraries (what we actually want)
    cmake --build llvm-build --target \
      MLIRCAPIIR \
      MLIRCAPIArith \
      MLIRCAPIMath \
      MLIRCAPISCF \
      MLIRCAPITransforms \
      MLIRCAPIFunc \
      MLIRCAPITensor
  '';

  # installPhase = ''
  #   set -euo pipefail
  #
  #   mkdir -p "$out/include" "$out/lib"
  #
  #   # MLIR C API headers
  #   cp -r "${llvmSrc}/mlir/include/mlir-c" "$out/include/"
  #
  #   # StableHLO C API headers
  #   mkdir -p "$out/include/stablehlo/integrations"
  #   cp -r "${stablehloSrc}/stablehlo/integrations/c" "$out/include/stablehlo/integrations/"
  #
  #   # PJRT C API headers
  #   mkdir -p "$out/include/xla/pjrt/c"
  #   cp -r "${xlaSrc}/xla/pjrt/c/"*.h "$out/include/xla/pjrt/c/"
  #
  #   # Copy only the relevant libs (MLIR CAPI + optional aggregate + StablehloCAPI)
  #   copy_selected_libs() {
  #     local root="$1"
  #     [ -d "$root" ] || return 0
  #     find "$root" -type f \
  #       \( -name "libMLIRCAPI*.a" -o -name "libMLIRCAPI*.so*" -o -name "libMLIR-C*.a" -o -name "libMLIR-C*.so*" \
  #         -o -name "libStablehloCAPI*.a" -o -name "libStablehloCAPI*.so*" \) \
  #       -print -exec cp -v {} "$out/lib/" \;
  #   }
  #
  #   copy_selected_libs llvm-build
  #   copy_selected_libs stablehlo-build
  #
  #   # Hard check: we expect at least one MLIR CAPI library
  #   if ! find "$out/lib" -maxdepth 1 -type f -name "libMLIRCAPI*" | grep -q .; then
  #     echo "ERROR: No MLIR CAPI libraries were copied into $out/lib" >&2
  #     echo "DEBUG: listing likely locations under llvm-build (top 200 entries):" >&2
  #     find llvm-build -maxdepth 4 -type f \( -name "libMLIR*" -o -name "libLLVM*" \) | head -n 200 >&2
  #     exit 1
  #   fi
  # '';


  installPhase = ''
    set -euo pipefail

    mkdir -p "$out/include" "$out/lib"

    # MLIR C API headers
    cp -r "${llvmSrc}/mlir/include/mlir-c" "$out/include/"

    # StableHLO C API headers (header-only; do not build stablehlo)
    mkdir -p "$out/include/stablehlo/integrations"
    cp -r "${stablehloSrc}/stablehlo/integrations/c" "$out/include/stablehlo/integrations/"

    # PJRT C API headers
    mkdir -p "$out/include/xla/pjrt/c"
    cp -r "${xlaSrc}/xla/pjrt/c/"*.h "$out/include/xla/pjrt/c/"

    find stablehlo-build -type f \
      \( -name "libStablehloCAPI*.a" -o -name "libStablehloCAPI*.so*" -o -name "libStablehloCAPI*.dylib" \) \
      -print -exec cp -v {} "$out/lib/" \;


    # Copy MLIR CAPI libs from build tree
    find llvm-build -type f \
      \( -name "libMLIRCAPI*.a" -o -name "libMLIRCAPI*.so*" -o -name "libMLIR-C*.a" -o -name "libMLIR-C*.so*" \) \
      -print -exec cp -v {} "$out/lib/" \;

    # Ensure the unversioned linker name exists (Zig searches libMLIR-C.so)
    mlir_c_so="$(ls -1 "$out/lib/libMLIR-C.so."* 2>/dev/null | head -n1 || true)"
    if [ -n "$mlir_c_so" ]; then
      ln -sfn "$(basename "$mlir_c_so")" "$out/lib/libMLIR-C.so"
    fi


    # Hard check: we expect at least one MLIR CAPI library
    if ! find "$out/lib" -maxdepth 1 -type f -name "libMLIRCAPI*" | grep -q .; then
      echo "ERROR: No MLIR CAPI libraries were copied into $out/lib" >&2
      find llvm-build -maxdepth 4 -type f -name "libMLIR*" | head -n 200 >&2
      exit 1
    fi

    # Hard check for stablehlo
    if ! find "$out/lib" -maxdepth 1 -type f -name "libStablehloCAPI*" | grep -q .; then
      echo "ERROR: No StablehloCAPI library was copied into $out/lib" >&2
      find stablehlo-build -maxdepth 4 -type f -name "libStablehlo*" | head -n 200 >&2
      exit 1
    fi
  '';

  meta = {
    description = "MLIR + StableHLO + PJRT C API SDK derived from JAX->XLA lock.json";
    license = lib.licenses.asl20;
  };
}

