{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  ninja,
  python3,
  patch,
  # Standard nixpkgs LLVM (used when llvm is null)
  llvmPackages ? null,
  # LLVM 22 built from XLA-pinned sources (preferred - ensures ABI compat with SDK)
  llvm ? null,
  pkg-config,
  git,
  patchelf,
  zlib,
  ncurses,
  libxml2,
  autoAddDriverRunpath,
  # cudatoolkit-style layout (bin/nvcc, include/, lib/, lib/stubs/libcuda.so).
  #  Provided by cuda-redist.dev; passed in by sdk.nix.
  cudaToolkit ? null,
  gccHost ? null,
  cudaSupport ? true,
  cudaArchitectures ? [],
  enableCublas ? false,
  enableCudnn ? false,
  enableCutlass ? false,
  # When true, build & install TVM's Python/FFI bindings. Independent from the
  #  `out`/`dev` output split: this controls *what gets compiled*, while outputs
  #  control *which built artifacts go where*.
  withPythonBindings ? false,
  tvmSrcOverride ? null,
  tvmRev ? "v0.22.0",
  tvmHash ? "sha256-KcHUcblwtqxNofHKofuQHu2d7hIqS9FUvc41OkCVtnY=",
}: let
  boolToCmake = v:
    if v
    then "ON"
    else "OFF";
  cudaEnabled = cudaSupport && cudaToolkit != null;
  cudaArchStr = lib.concatStringsSep ";" cudaArchitectures;

  # When llvm is provided, use shared LLVM to match the SDK's LLVM 22.
  # When using nixpkgs llvmPackages, use static linking to isolate symbols.
  useCustomLlvm = llvm != null;
  llvmDev = if useCustomLlvm then llvm else llvmPackages.llvm.dev;
  llvmLib = if useCustomLlvm then llvm else llvmPackages.llvm.lib;
  llvmConfigCmd = if useCustomLlvm
    then "${llvm}/bin/llvm-config"  # Shared linking
    else "${llvmPackages.llvm.dev}/bin/llvm-config --link-static";  # Static isolation

  src =
    if tvmSrcOverride != null
    then tvmSrcOverride
    else
      fetchFromGitHub {
        owner = "apache";
        repo = "tvm";
        rev = tvmRev;
        hash = tvmHash;
        fetchSubmodules = true;
      };
in
  assert lib.assertMsg (!cudaSupport || cudaToolkit != null) "tvm: cudaSupport=true requires cudaToolkit";
  assert lib.assertMsg (llvm != null || llvmPackages != null) "tvm: requires either llvm or llvmPackages";
    stdenv.mkDerivation {
      pname = "tvm";
      version = tvmRev;
      inherit src;
      outputs = ["out" "dev"];

      strictDeps = true;
      dontStrip = true;

      postPatch = ''
        ${lib.optionalString useCustomLlvm ''
          # Apply LLVM 22 API compatibility patch
          patch -p1 < ${../patches/tvm-llvm22.patch}
        ''}

        ${lib.optionalString withPythonBindings ''
          # Enable Python module build in tvm-ffi subproject
          # By default, tvm-ffi skips Python module when used as a subdirectory
          patch -p1 < ${../patches/tvm-ffi-python.patch}
        ''}
      '';

    nativeBuildInputs = [
      cmake
      ninja
      python3
      python3.pkgs.cython  # Required to build tvm_ffi Cython extension (core.pyx)
      pkg-config
      git
      patchelf
      patch
      llvmDev  # Provides llvm-config (either custom LLVM or llvmPackages.llvm.dev)
    ] ++ lib.optionals cudaEnabled [
      autoAddDriverRunpath  # automatically patches rpath to include /run/opengl-driver for libcuda.so
    ];

      buildInputs = [
        # LLVM runtime library (either custom LLVM or llvmPackages.llvm.lib)
        llvmLib
        # LLVM's dependencies (required for linking)
        zlib
        ncurses   # provides libtinfo
        libxml2
      ] ++ lib.optionals cudaEnabled [
        cudaToolkit
      ];

      # Note: Using `.` (current dir) for cmake -S because postPatch patches the source in-place
      # and using ${src} would reference the unpatched original source in the nix store.
      configurePhase = ''
        set -euo pipefail

        mkdir -p build
        cp cmake/config.cmake build/config.cmake
        chmod u+w build/config.cmake

        cat >> build/config.cmake <<EOF
        set(CMAKE_BUILD_TYPE RelWithDebInfo)
        set(CMAKE_CXX_STANDARD 17)
        set(USE_LLVM "${llvmConfigCmd}")
        set(HIDE_PRIVATE_SYMBOLS ON)
        set(USE_CUDA ${boolToCmake cudaEnabled})
        set(USE_METAL OFF)
        set(USE_VULKAN OFF)
        set(USE_OPENCL OFF)
        set(USE_CUBLAS ${boolToCmake enableCublas})
        set(USE_CUDNN ${boolToCmake enableCudnn})
        set(USE_CUTLASS ${boolToCmake enableCutlass})
        ${lib.optionalString withPythonBindings "set(TVM_FFI_BUILD_PYTHON_MODULE ON)"}
        EOF

        ${lib.optionalString cudaEnabled ''
          cat >> build/config.cmake <<EOF
          set(CUDAToolkit_ROOT "${cudaToolkit}")
          set(CUDA_TOOLKIT_ROOT_DIR "${cudaToolkit}")
          set(CMAKE_CUDA_COMPILER "${cudaToolkit}/bin/nvcc")
          set(CUDA_CUDA_LIBRARY "${cudaToolkit}/lib/stubs/libcuda.so")
          EOF

          ${lib.optionalString (cudaArchitectures != []) ''
            cat >> build/config.cmake <<EOF
            set(CMAKE_CUDA_ARCHITECTURES "${cudaArchStr}")
            EOF
          ''}

          ${lib.optionalString (gccHost != null) ''
            cat >> build/config.cmake <<EOF
            set(CMAKE_CUDA_HOST_COMPILER "${gccHost}/bin/g++")
            EOF
          ''}
        ''}

        cmake -S . -B build -G Ninja
      '';

      buildPhase = ''
        cmake --build build --parallel $NIX_BUILD_CORES
      '';

    # Note: we only ship headers + shared libs by default.
    # Python/ffi bindings included when withPythonBindings=true.
    installPhase = ''
      set -euo pipefail

      mkdir -p $out/lib $dev/include

      # No custom headers needed - NVRTC uses CUDA's bundled libcxx

      for f in build/libtvm* build/libtvm_runtime*; do
        if [ -e "$f" ]; then
          cp -v "$f" $out/lib/
        fi
      done

      # Copy tvm_ffi_testing.so if it exists (needed by Python Cython extension)
      echo "Searching for libtvm_ffi_testing.so..."
      find build -name "libtvm_ffi_testing.so*" -type f
      testing_lib="$(find build -name "libtvm_ffi_testing.so" -type f | head -n1 || true)"
      if [ -n "$testing_lib" ] && [ -f "$testing_lib" ]; then
        echo "Found testing library at: $testing_lib"
        cp -v "$testing_lib" $out/lib/
      else
        echo "WARNING: libtvm_ffi_testing.so not found in build directory"
        echo "This may cause Python imports to fail if core.abi3.so depends on it"
      fi

      # v0.22+: TVM runtime depends on a separate libtvm_ffi.so.
      # It may live under build/3rdparty/tvm-ffi/, so copy it explicitly.
      if [ ! -e "$out/lib/libtvm_ffi.so" ]; then
        found="$(find build -type f -name 'libtvm_ffi.so*' | head -n1 || true)"
        if [ -n "$found" ]; then
          cp -v "$found" $out/lib/
        fi
      fi

      cp -r --no-preserve=mode,ownership include/. $dev/include/
      chmod -R u+w $dev/include

      if [ -d 3rdparty/tvm-ffi/include ]; then
        cp -r --no-preserve=mode,ownership 3rdparty/tvm-ffi/include/. $dev/include/
      fi

      # DLPack headers are a TVM dependency used by the runtime C API.
      if [ -d 3rdparty/dlpack/include/dlpack ]; then
        mkdir -p $dev/include/dlpack
        cp -r 3rdparty/dlpack/include/dlpack/. $dev/include/dlpack/
      elif [ -d 3rdparty/tvm-ffi/3rdparty/dlpack/include/dlpack ]; then
        mkdir -p $dev/include/dlpack
        cp -r 3rdparty/tvm-ffi/3rdparty/dlpack/include/dlpack/. $dev/include/dlpack/
      fi

      # DMLC headers are required by TVM C++ APIs.
      if [ -d 3rdparty/dmlc-core/include/dmlc ]; then
        mkdir -p $dev/include/dmlc
        cp -r 3rdparty/dmlc-core/include/dmlc/. $dev/include/dmlc/
      fi

      # Build rpath for TVM libs.
      # - Include LLVM library in rpath so libtvm.so finds libLLVM.so.
      # - Always include LLVM's runtime deps (zlib, ncurses, libxml2).
      # - Note: libcuda.so.1 is provided by the driver and resolved via autoAddDriverRunpath or LD_LIBRARY_PATH.
      rpath="\$ORIGIN:${lib.makeLibraryPath (
        [stdenv.cc.cc.lib zlib ncurses libxml2 llvmLib]
        ++ lib.optionals cudaEnabled [cudaToolkit]
      )}"
      for f in $out/lib/*.so*; do
        [ -e "$f" ] || continue
        patchelf --set-rpath "$rpath" "$f"
      done

      ${lib.optionalString withPythonBindings ''
        echo "Including Python bindings (withPythonBindings=true)"
        mkdir -p $out/python

        # Copy main TVM Python package
        cp -r python/tvm $out/python/

        # Copy tvm_ffi from 3rdparty
        if [ -d 3rdparty/tvm-ffi/python/tvm_ffi ]; then
          cp -r 3rdparty/tvm-ffi/python/tvm_ffi $out/python/
        fi

        # Copy compiled Cython extension (core*.so) to tvm_ffi/
        # This is built by TVM_FFI_BUILD_PYTHON_MODULE and is required for Python imports
        echo "Searching for Cython core extension..."
        find build -name "core*.so" -type f
        core_so="$(find build -name "core*.so" -type f | head -n1 || true)"
        if [ -n "$core_so" ] && [ -f "$core_so" ]; then
          cp -v "$core_so" $out/python/tvm_ffi/

          # Patch RPATH to find libtvm_ffi.so in $out/lib (two directories up)
          # The Cython module is at $out/python/tvm_ffi/core.abi3.so
          # libtvm_ffi.so is at $out/lib/libtvm_ffi.so
          # So we need $ORIGIN/../../lib
          patchelf --set-rpath "\$ORIGIN/../../lib:$rpath" "$out/python/tvm_ffi/$(basename "$core_so")"

          echo "Installed Cython core extension: $core_so"
        else
          echo "WARNING: Cython core extension not found - Python imports will fail"
          echo "Expected location: build/3rdparty/tvm-ffi/core.abi3.so"
        fi

        # Make bindings writable for any post-install modifications
        chmod -R u+w $out/python
      ''}
    '';

      meta = {
        description = "Apache TVM compiler stack";
        license = lib.licenses.asl20;
        platforms = lib.platforms.linux;
      };
    }
