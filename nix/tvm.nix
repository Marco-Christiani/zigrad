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
  cudaPackages ? null,
  gccHost ? null,
  cudaSupport ? true,
  cudaArchitectures ? [],
  enableCublas ? false,
  enableCudnn ? false,
  enableCutlass ? false,
  devel ? false,
  tvmSrcOverride ? null,
  tvmRev ? "v0.22.0",
  tvmHash ? "sha256-KcHUcblwtqxNofHKofuQHu2d7hIqS9FUvc41OkCVtnY=",
}: let
  boolToCmake = v:
    if v
    then "ON"
    else "OFF";
  cudaEnabled = cudaSupport && cudaPackages != null;
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
  assert lib.assertMsg (!cudaSupport || cudaPackages != null) "tvm: cudaSupport=true requires cudaPackages";
  assert lib.assertMsg (llvm != null || llvmPackages != null) "tvm: requires either llvm or llvmPackages";
    stdenv.mkDerivation {
      pname = "tvm";
      version = tvmRev;
      inherit src;

      strictDeps = true;
      dontStrip = true;

      # Fix: TVM v0.22 unconditionally adds #include <cuda.h> and #include <cstdint> to generated
      # CUDA code, but NVRTC can't compile code that includes these headers (cuda.h includes stdlib.h,
      # and cstdint is a C++ STL header not available in NVRTC).
      # For simple kernels, these aren't needed - only cuda_fp16.h etc for special types.
      # Patch: Only include these headers when need_include_path() returns true.
      postPatch = ''
        ${lib.optionalString cudaEnabled ''
          substituteInPlace src/target/source/codegen_cuda.cc \
            --replace-fail 'decl_stream << "#include <cuda.h>\n";' \
                           'if (need_include_path()) { decl_stream << "#include <cuda.h>\\n"; }' \
            --replace-fail 'decl_stream << "#include <cstdint>\n";' \
                           'if (need_include_path()) { decl_stream << "#include <cstdint>\\n"; }'
        ''}

        ${lib.optionalString useCustomLlvm ''
          # Apply LLVM 22 API compatibility patch
          patch -p1 < ${./tvm-llvm22.patch}
        ''}
      '';

    nativeBuildInputs = [
      cmake
      ninja
      python3
      pkg-config
      git
      patchelf
      patch
      llvmDev
    ] ++ lib.optionals (!useCustomLlvm) [
      llvmPackages.llvm
    ] ++ lib.optionals cudaEnabled [
      autoAddDriverRunpath  # automatically patches rpath to include /run/opengl-driver for libcuda.so
    ];

      buildInputs = [
        # Required for static LLVM linking - these are LLVM's dependencies
        zlib
        ncurses   # provides libtinfo
        libxml2
      ] ++ lib.optionals cudaEnabled [
        cudaPackages.cudatoolkit
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
        EOF

        ${lib.optionalString cudaEnabled ''
          cat >> build/config.cmake <<EOF
          set(CUDAToolkit_ROOT "${cudaPackages.cudatoolkit}")
          set(CUDA_TOOLKIT_ROOT_DIR "${cudaPackages.cudatoolkit}")
          set(CMAKE_CUDA_COMPILER "${cudaPackages.cudatoolkit}/bin/nvcc")
          set(CUDA_CUDA_LIBRARY "${cudaPackages.cudatoolkit}/lib/stubs/libcuda.so")
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
    # Python/ffi bindings are included when devel=true for development/testing.
    installPhase = ''
      set -euo pipefail

      mkdir -p $out/lib $out/include

      for f in build/libtvm* build/libtvm_runtime*; do
        if [ -e "$f" ]; then
          cp -v "$f" $out/lib/
        fi
      done

      # v0.22+: TVM runtime depends on a separate libtvm_ffi.so.
      # It may live under build/3rdparty/tvm-ffi/, so copy it explicitly.
      if [ ! -e "$out/lib/libtvm_ffi.so" ]; then
        found="$(find build -type f -name 'libtvm_ffi.so*' | head -n1 || true)"
        if [ -n "$found" ]; then
          cp -v "$found" $out/lib/
        fi
      fi

      cp -r --no-preserve=mode,ownership include/. $out/include/
      chmod -R u+w $out/include

      if [ -d 3rdparty/tvm-ffi/include ]; then
        cp -r --no-preserve=mode,ownership 3rdparty/tvm-ffi/include/. $out/include/
      fi

      # DLPack headers are a TVM dependency used by the runtime C API.
      if [ -d 3rdparty/dlpack/include/dlpack ]; then
        mkdir -p $out/include/dlpack
        cp -r 3rdparty/dlpack/include/dlpack/. $out/include/dlpack/
      elif [ -d 3rdparty/tvm-ffi/3rdparty/dlpack/include/dlpack ]; then
        mkdir -p $out/include/dlpack
        cp -r 3rdparty/tvm-ffi/3rdparty/dlpack/include/dlpack/. $out/include/dlpack/
      fi

      # DMLC headers are required by TVM C++ APIs.
      if [ -d 3rdparty/dmlc-core/include/dmlc ]; then
        mkdir -p $out/include/dmlc
        cp -r 3rdparty/dmlc-core/include/dmlc/. $out/include/dmlc/
      fi

      # Build rpath for TVM libs.
      # - When using custom LLVM (shared), include it in rpath so libtvm.so finds libLLVM.so.
      # - Always include LLVM's runtime deps (zlib, ncurses, libxml2).
      # - Note: libcuda.so.1 is provided by the driver and resolved via autoAddDriverRunpath or LD_LIBRARY_PATH.
      rpath="\$ORIGIN:${lib.makeLibraryPath (
        [stdenv.cc.cc.lib zlib ncurses libxml2]
        ++ lib.optionals useCustomLlvm [llvm]
        ++ lib.optionals cudaEnabled [cudaPackages.cudatoolkit]
      )}"
      for f in $out/lib/*.so*; do
        [ -e "$f" ] || continue
        patchelf --set-rpath "$rpath" "$f"
      done

      ${lib.optionalString devel ''
        # Include Python bindings for development/testing
        echo "Including Python bindings (devel mode)"
        mkdir -p $out/python

        # Copy main TVM Python package
        cp -r python/tvm $out/python/

        # Copy tvm_ffi from 3rdparty
        if [ -d 3rdparty/tvm-ffi/python/tvm_ffi ]; then
          cp -r 3rdparty/tvm-ffi/python/tvm_ffi $out/python/
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
