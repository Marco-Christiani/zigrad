{
  lib,
  stdenv,
  cmake,
  autoAddDriverRunpath,
  autoPatchelfHook,
  fetchFromGitHub,
  gccHost,
  cudaToolkit,
  cudaRuntime,
  z3,
  nlohmann_json,
  mirageRustLibs,
  src,
  sourceRoot,
  revision,
  cudaArchitectures ? [],
  withDebugSymbols ? false,
  withNativeTuning ? false,
  enableLto ? false,
  extraCxxFlags ? [],
  extraLdFlags ? [],
}: let
  upstreamCudaArchitectures = ["75" "80" "86" "89" "90"];
  selectedCudaArchitectures =
    if cudaArchitectures == []
    then upstreamCudaArchitectures
    else cudaArchitectures;
  cudaArchStr = lib.concatStringsSep ";" selectedCudaArchitectures;
  buildType =
    if withDebugSymbols
    then "RelWithDebInfo"
    else "Release";
  cxxFlags =
    extraCxxFlags
    ++ lib.optional withNativeTuning "-march=native";

  cutlassSrc = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "cutlass";
    rev = "v4.2.1";
    hash = "sha256-iP560D5Vwuj6wX1otJhwbvqe/X4mYVeKTpK533Wr5gY=";
  };
in
  stdenv.mkDerivation {
    pname = "mirage";
    version = "0-unstable-2026-04-17";

    inherit src sourceRoot;
    patches = [../patches/mirage-transpile-metadata.patch];
    outputs = ["out" "dev"];
    strictDeps = true;

    nativeBuildInputs = [
      cmake
      autoPatchelfHook
      autoAddDriverRunpath
      gccHost
    ];

    buildInputs = [
      cudaRuntime
      cudaToolkit
      z3
      nlohmann_json
      mirageRustLibs.abstract_subexpr
      mirageRustLibs.formal_verifier
    ];

    # The NVIDIA driver provides libcuda.so.1 at runtime.
    autoPatchelfIgnoreMissingDeps = ["libcuda.so.1"];

    postPatch = ''
            rm -rf deps/cutlass deps/json
            ln -s ${cutlassSrc} deps/cutlass

            substituteInPlace cmake/cuda.cmake \
              --replace-fail \
                'set(CUDA_CUDA_LIBRARY ''${CUDAToolkit_LIBRARY_DIR}/libcuda.so)' \
                'set(CUDA_CUDA_LIBRARY ''${CUDAToolkit_LIBRARY_DIR}/stubs/libcuda.so)'

            substituteInPlace CMakeLists.txt \
              --replace-fail \
                'add_subdirectory(deps/json)' \
                'find_package(nlohmann_json REQUIRED)' \
              --replace-fail \
                'list(APPEND MIRAGE_LINK_LIBS abstract_subexpr)' \
                '# Rust library paths are appended below' \
              --replace-fail \
                'execute_process(COMMAND rustc --version' \
                'execute_process(COMMAND true #' \
              --replace-fail \
                'execute_process(COMMAND cargo --version' \
                'execute_process(COMMAND true #' \
              --replace-fail \
                'set(ABSTRACT_SUBEXPR_LIBRARIES ''${PROJECT_SOURCE_DIR}/build/abstract_subexpr/release/libabstract_subexpr.so)' \
                '# Rust libraries are supplied by the package' \
              --replace-fail \
                'set(FORMAL_VERIFIER_LIBRARIES ''${PROJECT_SOURCE_DIR}/build/formal_verifier/release/libformal_verifier.so)' \
                '# Rust libraries are supplied by the package' \
              --replace-fail \
                'set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")' \
                '# CMAKE_CUDA_ARCHITECTURES is supplied by the package' \
              --replace-fail \
                'set_target_properties(mirage_runtime PROPERTIES CUDA_ARCHITECTURES "75;80;86;89;90")' \
                '# CUDA_ARCHITECTURES is inherited from CMAKE_CUDA_ARCHITECTURES'

            substituteInPlace CMakeLists.txt \
              --replace-fail \
                'file(GLOB_RECURSE MIRAGE_SRCS
        src/*.cc
      )' \
                'file(GLOB_RECURSE MIRAGE_SRCS
        src/*.cc
      )
      list(REMOVE_ITEM MIRAGE_SRCS ''${PROJECT_SOURCE_DIR}/src/layout.cc)'

            substituteInPlace src/kernel/chunk.cc \
              --replace-fail \
                '#ifdef MIRAGE_FINGERPRINT_USE_CPU' \
                '#if 1'
    '';

    cmakeFlags = [
      "-DCMAKE_BUILD_TYPE=${buildType}"
      "-DBUILD_SHARED_LIBS=ON"
      "-DBUILD_CPP_EXAMPLES=OFF"
      "-DMIRAGE_BUILD_UNIT_TEST=OFF"
      "-DUSE_CUDA=ON"
      "-DUSE_FORMAL_VERIFIER=ON"
      "-DUSE_NKI=OFF"
      "-DCMAKE_C_COMPILER=${gccHost}/bin/gcc"
      "-DCMAKE_CXX_COMPILER=${gccHost}/bin/g++"
      "-DCMAKE_CUDA_COMPILER=${cudaToolkit}/bin/nvcc"
      "-DCMAKE_CUDA_HOST_COMPILER=${gccHost}/bin/g++"
      "-DCMAKE_CUDA_ARCHITECTURES=${cudaArchStr}"
      "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=${
        if enableLto
        then "ON"
        else "OFF"
      }"
      "-DCMAKE_CXX_FLAGS=${lib.concatStringsSep " " cxxFlags}"
      "-DCMAKE_SHARED_LINKER_FLAGS=${lib.concatStringsSep " " extraLdFlags}"
      "-DZ3_CXX_INCLUDE_DIRS=${lib.getDev z3}/include"
      "-DZ3_LIBRARIES=${lib.getLib z3}/lib/libz3.so"
      "-DABSTRACT_SUBEXPR_LIB=${mirageRustLibs.abstract_subexpr}/lib"
      "-DABSTRACT_SUBEXPR_LIBRARIES=${mirageRustLibs.abstract_subexpr}/lib/libabstract_subexpr.so"
      "-DFORMAL_VERIFIER_LIB=${mirageRustLibs.formal_verifier}/lib"
      "-DFORMAL_VERIFIER_LIBRARIES=${mirageRustLibs.formal_verifier}/lib/libformal_verifier.so"
    ];

    postInstall = ''
      install -Dm755 \
        ${mirageRustLibs.abstract_subexpr}/lib/libabstract_subexpr.so \
        "$out/lib/libabstract_subexpr.so"
      install -Dm755 \
        ${mirageRustLibs.formal_verifier}/lib/libformal_verifier.so \
        "$out/lib/libformal_verifier.so"

      mkdir -p "$dev/include"
      cp -r --no-preserve=mode,ownership \
        ${cutlassSrc}/include/. \
        "$dev/include/"
      cp -r --no-preserve=mode,ownership \
        ${cutlassSrc}/tools/util/include/. \
        "$dev/include/"
    '';

    passthru = {
      sourceRevision = revision;
      inherit selectedCudaArchitectures;
    };

    meta = {
      description = "Mirage symbolic superoptimizer and CUDA transpiler";
      # TODO(release): license, etc.
      license = lib.licenses.asl20;
      homepage = "https://github.com/mirage-project/mirage";
      platforms = lib.platforms.linux;
    };
  }
