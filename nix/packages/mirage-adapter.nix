{
  lib,
  stdenv,
  cmake,
  autoAddDriverRunpath,
  autoPatchelfHook,
  gccHost,
  cudaToolkit,
  cudaRuntime,
  nlohmann_json,
  mirage,
  src,
  withDebugSymbols ? false,
  enableLto ? false,
  extraCxxFlags ? [],
  extraLdFlags ? [],
}: let
  buildType =
    if withDebugSymbols
    then "RelWithDebInfo"
    else "Release";
in
  stdenv.mkDerivation {
    pname = "zigrad-mirage-adapter";
    version = "1";

    inherit src;
    outputs = ["out" "dev"];
    strictDeps = true;
    dontStrip = withDebugSymbols;

    nativeBuildInputs = [
      cmake
      autoPatchelfHook
      autoAddDriverRunpath
      gccHost
    ];

    buildInputs = [
      cudaRuntime
      cudaToolkit
      nlohmann_json
      mirage
      mirage.dev
    ];

    cmakeFlags = [
      "-DCMAKE_BUILD_TYPE=${buildType}"
      "-DCMAKE_CXX_COMPILER=${gccHost}/bin/g++"
      "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=${
        if enableLto
        then "ON"
        else "OFF"
      }"
      "-DCMAKE_CXX_FLAGS=${lib.concatStringsSep " " extraCxxFlags}"
      "-DCMAKE_SHARED_LINKER_FLAGS=${lib.concatStringsSep " " extraLdFlags}"
      "-DCMAKE_INSTALL_INCLUDEDIR=${placeholder "dev"}/include"
      "-DCMAKE_INSTALL_LIBDIR=lib"
      "-DCUDAToolkit_ROOT=${cudaToolkit}"
      "-DMIRAGE_INCLUDE_DIR=${mirage.dev}/include"
      "-DMIRAGE_RUNTIME_LIBRARY=${mirage}/lib/libmirage_runtime.so"
      "-DNLOHMANN_JSON_INCLUDE_DIR=${lib.getDev nlohmann_json}/include"
    ];

    meta = {
      description = "Zigrad-owned C adapter for Mirage";
      license = lib.licenses.asl20;
      platforms = lib.platforms.linux;
    };
  }
