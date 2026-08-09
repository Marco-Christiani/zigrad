# nix/flake/examples.nix
#
# Packages and development inputs for maintained examples.
{
  perSystem = {
    pkgs,
    config,
    repoRoot,
    zigradBuildConfigurations,
    ...
  }: let
    inherit (pkgs) lib;
    fs = lib.fileset;
    zigradSrc = import ../helpers/source-filter.nix {
      inherit lib;
      root = repoRoot;
    };
    zigExampleSrc = name:
      fs.toSource {
        root = repoRoot;
        fileset = fs.unions [
          (repoRoot + /build.zig)
          (repoRoot + /build.zig.zon)
          (repoRoot + /assets)
          (repoRoot + /src)
          (repoRoot + /tools)
          (repoRoot + "/examples/${name}")
        ];
      };
    mkZigExample = {
      name,
      mainProgram,
      profile,
    }:
      pkgs.callPackage ../packages/zigrad-zig-example.nix {
        configuration = profile.package;
        inherit mainProgram;
        pname = "zigrad-example-${name}";
        src = zigExampleSrc name;
        sourceSubdir = "examples/${name}";
        usePackagedZigrad = true;
        inherit zigradSrc;
      };
    benchmark = mkZigExample {
      name = "benchmark";
      mainProgram = "benchmark";
      profile = zigradBuildConfigurations.example-benchmark;
    };
    mnist = mkZigExample {
      name = "mnist";
      mainProgram = "mnist";
      profile = zigradBuildConfigurations.xla-iree-cpu;
    };
    mlirCpp = pkgs.callPackage ../packages/zigrad-mlir-cpp-example.nix {
      xlaMlirStablehloCapiSdk = config.packages.xla-mlir-stablehlo-capi-sdk;
      llvm = config.packages.llvm;
      src = let
        fs = lib.fileset;
      in
        fs.toSource {
          root = repoRoot + /examples/mlir-cpp;
          fileset = fs.unions [
            (repoRoot + /examples/mlir-cpp/CMakeLists.txt)
            (repoRoot + /examples/mlir-cpp/mlir_ext.cc)
            (repoRoot + /examples/mlir-cpp/zigrad)
            (repoRoot + /examples/mlir-cpp/dev)
            (repoRoot + /examples/mlir-cpp/test)
          ];
        };
    };
    configureMlirCpp = pkgs.writeShellApplication {
      name = "configure-mlir-cpp";
      runtimeInputs = [pkgs.cmake pkgs.git pkgs.ninja];
      text = ''
        root="$(git rev-parse --show-toplevel)"

        cmake \
          -S "$root/examples/mlir-cpp" \
          -B "$root/examples/mlir-cpp/build" \
          -G Ninja \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          -DZG_ENABLE_DEV_TARGETS=ON \
          -DZG_SDK_INCLUDE=${config.packages.xla-mlir-stablehlo-capi-sdk.dev}/include \
          -DZG_SDK_LIB=${config.packages.xla-mlir-stablehlo-capi-sdk.out}/lib \
          -DZG_LLVM_ROOT=${config.packages.llvm} \
          -DZG_INSTALL_RPATH="${config.packages.xla-mlir-stablehlo-capi-sdk.out}/lib:${config.packages.llvm}/lib"

        cmake --build "$root/examples/mlir-cpp/build" --target zigrad_tblgen
      '';
    };
  in {
    packages = {
      configure-mlir-cpp = configureMlirCpp;
      zigrad-example-benchmark = benchmark;
      zigrad-example-mnist = mnist;
      zigrad-example-mlir-cpp = mlirCpp;
      zigrad-example-mlir-cpp-dev = mlirCpp.dev;
    };

    apps = {
      configure-mlir-cpp = {
        type = "app";
        program = "${configureMlirCpp}/bin/configure-mlir-cpp";
        meta.description = "Configure the local C++ MLIR example build";
      };
      zigrad-example-benchmark = {
        type = "app";
        program = "${benchmark}/bin/benchmark";
        meta.description = "Run the Zigrad backend benchmark";
      };
      zigrad-example-mnist = {
        type = "app";
        program = "${mnist}/bin/mnist";
        meta.description = "Run the Zigrad MNIST training example";
      };
    };
  };
}
