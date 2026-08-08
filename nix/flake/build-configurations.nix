{
  lib,
  pkgs,
  zigradSrc,
  version,
  cudaArchitectures ? [],
  parts,
}: let
  featureDefaults = {
    withPjrt = false;
    withMlir = false;
    withTvm = false;
    withMirage = false;
    withIree = false;
    withNvrtc = false;
    withCudaRuntime = false;
  };

  nodes = {
    "cuda-runtime" = {
      requires = ["cuda-driver"];
      compile = [];
      runtime = [parts.cudaRuntime];
      features.withCudaRuntime = true;
    };

    "cuda-driver" = {
      requires = [];
      compile = [];
      runtime = [];
      features = {};
    };

    nvrtc = {
      requires = ["cuda-runtime"];
      compile = [parts.cudaToolkit];
      runtime = [];
      features.withNvrtc = true;
    };

    "stablehlo-mlir" = {
      requires = [];
      compile = [parts.xlaMlirStablehloCapiSdk.dev];
      runtime = [];
      features.withMlir = true;
      compatibility = [
        {
          group = "host-llvm";
          consumer = "stablehlo-mlir";
          requirement = "xla-llvm";
          isolation = "in-process";
        }
      ];
    };

    "pjrt-api" = {
      requires = [];
      compile = [
        parts.pjrtHeaders
        parts.xlaProtos
      ];
      runtime = [];
      features.withPjrt = true;
    };

    "pjrt-cpu" = {
      requires = ["pjrt-api"];
      compile = [];
      runtime = [parts.xlaPjrtPlugins];
      features = {};
      conflicts = ["pjrt-cuda"];
    };

    "pjrt-cuda" = {
      requires = [
        "cuda-runtime"
        "pjrt-api"
      ];
      compile = [];
      runtime = [
        parts.xlaCudaRuntime
        parts.xlaPjrtPluginsCuda
      ];
      features = {};
      conflicts = ["pjrt-cpu"];
    };

    iree = {
      requires = [];
      compile = [parts.ireeRuntime];
      runtime = [parts.ireeCompiler];
      features.withIree = true;
      compatibility = [
        {
          group = "iree-build-llvm";
          consumer = "iree";
          requirement = "iree-llvm";
          isolation = "out-of-process";
        }
      ];
    };

    "tvm-cpu" = {
      requires = [];
      compile = [parts.tvmCpu.dev];
      runtime = [parts.tvmCpu];
      features.withTvm = true;
      compatibility = [
        {
          group = "host-llvm";
          consumer = "tvm";
          requirement = "tvm-llvm";
          isolation = "in-process";
        }
      ];
      conflicts = [
        "tvm-cuda"
        "tvm-python-cuda"
      ];
    };

    "tvm-cuda" = {
      requires = ["nvrtc"];
      compile = [parts.tvm.dev];
      runtime = [parts.tvm];
      features.withTvm = true;
      compatibility = [
        {
          group = "host-llvm";
          consumer = "tvm";
          requirement = "tvm-llvm";
          isolation = "in-process";
        }
      ];
      conflicts = [
        "tvm-cpu"
        "tvm-python-cuda"
      ];
    };

    "tvm-python-cuda" = {
      requires = ["nvrtc"];
      compile = [parts.tvmFullDev.dev];
      runtime = [parts.tvmFullDev];
      features.withTvm = true;
      compatibility = [
        {
          group = "host-llvm";
          consumer = "tvm";
          requirement = "tvm-llvm";
          isolation = "in-process";
        }
      ];
      conflicts = [
        "tvm-cpu"
        "tvm-cuda"
      ];
    };

    "mirage-cuda" = {
      requires = [
        "cuda-driver"
        "nvrtc"
      ];
      compile = [
        parts.mirage.dev
        parts.mirageAdapter.dev
      ];
      runtime = [parts.mirageAdapter];
      features.withMirage = true;
    };

    "xla-cpu-execution" = {
      requires = [
        "pjrt-cpu"
        "stablehlo-mlir"
      ];
      compile = [];
      runtime = [];
      features = {};
    };

    "xla-cuda-execution" = {
      requires = [
        "pjrt-cuda"
        "stablehlo-mlir"
      ];
      compile = [];
      runtime = [];
      features = {};
    };

    "iree-cpu-execution" = {
      requires = [
        "iree"
        "stablehlo-mlir"
      ];
      compile = [];
      runtime = [];
      features = {};
    };

    "tvm-xla-cpu-execution" = {
      requires = [
        "tvm-cpu"
        "xla-cpu-execution"
      ];
      compile = [];
      runtime = [];
      features = {};
    };

    "tvm-xla-cuda-execution" = {
      requires = [
        "tvm-cuda"
        "xla-cuda-execution"
      ];
      compile = [];
      runtime = [];
      features = {};
    };

    "mirage-xla-cuda-execution" = {
      requires = [
        "mirage-cuda"
        "xla-cuda-execution"
      ];
      compile = [];
      runtime = [];
      features = {};
    };
  };

  definitions = {
    zigrad = {
      demands = [];
      description = "Zigrad without external compiler or runtime integrations";
      packageName = "zigrad";
    };

    xla-cpu = {
      demands = ["xla-cpu-execution"];
      description = "Zigrad with the current XLA and PJRT CPU execution path";
    };

    xla-cuda = {
      demands = ["xla-cuda-execution"];
      description = "Zigrad with the current XLA and PJRT CUDA execution path";
    };

    iree-cpu = {
      demands = ["iree-cpu-execution"];
      description = "Zigrad with the current IREE CPU compile and execute path";
    };

    tvm-cpu = {
      demands = ["tvm-cpu"];
      description = "Zigrad with standalone TVM CPU tuning and execution";
    };

    tvm-cuda = {
      demands = ["tvm-cuda"];
      description = "Zigrad with standalone TVM CUDA tuning and execution";
    };

    tvm-xla-cpu = {
      demands = ["tvm-xla-cpu-execution"];
      description = "Zigrad with TVM CPU specialization on the current XLA path";
    };

    tvm-xla-cuda = {
      demands = ["tvm-xla-cuda-execution"];
      description = "Zigrad with TVM CUDA specialization on the current XLA path";
    };

    mirage-xla-cuda = {
      demands = ["mirage-xla-cuda-execution"];
      description = "Zigrad with Mirage CUDA specialization on the current XLA path";
    };

    dev-cuda = {
      demands = [
        "iree-cpu-execution"
        "mirage-xla-cuda-execution"
        "tvm-xla-cuda-execution"
      ];
      description = "Broad Zigrad development build with every current CUDA integration";
    };

    dev-cuda-tvm-python = {
      demands = [
        "iree-cpu-execution"
        "mirage-xla-cuda-execution"
        "tvm-python-cuda"
      ];
      description = "Broad CUDA development build with TVM Python bindings";
      expose = false;
    };

    example-mnist = {
      demands = [
        "iree-cpu-execution"
        "xla-cuda-execution"
      ];
      description = "Dependencies for the MNIST example";
      expose = false;
      packageName = "zigrad-example-mnist-dependencies";
    };

    example-benchmark = {
      demands = [
        "iree-cpu-execution"
        "tvm-xla-cuda-execution"
      ];
      description = "Dependencies for the benchmark example";
      expose = false;
      packageName = "zigrad-example-benchmark-dependencies";
    };
  };

  nodeFor = name:
    nodes.${name}
    or (throw "unknown Zigrad build dependency '${name}'");

  normalize = names:
    lib.sort builtins.lessThan (lib.unique names);

  closeDemands = demands: let
    current = normalize demands;
    expanded = normalize (
      current
      ++ lib.concatMap (name: (nodeFor name).requires or []) current
    );
  in
    if expanded == current
    then current
    else closeDemands expanded;

  conflictFor = resolved:
    lib.findFirst
    (entry: entry != null)
    null
    (lib.concatMap
      (name:
        map
        (other:
          if lib.elem other resolved
          then "${name} conflicts with ${other}"
          else null)
        ((nodeFor name).conflicts or []))
      resolved);

  mkExternalInputs = packageName: resolved: let
    compilePaths = lib.unique (
      lib.concatMap (node: (nodeFor node).compile or []) resolved
    );
    runtimePaths = lib.unique (
      lib.concatMap (node: (nodeFor node).runtime or []) resolved
    );
  in rec {
    compile = pkgs.symlinkJoin {
      name = "${packageName}-external-compile-inputs";
      paths = compilePaths;
    };

    runtime = pkgs.symlinkJoin {
      name = "${packageName}-external-runtime-inputs";
      paths = runtimePaths;
    };

    combined = pkgs.symlinkJoin {
      name = "${packageName}-external-inputs";
      paths = [
        compile
        runtime
      ];
      passthru = {
        inherit compile runtime;
      };
    };

    inherit compilePaths runtimePaths;
  };

  mkConfiguration = name: definition: let
    resolved = closeDemands definition.demands;
    conflict = conflictFor resolved;
    packageName = definition.packageName or "zigrad-${name}";
    externalInputs = mkExternalInputs packageName resolved;
    has = node: lib.elem node resolved;
    featureArgs =
      lib.foldl'
      (features: node: features // ((nodeFor node).features or {}))
      featureDefaults
      resolved;
    zigFeatureArgs = [
      "-Dpjrt=${lib.boolToString featureArgs.withPjrt}"
      "-Dmlir=${lib.boolToString featureArgs.withMlir}"
      "-Dtvm=${lib.boolToString featureArgs.withTvm}"
      "-Dmirage=${lib.boolToString featureArgs.withMirage}"
      "-Diree=${lib.boolToString featureArgs.withIree}"
      "-Dnvrtc=${lib.boolToString featureArgs.withNvrtc}"
      "-Dcuda-runtime=${lib.boolToString featureArgs.withCudaRuntime}"
    ];
    zigFeatureFlags = lib.concatStringsSep " " zigFeatureArgs;
    compatibility = lib.unique (
      lib.concatMap (node: (nodeFor node).compatibility or []) resolved
    );
    runtimeEnv =
      lib.optionalAttrs (has "pjrt-cpu" || has "pjrt-cuda") {
        PJRT_PLUGIN_PATH =
          if has "pjrt-cuda"
          then "${externalInputs.runtime}/runtime/xla/pjrt/c/pjrt_c_api_gpu_plugin.so"
          else "${externalInputs.runtime}/runtime/xla/pjrt/c/pjrt_c_api_cpu_plugin.so";
      }
      // lib.optionalAttrs (has "tvm-cpu" || has "tvm-cuda" || has "tvm-python-cuda") {
        ZG_ELF_LINKER_PATH = "${parts.llvm}/bin/ld.lld";
      }
      // lib.optionalAttrs (has "iree") {
        ZG_IREE_COMPILER_PATH = "${parts.ireeCompiler}/bin/iree-compile";
      }
      // lib.optionalAttrs (has "nvrtc") {
        CUDA_HOME = "${parts.cudaToolkit}";
        NIX_GLIBC_INCLUDE = "${pkgs.stdenv.cc.libc.dev}/include";
        NIX_GCC_INCLUDE = "${pkgs.stdenv.cc.cc}/lib/gcc/${pkgs.stdenv.hostPlatform.config}/${lib.getVersion pkgs.stdenv.cc.cc}/include";
      }
      // lib.optionalAttrs (has "mirage-cuda") {
        ZG_EXTERNAL_SDK_ROOT = "${externalInputs.combined}";
        ZG_MIRAGE_INCLUDE_DIR = "${parts.mirage.dev}/include";
      };
    runtimeEnvPrefixes = lib.optionalAttrs (has "pjrt-cuda") {
      XLA_FLAGS = "--xla_gpu_cuda_data_dir=${externalInputs.runtime}/runtime/nvidia";
    };
    effectiveRuntimeEnv =
      runtimeEnv
      // runtimeEnvPrefixes;
    package = pkgs.callPackage ../packages/zigrad.nix {
      inherit
        packageName
        runtimeEnv
        runtimeEnvPrefixes
        version
        zigradSrc
        ;
      compileInputs = externalInputs.compile;
      inherit zigFeatureArgs;
      needsPjrtDependencies = featureArgs.withPjrt;
      needsCudaDriverRunpath = has "cuda-driver";
      runtimeLibraryPaths = lib.optionals (externalInputs.runtimePaths != []) [
        "${externalInputs.runtime}/lib"
        "${externalInputs.runtime}/runtime/sys/lib"
      ];
      runtimeInputs =
        if externalInputs.runtimePaths == []
        then null
        else externalInputs.runtime;
      optimize = "ReleaseFast";
      passthru = {
        configuration = {
          inherit
            compatibility
            name
            packageName
            resolved
            zigFeatureArgs
            zigFeatureFlags
            ;
          runtimeEnv = effectiveRuntimeEnv;
          runtimeEnvPolicy = {
            fixed = runtimeEnv;
            defaults = {};
            prefixes = runtimeEnvPrefixes;
          };
          demands = definition.demands;
        };
        externalInputs = externalInputs.combined;
      };
    };
  in
    assert lib.assertMsg (conflict == null)
    "Zigrad build configuration '${name}' is invalid: ${conflict}"; {
      inherit
        compatibility
        featureArgs
        name
        package
        packageName
        resolved
        externalInputs
        zigFeatureArgs
        zigFeatureFlags
        ;
      runtimeEnv = effectiveRuntimeEnv;
      runtimeEnvPolicy = {
        fixed = runtimeEnv;
        defaults = {};
        prefixes = runtimeEnvPrefixes;
      };
      inherit (definition) demands description;
      expose = definition.expose or true;
    };

  configurations = lib.mapAttrs mkConfiguration definitions;

  manifest =
    lib.mapAttrs
    (_: configuration: {
      inherit
        (configuration)
        compatibility
        demands
        description
        expose
        packageName
        resolved
        zigFeatureFlags
        ;
    })
    configurations;
in {
  inherit
    configurations
    definitions
    manifest
    nodes
    ;
}
