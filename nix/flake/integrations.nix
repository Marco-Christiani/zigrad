# nix/flake/integrations.nix
#
# External integration derivations and demand-driven build configurations.
{inputs, ...}: let
  zigradVersion = inputs.self.shortRev or inputs.self.dirtyShortRev or "dev";
in {
  perSystem = {
    pkgs,
    cudaCfg,
    buildCfg,
    ...
  }: let
    # Shared settings for source builds that can demand long-running derivations.
    #
    # Definitions and defaults live in flake.nix.
    inherit
      (buildCfg)
      withDebugSymbols
      withNativeTuning
      cudaArchitectures
      enableLto
      extraCxxFlags
      extraLdFlags
      extraBazelFlags
      ;
    inherit (inputs) xlaSrc llvmSrc stablehloSrc;
    inherit (inputs) ireeSrc ireeLlvmSrc ireeStablehloSrc ireeFlatccSrc ireeBenchmarkSrc;
    inherit (pkgs) lib;

    zigradSrc = import ../helpers/source-filter.nix {
      inherit (pkgs) lib;
      root = ../..;
    };

    zigradAutodoc = pkgs.callPackage ../packages/zigrad-autodoc.nix {
      inherit zigradSrc;
    };

    cuda = import ./integrations/cuda.nix {
      inherit pkgs cudaCfg;
    };
    inherit (cuda) cudaRedist cudaToolkit gccHost;

    llvm = import ./compiler-support/llvm.nix {
      inherit
        pkgs
        xlaSrc
        llvmSrc
        withDebugSymbols
        enableLto
        extraCxxFlags
        extraLdFlags
        ;
    };

    xla = import ./integrations/xla.nix {
      inherit
        pkgs
        lib
        cudaCfg
        cudaArchitectures
        xlaSrc
        stablehloSrc
        llvm
        withDebugSymbols
        withNativeTuning
        enableLto
        extraCxxFlags
        extraLdFlags
        extraBazelFlags
        ;
    };
    inherit
      (xla)
      pjrtHeaders
      xlaMlirStablehloCapiSdk
      xlaPjrtPlugins
      xlaPjrtPluginsCuda
      xlaProtos
      zigradMlirExt
      ;

    mirageParts = import ./integrations/mirage.nix {
      inherit
        pkgs
        lib
        cudaToolkit
        gccHost
        cudaArchitectures
        withDebugSymbols
        withNativeTuning
        enableLto
        extraCxxFlags
        extraLdFlags
        ;
      cudaRuntime = cudaRedist.out;
    };
    inherit (mirageParts) mirage mirageAdapter mirageRustLibs;

    tvmParts = import ./integrations/tvm.nix {
      inherit
        pkgs
        cudaToolkit
        gccHost
        llvm
        cudaArchitectures
        withDebugSymbols
        withNativeTuning
        enableLto
        extraCxxFlags
        extraLdFlags
        ;
      cudaRuntime = cudaRedist.out;
    };
    inherit (tvmParts) tvm tvmCpu tvmFullDev;

    iree = import ./integrations/iree.nix {
      inherit
        pkgs
        ireeSrc
        ireeLlvmSrc
        ireeStablehloSrc
        ireeFlatccSrc
        ireeBenchmarkSrc
        withDebugSymbols
        withNativeTuning
        enableLto
        extraCxxFlags
        extraLdFlags
        ;
    };
    inherit (iree) ireeCompiler ireeLlvm ireeRuntime;

    buildGraph = import ./build-configurations.nix {
      inherit
        lib
        pkgs
        zigradSrc
        ;
      version = zigradVersion;
      inherit cudaArchitectures;
      parts = {
        inherit
          cudaRedist
          cudaToolkit
          ireeCompiler
          ireeRuntime
          llvm
          mirage
          mirageAdapter
          pjrtHeaders
          tvm
          tvmCpu
          tvmFullDev
          xlaMlirStablehloCapiSdk
          xlaPjrtPlugins
          xlaPjrtPluginsCuda
          xlaProtos
          ;
      };
    };

    buildConfigurations = buildGraph.configurations;
    configurationPackages =
      lib.concatMapAttrs
      (_: configuration:
        lib.optionalAttrs configuration.expose {
          "${configuration.packageName}" = configuration.package;
        })
      buildConfigurations;
    configurationApps =
      lib.mapAttrs'
      (_: configuration:
        lib.nameValuePair configuration.packageName {
          type = "app";
          program = "${configuration.package}/bin/zigrad";
          meta.description = configuration.description;
        })
      (lib.filterAttrs (_: configuration: configuration.expose) buildConfigurations);
    buildConfigurationManifest = pkgs.writeText "zigrad-build-configurations.json" (
      builtins.toJSON buildGraph.manifest
    );
  in {
    _module.args.zigradBuildConfigurations = buildConfigurations;

    packages =
      configurationPackages
      // {
        mirage-adapter = mirageAdapter;
        mirage-adapter-dev = mirageAdapter.dev;
        inherit mirage;
        mirage-dev = mirage.dev;
        mirage-abstract-subexpr = mirageRustLibs.abstract_subexpr;
        mirage-formal-verifier = mirageRustLibs.formal_verifier;
        zigrad-build-configurations = buildConfigurationManifest;
        zigrad-autodoc = zigradAutodoc;
        inherit llvm tvm;
        tvm-dev = tvm.dev;
        tvm-cpu = tvmCpu;
        tvm-full-dev = tvmFullDev.dev;
        cuda-redist = cudaRedist;
        cuda-redist-dev = cudaRedist.dev;
        xla-mlir-stablehlo-capi-sdk = xlaMlirStablehloCapiSdk;
        xla-mlir-stablehlo-capi-sdk-dev = xlaMlirStablehloCapiSdk.dev;
        xla-pjrt-plugins = xlaPjrtPlugins;
        xla-pjrt-plugins-cuda = xlaPjrtPluginsCuda;
        iree-llvm = ireeLlvm;
        iree-compiler = ireeCompiler;
        iree-runtime = ireeRuntime;
        pjrt-headers = pjrtHeaders;
        # Disconnected C++ dialect, pass, and language-server tooling.
        zigrad-mlir-ext = zigradMlirExt;
        zigrad-mlir-ext-dev = zigradMlirExt.dev;
      };

    apps = configurationApps;
  };
}
