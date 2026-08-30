{
  lib,
  flake-parts-lib,
  ...
}: let
  inherit (lib) mkOption types;

  componentType = types.submodule {
    options = {
      provider = mkOption {
        type = types.nonEmptyStr;
        description = "Compiler, kernel provider, or runtime implementation.";
      };

      target = mkOption {
        type = types.nonEmptyStr;
        description = "Execution target selected for the provider.";
      };
    };
  };

  configurationType = types.submodule {
    options = {
      compilers = mkOption {
        type = types.listOf componentType;
        default = [];
        description = "Whole-program compilers available in the configuration.";
      };

      kernelProviders = mkOption {
        type = types.listOf componentType;
        default = [];
        description = "Kernel providers available to specialize selected regions.";
      };

      runtimes = mkOption {
        type = types.listOf componentType;
        default = [];
        description = "Runtime implementations available to execute compiled artifacts.";
      };

      description = mkOption {
        type = types.nonEmptyStr;
        description = "User-facing summary of the configuration's direct intent.";
      };

      expose = mkOption {
        type = types.bool;
        default = true;
        description = "Publish the configuration as a package and app output.";
      };

      pname = mkOption {
        type = types.nullOr types.nonEmptyStr;
        default = null;
        description = "Internal derivation name override.";
      };
    };
  };

  buildOptionsType = types.submodule {
    options = {
      withDebugSymbols = mkOption {
        type = types.bool;
        default = false;
        description = "Retain debug information in external integration builds.";
      };

      withNativeTuning = mkOption {
        type = types.bool;
        default = false;
        description = "Allow host-specific CPU tuning in supported integration builds.";
      };

      cudaArchitectures = mkOption {
        type = types.listOf types.nonEmptyStr;
        default = [];
        description = "CUDA compute capabilities compiled by supported integrations.";
      };

      enableLto = mkOption {
        type = types.bool;
        default = false;
        description = "Enable link-time optimization in supported integration builds.";
      };

      extraCxxFlags = mkOption {
        type = types.listOf types.str;
        default = [];
        description = "Additional C++ compiler flags for integration experiments.";
      };

      extraLdFlags = mkOption {
        type = types.listOf types.str;
        default = [];
        description = "Additional linker flags for integration experiments.";
      };

      extraBazelFlags = mkOption {
        type = types.listOf types.str;
        default = [];
        description = "Additional Bazel flags for integration experiments.";
      };
    };
  };
in {
  options.perSystem = flake-parts-lib.mkPerSystemOption {
    options.zigrad = {
      build = mkOption {
        type = buildOptionsType;
        default = {};
        description = "Build policy applied to external integration derivations.";
      };

      configurations = mkOption {
        type = types.lazyAttrsOf configurationType;
        default = {};
        description = "Named Zigrad compiler, kernel-provider, and runtime compositions.";
      };

      resolvedConfigurations = mkOption {
        type = types.lazyAttrsOf types.raw;
        readOnly = true;
        internal = true;
        description = "Resolved build products derived from Zigrad configurations.";
      };
    };
  };
}
