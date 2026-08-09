{zigradSrc}: let
  shellValue = name: "\${${name}}";
  shellIsSet = name: "\${${name}+x}";
  shellHasValue = name: "\${${name}:+ }";
in {
  mkZigApplication = {
    pkgs,
    configuration,
    mainProgram,
    pname,
    src,
    sourceSubdir ? ".",
  }:
    pkgs.callPackage ../packages/zigrad-zig-example.nix {
      inherit
        configuration
        mainProgram
        pname
        src
        sourceSubdir
        zigradSrc
        ;
    };

  mkDevShell = {
    pkgs,
    configuration,
  }: let
    inherit (pkgs) lib;
    externalInputs = configuration.externalInputs;
    runtimeInputs = externalInputs.runtime;
    runtimePolicy = configuration.configuration.runtimeEnvPolicy;
    runtimeLibraryPaths =
      [
        "${runtimeInputs}/lib"
        "${runtimeInputs}/runtime/sys/lib"
      ]
      ++ lib.optional
      (lib.elem "cuda-driver" configuration.configuration.resolved)
      "/run/opengl-driver/lib";
    defaultHooks =
      lib.mapAttrsToList
      (name: value: ''
        if [ -z "${shellIsSet name}" ]; then
          export ${name}=${lib.escapeShellArg (toString value)}
        fi
      '')
      runtimePolicy.defaults;
    prefixHooks =
      lib.mapAttrsToList
      (name: value: ''
        export ${name}=${lib.escapeShellArg (toString value)}"${shellHasValue name}${shellValue name}"
      '')
      runtimePolicy.prefixes;
  in
    pkgs.mkShellNoCC {
      packages = [
        pkgs.stdenv.cc
        pkgs.zig
        pkgs.zls
        externalInputs
      ];

      env =
        runtimePolicy.fixed
        // {
          ZG_EXTERNAL_SDK_ROOT = toString externalInputs;
          ZG_RUNTIME_SDK_ROOT = toString runtimeInputs;
          ZG_ZIG_BUILD_ARGS = configuration.configuration.zigFeatureFlags;
        };

      shellHook = lib.concatStrings (
        defaultHooks
        ++ prefixHooks
        ++ [
          ''
            export LD_LIBRARY_PATH=${lib.escapeShellArg (lib.concatStringsSep ":" runtimeLibraryPaths)}"''${LD_LIBRARY_PATH:+:}''${LD_LIBRARY_PATH:-}"
            unset NIX_CFLAGS_COMPILE
          ''
        ]
      );
    };
}
