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
    pkgs.callPackage ../packages/zigrad-application.nix (
      {
        inherit
          (configuration)
          externalInputs
          needsCudaDriverRunpath
          runtimePolicy
          zigDependencySets
          ;
      }
      // {
        inherit
          mainProgram
          pname
          src
          sourceSubdir
          zigradSrc
          ;
      }
    );

  mkDevShell = {
    pkgs,
    configuration,
  }: let
    inherit (pkgs) lib;
    inherit (configuration) externalInputs runtimePolicy;
    runtimeInputs = externalInputs.runtime;
    runtimeLibraryPaths =
      [
        "${runtimeInputs}/lib"
        "${runtimeInputs}/runtime/sys/lib"
      ]
      ++ lib.optional
      configuration.needsCudaDriverRunpath
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
        externalInputs.combined
      ];

      env =
        runtimePolicy.fixed
        // {
          ZG_EXTERNAL_SDK_ROOT = toString externalInputs.combined;
          ZG_RUNTIME_SDK_ROOT = toString runtimeInputs;
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
