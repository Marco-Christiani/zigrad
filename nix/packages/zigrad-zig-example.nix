{
  autoAddDriverRunpath,
  autoPatchelfHook,
  callPackage,
  lib,
  makeWrapper,
  stdenvNoCC,
  zig,
  exampleName,
  mainProgram,
  profile,
  src,
}: let
  externalInputs = profile.externalInputs.combined;
  runtimeInputs = externalInputs.runtime;
  runtimePolicy = profile.runtimeEnvPolicy;
  runtimeLibraryPaths = [
    "${runtimeInputs}/lib"
    "${runtimeInputs}/runtime/sys/lib"
  ];
  runtimeLibraryPath = lib.concatStringsSep ":" runtimeLibraryPaths;
  needsCudaDriverRunpath = lib.elem "cuda-driver" profile.resolved;
  zigDeps = callPackage ./zig-dependencies.nix {
    withPjrt = true;
  };
  runtimeWrapperArgs = lib.concatStringsSep " \\\n" (
    (lib.mapAttrsToList (
        name: value: "--set ${lib.escapeShellArg name} ${lib.escapeShellArg (toString value)}"
      )
      runtimePolicy.fixed)
    ++ (lib.mapAttrsToList (
        name: value: "--set-default ${lib.escapeShellArg name} ${lib.escapeShellArg (toString value)}"
      )
      runtimePolicy.defaults)
    ++ (lib.mapAttrsToList (
        name: value: "--prefix ${lib.escapeShellArg name} ' ' ${lib.escapeShellArg (toString value)}"
      )
      runtimePolicy.prefixes)
    ++ ["--prefix LD_LIBRARY_PATH : ${lib.escapeShellArg runtimeLibraryPath}"]
  );
in
  stdenvNoCC.mkDerivation {
    pname = "zigrad-example-${exampleName}";
    version = "dev";

    inherit src;

    strictDeps = true;

    nativeBuildInputs =
      [
        zig
        autoPatchelfHook
        makeWrapper
      ]
      ++ lib.optional needsCudaDriverRunpath autoAddDriverRunpath;

    buildInputs = [externalInputs];
    runtimeDependencies = [runtimeInputs];

    configurePhase = ''
      runHook preConfigure
      export ZIG_GLOBAL_CACHE_DIR=$(mktemp -d)
      runHook postConfigure
    '';

    buildPhase = ''
      runHook preBuild
      cd "examples/${exampleName}"
      TERM=dumb zig build \
        -j"$NIX_BUILD_CORES" \
        -Doptimize=ReleaseFast \
        -Dsdk=${externalInputs} \
        -Dtarget=native-native-gnu \
        --system ${zigDeps} \
        --verbose
      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      TERM=dumb zig build install \
        -j"$NIX_BUILD_CORES" \
        -Doptimize=ReleaseFast \
        -Dsdk=${externalInputs} \
        -Dtarget=native-native-gnu \
        --system ${zigDeps} \
        --prefix "$out" \
        --verbose
      runHook postInstall
    '';

    postFixup =
      lib.optionalString needsCudaDriverRunpath ''
        addDriverRunpath "$out/bin/${mainProgram}"
      ''
      + ''
        wrapProgram "$out/bin/${mainProgram}" \
          ${runtimeWrapperArgs}
      '';

    meta = {
      description = "Zigrad ${exampleName} example";
      license = lib.licenses.asl20;
      inherit mainProgram;
      platforms = lib.platforms.linux;
    };
  }
