# Package a Zig executable that imports Zigrad with one named configuration.
{
  autoAddDriverRunpath,
  autoPatchelfHook,
  callPackage,
  lib,
  makeWrapper,
  stdenvNoCC,
  zig,
  configuration,
  mainProgram,
  pname,
  src,
  sourceSubdir ? ".",
  zigradSrc,
  usePackagedZigrad ? false,
  externalInputs ? configuration.externalInputs,
  runtimePolicy ? configuration.configuration.runtimeEnvPolicy,
  targetPkgs ? null,
  zigArgs ? [],
  optimize ? "ReleaseFast",
  runTests ? false,
  testProgram ? null,
  withRuntimeEnvironment ? true,
}: let
  runtimeInputs = externalInputs.runtime;
  zigTarget =
    if targetPkgs == null
    then "native-native-gnu"
    else
      lib.concatStringsSep "-" (with targetPkgs.stdenv.hostPlatform.parsed; [
        cpu.name
        kernel.name
        abi.name
      ]);
  runtimeLibraryPaths = [
    "${runtimeInputs}/lib"
    "${runtimeInputs}/runtime/sys/lib"
  ];
  runtimeLibraryPath = lib.concatStringsSep ":" runtimeLibraryPaths;
  needsCudaDriverRunpath = lib.elem "cuda-driver" configuration.configuration.resolved;
  zigDeps = callPackage ./zig-dependencies.nix {
    withPjrt =
      lib.elem "pjrt-cpu" configuration.configuration.resolved
      || lib.elem "pjrt-cuda" configuration.configuration.resolved;
  };
  zigFeatureArgs = lib.escapeShellArgs configuration.configuration.zigFeatureArgs;
  testProgramName =
    if testProgram != null
    then testProgram
    else "${mainProgram}-tests";
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
    inherit pname;
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

      export ZG_ZIG_SYSTEM_PACKAGES="$TMPDIR/zigrad-system-packages"
      mkdir -p "$ZG_ZIG_SYSTEM_PACKAGES"
      cp -a ${zigDeps}/. "$ZG_ZIG_SYSTEM_PACKAGES/"
      chmod u+w "$ZG_ZIG_SYSTEM_PACKAGES"

      ${
        if usePackagedZigrad
        then ''
          sed -i \
            '/^[[:space:]]*\.zigrad = \.{[[:space:]]*$/,/^[[:space:]]*},[[:space:]]*$/c\        .zigrad = .{ .path = "../.." },' \
            ${lib.escapeShellArg sourceSubdir}/build.zig.zon
        ''
        else ''
          zigrad_package_name="$(zig fetch ${zigradSrc})"
          test -n "$zigrad_package_name"
          ln -s ${zigradSrc} "$ZG_ZIG_SYSTEM_PACKAGES/$zigrad_package_name"
        ''
      }
      runHook postConfigure
    '';

    buildPhase = ''
      runHook preBuild
      export ZG_ZIG_SYSTEM_PACKAGES="$TMPDIR/zigrad-system-packages"
      cd ${lib.escapeShellArg sourceSubdir}
      TERM=dumb zig build \
        -j"$NIX_BUILD_CORES" \
        -Doptimize=${lib.escapeShellArg optimize} \
        -Dsdk=${externalInputs} \
        -Dtarget=${lib.escapeShellArg zigTarget} \
        ${zigFeatureArgs} \
        ${lib.escapeShellArgs zigArgs} \
        --system "$ZG_ZIG_SYSTEM_PACKAGES" \
        --verbose
      runHook postBuild
    '';

    checkPhase = ''
      runHook preCheck
      export ZG_ZIG_SYSTEM_PACKAGES="$TMPDIR/zigrad-system-packages"
      testRoot=$(mktemp -d)
      TERM=dumb zig build test-compile \
        -j"$NIX_BUILD_CORES" \
        -Doptimize=${lib.escapeShellArg optimize} \
        -Dsdk=${externalInputs} \
        -Dtarget=${lib.escapeShellArg zigTarget} \
        ${lib.escapeShellArgs zigArgs} \
        --system "$ZG_ZIG_SYSTEM_PACKAGES" \
        --prefix "$testRoot" \
        --verbose
      autoPatchelf "$testRoot/bin"
      "$testRoot/bin/${testProgramName}"
      runHook postCheck
    '';

    doCheck = runTests;

    installPhase = ''
      runHook preInstall
      export ZG_ZIG_SYSTEM_PACKAGES="$TMPDIR/zigrad-system-packages"
      TERM=dumb zig build install \
        -j"$NIX_BUILD_CORES" \
        -Doptimize=${lib.escapeShellArg optimize} \
        -Dsdk=${externalInputs} \
        -Dtarget=${lib.escapeShellArg zigTarget} \
        ${zigFeatureArgs} \
        ${lib.escapeShellArgs zigArgs} \
        --system "$ZG_ZIG_SYSTEM_PACKAGES" \
        --prefix "$out" \
        --verbose
      runHook postInstall
    '';

    postFixup =
      lib.optionalString (targetPkgs != null) ''
        patchelf --set-interpreter ${lib.escapeShellArg targetPkgs.stdenv.cc.bintools.dynamicLinker} "$out/bin/${mainProgram}"
      ''
      + lib.optionalString needsCudaDriverRunpath ''
        addDriverRunpath "$out/bin/${mainProgram}"
      ''
      + lib.optionalString withRuntimeEnvironment ''
        wrapProgram "$out/bin/${mainProgram}" \
          ${runtimeWrapperArgs}
      '';

    meta = {
      description = "Zigrad application ${pname}";
      # TODO(release): license, etc.
      license = lib.licenses.asl20;
      inherit mainProgram;
      platforms = lib.platforms.linux;
    };
  }
