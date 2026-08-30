# Compile a Zigrad PR with IREE and embed the executable artifact in a Zig runner.
{
  lib,
  runCommand,
  mkZigApplication,
}: {
  name,
  description,
  source,
  sourceSubdir ? ".",
  zigradSrc,
  program,
  programName,
  vmfbName,
  compilerConfiguration,
  runnerConfiguration,
  backend,
  mainProgram,
  runnerArguments,
  compilerArguments ? [],
  compilerEnvironment ? {},
  embeddedFiles ? [],
  installCheckCommand ? null,
  runnerExternalInputs ? runnerConfiguration.externalInputs,
  targetPkgs ? null,
  withRuntimeEnvironment ? false,
  passthru ? {},
}: let
  vmfb = runCommand "${name}-vmfb" compilerEnvironment ''
    mkdir -p "$out"
    ${lib.getExe compilerConfiguration.package} --quiet iree compile \
      ${program}/${programName} \
      --target=${lib.escapeShellArg backend} \
      --output="$out/${vmfbName}" \
      -- \
      ${lib.escapeShellArgs compilerArguments}
  '';
  copyEmbeddedFiles =
    lib.concatMapStringsSep "\n" (file: ''
      cp ${file.source}/${file.sourceName} ${sourceSubdir}/src/${file.destinationName}
    '')
    embeddedFiles;
  runner = mkZigApplication runnerConfiguration {
    pname = name;
    src = source;
    inherit mainProgram sourceSubdir zigradSrc targetPkgs;
    externalInputs = runnerExternalInputs;
    zigArgs = runnerArguments;
    inherit withRuntimeEnvironment;
  };
in
  runner.overrideAttrs (old: {
    postPatch =
      (old.postPatch or "")
      + ''
        cp ${vmfb}/${vmfbName} ${sourceSubdir}/src/${vmfbName}
        ${copyEmbeddedFiles}
      '';

    doInstallCheck = installCheckCommand != null;
    installCheckPhase = lib.optionalString (installCheckCommand != null) ''
      runHook preInstallCheck
      ${installCheckCommand}
      runHook postInstallCheck
    '';

    passthru = (old.passthru or {}) // passthru // {inherit program vmfb;};
    meta = old.meta // {inherit description mainProgram;};
  })
