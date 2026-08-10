{
  lib,
  runCommand,
  mkZigradZigPackage,
  source,
  zigradSrc,
  emitterConfiguration,
}: let
  sourceSubdir = "examples/basic-deployment";
  programName = "model.zgpr";
  vmfbName = "model.vmfb";

  emitter = mkZigradZigPackage {
    configuration = emitterConfiguration.package;
    mainProgram = "emit-pr";
    pname = "zigrad-example-basic-deployment-pr-emitter";
    src = source;
    inherit sourceSubdir zigradSrc;
    usePackagedZigrad = true;
    zigArgs = ["-Dmode=emit_pr"];
    withRuntimeEnvironment = false;
  };

  program = runCommand "zigrad-example-basic-deployment-program" {} ''
    mkdir -p "$out"
    ${lib.getExe emitter} "$out/${programName}"
  '';
in {
  name,
  backend,
  compilerConfiguration,
  runnerConfiguration,
  compilerArguments ? [],
  compilerEnvironment ? {},
  runnerArguments ? [],
  runnerExternalInputs ? runnerConfiguration.externalInputs.combined,
  targetPkgs ? null,
  strip ? false,
  withRuntimeEnvironment ? false,
  canExecute ? false,
}: let
  vmfb = runCommand "zigrad-example-basic-deployment-${name}-vmfb" compilerEnvironment ''
    mkdir -p "$out"
    ${lib.getExe compilerConfiguration.package} --quiet iree compile \
      ${program}/${programName} \
      --target=${lib.escapeShellArg backend} \
      --output="$out/${vmfbName}" \
      -- \
      ${lib.escapeShellArgs compilerArguments}
  '';

  runner = mkZigradZigPackage {
    configuration = runnerConfiguration.package;
    mainProgram = "basic-deployment";
    pname = "zigrad-example-basic-deployment-${name}";
    src = source;
    inherit sourceSubdir zigradSrc targetPkgs;
    externalInputs = runnerExternalInputs;
    usePackagedZigrad = true;
    zigArgs =
      ["-Dmode=run"]
      ++ lib.optional strip "-Dstrip=true"
      ++ runnerArguments;
    inherit withRuntimeEnvironment;
  };
in
  runner.overrideAttrs (old: {
    postPatch =
      (old.postPatch or "")
      + ''
        cp ${vmfb}/${vmfbName} ${sourceSubdir}/src/${vmfbName}
      '';

    doInstallCheck = canExecute;
    installCheckPhase = lib.optionalString canExecute ''
      runHook preInstallCheck
      "$out/bin/basic-deployment"
      runHook postInstallCheck
    '';

    passthru = (old.passthru or {}) // {inherit program vmfb;};
    meta =
      old.meta
      // {
        description = "Zigrad basic ${name} deployment example";
        mainProgram = "basic-deployment";
      };
  })
