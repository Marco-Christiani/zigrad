{
  callPackage,
  lib,
  runCommand,
  mkZigradZigPackage,
  source,
  zigradSrc,
  emitterConfiguration,
}: let
  sourceSubdir = "examples/basic-deployment";
  programName = "model.zgpr";
  mkIreeEmbeddedExample = callPackage ../../nix/packages/iree-embedded-example.nix {
    inherit mkZigradZigPackage;
  };
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
in
  {
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
  }:
    mkIreeEmbeddedExample {
      name = "zigrad-example-basic-deployment-${name}";
      description = "Zigrad basic ${name} deployment example";
      vmfbName = "model.vmfb";
      inherit source sourceSubdir zigradSrc program programName backend;
      inherit compilerConfiguration compilerArguments compilerEnvironment;
      inherit runnerConfiguration runnerExternalInputs targetPkgs withRuntimeEnvironment;
      mainProgram = "basic-deployment";
      runnerArguments =
        ["-Dmode=run"]
        ++ lib.optional strip "-Dstrip=true"
        ++ runnerArguments;
      installCheckCommand =
        if canExecute
        then ''"$out/bin/basic-deployment"''
        else null;
    }
