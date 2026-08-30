{
  callPackage,
  lib,
  runCommand,
  mkZigApplication,
  source,
  zigradSrc,
  emitterConfiguration,
}: let
  programName = "model.zgpr";
  mkIreeEmbeddedExample = callPackage ../../nix/packages/iree-embedded-example.nix {
    inherit mkZigApplication;
  };
  emitter = mkZigApplication emitterConfiguration {
    mainProgram = "emit-pr";
    pname = "zigrad-example-basic-deployment-pr-emitter";
    src = source;
    inherit zigradSrc;
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
    runnerExternalInputs ? runnerConfiguration.externalInputs,
    targetPkgs ? null,
    strip ? false,
    withRuntimeEnvironment ? false,
    canExecute ? false,
  }:
    mkIreeEmbeddedExample {
      name = "zigrad-example-basic-deployment-${name}";
      description = "Zigrad basic ${name} deployment example";
      vmfbName = "model.vmfb";
      inherit source zigradSrc program programName backend;
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
