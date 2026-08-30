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
  vmfbName = "model.vmfb";
  checkpointName = "model.safetensors";
  mkIreeEmbeddedExample = callPackage ../../nix/packages/iree-embedded-example.nix {
    inherit mkZigApplication;
  };
  mkTool = {
    name,
    mode,
  }:
    mkZigApplication emitterConfiguration {
      mainProgram = name;
      pname = "zigrad-example-cifar10-${name}";
      src = source;
      inherit zigradSrc;
      zigArgs = ["-Dmode=${mode}"];
      withRuntimeEnvironment = false;
    };
  emitter = mkTool {
    name = "cifar10-emit-pr";
    mode = "emit_pr";
  };
  initializer = mkTool {
    name = "cifar10-init";
    mode = "init";
  };
  program = runCommand "zigrad-example-cifar10-program" {} ''
    mkdir -p "$out"
    ${lib.getExe emitter} "$out/${programName}"
  '';
  initialCheckpoint = runCommand "zigrad-example-cifar10-initial-checkpoint" {} ''
    mkdir -p "$out"
    ${lib.getExe initializer} "$out/${checkpointName}"
  '';
in
  {
    name,
    backend,
    compilerConfiguration,
    runnerConfiguration,
    checkpoint ? initialCheckpoint,
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
      name = "zigrad-example-cifar10-${name}";
      description = "Zigrad CIFAR-10 ${name} inference example";
      inherit source zigradSrc program programName vmfbName backend;
      inherit compilerConfiguration compilerArguments compilerEnvironment;
      inherit runnerConfiguration runnerExternalInputs targetPkgs withRuntimeEnvironment;
      mainProgram = "cifar10-infer";
      runnerArguments =
        ["-Dmode=infer"]
        ++ lib.optional strip "-Dstrip=true"
        ++ runnerArguments;
      embeddedFiles = [
        {
          source = checkpoint;
          sourceName = checkpointName;
          destinationName = checkpointName;
        }
      ];
      installCheckCommand =
        if canExecute
        then ''
          head -c 3073 /dev/zero > cifar-record.bin
          "$out/bin/cifar10-infer" cifar-record.bin
        ''
        else null;
      passthru = {inherit checkpoint;};
    }
