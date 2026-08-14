{
  callPackage,
  lib,
  runCommand,
  mkZigradZigPackage,
  source,
  zigradSrc,
  emitterConfiguration,
}: let
  sourceSubdir = "examples/cifar10-lifecycle";
  programName = "model.zgpr";
  vmfbName = "model.vmfb";
  checkpointName = "model.safetensors";
  mkIreeEmbeddedExample = callPackage ../../nix/packages/iree-embedded-example.nix {
    inherit mkZigradZigPackage;
  };
  mkTool = {
    name,
    mode,
  }:
    mkZigradZigPackage {
      configuration = emitterConfiguration.package;
      mainProgram = name;
      pname = "zigrad-example-cifar10-${name}";
      src = source;
      inherit sourceSubdir zigradSrc;
      usePackagedZigrad = true;
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
    runnerExternalInputs ? runnerConfiguration.externalInputs.combined,
    targetPkgs ? null,
    strip ? false,
    withRuntimeEnvironment ? false,
    canExecute ? false,
  }:
    mkIreeEmbeddedExample {
      name = "zigrad-example-cifar10-${name}";
      description = "Zigrad CIFAR-10 ${name} inference example";
      inherit source sourceSubdir zigradSrc program programName vmfbName backend;
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
