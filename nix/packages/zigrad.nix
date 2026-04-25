{
  callPackage,
  lib,
  stdenvNoCC,
  zig,
  autoAddDriverRunpath,
  zigradSrc,
  sdk,
  version ? "dev",
  optimize ? "ReleaseFast",
  runTests ? false,
}: let
  pname = "zigrad";
  zigDeps = callPackage ./build.zig.zon.nix {};
in
  stdenvNoCC.mkDerivation {
    inherit pname version;
    src = zigradSrc;

    strictDeps = true;

    nativeBuildInputs = [
      zig.hook
      autoAddDriverRunpath
    ];

    buildInputs = [
      sdk
    ];

    zigBuildFlags = [
      "-Doptimize=${optimize}"
      "-Dsdk=${sdk}"
      "-Dinstall-runtime-link=false"
      "-freference-trace=10"
      "--system"
      zigDeps
    ];

    zigCheckFlags = [
      "-Doptimize=${optimize}"
      "-Dsdk=${sdk}"
      "-Dinstall-runtime-link=false"
      "-freference-trace=10"
      "--system"
      zigDeps
    ];

    doCheck = runTests;

    postFixup = ''
      addDriverRunpath "$out/bin/zigrad"
    '';

    meta = {
      description = "Zigrad: differentiable computation framework";
      license = lib.licenses.asl20;
      mainProgram = pname;
      platforms = lib.platforms.linux;
    };
  }
