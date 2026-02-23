{
  callPackage,
  lib,
  stdenvNoCC,
  zig,
  autoAddDriverRunpath,
  zigradSrc,
  sdk,
  optimize ? "ReleaseFast",
  runTests ? false,
}: let
  pname = "zigrad";
  zigDeps = callPackage ./build.zig.zon.nix { };
in
  stdenvNoCC.mkDerivation {
    inherit pname;
    version = "dev";
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
      "--system"
      zigDeps
    ];

    zigCheckFlags = [
      "-Doptimize=${optimize}"
      "-Dsdk=${sdk}"
      "-Dinstall-runtime-link=false"
      "--system"
      zigDeps
    ];

    doCheck = runTests;

    postFixup = ''
      addDriverRunpath "$out/bin/zigrad"
    '';

    meta = {
      mainProgram = pname;
      platforms = lib.platforms.linux;
    };
  }
