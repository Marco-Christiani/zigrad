{
  callPackage,
  lib,
  stdenvNoCC,
  zig,
  zigradSrc,
}: let
  zigDeps = callPackage ./zig-dependencies.nix {
    withPjrt = false;
  };
in
  stdenvNoCC.mkDerivation {
    pname = "zigrad-autodoc";
    version = "dev";
    src = zigradSrc;

    strictDeps = true;
    nativeBuildInputs = [zig];

    configurePhase = ''
      runHook preConfigure
      export ZIG_GLOBAL_CACHE_DIR=$(mktemp -d)
      runHook postConfigure
    '';

    dontBuild = true;

    installPhase = ''
      runHook preInstall
      TERM=dumb zig build docs \
        -j"$NIX_BUILD_CORES" \
        -Doptimize=ReleaseFast \
        -Dtarget=native-native-gnu \
        -Dpjrt=false \
        -Dmlir=false \
        -Dtvm=false \
        -Dmirage=false \
        -Diree=false \
        -Dnvrtc=false \
        -Dcuda-runtime=false \
        --system ${zigDeps} \
        --prefix "$out"
      runHook postInstall
    '';

    passthru.zigVersion = zig.version;

    meta = {
      description = "Zigrad Zig autodoc bundle";
      license = lib.licenses.asl20;
      platforms = lib.platforms.linux;
    };
  }
