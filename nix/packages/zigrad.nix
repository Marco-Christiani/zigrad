{
  callPackage,
  lib,
  stdenvNoCC,
  zig,
  autoAddDriverRunpath,
  autoPatchelfHook,
  zigradSrc,
  sdk,
  # `cuda-redist.dev` supplies nvrtc.h and other CUDA dev headers.
  #  build.zig reads `CUDA_HOME` from `b.graph.environ_map` to pull the
  #  nvrtc include path. The nix sandbox doesn't carry env vars by default,
  #  so we set it via the derivation's `env` attribute.
  cudaHome ? null,
  version ? "dev",
  optimize ? "ReleaseFast",
  runTests ? false,
}: let
  pname = "zigrad";
  zigDeps = callPackage ./build.zig.zon.nix {};
in
  # `stdenvNoCC` is correct here. zig ships its own toolchain and pulling
  #  in nixpkgs's gcc/glibc via plain `stdenv` pollutes the include path
  #  enough to confuse zig's bundled libcxx. The libc choice is forced via
  #  `-Dtarget=native-native-gnu` below so zig uses its bundled glibc
  #  headers instead of falling back to musl (which embeds
  #  `PT_INTERP=/lib/ld-musl-x86_64.so.1`, a path that does not exist on
  #  glibc hosts).
  stdenvNoCC.mkDerivation {
    inherit pname version;
    src = zigradSrc;

    strictDeps = true;

    nativeBuildInputs = [
      zig
      # autoPatchelfHook rewrites `PT_INTERP` and `RUNPATH` to point at
      #  nix-store paths. zig links the binary with the conventional
      #  `/lib64/ld-linux-x86-64.so.2`, which on NixOS is a stub that
      #  only delegates to nix-ld for FHS-style binaries. autoPatchelfHook
      #  swaps that for the real glibc dyld in the build closure so the
      #  binary runs natively.
      autoPatchelfHook
      autoAddDriverRunpath
    ];

    buildInputs =
      [sdk]
      ++ lib.optional (cudaHome != null) cudaHome;

    # build.zig reads `CUDA_HOME` to find nvrtc.h. Setting it in the
    #  derivation env makes it visible during the build phase.
    env = lib.optionalAttrs (cudaHome != null) {
      CUDA_HOME = "${cudaHome}";
    };

    # Inline phases: zig-overlay's binary install ships no `setupHook`, and
    #  modern nixpkgs's `zig.passthru.hook` collapses to the zig drv itself
    #  (passthru.nix line: `hook = zig;`). Rather than vendor a copy of
    #  nixpkgs's setup-hook script, we run the two zig commands directly.
    #  Small enough to inline, no version drift to track.
    #
    # `-Dgen-cli-meta=false` skips the cova completion/manpage generator. The
    #  gen exe is built and run during `zig build`, before autoPatchelfHook
    #  fixes interpreter paths. zig autodetects the host abi from what's
    #  visible on the link line: with no glibc available it picks musl
    #  (PT_INTERP=/lib/ld-musl-x86_64.so.1), with glibc it picks the
    #  conventional /lib64/ld-linux-x86-64.so.2. Neither path exists in the
    #  nix sandbox, so the gen exe fails to exec with ENOENT mid-build.
    #  Devshell builds still generate cli docs; only the hermetic nix build
    #  ships without them.
    configurePhase = ''
      runHook preConfigure
      export ZIG_GLOBAL_CACHE_DIR=$(mktemp -d)
      runHook postConfigure
    '';

    buildPhase = ''
      runHook preBuild
      TERM=dumb zig build \
        -j"$NIX_BUILD_CORES" \
        -Doptimize=${optimize} \
        -Dsdk=${sdk} \
        -Dtarget=native-native-gnu \
        -Dinstall-runtime-link=false \
        -Dgen-cli-meta=false \
        -freference-trace=10 \
        --system ${zigDeps} \
        --verbose
      runHook postBuild
    '';

    checkPhase = ''
      runHook preCheck
      TERM=dumb zig build test \
        -j"$NIX_BUILD_CORES" \
        -Doptimize=${optimize} \
        -Dsdk=${sdk} \
        -Dtarget=native-native-gnu \
        -Dinstall-runtime-link=false \
        -Dgen-cli-meta=false \
        -freference-trace=10 \
        --system ${zigDeps} \
        --verbose
      runHook postCheck
    '';

    installPhase = ''
      runHook preInstall
      TERM=dumb zig build install \
        -j"$NIX_BUILD_CORES" \
        -Doptimize=${optimize} \
        -Dsdk=${sdk} \
        -Dtarget=native-native-gnu \
        -Dinstall-runtime-link=false \
        -Dgen-cli-meta=false \
        -freference-trace=10 \
        --system ${zigDeps} \
        --prefix "$out" \
        --verbose
      runHook postInstall
    '';

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
