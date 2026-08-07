{
  inputs = {
    # zig 0.16 added in bbbf018e74b80af4a4692a4b3fca8e91bb0ed7ee
    nixpkgs.url = "github:NixOS/nixpkgs/bbbf018e74b80af4a4692a4b3fca8e91bb0ed7ee";

    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };

    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    zig-overlay = {
      url = "github:mitchellh/zig-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Source pins.
    #
    # Update with: nix flake update --impure --update-input xla-src (etc.)
    #
    # LLVM and StableHLO commits are derived from XLA's third_party/.
    # after updating xla-src, check third_party/llvm/workspace.bzl
    # and third_party/stablehlo/workspace.bzl for new commits.
    xlaSrc = {
      url = "github:openxla/xla/913ae2eaa3cb88971003592a90959685a78c9e30";
      flake = false;
    };

    llvmSrc = {
      url = "github:llvm/llvm-project/8f264586d7521b0e305ca7bb78825aa3382ffef7";
      flake = false;
    };

    stablehloSrc = {
      url = "github:openxla/stablehlo/1ef9e390b5295e676d2b864fe1924bc2f3f4cf0f";
      flake = false;
    };

    # IREE and its pinned source dependencies.
    ireeSrc = {
      url = "github:iree-org/iree/776210bd36896f8ca14288637592a9d5cebfcea1";
      flake = false;
    };
    ireeLlvmSrc = {
      url = "github:iree-org/llvm-project/c95bd0bba5be9710292ce3a29832b67a0e33f887";
      flake = false;
    };
    ireeStablehloSrc = {
      url = "github:iree-org/stablehlo/6fabd27b15885179a3b6a601ea1e4171f2ed2c91";
      flake = false;
    };
    ireeFlatccSrc = {
      url = "github:dvidelabs/flatcc/9362cd00f0007d8cbee7bff86e90fb4b6b227ff3";
      flake = false;
    };
    ireeBenchmarkSrc = {
      url = "github:google/benchmark/192ef10025eb2c4cdd392bc502f0c852196baa48";
      flake = false;
    };
  };

  outputs = inputs @ {
    flake-parts,
    treefmt-nix,
    ...
  }: let
    zigOverlay = final: prev: {
      zig = inputs.zig-overlay.packages.${prev.stdenv.hostPlatform.system}."0.16.0";
      zls = prev.stdenvNoCC.mkDerivation (finalAttrs: {
        pname = "zls";
        version = "0.16.0";

        src = let
          assets = {
            x86_64-linux = {
              url = "https://github.com/zigtools/zls/releases/download/${finalAttrs.version}/zls-x86_64-linux.tar.xz";
              hash = "sha256-3tbVYqC4buh4sd33D/qyeXzjzco7AtYHdUj51W3/lrY=";
            };
            aarch64-linux = {
              url = "https://github.com/zigtools/zls/releases/download/${finalAttrs.version}/zls-aarch64-linux.tar.xz";
              hash = "sha256-QwzSk9IB63CuJRnbyWyFS/h5G433/JOS6NLcloCivtc=";
            };
          };
          asset =
            assets.${prev.stdenvNoCC.hostPlatform.system}
            or (throw "zls-bin: unsupported system ${prev.stdenvNoCC.hostPlatform.system}");
        in
          prev.fetchurl asset;

        sourceRoot = ".";

        installPhase = ''
          runHook preInstall

          install -Dm755 zls "$out/bin/zls"
          install -Dm644 README.md "$out/share/doc/zls/README.md"
          install -Dm644 LICENSE "$out/share/licenses/zls/LICENSE"

          runHook postInstall
        '';

        meta = {
          description = "Zig LSP implementation and Zig language server";
          mainProgram = "zls";
          homepage = "https://github.com/zigtools/zls";
          license = prev.lib.licenses.mit;
          platforms = [
            "x86_64-linux"
            "aarch64-linux"
          ];
        };
      });
    };

    # Shared CUDA package and target configuration.
    cudaCfg = {
      gccHostAttr = "gcc14";
      cudaPackagesAttr = "cudaPackages_12_9";
      cudaVersion = "12.9.1";
    };

    # Build options threaded into external source derivations.
    #
    #  Project defaults avoid host-specific CPU and CUDA tuning. Named
    #  configurations decide which packages are demanded. This attrset decides
    #  how those external packages are compiled.
    #
    #  Per-user overrides live in ./local-build-cfg.nix (gitignored).
    #  Copy ./local-build-cfg.example.nix to start. To apply the override:
    #
    #    nix build --impure .#<configuration>
    #
    #  The --impure flag is required so the flake can read the user's
    #  current working directory for the file.
    #
    #  See the Building section of the docs site for the full knob reference
    #  and use-case recipes.
    buildCfg = let
      base = {
        withDebugSymbols = false; # stripped, NDEBUG (production)
        withNativeTuning = false; # portable: no -march=native
        cudaArchitectures = []; # upstream fat list
        enableLto = false; # opt-in via local override or release tooling
        extraCxxFlags = [];
        extraLdFlags = [];
        extraBazelFlags = [];
      };
      # Impure evaluation supplies PWD for the optional local override.
      pwd = builtins.getEnv "PWD";
      localFile =
        if pwd != ""
        then "${pwd}/local-build-cfg.nix"
        else null;
      localOverrides =
        if localFile != null && builtins.pathExists localFile
        then import localFile
        else {};
    in
      base // localOverrides;
  in
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      imports = [
        ./nix/flake/integrations.nix
        ./nix/flake/tools.nix
        ./nix/flake/devshells.nix
        ./nix/flake/checks.nix
      ];

      perSystem = {system, ...}: let
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [zigOverlay];
          config = {
            allowUnfree = true;
            # TODO(nix): Narrow unfree package acceptance to required inputs.
            # Enabling cudaSupport globally changes unrelated nixpkgs packages.
            # cudaSupport = true;
          };
        };
        treefmt = treefmt-nix.lib.evalModule pkgs ./treefmt.nix;

        # cudaCfg pins two independently versioned inputs.
        #  cudaVersion drives versions.json (cuda-redist tarballs from nvidia).
        #  cudaPackagesAttr selects a nixpkgs cuda set whose minor version
        #  floats with the nixpkgs lock. Assert their major.minor agree so a
        #  version mismatch fails during evaluation.
        pkgsCudartVersion = pkgs.${cudaCfg.cudaPackagesAttr}.cuda_cudart.version;
        cudaVerMM = pkgs.lib.versions.majorMinor cudaCfg.cudaVersion;
        pkgsCudartVerMM = pkgs.lib.versions.majorMinor pkgsCudartVersion;
      in
        assert pkgs.lib.assertMsg
        (cudaVerMM == pkgsCudartVerMM)
        ''
          cudaCfg drift: cudaVersion=${cudaCfg.cudaVersion} (major.minor ${cudaVerMM})
            disagrees with nixpkgs ${cudaCfg.cudaPackagesAttr}.cuda_cudart.version=${pkgsCudartVersion} (major.minor ${pkgsCudartVerMM}).
            Update flake.nix cudaCfg or the nixpkgs lock.
        ''; {
          _module.args = {
            inherit pkgs cudaCfg buildCfg;
          };
          formatter = treefmt.config.build.wrapper;
          checks.formatting = treefmt.config.build.check inputs.self;
        };
    };
}
