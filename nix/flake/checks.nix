# nix/flake/checks.nix
#
# Hermetic flake checks (CI-ready, no host GPU required).
{inputs, ...}: let
  zigradVersion = inputs.self.shortRev or inputs.self.dirtyShortRev or "dev";
in {
  perSystem = {
    pkgs,
    config,
    ...
  }: let
    zigradSrc = import ../source-filter.nix {
      inherit (pkgs) lib;
      root = ../..;
    };

    zigrad = config.packages.zigrad;

    zigradTests = pkgs.callPackage ../zigrad.nix {
      inherit zigradSrc;
      version = zigradVersion;
      sdk = config.packages.zigrad-external-sdk-build-devel;
      optimize = "ReleaseSafe";
      runTests = true;
    };

    checkTvmRuntimeFullFfi =
      pkgs.runCommand "check-zigrad-tvm-runtime-full-ffi" {
        nativeBuildInputs = [
          zigrad
        ];
      } ''
        set -euo pipefail
        export HOME="$TMPDIR"
        runtime_root="${config.packages.zigrad-external-sdk-runtime-full-devel}"
        export LD_LIBRARY_PATH="$runtime_root/lib:$runtime_root/runtime/sys/lib:$runtime_root/runtime/nvidia/nvrtc/lib:$runtime_root/runtime/nvidia/nvjitlink/lib"

        ${zigrad}/bin/zigrad tvm-dump-symbols > "$TMPDIR/tvm-symbols.txt"
        test -s "$TMPDIR/tvm-symbols.txt"

        mkdir -p "$out"
        cp "$TMPDIR/tvm-symbols.txt" "$out/tvm-symbols.txt"
      '';

    checkTvmRuntimeNoTvm =
      pkgs.runCommand "check-zigrad-tvm-runtime-no-tvm" {
        nativeBuildInputs = [
          zigrad
        ];
      } ''
        set -euo pipefail
        export HOME="$TMPDIR"
        runtime_root="${config.packages.zigrad-external-sdk-runtime-no-tvm-devel}"
        export LD_LIBRARY_PATH="$runtime_root/lib:$runtime_root/runtime/sys/lib:$runtime_root/runtime/nvidia/nvrtc/lib:$runtime_root/runtime/nvidia/nvjitlink/lib"

        if ${zigrad}/bin/zigrad tvm-dump-symbols > "$TMPDIR/stdout.txt" 2> "$TMPDIR/stderr.txt"; then
          echo "expected tvm-dump-symbols to fail without TVM runtime libraries" >&2
          exit 1
        fi

        if ! grep -Eq "(TvmLoadFailed|failed to load TVM FFI runtime|dlopen)" "$TMPDIR/stderr.txt"; then
          echo "expected loader diagnostics in stderr" >&2
          cat "$TMPDIR/stderr.txt" >&2
          exit 1
        fi

        mkdir -p "$out"
        cp "$TMPDIR/stderr.txt" "$out/tvm-missing-stderr.txt"
      '';
  in {
    checks = {
      zigrad-build = zigrad;
      zigrad-unit-tests = zigradTests;
      tvm-runtime-full-ffi = checkTvmRuntimeFullFfi;
      tvm-runtime-no-tvm = checkTvmRuntimeNoTvm;
    };
  };
}
