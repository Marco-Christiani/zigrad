# nix/pjrt-cuda-bundle.nix
{
  pkgs,
  lockFile,
  withNvidiaHeaders ? false,
}:

let
  inherit (pkgs)
    fetchurl
    unzip
    patchelf
    lib
    ;

  lock = builtins.fromJSON (builtins.readFile lockFile);
  wheels = lock.wheels;

  # Fetch all wheels listed in lock.json
  wheelFetches = map (
    w:
    fetchurl {
      url = w.url;
      hash = w.hash_sri;
    }
  ) wheels;

in
pkgs.stdenvNoCC.mkDerivation {
  pname = "pjrt-cuda-bundle";
  version =
    let
      jax = lock.pins.jax.git_hash;
    in
    "jax-${builtins.substring 0 7 jax}";

  dontUnpack = true;
  nativeBuildInputs = [
    unzip
    patchelf
  ];

  installPhase = ''
    set -euo pipefail
    mkdir -p $out/runtime

    # ------------------------------------------------------------------
    # Unpack wheels
    # ------------------------------------------------------------------
    mkdir tmp
    for whl in ${lib.concatStringsSep " " wheelFetches}; do
      echo "Extracting $whl"
      ${unzip}/bin/unzip -oq "$whl" -d tmp
    done

    # ------------------------------------------------------------------
    # PJRT CUDA plugin
    # ------------------------------------------------------------------
    if [ -d tmp/jax_plugins ]; then
      cp -r tmp/jax_plugins $out/runtime/
    else
      echo "ERROR: jax_plugins not found in wheels"
      exit 1
    fi

    # ------------------------------------------------------------------
    # Optional PJRT runtime Python package
    # ------------------------------------------------------------------
    if [ -d tmp/jax_cuda13_pjrt ]; then
      cp -r tmp/jax_cuda13_pjrt $out/runtime/
    fi

    # ------------------------------------------------------------------
    # NVIDIA user-space libraries
    # ------------------------------------------------------------------
    mkdir -p $out/runtime/nvidia
    for pkg in cu13 cudnn cublas nccl nvshmem cuda_nvrtc nvjitlink; do
      if [ -d "tmp/nvidia/$pkg" ]; then
        if [ "${lib.boolToString withNvidiaHeaders}" != "true" ] && [ -d "tmp/nvidia/$pkg/include" ]; then
          rm -r tmp/nvidia/$pkg/include
        fi
        cp -r -P "tmp/nvidia/$pkg" "$out/runtime/nvidia/"
      fi
    done

    # ------------------------------------------------------------------
    # C++ runtime dependencies
    # ------------------------------------------------------------------
    mkdir -p $out/runtime/sys/lib
    cp ${pkgs.stdenv.cc.cc.lib}/lib/libstdc++.so.6 $out/runtime/sys/lib/
    cp ${pkgs.stdenv.cc.cc.lib}/lib/libgcc_s.so.1  $out/runtime/sys/lib/
    cp ${pkgs.zlib}/lib/libz.so.1 $out/runtime/sys/lib/

    # ------------------------------------------------------------------
    # RPATH patching
    # ------------------------------------------------------------------
    patch_append_rpath() {
      local so="$1"
      local add="$2"
      local old
      old="$(patchelf --print-rpath "$so" 2>/dev/null || true)"
      [ -n "$old" ] || old=""
      case ":$old:" in
        *":$add:"*) return 0 ;;
      esac
      if [ -n "$old" ]; then
        patchelf --set-rpath "$old:$add" "$so"
      else
        patchelf --set-rpath "$add" "$so"
      fi
    }

    # Plugin: runtime/jax_plugins/xla_cuda13 -> runtime/sys/lib
    plugin="$out/runtime/jax_plugins/xla_cuda13/xla_cuda_plugin.so"
    patch_append_rpath "$plugin" '$ORIGIN/../../sys/lib'

    # NVIDIA DSOs: runtime/nvidia/<pkg>/lib -> runtime/sys/lib
    for so in $out/runtime/nvidia/*/lib/*.so*; do
      [ -f "$so" ] || continue
      patch_append_rpath "$so" '$ORIGIN/../../../sys/lib'
    done

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    mkdir -p $out/runtime
    cat > $out/runtime/PROVENANCE.json <<EOF
    ${builtins.toJSON lock}
    EOF

    # Cleanup
    rm -rf tmp
  '';

  meta = {
    description = "PJRT CUDA runtime bundle (wheel-derived, hermetic)";
    platforms = [ "x86_64-linux" ];
  };
}
