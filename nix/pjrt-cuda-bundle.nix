{ pkgs
, wheelSources
, withHeaders ? false
}:

let
  inherit (pkgs) fetchurl unzip patchelf;

  # Flatten the attrset into a list of fetched wheel derivations
  wheels =
    pkgs.lib.flatten (
      pkgs.lib.mapAttrsToList
        (_pkg: entries:
          map
            (e: fetchurl {
              inherit (e) url sha256;
            })
            entries
        )
        wheelSources
    );
in
pkgs.stdenvNoCC.mkDerivation {
  pname = "pjrt-cuda-bundle";
  version = "0.8.3.dev20251228-cuda13";
  src = null;

  dontUnpack = true;

  nativeBuildInputs = [ unzip patchelf ];

  installPhase = ''
    mkdir -p $out/runtime

    for whl in ${pkgs.lib.concatStringsSep " " wheels}; do
      echo "Extracting $whl"
      ${unzip}/bin/unzip -q "$whl" -d tmp
    done

    # Copy PJRT CUDA plugin
    if [ -d tmp/jax_plugins ]; then
      mkdir -p $out/runtime
      cp -r tmp/jax_plugins $out/runtime/
    fi
    rm -f $out/runtime/jax_plugins/xla_cuda13/__init__.py
    rm -f $out/runtime/jax_plugins/xla_cuda13/version.py

    # Copy PJRT runtime package if its there
    if [ -d tmp/jax_cuda13_pjrt ]; then
      cp -r tmp/jax_cuda13_pjrt $out/runtime/
    fi

    # Copy NVIDIA user-space libs
    mkdir -p $out/runtime/nvidia
    for pkg in cu13 cudnn cublas nccl nvshmem cuda_nvrtc nvjitlink; do
      if [ "$withHeaders" != "1" ] && [ -d "tmp/nvidia/$pkg/include" ]; then
        # prune headers
        rm -r tmp/nvidia/$pkg/include
      fi
      if [ -d "tmp/nvidia/$pkg" ]; then
        cp -r -P "tmp/nvidia/$pkg" "$out/runtime/nvidia/"
      fi
    done

    # cpp deps
    mkdir -p $out/runtime/sys/lib

   # cpp runtime
    cp ${pkgs.stdenv.cc.cc.lib}/lib/libstdc++.so.6 $out/runtime/sys/lib/
    cp ${pkgs.stdenv.cc.cc.lib}/lib/libgcc_s.so.1  $out/runtime/sys/lib/

    # zlib (wanted by cudnn)
    cp ${pkgs.zlib}/lib/libz.so.1 $out/runtime/sys/lib/

    # patch DSOs so they look in our bundle
    sys_rpath='$ORIGIN/../../sys/lib'

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

    # Plugin: jax_plugins/xla_cuda13 -> runtime/sys/lib is ../../sys/lib
    plugin="$out/runtime/jax_plugins/xla_cuda13/xla_cuda_plugin.so"
    patch_append_rpath "$plugin" '$ORIGIN/../../sys/lib'

    # NVIDIA DSOs: nvidia/<pkg>/lib -> runtime/sys/lib is ../../../sys/lib
    for so in $out/runtime/nvidia/*/lib/*.so*; do
      [ -f "$so" ] || continue
      patch_append_rpath "$so" '$ORIGIN/../../../sys/lib'
    done

    # Cleanup 
    rm -rf tmp
  '';

  meta = {
    description = "PJRT CUDA runtime bundle";
    platforms = [ "x86_64-linux" ];
  };
}

