{ pkgs
, wheelSources
, withHeaders ? false
}:

let
  inherit (pkgs) fetchurl unzip;

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

  nativeBuildInputs = [ unzip ];

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
    # This is fine, but will cause headers to be included
    # if [ -d tmp/nvidia ]; then
    #   cp -r tmp/nvidia $out/runtime/
    # fi
    # Being a bit more precise, although this hardcodes, not sure if I like 
    #  the explicitness more than the risk of missing things in the general/future case.
    mkdir -p $out/runtime/nvidia
    for pkg in cu13 cudnn cublas nccl nvshmem cuda_nvrtc nvjitlink; do
      if [ -d "tmp/nvidia/$pkg/lib" ]; then
        mkdir -p "$out/runtime/nvidia/$pkg/lib"
        cp -P tmp/nvidia/$pkg/lib/*.so* "$out/runtime/nvidia/$pkg/lib/"
      fi
    done
    # optionally pull in the headers if requested
    if [ "$withHeaders" = "1" ] && [ -d "tmp/nvidia/$pkg/include" ]; then
      mkdir -p "$out/runtime/nvidia/$pkg/include"
      cp -r tmp/nvidia/$pkg/include/* "$out/runtime/nvidia/$pkg/include/"
    fi

    # Cleanup 
    rm -rf tmp
  '';

  meta = {
    description = "PJRT CUDA runtime bundle";
    platforms = [ "x86_64-linux" ];
  };
}

