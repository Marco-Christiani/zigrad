# TODO: this will include some text files we dont need like headers in include/ subdirs and python files, clean up later
{ pkgs
, wheelSources
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

    # Copy PJRT runtime package if its there
    if [ -d tmp/jax_cuda13_pjrt ]; then
      cp -r tmp/jax_cuda13_pjrt $out/runtime/
    fi

    # Copy NVIDIA user-space libs
    if [ -d tmp/nvidia ]; then
      cp -r tmp/nvidia $out/runtime/
    fi

    # Cleanup 
    rm -rf tmp
  '';

  meta = {
    description = "PJRT CUDA runtime bundle";
    platforms = [ "x86_64-linux" ];
  };
}

