# nix/cuda-redist.nix
#
# Pure-fetch derivation that downloads CUDA redistributable tarballs from NVIDIA's CDN and
#  assembles the runtime/ layout expected by build.zig.
#
# Decoupled from the Bazel build - these are pre-built NVIDIA binaries, not compiled artifacts.
{
  lib,
  stdenv,
  fetchurl,
  patchelf,
  file,
  zlib,
  unzip,
  # CUDA version to resolve from nix/versions.json (e.g. "12.8.1", "12.9.1").
  cudaVersion,
}: let
  versionsJson = builtins.fromJSON (builtins.readFile ./versions.json);

  availableVersions = builtins.attrNames versionsJson.cuda;

  # Find the best matching version: try exact match first, then prefix match
  # picking the newest patch release via semantic version comparison.
  # This handles e.g. cudaVersion="12.8" matching "12.8.1" in versions.json.
  resolvedVersion = let
    exact = versionsJson.cuda.${cudaVersion} or null;
    prefixMatches = builtins.filter (v: lib.hasPrefix cudaVersion v) availableVersions;
    # Sort descending by semantic version (lib.versionOlder does proper
    # numeric comparison per component), then take the first (newest).
    bestPrefix =
      if builtins.length prefixMatches > 0
      then builtins.head (builtins.sort (a: b: lib.versionOlder b a) prefixMatches)
      else null;
  in
    if exact != null
    then cudaVersion
    else if bestPrefix != null
    then bestPrefix
    else throw "CUDA version ${cudaVersion} not found in nix/versions.json. Available: ${builtins.concatStringsSep ", " availableVersions}";

  cudaEntry = versionsJson.cuda.${resolvedVersion};

  components = cudaEntry.components;

  # Fetch each component as a fixed-output derivation.
  fetchComponent = name: comp:
    fetchurl {
      url = comp.url;
      hash = comp.hash_sri;
      name = "${name}-${comp.version}.${
        if comp.kind == "wheel"
        then "whl"
        else "tar.xz"
      }";
    };

  # Build a map of name -> fetched path.
  fetched = builtins.mapAttrs fetchComponent components;

  # Components that produce lib/*.so* (standard tarball layout).
  libComponents = lib.filterAttrs (_: c: c.kind == "tarball" && c.runtime_dir != "cuda_nvcc") components;

  # cuda_nvcc provides tools (ptxas, nvlink), not libraries.
  hasNvcc = components ? cuda_nvcc;

  # NCCL comes as a wheel (zip with .so inside).
  hasNccl = components ? nccl && components.nccl.kind == "wheel";
in
  stdenv.mkDerivation {
    pname = "cuda-redist";
    version = cudaVersion;

    dontUnpack = true;

    nativeBuildInputs = [patchelf file] ++ lib.optionals hasNccl [unzip];

    installPhase = ''
      set -euo pipefail
      mkdir -p "$out/runtime/sys/lib"

      # System libs (self-contained runtime, matches xla-pjrt-runtime-bazel.nix).
      cp -v ${stdenv.cc.cc.lib}/lib/libstdc++.so.6 "$out/runtime/sys/lib/"
      cp -v ${stdenv.cc.cc.lib}/lib/libgcc_s.so.1 "$out/runtime/sys/lib/"
      cp -v ${zlib}/lib/libz.so.1 "$out/runtime/sys/lib/"

      # Extract tarball components into runtime/nvidia/<runtime_dir>/lib/.
      ${lib.concatStringsSep "\n" (lib.mapAttrsToList (name: comp: let
        src = fetched.${name};
        dir = comp.runtime_dir;
      in
        lib.optionalString (comp.kind == "tarball" && dir != "cuda_nvcc") ''
          echo "[cuda-redist] extracting ${name} -> nvidia/${dir}/lib"
          mkdir -p "$out/runtime/nvidia/${dir}/lib"
          tmp_extract="$(mktemp -d)"
          tar -xf "${src}" -C "$tmp_extract"
          # Tarballs have a top-level dir; find lib/ underneath.
          for libdir in "$tmp_extract"/*/lib "$tmp_extract"/*/lib64; do
            [ -d "$libdir" ] || continue
            for f in "$libdir"/*.so*; do
              [ -e "$f" ] || continue
              cp -aL "$f" "$out/runtime/nvidia/${dir}/lib/"
            done
          done
          rm -rf "$tmp_extract"
        '')
      components)}

      ${lib.optionalString hasNvcc (let
        src = fetched.cuda_nvcc;
      in ''
        # cuda_nvcc: extract ptxas and nvlink.
        echo "[cuda-redist] extracting cuda_nvcc tools"
        mkdir -p "$out/runtime/nvidia/cuda_nvcc/bin" "$out/runtime/nvidia/bin"
        tmp_extract="$(mktemp -d)"
        tar -xf "${src}" -C "$tmp_extract"
        for tool in ptxas nvlink; do
          for candidate in "$tmp_extract"/*/bin/"$tool"; do
            if [ -f "$candidate" ] && [ -x "$candidate" ]; then
              cp -aL "$candidate" "$out/runtime/nvidia/cuda_nvcc/bin/$tool"
              cp -aL "$candidate" "$out/runtime/nvidia/bin/$tool"
              break
            fi
          done
        done
        # libdevice bitcode
        for devdir in "$tmp_extract"/*/nvvm/libdevice; do
          if [ -d "$devdir" ]; then
            mkdir -p "$out/runtime/nvidia/nvvm/libdevice"
            cp -aL "$devdir"/. "$out/runtime/nvidia/nvvm/libdevice/"
          fi
        done
        rm -rf "$tmp_extract"
      '')}

      ${lib.optionalString hasNccl (let
        src = fetched.nccl;
      in ''
        # NCCL: extract from wheel (zip).
        echo "[cuda-redist] extracting nccl from wheel"
        mkdir -p "$out/runtime/nvidia/nccl/lib"
        tmp_extract="$(mktemp -d)"
        unzip -q "${src}" -d "$tmp_extract"
        find "$tmp_extract" -name '*.so*' -type f | while read -r f; do
          cp -aL "$f" "$out/runtime/nvidia/nccl/lib/"
        done
        rm -rf "$tmp_extract"
      '')}

      # Set RPATH on all NVIDIA DSOs.
      # $ORIGIN: sibling libs in same dir (e.g. libnvrtc -> libnvrtc-builtins).
      # $ORIGIN/../../../sys/lib: bundled system libs (libstdc++, libgcc_s, libz).
      echo "[cuda-redist] patching RPATHs on NVIDIA DSOs"
      find "$out/runtime/nvidia" -type f -name '*.so*' | while read -r so; do
        patchelf --set-rpath '$ORIGIN:$ORIGIN/../../../sys/lib' "$so" || true
      done

      # Patch interpreter on tools (ptxas, nvlink).
      for tool in "$out/runtime/nvidia/cuda_nvcc/bin"/* "$out/runtime/nvidia/bin"/*; do
        [ -f "$tool" ] || continue
        if file -L "$tool" 2>/dev/null | grep -q 'ELF '; then
          patchelf --set-interpreter "${stdenv.cc.bintools.dynamicLinker}" "$tool" || true
          patchelf --set-rpath '${lib.makeLibraryPath [stdenv.cc.cc stdenv.cc.libc zlib]}' "$tool" || true
        fi
      done

      chmod -R u+w "$out/runtime"

      # Verify
      so_count="$(find "$out/runtime/nvidia" -type f -name '*.so*' | wc -l)"
      echo "[cuda-redist] $so_count DSOs installed"
      if [ "$so_count" -lt 10 ]; then
        echo "ERROR: expected at least 10 CUDA DSOs, got $so_count" >&2
        exit 1
      fi
    '';

    meta = {
      description = "CUDA redistributable runtime bundle (${cudaVersion})";
      platforms = lib.platforms.linux;
    };
  }
