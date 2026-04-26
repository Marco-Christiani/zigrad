# nix/packages/cuda-redist.nix
#
# Pure-fetch derivation that downloads CUDA redistributable tarballs from NVIDIA's CDN
#  and assembles two layouts:
#
#  out (runtime layout, consumed by SDK runtime profiles):
#    $out/runtime/nvidia/<component>/lib/*.so*
#    $out/runtime/nvidia/cuda_nvcc/bin/{ptxas,nvlink}
#    $out/runtime/nvidia/nvvm/libdevice/
#    $out/runtime/sys/lib/{libstdc++.so.6,libgcc_s.so.1,libz.so.1}
#
#  dev (cudatoolkit-style layout, consumed by build-time tooling - tvm, zig
#  build, devshell):
#    $dev/bin/{nvcc,ptxas,nvlink,cicc}
#    $dev/include/             - flat header tree from all components
#    $dev/lib/*.so*            - flat lib tree
#    $dev/lib/stubs/libcuda.so - driver stub for sandbox link
#    $dev/nvvm/libdevice/
#
# Decoupled from the Bazel build - these are pre-built NVIDIA binaries.
{
  lib,
  stdenv,
  fetchurl,
  patchelf,
  autoPatchelfHook,
  file,
  zlib,
  unzip,
  # CUDA version to resolve from nix/versions.json (e.g. "12.8.1", "12.9.1").
  cudaVersion,
}: let
  versionsJson = builtins.fromJSON (builtins.readFile ../versions.json);

  availableVersions = builtins.attrNames versionsJson.cuda;

  # Find the best matching version: try exact match first, then prefix match
  # picking the newest patch release via semantic version comparison.
  resolvedVersion = let
    exact = versionsJson.cuda.${cudaVersion} or null;
    prefixMatches = builtins.filter (v: lib.hasPrefix cudaVersion v) availableVersions;
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

  fetched = builtins.mapAttrs fetchComponent components;

  hasNccl = components ? nccl && components.nccl.kind == "wheel";

  # Link-time allowlist for $dev/lib. Anything not listed is shipped only in
  #  $out (runtime layout) and must be loaded via LD_LIBRARY_PATH or the
  #  consumer's runtime rpath. Keeps $dev small (~500MB instead of ~14GiB).
  #
  # Includes:
  # - libcudart + static deps (cmake's CUDA toolchain probe links -lcudart
  #   -lcudart_static -lcudadevrt; libculibos is a cudart_static internal dep).
  # - libnvrtc family (runtime kernel compilation; cmake probes for it).
  # - stubs/libcuda.so (build-time stub for the driver; real driver loaded at
  #   runtime via NixOS's /run/opengl-driver).
  devLinkTimeLibPattern = lib.concatStringsSep "|" [
    "libcudart.so*"
    "libcudart_static.a"
    "libcudadevrt.a"
    "libculibos.a"
    "libnvrtc.so*"
    "libnvrtc-builtins.so*"
    "libnvJitLink.so*"
  ];
in
  stdenv.mkDerivation {
    pname = "cuda-redist";
    version = cudaVersion;

    outputs = ["out" "dev"];

    dontUnpack = true;
    # The dev output ships nvcc and headers via `bin/` + `include/` at output
    #  root, which the stdenv multi-output hook would otherwise relocate. We
    #  populate $dev directly with the canonical cudatoolkit-style layout, so
    #  disable the hook to keep our placement.
    setOutputFlags = false;

    nativeBuildInputs =
      [patchelf autoPatchelfHook file]
      ++ lib.optionals hasNccl [unzip];

    # autoPatchelfHook scans every ELF in $out and $dev, resolves NEEDED libs
    #  against buildInputs, and rewrites RPATH without breaking version_r.
    #  Required for nvcc et al. which encode GLIBC version requirements that
    #  manual patchelf --set-rpath silently corrupts.
    buildInputs = [stdenv.cc.cc.lib stdenv.cc.libc zlib];

    # nvshmem ships optional bootstrap/transport plugins for MPI, UCX, OFI,
    #  Mellanox IB, etc. We don't use any of them; the core nvshmem .so works
    #  without them. Tell autoPatchelfHook to skip these deps instead of
    #  failing the build. Mirrors nixpkgs's libnvshmem packaging.
    autoPatchelfIgnoreMissingDeps = [
      "libmpi.so.40"
      "libpmix.so.2"
      "liboshmem.so.40"
      "libmlx5.so.1"
      "libfabric.so.1"
      "libucs.so.0"
      "libucp.so.0"
    ];

    installPhase = ''
      set -eo pipefail
      mkdir -p "$out/runtime/sys/lib"
      mkdir -p "$dev/bin" "$dev/include" "$dev/lib/stubs" "$dev/nvvm"

      # System libs (self-contained runtime layout in out only).
      cp -v ${stdenv.cc.cc.lib}/lib/libstdc++.so.6 "$out/runtime/sys/lib/"
      cp -v ${stdenv.cc.cc.lib}/lib/libgcc_s.so.1 "$out/runtime/sys/lib/"
      cp -v ${zlib}/lib/libz.so.1 "$out/runtime/sys/lib/"

      # Helper: copy headers from a component's extracted include/ dir into $dev/include.
      # Tarballs ship include/ at <topdir>/include or <topdir>/include/<header>.
      copy_headers() {
        local extract_root="$1"
        for incdir in "$extract_root"/*/include; do
          [ -d "$incdir" ] || continue
          cp -aLr "$incdir/." "$dev/include/"
        done
      }

      # Helper: copy link-time libs (whitelisted) and stubs from a component's
      #  extract dir. Runtime-only libs (cudnn, cublas, cufft, cusparse,
      #  nvshmem, libcusolver, ...) are NOT shipped in dev — consumers load
      #  them via LD_LIBRARY_PATH or rpath against the SDK's runtime profile.
      #  Keeps dev's closure ~30x smaller than copying everything.
      copy_libs_to_dev() {
        local extract_root="$1"
        for libdir in "$extract_root"/*/lib "$extract_root"/*/lib64; do
          [ -d "$libdir" ] || continue
          for f in "$libdir"/*.so* "$libdir"/*.a; do
            [ -e "$f" ] || continue
            local base="$(basename "$f")"
            case "$base" in
              ${devLinkTimeLibPattern}) cp -aL "$f" "$dev/lib/" ;;
            esac
          done
          # Stubs are always small + always needed at link time (libcuda.so).
          if [ -d "$libdir/stubs" ]; then
            for f in "$libdir/stubs"/*.so*; do
              [ -e "$f" ] || continue
              cp -aL "$f" "$dev/lib/stubs/"
            done
          fi
        done
      }

      # Extract each tarball component once; populate both outputs.
      ${lib.concatStringsSep "\n" (lib.mapAttrsToList (name: comp: let
        src = fetched.${name};
        dir = comp.runtime_dir;
      in
        lib.optionalString (comp.kind == "tarball") ''
          echo "[cuda-redist] extracting ${name} (${dir})"
          tmp_extract="$(mktemp -d)"
          tar -xf "${src}" -C "$tmp_extract"

          ${lib.optionalString (dir != "cuda_nvcc") ''
            # out: keyed runtime/nvidia/<dir>/lib/ layout (all components except nvcc).
            mkdir -p "$out/runtime/nvidia/${dir}/lib"
            for libdir in "$tmp_extract"/*/lib "$tmp_extract"/*/lib64; do
              [ -d "$libdir" ] || continue
              for f in "$libdir"/*.so*; do
                [ -e "$f" ] || continue
                cp -aL "$f" "$out/runtime/nvidia/${dir}/lib/"
              done
            done
          ''}

          # dev: flat lib + headers from every component (incl. cuda_nvcc).
          copy_libs_to_dev "$tmp_extract"
          copy_headers "$tmp_extract"

          ${lib.optionalString (dir == "cuda_nvcc") ''
            # dev: mirror cuda_nvcc's tarball layout 1:1. nvcc and friends use
            #  path-relative discovery for sibling tools (cicc at nvvm/bin/,
            #  link.stub at bin/crt/, libnvvm.so at nvvm/lib64/, etc.). Easier
            #  to mirror the whole subtree than to enumerate each consumer.
            for srctop in "$tmp_extract"/*/; do
              [ -d "$srctop/bin" ] || continue
              cp -aLr "$srctop/bin"/. "$dev/bin/"
              [ -d "$srctop/nvvm" ] && cp -aLr "$srctop/nvvm"/. "$dev/nvvm/"
            done

            # out: ptxas + nvlink in their existing component-keyed layout.
            mkdir -p "$out/runtime/nvidia/cuda_nvcc/bin" "$out/runtime/nvidia/bin"
            for tool in ptxas nvlink; do
              for candidate in "$tmp_extract"/*/bin/"$tool"; do
                if [ -f "$candidate" ] && [ -x "$candidate" ]; then
                  cp -aL "$candidate" "$out/runtime/nvidia/cuda_nvcc/bin/$tool"
                  cp -aL "$candidate" "$out/runtime/nvidia/bin/$tool"
                  break
                fi
              done
            done
            # out: libdevice bitcode (used by NVRTC at runtime).
            for libdev in "$tmp_extract"/*/nvvm/libdevice; do
              if [ -d "$libdev" ]; then
                mkdir -p "$out/runtime/nvidia/nvvm/libdevice"
                cp -aL "$libdev"/. "$out/runtime/nvidia/nvvm/libdevice/"
              fi
            done
          ''}

          rm -rf "$tmp_extract"
        '')
      components)}

      ${lib.optionalString hasNccl (let
        src = fetched.nccl;
      in ''
        # NCCL: extract from wheel (zip), populate both outputs.
        echo "[cuda-redist] extracting nccl from wheel"
        mkdir -p "$out/runtime/nvidia/nccl/lib"
        tmp_extract="$(mktemp -d)"
        unzip -q "${src}" -d "$tmp_extract"
        find "$tmp_extract" -name '*.so*' -type f | while read -r f; do
          cp -aL "$f" "$out/runtime/nvidia/nccl/lib/"
          cp -aL "$f" "$dev/lib/"
        done
        rm -rf "$tmp_extract"
      '')}

      # autoPatchelfHook (in fixupPhase) handles RPATHs and interpreters for
      #  every ELF across $out and $dev. It scans NEEDED entries, resolves
      #  against buildInputs and same-derivation outputs, and rewrites RPATH
      #  without corrupting versioned-symbol requirements (which manual
      #  patchelf was breaking on nvcc, surfacing as
      #  "undefined symbol: , version GLIBC_2.2.5" at runtime).

      chmod -R u+w "$out/runtime" "$dev"

      # Sanity checks: assert specific critical files. Counting .so* is fragile
      #  with the dev-side whitelist (only ~12 versioned files expected).
      out_so_count="$(find "$out/runtime/nvidia" -type f -name '*.so*' | wc -l)"
      echo "[cuda-redist] out runtime: $out_so_count DSOs"
      if [ "$out_so_count" -lt 10 ]; then
        echo "ERROR: expected >=10 out DSOs, got $out_so_count" >&2
        exit 1
      fi
      missing=()
      for required in \
          "$dev/bin/nvcc" \
          "$dev/bin/cudafe++" \
          "$dev/nvvm/bin/cicc" \
          "$dev/lib/libcudart.so" \
          "$dev/lib/libcudart_static.a" \
          "$dev/lib/libcudadevrt.a" \
          "$dev/lib/libnvrtc.so" \
          "$dev/lib/stubs/libcuda.so"; do
        [ -e "$required" ] || missing+=("$required")
      done
      if [ "''${#missing[@]}" -gt 0 ]; then
        echo "ERROR: dev output missing critical files:" >&2
        printf '  %s\n' "''${missing[@]}" >&2
        exit 1
      fi
      echo "[cuda-redist] dev: critical files present"
    '';

    meta = {
      description = "CUDA redistributable runtime + cudatoolkit-style dev (${cudaVersion})";
      platforms = lib.platforms.linux;
    };
  }
