# nix/packages/cuda-redist.nix
#
# Assembles selected CUDA redistributable components into runtime and development
#  layouts. Callers choose the component set and which layout receives files.
#
#  out (runtime layout, composed into CUDA runtime closures):
#    $out/runtime/nvidia/<component>/lib/*.so*
#    $out/runtime/nvidia/cuda_nvcc/bin/{ptxas,nvlink}
#    $out/runtime/nvidia/nvvm/libdevice/
#    $out/runtime/sys/lib/{libstdc++.so.6,libgcc_s.so.1,libz.so.1}
#
#  dev (CUDA toolkit layout consumed by TVM, Zig builds, and development shells):
#    $dev/bin/{nvcc,ptxas,nvlink,cicc}
#    $dev/include/             - flat header tree from all components
#    $dev/lib/*.so*            - flat lib tree
#    $dev/lib/stubs/libcuda.so - driver stub for sandbox link
#    $dev/nvvm/libdevice/
#
{
  lib,
  stdenv,
  fetchurl,
  patchelf,
  autoPatchelfHook,
  file,
  zlib,
  unzip,
  # CUDA version recorded in nix/cuda-redist.json.
  cudaVersion,
  componentNames,
  includeDevelopmentFiles ? false,
  includeRuntimeFiles ? true,
  includeSystemRuntime ? false,
  componentDependencies ? [],
  nameSuffix ? "",
}: let
  catalog = builtins.fromJSON (builtins.readFile ../cuda-redist.json);
  availableVersions = builtins.attrNames catalog.cuda;
  cudaEntry =
    catalog.cuda.${cudaVersion}
    or (throw "CUDA version ${cudaVersion} not found in nix/cuda-redist.json. Available: ${builtins.concatStringsSep ", " availableVersions}");
  missingComponents = builtins.filter (name: !(builtins.hasAttr name cudaEntry.components)) componentNames;
  components = lib.getAttrs componentNames cudaEntry.components;

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
  hasComponent = name: builtins.elem name componentNames;
  developmentRequiredFiles =
    lib.optionals (hasComponent "cuda_nvcc") [
      "$dev/bin/nvcc"
      "$dev/bin/cudafe++"
      "$dev/nvvm/bin/cicc"
    ]
    ++ lib.optionals (hasComponent "cuda_cudart") [
      "$dev/lib/libcudart.so"
      "$dev/lib/libcudart_static.a"
      "$dev/lib/libcudadevrt.a"
      "$dev/lib/stubs/libcuda.so"
    ]
    ++ lib.optional (hasComponent "cuda_nvrtc") "$dev/lib/libnvrtc.so"
    ++ lib.optional (hasComponent "cuda_cccl") "$dev/include/nv/target";

  # Link-time allowlist for $dev/lib. Runtime libraries outside this set remain
  #  in the runtime output selected by the caller.
  #
  # This set covers CMake's CUDA probe, NVRTC compilation, and the driver stub
  #  used for sandboxed linking.
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
  assert lib.assertMsg
  (missingComponents == [])
  "CUDA ${cudaVersion} lacks components: ${builtins.concatStringsSep ", " missingComponents}";
    stdenv.mkDerivation {
      pname = "cuda-redist${nameSuffix}";
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

      # autoPatchelfHook owns interpreter and RPATH updates for the packaged
      #  NVIDIA binaries so versioned symbol requirements remain intact.
      buildInputs = [stdenv.cc.cc.lib stdenv.cc.libc zlib] ++ componentDependencies;

      # nvshmem ships optional bootstrap and transport plugins.
      #
      # The runtime closure uses none of these plugins. Ignore their dependencies
      #  while retaining the core nvshmem library.
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
        mkdir -p "$out/runtime/nvidia" "$out/lib"
        mkdir -p "$dev/bin" "$dev/include" "$dev/lib/stubs" "$dev/nvvm"

        ${lib.optionalString includeSystemRuntime ''
          mkdir -p "$out/runtime/sys/lib"
          cp -v ${stdenv.cc.cc.lib}/lib/libstdc++.so.6 "$out/runtime/sys/lib/"
          cp -v ${stdenv.cc.cc.lib}/lib/libgcc_s.so.1 "$out/runtime/sys/lib/"
          cp -v ${zlib}/lib/libz.so.1 "$out/runtime/sys/lib/"
        ''}

        # NVIDIA tarballs place headers below a single archive root.
        copy_headers() {
          local extract_root="$1"
          for incdir in "$extract_root"/*/include; do
            [ -d "$incdir" ] || continue
            cp -aLr "$incdir/." "$dev/include/"
          done
        }

        # Development outputs contain link-time libraries and the driver stub.
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
            # The driver stub is required for sandboxed linking.
            if [ -d "$libdir/stubs" ]; then
              for f in "$libdir/stubs"/*.so*; do
                [ -e "$f" ] || continue
                cp -aL "$f" "$dev/lib/stubs/"
              done
            fi
          done
        }

        # Extract each tarball component once and populate both outputs.
        ${lib.concatStringsSep "\n" (lib.mapAttrsToList (name: comp: let
          src = fetched.${name};
          dir = comp.runtime_dir;
        in
          lib.optionalString (comp.kind == "tarball") ''
            echo "[cuda-redist] extracting ${name} (${dir})"
            tmp_extract="$(mktemp -d)"
            tar -xf "${src}" -C "$tmp_extract"

            ${lib.optionalString (includeRuntimeFiles && dir != "cuda_nvcc") ''
              # Runtime libraries retain their component directory.
              mkdir -p "$out/runtime/nvidia/${dir}/lib"
              for libdir in "$tmp_extract"/*/lib "$tmp_extract"/*/lib64; do
                [ -d "$libdir" ] || continue
                for f in "$libdir"/*.so*; do
                  [ -e "$f" ] || continue
                  cp -aL "$f" "$out/runtime/nvidia/${dir}/lib/"
                done
              done
            ''}

            ${lib.optionalString includeDevelopmentFiles ''
              copy_libs_to_dev "$tmp_extract"
              copy_headers "$tmp_extract"
            ''}

            ${lib.optionalString (dir == "cuda_nvcc") ''
              ${lib.optionalString includeDevelopmentFiles ''
                # nvcc discovers sibling tools and libraries by relative path.
                for srctop in "$tmp_extract"/*/; do
                  [ -d "$srctop/bin" ] || continue
                  cp -aLr "$srctop/bin"/. "$dev/bin/"
                  [ -d "$srctop/nvvm" ] && cp -aLr "$srctop/nvvm"/. "$dev/nvvm/"
                done
              ''}

              ${lib.optionalString includeRuntimeFiles ''
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
                for libdev in "$tmp_extract"/*/nvvm/libdevice; do
                  if [ -d "$libdev" ]; then
                    mkdir -p "$out/runtime/nvidia/nvvm/libdevice"
                    cp -aL "$libdev"/. "$out/runtime/nvidia/nvvm/libdevice/"
                  fi
                done
              ''}
            ''}

            rm -rf "$tmp_extract"
          '')
        components)}

        ${lib.optionalString hasNccl (let
          src = fetched.nccl;
        in ''
          # XLA distributes its selected NCCL artifact as a Python wheel.
          echo "[cuda-redist] extracting nccl from wheel"
          ${lib.optionalString includeRuntimeFiles ''mkdir -p "$out/runtime/nvidia/nccl/lib"''}
          tmp_extract="$(mktemp -d)"
          unzip -q "${src}" -d "$tmp_extract"
          find "$tmp_extract" -name '*.so*' -type f | while read -r f; do
            ${lib.optionalString includeRuntimeFiles ''cp -aL "$f" "$out/runtime/nvidia/nccl/lib/"''}
            ${lib.optionalString includeDevelopmentFiles ''cp -aL "$f" "$dev/lib/"''}
          done
          rm -rf "$tmp_extract"
        '')}

        # The flat library view gives general consumers one runtime search path.
        #  Component directories remain available for integration-specific RPATHs.
        mkdir -p "$out/lib"
        for sodir in "$out"/runtime/nvidia/*/lib; do
          [ -d "$sodir" ] || continue
          for so in "$sodir"/*.so*; do
            [ -e "$so" ] || continue
            ln -sf "$so" "$out/lib/$(basename "$so")"
          done
        done

        chmod -R u+w "$out/runtime" "$dev"

        ${lib.optionalString includeDevelopmentFiles ''
          missing=()
          for required in ${builtins.concatStringsSep " " developmentRequiredFiles}; do
            [ -e "$required" ] || missing+=("$required")
          done
          if [ "''${#missing[@]}" -gt 0 ]; then
            printf 'missing CUDA development file: %s\n' "''${missing[@]}" >&2
            exit 1
          fi
        ''}
      '';

      meta = {
        description = "Selected CUDA redistributable components (${cudaVersion})";
        platforms = lib.platforms.linux;
      };
    }
