{
  lib,
  stdenv,
  fetchurl,
  runCommand,
  python3,
  bazel_7,
  symlinkJoin,
  binutils,
  patchelf,
  # CUDA (optional)
  cudaSupport ? false,
  cudaPackages ? null,
  cudaArchitectures ? null,
  lockFile,
  devel ? false,
  # Dev-speed knobs (keep false for hermetic builds)
  persistentBazelOutputBase ? false,
  bazelLogEvents ? false,
}: let
  lock = builtins.fromJSON (builtins.readFile lockFile);
  xla = lock.pins.xla;

  xlaTar = fetchurl {
    url = xla.tarball_url;
    hash = xla.hash_sri;
  };

  xlaSrc = runCommand "xla-src-${builtins.substring 0 12 xla.commit}" {} ''
    set -euo pipefail
    mkdir -p "$out"
    tar -xzf ${xlaTar} -C "$out" --strip-components=1
  '';

  effectiveStdenv =
    if cudaSupport
    then
      (
        if cudaPackages == null
        then throw "cudaSupport=true requires cudaPackages"
        else cudaPackages.backendStdenv
      )
    else stdenv;

  cudnnMerged =
    if cudaSupport
    then symlinkJoin {
      name = "cudnn-merged";
      paths = with cudaPackages; [(lib.getDev cudnn) (lib.getLib cudnn)];
    }
    else null;

  cudaLibsJoined =
    if cudaSupport
    then symlinkJoin {
      name = "cuda-libs-joined";
      paths = with cudaPackages; [
        (lib.getLib cuda_cudart)
        (lib.getLib cuda_cupti)
        (lib.getLib libcublas)
        (lib.getLib libcufft)
        # TODO: No DT_NEEDED evidence yet for curand/cusolver. Keep commented unless required.
        # (lib.getLib libcurand)
        # (lib.getLib libcusolver)
        (lib.getLib libcusparse)
      ];
    }
    else null;

  cudaBuildDepsJoined =
    if cudaSupport
    then symlinkJoin {
      name = "cuda-build-deps-joined";
      paths = with cudaPackages; [
        cudaLibsJoined
        (lib.getBin cuda_nvcc)
        (lib.getOutput "static" cuda_cudart)
        (lib.getDev cuda_cccl)
        (lib.getDev cuda_cudart)
        (lib.getDev cuda_cupti)
        (lib.getDev cuda_nvcc)
        (lib.getDev cuda_nvml_dev)
        (lib.getDev cuda_nvtx)
        (lib.getDev libcublas)
        (lib.getDev libcufft)
        (lib.getDev libcurand)
        (lib.getDev libcusolver)
        (lib.getDev libcusparse)
      ];
      # TODO: Trim build deps to the minimum required for Bazel CUDA toolchain.
    }
    else null;

  cudaRuntimeLibs =
    lib.optionals cudaSupport
      (
        [
          {
            name = "cudnn";
            pkg = lib.getLib cudaPackages.cudnn;
          }
          {
            name = "cublas";
            pkg = lib.getLib cudaPackages.libcublas;
          }
          {
            name = "cufft";
            pkg = lib.getLib cudaPackages.libcufft;
          }
          # TODO: No DT_NEEDED evidence yet for curand/cusolver. Enable if runtime requires.
          # {
          #   name = "curand";
          #   pkg = lib.getLib cudaPackages.libcurand;
          # }
          # {
          #   name = "cusolver";
          #   pkg = lib.getLib cudaPackages.libcusolver;
          # }
          {
            name = "cusparse";
            pkg = lib.getLib cudaPackages.libcusparse;
          }
          {
            name = "cudart";
            pkg = lib.getLib cudaPackages.cuda_cudart;
          }
          {
            name = "cupti";
            pkg = lib.getLib cudaPackages.cuda_cupti;
          }
          {
            name = "nvrtc";
            pkg = lib.getLib cudaPackages.cuda_nvrtc;
          }
        ]
        ++ lib.optionals (cudaPackages ? cuda_nvjitlink) [
          {
            name = "nvjitlink";
            pkg = lib.getLib cudaPackages.cuda_nvjitlink;
          }
        ]
        ++ lib.optionals (cudaPackages ? nccl) [
          {
            name = "nccl";
            pkg = lib.getLib cudaPackages.nccl;
          }
        ]
        ++ lib.optionals (cudaPackages ? nvshmem) [
          {
            name = "nvshmem";
            pkg = lib.getLib cudaPackages.nvshmem;
          }
        ]
      );

  cudaComputeCapabilities =
    if cudaSupport && cudaArchitectures != null
    then
      let
        toCap = arch:
          let
            len = builtins.stringLength arch;
          in
            if len == 2 then "${builtins.substring 0 1 arch}.${builtins.substring 1 1 arch}"
            else if len == 3 then "${builtins.substring 0 2 arch}.${builtins.substring 2 1 arch}"
            else arch;
      in
        lib.concatStringsSep "," (map toCap cudaArchitectures)
    else null;

  bazelTargets =
    [
      "//xla/pjrt/c:pjrt_c_api_cpu_plugin.so"
    ]
    ++ lib.optionals cudaSupport [
      "//xla/pjrt/c:pjrt_c_api_gpu_plugin.so"
    ];

  builtOutputs =
    [
      "bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"
    ]
    ++ lib.optionals cudaSupport [
      "bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so"
    ];

  # see jaxlib
  xlaBazelrc =
    ''
      try-import %workspace%/tensorflow.bazelrc
      common --enable_bzlmod=false
      common --experimental_repo_remote_exec
      build --enable_bzlmod=false
      ${lib.optionalString persistentBazelOutputBase
        "build --repository_cache=/nix/var/cache/bazel/xla-${builtins.substring 0 12 xla.commit}/repo-cache"}
      ${lib.optionalString persistentBazelOutputBase
        "build --disk_cache=/nix/var/cache/bazel/xla-${builtins.substring 0 12 xla.commit}/disk-cache"}
      build --repo_env PYTHON_BIN_PATH="${python3}/bin/python"
      build --python_path="${python3}/bin/python"

      # Make Bazel action environment see ccache variables too (even though the wrapper sets them).
      build --action_env CCACHE_DIR="/nix/var/cache/ccache"
      build --action_env CCACHE_COMPRESS="1"
      build --action_env CCACHE_SLOPPINESS="random_seed"
      build --action_env CCACHE_COMPILERCHECK="content"

      # Normalize build-root paths only when they are unstable.
      # If sandboxed, NIX_BUILD_TOP is typically /build and normalization is unnecessary.
      build --action_env CCACHE_BASEDIR="''${NIX_BUILD_TOP}"

      # Allow Bazel sandbox actions to write to the shared ccache dir.
      build --sandbox_writable_path=/nix/var/cache/ccache
    ''
    + lib.optionalString cudaSupport ''
      build --config=cuda
      build --action_env CUDA_TOOLKIT_PATH="${cudaBuildDepsJoined}"
      build --action_env CUDNN_INSTALL_PATH="${cudnnMerged}"
      build --action_env TF_CUDA_PATHS="${cudaBuildDepsJoined},${cudnnMerged}"
      build --action_env TF_CUDA_VERSION="${cudaPackages.cudaMajorMinorVersion}"
      build --action_env TF_CUDNN_VERSION="${lib.versions.major cudaPackages.cudnn.version}"
      ${lib.optionalString (cudaComputeCapabilities != null)
        "build --action_env TF_CUDA_COMPUTE_CAPABILITIES=\"${cudaComputeCapabilities}\""}
    '';

  cudaPluginRpath = lib.optionalString cudaSupport (
    lib.concatStringsSep ":" (
      [
        "$ORIGIN/../../../sys/lib"
      ]
      ++ map (entry: "$ORIGIN/../../../nvidia/${entry.name}/lib") cudaRuntimeLibs
      ++ [
        "$ORIGIN/../../../nvidia/nvshmem/lib"
        "$ORIGIN/../../../nvidia/nvjitlink/lib"
      ]
    )
  );
in
  effectiveStdenv.mkDerivation {
    pname = "xla-pjrt-plugins";
    version =
      "xla-${builtins.substring 0 12 xla.commit}"
      + lib.optionalString devel "-devel"
      + lib.optionalString cudaSupport "-cuda";

    src = xlaSrc;

    nativeBuildInputs = [
      bazel_7
      python3
      binutils
      patchelf
    ];

    buildInputs = lib.optionals cudaSupport [
      cudaPackages.cudatoolkit
    ];

    # Remove any Bazel pin file if present, and patch python repo glue.
    postPatch = ''
      rm -f .bazelversion || true
      substituteInPlace third_party/extensions/python_version.bzl \
        --replace 'USE_PYWRAP_RULES = {use_pywrap_rules}' \
                  'USE_PYWRAP_RULES = {use_pywrap_rules}\nHERMETIC_PYTHON_URL = ""\nHERMETIC_PYTHON_SHA256 = ""\nHERMETIC_PYTHON_PREFIX = "python"'
      substituteInPlace third_party/py/python_init_toolchains.bzl \
        --replace '            python_version_kind = HERMETIC_PYTHON_VERSION_KIND,' ""
      sed -i '/remotable = True,/d' third_party/py/python_configure.bzl
      mkdir -p third_party/local_config_cuda_stub
      cat > third_party/local_config_cuda_stub/BUILD.bazel <<'EOF'
      config_setting(
          name = "is_cuda",
          values = {"define": "using_cuda=true"},
      )
      EOF
      cat >> WORKSPACE <<'EOF'

      load("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
      local_repository(
          name = "local_config_cuda",
          path = "third_party/local_config_cuda_stub",
      )
      EOF
    '';

    preConfigure = ''
      set -euo pipefail
      export HOME="$PWD/.home"
      mkdir -p "$HOME"

      cat > ./.bazelrc <<'CFG'
      ${xlaBazelrc}
      CFG

      if [ "${lib.boolToString devel}" = "true" ] && [ "${lib.boolToString bazelLogEvents}" = "true" ]; then
        cat >> ./.bazelrc <<'CFG'
      build --build_event_json_file=./bazel-bep.json
      build --execution_log_json_file=./bazel-exec.json
      CFG
      fi
    '';

    buildPhase = ''
      runHook preBuild
      ${lib.optionalString devel ''
        bazel info output_base
        bazel info repository_cache
      ''}
      bazel \
        --batch \
        ${lib.optionalString persistentBazelOutputBase
          "--output_base=$PWD/.bazel-output-base"} \
        build \
        -c opt \
        ${lib.concatStringsSep " " bazelTargets}
      runHook postBuild
    '';

    installPhase = ''
      set -euo pipefail
      mkdir -p "$out/runtime/xla/pjrt/c" "$out/runtime/sys/lib"
      mkdir -p "$out/lib"

      for f in ${lib.concatStringsSep " " builtOutputs}; do
        if [[ ! -f "$f" ]]; then
          echo "ERROR: missing expected Bazel output: $f" >&2
          find bazel-bin -type f -name '*.so*' -maxdepth 8 -print >&2 || true
          exit 1
        fi
      done

      cpu_out="bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"
      cp -v --no-preserve=mode "$cpu_out" "$out/runtime/xla/pjrt/c/"

      ${lib.optionalString cudaSupport ''
        mkdir -p "$out/runtime/xla/pjrt/c"
        gpu_out="bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so"
        cp -v --no-preserve=mode "$gpu_out" "$out/runtime/xla/pjrt/c/"
      ''}

      # Runtime sys libs (self-contained)
      cp -v ${stdenv.cc.cc.lib}/lib/libstdc++.so.6 "$out/runtime/sys/lib/"
      cp -v ${stdenv.cc.cc.lib}/lib/libgcc_s.so.1 "$out/runtime/sys/lib/"

      ${lib.optionalString cudaSupport ''
        # Copy CUDA runtime libs into runtime/nvidia/<pkg>/lib
        copy_cuda_lib() {
          local name="$1"
          local pkg="$2"
          local src=""
          if [ -d "$pkg/lib" ]; then
            src="$pkg/lib"
          elif [ -d "$pkg/lib64" ]; then
            src="$pkg/lib64"
          else
            echo "WARNING: no lib dir for $name at $pkg" >&2
            return 0
          fi
          mkdir -p "$out/runtime/nvidia/$name/lib"
          cp -a "$src/." "$out/runtime/nvidia/$name/lib/"
          chmod -R u+w "$out/runtime/nvidia/$name/lib"
        }
      ''}

      ${lib.optionalString cudaSupport (lib.concatStringsSep "\n" (map (entry: ''
        copy_cuda_lib "${entry.name}" "${entry.pkg}"
      '') cudaRuntimeLibs))}

      ${lib.optionalString cudaSupport ''
        # Prefer Bazel runfiles (_solib) for hermetic CUDA/NVSHMEM DSOs.
        log_copy() {
          echo "[cuda-copy] $*"
        }

        copy_one() {
          local src="$1"
          local dest_dir="$2"
          local base=""
          base="$(basename "$src")"
          mkdir -p "$dest_dir"
          if [ -e "$dest_dir/$base" ]; then
            log_copy "overwrite $dest_dir/$base"
          fi
          rm -f "$dest_dir/$base"
          if command -v readlink >/dev/null 2>&1; then
            local resolved=""
            resolved="$(readlink -f "$src" 2>/dev/null || true)"
            if [ -n "$resolved" ] && [ -f "$resolved" ]; then
              log_copy "copy resolved $src -> $resolved -> $dest_dir/$base"
              cp -a "$resolved" "$dest_dir/$base"
              return 0
            fi
          fi
          log_copy "copy (no resolve) $src -> $dest_dir/$base"
          cp -aL "$src" "$dest_dir/$base"
        }

        copy_bazel_libs_from() {
          local src_dir="$1"
          local dest_dir="$2"
          local pattern="$3"
          if [ -d "$src_dir" ]; then
            log_copy "scan $src_dir (pattern=$pattern)"
            mkdir -p "$dest_dir"
            for f in "$src_dir"/$pattern; do
              [ -e "$f" ] || continue
              copy_one "$f" "$dest_dir"
            done
            chmod -R u+w "$dest_dir"
          fi
        }

        solib_candidates=(
          "bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/xla/_solib_x86_64"
          "bazel-bin/_solib_x86_64"
          "bazel-out/k8-opt/bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/xla/_solib_x86_64"
          "bazel-out/k8-opt/bin/_solib_x86_64"
        )

        log_copy "solib candidates: ''${solib_candidates[*]}"
        for d in "''${solib_candidates[@]}"; do
          copy_bazel_libs_from "$d" "$out/runtime/nvidia/cublas/lib" "libcublas*.so*"
          copy_bazel_libs_from "$d" "$out/runtime/nvidia/cublas/lib" "libcublasLt*.so*"
          copy_bazel_libs_from "$d" "$out/runtime/nvidia/cudnn/lib" "libcudnn*.so*"
          copy_bazel_libs_from "$d" "$out/runtime/nvidia/cufft/lib" "libcufft*.so*"
          copy_bazel_libs_from "$d" "$out/runtime/nvidia/cupti/lib" "libcupti*.so*"
          copy_bazel_libs_from "$d" "$out/runtime/nvidia/cusparse/lib" "libcusparse*.so*"
          copy_bazel_libs_from "$d" "$out/runtime/nvidia/nvjitlink/lib" "libnvJitLink*.so*"
          copy_bazel_libs_from "$d" "$out/runtime/nvidia/nvrtc/lib" "libnvrtc*.so*"
          copy_bazel_libs_from "$d" "$out/runtime/nvidia/nvshmem/lib" "libnvshmem_host.so*"
          copy_bazel_libs_from "$d" "$out/runtime/nvidia/nvshmem/lib" "nvshmem_bootstrap_uid.so*"
          copy_bazel_libs_from "$d" "$out/runtime/nvidia/nvshmem/lib" "nvshmem_transport_ibrc.so*"
        done

        # Fallback: find missing hermetic DSOs anywhere under bazel-bin/bazel-out.
        copy_find() {
          local pattern="$1"
          local dest="$2"
          log_copy "fallback find pattern=$pattern"
          mkdir -p "$dest"
          while IFS= read -r -d $'\0' f; do
            copy_one "$f" "$dest"
          done < <(find -L bazel-bin bazel-out -type f -name "$pattern" -print0 2>/dev/null || true)
          chmod -R u+w "$dest"
        }

        copy_find "libnvshmem_host.so.3*" "$out/runtime/nvidia/nvshmem/lib"
        copy_find "nvshmem_bootstrap_uid.so.3*" "$out/runtime/nvidia/nvshmem/lib"
        copy_find "nvshmem_transport_ibrc.so.3*" "$out/runtime/nvidia/nvshmem/lib"
        copy_find "libnvJitLink.so.12*" "$out/runtime/nvidia/nvjitlink/lib"
        copy_find "libnvrtc-builtins.so.12.9*" "$out/runtime/nvidia/nvrtc/lib"

        # Prefer runfiles cuda_nvcc tools if present.
        tool_candidates=(
          "bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/cuda_nvcc/bin"
          "bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/xla/external/cuda_nvcc/bin"
          "bazel-out/k8-opt/bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/cuda_nvcc/bin"
          "bazel-out/k8-opt/bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/xla/external/cuda_nvcc/bin"
        )
        for d in "''${tool_candidates[@]}"; do
          copy_bazel_libs_from "$d" "$out/runtime/nvidia/cuda_nvcc/bin" "ptxas"
          copy_bazel_libs_from "$d" "$out/runtime/nvidia/cuda_nvcc/bin" "nvlink"
        done

        # Fallback to generic search if runfiles paths change.
        copy_find "ptxas" "$out/runtime/nvidia/cuda_nvcc/bin"
        copy_find "nvlink" "$out/runtime/nvidia/cuda_nvcc/bin"

        # XLA also searches <cuda_data_dir>/bin, so mirror tools there too.
        if [ -f "$out/runtime/nvidia/cuda_nvcc/bin/ptxas" ]; then
          copy_one "$out/runtime/nvidia/cuda_nvcc/bin/ptxas" "$out/runtime/nvidia/bin"
        fi
        if [ -f "$out/runtime/nvidia/cuda_nvcc/bin/nvlink" ]; then
          copy_one "$out/runtime/nvidia/cuda_nvcc/bin/nvlink" "$out/runtime/nvidia/bin"
        fi

        # libdevice bitcode for NVVM (fixes libdevice lookup warning)
        nvvm_candidates=(
          "bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/cuda_nvvm/nvvm/libdevice"
          "bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/xla/external/cuda_nvvm/nvvm/libdevice"
          "bazel-out/k8-opt/bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/cuda_nvvm/nvvm/libdevice"
          "bazel-out/k8-opt/bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/xla/external/cuda_nvvm/nvvm/libdevice"
        )
        for d in "''${nvvm_candidates[@]}"; do
          if [ -d "$d" ]; then
            log_copy "copy libdevice from $d"
            mkdir -p "$out/runtime/nvidia/nvvm/libdevice"
            cp -aL "$d/." "$out/runtime/nvidia/nvvm/libdevice/"
            chmod -R u+w "$out/runtime/nvidia/nvvm/libdevice"
          fi
        done
        if [ ! -d "$out/runtime/nvidia/nvvm/libdevice" ]; then
          log_copy "fallback find libdevice"
          mkdir -p "$out/runtime/nvidia/nvvm/libdevice"
          while IFS= read -r -d $'\0' f; do
            copy_one "$f" "$out/runtime/nvidia/nvvm/libdevice"
          done < <(find -L bazel-bin bazel-out -type f -path "*/nvvm/libdevice/*" -print0 2>/dev/null || true)
          chmod -R u+w "$out/runtime/nvidia/nvvm/libdevice"
        fi

        if [[ "${lib.boolToString devel}" == "true" ]]; then
          mkdir -p "$out/logs"
          {
            echo "[debug] bazel-bin/bazel-out scan for missing CUDA DSOs"
            find -L bazel-bin bazel-out -type f -name "libnvshmem_host.so.3*" -o -name "nvshmem_bootstrap_uid.so.3*" -o -name "nvshmem_transport_ibrc.so.3*" -o -name "libnvJitLink.so.12*" -o -name "libnvrtc-builtins.so.12.9*" 2>/dev/null || true
            echo "[debug] bazel-bin/bazel-out scan for ptxas/nvlink"
            find -L bazel-bin bazel-out -type f -name "ptxas" -o -name "nvlink" 2>/dev/null || true
          } > "$out/logs/bazel-solib-scan.txt"
        fi

        # Ensure NVIDIA libs find the bundled sys libs
        for so in "$out/runtime/nvidia/"*/lib/*.so*; do
          [ -f "$so" ] || continue
          patchelf --set-rpath '$ORIGIN/../../../sys/lib' "$so" || true
        done
      ''}

      # Patch plugin rpaths to the bundled runtime lib dirs
      chmod u+w "$out/runtime/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"
      patchelf --set-rpath '$ORIGIN/../../../sys/lib' "$out/runtime/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"
      ${lib.optionalString cudaSupport ''
        cuda_rpath='${cudaPluginRpath}'
        chmod u+w "$out/runtime/xla/pjrt/c/pjrt_c_api_gpu_plugin.so"
        patchelf --set-rpath "$cuda_rpath" "$out/runtime/xla/pjrt/c/pjrt_c_api_gpu_plugin.so"
      ''}

      if [[ "${lib.boolToString devel}" == "true" ]]; then
        echo "[devel] copying additional bazel-bin artifacts (*.so*, *.a)"
        find bazel-bin -type f \( -name "*.so*" -o -name "*.a" \) -print -exec cp -vn {} "$out/lib/" \;

        if [[ "${lib.boolToString bazelLogEvents}" == "true" ]]; then
          # Preserve Bazel cache diagnostics for inspection.
          mkdir -p "$out/logs"
          if [[ -f bazel-bep.json ]]; then
            cp -v bazel-bep.json "$out/logs/"
          fi
          if [[ -f bazel-exec.json ]]; then
            cp -v bazel-exec.json "$out/logs/"
          fi
        fi
      fi

      # Provenance
      cat > "$out/runtime/PROVENANCE.json" <<EOF
      ${builtins.toJSON lock}
      EOF
    '';

    meta = {
      description = "XLA PJRT C API runtime plugins";
      platforms = lib.platforms.linux;
    };

    # Strip step is brittle for Bazel outputs in devel builds; keep symbols.
    dontStrip = devel;
  }
