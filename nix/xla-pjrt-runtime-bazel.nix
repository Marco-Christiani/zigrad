{
  lib,
  stdenv,
  buildBazelPackage,
  # native build inputs
  python3,
  bazel_7,
  patchelf,
  file,
  zlib,
  ncurses,
  # CUDA (optional)
  cudaSupport ? false,
  cudaVersion ? null,
  cudaArchitectures ? null,
  # Copy NCCL/NVSHMEM DSOs into the runtime bundle (usually required for GPU).
  copyNcclNvshmem ? true,
  # Copy CUDA tools (ptxas/nvlink) into runtime/nvidia.
  copyCudaTools ? true,
  # Copy libdevice bitcode into runtime/nvidia/nvvm/libdevice.
  copyLibdevice ? true,
  # XLA source (flake input).
  xlaSrc,
  depsHash,
  # CPU math library for the PJRT CPU plugin.
  #   "eigen"        - Eigen + XNNPACK only (default).
  #   "onednn"       - open-source oneDNN v3.7.3, JIT contraction kernel, threadpool.
  #   "onednn-thunk" - onednn + compiler rewrites eligible ops to oneDNN thunks (dev branch).
  #   "onednn-omp"   - same as "onednn" but uses OpenMP (libiomp5) instead of threadpool.
  #                    Note: the old proprietary MKL-ML BLAS blobs have been removed from XLA;
  #                    this variant only differs in threading model. Threadpool is generally
  #                    preferred (avoids oversubscription with Eigen's threadpool).
  cpuMathLibrary ? "eigen",
  # Emit -march=native -mavx2 -mfma for both target and host.
  cpuNativeTuning ? false,
  ...
}: let
  cudaVersionChecked =
    if cudaSupport && cudaVersion == null
    then throw "cudaVersion required when cudaSupport is true"
    else cudaVersion;

  cudaRuntimeNames = lib.optionals cudaSupport (
    [
      "cudnn"
      "cublas"
      "cufft"
      "cusparse"
      "cudart"
      "cupti"
      "nvrtc"
      "nvjitlink"
    ]
    ++ lib.optionals copyNcclNvshmem [
      "nccl"
      "nvshmem"
    ]
  );

  cudaComputeCapabilities =
    if cudaSupport && cudaArchitectures != null
    then let
      toCap = arch: let
        len = builtins.stringLength arch;
      in
        if len == 2
        then "${builtins.substring 0 1 arch}.${builtins.substring 1 1 arch}"
        else if len == 3
        then "${builtins.substring 0 2 arch}.${builtins.substring 2 1 arch}"
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

  # -- oneDNN flags --------------------------------------------------------
  # build_with_mkl:          compile oneDNN sources, set XLA_ONEDNN / ENABLE_ONEDNN_V3 macros.
  # enable_mkl:              activate runtime paths (ENABLE_MKL). Without this, oneDNN compiles but is inert.
  # build_with_mkl_opensource: exclude proprietary MKL-ML blobs; only open-source oneDNN.
  # build_with_openmp:       oneDNN uses OpenMP (links libiomp5). Omit -> threadpool.
  # build_with_onednn_async: async thunk runtime (ENABLE_ONEDNN_ASYNC). Required for
  #                          IsOneDnnCompatible() -> compiler rewrites ops to __onednn$* custom calls.
  #                          Switches dep from stable v3.7.3 to dev-v3.7-thunk-preview.
  # tensorflow_mkldnn_contraction_kernel:
  #   =1  replace Eigen's SGEMM (gebp_kernel) with oneDNN JIT (avx/avx2/fma/avx512 via CPUID).
  #   =0  keep Eigen's default contraction kernel.

  onednnBaseFlags = ''
    build --define=build_with_mkl=true
    build --define=enable_mkl=true
    build --define=tensorflow_mkldnn_contraction_kernel=1
  '';

  asyncFlags = ''
    build --define=build_with_onednn_async=true
  '';

  mathLibraryFlags =
    {
      "eigen" = "";

      "onednn" = ''
        ${onednnBaseFlags}
        build --define=build_with_mkl_opensource=true
      '';

      "onednn-thunk" = ''
        ${onednnBaseFlags}
        build --define=build_with_mkl_opensource=true
        ${asyncFlags}
      '';

      # OpenMP threading variant. Only differs from "onednn" in threading model
      # (libiomp5 vs Eigen threadpool). The old proprietary MKL-ML BLAS blobs
      # have been removed from XLA; intel_binary_blob now just provides libiomp5.
      "onednn-omp" = ''
        ${onednnBaseFlags}
        build --define=build_with_openmp=true
      '';
    }.${
      cpuMathLibrary
    };

  nativeTuningFlags = lib.optionalString cpuNativeTuning ''
    build --copt=-march=native
    build --copt=-mtune=native
    build --copt=-mavx2
    build --copt=-mfma
    build --host_copt=-march=native
    build --host_copt=-mtune=native
    build --host_copt=-mavx2
    build --host_copt=-mfma
  '';

  cpuFlags = "${mathLibraryFlags}${nativeTuningFlags}";

  xlaBazelrc =
    ''
      try-import %workspace%/tensorflow.bazelrc
      common --enable_bzlmod=false
      common --experimental_repo_remote_exec
      build --enable_bzlmod=false
      build --repo_env PYTHON_BIN_PATH="${python3}/bin/python"
      build --python_path="${python3}/bin/python"
      build --action_env PYTHON_BIN_PATH="${python3}/bin/python"
      ${cpuFlags}
      common --host_linkopt=-Wl,--dynamic-linker=${stdenv.cc.bintools.dynamicLinker}
      common --host_linkopt=-Wl,-rpath,${lib.makeLibraryPath [stdenv.cc.cc stdenv.cc.libc zlib ncurses]}
      common --verbose_failures
    ''
    + lib.optionalString cudaSupport ''
      build --config=pjrt_cuda12
      build --action_env TF_CUDA_VERSION="${cudaVersionChecked}"
      ${lib.optionalString (cudaComputeCapabilities != null)
        "build --action_env TF_CUDA_COMPUTE_CAPABILITIES=\"${cudaComputeCapabilities}\""}
    '';

  cudaPluginRpath = lib.optionalString cudaSupport (
    lib.concatStringsSep ":" (
      [
        "$ORIGIN/../../../sys/lib"
      ]
      ++ map (name: "$ORIGIN/../../../nvidia/${name}/lib") cudaRuntimeNames
    )
  );

  postPatchScript = ''
    rm -f .bazelversion || true
    substituteInPlace third_party/extensions/python_version.bzl \
      --replace-quiet 'USE_PYWRAP_RULES = {use_pywrap_rules}' \
                'USE_PYWRAP_RULES = {use_pywrap_rules}\nHERMETIC_PYTHON_URL = ""\nHERMETIC_PYTHON_SHA256 = ""\nHERMETIC_PYTHON_PREFIX = "python"'
    substituteInPlace third_party/py/python_init_toolchains.bzl \
      --replace-quiet '            python_version_kind = HERMETIC_PYTHON_VERSION_KIND,' ""
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

  writeBazelrc = ''
    cat > ./.bazelrc <<'CFG'
    ${xlaBazelrc}
    CFG
  '';

  fetchPreInstall = ''
    set -euo pipefail

    tool_rpath="\$ORIGIN/../lib:\$ORIGIN/../lib64:${lib.makeLibraryPath [stdenv.cc.cc zlib ncurses]}"

    patch_elf_one() {
      local p="$1"
      [ -e "$p" ] || return 0

      local real="$p"
      if [ -L "$p" ] && command -v readlink >/dev/null 2>&1; then
        real="$(readlink -f "$p" 2>/dev/null || echo "$p")"
      fi
      [ -f "$real" ] || return 0

      chmod u+w "$real" 2>/dev/null || true

      if file -L "$p" 2>/dev/null | grep -q 'ELF '; then
        local interp=""
        interp="$(patchelf --print-interpreter "$real" 2>/dev/null || true)"
        case "$interp" in
          /lib64/ld-linux-x86-64.so.2|/lib/ld-linux-x86-64.so.2)
            patchelf --set-interpreter "${stdenv.cc.bintools.dynamicLinker}" "$real" || true
            patchelf --set-rpath "$tool_rpath" "$real" || true
            ;;
        esac
      fi
    }

    patch_elf_tree() {
      local root="$1"
      [ -d "$root" ] || return 0
      chmod -R u+w "$root" 2>/dev/null || true

      find -L "$root" \
        \( -path '*/bin/*' -o -path '*/tools/*' -o -path '*/.runfiles/*/bin/*' -o -path '*/python_*/*/bin/*' \) \
        \( -type f -o -type l \) \
        -print0 2>/dev/null |
      while IFS= read -r -d $'\0' f; do
        patch_elf_one "$f"
      done
    }

    patch_wrappers_shebangs() {
      local root="$1"
      [ -d "$root" ] || return 0
      find -L "$root" -type d -name wrappers -print0 2>/dev/null |
      while IFS= read -r -d $'\0' d; do
        chmod -R u+w "$d" 2>/dev/null || true
        patchShebangs "$d" || true
      done
    }

    patch_wrappers_shebangs "$bazelOut/external/rules_ml_toolchain"
      if [ -d "$bazelOut/external/rules_ml_toolchain" ]; then
        while IFS= read -r f; do
          sed -i "s|/usr/bin/env python3|${python3}/bin/python|g" "$f" || true
          sed -i "s|/usr/bin/env python|${python3}/bin/python|g" "$f" || true
        done < <(grep -rl "/usr/bin/env python3" "$bazelOut/external/rules_ml_toolchain" 2>/dev/null || true)
        while IFS= read -r f; do
          sed -i "s|/usr/bin/env python|${python3}/bin/python|g" "$f" || true
        done < <(grep -rl "/usr/bin/env python" "$bazelOut/external/rules_ml_toolchain" 2>/dev/null || true)
      fi
    patch_elf_tree "$bazelOut/external"
    patch_elf_tree "$bazelOut/execroot"
  '';
in
  buildBazelPackage {
    name =
      "xla-pjrt-plugins"
      + lib.optionalString cudaSupport "-cuda";

    removeRulesCC = false;

    bazel = bazel_7;
    dontAddBazelOpts = true;

    bazelFlags = [
      "--enable_bzlmod=false"
    ];

    bazelBuildFlags = [
      "-c"
      "opt"
      "--nofetch"
    ];

    inherit bazelTargets;

    buildAttrs = {
      pname = "xla-pjrt-plugins";
      version = "xla-${xlaSrc.shortRev or "unknown"}";
      src = xlaSrc;

      nativeBuildInputs = [
        python3
        patchelf
        file
      ];

      postPatch = postPatchScript;

      preBuild = ''
        ${writeBazelrc}

        python3 ${./parse_bazelrc.py} --output ./bazel-config.json

        install_base="$(bazel --batch --output_base="$bazelOut" --output_user_root="$bazelUserRoot" info install_base)"
        rm -rf "$bazelOut/external/bazel_tools"
        ln -s "$install_base/embedded_tools" "$bazelOut/external/bazel_tools"
        rm -rf "$bazelOut/external/rules_java_builtin"
        ln -s "$install_base/rules_java" "$bazelOut/external/rules_java_builtin"
        rm -rf "$bazelOut/external/internal_platforms_do_not_use"
        ln -s "$install_base/platforms" "$bazelOut/external/internal_platforms_do_not_use"

        while IFS= read -r f; do
          sed -i "s|/usr/bin/env python3|${python3}/bin/python|g" "$f" || true
          sed -i "s|/usr/bin/env python|${python3}/bin/python|g" "$f" || true
        done < <(grep -rl "/usr/bin/env python" "$install_base" 2>/dev/null || true)

        if [ -d "$bazelOut/external/XNNPACK" ]; then
          patchShebangs "$bazelOut/external/XNNPACK" || true
          substituteInPlace "$bazelOut/external/XNNPACK/ynnpack/build_defs.bzl" \
            --replace 'cmd = "$(location ' 'cmd = "$${PYTHON_BIN_PATH} $(location '
        fi

        if [ -f "$bazelOut/external/rules_python/python/private/python_bootstrap_template.txt" ]; then
          sed -i "s|%shebang%|#!${python3}/bin/python|g" \
            "$bazelOut/external/rules_python/python/private/python_bootstrap_template.txt"
          sed -i "s|/usr/bin/env python3|${python3}/bin/python|g" \
            "$bazelOut/external/rules_python/python/private/python_bootstrap_template.txt"
          sed -i "s|/usr/bin/env python|${python3}/bin/python|g" \
            "$bazelOut/external/rules_python/python/private/python_bootstrap_template.txt"
        fi

      '';

      installPhase = ''
        set -euo pipefail
        mkdir -p "$out/runtime/xla/pjrt/c" "$out/runtime/sys/lib"

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
        cp -v ${zlib}/lib/libz.so.1 "$out/runtime/sys/lib/"

        ${lib.optionalString cudaSupport ''
          # Prefer Bazel runfiles (_solib) for hermetic CUDA/NVSHMEM DSOs.
          log_copy() {
            echo "[cuda-copy] $*"
          }

          is_valid_tool() {
            local p="$1"
            [ -f "$p" ] || return 1
            [ -s "$p" ] || return 1
            [ -x "$p" ] || return 1
            file -L "$p" | grep -q 'ELF ' || return 1
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
            local find_roots=()
            if [ -d "bazel-bin" ]; then
              find_roots+=("bazel-bin")
            fi
            if [ -d "bazel-out" ]; then
              find_roots+=("bazel-out")
            fi
            if [ "''${#find_roots[@]}" -eq 0 ]; then
              log_copy "fallback find skipped: no bazel roots found"
              return 0
            fi
            while IFS= read -r -d $'\0' f; do
              copy_one "$f" "$dest"
            done < <(find -L "''${find_roots[@]}" -type f -name "$pattern" -print0 2>/dev/null || true)
            chmod -R u+w "$dest"
          }

          copy_find "libcublas*.so*" "$out/runtime/nvidia/cublas/lib"
          copy_find "libcublasLt*.so*" "$out/runtime/nvidia/cublas/lib"
          copy_find "libcudnn*.so*" "$out/runtime/nvidia/cudnn/lib"
          copy_find "libcufft*.so*" "$out/runtime/nvidia/cufft/lib"
          copy_find "libcupti*.so*" "$out/runtime/nvidia/cupti/lib"
          copy_find "libcusparse*.so*" "$out/runtime/nvidia/cusparse/lib"
          copy_find "libcudart*.so*" "$out/runtime/nvidia/cudart/lib"
          copy_find "libnvrtc*.so*" "$out/runtime/nvidia/nvrtc/lib"
          copy_find "libnvJitLink*.so*" "$out/runtime/nvidia/nvjitlink/lib"
          ${lib.optionalString copyNcclNvshmem ''
            copy_find "libnccl*.so*" "$out/runtime/nvidia/nccl/lib"
            copy_find "libnvshmem_host.so.3*" "$out/runtime/nvidia/nvshmem/lib"
            copy_find "nvshmem_bootstrap_uid.so.3*" "$out/runtime/nvidia/nvshmem/lib"
            copy_find "nvshmem_transport_ibrc.so.3*" "$out/runtime/nvidia/nvshmem/lib"
          ''}
          copy_find "libnvrtc-builtins.so.*" "$out/runtime/nvidia/nvrtc/lib"

          ${lib.optionalString copyCudaTools ''
            copy_tool_safe() {
              local name="$1"
              local dest="$2"

              mkdir -p "$dest"

              tool_candidates=(
                "bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/cuda_nvcc/bin"
                "bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/xla/external/cuda_nvcc/bin"
                "bazel-out/k8-opt/bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/cuda_nvcc/bin"
                "bazel-out/k8-opt/bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/xla/external/cuda_nvcc/bin"
              )

              for d in "''${tool_candidates[@]}"; do
                candidate="$d/$name"
                if is_valid_tool "$candidate"; then
                  copy_one "$candidate" "$dest"
                  return 0
                fi
              done

              echo "ERROR: no valid $name found (ELF + executable + non-empty)" >&2
              return 1
            }

            copy_tool_safe "ptxas"  "$out/runtime/nvidia/cuda_nvcc/bin"
            copy_tool_safe "nvlink" "$out/runtime/nvidia/cuda_nvcc/bin"

            # Mirror into <cuda_data_dir>/bin (XLA lookup path)
            copy_one "$out/runtime/nvidia/cuda_nvcc/bin/ptxas"  "$out/runtime/nvidia/bin"
            copy_one "$out/runtime/nvidia/cuda_nvcc/bin/nvlink" "$out/runtime/nvidia/bin"
          ''}

          ${lib.optionalString copyLibdevice ''
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
          ''}

          # Ensure NVIDIA libs find the bundled sys libs AND sibling libs in same directory
          # ($ORIGIN needed for libnvrtc -> libnvrtc-builtins internal dlopen)
          for so in "$out/runtime/nvidia/"*/lib/*.so*; do
            [ -f "$so" ] || continue
            patchelf --set-rpath '$ORIGIN:$ORIGIN/../../../sys/lib' "$so" || true
          done

          if ! find "$out/runtime/nvidia" -type f -name "*.so*" -print -quit | grep -q .; then
            echo "ERROR: no CUDA DSOs copied into runtime/nvidia; check Bazel runfiles paths." >&2
            exit 1
          fi
        ''}

        # Patch plugin rpaths to the bundled runtime lib dirs
        chmod u+w "$out/runtime/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"
        patchelf --set-rpath '$ORIGIN/../../../sys/lib' "$out/runtime/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"
        ${lib.optionalString cudaSupport ''
          cuda_rpath='${cudaPluginRpath}'
          chmod u+w "$out/runtime/xla/pjrt/c/pjrt_c_api_gpu_plugin.so"
          patchelf --set-rpath "$cuda_rpath" "$out/runtime/xla/pjrt/c/pjrt_c_api_gpu_plugin.so"
        ''}

        # Provenance
        cat > "$out/runtime/PROVENANCE.json" <<EOF
        ${builtins.toJSON {xla-rev = xlaSrc.rev or xlaSrc.shortRev or "unknown";}}
        EOF
        mkdir -p "$out/runtime/logs"
        cp -v ./bazel-config.json "$out/runtime/BAZEL_CONFIG.json"
        cp -v ./bazel-config.json "$out/runtime/logs/bazel-config.json"
        if [ -f tensorflow.bazelrc ]; then
          head -n 120 tensorflow.bazelrc > "$out/runtime/logs/bazelrc-head.txt" || true
          grep -nE '^(common|build):' tensorflow.bazelrc > "$out/runtime/logs/bazelrc-configs.txt" || true
        fi
      '';

      meta = {
        description = "XLA PJRT C API runtime plugins";
        longDescription = ''
          Bundles XLA PJRT CPU/GPU plugins plus a self-contained CUDA runtime
          tree. CUDA/NVSHMEM DSOs are copied from Bazel runfiles (_solib).
        '';
        platforms = lib.platforms.linux;
      };

      dontStrip = true;
    };

    fetchAttrs = {
      hash = depsHash;
      src = xlaSrc;
      nativeBuildInputs = [
        patchelf
        file
      ];
      postPatch = postPatchScript;
      preBuild = writeBazelrc;
      preInstall = fetchPreInstall;
      installPhase = ''
        runHook preInstall

        # Remove all vcs files
        rm -rf $(find $bazelOut/external -type d -name .git)
        rm -rf $(find $bazelOut/external -type d -name .svn)
        rm -rf $(find $bazelOut/external -type d -name .hg)

        if [ -e "$bazelOut/external/bazel_tools" ]; then
          echo "[deps] bazel_tools already present at $bazelOut/external/bazel_tools"
        else
          install_base="$(bazel info install_base)"
          echo "[deps] creating bazel_tools symlink to $install_base/embedded_tools"
          ln -s "$install_base/embedded_tools" "$bazelOut/external/bazel_tools"
        fi

        # Patching symlinks to remove build directory reference
        find $bazelOut/external -type l | while read symlink; do
          new_target="$(readlink "$symlink" | sed "s,$NIX_BUILD_TOP,NIX_BUILD_TOP,")"
          rm "$symlink"
          ln -sf "$new_target" "$symlink"
        done

        echo '${bazel_7.name}' > $bazelOut/external/.nix-bazel-version

        (cd $bazelOut/ && tar czf $out --sort=name --mtime='@1' --owner=0 --group=0 --numeric-owner external/)

        runHook postInstall
      '';
    };
  }
