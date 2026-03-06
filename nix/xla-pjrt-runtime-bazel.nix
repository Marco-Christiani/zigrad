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

  # Runtime dir names for CUDA DSOs. Used only for GPU plugin RPATH computation.
  # The actual CUDA DSOs are provided by cuda-redist.nix, not extracted from Bazel.
  cudaRuntimeNames = lib.optionals cudaSupport [
    "cudnn"
    "cublas"
    "cufft"
    "cusparse"
    "cudart"
    "cupti"
    "nvrtc"
    "nvjitlink"
    "nccl"
    "nvshmem"
  ];

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

        # Re-patch shebangs in the unpacked deps with current nixpkgs tools.
        # The deps tarball may contain stale shebangs from a previous fetch
        # (e.g. /nix/store/<old-hash>-bash that no longer exists).
        if [ -d "$bazelOut/external/rules_ml_toolchain" ]; then
          patchShebangs "$bazelOut/external/rules_ml_toolchain" || true
        fi

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
        mkdir -p "$out/runtime/xla/pjrt/c"

        for f in ${lib.concatStringsSep " " builtOutputs}; do
          if [[ ! -f "$f" ]]; then
            echo "ERROR: missing expected Bazel output: $f" >&2
            find bazel-bin -type f -name '*.so*' -maxdepth 8 -print >&2 || true
            exit 1
          fi
        done

        # Copy plugin .so files
        cpu_out="bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"
        cp -v --no-preserve=mode "$cpu_out" "$out/runtime/xla/pjrt/c/"

        ${lib.optionalString cudaSupport ''
          gpu_out="bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so"
          cp -v --no-preserve=mode "$gpu_out" "$out/runtime/xla/pjrt/c/"
        ''}

        # Patch plugin rpaths to the bundled runtime lib dirs.
        # CUDA DSOs and sys libs are provided by cuda-redist.nix (merged via symlinkJoin).
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
        description = "XLA PJRT C API runtime plugins (CPU + GPU .so only)";
        longDescription = ''
          Builds XLA PJRT CPU/GPU plugin .so files via Bazel. CUDA runtime DSOs
          are provided separately by cuda-redist.nix and merged at the SDK level.
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
