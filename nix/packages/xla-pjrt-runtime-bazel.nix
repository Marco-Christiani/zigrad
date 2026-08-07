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
  # Optional explicit PJRT CUDA track selector. If null, inferred from cudaVersion.
  # Valid values: "cuda12" | "cuda13"
  pjrtCudaTrack ? null,
  cudaArchitectures ? null,
  # XLA source and its selected revision.
  xlaSrc,
  xlaRevision,
  depsHash,
  # CPU math library for the PJRT CPU plugin.
  #   "eigen"        - Eigen + XNNPACK only (default).
  #   "onednn"       - open-source oneDNN v3.7.3, JIT contraction kernel, threadpool.
  #   "onednn-thunk" - onednn + compiler rewrites eligible ops to oneDNN thunks (dev branch).
  #   "onednn-omp"   - same as "onednn" but uses OpenMP (libiomp5) instead of threadpool.
  #                    This variant differs only in threading model. Threadpool avoids
  #                    oversubscription with Eigen's threadpool.
  cpuMathLibrary ? "eigen",
  # Emit -march=native -mavx2 -mfma for both target and host.
  cpuNativeTuning ? false,
  # When true: pass --copt=-g to bazel, retain DWARF, don't strip.
  # When false (default, production): bazel -c opt only, stripped.
  withDebugSymbols ? false,
  # LTO uses Bazel's --features=thin_lto. Some XLA targets reject the feature
  #  with a warning.
  enableLto ? false,
  # Extra build flags appended to bazelBuildFlags. Use for one-off
  #  experiments (e.g. ["--copt=-funroll-loops"]).
  extraBazelFlags ? [],
  ...
}: let
  cudaVersionChecked =
    if cudaSupport && cudaVersion == null
    then throw "cudaVersion required when cudaSupport is true"
    else cudaVersion;

  pjrtCudaTrackResolved =
    if !cudaSupport
    then null
    else if pjrtCudaTrack != null
    then pjrtCudaTrack
    else if lib.hasPrefix "13" cudaVersionChecked
    then "cuda13"
    else "cuda12";

  expectedCudaMajor =
    if pjrtCudaTrackResolved == "cuda13"
    then "13"
    else "12";

  pjrtCudaConfig =
    if pjrtCudaTrackResolved == "cuda13"
    then "pjrt_cuda13"
    else "pjrt_cuda12";

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
  # build_with_mkl_opensource: include only open-source oneDNN.
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

      # OpenMP uses libiomp5 instead of the Eigen threadpool.
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
      build --config=${pjrtCudaConfig}
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

  validateHermeticCudaSupport = lib.optionalString cudaSupport ''
    export ZG_RULES_ML_TOOLCHAIN_CUDA_REDIST_VERSIONS="$bazelOut/external/rules_ml_toolchain/third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl"
    python3 - <<'PY'
    import os
    import pathlib
    import re
    import sys

    cuda_version = "${cudaVersionChecked}"
    pjrt_track = "${pjrtCudaTrackResolved}"
    expected_major = "${expectedCudaMajor}"
    rules_file = pathlib.Path(os.environ["ZG_RULES_ML_TOOLCHAIN_CUDA_REDIST_VERSIONS"])

    if not rules_file.exists():
        print(
            f"xla-pjrt-runtime-bazel: expected rules_ml_toolchain metadata at {rules_file}, but it does not exist",
            file=sys.stderr,
        )
        sys.exit(1)

    if not cuda_version.startswith(expected_major + "."):
        print(
            f"xla-pjrt-runtime-bazel: {pjrt_track} requires CUDA {expected_major}.x, got {cuda_version}",
            file=sys.stderr,
        )
        sys.exit(1)

    text = rules_file.read_text()

    cuda_redist_match = re.search(r"CUDA_REDIST_JSON_DICT\s*=\s*\{(.*?)\n\}", text, re.DOTALL)
    if not cuda_redist_match:
        print(
            f"xla-pjrt-runtime-bazel: failed to parse CUDA_REDIST_JSON_DICT from {rules_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    supported_versions = sorted(set(re.findall(r'"([0-9]+\.[0-9]+(?:\.[0-9]+)?)"\s*:', cuda_redist_match.group(1))))
    if cuda_version not in supported_versions:
        major_versions = [v for v in supported_versions if v.startswith(expected_major + ".")]
        print(
            "xla-pjrt-runtime-bazel: requested cudaVersion="
            f"{cuda_version} is not supported by pinned rules_ml_toolchain for {pjrt_track}. "
            f"Supported {expected_major}.x versions: {', '.join(major_versions)}",
            file=sys.stderr,
        )
        sys.exit(1)

    nccl_map_match = re.search(r"CUDA_NCCL_WHEELS\s*=\s*(\{.*?)(?:\n\n|# Ensures)", text, re.DOTALL)
    if not nccl_map_match:
        print(
            f"xla-pjrt-runtime-bazel: failed to parse CUDA_NCCL_WHEELS from {rules_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    nccl_expr = nccl_map_match.group(1).strip()
    if re.search(rf'"{re.escape(cuda_version)}"\s*:', nccl_expr):
        sys.exit(0)

    generated_dict = f"CUDA_{expected_major}_NCCL_WHEEL_DICT"
    generated_pattern = (
        rf'v:\s*{re.escape(generated_dict)}\s+for\s+v\s+in\s+CUDA_REDIST_JSON_DICT\.keys\(\)\s+'
        rf'if\s+v\.startswith\("{re.escape(expected_major)}"\)'
    )
    if generated_dict in text and re.search(generated_pattern, nccl_expr):
        sys.exit(0)

    print(
        "xla-pjrt-runtime-bazel: requested cudaVersion="
        f"{cuda_version} has no NCCL wheel mapping in pinned rules_ml_toolchain metadata ({rules_file})",
        file=sys.stderr,
    )
    sys.exit(1)
    PY
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
  assert lib.assertMsg
  (!cudaSupport || lib.elem pjrtCudaTrackResolved ["cuda12" "cuda13"])
  "xla-pjrt-runtime-bazel: invalid pjrtCudaTrack=${pjrtCudaTrackResolved}; expected cuda12 or cuda13";
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

      bazelBuildFlags =
        [
          "-c"
          "opt"
          "--nofetch"
        ]
        ++ lib.optionals withDebugSymbols [
          "--copt=-g"
          "--strip=never"
        ]
        ++ lib.optionals enableLto [
          "--copt=-flto=thin"
          "--linkopt=-flto=thin"
        ]
        ++ extraBazelFlags;

      inherit bazelTargets;

      buildAttrs = {
        pname = "xla-pjrt-plugins";
        version = "xla-${builtins.substring 0 7 xlaRevision}";
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

          ${validateHermeticCudaSupport}

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
          ${builtins.toJSON {xla-rev = xlaRevision;}}
          EOF
          mkdir -p "$out/runtime/logs"
          cp -v ./bazel-config.json "$out/runtime/BAZEL_CONFIG.json"
          cp -v ./bazel-config.json "$out/runtime/logs/bazel-config.json"
          if [ -f tensorflow.bazelrc ]; then
            head -n 120 tensorflow.bazelrc > "$out/runtime/logs/bazelrc-head.txt" || true
            grep -nE '^(common|build):' tensorflow.bazelrc > "$out/runtime/logs/bazelrc-configs.txt" || true
          fi

          # Restore -u to default so fixupPhase's strip-hook doesn't trip.
          set +u
        '';

        meta = {
          description = "XLA PJRT C API runtime plugins (CPU + GPU .so only)";
          longDescription = ''
            Builds XLA PJRT CPU/GPU plugin .so files via Bazel. CUDA runtime DSOs
            are provided separately by cuda-redist.nix and merged at the SDK level.
          '';
          platforms = lib.platforms.linux;
        };

        dontStrip = withDebugSymbols;
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
