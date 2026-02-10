{
  lib,
  stdenv,
  fetchurl,
  runCommand,
  # native build inputs
  python3,
  bazel_7,
  symlinkJoin,
  binutils,
  patchelf,
  gnutar,
  xz,
  gzip,
  file,
  zlib,
  ncurses,
  # CUDA (optional)
  cudaSupport ? false,
  cudaPackages ? null,
  cudaVersion ? null,
  cudaArchitectures ? null,
  # Copy CUDA runtime libs from nixpkgs into the bundle (can be huge).
  copyCudaFromNix ? false,
  # Copy NCCL/NVSHMEM DSOs into the runtime bundle (usually required for GPU).
  copyNcclNvshmem ? true,
  # Copy CUDA tools (ptxas/nvlink) into runtime/nvidia.
  copyCudaTools ? true,
  # Copy libdevice bitcode into runtime/nvidia/nvvm/libdevice.
  copyLibdevice ? true,
  # Use cudaPackages.backendStdenv for CUDA builds. This can pull a large
  # toolchain closure into the build environment.
  useCudaStdenv ? true,
  lockFile,
  xlaSrcOverride ? null,
  devel ? false,
  # Dev-speed knobs (keep false for hermetic builds)
  persistentBazelOutputBase ? false,
  bazelLogEvents ? false,
  extraCpuFlags ? false,
}: let
  # -----------------------------------------------------------------------------
  # XLA PJRT runtime bundle
  #
  # Outputs:
  #   $out/runtime/xla/pjrt/c/pjrt_c_api_{cpu,gpu}_plugin.so
  #   $out/runtime/sys/lib/{libstdc++.so.6,libgcc_s.so.1}
  #   $out/runtime/nvidia/... (CUDA runtime + tools when cudaSupport=true)
  #
  # CUDA bundle strategy:
  #   - Default: copy CUDA/NVSHMEM DSOs + tools from Bazel runfiles/_solib
  #     (hermetic, avoids nixpkgs CUDA runtime closure).
  #   - Optional: copy from nixpkgs CUDA packages (copyCudaFromNix=true).
  #
  # Key switches:
  #   - cudaSupport: build GPU plugin and populate runtime/nvidia
  #   - copyCudaFromNix: include nixpkgs CUDA runtime DSOs in bundle
  #   - useCudaStdenv: use cudaPackages.backendStdenv (can enlarge build closure)
  #   - devel: keep symbols, copy extra bazel-bin artifacts, write debug logs
  #   - bazelLogEvents: write BEP/exec logs (can be multi-GB) when devel=true
  #
  # Expected Bazel output layout (for runfiles discovery):
  #   bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/...
  #   bazel-out/.../xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/...
  #   or under Bazel output_base (queried via `bazel info output_base`).
  # -----------------------------------------------------------------------------
  lock = builtins.fromJSON (builtins.readFile lockFile);
  inherit (lock.pins) xla;

  xlaTar = fetchurl {
    url = xla.tarball_url;
    hash = xla.hash_sri;
  };

  xlaSrc =
    if xlaSrcOverride != null
    then
      builtins.path {
        path = xlaSrcOverride;
        name = "xla-src-local";
        filter = path: type: true;
      }
    else
      runCommand "xla-src-${builtins.substring 0 12 xla.commit}" {} ''
        set -euo pipefail
        mkdir -p "$out"
        tar -xzf ${xlaTar} -C "$out" --strip-components=1
      '';

  cudaPackagesRequired = cudaSupport && (copyCudaFromNix || useCudaStdenv);

  cudaPackagesChecked =
    if cudaPackagesRequired && cudaPackages == null
    then throw "cudaPackages required when copyCudaFromNix or useCudaStdenv is true"
    else cudaPackages;

  cudaVersionChecked =
    if cudaSupport && cudaVersion == null
    then throw "cudaVersion required when cudaSupport is true"
    else cudaVersion;

  effectiveStdenv =
    if cudaSupport && useCudaStdenv
    then cudaPackagesChecked.backendStdenv
    else stdenv;

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

  cudaRuntimeLibs =
    lib.optionals (cudaSupport && copyCudaFromNix)
    (
      [
        {
          name = "cudnn";
          pkg = lib.getLib cudaPackagesChecked.cudnn;
        }
        {
          name = "cublas";
          pkg = lib.getLib cudaPackagesChecked.libcublas;
        }
        {
          name = "cufft";
          pkg = lib.getLib cudaPackagesChecked.libcufft;
        }
        # TODO: No DT_NEEDED evidence yet for curand/cusolver. Enable if runtime requires.
        # {
        #   name = "curand";
        #   pkg = lib.getLib cudaPackagesChecked.libcurand;
        # }
        # {
        #   name = "cusolver";
        #   pkg = lib.getLib cudaPackagesChecked.libcusolver;
        # }
        {
          name = "cusparse";
          pkg = lib.getLib cudaPackagesChecked.libcusparse;
        }
        {
          name = "cudart";
          pkg = lib.getLib cudaPackagesChecked.cuda_cudart;
        }
        {
          name = "cupti";
          pkg = lib.getLib cudaPackagesChecked.cuda_cupti;
        }
        {
          name = "nvrtc";
          pkg = lib.getLib cudaPackagesChecked.cuda_nvrtc;
        }
      ]
      ++ lib.optionals (cudaPackagesChecked ? cuda_nvjitlink) [
        {
          name = "nvjitlink";
          pkg = lib.getLib cudaPackagesChecked.cuda_nvjitlink;
        }
      ]
      ++ lib.optionals (cudaPackagesChecked ? nccl) [
        {
          name = "nccl";
          pkg = lib.getLib cudaPackagesChecked.nccl;
        }
      ]
      ++ lib.optionals (cudaPackagesChecked ? nvshmem) [
        {
          name = "nvshmem";
          pkg = lib.getLib cudaPackagesChecked.nvshmem;
        }
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

  cpuFlags = lib.optionalString extraCpuFlags ''
    build --config=mkl_threadpool
    build --define=tensorflow_mkldnn_contraction_kernel=1
    build --copt=-march=native
    build --copt=-mtune=native
    build --copt=-mavx2
    build --copt=-mfma
    build --host_copt=-march=native
    build --host_copt=-mtune=native
    build --host_copt=-mavx2
    build --host_copt=-mfma'';

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
      ${cpuFlags}

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
      # build --sandbox_writable_path=/tmp

      common --host_linkopt=-Wl,--dynamic-linker=${stdenv.cc.bintools.dynamicLinker}
      common --host_linkopt=-Wl,-rpath,${lib.makeLibraryPath [stdenv.cc.cc stdenv.cc.libc zlib ncurses]}
      common --verbose_failures
      common --sandbox_debug
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
in
  effectiveStdenv.mkDerivation {
    pname = "xla-pjrt-plugins";
    version =
      "xla-${builtins.substring 0 12 xla.commit}"
      + lib.optionalString (xlaSrcOverride != null) "-local"
      + lib.optionalString devel "-devel"
      + lib.optionalString cudaSupport "-cuda";

    src = xlaSrc;

    nativeBuildInputs = [
      bazel_7
      python3
      binutils
      patchelf
      gnutar
      xz
      gzip
      file
    ];

    buildInputs =
      lib.optionals cudaSupport [
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
      # Ensure CPU PJRT plugin links PJRT compiler registration for AOT.
      python3 - <<'PY'
      from pathlib import Path

      path = Path("xla/pjrt/c/BUILD")
      text = path.read_text()
      if "xla/pjrt/cpu:cpu_pjrt_compiler" in text:
          raise SystemExit(0)

      name_anchor = 'name = "pjrt_c_api_cpu_plugin.so"'
      name_idx = text.find(name_anchor)
      if name_idx < 0:
          raise SystemExit("failed to patch xla/pjrt/c/BUILD: plugin target not found")

      deps_anchor = "deps = ["
      deps_idx = text.find(deps_anchor, name_idx)
      if deps_idx < 0:
          raise SystemExit("failed to patch xla/pjrt/c/BUILD: deps block not found")

      insert_pos = deps_idx + len(deps_anchor)
      insert_text = '\n        "//xla/pjrt/cpu:cpu_pjrt_compiler",'
      updated = text[:insert_pos] + insert_text + text[insert_pos:]
      path.write_text(updated)
      PY
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
        # -----------------------------------------------------------------------------
        # Build strategy (NixOS + Bazel hermetic toolchains)
        #
        # Problem:
        #   Bazel downloads some prebuilt, dynamically-linked Linux executables (LLVM
        #   toolchain, hermetic Python, etc). On NixOS these fail at runtime because
        #   their ELF interpreter is /lib64/ld-linux-x86-64.so.2 (stub-ld on NixOS).
        #
        # Approach:
        #   1) Use a deterministic Bazel output_base inside the build directory.
        #   2) Prime externals with `bazel build --nobuild` to populate output_base.
        #   3) Patch:
        #        - scripts: patch shebangs for wrapper dirs
        #        - ELFs: rewrite stub interpreter -> real glibc loader + add minimal RPATH
        #      This preserves Bazel's pinned deps and patches; we only make executables runnable.
        #   4) Run the real `bazel build` using the same output_base.
        # -----------------------------------------------------------------------------

        runHook preBuild

        python3 ${./parse_bazelrc.py} --output ./bazel-config.json

        echo "[bazelrc] wrote ./bazel-config.json"
        echo "[bazelrc] parsed $(python3 - <<'PY'
      import json
      data = json.load(open("./bazel-config.json", "r", encoding="utf-8"))
      src = data.get("source") or "unknown"
      count = len(data.get("available_configs", []))
      print(f"source={src} configs={count}")
      PY
        )"
        echo "[bazelrc] cuda (resolved): $(python3 - <<'PY'
      import json
      data = json.load(open("./bazel-config.json", "r", encoding="utf-8"))
      for key in ("cuda", "pjrt_cuda12", "pjrt_cuda13"):
          env = data.get("repo_env_resolved", {}).get(key, {})
          cuda = env.get("HERMETIC_CUDA_VERSION", "n/a")
          cudnn = env.get("HERMETIC_CUDNN_VERSION", "n/a")
          nvsh = env.get("HERMETIC_NVSHMEM_VERSION", "n/a")
          print(f"{key}={cuda} (cudnn={cudnn}, nvshmem={nvsh})")
      PY
        )"

        # Always deterministic output_base for this derivation.
        OUTPUT_BASE="$PWD/.bazel-output-base"
        output_base_arg=( "--output_base=$OUTPUT_BASE" )

        # Keep Bazel rc usage hermetic for this build.
        # bazel_rc_args=( --nosystem_rc --nohome_rc )
        bazel_rc_args=( --nohome_rc )

        # Record the output_base for installPhase (copy_find fallback) and debugging.
        bazel "''${bazel_rc_args[@]}" "''${output_base_arg[@]}" info output_base | tee ./bazel-output-base-path

        ${lib.optionalString devel ''
        bazel "''${bazel_rc_args[@]}" "''${output_base_arg[@]}" info output_base
        bazel "''${bazel_rc_args[@]}" "''${output_base_arg[@]}" info repository_cache
      ''}

        # 1) Prime: populate external repos + runfiles without compiling targets.
        bazel \
          --batch \
          "''${bazel_rc_args[@]}" \
          "''${output_base_arg[@]}" \
          build \
          --nobuild \
          -c opt \
          ${lib.concatStringsSep " " bazelTargets}

        # Minimal RPATH fallback for prebuilt tools.
        # Prefer bundled libs first, then a small Nix fallback for common deps.
        tool_rpath="\$ORIGIN/../lib:\$ORIGIN/../lib64:${lib.makeLibraryPath [stdenv.cc.cc zlib ncurses]}"

        patch_elf_one() {
          local p="$1"
          [ -e "$p" ] || return 0

          # Follow symlinks so we patch the actual ELF.
          local real="$p"
          if [ -L "$p" ] && command -v readlink >/dev/null 2>&1; then
            real="$(readlink -f "$p" 2>/dev/null || echo "$p")"
          fi
          [ -f "$real" ] || return 0

          chmod u+w "$real" 2>/dev/null || true

          # Only touch ELF files whose interpreter is the NixOS stub loader.
          if file -L "$p" 2>/dev/null | grep -q 'ELF '; then
            local interp=""
            interp="$(patchelf --print-interpreter "$real" 2>/dev/null || true)"
            case "$interp" in
              /lib64/ld-linux-x86-64.so.2|/lib/ld-linux-x86-64.so.2)
                echo "[elf-fix] patch $p (real=$real)"
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

          # Keep scan bounded so it scales:
          # - bin/*, tools/*, and runfiles/*/bin/* are the typical "executed tools" locations.
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

          # Patch wrapper script shebangs (common source of ENOENT on NixOS sandboxes).
          find -L "$root" -type d -name wrappers -print0 2>/dev/null |
          while IFS= read -r -d $'\0' d; do
            chmod -R u+w "$d" 2>/dev/null || true
            echo "[shebangs] patchShebangs $d"
            patchShebangs "$d" || true
          done
        }

        # 2) Apply fixes:
        #   - scripts in rules_ml_toolchain wrappers
        #   - prebuilt ELF tools under external + execroot (covers hermetic python in runfiles)
        patch_wrappers_shebangs "$OUTPUT_BASE/external/rules_ml_toolchain"
        patch_elf_tree "$OUTPUT_BASE/external"
        patch_elf_tree "$OUTPUT_BASE/execroot"

        # 3) Extra hardening: ensure the LLVM clang entrypoint is actually patched.
        tc="$OUTPUT_BASE/external/llvm18_linux_x86_64"
        if [ -d "$tc/bin" ]; then
          chmod -R u+w "$tc" 2>/dev/null || true
          for b in clang clang-18 clang++ clang++-18 ld.lld ld.lld-18 llvm-ar llvm-strip; do
            patch_elf_one "$tc/bin/$b"
          done

          clang_real="$(readlink -f "$tc/bin/clang" 2>/dev/null || echo "$tc/bin/clang")"
          echo "[toolchain] clang real=$clang_real interp=$(patchelf --print-interpreter "$clang_real" 2>/dev/null || true)"
          if [ "$(patchelf --print-interpreter "$clang_real" 2>/dev/null || true)" != "${stdenv.cc.bintools.dynamicLinker}" ]; then
            echo "ERROR: clang still has stub interpreter after patch pass" >&2
            ls -la "$tc/bin" >&2 || true
            exit 1
          fi
        else
          echo "ERROR: expected llvm toolchain missing at $tc/bin" >&2
          find "$OUTPUT_BASE/external" -maxdepth 2 -type d -name 'llvm*' -print >&2 || true
          exit 1
        fi

        # --- hard patch + assert: llvm toolchain (clang / clang-18 etc.) ----------------
        tc="$OUTPUT_BASE/external/llvm18_linux_x86_64"
        [ -d "$tc/bin" ] || { echo "ERROR: missing $tc/bin" >&2; exit 1; }
        chmod -R u+w "$tc" 2>/dev/null || true

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

          # Treat symlinks as their targets
          if file -L "$p" 2>/dev/null | grep -q 'ELF '; then
            local interp
            interp="$(patchelf --print-interpreter "$real" 2>/dev/null || true)"
            case "$interp" in
              /lib64/ld-linux-x86-64.so.2|/lib/ld-linux-x86-64.so.2)
                patchelf --set-interpreter "${stdenv.cc.bintools.dynamicLinker}" "$real"
                patchelf --set-rpath "$tool_rpath" "$real" || true
                ;;
            esac
          fi
        }

        # Patch both entrypoint symlinks and common versioned targets.
        for b in clang clang-18 clang++ clang++-18 ld.lld ld.lld-18 llvm-ar llvm-strip; do
          patch_elf_one "$tc/bin/$b"
        done

        clang_real="$(readlink -f "$tc/bin/clang" 2>/dev/null || echo "$tc/bin/clang")"
        echo "[toolchain] clang real=$clang_real"
        echo "[toolchain] clang interp=$(patchelf --print-interpreter "$clang_real" 2>/dev/null || true)"
        echo "[toolchain] clang rpath=$(patchelf --print-rpath "$clang_real" 2>/dev/null || true)"

        if [ "$(patchelf --print-interpreter "$clang_real" 2>/dev/null || true)" != "${stdenv.cc.bintools.dynamicLinker}" ]; then
          echo "ERROR: clang still has stub interpreter after patch" >&2
          ls -la "$tc/bin" >&2 || true
          exit 1
        fi
        # -------------------------------------------------------------------------------

        # 4) Real build (must use the same output_base as priming + patching).
        bazel \
          --batch \
          "''${bazel_rc_args[@]}" \
          "''${output_base_arg[@]}" \
          build \
          --nofetch \
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
      cp -v ${zlib}/lib/libz.so.1 "$out/runtime/sys/lib/"

      ${lib.optionalString (cudaSupport && copyCudaFromNix) ''
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

      ${lib.optionalString (cudaSupport && copyCudaFromNix) (lib.concatStringsSep "\n" (map (entry: ''
          copy_cuda_lib "${entry.name}" "${entry.pkg}"
        '')
        cudaRuntimeLibs))}

      ${lib.optionalString cudaSupport ''
        output_base=""
        if [ -f ./bazel-output-base-path ]; then
          output_base="$(cat ./bazel-output-base-path)"
        fi

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
          if [ -n "$output_base" ] && [ -d "$output_base" ]; then
            find_roots+=("$output_base")
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

            # 1) Prefer canonical Bazel CUDA repo location
            if [ -n "$output_base" ]; then
              candidate="$output_base/external/cuda_nvcc/bin/$name"
              if is_valid_tool "$candidate"; then
                log_copy "tool ok: $name from $candidate"
                copy_one "$candidate" "$dest"
                return 0
              fi
            fi

            # 2) Try runfiles candidates
            tool_candidates=(
              "bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/cuda_nvcc/bin"
              "bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/xla/external/cuda_nvcc/bin"
              "bazel-out/k8-opt/bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/cuda_nvcc/bin"
              "bazel-out/k8-opt/bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/xla/external/cuda_nvcc/bin"
            )

            for d in "''${tool_candidates[@]}"; do
              candidate="$d/$name"
              if is_valid_tool "$candidate"; then
                log_copy "tool ok: $name from $candidate"
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

        if [[ "${lib.boolToString devel}" == "true" ]]; then
          mkdir -p "$out/logs"
          {
            echo "[debug] bazel-bin/bazel-out scan for missing CUDA DSOs"
            find -L bazel-bin bazel-out -type f -name "libnvshmem_host.so.3*" -o -name "nvshmem_bootstrap_uid.so.3*" -o -name "nvshmem_transport_ibrc.so.3*" -o -name "libnvJitLink.so.12*" -o -name "libnvrtc-builtins.so.12.9*" 2>/dev/null || true
            echo "[debug] bazel-bin/bazel-out scan for ptxas/nvlink"
            find -L bazel-bin bazel-out -type f -name "ptxas" -o -name "nvlink" 2>/dev/null || true
          } > "$out/logs/bazel-solib-scan.txt"
        fi

        # Ensure NVIDIA libs find the bundled sys libs and sibling libs in same directory
        # E.g., sibling ($ORIGIN) needed for libnvrtc -> libnvrtc-builtins internal dlopen
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
      mkdir -p "$out/runtime/logs"
      cp -v ./bazel-config.json "$out/runtime/BAZEL_CONFIG.json"
      cp -v ./bazel-config.json "$out/runtime/logs/bazel-config.json"
      if [ -f tensorflow.bazelrc ]; then
        head -n 120 tensorflow.bazelrc > "$out/runtime/logs/bazelrc-head.txt" || true
        grep -nE '^(common|build):' tensorflow.bazelrc > "$out/runtime/logs/bazelrc-configs.txt" || true
      fi
      # Lightweight README for consumers inspecting the runtime bundle.
      cat > "$out/runtime/README.txt" <<'EOF'
      zigrad XLA PJRT runtime bundle

      Layout:
        runtime/xla/pjrt/c/            PJRT C API plugins (cpu/gpu)
        runtime/sys/lib/              libstdc++ and libgcc_s for plugin rpaths
        runtime/nvidia/               CUDA runtime DSOs, tools, nvvm/libdevice

      CUDA bundle behavior:
        - Default: copy CUDA/NVSHMEM DSOs + tools from Bazel runfiles/_solib.
        - copyCudaFromNix=true: also copy CUDA runtime libs from nixpkgs.

      Build options of note:
        - cudaSupport: build GPU plugin and populate runtime/nvidia
        - cudaVersion: TF_CUDA_VERSION (should match Bazel hermetic CUDA)
        - useCudaStdenv: use cudaPackages.backendStdenv (larger build closure)
        - devel: copy extra bazel-bin artifacts into $out/lib
        - bazelLogEvents: emit BEP/exec logs (can be huge) when devel=true

      Expected Bazel paths:
        bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/...
        bazel-out/.../xla/pjrt/c/pjrt_c_api_gpu_plugin.so.runfiles/...
        output_base (from `bazel info output_base`)
      EOF
    '';

    meta = {
      description = "XLA PJRT C API runtime plugins";
      longDescription = ''
        Bundles XLA PJRT CPU/GPU plugins plus a self-contained CUDA runtime
        tree. By default the CUDA/NVSHMEM DSOs are copied from Bazel runfiles
        (_solib) to avoid dragging the full nixpkgs CUDA runtime closure. The
        bundle layout lives under $out/runtime; see $out/runtime/README.txt for
        details on layout, switches, and expected Bazel paths.
      '';
      platforms = lib.platforms.linux;
    };

    # Strip step is brittle for Bazel outputs in devel builds; keep symbols.
    dontStrip = devel;
  }
