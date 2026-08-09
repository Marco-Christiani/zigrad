# Per-user overrides for flake.nix's buildCfg. Copy this file to
# local-build-cfg.nix (gitignored) and adjust for your machine.
#
# Each key listed here overrides the project default in flake.nix.
# Omit a key to inherit the project default. The defaults are
# deliberately portable. Your local file opts into machine-specific
# optimizations.
#
# The repository direnv environment loads this file. Direct Nix commands use
# the portable defaults.
#
# Full knob reference: docs site, "Building / Optimization knobs".
{
  # === CUDA architecture pin ==========================================
  #
  # Compile CUDA only for these GPU compute capabilities. This avoids
  # compiling each supported architecture on a single-GPU workstation.
  #
  # Find your GPU's compute capability:
  #   nvidia-smi --query-gpu=compute_cap --format=csv,noheader
  # Then map "8.6" to ["86"], "8.9" to ["89"], "9.0" to ["90"], etc.
  #
  # Examples:
  #   ["86"]              # RTX 30-series, A10
  #   ["89"]              # RTX 40-series (Ada Lovelace)
  #   ["90"]              # H100 (Hopper)
  #   ["80"]              # A100 (Ampere data-center)
  #   ["80" "86" "89"]    # Ampere + Ada (broader local pool)
  #
  # cudaArchitectures = ["86"];

  # === Native CPU tuning ==============================================
  #
  # Emit -march=native -mtune=native and AVX2/FMA for hot-path libs
  # (TVM, IREE runtime, xla-pjrt CPU plugin). Compiler-infra libs (LLVM,
  # XLA-MLIR, IREE-LLVM) skip this regardless. They're loaded by every
  # consumer and non-portable codegen there would be a problem.
  #
  # Workstation-class build host: turn on for max perf.
  # CI / cross-machine builds: leave off.
  #
  # withNativeTuning = true;

  # === Debug build ====================================================
  #
  # Retain DWARF, build with RelWithDebInfo, and do not strip. Useful for
  # GDB-stepping into integration libraries such as libtvm.so and
  #  libIREECompiler.so.
  #
  # withDebugSymbols = true;

  # === Link-time optimization (opt-in) ================================
  #
  # Do not enable without a benchmark workload to compare.
  #
  # enableLto = true;

  # === Escape hatches =================================================
  #
  # Free-form flag injection for experimentation. Each list is appended
  # to the corresponding cmake/bazel flag set across all derivations.
  #
  # extraCxxFlags = [ "-funroll-loops" ];
  # extraLdFlags = [ "-Wl,--gc-sections" ];
  # extraBazelFlags = [ "--copt=-funroll-loops" ];
}
