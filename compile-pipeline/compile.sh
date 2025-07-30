#!/usr/bin/env zsh
# set -euo pipefail

ZIG_SOURCE="main.zig"
RAW_LL="main.ll"

typeset -A OPT_PASSES
OPT_PASSES["baseline"]="default<O3>"
OPT_PASSES["noopt"]=""
OPT_PASSES["sroa_only"]="sroa"
OPT_PASSES["inline_only"]="inline"
OPT_PASSES["unroll_only"]="loop-unroll"
OPT_PASSES["minimal"]="sroa,mem2reg,instcombine"
# OPT_PASSES["loop_focus"]="loop-simplify,loop-mssa,licm,loop-unroll"
OPT_PASSES["mem_focused"]="mem2reg,sroa,instcombine,dce"
OPT_PASSES["inline_unroll"]="inline,loop-unroll"
# OPT_PASSES["conservative"]="mem2reg,sroa,function(inline),instcombine"

OPT_PASSES["custom_aggressive"]="function-attrs,infer-address-spaces,inline,mem2reg,instcombine,early-cse,loop-simplify,loop-unroll,loop-vectorize,slp-vectorizer,simplifycfg,dce,sroa,instcombine,tailcallelim"

OPT_PASSES["o1_equiv"]="default<O1>"
OPT_PASSES["o2_equiv"]="default<O2>"
OPT_PASSES["o3_equiv"]="default<O3>"
OPT_PASSES["oz_equiv"]="default<Oz>" # size

# Categories
# OPT_PASSES["just_inline"]="inline"
OPT_PASSES["just_unroll"]="loop-unroll"
OPT_PASSES["just_mem"]="mem2reg,sroa"
OPT_PASSES["just_combine"]="instcombine"

# TODO: PGO
# OPT_PASSES["pgo"]="pgo-instr-gen"
# OPT_PASSES["pgo_use"]="pgo-instr-use"

# build llvm ir
echo "[1] Emit raw LLVM IR from Zig"
zig build-exe "$ZIG_SOURCE" \
  -OReleaseFast \
  -femit-llvm-ir \
  -fno-emit-bin \
  -fno-emit-h \
  --name "${RAW_LL/\.ll/}"

executables=()
# for key in "${!OPT_PASSES[@]}"; do
for key in ${(@k)OPT_PASSES}; do
  echo "[2] Compile variant: $key"

  OPT_LL="main_opt_$key.ll"
  OBJ="main_opt_$key.o"
  EXE="main_$key"
  PASSES="${OPT_PASSES[$key]}"

  # Optimize
  if [[ -n "$PASSES" ]]; then
    opt -S -passes="$PASSES" "$RAW_LL" -o "$OPT_LL"
  else
    cp "$RAW_LL" "$OPT_LL"
  fi

  # lower to o
  llc -filetype=obj "$OPT_LL" -o "$OBJ"

  # Link with zig runtime
  zig build-exe "$OBJ" --name "$EXE"

  executables+=("./$EXE")
done

# Build Zig bin the normal way
EXE_BUILTIN="main_builtin"
echo "[3] Build Zig built-in ReleaseFast binary"
zig build-exe "$ZIG_SOURCE" -OReleaseFast --name "$EXE_BUILTIN"
executables+=("./$EXE_BUILTIN")

echo -e "\n[4] Binaries:"
ls -lh "${executables[@]}"

# Benchmark
echo -e "\n[5] Benchmark runtimes:"
hyperfine "${executables[@]}"
