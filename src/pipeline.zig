//! Entrypoint for pass pipeline infrastructure.
//!
//! The pipeline is a chain of passes that transform a program from PR
//!  through one or more lowered IRs. Compilation to an executable is a
//!  backend responsibility; the pipeline conventionally ends at an IR
//!  ready to be compiled by a backend toolchain.
//!
//! Dialect-specific passes live alongside their lowering (e.g. `lower/mlir/`
//!  for MLIR-side passes that operate on `.stablehlo` artifacts).
//!
//! See `pass` for more.
const pass = @import("pipeline/pass.zig");

// Re-export pass types
pub const Artifact = pass.Artifact;
pub const ArtifactKind = pass.ArtifactKind;
pub const Encoding = pass.Encoding;
pub const PassContext = pass.PassContext;
pub const Pass = pass.Pass;
pub const PassError = pass.PassError;
pub const Pipeline = pass.Pipeline;
pub const RunOptions = pass.RunOptions;
pub const VerifierPolicy = pass.VerifierPolicy;

const dump = @import("pipeline/dump.zig");
pub const DumpConfig = dump.DumpConfig;
pub const DumpTarget = dump.DumpTarget;
pub const DumpSpec = dump.DumpSpec;
pub const dump_pr_pass_with_config = dump.dump_pr_pass_with_config;
pub const dump_mlir_pass_with_config = dump.dump_mlir_pass_with_config;
pub const dump_optimized_program = dump.dump_optimized_program;

pub const kernelize = @import("pipeline/kernelize.zig");
pub const KernelizePass = kernelize.KernelizePass;

test {
    @import("std").testing.refAllDecls(@This());
}
