//! Entrypoint for pass pipeline infrastructure.
//!
//! The pipeline is a chain of passes that transform artifacts from PR through
//!  to IM (MLIR). Compilation from IM to executable artifacts is a backend
//!  responsibility, the pipeline conventionally ends at an IM ready to be
//!  compiled by a backend toolchain.
//!
//! MLIR-specific passes (select, legalize) live in `lower/mlir/`.
//!
//! See `pass` for more.
const pass = @import("pipeline/pass.zig");

// Re-export pass types
pub const Artifact = pass.Artifact;
pub const ArtifactKind = pass.ArtifactKind;
pub const MlirArtifact = pass.MlirArtifact;
pub const MlirEncoding = pass.MlirEncoding;
pub const PassContext = pass.PassContext;
pub const Pass = pass.Pass;
pub const PassError = pass.PassError;
pub const Pipeline = pass.Pipeline;

const dump = @import("pipeline/dump.zig");
pub const DumpConfig = dump.DumpConfig;
pub const DumpTarget = dump.DumpTarget;
pub const DumpSpec = dump.DumpSpec;
pub const dump_pr_pass_with_config = dump.dump_pr_pass_with_config;
pub const dump_mlir_pass_with_config = dump.dump_mlir_pass_with_config;
pub const dump_optimized_program = dump.dump_optimized_program;

pub const kernelize = @import("pipeline/kernelize.zig");
pub const KernelizePass = kernelize.KernelizePass;

const validate_mod = @import("pipeline/validate.zig");
pub const validate_pass = validate_mod.validate_pass;

test {
    @import("std").testing.refAllDecls(@This());
}
