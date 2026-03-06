/// Pipeline Module
///
/// Pass-based pipeline infrastructure for compilation.
/// The pipeline operates on PR and MLIR only — compilation to executable
/// artifacts is a backend responsibility.
///
/// Key types:
/// - Artifact: Tagged union representing IR at various stages (PR, MLIR)
/// - PassContext: Shared state threaded through passes
/// - Pass: Pass descriptor (name + input/output kinds + run)
pub const pass = @import("pass.zig");

// Re-export pass types
pub const Artifact = pass.Artifact;
pub const ArtifactKind = pass.ArtifactKind;
pub const MlirArtifact = pass.MlirArtifact;
pub const MlirEncoding = pass.MlirEncoding;
pub const PassContext = pass.PassContext;
pub const Pass = pass.Pass;
pub const PassError = pass.PassError;
pub const Pipeline = pass.Pipeline;

pub const dump = @import("dump.zig");
pub const DumpConfig = dump.DumpConfig;
pub const DumpTarget = dump.DumpTarget;
pub const DumpSpec = dump.DumpSpec;
pub const dump_pr_pass_with_config = dump.dump_pr_pass_with_config;
pub const dump_mlir_pass_with_config = dump.dump_mlir_pass_with_config;

pub const kernelize = @import("kernelize.zig");
pub const KernelizePass = kernelize.KernelizePass;

pub const mlir_materialize = @import("mlir_materialize.zig");
pub const MlirKernelMaterializePass = mlir_materialize.MlirKernelMaterializePass;

pub const mlir_passes = @import("mlir_passes.zig");
pub const MlirSelectPass = mlir_passes.MlirSelectPass;
pub const MlirLegalizePass = mlir_passes.MlirLegalizePass;

test {
    @import("std").testing.refAllDecls(@This());
}
