//! PR Validate Pass
const pr = @import("../pr/pr.zig");
const pass = @import("pass.zig");

/// Validate pass: PR artifact -> PR artifact.
fn validate_pass_run(_: *anyopaque, artifact: *pass.Artifact, _: *pass.PassContext) pass.PassError!void {
    if (artifact.kind() != .pr) return error.ArtifactKindMismatch;
    pr.validate_program(artifact.pr) catch return error.ValidationFailed;
}

/// Metadata for the validate pass.
pub const validate_pass = pass.Pass{
    .ptr = undefined,
    .run_fn = validate_pass_run,
    .name = "pr_validate",
    .input_kind = .pr,
    .output_kind = .pr,
};
