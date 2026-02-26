/// Pass-Based Pipeline Infrastructure
///
/// This module defines the core abstractions for the pass-based compilation model:
/// - Artifact: Tagged union representing IR at various stages (PR, MLIR)
/// - PassContext: Shared state threaded through passes
/// - Pass: Pass metadata + runnable function + optional user config
/// - Pipeline: Pass sequence with validation and execution
///
/// The pipeline operates on PR and MLIR only. Compilation (MLIR -> EA) is a
/// backend responsibility, called separately after the pipeline completes.
///
/// Key design principles:
/// - Passes declare input/output artifact kinds for validation
/// - Artifact kinds are runtime-validated at pass composition
/// - PassContext is backend-agnostic; pass-specific state lives behind Pass.ptr
///
/// See KB: "Pass-Based Pipeline Direction (Design Update)"
const std = @import("std");

const pr_mod = @import("../pr/pr.zig");

/// Artifact kinds for pass input/output validation.
pub const ArtifactKind = enum {
    /// Zigrad PR program (internal, toolchain-neutral)
    pr,

    /// MLIR module bytes (StableHLO dialect, text or bytecode)
    mlir,
};

/// MLIR encoding format
pub const MlirEncoding = enum {
    text,
    bytecode,
};

/// Artifact: The "IR at some point" in the pipeline.
///
/// This is a tagged union representing the various forms that a program
/// takes as it flows through pipeline passes. The pipeline operates on
/// PR and MLIR only — compilation to executable artifacts is handled
/// by the backend after the pipeline completes.
pub const Artifact = union(ArtifactKind) {
    /// PR program (Zigrad-owned)
    pr: *pr_mod.Program,

    /// MLIR module (serialized bytes)
    mlir: MlirArtifact,

    pub fn kind(self: Artifact) ArtifactKind {
        return @as(ArtifactKind, self);
    }

    /// Free owned resources. Not all variants own memory.
    pub fn deinit(self: *Artifact, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .pr => {}, // PR program is borrowed, not owned here
            .mlir => |*m| m.deinit(allocator),
        }
    }

    /// Replace this artifact with a new one, deinitializing the old value.
    pub fn replace(self: *Artifact, allocator: std.mem.Allocator, next: Artifact) void {
        self.deinit(allocator);
        self.* = next;
    }
};

/// MLIR artifact with encoding metadata
pub const MlirArtifact = struct {
    bytes: []u8,
    encoding: MlirEncoding,

    pub fn deinit(self: *MlirArtifact, allocator: std.mem.Allocator) void {
        allocator.free(self.bytes);
    }
};

/// Pass context: shared state threaded through pass execution.
///
/// Passes receive this context for access to allocators and other shared
/// resources without needing to thread them explicitly.
pub const PassContext = struct {
    allocator: std.mem.Allocator,
};

/// Pass execution errors
pub const PassError = error{
    /// Input artifact kind does not match pass expectation
    ArtifactKindMismatch,

    /// Pass-specific validation failed
    ValidationFailed,

    /// PR program failed structural/typing validation.
    InvalidProgram,

    /// Lowering failed (PR -> MLIR)
    LoweringFailed,

    /// Lowered MLIR failed verification.
    InvalidMlir,

    /// TVM runtime/compiler library loading failed.
    TvmLoadFailed,

    /// TVM function invocation failed.
    TvmCallFailed,

    /// Required TVM global function was not found.
    TvmFunctionNotFound,

    /// TVM value type did not match the expected representation.
    UnexpectedTvmType,

    /// Mirage runtime/API loading failed.
    MirageLoadFailed,

    /// Mirage API returned invalid arguments (contract mismatch).
    MirageInvalidArgument,

    /// Mirage API reported an internal runtime failure.
    MirageInternalError,

    /// Mirage runtime reported unsupported outside region-capability matching.
    MirageApiUnsupported,

    /// Mirage compile invocation failed.
    MirageCompileFailed,

    /// Mirage execution contract was invalid.
    MirageContractError,

    /// Provider compilation failed for an internal reason.
    CompileFailed,

    /// Provider reported a region as unsupported.
    Unsupported,

    /// Kernelize custom-call rewrite encountered an invalid region shape.
    InvalidRegion,

    /// Duplicate kernel key while registering provider artifact.
    DuplicateKey,

    /// Missing required context (e.g., no backend session)
    MissingContext,

    /// Out of memory
    OutOfMemory,
};

/// Pass: the unit of transformation in the pipeline.
///
/// Follows the Zig interface pattern (ptr + run_fn). Stateful passes
/// store their configuration behind `ptr`; stateless passes leave it
/// undefined.
pub const Pass = struct {
    ptr: *anyopaque,
    run_fn: *const fn (ptr: *anyopaque, artifact: *Artifact, ctx: *PassContext) PassError!void,
    name: []const u8,
    input_kind: ArtifactKind,
    output_kind: ArtifactKind,

    pub fn run(self: Pass, artifact: *Artifact, ctx: *PassContext) PassError!void {
        return self.run_fn(self.ptr, artifact, ctx);
    }
};

/// Pipeline: a sequence of passes with validation and execution.
pub const Pipeline = struct {
    passes: []const Pass,

    pub fn validate(self: *const Pipeline) PassError!void {
        if (self.passes.len < 2) return;
        for (0..self.passes.len - 1) |i| {
            if (self.passes[i].output_kind != self.passes[i + 1].input_kind) {
                return error.ArtifactKindMismatch;
            }
        }
    }

    pub fn run(self: *const Pipeline, initial: Artifact, ctx: *PassContext) PassError!Artifact {
        try self.validate();

        if (self.passes.len > 0 and initial.kind() != self.passes[0].input_kind) {
            return error.ArtifactKindMismatch;
        }

        var current = initial;
        errdefer current.deinit(ctx.allocator);
        for (self.passes) |p| {
            if (current.kind() != p.input_kind) return error.ArtifactKindMismatch;
            try p.run(&current, ctx);
            if (current.kind() != p.output_kind) return error.ArtifactKindMismatch;
        }

        return current;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "artifact kind tagging" {
    const testing = std.testing;

    var program = pr_mod.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr_mod.FunctionBuilder.init(&program, "test");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const func = try b.finish(&.{x});
    try program.add_function(func);

    const artifact = Artifact{ .pr = &program };
    try testing.expectEqual(ArtifactKind.pr, artifact.kind());
}

test "pass chain validation" {
    const noop = struct {
        fn f(_: *anyopaque, _: *Artifact, _: *PassContext) PassError!void {}
    }.f;

    const passes = [_]Pass{
        .{ .ptr = undefined, .run_fn = noop, .name = "validate", .input_kind = .pr, .output_kind = .pr },
        .{ .ptr = undefined, .run_fn = noop, .name = "lower", .input_kind = .pr, .output_kind = .mlir },
    };

    const pipeline = Pipeline{ .passes = &passes };
    try pipeline.validate();
}

test "pass chain validation rejects mismatch" {
    const noop = struct {
        fn f(_: *anyopaque, _: *Artifact, _: *PassContext) PassError!void {}
    }.f;

    const passes = [_]Pass{
        .{ .ptr = undefined, .run_fn = noop, .name = "validate", .input_kind = .pr, .output_kind = .pr },
        .{ .ptr = undefined, .run_fn = noop, .name = "lower", .input_kind = .mlir, .output_kind = .mlir },
    };

    const pipeline = Pipeline{ .passes = &passes };
    try std.testing.expectError(error.ArtifactKindMismatch, pipeline.validate());
}

test "pipeline run transforms artifacts" {
    const testing = std.testing;

    var program = pr_mod.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr_mod.FunctionBuilder.init(&program, "main");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{2});
    const func = try b.finish(&.{x});
    try program.add_function(func);

    const to_mlir = struct {
        fn f(_: *anyopaque, a: *Artifact, ctx: *PassContext) PassError!void {
            const bytes = try ctx.allocator.dupe(u8, "mlir_output");
            a.replace(ctx.allocator, .{ .mlir = .{ .bytes = bytes, .encoding = .text } });
        }
    }.f;

    const passes = [_]Pass{
        .{ .ptr = undefined, .run_fn = to_mlir, .name = "to_mlir", .input_kind = .pr, .output_kind = .mlir },
    };

    var ctx = PassContext{ .allocator = testing.allocator };
    const pipeline = Pipeline{ .passes = &passes };
    var artifact = try pipeline.run(.{ .pr = &program }, &ctx);
    defer artifact.deinit(testing.allocator);

    switch (artifact) {
        .mlir => |m| try testing.expectEqualStrings("mlir_output", m.bytes),
        else => return error.ArtifactKindMismatch,
    }
}
