/// Pass-Based Pipeline Infrastructure
///
/// This module defines the core abstractions for the pass-based compilation model:
/// - Artifact: Tagged union representing IR at various stages (PR, MLIR, EA)
/// - PassContext: Shared state threaded through passes
/// - Pass: Pass metadata + runnable function + optional user config
/// - Pipeline: Pass sequence with validation and execution
///
/// Key design principles:
/// - Passes declare input/output artifact kinds for validation
/// - Artifact kinds are runtime-validated at pass composition
/// - PassContext is backend-agnostic; backend-specific config lives in Pass.userdata
///
/// See: .internal/2026-01-16-03_PASS_BASED_PIPELINE.md
const std = @import("std");

const pr_mod = @import("../pr/pr.zig");
const pjrt_types = @import("../ffi/pjrt/types.zig");

/// Artifact kinds for pass input/output validation.
pub const ArtifactKind = enum {
    /// Zigrad PR program (internal, toolchain-neutral)
    pr,

    /// MLIR module bytes (StableHLO dialect, text or bytecode)
    mlir,

    /// Executable artifact (backend-specific, ready for execution)
    ea,

    /// Serialized executable bytes (for caching)
    serialized_ea,
};

/// MLIR encoding format
pub const MlirEncoding = enum {
    text,
    bytecode,
};

/// Artifact: The "IR at some point" in the pipeline.
///
/// This is a tagged union representing the various forms that a program
/// takes as it flows through compilation passes.
pub const Artifact = union(ArtifactKind) {
    /// PR program (Zigrad-owned)
    pr: *pr_mod.Program,

    /// MLIR module (serialized bytes)
    mlir: MlirArtifact,

    /// Executable artifact (backend-bound handle)
    ea: ExecutableArtifact,

    /// Serialized executable (for cache/AOT)
    serialized_ea: []u8,

    pub fn kind(self: Artifact) ArtifactKind {
        return @as(ArtifactKind, self);
    }

    /// Free owned resources. Not all variants own memory.
    pub fn deinit(self: *Artifact, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .pr => {}, // PR program is borrowed, not owned here
            .mlir => |*m| m.deinit(allocator),
            .ea => |*e| e.deinit(),
            .serialized_ea => |bytes| allocator.free(bytes),
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

/// Executable artifact (backend-specific handle)
pub const ExecutableArtifact = union(enum) {
    /// PJRT loaded executable (JIT)
    pjrt: pjrt_types.LoadedExecutable,

    pub fn deinit(self: *ExecutableArtifact) void {
        switch (self.*) {
            .pjrt => |*exe| exe.deinit(),
        }
    }
};

/// Pass context: shared state threaded through pass execution.
///
/// Passes receive this context for access to allocators, backend sessions,
/// and other shared resources without needing to thread them explicitly.
pub const PassContext = struct {
    allocator: std.mem.Allocator,
};

/// Pass execution errors
pub const PassError = error{
    /// Input artifact kind does not match pass expectation
    ArtifactKindMismatch,

    /// Pass-specific validation failed
    ValidationFailed,

    /// Lowering failed (PR -> MLIR)
    LoweringFailed,

    /// Compilation failed (MLIR -> EA)
    CompilationFailed,

    /// Missing required context (e.g., no backend session)
    MissingContext,

    /// Backend-specific error
    BackendError,

    /// Out of memory
    OutOfMemory,
};

/// Pass descriptor: metadata + function + optional user config.
pub const Pass = struct {
    name: []const u8,
    input_kind: ArtifactKind,
    output_kind: ArtifactKind,
    run: *const fn (*Artifact, *PassContext, ?*anyopaque) PassError!void,
    userdata: ?*anyopaque = null,
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
            try p.run(&current, ctx, p.userdata);
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
        fn run(a: *Artifact, ctx: *PassContext, _: ?*anyopaque) PassError!void {
            _ = a;
            _ = ctx;
        }
    }.run;

    const passes = [_]Pass{
        .{ .name = "validate", .input_kind = .pr, .output_kind = .pr, .run = noop },
        .{ .name = "lower", .input_kind = .pr, .output_kind = .mlir, .run = noop },
        .{ .name = "compile", .input_kind = .mlir, .output_kind = .ea, .run = noop },
    };

    const pipeline = Pipeline{ .passes = &passes };
    try pipeline.validate();
}

test "pass chain validation rejects mismatch" {
    const noop = struct {
        fn run(a: *Artifact, ctx: *PassContext, _: ?*anyopaque) PassError!void {
            _ = a;
            _ = ctx;
        }
    }.run;

    const passes = [_]Pass{
        .{ .name = "validate", .input_kind = .pr, .output_kind = .pr, .run = noop },
        .{ .name = "compile", .input_kind = .mlir, .output_kind = .ea, .run = noop },
    };

    const pipeline = Pipeline{ .passes = &passes };
    try std.testing.expectError(error.ArtifactKindMismatch, pipeline.validate());
}
