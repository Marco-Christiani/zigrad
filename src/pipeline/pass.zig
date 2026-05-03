//! Pass Pipeline Infrastructure
//!
//! This module defines the core abstractions for the pass-based compilation model:
//!  - `Artifact`: Tagged union representing a program at various pipeline stages
//!  - `PassContext`: Shared state threaded through passes
//!  - `Pass`: Pass metadata + runnable function + optional user config
//!  - `Pipeline`: Pass sequence with validation and execution
//!
//! ## ADR
//!  - The pipeline carries a program from PR through one or more lowered IRs.
//!     Compilation to an executable artifact is a backend responsibility,
//!     called separately after the pipeline completes.
//!  - Passes declare input/output artifact kinds for validation.
//!  - Artifact kinds are runtime-validated at pass composition.
//!  - PassContext is backend-agnostic. Pass-specific state lives behind Pass.ptr.
//!  - `ArtifactKind` is non-exhaustive so consumer code can extend it with
//!     additional dialect tags without modifying core. Built-in tags name
//!     specific dialects (e.g. `stablehlo`), not framework families.
const std = @import("std");
const log = std.log.scoped(.@"zg/pipeline");

const pr_mod = @import("../pr/pr.zig");

/// Artifact kinds for pass input/output validation.
///
/// Non-exhaustive so additional dialect tags can be introduced without
/// modifying core. Tags name a specific dialect, not a framework.
pub const ArtifactKind = enum(u32) {
    /// Zigrad PR (toolchain agnostic).
    pr,

    /// Serialized StableHLO. Encoding (text vs binary) is on the payload.
    stablehlo,

    _,
};

/// Wire-format encoding for serialized IR bytes. Crosses the pipeline ->
///  backend boundary; backends compile against bytes plus this tag.
pub const Encoding = enum { text, binary };

/// Artifact: the program at a pipeline stage.
///
/// Tagged union over `ArtifactKind`. The pipeline carries one of these
/// between passes; compilation (IR -> executable) is a backend concern
/// invoked after the pipeline finishes.
pub const Artifact = union(ArtifactKind) {
    /// PR program (Zigrad-owned, borrowed).
    pr: *pr_mod.Program,

    /// Serialized StableHLO bytes plus encoding metadata. Owns `bytes`.
    stablehlo: struct {
        bytes: []u8,
        encoding: Encoding,
    },

    pub fn kind(self: Artifact) ArtifactKind {
        return std.meta.activeTag(self);
    }

    /// Free owned resources. Not all variants own memory.
    pub fn deinit(self: *Artifact, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .pr => {}, // PR program is borrowed
            .stablehlo => |s| allocator.free(s.bytes),
        }
    }

    /// Replace this artifact with a new one, deinitializing the old value.
    pub fn replace(self: *Artifact, allocator: std.mem.Allocator, next: Artifact) void {
        self.deinit(allocator);
        self.* = next;
    }
};

/// Pass context: shared state threaded through pass execution.
///
/// Passes receive this context for access to allocators and other shared
/// resources without needing to thread them explicitly.
pub const PassContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
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

    /// Provider runtime not available (library loading failed).
    ProviderLoadFailed,

    /// Provider API call failed or returned unexpected data.
    ProviderCallFailed,

    /// Provider compilation failed for an internal reason.
    CompileFailed,

    /// Provider reported a region as unsupported.
    Unsupported,

    /// Kernelize custom-call rewrite encountered an invalid region shape.
    InvalidRegion,

    /// Duplicate kernel key while registering provider artifact.
    DuplicateKey,

    /// Failure inside a `std.Io.Writer` (e.g. the Allocating variant).
    WriteFailed,

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
///
/// TODO: adding support for lifecycle hooks makes sense here, some passes need
///  setup with widened error unions (eg initializing an mlir session).
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
/// TODO: See above comments in `Pass`.
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

        const pipeline_start = std.Io.Timestamp.now(ctx.io, .awake);

        var current = initial;
        errdefer current.deinit(ctx.allocator);
        for (self.passes) |p| {
            if (current.kind() != p.input_kind) return error.ArtifactKindMismatch;

            const pass_start = std.Io.Timestamp.now(ctx.io, .awake);
            try p.run(&current, ctx);
            log.info("pass '{s}' completed in {d:.2}ms", .{
                p.name, ns_to_ms(@intCast(pass_start.untilNow(ctx.io, .awake).toNanoseconds())),
            });

            if (current.kind() != p.output_kind) return error.ArtifactKindMismatch;
        }

        log.info("pipeline completed: {d} passes in {d:.2}ms", .{
            self.passes.len, ns_to_ms(@intCast(pipeline_start.untilNow(ctx.io, .awake).toNanoseconds())),
        });

        return current;
    }
};

fn ns_to_ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}

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
        .{ .ptr = undefined, .run_fn = noop, .name = "lower", .input_kind = .pr, .output_kind = .stablehlo },
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
        .{ .ptr = undefined, .run_fn = noop, .name = "lower", .input_kind = .stablehlo, .output_kind = .stablehlo },
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

    const to_stablehlo = struct {
        fn f(_: *anyopaque, a: *Artifact, ctx: *PassContext) PassError!void {
            const bytes = try ctx.allocator.dupe(u8, "stablehlo_output");
            a.replace(ctx.allocator, .{ .stablehlo = .{ .bytes = bytes, .encoding = .text } });
        }
    }.f;

    const passes = [_]Pass{
        .{ .ptr = undefined, .run_fn = to_stablehlo, .name = "to_stablehlo", .input_kind = .pr, .output_kind = .stablehlo },
    };

    var ctx = PassContext{ .allocator = testing.allocator, .io = std.testing.io };
    const pipeline = Pipeline{ .passes = &passes };
    var artifact = try pipeline.run(.{ .pr = &program }, &ctx);
    defer artifact.deinit(testing.allocator);

    switch (artifact) {
        .stablehlo => |s| try testing.expectEqualStrings("stablehlo_output", s.bytes),
        else => return error.ArtifactKindMismatch,
    }
}
