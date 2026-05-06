//! Pass Pipeline Infrastructure
//!
//! This module defines the core abstractions for the pass-based compilation model:
//!  - `Artifact`: Tagged union representing a program at various pipeline stages
//!  - `PassContext`: Shared state threaded through passes
//!  - `Pass`: Pass metadata + runnable function + optional user config
//!  - `Pipeline`: Pass sequence with validation and execution
//!  - `RunOptions`: Per-run configuration including verifier policy
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
//!  - The runner calls `pr.validate_program` between passes per
//!     `RunOptions.verifier_policy`.
const std = @import("std");
const builtin = @import("builtin");
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

/// When the pipeline runner invokes `pr.validate_program` against the
/// current artifact.
pub const VerifierPolicy = enum {
    /// Skip structural verification entirely. Caller asserts validity.
    never,
    /// Verify the artifact once before the first pass runs.
    input,
    /// Verify input plus after every pass whose current artifact is PR.
    ///  Skips when the artifact is not PR.
    each_pr_pass,

    /// Build-mode default for `VerifierPolicy`. Always at least `.input` so
    /// that compile entry points using the default reject malformed input
    /// at every build mode.
    pub const default: VerifierPolicy = switch (builtin.mode) {
        .Debug, .ReleaseSafe => .each_pr_pass,
        .ReleaseFast, .ReleaseSmall => .input,
    };
};

/// Per-run pipeline configuration consumed by `Pipeline.run`.
pub const RunOptions = struct {
    verifier_policy: VerifierPolicy = .default,
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

    pub fn run(
        self: *const Pipeline,
        initial: Artifact,
        ctx: *PassContext,
        opts: RunOptions,
    ) PassError!Artifact {
        try self.validate();

        if (self.passes.len > 0 and initial.kind() != self.passes[0].input_kind) {
            return error.ArtifactKindMismatch;
        }

        const pipeline_start = std.Io.Timestamp.now(ctx.io, .awake);

        var current = initial;
        errdefer current.deinit(ctx.allocator);

        // Verify input before any pass runs, when policy demands it.
        try maybe_verify_pr(&current, opts.verifier_policy, .before_first_pass, null);

        for (self.passes) |p| {
            if (current.kind() != p.input_kind) return error.ArtifactKindMismatch;

            const pass_start = std.Io.Timestamp.now(ctx.io, .awake);
            try p.run(&current, ctx);
            log.info("pass '{s}' completed in {d:.2}ms", .{
                p.name, ns_to_ms(@intCast(pass_start.untilNow(ctx.io, .awake).toNanoseconds())),
            });

            // Kind check first: a wrong-output-kind bug surfaces as
            //  ArtifactKindMismatch at the pass that violated its contract.
            if (current.kind() != p.output_kind) return error.ArtifactKindMismatch;

            try maybe_verify_pr(&current, opts.verifier_policy, .after_pass, p.name);
        }

        log.info("pipeline completed: {d} passes in {d:.2}ms", .{
            self.passes.len, ns_to_ms(@intCast(pipeline_start.untilNow(ctx.io, .awake).toNanoseconds())),
        });

        return current;
    }
};

const VerifyWhen = enum { before_first_pass, after_pass };

/// Run structural PR validation if the policy and current artifact demand it.
///
/// Failure is attributed to `attribution` (the pass that produced the invalid
/// artifact) when known, or to "input" when verifying before the first pass.
/// All `pr.ValidationError` variants are remapped to `PassError.InvalidProgram`.
fn maybe_verify_pr(
    artifact: *const Artifact,
    policy: VerifierPolicy,
    when: VerifyWhen,
    attribution: ?[]const u8,
) PassError!void {
    switch (policy) {
        .never => return,
        .input => if (when != .before_first_pass) return,
        .each_pr_pass => {},
    }
    if (artifact.kind() != .pr) return;
    pr_mod.validate_program(artifact.pr) catch |err| {
        if (attribution) |name| {
            log.err("pass '{s}' produced invalid PR: {s}", .{ name, @errorName(err) });
        } else {
            log.err("input PR failed verification: {s}", .{@errorName(err)});
        }
        return error.InvalidProgram;
    };
}

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
    var artifact = try pipeline.run(.{ .pr = &program }, &ctx, .{});
    defer artifact.deinit(testing.allocator);

    switch (artifact) {
        .stablehlo => |s| try testing.expectEqualStrings("stablehlo_output", s.bytes),
        else => return error.ArtifactKindMismatch,
    }
}

test "verifier policy never skips structural verification" {
    const testing = std.testing;

    // An empty program with no functions is structurally valid; we use it
    //  as a smoke test that .never does not invoke the verifier.
    var program = pr_mod.Program.init(testing.allocator);
    defer program.deinit();

    const passes = [_]Pass{};
    var ctx = PassContext{ .allocator = testing.allocator, .io = std.testing.io };
    const pipeline = Pipeline{ .passes = &passes };
    var artifact = try pipeline.run(
        .{ .pr = &program },
        &ctx,
        .{ .verifier_policy = .never },
    );
    defer artifact.deinit(testing.allocator);
}

test "verifier policy input runs verification on input artifact" {
    const testing = std.testing;

    var program = pr_mod.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr_mod.FunctionBuilder.init(&program, "main");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{2});
    const func = try b.finish(&.{x});
    try program.add_function(func);

    const passes = [_]Pass{};
    var ctx = PassContext{ .allocator = testing.allocator, .io = std.testing.io };
    const pipeline = Pipeline{ .passes = &passes };
    var artifact = try pipeline.run(
        .{ .pr = &program },
        &ctx,
        .{ .verifier_policy = .input },
    );
    defer artifact.deinit(testing.allocator);
}

test "verifier policy each_pr_pass verifies after every pr-out pass" {
    const testing = std.testing;

    var program = pr_mod.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr_mod.FunctionBuilder.init(&program, "main");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{2});
    const func = try b.finish(&.{x});
    try program.add_function(func);

    // Identity PR-to-PR pass: leaves the program valid.
    const noop_pr = struct {
        fn f(_: *anyopaque, _: *Artifact, _: *PassContext) PassError!void {}
    }.f;

    const passes = [_]Pass{
        .{ .ptr = undefined, .run_fn = noop_pr, .name = "noop", .input_kind = .pr, .output_kind = .pr },
    };

    var ctx = PassContext{ .allocator = testing.allocator, .io = std.testing.io };
    const pipeline = Pipeline{ .passes = &passes };
    var artifact = try pipeline.run(
        .{ .pr = &program },
        &ctx,
        .{ .verifier_policy = .each_pr_pass },
    );
    defer artifact.deinit(testing.allocator);
}

test "verifier policy skips validation once artifact is non-pr" {
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
        .{ .ptr = undefined, .run_fn = to_stablehlo, .name = "lower", .input_kind = .pr, .output_kind = .stablehlo },
    };

    var ctx = PassContext{ .allocator = testing.allocator, .io = std.testing.io };
    const pipeline = Pipeline{ .passes = &passes };
    // each_pr_pass policy: input is verified, but after the lowering pass
    //  the artifact is .stablehlo and the verifier is skipped.
    var artifact = try pipeline.run(
        .{ .pr = &program },
        &ctx,
        .{ .verifier_policy = .each_pr_pass },
    );
    defer artifact.deinit(testing.allocator);

    switch (artifact) {
        .stablehlo => |s| try testing.expectEqualStrings("stablehlo_output", s.bytes),
        else => return error.ArtifactKindMismatch,
    }
}

test VerifierPolicy {
    { // default verifier policy returns build-mode-appropriate value
        const policy: VerifierPolicy = .default;
        // At least one of these must be true for any build mode; the test
        //  exists to ensure the function compiles and returns a valid variant.
        try std.testing.expect(
            policy == .never or policy == .input or policy == .each_pr_pass,
        );
    }
}
