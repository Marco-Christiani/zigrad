//! TVM kernel provider.
//!
//! The provider accepts PR matrix-multiply functions and emits kernel artifacts
//!  values through MetaSchedule autotuning. TVM C types are internal.
const std = @import("std");
const contraction = @import("../pr/analysis/contraction.zig");
const pattern = @import("../pr/analysis/pattern.zig");

const Cache = @import("../cache.zig").Cache;
const device = @import("../device.zig");
const kernel = @import("../kernel.zig");
const pr = @import("../pr/pr.zig");
const TypedPtr = @import("../utils/rtti.zig").TypedPtr;
const config = @import("config.zig");
const mm = @import("matmul.zig");
const tvm_runtime = @import("runtime.zig");
const TvmDispatchState = @import("dispatch.zig").TvmDispatchState;

const log = std.log.scoped(.@"zg/tvm_provider");

pub const TvmProvider = struct {
    io: std.Io,
    compile_config: config.CompileConfig,
    cache: Cache,
    max_trials: u32 = 64,
    trials_per_iter: u32 = 16,

    /// Shared dispatch state owning the TVM module cache.
    ///
    /// The state must outlive all artifacts produced by this provider.
    dispatch_state: *TvmDispatchState,

    /// Inputs required to initialize a TVM provider.
    pub const InitOptions = struct {
        /// Target and compiler inputs resolved by application composition.
        compile: config.CompileConfig,

        /// Maximum measured candidates.
        max_trials: u32 = 64,

        /// Candidates submitted per tuning iteration.
        trials_per_iter: u32 = 16,
    };

    /// Initialize a provider after loading its configured TVM runtime.
    pub fn init(
        io: std.Io,
        cache: Cache,
        dispatch_state: *TvmDispatchState,
        options: InitOptions,
    ) tvm_runtime.Error!TvmProvider {
        try tvm_runtime.ensure_loaded(.compiler);
        return .{
            .io = io,
            .compile_config = options.compile,
            .cache = cache,
            .max_trials = options.max_trials,
            .trials_per_iter = options.trials_per_iter,
            .dispatch_state = dispatch_state,
        };
    }

    /// Return a KernelProvider interface backed by this TvmProvider.
    pub fn kernel_provider(self: *TvmProvider) kernel.KernelProvider {
        return .{
            .name = "tvm",
            .ptr = @ptrCast(self),
            .compile_fn = compile_impl,
            .match_fn = match_impl,
            .prepare_fn = &TvmDispatchState.prepare,
            .dispatch_fn = &TvmDispatchState.dispatch,
            .dispatch_ctx = TypedPtr.init(self.dispatch_state),
        };
    }

    fn match_impl(_: *anyopaque, func: pr.Function, start: usize) ?kernel.Match {
        if (start >= func.ops.len or validate_matmul_op(func.ops[start]) == null) return null;
        return .{ .op_count = 1 };
    }

    fn compile_impl(ptr: *anyopaque, func: pr.Function, selected_device: device.Device, allocator: std.mem.Allocator) kernel.CompileError!kernel.Artifact {
        const self: *TvmProvider = @ptrCast(@alignCast(ptr));
        return try self.compile(func, selected_device, allocator);
    }

    /// Compile one supported matrix-multiply function into a kernel artifact.
    ///
    /// A stable cached artifact is reused when present. A cache miss runs the
    ///  shared TVM matmul tuner and stores its selected artifact.
    fn compile(
        self: *TvmProvider,
        func: pr.Function,
        selected_device: device.Device,
        allocator: std.mem.Allocator,
    ) kernel.CompileError!kernel.Artifact {
        if (!self.compile_config.target.accepts(selected_device)) {
            return error.Unsupported;
        }
        const mm_shape = validate_matmul_function(func) orelse return error.Unsupported;

        log.info("compiling matmul kernel: {s} ({d}x{d}x{d})", .{
            func.name, mm_shape.m, mm_shape.n, mm_shape.k,
        });

        const cached = mm.load_artifact(
            self.io,
            allocator,
            self.cache,
            mm_shape,
            self.compile_config.target,
            selected_device,
        ) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => {
                log.err("failed to read the TVM artifact cache: {s}", .{@errorName(err)});
                return error.CompileFailed;
            },
        };
        if (cached) |artifact| return make_kernel_artifact(artifact);

        const result = mm.tune(
            self.io,
            allocator,
            self.cache,
            mm_shape,
            .{
                .compile = self.compile_config,
                .device = selected_device,
                .max_trials = self.max_trials,
                .trials_per_iter = self.trials_per_iter,
            },
        ) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.TvmLoadFailed => {
                log.err("TVM runtime unavailable: {s}", .{@errorName(err)});
                return error.ProviderLoadFailed;
            },
            error.TvmCallFailed, error.TvmFunctionNotFound, error.UnexpectedTvmType => {
                log.err("TVM API call failed during tuning: {s}", .{@errorName(err)});
                return error.ProviderCallFailed;
            },
            else => {
                log.err("tuning failed: {s}", .{@errorName(err)});
                return error.CompileFailed;
            },
        };

        const artifact = mm.load_artifact(
            self.io,
            allocator,
            self.cache,
            mm_shape,
            self.compile_config.target,
            selected_device,
        ) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => {
                log.err("failed to read cached kernel: {s}", .{@errorName(err)});
                return error.CompileFailed;
            },
        } orelse return error.CompileFailed;

        log.info("compiled kernel: {s} (candidate {d}, {d:.2} us, {d} bytes)", .{
            func.name, result.best_candidate, result.best_time_us, artifact.bytes.len,
        });

        return make_kernel_artifact(artifact);
    }
};

/// Validate that a function describes a single rank-two matrix multiplication.
///
/// Returns the matrix dimensions, or null when the function is unsupported.
fn validate_matmul_function(func: pr.Function) ?mm.Shape {
    if (func.ops.len != 1) return null;
    if (func.params.len != 2 or func.returns.len != 1) return null;
    return validate_matmul_op(func.ops[0]);
}

fn validate_matmul_op(op: *const pr.Op) ?mm.Shape {
    if (!(pattern.Operation{
        .input_count = 2,
        .output_count = 1,
        .first_output_dtype = .f32,
        .first_output_rank = 2,
    }).matches(op)) return null;

    switch (op.params) {
        .mm => {},
        .dot_general => |dg| {
            if (!contraction.is_matrix_matmul(dg)) return null;
        },
        else => return null,
    }

    const a = op.inputs[0].value.as_tensor();
    const b = op.inputs[1].value.as_tensor();
    const c_tensor = op.outputs[0].as_tensor();

    if (a.shape.rank() != 2 or b.shape.rank() != 2 or c_tensor.shape.rank() != 2) return null;

    const m = a.shape.dims[0];
    const k = a.shape.dims[1];
    const n = b.shape.dims[1];

    if (b.shape.dims[0] != k) return null;
    if (c_tensor.shape.dims[0] != m or c_tensor.shape.dims[1] != n) return null;

    if (a.dtype != .f32 or b.dtype != .f32) return null;

    return .{ .m = m, .n = n, .k = k };
}

fn make_kernel_artifact(artifact: mm.CachedArtifact) kernel.Artifact {
    return .{
        .data = artifact.bytes,
    };
}

test "TVM matcher recognizes matrix matmul" {
    const testing = std.testing;
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const lhs = try builder.param_tensor(.f32, &.{ 4, 8 });
    const rhs = try builder.param_tensor(.f32, &.{ 8, 2 });
    const output = try builder.mm(lhs, rhs);
    const func = try builder.finish(.{ .returns = &.{output} });

    const matched = TvmProvider.match_impl(undefined, func, 0) orelse
        return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 1), matched.op_count);
}
