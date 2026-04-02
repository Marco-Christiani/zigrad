//! TVM kernel provider.
//!
//! Implements the KernelProvider interface (src/kernel.zig) for TVM.
//! Handles matmul (dot/dot_general) kernels via MetaSchedule autotuning.
//! No TVM C types cross this boundary -- only PR types and KernelArtifact.
const std = @import("std");
const tir = @import("../c/tvm/tir.zig");
const tvm_api = @import("../c/tvm/api.zig");
const tune_mod = @import("tune.zig");
const tuned_module = @import("module.zig");
const kernel = @import("../kernel.zig");
const dispatch_mod = @import("dispatch.zig");
const pr = @import("../pr/pr.zig");
const TargetKind = tir.TargetKind;

const log = std.log.scoped(.@"zg/tvm_provider");

pub const TvmProvider = struct {
    allocator: std.mem.Allocator,
    target_kind: TargetKind,
    work_dir: []const u8,
    max_trials: u32 = 64,
    trials_per_iter: u32 = 16,

    /// Shared dispatch state owning the TVM module cache.
    /// Must outlive all KernelArtifacts produced by this provider.
    dispatch_state: *dispatch_mod.TvmDispatchState,

    /// Return a KernelProvider interface backed by this TvmProvider.
    pub fn kernel_provider(self: *TvmProvider) kernel.KernelProvider {
        return .{
            .name = "tvm",
            .ptr = @ptrCast(self),
            .compile_fn = compile_impl,
            .dispatch_fn = &dispatch_mod.TvmDispatchState.dispatch,
            .dispatch_ctx = @ptrCast(self.dispatch_state),
        };
    }

    fn compile_impl(ptr: *anyopaque, desc: kernel.RegionDescriptor, _: kernel.CompileContext, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
        const self: *TvmProvider = @ptrCast(@alignCast(ptr));
        return self.compile(desc, allocator);
    }

    /// Compile a region descriptor into a TVM kernel artifact.
    ///
    /// Currently supports only single-equation matmul regions (dot or dot_general).
    /// 1. Validates region shape (single matmul equation).
    /// 2. Extracts M, N, K from input/output types.
    /// 3. Builds a matmul IRModule via TE.
    /// 4. Tunes via MetaSchedule (produces candidate .so files with device code).
    /// 5. Loads the best candidate .so and returns its bytes as a KernelArtifact.
    fn compile(self: *TvmProvider, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
        // Validate: single matmul equation
        const matmul = validate_matmul_region(desc) orelse return error.Unsupported;

        log.info("compiling matmul kernel: {s} ({d}x{d}x{d})", .{
            desc.name, matmul.m, matmul.n, matmul.k,
        });

        // Ensure TVM is loaded
        tvm_api.ensure_loaded(allocator, .{}) catch |err| {
            log.err("TVM runtime unavailable: {s}", .{@errorName(err)});
            return if (err == error.OutOfMemory) error.OutOfMemory else error.ProviderLoadFailed;
        };

        // Build matmul IRModule
        var ir_mod = tir.build_matmul_tir(allocator, matmul.m, matmul.n, matmul.k) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.TvmLoadFailed => {
                log.err("TVM runtime unavailable: {s}", .{@errorName(err)});
                return error.ProviderLoadFailed;
            },
            error.TvmCallFailed, error.TvmFunctionNotFound, error.UnexpectedTvmType => {
                log.err("TVM API call failed building IRModule: {s}", .{@errorName(err)});
                return error.ProviderCallFailed;
            },
        };
        defer ir_mod.deinit();

        // Create target
        var target = tir.Target.create(allocator, self.target_kind) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.TvmLoadFailed => {
                log.err("TVM runtime unavailable: {s}", .{@errorName(err)});
                return error.ProviderLoadFailed;
            },
            error.TvmCallFailed, error.TvmFunctionNotFound, error.UnexpectedTvmType => {
                log.err("TVM API call failed creating target: {s}", .{@errorName(err)});
                return error.ProviderCallFailed;
            },
        };
        defer target.deinit();

        // Tune
        const shape_a = [_]i64{ matmul.m, matmul.k };
        const shape_b = [_]i64{ matmul.k, matmul.n };
        const shape_c = [_]i64{ matmul.m, matmul.n };
        const shapes: [3][]const i64 = .{ &shape_a, &shape_b, &shape_c };

        // Per-target base dir for cache index and per-kernel subdirectories.
        const target_suffix = switch (self.target_kind) {
            .cpu => "cpu",
            .cuda => "cuda",
        };
        const base_dir = std.fmt.allocPrint(allocator, "{s}/{s}", .{ self.work_dir, target_suffix }) catch
            return error.OutOfMemory;
        defer allocator.free(base_dir);
        std.fs.cwd().makePath(base_dir) catch {};

        const key = try tuned_module.matmul_cache_key(allocator, self.target_kind, matmul.m, matmul.n, matmul.k);
        defer allocator.free(key);

        if (load_cached_kernel(allocator, base_dir, key, self.target_kind) catch null) |cached| {
            return .{
                .provider_name = "tvm",
                .data = cached,
                .target_name = try allocator.dupe(u8, key),
            };
        }

        const work_dir = tuned_module.ensure_cache_dir(allocator, base_dir, key) catch
            return error.OutOfMemory;
        defer allocator.free(work_dir);

        tune_mod.tune(allocator, ir_mod, target, self.target_kind, &shapes, .{
            .work_dir = work_dir,
            .max_trials = self.max_trials,
            .trials_per_iter = self.trials_per_iter,
        }) catch |err| switch (err) {
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

        const update = tuned_module.update_cache_from_work_dir(
            allocator,
            base_dir,
            work_dir,
            key,
            self.target_kind,
        ) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => {
                log.err("failed to update kernel cache index: {s}", .{@errorName(err)});
                return error.CompileFailed;
            },
        };
        defer allocator.free(update.stable_path);

        const so_bytes = std.fs.cwd().readFileAlloc(allocator, update.stable_path, 100 * 1024 * 1024) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => {
                log.err("failed to read {s}: {s}", .{ update.stable_path, @errorName(err) });
                return error.CompileFailed;
            },
        };

        log.info("compiled kernel: {s} (candidate {d}, {d:.2} us, {d} bytes)", .{
            desc.name, update.best_candidate, update.best_time_us, so_bytes.len,
        });

        return .{
            .provider_name = "tvm",
            .data = so_bytes,
            .target_name = try allocator.dupe(u8, key),
        };
    }
};

// ============================================================================
// Matmul validation
// ============================================================================

const MatmulShape = struct {
    m: i64,
    n: i64,
    k: i64,
};

/// Validate that a region describes a single matmul (dot or dot_general).
/// Returns the M, N, K dimensions, or null if unsupported.
fn validate_matmul_region(desc: kernel.RegionDescriptor) ?MatmulShape {
    // Must be exactly one op
    if (desc.ops.len != 1) return null;
    if (desc.inputs.len != 2 or desc.outputs.len != 1) return null;
    const op = desc.ops[0];

    // Must be dot or dot_general
    switch (op.params) {
        .dot => {},
        .dot_general => |dg| {
            if (!kernel.dot_general_is_matrix_matmul(dg)) return null;
        },
        else => return null,
    }

    // Must have 2 inputs and 1 output
    if (op.inputs.len != 2 or op.outputs.len != 1) return null;

    // Get types
    const a = op.inputs[0].value.as_tensor();
    const b = op.inputs[1].value.as_tensor();
    const c_tensor = op.outputs[0].as_tensor();

    // Must be rank-2 (matrix)
    if (a.shape.rank() != 2 or b.shape.rank() != 2 or c_tensor.shape.rank() != 2) return null;

    // A[M,K] @ B[K,N] = C[M,N]
    const m = a.shape.dims[0];
    const k = a.shape.dims[1];
    const n = b.shape.dims[1];

    // Validate consistency
    if (b.shape.dims[0] != k) return null;
    if (c_tensor.shape.dims[0] != m or c_tensor.shape.dims[1] != n) return null;

    // Only f32 for now
    if (a.dtype != .f32 or b.dtype != .f32) return null;

    return .{ .m = m, .n = n, .k = k };
}

fn load_cached_kernel(
    allocator: std.mem.Allocator,
    base_dir: []const u8,
    key: []const u8,
    target_kind: TargetKind,
) !?[]u8 {
    const cached = try tuned_module.cache_lookup(allocator, base_dir, key, target_kind);
    if (cached == null) return null;
    defer {
        allocator.free(cached.?.key);
        allocator.free(cached.?.artifact_path);
    }

    std.fs.cwd().access(cached.?.artifact_path, .{}) catch return null;
    return std.fs.cwd().readFileAlloc(allocator, cached.?.artifact_path, 100 * 1024 * 1024) catch null;
}
