//! TVM kernel provider.
//!
//! Implements the KernelProvider interface (src/kernel.zig) for TVM.
//! Handles matmul (dot/dot_general) kernels via MetaSchedule autotuning.
//! No TVM C types cross this boundary — only PR types and KernelArtifact.
const std = @import("std");
const tvm_types = @import("../ffi/tvm/types.zig");
const tvm_api = @import("../ffi/tvm/api.zig");
const tune_mod = @import("tune.zig");
const kernel = @import("../kernel.zig");
const pr = @import("../pr/pr.zig");
const TargetKind = @import("../ffi/tvm/types.zig").TargetKind;

const log = std.log.scoped(.@"zg/tvm_provider");

pub const TvmProvider = struct {
    allocator: std.mem.Allocator,
    target_kind: TargetKind,
    work_dir: []const u8,
    max_trials: u32 = 64,
    trials_per_iter: u32 = 16,

    /// Return a KernelProvider interface backed by this TvmProvider.
    pub fn kernel_provider(self: *TvmProvider) kernel.KernelProvider {
        return .{
            .name = "tvm",
            .ptr = @ptrCast(self),
            .compile_fn = compile_impl,
        };
    }

    fn compile_impl(ptr: *anyopaque, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
        const self: *TvmProvider = @ptrCast(@alignCast(ptr));
        return self.compile(desc, allocator);
    }

    /// Compile a region descriptor into a TVM kernel artifact.
    ///
    /// Currently supports only single-equation matmul regions (dot or dot_general).
    /// 1. Validates region shape (single matmul equation).
    /// 2. Extracts M, N, K from input/output types.
    /// 3. Builds a matmul IRModule via TE.
    /// 4. Tunes via MetaSchedule.
    /// 5. Lowers, builds, and exports to .so.
    /// 6. Reads .so bytes into a KernelArtifact.
    fn compile(self: *TvmProvider, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
        // Validate: single matmul equation
        const matmul = validate_matmul_region(desc) orelse return error.Unsupported;

        log.info("compiling matmul kernel: {s} ({d}x{d}x{d})", .{
            desc.name, matmul.m, matmul.n, matmul.k,
        });

        // Ensure TVM is loaded
        tvm_api.ensure_loaded(allocator) catch return error.CompileFailed;

        // Build matmul IRModule
        var ir_mod = tvm_types.build_matmul_tir(allocator, matmul.m, matmul.n, matmul.k) catch |err| {
            log.err("build_matmul_tir failed: {s}", .{@errorName(err)});
            return error.CompileFailed;
        };
        defer ir_mod.deinit();

        // Create target
        var target = tvm_types.Target.create(allocator, self.target_kind) catch return error.CompileFailed;
        defer target.deinit();

        // Tune
        const shape_a = [_]i64{ @intCast(matmul.m), @intCast(matmul.k) };
        const shape_b = [_]i64{ @intCast(matmul.k), @intCast(matmul.n) };
        const shape_c = [_]i64{ @intCast(matmul.m), @intCast(matmul.n) };
        const shapes: [3][]const i64 = .{ &shape_a, &shape_b, &shape_c };

        // Per-target work dir to avoid database contamination
        const target_suffix = switch (self.target_kind) {
            .cpu => "cpu",
            .cuda => "cuda",
        };
        const work_dir = std.fmt.allocPrint(allocator, "{s}/{s}", .{ self.work_dir, target_suffix }) catch
            return error.OutOfMemory;
        defer allocator.free(work_dir);

        tune_mod.tune(allocator, ir_mod, target, self.target_kind, &shapes, .{
            .work_dir = work_dir,
            .max_trials = self.max_trials,
            .trials_per_iter = self.trials_per_iter,
        }) catch |err| {
            log.err("tuning failed: {s}", .{@errorName(err)});
            return error.CompileFailed;
        };

        // Rebuild with tuned schedule (re-create IRModule since tuning consumed it)
        var tuned_mod = tvm_types.build_matmul_tir(allocator, matmul.m, matmul.n, matmul.k) catch
            return error.CompileFailed;

        // Lower and build
        var built = tvm_types.lower_and_build(allocator, &tuned_mod, target, self.target_kind) catch |err| {
            log.err("lower_and_build failed: {s}", .{@errorName(err)});
            tuned_mod.deinit();
            return error.CompileFailed;
        };
        defer built.deinit();

        // Export to .so
        const so_path = std.fmt.allocPrint(allocator, "{s}/{s}.so", .{ work_dir, desc.name }) catch
            return error.OutOfMemory;
        defer allocator.free(so_path);

        built.export_shared(allocator, so_path, self.target_kind) catch |err| {
            log.err("export_shared failed: {s}", .{@errorName(err)});
            return error.CompileFailed;
        };

        // Read .so bytes
        const so_bytes = std.fs.cwd().readFileAlloc(allocator, so_path, 100 * 1024 * 1024) catch |err| {
            log.err("failed to read {s}: {s}", .{ so_path, @errorName(err) });
            return error.CompileFailed;
        };

        log.info("compiled kernel: {s} ({d} bytes)", .{ desc.name, so_bytes.len });

        return .{
            .provider_name = "tvm",
            .data = so_bytes,
            .target_name = desc.name,
        };
    }
};

// ============================================================================
// Matmul validation
// ============================================================================

const MatmulShape = struct {
    m: usize,
    n: usize,
    k: usize,
};

/// Validate that a region describes a single matmul (dot or dot_general).
/// Returns the M, N, K dimensions, or null if unsupported.
fn validate_matmul_region(desc: kernel.RegionDescriptor) ?MatmulShape {
    // Must be exactly one equation
    if (desc.eqns.len != 1) return null;
    const eqn = desc.eqns[0];

    // Must be dot or dot_general
    switch (eqn.prim) {
        .dot, .dot_general => {},
        else => return null,
    }

    // Must have 2 inputs and 1 output
    const inputs = eqn.inputs.slice(pr.VarId, desc.varids_store);
    const outputs = eqn.outputs.slice(pr.VarId, desc.varids_store);
    if (inputs.len != 2 or outputs.len != 1) return null;

    // Get types
    const a_aval = desc.aval_of(inputs[0]) orelse return null;
    const b_aval = desc.aval_of(inputs[1]) orelse return null;
    const c_aval = desc.aval_of(outputs[0]) orelse return null;

    const a = a_aval.as_tensor() orelse return null;
    const b = b_aval.as_tensor() orelse return null;
    const c_tensor = c_aval.as_tensor() orelse return null;

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
