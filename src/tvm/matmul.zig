//! TVM matmul tuning and execution.

const std = @import("std");

const device = @import("../device.zig");
const dlpack = @import("../c/dlpack.zig");
const tvm_runtime = @import("../c/tvm/runtime.zig");
const tir = @import("../c/tvm/tir.zig");
const Cache = @import("../cache.zig").Cache;
const artifact = @import("artifact.zig");
const config = @import("config.zig");
pub const TargetKind = config.TargetKind;
const integration_runtime = @import("runtime.zig");
const tune_mod = @import("tune.zig");

/// Logical dimensions for an f32 matrix multiplication.
pub const Shape = struct {
    m: i64,
    n: i64,
    k: i64,
};

/// Options for tuning one matrix multiplication shape.
pub const TuneOptions = struct {
    /// Target-specific compiler inputs resolved by application composition.
    compile: config.CompileConfig,

    /// Device used for target detection and candidate measurement.
    device: device.Device,

    /// Delete existing work artifacts before tuning.
    retune: bool = false,

    /// Maximum measured candidates.
    max_trials: u32 = 64,

    /// Candidates submitted per tuning iteration.
    trials_per_iter: u32 = 16,
};

/// Selected result from one tuning run.
pub const TuneResult = struct {
    best_candidate: usize,
    best_time_us: f64,
};

/// Tune one f32 matrix multiplication and update the stable cache entry.
pub fn tune(
    io: std.Io,
    allocator: std.mem.Allocator,
    artifact_cache: Cache,
    shape: Shape,
    options: TuneOptions,
) !TuneResult {
    try integration_runtime.ensure_loaded(.compiler);

    const target = options.compile.target;
    if (!target.accepts(options.device)) return error.TargetDeviceMismatch;
    var tvm_target = try tir.Target.resolve(
        allocator,
        target,
        options.device.ordinal,
    );
    defer tvm_target.deinit();
    const location = try cache_location(
        io,
        artifact_cache,
        shape,
        tvm_target.description,
    );
    const work_cache = try location.base.subdir(io, location.key.slice(), .{});

    if (options.retune) {
        std.Io.Dir.cwd().deleteTree(io, work_cache.path()) catch {};
        try std.Io.Dir.cwd().createDirPath(io, work_cache.path());
    }

    var ir_module = try tir.build_matmul_tir(
        allocator,
        shape.m,
        shape.n,
        shape.k,
    );
    defer ir_module.deinit();
    const shape_a = [_]i64{ shape.m, shape.k };
    const shape_b = [_]i64{ shape.k, shape.n };
    const shape_c = [_]i64{ shape.m, shape.n };
    const shapes = [_][]const i64{ &shape_a, &shape_b, &shape_c };

    try tune_mod.tune(
        io,
        allocator,
        ir_module,
        tvm_target.target,
        &shapes,
        .{
            .compile = options.compile,
            .device = options.device,
            .gpu_arch = tvm_target.gpu_arch,
            .work_cache = work_cache,
            .max_trials = options.max_trials,
            .trials_per_iter = options.trials_per_iter,
        },
    );

    const update = try artifact.update_cache_from_work_dir(
        io,
        allocator,
        location.base,
        work_cache,
        location.key.slice(),
        target,
    );
    return .{
        .best_candidate = update.best_candidate,
        .best_time_us = update.best_time_us,
    };
}

/// Bytes and cache identity for one stable matmul artifact.
///
/// The caller frees `bytes` with the allocator passed to `load_artifact`.
pub const CachedArtifact = struct {
    key: artifact.CacheKey,
    bytes: []u8,
};

/// Read a stable matmul artifact without exposing cache-index details.
pub fn load_artifact(
    io: std.Io,
    allocator: std.mem.Allocator,
    artifact_cache: Cache,
    shape: Shape,
    target: TargetKind,
    selected_device: device.Device,
) !?CachedArtifact {
    try integration_runtime.ensure_loaded(.compiler);
    if (!target.accepts(selected_device)) return error.TargetDeviceMismatch;
    var target_description = try tir.Target.describe(
        allocator,
        target,
        selected_device.ordinal,
    );
    defer target_description.deinit();
    const location = try cache_location(
        io,
        artifact_cache,
        shape,
        target_description.description,
    );
    const bytes = try artifact.read_cached_bytes(
        io,
        allocator,
        location.base,
        location.key.slice(),
        target,
    ) orelse return null;
    return .{ .key = location.key, .bytes = bytes };
}

const CachedMatmulState = struct {
    allocator: std.mem.Allocator,
    target: TargetKind,
    device: device.Device,
    shape: Shape,
    artifact: artifact.LoadedArtifact,
};

/// Loaded cached matmul with no raw TVM or DLPack types in its contract.
pub const CachedMatmul = opaque {
    /// Load the stable cached implementation for one shape and target.
    pub fn load(
        io: std.Io,
        allocator: std.mem.Allocator,
        artifact_cache: Cache,
        shape: Shape,
        target: TargetKind,
        selected_device: device.Device,
    ) !?*CachedMatmul {
        try integration_runtime.ensure_loaded(.runtime);

        if (!target.accepts(selected_device)) return error.TargetDeviceMismatch;
        var target_description = try tir.Target.describe(
            allocator,
            target,
            selected_device.ordinal,
        );
        defer target_description.deinit();
        const location = try cache_location(
            io,
            artifact_cache,
            shape,
            target_description.description,
        );
        var loaded = try artifact.load_cached(
            io,
            allocator,
            location.base,
            location.key.slice(),
            target,
        ) orelse return null;
        errdefer loaded.deinit();

        const state = try allocator.create(CachedMatmulState);
        state.* = .{
            .allocator = allocator,
            .target = target,
            .device = selected_device,
            .shape = shape,
            .artifact = loaded,
        };
        return @ptrCast(state);
    }

    /// Release the loaded module and wrapper state.
    pub fn deinit(self: *CachedMatmul) void {
        const state = state_from(self);
        const allocator = state.allocator;
        state.artifact.deinit();
        allocator.destroy(state);
    }

    /// Execute the loaded f32 matrix multiplication with borrowed host slices.
    pub fn execute(
        self: *CachedMatmul,
        lhs: []const f32,
        rhs: []const f32,
        output: []f32,
    ) !void {
        const state = state_from(self);
        const shape = state.shape;
        const counts = try element_counts(shape);
        if (lhs.len != counts.lhs or
            rhs.len != counts.rhs or
            output.len != counts.output)
        {
            return error.InvalidShape;
        }

        var shape_lhs = [_]i64{ shape.m, shape.k };
        var shape_rhs = [_]i64{ shape.k, shape.n };
        var shape_output = [_]i64{ shape.m, shape.n };

        const allocator = state.allocator;
        switch (state.target) {
            .cpu => {
                var dl_lhs = dlpack.ManagedTensor.borrowing(
                    dlpack.Tensor.init_contiguous(f32, @constCast(lhs), &shape_lhs),
                );
                var dl_rhs = dlpack.ManagedTensor.borrowing(
                    dlpack.Tensor.init_contiguous(f32, @constCast(rhs), &shape_rhs),
                );
                var dl_output = dlpack.ManagedTensor.borrowing(
                    dlpack.Tensor.init_contiguous(f32, output, &shape_output),
                );

                var tvm_lhs = try tvm_runtime.Tensor.from_dlpack(&dl_lhs);
                defer tvm_lhs.deinit();
                var tvm_rhs = try tvm_runtime.Tensor.from_dlpack(&dl_rhs);
                defer tvm_rhs.deinit();
                var tvm_output = try tvm_runtime.Tensor.from_dlpack(&dl_output);
                defer tvm_output.deinit();

                try state.artifact.invoke(allocator, &.{
                    tvm_lhs.as_value(),
                    tvm_rhs.as_value(),
                    tvm_output.as_value(),
                });
            },
            .cuda => {
                var tvm_lhs = try tvm_runtime.Tensor.allocate(
                    allocator,
                    @constCast(lhs),
                    &shape_lhs,
                    .cuda,
                    state.device.ordinal,
                );
                defer tvm_lhs.deinit();
                var tvm_rhs = try tvm_runtime.Tensor.allocate(
                    allocator,
                    @constCast(rhs),
                    &shape_rhs,
                    .cuda,
                    state.device.ordinal,
                );
                defer tvm_rhs.deinit();

                const output_init = try allocator.alloc(f32, output.len);
                defer allocator.free(output_init);
                @memset(output_init, 0);
                var tvm_output = try tvm_runtime.Tensor.allocate(
                    allocator,
                    output_init,
                    &shape_output,
                    .cuda,
                    state.device.ordinal,
                );
                defer tvm_output.deinit();

                try state.artifact.invoke(allocator, &.{
                    tvm_lhs.as_value(),
                    tvm_rhs.as_value(),
                    tvm_output.as_value(),
                });
                try tvm_output.copy_to_host(allocator, output);
            },
        }
    }

    fn state_from(self: *CachedMatmul) *CachedMatmulState {
        return @ptrCast(@alignCast(self));
    }
};

const ElementCounts = struct {
    lhs: usize,
    rhs: usize,
    output: usize,
};

const CacheLocation = struct {
    base: Cache,
    key: artifact.CacheKey,
};

fn cache_location(
    io: std.Io,
    artifact_cache: Cache,
    shape: Shape,
    target_description: []const u8,
) !CacheLocation {
    try validate_shape(shape);
    return .{
        .base = try artifact_cache.subdir(io, "tvm", .{}),
        .key = artifact.matmul_cache_key(
            target_description,
            shape.m,
            shape.n,
            shape.k,
        ),
    };
}

fn validate_shape(shape: Shape) error{InvalidShape}!void {
    if (shape.m <= 0 or shape.n <= 0 or shape.k <= 0)
        return error.InvalidShape;
}

fn element_counts(shape: Shape) error{InvalidShape}!ElementCounts {
    try validate_shape(shape);
    const m_size = std.math.cast(usize, shape.m) orelse return error.InvalidShape;
    const n_size = std.math.cast(usize, shape.n) orelse return error.InvalidShape;
    const k_size = std.math.cast(usize, shape.k) orelse return error.InvalidShape;
    return .{
        .lhs = std.math.mul(usize, m_size, k_size) catch return error.InvalidShape,
        .rhs = std.math.mul(usize, k_size, n_size) catch return error.InvalidShape,
        .output = std.math.mul(usize, m_size, n_size) catch return error.InvalidShape,
    };
}

test element_counts {
    try std.testing.expectEqualDeep(
        ElementCounts{ .lhs = 6, .rhs = 6, .output = 4 },
        try element_counts(.{ .m = 2, .n = 2, .k = 3 }),
    );
    try std.testing.expectError(
        error.InvalidShape,
        element_counts(.{ .m = 0, .n = 2, .k = 3 }),
    );
}
