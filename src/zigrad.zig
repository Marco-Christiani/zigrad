//! - Treat this as a way to search and browse the source code, documentation is otherwise lacking due to how quickly Zigrad has been iterating.
//! - In this documentation COM means "caller owns memory"
//! - ADR denotes decision records, this is essentially a discourse explaining rationale because things get complicated
//! - Grep for "TODO" if you are interested in helping
//! - If you got this far there are more detailed roadmap and arch notes under docs/
const root = @import("root");
const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

pub const NDArray = @import("ndarray.zig").NDArray;
pub const Shape = @import("ndarray.zig").Shape;
pub const arrayutils = @import("ndarray.zig").utils;
pub const NDTensor = @import("ndtensor.zig").NDTensor;
pub const Op = @import("ndtensor.zig").Op;
pub const loss = @import("nn/loss.zig");
pub const mnist = @import("nn/mnist.zig");
pub const Graph = @import("graph.zig");
pub const TensorOpts = @import("ndtensor/utils.zig").TensorOpts;
//pub const Model = @import("nn/model.zig").Model;
pub const conv_utils = @import("nn/conv_utils.zig");
pub const utils = @import("nn/utils.zig");
//pub const layer = @import("nn/layer.zig");
pub const optim = @import("nn/optim.zig");
pub const Optimizer = optim.Optimizer;
pub const nn = @import("nn/nn.zig");

/// The category tag is used to categorize different
/// mathematical objects based on their mathematical
/// category. This is used as a convenience tag to
/// make comptime deduction easier.
pub const Category = enum {
    dense, // NDTensor
    sparse, // NDSparse
};

const device_root = @import("device.zig");
pub const device = struct {
    pub const Error = device_root.Error;
    pub const HostDevice = device_root.HostDevice;
    pub const CudaDevice = device_root.CudaDevice;
    pub const DeviceData = device_root.DeviceData;
};
pub const opspec = device_root.opspec;
pub const DeviceReference = device_root.DeviceReference;
pub const ReduceType = device_root.ReduceType;
pub const RandType = device_root.RandType;
pub const SmaxType = device_root.SmaxType;

// we should make this a build option like

/// lib-wide default options that can be overridden by the root file.
/// Note that these values can be overridden at call-site, this is just a way to configure global defaults.
/// E.g. in your main file `const zigrad_settings = .{ .gradEnabled = true };`
pub const settings: Settings = if (@hasDecl(root, "zigrad_settings")) root.zigrad_settings else .{};

pub const constants = @import("allocators.zig").constants;

/// Default values
pub const Settings = struct {
    grad_clip_max_norm: f32 = 10.0,
    grad_clip_delta: f32 = 1e-6,
    grad_clip_enabled: bool = true,
    seed: u64 = 81761,
    backward_children_capacity: usize = 8,
    label_capacity: usize = 32,
    thread_safe: bool = !builtin.single_threaded,
};

/// Global flag for enabling/disabling gradient tracking.
/// NOTE: there should be an inference mode coming, there was a comptime disable flag to allow for
/// more optimizations tbd if it will be added back in the future.
pub const runtime = struct {
    pub var grad_enabled: bool = true;
    // TODO: At some point, I'd rather turn this into
    // a system memory check to verify availability.
    // Right now, 64 gigs until more research...
    pub var max_cache_size: usize = constants.@"1Gb" * 16;
};

var prng = std.Random.DefaultPrng.init(settings.seed);
/// currently only used for generating node labels when tracing the comp graph
pub const random = prng.random();

// Global computation graph - can be used as a convenience to avoid passing graphs
// along control-flow paths.
var global_graph: ?Graph = null;

pub fn global_graph_init(allocator: std.mem.Allocator, config: Graph.Opts) void {
    global_graph = Graph.init(allocator, config);
}
pub fn global_graph_deinit() void {
    if (global_graph) |*gg| gg.deinit();
}
pub fn global_graph_get() *Graph {
    return &(global_graph orelse @panic("Global graph is uninitialized, call init_global_graph first."));
}

// see zls
// The type of `log_level` is not `std.log.Level`.
var actual_log_level: std.log.Level = @enumFromInt(@intFromEnum(build_options.log_level));

fn logFn(
    comptime level: std.log.Level,
    comptime scope: @TypeOf(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    if (@intFromEnum(level) > @intFromEnum(actual_log_level)) return;

    const level_txt = comptime level.asText();
    const scope_txt = comptime @tagName(scope);
    const stderr = std.io.getStdErr().writer();
    std.debug.lockStdErr();
    defer std.debug.unlockStdErr();

    stderr.print("[{s:<5}] {s:^6}: ", .{ level_txt, if (comptime std.mem.startsWith(u8, scope_txt, "zg_")) scope_txt[3..] else scope_txt }) catch return;
    stderr.print(format, args) catch return;
    stderr.writeByte('\n') catch return;
}

pub const std_options = std.Options{
    // Always set this to debug to make std.log call into our handler, then control the runtime
    // value in logFn itself
    .log_level = .debug,
    .logFn = logFn,
};

// TODO: lib tests, recursive inclusion *in progress*
test {
    _ = @import("ndarray.zig");
    _ = @import("nn/tests/test_loss.zig");
    std.testing.refAllDecls(@This());
}
