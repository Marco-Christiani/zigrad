const root = @import("root");
const std = @import("std");
const zig_builtin = @import("builtin");
const build_options = @import("build_options");
pub const zarray = @import("zarray");
// pub const blas = @import("backend/blas.zig");

/// lib-wide options that can be overridden by the root file.
/// E.g. const zigrad_settings = .{ .gradEnabled = true };
pub const settings: Settings = if (@hasDecl(root, "zigrad_settings")) root.zigrad_settings else .{};

pub const Settings = struct {
    grad_enabled: bool = true, // TODO: implement grad_enabled
    grad_clip_max_norm: f32 = 10.0,
    grad_clip_delta: f32 = 1e-6,
    grad_clip_enabled: bool = true,
    seed: u64 = 4545611,
};

var prng = std.rand.DefaultPrng.init(settings.seed);
pub const random = prng.random();

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

// test {
//     _ = @import("tensor/zarray.zig");
//     _ = @import("tensor/tensor.zig");
//     std.testing.refAllDecls(@This());
// }
