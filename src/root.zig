const root = @import("root");
const std = @import("std");

/// lib-wide options that can be overridden by the root file.
/// E.g. const zigrad_settings = .{ .gradEnabled = true };
pub const settings: Settings = if (@hasDecl(root, "zigrad_settings")) root.zigrad_settings else .{};

// pub const settings: Settings = blk: {
//     if (@hasDecl(root, "zigrad_settings")) {
//         break :blk root.zigrad_settings;
//     } else {
//         break :blk .{};
//     }
// };

pub const Settings = struct {
    grad_enabled: bool = true, // TODO: implement grad_enabled
    grad_clip_max_norm: f32 = 10.0,
    grad_clip_delta: f32 = 1e-6,
    grad_clip_enabled: bool = true,
};

pub fn foo() !void {
    std.debug.print("root is: {}\n", .{root});
}

// comptime and mutable globals can be frustrating. need a better idea
// pub const Settings = struct {
//     grad_enabled: bool,
//     max_dim: usize,
//     grad_clip_max_norm: f32,
//     grad_clip_delta: f32,
//     grad_clip_enabled: bool,
//
//     var default: Settings = .{
//         .grad_enabled = true,
//         .max_dim = 4,
//         .grad_clip_max_norm = 10.0,
//         .grad_clip_delta = 1e-6,
//         .grad_clip_enabled = true,
//     };
// };
