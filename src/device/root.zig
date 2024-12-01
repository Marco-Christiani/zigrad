pub const Backend = enum{ HOST, CUDA };
pub const backend: Backend = .HOST;
pub const device = @import("host_device.zig");
