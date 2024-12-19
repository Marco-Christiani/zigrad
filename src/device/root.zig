pub const Backend = enum{ HOST, CUDA };
pub const backend: Backend = .HOST;
pub const device = @import("host_device.zig");
pub const ReduceType = @import("device_common.zig").ReduceType;
pub const SmaxType = @import("device_common.zig").SmaxType;
pub const RandType = @import("device_common.zig").RandType;
pub const using_mkl = @import("host_device.zig").using_mkl;