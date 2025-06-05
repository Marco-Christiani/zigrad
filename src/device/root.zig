pub const HostDevice = @import("host_device.zig");
pub const CudaDevice = @import("host_device.zig");
pub const DeviceReference = @import("device_reference.zig");
pub const ReduceType = @import("device_common.zig").ReduceType;
pub const SmaxType = @import("device_common.zig").SmaxType;
pub const RandType = @import("device_common.zig").RandType;
pub const using_mkl = @import("host_device.zig").using_mkl;
pub const opspec = @import("opspec.zig");
