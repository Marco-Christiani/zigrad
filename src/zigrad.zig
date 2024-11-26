pub const device = @import("device");
const DeviceReference = device.DeviceReference;

pub const Tensor = struct {
    device: DeviceReference,
    pub fn init(device_ptr: DeviceReference) Tensor {
        return .{ .device = device_ptr };    
    }  
};

