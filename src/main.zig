const std = @import("std");
const zg = @import("zigrad");

pub fn main() !void {
    // only compiles if -Dcuda=true
    //var device = zg.device.CudaDevice{};
    var device = zg.device.HostDevice{};
    const t = zg.Tensor.init(device.reference());
    t.device.print();
}
