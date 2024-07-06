const NDTensor = @import("tensor.zig").NDTensor;
const random = @import("zigrad").random;

pub fn heInit(comptime T: type, tensor: NDTensor(T)) void {
    const fan_in: T = @floatFromInt(tensor.data.shape.shape[1]);
    const std_dev = @sqrt(2.0 / fan_in);
    for (tensor.data.data) |*value| {
        value.* = random.floatNorm(T) * std_dev;
    }
}
