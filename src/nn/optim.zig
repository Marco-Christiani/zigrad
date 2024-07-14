const NDTensor = @import("../root.zig").NDTensor;

pub fn SGD(comptime T: type) type {
    return struct {
        const Self = @This();
        lr: T,

        pub fn step(self: Self, params: []*const NDTensor(T)) void {
            // lol. go to bed.
            for (params) |param| {
                // param.data._sub(param.grad.?._mul(self.lr)); // TODO: really need to implement scalar ops...
                for (0..param.data.data.len) |j| {
                    param.data.data[j] -= self.lr * param.grad.?.data[j];
                }
            }
        }
    };
}