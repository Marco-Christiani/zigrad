const zg = @import("../root.zig");
const NDTensor = zg.NDTensor;
const settings = zg.settings;

pub fn clipGrads(T: type, params: []*const NDTensor(T), opts: NDTensor(T).ClipOptions) void {
    for (params) |param| if (param.grad) |_| param.clip_grad_norm_delta(opts);
}

pub fn SGD(comptime T: type) type {
    return struct {
        const Self = @This();
        lr: T,
        grad_clip_enabled: bool = settings.grad_clip_enabled,
        grad_clip_max_norm: f32 = settings.grad_clip_max_norm,
        grad_clip_delta: f32 = settings.grad_clip_delta,

        pub fn step(self: Self, params: []*const NDTensor(T)) void {
            if (self.grad_clip_enabled) clipGrads(T, params, .{
                .max_norm = self.grad_clip_max_norm,
                .delta = self.grad_clip_delta,
            });

            // lol. go to bed.
            for (params) |param| {
                // param.grad.?._scale(self.lr);
                // _ = param.data._sub(param.grad.?) catch unreachable;
                for (0..param.data.data.len) |j| {
                    param.data.data[j] -= self.lr * param.grad.?.data[j];
                }
            }
        }
    };
}
