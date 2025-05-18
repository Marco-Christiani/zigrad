const std = @import("std");
const math = std.math;

const zg = @import("../zigrad.zig");
const NDTensor = zg.NDTensor;
const settings = zg.settings;
const opspec = zg.opspec;

pub fn clip_grads(T: type, params: []*NDTensor(T), opts: NDTensor(T).ClipOptions) void {
    for (params) |param| if (param.grad) |_| param._clip_grad_norm(opts);
}

pub fn SGD(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);
        lr: T,
        grad_clip_enabled: bool = settings.grad_clip_enabled,
        grad_clip_max_norm: f32 = settings.grad_clip_max_norm,
        grad_clip_delta: f32 = settings.grad_clip_delta,

        pub fn step(self: *Self, params: []const *Tensor) void {
            const nlr = -self.lr;

            for (params) |param| {
                if (self.grad_clip_enabled) param._clip_grad_norm(.{
                    .max_norm = self.grad_clip_max_norm,
                    .delta = self.grad_clip_delta,
                });
                // I suppose the idiomatic way would be to use the method
                // for (params) |param| param.data._axpy(param.grad.?, nlr, param.device);
                // But, can use direct access to skip the shape checks
                param.device.dispatch(opspec.axpy(T){
                    .x = param.assume_grad_data(),
                    .y = param.get_data(),
                    .alpha = &nlr,
                });
            }
        }
    };
}

pub fn Adam(comptime T: type) type {
    return struct {
        const Self = @This();

        lr: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        t: usize,
        m: std.AutoHashMap(*const NDTensor(T), []T),
        v: std.AutoHashMap(*const NDTensor(T), []T),
        allocator: std.mem.Allocator,
        grad_clip_enabled: bool = settings.grad_clip_enabled,
        grad_clip_max_norm: f32 = settings.grad_clip_max_norm,
        grad_clip_delta: f32 = settings.grad_clip_delta,

        pub fn init(allocator: std.mem.Allocator, learning_rate: T, beta1: T, beta2: T, epsilon: T) Self {
            return Self{
                .lr = learning_rate,
                .beta1 = beta1,
                .beta2 = beta2,
                .epsilon = epsilon,
                .t = 0,
                .m = std.AutoHashMap(*NDTensor(T), []T).init(allocator),
                .v = std.AutoHashMap(*NDTensor(T), []T).init(allocator),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            var m_it = self.m.iterator();
            while (m_it.next()) |entry| {
                self.allocator.free(entry.value_ptr.*);
            }
            self.m.deinit();

            var v_it = self.v.iterator();
            while (v_it.next()) |entry| {
                self.allocator.free(entry.value_ptr.*);
            }
            self.v.deinit();
        }

        pub fn step(self: *Self, params: []const *NDTensor(T)) !void {
            if (self.grad_clip_enabled) clip_grads(T, params, .{
                .max_norm = self.grad_clip_max_norm,
                .delta = self.grad_clip_delta,
            });
            self.t += 1;
            const t_f: T = @floatFromInt(self.t);
            const lr_t = self.lr * math.sqrt(1 - math.pow(T, self.beta2, t_f)) / (1 - math.pow(T, self.beta1, t_f));

            for (params) |param| {
                const param_size = param.data.data.len;

                // initialize m and v
                if (!self.m.contains(param)) {
                    const m_data = try self.allocator.alloc(T, param_size);
                    @memset(m_data, 0);
                    try self.m.put(param, m_data);
                }
                if (!self.v.contains(param)) {
                    const v_data = try self.allocator.alloc(T, param_size);
                    @memset(v_data, 0);
                    try self.v.put(param, v_data);
                }

                const m = self.m.get(param).?;
                const v = self.v.get(param).?;

                // TODO: SIMD or BLAS
                for (0..param_size) |i| {
                    const g = param.grad.?.data[i];
                    m[i] = self.beta1 * m[i] + (1 - self.beta1) * g;
                    v[i] = self.beta2 * v[i] + (1 - self.beta2) * g * g;
                    param.data.data[i] -= lr_t * m[i] / (math.sqrt(v[i]) + self.epsilon);
                }
            }
        }
    };
}
