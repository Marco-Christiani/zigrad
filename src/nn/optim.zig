const std = @import("std");
const math = std.math;

const zg = @import("../root.zig");
const NDTensor = zg.NDTensor;
const settings = zg.settings;

pub fn clipGrads(T: type, params: []*const NDTensor(T), opts: NDTensor(T).ClipOptions) void {
    for (params) |param| if (param.grad) |_| param.clip_grad_norm_delta(opts);
}

pub fn Optimizer(comptime T: type) type {
    return struct {
        const Self = @This();

        vtable: *const VTable,
        ptr: *anyopaque,

        const VTable = struct {
            step: *const fn (ctx: *anyopaque, params: []*const NDTensor(T)) anyerror!void,
        };

        pub fn init(pointer: anytype) Self {
            const Ptr = @TypeOf(pointer);

            const gen = struct {
                pub fn stepFn(ctx: *anyopaque, params: []*const NDTensor(T)) anyerror!void {
                    const self: Ptr = @ptrCast(@alignCast(ctx));
                    return self.step(params);
                }
                const vtable = VTable{ .step = stepFn };
            };
            return .{
                .vtable = &gen.vtable,
                .ptr = pointer,
            };
        }

        pub fn step(self: Self, params: []*const NDTensor(T)) anyerror!void {
            return try self.vtable.step(self.ptr, params);
        }
    };
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

            for (params) |param| {
                for (param.data.data, param.grad.?.data) |*p, *g| {
                    // param.data.data[j] -= self.lr * param.grad.?.data[j]; // turns out this line is *really* slow, why cant compiler optimize this?
                    p.* -= self.lr * g.*;
                }
            }
        }

        pub fn optimizer(self: *Self) Optimizer(T) {
            return Optimizer(T).init(self);
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
                .m = std.AutoHashMap(*const NDTensor(T), []T).init(allocator),
                .v = std.AutoHashMap(*const NDTensor(T), []T).init(allocator),
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

        pub fn optimizer(self: *Self) Optimizer(T) {
            return Optimizer(T).init(self);
        }

        pub fn step(self: *Self, params: []*const NDTensor(T)) !void {
            if (self.grad_clip_enabled) clipGrads(T, params, .{
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
