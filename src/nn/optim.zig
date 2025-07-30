const std = @import("std");
const math = std.math;

const zg = @import("../zigrad.zig");
const NDTensor = zg.NDTensor;
const settings = zg.settings;
const opspec = zg.opspec;

const Error = zg.device.Error || std.mem.Allocator.Error;
const OptimTag = enum { sgd, adam };
const UpdateCallback = *const fn (*anyopaque, *zg.Graph.Node) Error!void;

/// ParamEntry is a closure to support
/// generic type optimization.
const ParamEntry = struct {
    node_ptr: *zg.Graph.Node,
    upd_call: UpdateCallback,
};

/// Every optimizer backend will have a ParamList that
/// tracks objects that the Optimizer has attached to.
const ParamList = std.ArrayList(ParamEntry);

/// Generic interface for handling optimizers. You can think
/// of this as being similar to an Allocator interface. Each
/// optimizer backend will have an `optimzer` function that
/// returns an instance of this class.
pub const Optimizer = struct {
    ptr: *anyopaque,
    tag: OptimTag,
    params: *ParamList,

    pub fn attach(self: Optimizer, object: anytype) !void {
        comptime std.debug.assert(@typeInfo(@TypeOf(object)) == .pointer);
        const Object = std.meta.Child(@TypeOf(object));

        if (comptime !@hasField(Object, "node"))
            @compileError("Object does not have a 'node' field: " ++ @typeName(Object));

        try self.params.append(.{
            .node_ptr = &object.node,
            .upd_call = update_selector(self.tag, Object),
        });
    }

    pub fn step(self: Optimizer) Error!void {
        for (self.params.items) |entry| {
            try entry.upd_call(self.ptr, entry.node_ptr);
        }
    }

    fn update_selector(tag: OptimTag, Param: type) UpdateCallback {
        return switch (tag) {
            .sgd => update_wrapper(SGD, Param),
            .adam => update_wrapper(Adam, Param),
        };
    }

    fn update_wrapper(Optim: type, Param: type) UpdateCallback {
        return struct {
            pub fn update(ctx: *anyopaque, node: *zg.Graph.Node) Error!void {
                const optim: *Optim = @ptrCast(@alignCast(ctx));
                const param: *Param = node.upcast(Param);
                try optim.update(param);
            }
        }.update;
    }
};

pub const SGD = struct {
    params: ParamList,
    grad_clip_enabled: bool,
    grad_clip_max_norm: f32,
    grad_clip_delta: f32,
    lr: f64,

    pub fn init(allocator: std.mem.Allocator, opts: struct {
        grad_clip_enabled: bool = settings.grad_clip_enabled,
        grad_clip_max_norm: f32 = settings.grad_clip_max_norm,
        grad_clip_delta: f32 = settings.grad_clip_delta,
        lr: f64,
    }) SGD {
        return .{
            .params = ParamList.init(allocator),
            .grad_clip_enabled = opts.grad_clip_enabled,
            .grad_clip_max_norm = opts.grad_clip_max_norm,
            .grad_clip_delta = opts.grad_clip_delta,
            .lr = opts.lr,
        };
    }

    pub fn deinit(self: *SGD) void {
        self.params.deinit();
    }

    pub fn optimizer(self: *SGD) Optimizer {
        return .{ .ptr = self, .tag = .sgd, .params = &self.params };
    }

    pub fn update(self: *SGD, param: anytype) Error!void {
        const Param = std.meta.Child(@TypeOf(param));

        const nlr = -@as(Param.ValueType, @floatCast(self.lr));

        switch (comptime Param.Category) {
            .dense => {
                if (self.grad_clip_enabled) param._clip_grad_norm(.{
                    .max_norm = self.grad_clip_max_norm,
                    .delta = self.grad_clip_delta,
                });
                // I suppose the idiomatic way would be to use the method
                // for (params) |param| param.data._axpy(param.grad.?, nlr, param.device);
                // But, can use direct access to skip the shape checks
                param.device.dispatch(opspec.axpy(Param.ValueType){
                    .x = param.assume_grad_data(),
                    .y = param.get_data(),
                    .alpha = &nlr,
                });
            },
            else => @compileError("Unimplemented: SGD for " ++ @typeName(Param)),
        }
    }
};

pub const Adam = struct {
    const MapEntry = struct { m: []u8, v: []u8, device: zg.DeviceReference };
    const ParamMap = std.AutoArrayHashMap(usize, MapEntry);

    params: ParamList,
    map: ParamMap,

    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    grad_clip_enabled: bool = settings.grad_clip_enabled,
    grad_clip_max_norm: f32 = settings.grad_clip_max_norm,
    grad_clip_delta: f32 = settings.grad_clip_delta,
    t: usize,

    pub fn init(allocator: std.mem.Allocator, opts: struct {
        lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        grad_clip_enabled: bool = settings.grad_clip_enabled,
        grad_clip_max_norm: f32 = settings.grad_clip_max_norm,
        grad_clip_delta: f32 = settings.grad_clip_delta,
    }) Adam {
        return .{
            .params = ParamList.init(allocator),
            .map = ParamMap.init(allocator),
            .lr = opts.lr,
            .beta1 = opts.beta1,
            .beta2 = opts.beta2,
            .epsilon = opts.epsilon,
            .grad_clip_enabled = opts.grad_clip_enabled,
            .grad_clip_max_norm = opts.grad_clip_max_norm,
            .grad_clip_delta = opts.grad_clip_delta,
            .t = 0,
        };
    }

    pub fn deinit(self: *Adam) void {
        self.params.deinit();
        for (self.map.values()) |entry| {
            entry.device.mem_free(entry.m);
            entry.device.mem_free(entry.v);
        }
        self.map.deinit();
    }

    pub fn optimizer(self: *Adam) Optimizer {
        return .{ .ptr = self, .tag = .adam, .params = &self.params };
    }

    pub fn update(self: *Adam, param: anytype) Error!void {
        if (!param.device.is_host())
            @panic("TODO: implement for non-host devices");

        const Param = std.meta.Child(@TypeOf(param));
        const T = Param.ValueType;

        const lr: T = @floatCast(self.lr);
        const beta1: T = @floatCast(self.beta1);
        const beta2: T = @floatCast(self.beta2);
        const epsilon: T = @floatCast(self.epsilon);

        self.t += 1;
        const t_f: T = @floatFromInt(self.t);
        const lr_t = lr * math.sqrt(1 - math.pow(T, beta2, t_f)) / (1 - math.pow(T, beta1, t_f));

        switch (comptime Param.Category) {
            .dense => {
                if (self.grad_clip_enabled) param._clip_grad_norm(.{
                    .max_norm = self.grad_clip_max_norm,
                    .delta = self.grad_clip_delta,
                });

                const param_size = param.get_size();

                const m, const v = blk: { // initialize m and v
                    const gop = try self.map.getOrPut(@intFromPtr(param));
                    if (!gop.found_existing) {
                        const m_bytes = try param.device.mem_alloc(u8, param_size * @sizeOf(T));
                        const v_bytes = try param.device.mem_alloc(u8, param_size * @sizeOf(T));
                        param.device.mem_fill(u8, m_bytes, 0);
                        param.device.mem_fill(u8, v_bytes, 0);
                        gop.value_ptr.* = .{
                            .m = m_bytes,
                            .v = v_bytes,
                            .device = param.device,
                        };
                    }
                    break :blk .{
                        std.mem.bytesAsSlice(T, gop.value_ptr.m),
                        std.mem.bytesAsSlice(T, gop.value_ptr.v),
                    };
                };

                const p_data = param.get_data();
                const p_grad = param.assume_grad_data();

                // TODO: SIMD or BLAS
                for (0..param_size) |i| {
                    const g = p_grad[i];
                    m[i] = beta1 * m[i] + (1 - beta1) * g;
                    v[i] = beta2 * v[i] + (1 - beta2) * g * g;
                    p_data[i] -= lr_t * m[i] / (math.sqrt(v[i]) + epsilon);
                }
            },
            else => @compileError("Unimplemented: Adam for " ++ @typeName(Param)),
        }
    }
};

test {
    std.testing.refAllDeclsRecursive(@This());
}
