const std = @import("std");
const zg = @import("../zigrad.zig");
const opspec = zg.opspec;
const NDTensor = zg.NDTensor;

/// rectified linear-unit
pub fn relu(T: type, x: *NDTensor(T)) !void {
    const Tensor = NDTensor(T);

    const BwdClosure = struct {
        pub fn callback(_y: *Tensor, children: *Tensor.Children) !void {
            const _x = children.get_bwd(0) orelse return;
            _y.device.dispatch(opspec.relu_bwd(T){
                .x = _x.get_data(),
                .x_g = try _x.ensure_grad_data(0),
                .y_g = _y.assume_grad_data(),
            });
        }
    };

    const y = try Tensor.DataType.empty(x.get_shape(), x.device);

    x.device.dispatch(opspec.relu_fwd(T){
        .x = x.get_data(),
        .y = y.get_data(),
    });

    return Tensor.create_dependent(BwdClosure, x, .{
        .data = y,
        .children = &.{x},
        .device = x.device,
        .callback = .{},
    });
}

/// inplace rectified linear-unit (masked)
pub fn relu_(T: type, x: *NDTensor(T)) !void {
    const Tensor = NDTensor(T);

    const BwdClosure = struct {
        version: u8,
        mask: []u8,
        pub fn callback(y: *Tensor, _: *Tensor.Children, ctx: *@This()) !void {
            std.debug.assert(ctx.version == y._version);
            y.device.dispatch(opspec.relu_mask_bwd(T){
                .x_g = try y.ensure_grad_data(0),
                .mask = ctx.mask,
            });
            y.device.mem_free(ctx.mask);
        }
    };

    if (!x.requires_grad()) {
        return x.device.dispatch(opspec.relu_fwd(T){
            .x = x.get_data(),
            .y = x.get_data(),
        });
    }

    const mask = try x.device.mem_alloc_byte_mask(x.get_size());
    errdefer x.device.mem_free(mask);

    x.device.dispatch(opspec.relu_mask_fwd(T){
        .x = x.get_data(),
        .mask = mask,
    });

    try Tensor.prepend_dependent(BwdClosure, x, .{
        .callback = .{
            .version = x._version +% 1,
            .mask = mask,
        },
        .children = &.{},
        .device = x.device,
    });
}

/// hyperbolic-tangent
pub fn tanh(T: type, x: *NDTensor(T)) !void {
    const Tensor = NDTensor(T);

    const BwdClosure = struct {
        pub fn callback(_y: *Tensor, children: *Tensor.Children) !void {
            const _x = children.get_bwd(0) orelse return;
            _y.device.dispatch(opspec.tanh_bwd(T){
                .x_g = try _x.ensure_grad_data(0),
                .y = _y.get_data(),
                .y_g = _y.assume_grad_data(),
            });
        }
    };

    const y = try Tensor.DataType.empty(x.get_shape(), x.device);

    x.device.dispatch(opspec.tanh_fwd(T){
        .x = x.get_data(),
        .y = y.data,
    });

    return Tensor.create_dependent(BwdClosure, x, .{
        .data = y,
        .children = &.{x},
        .device = x.device,
        .callback = .{},
    });
}

// inplace sigmoid
pub fn tanh_(T: type, x: *NDTensor(T)) !void {
    const Tensor = NDTensor(T);

    const BwdClosure = struct {
        version: u8,
        pub fn callback(_x: *Tensor, _: *Tensor.Children, ctx: *@This()) !void {
            std.debug.assert(ctx.version == _x._version);
            _x.device.dispatch(opspec.tanh_inplace_bwd(T){
                .x = _x.get_data(),
                .x_g = try _x.ensure_grad_data(0),
            });
        }
    };

    x.device.dispatch(opspec.tanh_fwd(T){
        .x = x.get_data(),
        .y = x.get_data(),
    });

    try Tensor.prepend_dependent(BwdClosure, x, .{
        .callback = .{ .version = x._version +% 1 },
        .children = &.{},
        .device = x.device,
    });
}

/// sigmoid
pub fn sigm(T: type, x: *NDTensor(T)) !void {
    const Tensor = NDTensor(T);

    const BwdClosure = struct {
        pub fn callback(_y: *Tensor, children: *Tensor.Children) !void {
            const _x = children.get_bwd(0) orelse return;
            _y.device.dispatch(opspec.sigm_bwd(T){
                .x_g = try _x.ensure_grad_data(0),
                .y = _y.get_data(),
                .y_g = _y.assume_grad_data(),
            });
        }
    };

    const y = try Tensor.DataType.empty(x.get_shape(), x.device);

    x.device.dispatch(opspec.sigm_fwd(T){
        .x = x.get_data(),
        .y = y.data,
    });

    return Tensor.create_dependent(BwdClosure, x, .{
        .data = y,
        .children = &.{x},
        .device = x.device,
        .callback = .{},
    });
}

// inplace sigmoid
pub fn sigm_(T: type, x: *NDTensor(T)) !void {
    const Tensor = NDTensor(T);

    const BwdClosure = struct {
        version: u8,
        pub fn callback(_x: *Tensor, _: *Tensor.Children, ctx: *@This()) !void {
            std.debug.assert(ctx.version == _x._version);
            _x.device.dispatch(opspec.sigm_inplace_bwd(T){
                .x = _x.get_data(),
                .x_g = try _x.ensure_grad_data(0),
            });
        }
    };

    x.device.dispatch(opspec.sigm_fwd(T){
        .x = x.get_data(),
        .y = x.get_data(),
    });

    try Tensor.prepend_dependent(BwdClosure, x, .{
        .callback = .{ .version = x._version +% 1 },
        .children = &.{},
        .device = x.device,
    });
}
