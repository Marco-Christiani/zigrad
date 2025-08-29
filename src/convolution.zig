const std = @import("std");
const zg = @import("zigrad.zig");
const settings = zg.settings;
const DeviceReference = zg.DeviceReference;
const backend = zg.backend;
const opspec = zg.opspec;
const utils = @import("ndtensor/utils.zig");

const Graph = zg.Graph;
const Node = Graph.Node;
const DeviceData = zg.device.DeviceData;

pub const Layout = enum { NCHW, NHWC };

pub fn Pointwise(scalar_type: type, comptime config: struct {
    layout: Layout,
    channels: usize,
    kernels: usize,
    stride: usize = 1,
}) type {
    if (config.channels == 0)
        @compileError("Channels must be greater than zero.");

    if (config.kernels == 0)
        @compileError("Kernels must be greater than zero.");

    if (config.stride == 0)
        @compileError("Stride must be greater than zero.");

    if (config.stride > 1)
        @compileError("TODO: Implement strided pointwise convolution");

    return struct {
        const Self = @This();
        const Tensor = zg.NDTensor(scalar_type);

        filter: *Tensor,

        pub fn init(device: DeviceReference, rand: zg.RandType, opts: zg.TensorOpts) !Self {
            return .{ .filter = try Tensor.random(device, &.{
                if (config.layout == .NCHW) config.kernels else config.channels,
                if (config.layout == .NCHW) config.channels else config.kernels,
            }, rand, opts) };
        }

        pub const apply = if (config.stride == 1)
            apply_1_stride
        else
            unreachable; // todo

        fn apply_1_stride(self: *Self, A: *Tensor) !*Tensor {
            std.debug.assert(3 <= A.get_ndims() and A.get_ndims() <= 4);

            // Prepare Q for batch matrics multiply by keeping N as the batch
            // dimension and compressing (H,W) to form a 3D tensor
            const Q = try if (A.get_ndims() == 3) A.unsqueeze(.view) else A.view();
            errdefer Q.deinit();

            const N, const C, const H, const W = if (config.layout == .NCHW)
                .{ Q.get_dim(0), Q.get_dim(1), Q.get_dim(2), Q.get_dim(3) }
            else
                .{ Q.get_dim(0), Q.get_dim(3), Q.get_dim(1), Q.get_dim(2) };

            switch (config.layout) {
                .NCHW => Q._reshape(&.{ N, C, H * W }),
                .NHWC => Q._reshape(&.{ N, H * C, W }),
            }

            defer Q.soft_deinit();

            // Apply filters to reduce channel dimension, then
            // reshape to break apart (H * W)
            const out = try switch (config.layout) {
                .NCHW => self.filter.bmm(Q, .{}),
                .NHWC => Q.bmm(self.filter, .{}),
            };

            const ReshapeBwd = struct {
                dims: [3]usize,
                pub fn backward(c: *Tensor, _: *Node.Children, ctx: *@This()) !void {
                    c._reshape(ctx.dims[0..]);
                }
            };

            try out.node.callbacks.prepend(Tensor, ReshapeBwd, .{
                .dims = out.get_shape()[0..3].*,
            }, &.{});

            switch (config.layout) {
                .NCHW => out._reshape(&.{ N, config.kernels, H, W }),
                .NHWC => out._reshape(&.{ N, H, W, config.kernels }),
            }

            out._squeeze();

            return out;
        }
    };
}
