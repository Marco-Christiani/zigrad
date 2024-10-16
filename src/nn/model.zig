const std = @import("std");
const zg = @import("../zigrad.zig");
const Layer = zg.layer.Layer;
const NDTensor = zg.NDTensor;

const log = std.log.scoped(.zg_model);

pub fn Model(comptime T: type) type {
    return struct {
        const Self = @This();
        layers: std.ArrayList(Layer(T)),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) !Self {
            return Self{
                .layers = std.ArrayList(Layer(T)).init(allocator),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.layers.items) |layer| layer.deinit();
            self.layers.deinit();
            self.* = undefined;
        }

        pub fn release(self: Self) void {
            for (self.layers.items) |layer| layer.release();
        }

        pub fn addLayer(self: *Self, layer: Layer(T)) !void {
            try self.layers.append(layer);
        }

        pub fn forward(self: Self, input: *NDTensor(T), fwd_allocator: std.mem.Allocator) !*NDTensor(T) {
            var output = input;
            for (self.layers.items) |layer| output = try layer.forward(output, fwd_allocator);
            return output;
        }

        /// COM.
        pub fn getParameters(self: Self) []*const NDTensor(T) {
            var params = std.ArrayList(*const NDTensor(T)).init(self.allocator);
            defer params.deinit();
            for (self.layers.items) |layer| {
                if (layer.getParameters()) |layer_params| {
                    for (layer_params) |p| params.append(p) catch unreachable;
                }
            }
            return params.toOwnedSlice() catch unreachable;
        }

        pub fn zeroGrad(self: Self) void {
            for (self.layers.items) |layer| layer.zeroGrad();
        }
    };
}
