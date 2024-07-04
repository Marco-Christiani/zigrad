const std = @import("std");
const Layer = @import("layer.zig").Layer;
const NDTensor = @import("tensor.zig").NDTensor;

const log = std.log.scoped(.zigrad_model);

pub fn Model(comptime T: type) type {
    return struct {
        const Self = @This();
        layers: std.ArrayList(Layer(T)),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) !*Self {
            const self = try allocator.create(Self);
            self.* = .{
                .layers = std.ArrayList(Layer(T)).init(allocator),
                .allocator = allocator,
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            for (self.layers.items, 0..) |layer, i| {
                log.info("deinit layer {d}", .{i});
                layer.deinit();
            }
            log.debug("deinit self.layers", .{});
            self.layers.deinit();
            log.debug("destroy self", .{});
            self.allocator.destroy(self);
            log.debug("model deinit done.", .{});
        }

        pub fn release(self: *Self) void {
            for (self.layers.items, 0..) |layer, i| {
                log.debug("release layer {d}", .{i});
                layer.release();
            }
        }

        pub fn addLayer(self: *Self, layer: Layer(T)) !void {
            try self.layers.append(layer);
        }

        pub fn forward(self: *Self, input: *NDTensor(T)) !*NDTensor(T) {
            var output = input;
            for (self.layers.items, 0..) |layer, i| {
                log.info("Layer {}: Input shape: {any}", .{ i + 1, output.data.shape.shape });
                output = try layer.forward(output, self.allocator);
                // output.label = output.label orelse try std.fmt.allocPrint(self.allocator, "out-layer-{}", .{i + 1});
                // output.label = output.label orelse blk: {
                //     var buf: [16]u8 = undefined;
                //     const label = try std.fmt.bufPrint(&buf, "layer-{}", .{i});
                //     break :blk label;
                // };
            }
            return output;
        }

        pub fn getParameters(self: *Self) []*NDTensor(T) {
            var params = std.ArrayList(*NDTensor(T)).init(self.allocator);
            defer params.deinit();
            for (self.layers.items) |layer| {
                if (layer.getParameters()) |layer_params| {
                    for (layer_params) |p| {
                        params.append(@constCast(p)) catch unreachable;
                    }
                }
            }
            return params.toOwnedSlice() catch unreachable;
        }

        pub fn zeroGrad(self: *Self) void {
            for (self.layers.items) |layer| {
                layer.zeroGrad();
            }
        }
    };
}
