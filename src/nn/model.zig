const std = @import("std");
const zg = @import("../root.zig");
const Layer = zg.layer.Layer;
const NDTensor = zg.tensor.NDTensor;
const ops = zg.ops;

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
            for (self.layers.items, 0..) |layer, i| {
                log.debug("deinit layer {d}", .{i});
                layer.deinit();
            }
            log.debug("deinit self.layers", .{});
            self.layers.deinit();
            self.* = undefined;
            log.debug("model deinit done.", .{});
        }

        pub fn release(self: Self) void {
            for (self.layers.items, 0..) |layer, i| {
                log.debug("release layer {d}", .{i});
                layer.release();
            }
        }

        pub fn addLayer(self: *Self, layer: Layer(T)) !void {
            try self.layers.append(layer);
        }

        pub fn forward(self: Self, input: *NDTensor(T), fwd_allocator: std.mem.Allocator) !*NDTensor(T) {
            var output = input;
            for (self.layers.items, 0..) |layer, i| {
                _ = i;
                output = try layer.forward(output, fwd_allocator);
                // log.info("layer-{} {d} output[..n]={d:.4}", .{ i, output.data.shape.shape, output.data.data[0..@min(output.data.data.len, 10)] });
                // output.label = try std.fmt.allocPrint(self.allocator, "layer-{}-{?s}", .{ i + 1, output.label });
                // output.label = output.label orelse try std.fmt.allocPrint(self.allocator, "out-layer-{}", .{i + 1});
                // output.label = output.label orelse blk: {
                //     var buf: [16]u8 = undefined;
                //     const label = try std.fmt.bufPrint(&buf, "layer-{}", .{i});
                //     break :blk label;
                // };
            }
            // log.info("FINAL output={d}", .{output.data.data[0..10]});
            // log.info("FINAL output={d}", .{output.data.data[10..20]});
            return output;
        }

        pub fn getParameters(self: Self) []*const NDTensor(T) {
            var params = std.ArrayList(*const NDTensor(T)).init(self.allocator);
            defer params.deinit();
            for (self.layers.items) |layer| {
                if (layer.getParameters()) |layer_params| {
                    for (layer_params) |p| {
                        params.append(p) catch unreachable;
                    }
                }
            }
            return params.toOwnedSlice() catch unreachable;
        }

        pub fn zeroGrad(self: Self) void {
            for (self.layers.items) |layer| {
                layer.zeroGrad();
            }
        }
    };
}
