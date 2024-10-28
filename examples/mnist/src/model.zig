const std = @import("std");
const zg = @import("zigrad");

const Model = zg.Model;
const LinearLayer = zg.layer.LinearLayer;
const ReLULayer = zg.layer.ReLULayer;
const FlattenLayer = zg.layer.FlattenLayer;

pub fn MnistModel(comptime T: type) type {
    return struct {
        const Self = @This();
        model: Model(T),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) !Self {
            var self = Self{
                .allocator = allocator,
                .model = try Model(T).init(allocator),
            };

            var reshape = try FlattenLayer(T).init(allocator);
            var fc1 = try LinearLayer(T).init(allocator, 28 * 28, 128);
            var relu1 = try ReLULayer(T).init(allocator);
            var fc2 = try LinearLayer(T).init(allocator, 128, 64);
            var relu2 = try ReLULayer(T).init(allocator);
            var fc3 = try LinearLayer(T).init(allocator, 64, 10);

            try self.model.addLayer(reshape.asLayer());

            try self.model.addLayer(fc1.asLayer());
            try self.model.addLayer(relu1.asLayer());

            try self.model.addLayer(fc2.asLayer());
            try self.model.addLayer(relu2.asLayer());

            try self.model.addLayer(fc3.asLayer());
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.model.deinit();
            self.* = undefined;
        }

        pub fn countParams(self: Self) usize {
            var total: usize = 0;
            const params = self.model.getParameters();
            defer self.model.allocator.free(params);
            for (params) |p| {
                total += p.grad.?.size();
            }
            return total;
        }
    };
}
