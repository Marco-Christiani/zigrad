const std = @import("std");
const zg = @import("zigrad");

const Model = zg.Model;
const LinearLayer = zg.layer.LinearLayer;
const ReLULayer = zg.layer.ReLULayer;
const FlattenLayer = zg.layer.FlattenLayer;
const DeviceReference = zg.DeviceReference;

pub fn MnistModel(comptime T: type) type {
    return struct {
        const Self = @This();
        model: Model(T),
        device: DeviceReference,

        pub fn init(device: DeviceReference) !Self {
            var self = Self{
                .device = device,
                .model = try Model(T).init(device),
            };

            var reshape = try FlattenLayer(T).init(device);
            var fc1 = try LinearLayer(T).init(device, 28 * 28, 128);
            var relu1 = try ReLULayer(T).init(device);
            var fc2 = try LinearLayer(T).init(device, 128, 64);
            var relu2 = try ReLULayer(T).init(device);
            var fc3 = try LinearLayer(T).init(device, 64, 10);

            try self.model.add_layer(reshape.as_layer());

            try self.model.add_layer(fc1.as_layer());
            try self.model.add_layer(relu1.as_layer());

            try self.model.add_layer(fc2.as_layer());
            try self.model.add_layer(relu2.as_layer());

            try self.model.add_layer(fc3.as_layer());
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.model.deinit();
            self.* = undefined;
        }

        pub fn countParams(self: Self) usize {
            var total: usize = 0;
            const params = self.model.get_parameters();
            defer self.model.device.allocator.free(params);
            for (params) |p| {
                total += p.grad.?.size();
            }
            return total;
        }
    };
}
