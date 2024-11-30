const std = @import("std");
const zg = @import("zigrad");
const NDTensor = zg.NDTensor;
const Layer = zg.layer.Layer;
const LinearLayer = zg.layer.LinearLayer;
const ReLULayer = zg.layer.ReLULayer;
const Model = zg.Model;

pub fn DQNModel(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        model: Model(T),
        params: []*const NDTensor(T),

        pub fn init(allocator: std.mem.Allocator, input_size: usize, hidden_size: usize, output_size: usize) !*Self {
            var model = try Model(T).init(allocator);

            try model.addLayer((try LinearLayer(T).init(allocator, input_size, hidden_size)).asLayer());
            try model.addLayer((try ReLULayer(T).init(allocator)).asLayer());
            try model.addLayer((try LinearLayer(T).init(allocator, hidden_size, hidden_size)).asLayer());
            try model.addLayer((try ReLULayer(T).init(allocator)).asLayer());
            try model.addLayer((try LinearLayer(T).init(allocator, hidden_size, output_size)).asLayer());
            const self = try allocator.create(Self);
            self.* = .{
                .allocator = allocator,
                .model = model,
                .params = model.getParameters(),
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.model.deinit();
            self.allocator.destroy(self);
        }

        pub fn forward(self: *Self, input: *NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
            return try self.model.forward(input, allocator);
        }

        pub fn getParameters(self: *Self) []*const NDTensor(T) {
            return self.model.getParameters();
        }

        pub fn train(self: Self) void {
            for (self.model.layers.items) |layer| layer.enableGrad();
        }

        pub fn eval(self: Self) void {
            for (self.model.layers.items) |layer| layer.disableGrad();
        }

        pub fn zeroGrad(self: Self) void {
            for (self.model.layers.items) |layer| layer.zeroGrad();
        }
    };
}
