const zg = @import("../root.zig");
const std = @import("std");
const json = std.json;
const NDTensor = zg.NDTensor;

pub fn ModelData(comptime T: type) type {
    return struct {
        const LayerDataT = zg.layer.Layer(T).LayerData;
        layers: []LayerDataT,
    };
}

pub fn serializeModel(comptime T: type, model: anytype, allocator: std.mem.Allocator) ![]u8 {
    var model_data = ModelData(T){
        .layers = try allocator.alloc(ModelData(T).LayerDataT, model.layers.items.len),
    };
    defer allocator.free(model_data.layers);

    for (model.layers.items, 0..) |layer, i| {
        model_data.layers[i] = try layer.serialize();
    }

    return try json.stringifyAlloc(allocator, model_data, .{});
}

pub fn deserializeModel(comptime T: type, json_str: []const u8, allocator: std.mem.Allocator) !zg.Model(T) {
    const model_data = try json.parseFromSliceLeaky(ModelData(T), allocator, json_str, .{});

    var model = try zg.Model(T).init(allocator);
    errdefer model.deinit();

    for (model_data.layers) |layer_data| {
        switch (layer_data) {
            .linear => {
                var linear_layer = try zg.layer.LinearLayer(T).deserialize(layer_data, allocator);
                errdefer linear_layer.deinit();
                try model.addLayer(linear_layer.asLayer());
            },
            .relu => try model.addLayer((try zg.layer.ReLULayer(T).init(allocator)).asLayer()),
            // other layers...
        }
    }

    return model;
}

pub fn saveModelToFile(comptime T: type, model: anytype, filepath: []const u8, allocator: std.mem.Allocator) !void {
    const json_str = try serializeModel(T, model, allocator);
    defer allocator.free(json_str);

    try std.fs.cwd().writeFile(filepath, json_str);
}

pub fn loadModelFromFile(comptime T: type, filepath: []const u8, allocator: std.mem.Allocator) !zg.Model(T) {
    const file_content = try std.fs.cwd().readFileAlloc(allocator, filepath, std.math.maxInt(usize));
    defer allocator.free(file_content);

    return try deserializeModel(T, file_content, allocator);
}

test "save and load" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;
    var model = try zg.Model(T).init(alloc);
    const fc1 = try zg.layer.LinearLayer(T).init(alloc, 2, 4);
    const fc2 = try zg.layer.LinearLayer(T).init(alloc, 4, 1);
    try model.addLayer(fc1.asLayer());
    try model.addLayer((try zg.layer.ReLULayer(T).init(alloc)).asLayer());
    try model.addLayer(fc2.asLayer());
    std.debug.print("Saving model...\n", .{});
    try saveModelToFile(T, model, "model_weights.json", alloc);
    var loaded_model = try loadModelFromFile(T, "model_weights.json", alloc);
    defer loaded_model.deinit();
    const input = try zg.NDTensor(T).init(&.{ 1.2, -0.4 }, &.{ 1, 2 }, true, alloc);
    const output = try loaded_model.forward(input, alloc);
    output.print();
    //
}

