const std = @import("std");
const zg = @import("zigrad");
const Optimizer = zg.optim.SGD(T);

const Dataset = @import("dataset.zig").Dataset;
const GCN = @import("model.zig").GCN;
const MaskLayer = @import("model.zig").MaskLayer;

const std_options = .{ .log_level = .info };
const log = std.log.scoped(.gnn);
const T = f32;

pub fn run_cora(data_dir: []const u8) !void {
    var debug_allocator = std.heap.DebugAllocator(.{}).init;
    var cpu = zg.device.HostDevice.init(debug_allocator.allocator());
    defer cpu.deinit();
    const device = cpu.reference();

    var buf1: [1024]u8 = undefined;
    var buf2: [1024]u8 = undefined;
    const edge_path = try std.fmt.bufPrint(&buf1, "{s}/cora/csv/cites.csv", .{data_dir});
    const node_path = try std.fmt.bufPrint(&buf2, "{s}/cora/csv/papers.csv", .{data_dir});
    const dataset = try Dataset(T).load_cora(device, node_path, edge_path);

    var optim: Optimizer = .{
        .lr = 0.01,
        .grad_clip_max_norm = 10.0,
        .grad_clip_delta = 1e-6,
        .grad_clip_enabled = false,
    };

    var gm = zg.GraphManager(zg.NDTensor(T)).init(cpu.allocator, .{
        .eager_teardown = true,
    });
    defer gm.deinit();

    var model = try GCN(T, Optimizer).init(
        device,
        dataset.num_features,
        dataset.num_classes,
        &optim,
    );
    defer model.deinit();

    const label = dataset.y;
    const mask_layer: MaskLayer(T) = .{};

    const num_epoochs = 50;
    for (0..num_epoochs) |epoch| {
        var loss_val: T = 0;
        var acc = [_]f32{ 0, 0, 0 };
        {
            zg.rt_grad_enabled = true;
            const output = try model.forward(dataset.x, dataset.edge_index);

            const output_ = try mask_layer.forward(output, dataset.train_mask);
            const label_ = try mask_layer.forward(label, dataset.train_mask);
            const loss = try zg.loss.softmax_cross_entropy_loss(T, output_, label_);
            loss_val = loss.get(0);

            try gm.backward(loss);
        }

        {
            zg.rt_grad_enabled = false;
            const output = try model.forward(dataset.x, dataset.edge_index);
            defer output.deinit();
            for ([_]*zg.NDTensor(bool){ dataset.train_mask, dataset.eval_mask, dataset.test_mask }, 0..) |mask, i| {
                var correct: f32 = 0;
                const output_ = try mask_layer.forward(output, mask);
                const label_ = try mask_layer.forward(label, mask);
                const total = output_.get_dim(0);

                for (0..total) |j| {
                    const start = j * dataset.num_classes;
                    const end = start + dataset.num_classes;
                    const yh = std.mem.indexOfMax(T, output_.data.data[start..end]);
                    const y = std.mem.indexOfMax(T, label_.data.data[start..end]);
                    correct += if (yh == y) 1 else 0;
                }
                acc[i] = correct / @as(f32, @floatFromInt(total));
            }
        }
        log.info("Epoch: {d:<2} Loss: {d:<5.4} Train_acc: {d:<2.2} Eval_acc: {d:<2.2} Test_acc: {d:<2.2}", .{ epoch + 1, loss_val, acc[0], acc[1], acc[2] });
    }
}

pub fn main() !void {
    const data_dir = std.posix.getenv("ZG_DATA_DIR") orelse "data";
    try run_cora(data_dir);
}
