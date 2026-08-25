//! Train the CIFAR-10 classifier through PJRT and write learned parameters.

const std = @import("std");
const zg = @import("zigrad");
const model = @import("model.zig");
const parameters = @import("parameters.zig");
const dataset_format = @import("dataset.zig");
const Tensor = zg.Tensor;

const Options = struct {
    data_dir: []const u8,
    output_path: []const u8 = "cifar10.safetensors",
    steps: usize = 5_000,
    evaluate: bool = true,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const options = try parse_options(allocator, init.minimal.args);
    defer allocator.free(options.data_dir);
    defer allocator.free(options.output_path);

    const dataset = try load_training_data(init.io, allocator, options.data_dir);
    defer allocator.free(dataset);

    const plugin_path = init.environ_map.get("PJRT_PLUGIN_PATH") orelse {
        std.log.err("set PJRT_PLUGIN_PATH to a PJRT plugin", .{});
        return error.MissingPlugin;
    };
    const pjrt_options = try zg.pjrt.config.from_environ(init.environ_map);
    var client = try zg.pjrt.Client.init(allocator, plugin_path, pjrt_options);
    defer client.deinit();
    const devices = try client.get_devices(allocator);
    defer allocator.free(devices);
    if (devices.len == 0) return error.NoDevices;

    var execution = try zg.pjrt.Execution.init(&client, devices[0], .{});
    var backend = zg.pjrt.Backend.init(&execution, .{});
    var ctx = zg.CompilationCtx{
        .allocator = allocator,
        .io = init.io,
        .device = execution.interface.device,
    };

    var traced = try zg.trace(
        model.train_step,
        allocator,
        .{ model.params_spec, model.training_batch_spec },
        .{ .name = "train_step" },
    );
    defer traced.deinit();
    var pipeline = try zg.pjrt.pipeline.create(allocator, &backend, .{});
    defer pipeline.deinit();
    var loaded = try pipeline.run(zg.Executor.LoadedProgram, &traced.program, &ctx);
    defer loaded.deinit();

    var host_params = try parameters.initialize(allocator);
    defer parameters.deinit(&host_params);
    var host_batch = try allocate_batch(allocator);
    defer deinit_batch(&host_batch);
    fill_batch(&host_batch, dataset, 0);

    var runtime = runtime: {
        var device_params = try parameters.map(host_params, &execution.interface, struct {
            fn upload(executor: *zg.Executor, tensor: Tensor, comptime _: []const u8) !Tensor {
                return try tensor.to_device(executor);
            }
        }.upload);
        errdefer parameters.deinit(&device_params);
        var device_batch = try allocate_batch_on_device(&host_batch, &execution.interface);
        errdefer deinit_batch(&device_batch);

        var inputs = try zg.utils.Tree(Tensor).from(allocator, .{ device_params, device_batch });
        defer inputs.deinit();
        const entry = traced.program.get_function_by_id(try traced.program.resolve_entry()) orelse return error.NoEntry;
        const donated = comptime zg.train.donated_input_indices(
            @TypeOf(.{ model.params_spec, model.training_batch_spec }),
            &.{0},
        );
        const state = try zg.train.TrainState.init(
            allocator,
            loaded,
            inputs.leaves,
            entry.returns.len,
            .{ .donated_input_indices = donated },
        );
        break :runtime .{ .state = state, .batch = device_batch };
    };
    defer runtime.state.deinit(.donatable);
    defer deinit_batch(&runtime.batch);

    for (0..options.steps) |step| {
        if (step != 0) {
            fill_batch(&host_batch, dataset, step * @as(usize, @intCast(model.training_batch_size)));
            const next_batch = try allocate_batch_on_device(&host_batch, &execution.interface);
            runtime.state.set_batch(&.{ next_batch.images, next_batch.labels });
            deinit_batch(&runtime.batch);
            runtime.batch = next_batch;
        }
        var result = try runtime.state.step();
        defer result.loss.deinit();
        if (result.event) |completion| {
            defer execution.interface.release_event(completion);
            try execution.interface.wait(completion);
        }
        const loss_value = try result.loss.item(f32);
        if (step % 50 == 0 or step + 1 == options.steps) {
            std.log.info("step {d}: loss {d:.4}", .{ step, loss_value });
        }
    }

    const device_params = try params_from_buffers(
        allocator,
        &execution.interface,
        runtime.state.input_buffers[0..runtime.state.donatable_count],
    );
    var trained = try download_params(allocator, device_params);
    defer parameters.deinit(&trained);
    const checkpoint = try zg.to_safetensors(model.Params, trained, allocator);
    defer allocator.free(checkpoint);
    {
        var output = try std.Io.Dir.cwd().createFile(init.io, options.output_path, .{ .truncate = true });
        defer output.close(init.io);
        try output.writeStreamingAll(init.io, checkpoint);
    }
    std.log.info("wrote {s}", .{options.output_path});

    if (options.evaluate) {
        try evaluate(
            init.io,
            allocator,
            options.data_dir,
            &backend,
            &ctx,
            device_params,
        );
    }
}

fn parse_options(allocator: std.mem.Allocator, args_iter: std.process.Args) !Options {
    const args = try args_iter.toSlice(allocator);
    defer allocator.free(args);
    var data_dir: ?[]const u8 = null;
    var output_path: []const u8 = "cifar10.safetensors";
    var steps: usize = 5_000;
    var evaluate_model = true;
    for (args[1..]) |arg| {
        if (std.mem.startsWith(u8, arg, "--data=")) {
            data_dir = arg["--data=".len..];
        } else if (std.mem.startsWith(u8, arg, "--output=")) {
            output_path = arg["--output=".len..];
        } else if (std.mem.startsWith(u8, arg, "--steps=")) {
            steps = try std.fmt.parseInt(usize, arg["--steps=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--no-eval")) {
            evaluate_model = false;
        } else {
            return error.InvalidArgument;
        }
    }
    const owned_data_dir = try allocator.dupe(u8, data_dir orelse return error.MissingDataDirectory);
    errdefer allocator.free(owned_data_dir);
    const owned_output_path = try allocator.dupe(u8, output_path);
    return .{
        .data_dir = owned_data_dir,
        .output_path = owned_output_path,
        .steps = steps,
        .evaluate = evaluate_model,
    };
}

fn evaluate(
    io: std.Io,
    allocator: std.mem.Allocator,
    data_dir: []const u8,
    backend: *zg.pjrt.Backend,
    ctx: *zg.CompilationCtx,
    device_params: model.Params,
) !void {
    const path = try std.fmt.allocPrint(allocator, "{s}/test_batch.bin", .{data_dir});
    defer allocator.free(path);
    const file_size = dataset_format.record_size * dataset_format.records_per_file;
    const dataset = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(file_size + 1));
    defer allocator.free(dataset);
    if (dataset.len != file_size) return error.InvalidDataset;
    try dataset_format.validate(dataset);

    var traced = try zg.trace(
        model.forward,
        allocator,
        .{ model.params_spec, model.evaluation_images_spec },
        .{ .name = "inference" },
    );
    defer traced.deinit();
    var pipeline = try zg.pjrt.pipeline.create(allocator, backend, .{});
    defer pipeline.deinit();
    var loaded = try pipeline.run(zg.Executor.LoadedProgram, &traced.program, ctx);
    var callable = traced.bind(loaded) catch |err| {
        loaded.deinit();
        return err;
    };
    defer callable.deinit();

    var host_images = try Tensor.host(.f32, model.evaluation_images_spec.shape.const_slice(), .{ .alloc = allocator });
    defer host_images.deinit();
    var correct: usize = 0;
    const batch_size: usize = @intCast(model.training_batch_size);
    var start: usize = 0;
    while (start < dataset_format.records_per_file) : (start += batch_size) {
        dataset_format.decode_images(host_images.as_slice(f32), dataset, start, batch_size);
        var device_images = try host_images.to_device(callable.loaded_program.executor);
        defer device_images.deinit();
        var eval_inputs = .{ device_params, device_images };
        var logits = try callable.call(&eval_inputs);
        defer logits.deinit();
        var host_logits = try logits.to_host(allocator);
        defer host_logits.deinit();

        const values = host_logits.as_const_slice(f32);
        const count = @min(batch_size, dataset_format.records_per_file - start);
        for (0..count) |batch_index| {
            const prediction = argmax(values[batch_index * @as(usize, @intCast(model.class_count)) ..][0..@intCast(model.class_count)]);
            const expected = dataset[(start + batch_index) * dataset_format.record_size];
            if (prediction == expected) correct += 1;
        }
    }
    const accuracy = @as(f64, @floatFromInt(correct)) / dataset_format.records_per_file;
    std.log.info("test accuracy {d:.2}% ({d}/{d})", .{ accuracy * 100.0, correct, dataset_format.records_per_file });
}

fn argmax(values: []const f32) usize {
    var best: usize = 0;
    for (values[1..], 1..) |value, index| if (value > values[best]) {
        best = index;
    };
    return best;
}

fn load_training_data(io: std.Io, allocator: std.mem.Allocator, directory: []const u8) ![]u8 {
    const file_size = dataset_format.record_size * dataset_format.records_per_file;
    const data = try allocator.alloc(u8, file_size * dataset_format.training_file_count);
    errdefer allocator.free(data);
    for (0..dataset_format.training_file_count) |index| {
        const path = try std.fmt.allocPrint(allocator, "{s}/data_batch_{d}.bin", .{ directory, index + 1 });
        defer allocator.free(path);
        const file = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(file_size + 1));
        defer allocator.free(file);
        if (file.len != file_size) return error.InvalidDataset;
        @memcpy(data[index * file_size ..][0..file_size], file);
    }
    try dataset_format.validate(data);
    return data;
}

fn allocate_batch(allocator: std.mem.Allocator) !model.Batch {
    var images = try Tensor.host(.f32, model.training_batch_spec.images.shape.const_slice(), .{ .alloc = allocator });
    errdefer images.deinit();
    const labels = try Tensor.host(.f32, model.training_batch_spec.labels.shape.const_slice(), .{ .alloc = allocator });
    return .{ .images = images, .labels = labels };
}

fn allocate_batch_on_device(batch: *const model.Batch, executor: *zg.Executor) !model.Batch {
    var images = try batch.images.to_device(executor);
    errdefer images.deinit();
    const labels = try batch.labels.to_device(executor);
    return .{ .images = images, .labels = labels };
}

fn fill_batch(batch: *model.Batch, dataset: []const u8, start_record: usize) void {
    const batch_size: usize = @intCast(model.training_batch_size);
    dataset_format.decode_images(batch.images.as_slice(f32), dataset, start_record, batch_size);
    dataset_format.decode_labels(batch.labels.as_slice(f32), dataset, start_record, batch_size);
}

fn params_from_buffers(
    allocator: std.mem.Allocator,
    executor: *zg.Executor,
    buffers: []const zg.Executor.Buffer,
) !model.Params {
    var specs = try zg.utils.Tree(Tensor).from(allocator, model.params_spec);
    defer specs.deinit();
    if (buffers.len != specs.leaves.len) return error.InvalidOutputArity;

    const leaves = try allocator.alloc(Tensor, buffers.len);
    defer allocator.free(leaves);
    for (specs.leaves, buffers, leaves) |spec, buffer, *tensor| {
        tensor.* = Tensor.from_buffer(executor, buffer, spec.dtype, spec.shape.const_slice());
    }
    return zg.utils.Tree(Tensor).unflatten(model.Params, leaves);
}

fn download_params(allocator: std.mem.Allocator, params: model.Params) !model.Params {
    return try parameters.map(params, allocator, struct {
        fn download(alloc: std.mem.Allocator, tensor: Tensor, comptime _: []const u8) !Tensor {
            return try tensor.to_host(alloc);
        }
    }.download);
}

fn deinit_batch(batch: *model.Batch) void {
    batch.images.deinit();
    batch.labels.deinit();
}
