//! MNIST training example.
//!
//! Demonstrates defining a model, compiling it, and running a training loop.
//!
//! Usage:
//!  `mnist` trains for 100 steps and `mnist --steps=50` selects a custom count.
const std = @import("std");
const zg = @import("zigrad");
const Tensor = zg.Tensor;

const BackendKind = enum {
    pjrt,
    iree,
};

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &.{},
};

// Model definition

/// Model parameters. Each field becomes a device buffer at runtime.
const Params = struct {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    w3: Tensor,
    b3: Tensor,
};

/// A single training batch.
const Batch = struct {
    x: Tensor, // [batch_size, 784]
    y: Tensor, // [batch_size, 10]  (one-hot targets)
};

// Architecture constants.
const batch_size: i64 = 64;
const input_dim: i64 = 784; // 28x28
const hidden1: i64 = 128;
const hidden2: i64 = 64;
const output_dim: i64 = 10;

/// Abstract specs describe shapes and dtypes without holding data.
///
/// TODO(api): Replace the ambiguous `abstract` terminology.
const params_spec: Params = .{
    .w1 = Tensor.abstract(.f32, &.{ input_dim, hidden1 }),
    .b1 = Tensor.abstract(.f32, &.{hidden1}),
    .w2 = Tensor.abstract(.f32, &.{ hidden1, hidden2 }),
    .b2 = Tensor.abstract(.f32, &.{hidden2}),
    .w3 = Tensor.abstract(.f32, &.{ hidden2, output_dim }),
    .b3 = Tensor.abstract(.f32, &.{output_dim}),
};
const batch_spec: Batch = .{
    .x = Tensor.abstract(.f32, &.{ batch_size, input_dim }),
    .y = Tensor.abstract(.f32, &.{ batch_size, output_dim }),
};

/// Forward pass: input -> 3 linear layers -> MSE loss.
fn loss(params: Params, batch: Batch) !Tensor {
    // Tracing records symbolic operations, so tensor values are unavailable here.
    //
    // TODO(debug): Define effect ordering for runtime debug operations.

    // TODO(api): Add linear, MMA, and BMMA operations with backend-neutral
    //  lowering instead of exposing contraction details at this level.
    const z1 = try batch.x.matmul(params.w1);
    const a1 = try z1.add(try params.b1.broadcast_in_dim(&.{ batch_size, hidden1 }, &.{1}));

    // Layer 2
    const z2 = try a1.matmul(params.w2);
    const a2 = try z2.add(try params.b2.broadcast_in_dim(&.{ batch_size, hidden2 }, &.{1}));

    // Layer 3 (output)
    const z3 = try a2.matmul(params.w3);
    const preds = try z3.add(try params.b3.broadcast_in_dim(&.{ batch_size, output_dim }, &.{1}));

    // TODO(api): Add an MSE operation.
    const diff = try preds.sub(batch.y);
    const sq = try diff.mul(diff);

    // TODO(api): Give full-tensor reduction a distinct method name.
    return try sq.reduce_sum(&.{ 0, 1 });
}

/// Full training step: forward + backward + SGD update.
///
/// This is what gets compiled into a single fused program.
/// `value_and_grad` traces the backward pass automatically.
fn train_step(params: Params, batch: Batch) !struct { loss_val: Tensor, updated: Params } {
    var vg = try zg.transforms.value_and_grad(loss, .{ params, batch });
    defer vg.deinit();

    const optim = zg.optim.SGD{ .lr = 1e-2 };

    // TODO(memory): Derive safe buffer reuse from PR def-use information and
    //  keep explicit donation as an opt-in override.
    var params_tree = try zg.utils.Tree(Tensor).from(vg.grads.allocator, params);
    defer params_tree.deinit();
    var updated = try params_tree.map2(Tensor, &vg.grads, Tensor, optim, zg.optim.SGD.update);
    defer updated.deinit();

    return .{
        .loss_val = vg.value,
        .updated = try updated.extract(Params),
    };
}

// Main

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var steps: usize = 100;
    var backend_kind: BackendKind = .pjrt;
    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);
    for (args[1..]) |arg| {
        if (std.mem.startsWith(u8, arg, "--steps=")) {
            steps = try std.fmt.parseInt(usize, arg["--steps=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--backend=")) {
            backend_kind = std.meta.stringToEnum(
                BackendKind,
                arg["--backend=".len..],
            ) orelse return error.InvalidBackend;
        }
    }

    return switch (backend_kind) {
        .pjrt => run_pjrt(io, allocator, init.environ_map, steps),
        .iree => run_iree(io, allocator, init.environ_map, steps),
    };
}

fn run_pjrt(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    steps: usize,
) !void {
    const plugin_path = environ.get("PJRT_PLUGIN_PATH") orelse {
        std.log.err("set PJRT_PLUGIN_PATH to a PJRT plugin (.so)", .{});
        return error.MissingPlugin;
    };
    const pjrt_options = try zg.pjrt.config.from_environ(environ);
    var pjrt_client = try zg.pjrt.Client.init(allocator, plugin_path, pjrt_options);
    defer pjrt_client.deinit();
    const devs = try pjrt_client.get_devices(allocator);
    defer allocator.free(devs);
    if (devs.len == 0) return error.NoDevices;
    var execution = try zg.pjrt.Execution.init(&pjrt_client, devs[0], .{});
    var compilation_context = zg.compilation.Context{
        .allocator = allocator,
        .io = io,
        .device = execution.interface.device,
    };
    var backend = zg.pjrt.Backend.init(&execution, .{});
    var pipeline = try zg.pjrt.pipeline.create(allocator, &backend, .{
        .stablehlo = .{ .entry_name = "train_step" },
    });
    defer pipeline.deinit();

    return try run_training(allocator, steps, &pipeline, &compilation_context);
}

fn run_iree(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    steps: usize,
) !void {
    const config = zg.iree.Config.from_environ(environ);
    var runtime = try zg.iree.Runtime.init(allocator, config.runtime);
    defer runtime.deinit();
    var execution = zg.iree.Execution.init(allocator, &runtime, config.runtime);
    var backend = zg.iree.Backend.init(&execution, config.compiler, "module.main");
    var compilation_context = zg.compilation.Context{
        .allocator = allocator,
        .io = io,
        .device = execution.interface.device,
    };
    var pipeline = try zg.iree.pipeline.create(allocator, .{ .loaded = &backend }, .{
        .stablehlo = .{ .entry_name = "train_step" },
    });
    defer pipeline.deinit();

    return try run_training(allocator, steps, &pipeline, &compilation_context);
}

fn run_training(
    allocator: std.mem.Allocator,
    steps: usize,
    pipeline: *zg.compilation.Pipeline,
    compilation_context: *zg.compilation.Context,
) !void {
    std.log.info("compiling train_step...", .{});
    var traced = try zg.trace_callable(
        train_step,
        allocator,
        .{ params_spec, batch_spec },
        .{ .entry_name = "train_step", .donate = &.{0} },
    );
    defer traced.deinit();

    var loaded_program = try pipeline.run(
        zg.Executor.LoadedProgram,
        &traced.program,
        compilation_context,
    );
    const executor = loaded_program.executor;
    var step_fn = traced.bind(loaded_program) catch |err| {
        loaded_program.deinit();
        return err;
    };
    defer step_fn.deinit();

    std.log.info("generating synthetic data...", .{});
    var spec_tree = try zg.utils.Tree(Tensor).from(allocator, .{ params_spec, batch_spec });
    defer spec_tree.deinit();

    // TODO: Need to add tree transfer and other tensor op conveniences without crossing the api boundary, this is verbose.
    var host_tensors = try spec_tree.map(Tensor, allocator, struct {
        fn f(alloc: std.mem.Allocator, spec: Tensor) !Tensor {
            return try Tensor.host(spec.dtype, spec.shape.const_slice(), .{ .alloc = alloc });
        }
    }.f);
    defer host_tensors.deinit_with(Tensor.deinit);

    // TODO(example): Add a real dataset input path.
    for (host_tensors.leaves) |t| fill_pattern(t.as_slice(f32));

    var dev_tensors = try host_tensors.map(Tensor, executor, struct {
        fn f(selected_executor: *zg.Executor, t: Tensor) !Tensor {
            return try t.to_device(selected_executor);
        }
    }.f);
    // Tensor buffer ownership transfers to `inputs`, so only tree arrays are freed here.
    defer dev_tensors.deinit();

    // Recover structured input from the flat device tensor tree.
    // InputType is derived from the compiled function.
    var inputs = try dev_tensors.extract(@TypeOf(step_fn).InputType);

    std.log.info("training for {} steps...", .{steps});
    for (0..steps) |step| {
        // call() takes a pointer to inputs. Donated args (params at index 0)
        //  are updated in place by swapping their buffers automatically.
        //  Only non-donated outputs (loss) are returned.
        var result = try step_fn.call(&inputs);

        const loss_val = try result.loss_val.item(f32);

        if (step % 10 == 0 or step == steps - 1) {
            std.log.info("step {d:>4}: loss = {d:.4}", .{ step, loss_val });
        }

        result.loss_val.deinit();
    }

    // Free all input buffers (final params from last step + batch data).
    step_fn.deinit_inputs(&inputs);

    std.log.info("done", .{});
}

/// Fill a buffer with a simple deterministic pattern.
fn fill_pattern(buf: []f32) void {
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 17)) * 1e-3 - 8e-3;
    }
}
