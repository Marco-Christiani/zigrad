//! MNIST training example.
//!
//! Demonstrates defining a model, compiling it, and running a training loop.
//!
//! Usage:
//!   mnist              Train for 100 steps (default)
//!   mnist --steps=50   Custom step count
const std = @import("std");
const zg = @import("zigrad");
const Tensor = zg.Tensor;

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &.{},
};

// ============================================================================
// Model definition
// ============================================================================

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

/// Abstract specs for tracing. These describe shapes and dtypes without
///  holding any data, Zigrad uses them to trace the computation graph.
/// Marking parameters as `donatable` lets the backend reuse their memory
///  for updated values (important for training loops). Later comments
///  explain the dontation concept further.
/// TODO: I still dont like the name abstract, its ambiguous.
const donatable: Tensor.AbstractOpts = .{ .donatable = true };
const params_spec: Params = .{
    .w1 = Tensor.abstract(.f32, &.{ input_dim, hidden1 }, donatable),
    .b1 = Tensor.abstract(.f32, &.{hidden1}, donatable),
    .w2 = Tensor.abstract(.f32, &.{ hidden1, hidden2 }, donatable),
    .b2 = Tensor.abstract(.f32, &.{hidden2}, donatable),
    .w3 = Tensor.abstract(.f32, &.{ hidden2, output_dim }, donatable),
    .b3 = Tensor.abstract(.f32, &.{output_dim}, donatable),
};
const batch_spec: Batch = .{
    .x = Tensor.abstract(.f32, &.{ batch_size, input_dim }, .{}),
    .y = Tensor.abstract(.f32, &.{ batch_size, output_dim }, .{}),
};

/// Forward pass: input -> 3 linear layers -> MSE loss.
fn loss(params: Params, batch: Batch) !Tensor {
    // No computation is actually done here, its symbolic, similar to other frameworks.
    // That means the lines of code here execute during tracing, not during runtime
    //  execution in training. You could not, for example, print the contents of these
    //  tensors. However, Zigrad will rightfully error you attempt something illegal
    //  so there wont be any surprises.
    // TODO: still need to implement debug ops for users, likely need token threading
    //  in the compiler.

    // Layer 1: linear (no activation, keeping it simple for now)
    // TODO: this is too simple
    // TODO: should add a linear method to tensor, and we still lack mma and bmma as conveniences
    //  using dot_general is just way too low level for this user facing api and not all backends
    //  can necessarily support this, we should be lowering to dot_general in a pass.
    const z1 = try batch.x.matmul(params.w1);
    const a1 = try z1.add(try params.b1.broadcast_in_dim(&.{ batch_size, hidden1 }, &.{1}));

    // Layer 2
    const z2 = try a1.matmul(params.w2);
    const a2 = try z2.add(try params.b2.broadcast_in_dim(&.{ batch_size, hidden2 }, &.{1}));

    // Layer 3 (output)
    const z3 = try a2.matmul(params.w3);
    const preds = try z3.add(try params.b3.broadcast_in_dim(&.{ batch_size, output_dim }, &.{1}));

    // MSE loss
    // TODO: should add an mse method to tensor
    const diff = try preds.sub(batch.y);
    const sq = try diff.mul(diff);

    // We want to sum all values, so we go from a 2D tensor to a scalar, summing dims 0 and 1.
    return sq.reduce_sum(&.{ 0, 1 }); // TODO: this isnt the right method name.
}

/// Full training step: forward + backward + SGD update.
///
/// This is what gets compiled into a single fused program.
/// `value_and_grad` traces the backward pass automatically.
fn train_step(params: Params, batch: Batch) !struct { loss_val: Tensor, updated: Params } {
    var vg = try zg.frontend.transforms.value_and_grad(loss, .{ params, batch });
    defer vg.deinit();

    const optim = zg.frontend.optim.SGD{ .lr = 1e-2 };

    // This updates the model, defined by the leaves of the tensor tree.
    // Since we are creating a single graph and care about performance, this is why
    //  we marked params as donatable so we can get in-place updates.
    // TODO: with the newer def-use features we still need to revisit automated liveness analysis
    //  dontable is likely better regarded as a lower level detail progressively disclosed
    //  following our opt-in-to-control philosophy.
    var params_tree = try zg.utils.Tree(Tensor).from(vg.grads.allocator, params);
    defer params_tree.deinit();
    var updated = try params_tree.map2(Tensor, &vg.grads, Tensor, optim, zg.frontend.optim.SGD.update);
    defer updated.deinit();

    return .{
        .loss_val = vg.value,
        .updated = updated.extract(Params),
    };
}

// ============================================================================
// Main
// ============================================================================

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    // --- Parse args ---
    var steps: usize = 100;
    var args = std.process.args();
    _ = args.next(); // skip program name
    while (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--steps=")) {
            steps = try std.fmt.parseInt(usize, arg["--steps=".len..], 10);
        }
    }

    // --- Backend ---
    const plugin_path = std.posix.getenv("PJRT_PLUGIN_PATH") orelse {
        std.log.err("set PJRT_PLUGIN_PATH to a PJRT plugin (.so)", .{});
        return error.MissingPlugin;
    };
    var pjrt_backend = try zg.backend.pjrt.Backend.init(allocator, plugin_path);
    defer pjrt_backend.deinit();
    const backend = &pjrt_backend.interface;
    const devs = try backend.get_devices(allocator);
    defer allocator.free(devs);
    if (devs.len == 0) return error.NoDevices;
    const device = devs[0];

    // --- Compile ---
    std.log.info("compiling train_step...", .{});
    // NOTE: specs must be runtime values, not comptime. Using `var` forces
    //  runtime evaluation. Without this, Tree.extract hits "cannot store
    //  runtime value in compile time variable" because Zig infers comptime
    //  struct fields from module-level `const` declarations.
    // TODO: fix in tree.zig unflatten_values to handle comptime-typed structs.
    var ps = params_spec;
    var bs = batch_spec;
    _ = .{ &ps, &bs };
    const inputs_spec = .{ ps, bs };
    var compiled = try zg.frontend.compile(
        train_step,
        allocator,
        backend,
        device,
        inputs_spec,
        .{ .entry_name = "train_step" },
    );
    // TODO: are we revisiting the idea that exe's hold references to backend so
    //  they deinit themselves? this isnt intuitive at all in the user layer.
    defer backend.deinit_executable(compiled.exe);
    defer compiled.deinit();

    // --- Synthetic data ---
    std.log.info("generating synthetic data...", .{});
    // Now create some real buffers that can hold data for our model.
    var spec_tree = try zg.utils.Tree(Tensor).from(allocator, inputs_spec);
    defer spec_tree.deinit();

    // This applies our function to every leaf of the tree.
    // So, for every parameter (defined above) we create a host buffer.
    var host_bufs = try spec_tree.map(zg.HostBuffer, allocator, struct {
        fn f(alloc: std.mem.Allocator, spec: Tensor) anyerror!zg.HostBuffer {
            return zg.HostBuffer.init(alloc, spec.shape, spec.dtype);
        }
    }.f);
    // The tree will free the leaves for us, we just tell it what function to use.
    defer host_bufs.deinit_with(zg.HostBuffer.deinit);

    // Fill params with small random-ish values
    for (host_bufs.leaves[0..6]) |*buf| fill_pattern(buf.as_slice(f32));
    // TODO: using synthetic values for now, will need to migrate to real data.
    // Fill batch input
    fill_pattern(host_bufs.leaves[6].as_slice(f32));
    // Fill targets
    fill_pattern(host_bufs.leaves[7].as_slice(f32));

    // --- Upload to device ---
    var dev_bufs = try backend.transfer(device, &host_bufs, .to_device);
    defer dev_bufs.deinit();

    // --- Training loop ---
    std.log.info("training for {} steps...", .{steps});
    const train = zg.frontend.train;
    var state = try train.TrainState.init_from_model(
        allocator,
        &compiled,
        backend,
        dev_bufs.leaves,
    );
    defer state.deinit(.all);

    var loss_host = try zg.HostBuffer.init(allocator, .{}, .f32);
    defer loss_host.deinit();

    for (0..steps) |step| {
        const result = try state.step();

        if (result.event) |ev| {
            try backend.await_event(ev);
            backend.deinit_event(ev);
        }

        // Read loss back to host
        // By default, we prefer async operations to create a "pit of success"
        //  style API. This is because it naturally encourages you to pipeline,
        //  which results in better performance. While the device (assuming its
        //  not CPU) is busy, the CPU can do other work like load the next batch.
        // TODO: we never finished the sync/async variants
        // TODO: same comments as the exe situation, this is pretty bad.
        if (try backend.buffer_to_host(result.loss_buf, loss_host.data_mut())) |ev| {
            try backend.await_event(ev);
            backend.deinit_event(ev);
        }

        // TODO: missing a .item() method
        const loss_val = loss_host.as_slice(f32)[0];

        if (step % 10 == 0 or step == steps - 1) {
            std.log.info("step {d:>4}: loss = {d:.4}", .{ step, loss_val });
        }
        // TODO: same comments as the exe situation
        backend.deinit_buffer(result.loss_buf);
    }

    std.log.info("done", .{});
}

/// Fill a buffer with a simple deterministic pattern.
fn fill_pattern(buf: []f32) void {
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 17)) * 1e-3 - 8e-3;
    }
}
