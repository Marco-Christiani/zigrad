const zg = @import("zigrad");
const Tensor = zg.Tensor;

pub const image_size: i64 = 32;
pub const input_channels: i64 = 3;
pub const class_count: i64 = 10;
pub const training_batch_size: i64 = 64;
pub const inference_batch_size: i64 = 1;

/// Trainable tensors shared by the training and inference programs.
pub const Params = struct {
    stem_kernel: Tensor,
    stem_bias: Tensor,
    block1_kernel1: Tensor,
    block1_bias1: Tensor,
    block1_kernel2: Tensor,
    block1_bias2: Tensor,
    down_kernel: Tensor,
    down_bias: Tensor,
    block2_kernel: Tensor,
    block2_bias: Tensor,
    classifier_kernel: Tensor,
    classifier_bias: Tensor,
};

/// One training batch of normalized images and one-hot labels.
pub const Batch = struct {
    images: Tensor,
    labels: Tensor,
};

/// Scalar loss and parameters after one optimizer step.
pub const TrainOutput = struct {
    loss_value: Tensor,
    updated: Params,
};

/// Abstract parameter shapes accepted by the model.
pub const params_spec: Params = .{
    .stem_kernel = Tensor.abstract(.f32, &.{ 3, 3, 3, 16 }),
    .stem_bias = Tensor.abstract(.f32, &.{16}),
    .block1_kernel1 = Tensor.abstract(.f32, &.{ 3, 3, 16, 16 }),
    .block1_bias1 = Tensor.abstract(.f32, &.{16}),
    .block1_kernel2 = Tensor.abstract(.f32, &.{ 3, 3, 16, 16 }),
    .block1_bias2 = Tensor.abstract(.f32, &.{16}),
    .down_kernel = Tensor.abstract(.f32, &.{ 3, 3, 16, 32 }),
    .down_bias = Tensor.abstract(.f32, &.{32}),
    .block2_kernel = Tensor.abstract(.f32, &.{ 3, 3, 32, 32 }),
    .block2_bias = Tensor.abstract(.f32, &.{32}),
    .classifier_kernel = Tensor.abstract(.f32, &.{ 32, class_count }),
    .classifier_bias = Tensor.abstract(.f32, &.{class_count}),
};

/// Abstract input shapes used by a compiled training step.
pub const training_batch_spec: Batch = .{
    .images = image_spec(training_batch_size),
    .labels = Tensor.abstract(.f32, &.{ training_batch_size, class_count }),
};

/// Fixed batch-one input used for packaged deployment.
pub const inference_images_spec = image_spec(inference_batch_size);
/// Batched input used for training-time evaluation.
pub const evaluation_images_spec = image_spec(training_batch_size);

/// Produce class logits for NHWC images with values normalized to `[0, 1]`.
pub fn forward(params: Params, images: Tensor) !Tensor {
    const batch_size = images.shape.const_slice()[0];

    var x = try conv_bias(images, params.stem_kernel, params.stem_bias, batch_size, 16, 1);
    x = try x.relu();

    var residual = try conv_bias(x, params.block1_kernel1, params.block1_bias1, batch_size, 16, 1);
    residual = try residual.relu();
    residual = try conv_bias(residual, params.block1_kernel2, params.block1_bias2, batch_size, 16, 1);
    x = try (try x.add(residual)).relu();

    x = try conv_bias(x, params.down_kernel, params.down_bias, batch_size, 32, 2);
    x = try x.relu();
    residual = try conv_bias(x, params.block2_kernel, params.block2_bias, batch_size, 32, 1);
    x = try (try x.add(residual)).relu();

    const pooled_sum = try x.reduce(.{ .axes = &.{ 1, 2 }, .operation = .sum });
    const pooled = try pooled_sum.mul(try Tensor.constant_like(pooled_sum, 1.0 / (16.0 * 16.0)));
    const logits = try pooled.mm(params.classifier_kernel);
    return try logits.add(try params.classifier_bias.broadcast_in_dim(
        &.{ batch_size, class_count },
        &.{1},
    ));
}

/// Mean categorical cross-entropy for one-hot labels.
pub fn loss(params: Params, batch: Batch) !Tensor {
    const logits = try forward(params, batch.images);
    const max_logits = try logits.reduce(.{ .axes = &.{1}, .operation = .maximum });
    const centered = try logits.sub(try max_logits.broadcast_in_dim(
        &.{ training_batch_size, class_count },
        &.{0},
    ));
    const exp_logits = try centered.exp();
    const exp_sum = try exp_logits.reduce(.{ .axes = &.{1}, .operation = .sum });
    const log_normalizer = try exp_sum.log();
    const log_probs = try centered.sub(try log_normalizer.broadcast_in_dim(
        &.{ training_batch_size, class_count },
        &.{0},
    ));
    const selected = try batch.labels.mul(log_probs);
    const total = try selected.reduce(.{ .axes = &.{ 0, 1 }, .operation = .sum });
    return try total.mul(try Tensor.constant_like(total, -1.0 / @as(f32, @floatFromInt(training_batch_size))));
}

/// Compute one differentiated loss and apply an SGD update.
pub fn train_step(params: Params, batch: Batch) !TrainOutput {
    var value_and_grad = try zg.transforms.value_and_grad(loss, .{ params, batch }, .{});
    defer value_and_grad.deinit();

    var params_tree = try zg.utils.Tree(Tensor).from(value_and_grad.grads.allocator, params);
    defer params_tree.deinit();
    const optimizer = zg.optim.SGD{ .lr = 0.01 };
    var updated = try params_tree.map2(
        Tensor,
        &value_and_grad.grads,
        Tensor,
        optimizer,
        zg.optim.SGD.update,
    );
    defer updated.deinit();
    return .{
        .loss_value = value_and_grad.outputs,
        .updated = try updated.extract(Params),
    };
}

fn image_spec(batch_size: i64) Tensor {
    return Tensor.abstract(.f32, &.{ batch_size, image_size, image_size, input_channels });
}

fn conv_bias(
    input: Tensor,
    kernel: Tensor,
    bias: Tensor,
    batch_size: i64,
    output_channels: i64,
    comptime stride: i64,
) !Tensor {
    const output_size = if (stride == 1) image_extent(input) else @divFloor(image_extent(input) + 1, 2);
    const convolved = try input.convolution(kernel, convolution_params(stride));
    return try convolved.add(try bias.broadcast_in_dim(
        &.{ batch_size, output_size, output_size, output_channels },
        &.{3},
    ));
}

fn image_extent(input: Tensor) i64 {
    return input.shape.const_slice()[1];
}

fn convolution_params(comptime stride: i64) zg.pr.ConvolutionParams {
    return .{
        .window_strides = &.{ stride, stride },
        .padding = &.{ 1, 1, 1, 1 },
        .lhs_dilation = &.{ 1, 1 },
        .rhs_dilation = &.{ 1, 1 },
        .window_reversal = &.{ false, false },
        .dimensions = .{
            .input_batch_dimension = 0,
            .input_feature_dimension = 3,
            .input_spatial_dimensions = &.{ 1, 2 },
            .kernel_input_feature_dimension = 2,
            .kernel_output_feature_dimension = 3,
            .kernel_spatial_dimensions = &.{ 0, 1 },
            .output_batch_dimension = 0,
            .output_feature_dimension = 3,
            .output_spatial_dimensions = &.{ 1, 2 },
        },
    };
}

test "trace inference and differentiated training" {
    const std = @import("std");
    const testing = std.testing;

    var inference = try zg.trace(
        forward,
        testing.allocator,
        .{ params_spec, inference_images_spec },
        .{},
    );
    defer inference.deinit();

    var training = try zg.trace(
        train_step,
        testing.allocator,
        .{ params_spec, training_batch_spec },
        .{ .name = "train_step" },
    );
    defer training.deinit();
}
