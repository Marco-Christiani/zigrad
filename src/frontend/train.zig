const std = @import("std");

const pr = @import("../pr/pr.zig");
const ad = @import("../pr/ad.zig");
const ops = @import("../pr/ops/ops.zig");
const backend = @import("../backend/root.zig");
const Backend = backend.Backend;
const frontend = @import("frontend.zig");

const Builder = frontend.Builder;
const Tensor = frontend.Tensor;
const TensorSpec = frontend.TensorSpec;

pub const OptimizerKind = enum { sgd };

pub const OptimizerOpts = struct {
    kind: OptimizerKind = .sgd,
    lr: f32 = 1e-3,
};

pub const TrainConfig = struct {
    optimizer: OptimizerOpts = .{},
    compile: frontend.CompileConfig = .{},
};

/// Compiled training step with buffer management metadata.
///
/// `param_count` and `batch_count` record how many of the flat inputs are
/// trainable parameters vs. batch data. This lets `TrainState` split the
/// input buffer array correctly.
pub const CompiledTrainStep = struct {
    exe: Backend.Executable,
    input_arity: usize,
    output_arity: usize,
    param_count: usize,
    batch_count: usize,
};

pub fn compile_train_step(
    allocator: std.mem.Allocator,
    backend_handle: *Backend,
    device: Backend.Device,
    func: anytype,
    inputs: anytype,
    param_count: usize,
    config: TrainConfig,
) !CompiledTrainStep {
    const compile_cfg = config.compile;

    var program = pr.Program.init(allocator);
    defer program.deinit();

    var loss_builder = try Builder.init(&program, "loss");
    defer loss_builder.deinit();

    const input_tensors = try frontend.build_inputs(&loss_builder, inputs);

    const result = if (@typeInfo(@TypeOf(input_tensors)) == .@"struct" and @typeInfo(@TypeOf(input_tensors)).@"struct".is_tuple)
        @call(.auto, func, input_tensors)
    else
        @call(.auto, func, .{input_tensors});

    const outputs = switch (@typeInfo(@TypeOf(result))) {
        .error_union => try result,
        else => result,
    };

    const output_tensors = try frontend.flatten_outputs(allocator, outputs);
    defer allocator.free(output_tensors);
    if (output_tensors.len != 1) return error.UnexpectedOutputs;

    const loss_func = try loss_builder.finish(output_tensors);

    const vjp_func = try ad.vjp_with_value(program.allocator(), &program, loss_func, "loss_vjp");
    try program.add_function(vjp_func);

    var step_builder = try pr.FunctionBuilder.init(&program, compile_cfg.entry_name);
    defer step_builder.deinit();

    const flat_specs = try frontend.flatten_specs(allocator, inputs);
    defer allocator.free(flat_specs);

    const primals = try allocator.alloc(pr.VarId, flat_specs.len);
    defer allocator.free(primals);
    for (flat_specs, 0..) |spec, i| {
        primals[i] = try step_builder.param_tensor(spec.dtype, spec.dims);
    }
    if (param_count > primals.len) return error.InvalidParams;

    const loss_tensor = output_tensors[0].tensor;
    const cot = try emit_cotangent(&step_builder, loss_tensor);

    const call_inputs = try allocator.alloc(pr.VarId, primals.len + 1);
    defer allocator.free(call_inputs);
    @memcpy(call_inputs[0..primals.len], primals);
    call_inputs[primals.len] = cot;

    const call_outputs = try step_builder.call("loss_vjp", call_inputs);
    if (call_outputs.len != primals.len + 1) return error.UnexpectedOutputs;

    const loss_value = call_outputs[0];
    const grads = call_outputs[1..];

    const updated = try emit_optimizer_updates(allocator, &step_builder, primals[0..param_count], grads[0..param_count], config.optimizer);
    defer allocator.free(updated);

    const returns = try allocator.alloc(pr.VarId, 1 + param_count);
    defer allocator.free(returns);
    returns[0] = loss_value;
    @memcpy(returns[1..], updated);

    const step_func = try step_builder.finish(returns);
    try program.add_function(step_func);

    const fwd_exe = try frontend.compile_program(
        backend_handle,
        allocator,
        &program,
        device,
        compile_cfg,
        compile_cfg.entry_name,
    );

    return .{
        .exe = fwd_exe,
        .input_arity = flat_specs.len,
        .output_arity = 1 + param_count,
        .param_count = param_count,
        .batch_count = flat_specs.len - param_count,
    };
}

/// Execute-swap-deinit loop for training steps.
///
/// Owns the parameter buffers (deinits old ones on swap). Batch buffers are
///  borrowed - the caller manages their lifetime and replaces them via
///  `set_batch`.
pub const TrainState = struct {
    input_bufs: []Backend.RawBuffer,
    output_bufs: []?Backend.RawBuffer,
    exe: Backend.Executable,
    backend_handle: *Backend,
    param_count: usize,
    non_donatable: []const i64,
    allocator: std.mem.Allocator,

    pub const StepResult = struct {
        loss_buf: Backend.Buffer,
        event: ?Backend.Event,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        compiled: *CompiledTrainStep,
        backend_handle: *Backend,
        initial_param_bufs: []const Backend.RawBuffer,
        initial_batch_bufs: []const Backend.RawBuffer,
    ) !TrainState {
        if (initial_param_bufs.len != compiled.param_count) return error.InvalidParams;
        if (initial_batch_bufs.len != compiled.batch_count) return error.InvalidBatchCount;

        const total = compiled.input_arity;
        const input_bufs = try allocator.alloc(Backend.RawBuffer, total);
        @memcpy(input_bufs[0..compiled.param_count], initial_param_bufs);
        @memcpy(input_bufs[compiled.param_count..], initial_batch_bufs);

        const output_bufs = try allocator.alloc(?Backend.RawBuffer, compiled.output_arity);
        @memset(output_bufs, null);

        const non_donatable = try allocator.alloc(i64, compiled.batch_count);
        for (0..compiled.batch_count) |i| {
            non_donatable[i] = @intCast(compiled.param_count + i);
        }

        return .{
            .input_bufs = input_bufs,
            .output_bufs = output_bufs,
            .exe = compiled.exe,
            .backend_handle = backend_handle,
            .param_count = compiled.param_count,
            .non_donatable = non_donatable,
            .allocator = allocator,
        };
    }

    /// Execute one training step. Swaps parameter buffers in-place.
    ///
    /// Returns `StepResult` with the loss buffer and optional completion event.
    /// Caller owns loss buffer.
    pub fn step(self: *TrainState) !StepResult {
        @memset(self.output_bufs, null);
        const event = try self.backend_handle.execute_into(
            self.exe,
            self.input_bufs,
            self.output_bufs,
            self.non_donatable,
            .{},
        );

        const loss_raw = self.output_bufs[0] orelse return error.NullOutputBuffer;
        self.output_bufs[0] = null;

        for (self.input_bufs[0..self.param_count], self.output_bufs[1 .. 1 + self.param_count]) |*old, new| {
            const new_raw = new orelse return error.NullOutputBuffer;
            if (new_raw == old.*) continue;
            self.backend_handle.deinit_buffer(.{ .handle = old.* });
            old.* = new_raw;
        }

        return .{
            .loss_buf = .{ .handle = loss_raw },
            .event = event,
        };
    }

    /// Replace batch input buffers. Old batch buffers are NOT deinited
    ///  (owned by caller).
    pub fn set_batch(self: *TrainState, batch_bufs: []const Backend.RawBuffer) void {
        @memcpy(self.input_bufs[self.param_count..], batch_bufs);
    }

    /// Deinit all owned buffers (params only, not batch).
    pub fn deinit(self: *TrainState) void {
        for (self.input_bufs[0..self.param_count]) |raw| {
            self.backend_handle.deinit_buffer(.{ .handle = raw });
        }
        self.allocator.free(self.input_bufs);
        self.allocator.free(self.output_bufs);
        self.allocator.free(self.non_donatable);
    }
};

fn emit_optimizer_updates(
    allocator: std.mem.Allocator,
    builder: *pr.FunctionBuilder,
    params: []const pr.VarId,
    grads: []const pr.VarId,
    opts: OptimizerOpts,
) ![]pr.VarId {
    const updated = try allocator.alloc(pr.VarId, params.len);
    for (params, grads, 0..) |param, grad, i| {
        updated[i] = switch (opts.kind) {
            .sgd => try emit_sgd_update(builder, param, grad, opts.lr),
        };
    }
    return updated;
}

fn emit_cotangent(builder: *pr.FunctionBuilder, tensor: pr.Tensor) pr.BuildError!pr.VarId {
    const lit = ops.types.scalar_literal(tensor.dtype, 1.0);
    const scalar = try builder.literal_scalar(lit);
    if (tensor.shape.rank() == 0) return scalar;
    return try builder.broadcast_in_dim(scalar, tensor.shape.dims, &.{});
}

fn emit_sgd_update(builder: *pr.FunctionBuilder, param: pr.VarId, grad: pr.VarId, lr: f32) pr.BuildError!pr.VarId {
    const param_tensor = builder.avals.items[@intCast(param)].as_tensor() orelse return error.UnsupportedAval;
    const grad_tensor = builder.avals.items[@intCast(grad)].as_tensor() orelse return error.UnsupportedAval;

    // Convert gradient to match param dtype (e.g. when loss computes in f32 but params are bf16).
    const matched_grad = if (grad_tensor.dtype != param_tensor.dtype)
        try builder.convert(grad, param_tensor.dtype)
    else
        grad;

    const lr_lit = ops.types.scalar_literal(param_tensor.dtype, lr);
    const lr_scalar = try builder.literal_scalar(lr_lit);
    const lr_broadcast = if (param_tensor.shape.rank() == 0)
        lr_scalar
    else
        try builder.broadcast_in_dim(lr_scalar, param_tensor.shape.dims, &.{});
    const scaled = try builder.multiply(matched_grad, lr_broadcast);
    return try builder.subtract(param, scaled);
}
