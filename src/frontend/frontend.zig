const std = @import("std");

const pr = @import("../pr/pr.zig");
const ad = @import("../pr/ad.zig");
const ops = @import("../pr/ops/ops.zig");
const lower = @import("../lower/root.zig");
const pipeline = @import("../pipeline/root.zig");
const dump = @import("../pipeline/dump.zig");
const backend = @import("../backend/root.zig");

pub const TensorSpec = struct {
    dtype: pr.DType,
    dims: []const usize,
};

pub const OpOptions = struct {
    outline: bool = false,
    kernelize_provider: ?[]const u8 = null,
};

pub const Tensor = struct {
    id: pr.VarId,
    tensor: pr.Tensor,
    builder: *Builder,
    options: OpOptions = .{},

    pub fn annotate(self: Tensor, opts: OpOptions) Tensor {
        var result = self;
        result.options = opts;
        return result;
    }

    pub fn add(self: Tensor, rhs: Tensor) !Tensor {
        return self.builder.emit_binary(.add, self, rhs, self.options);
    }

    pub fn sub(self: Tensor, rhs: Tensor) !Tensor {
        return self.builder.emit_binary(.subtract, self, rhs, self.options);
    }

    pub fn mul(self: Tensor, rhs: Tensor) !Tensor {
        return self.builder.emit_binary(.multiply, self, rhs, self.options);
    }

    pub fn max(self: Tensor, rhs: Tensor) !Tensor {
        return self.builder.emit_binary(.maximum, self, rhs, self.options);
    }

    pub fn matmul(self: Tensor, rhs: Tensor) !Tensor {
        return self.builder.emit_binary(.dot, self, rhs, self.options);
    }

    pub fn reshape(self: Tensor, dims: []const usize) !Tensor {
        const dims_copy = try self.builder.program.allocator().dupe(usize, dims);
        return self.builder.emit_unary(.reshape, self, &.{.{ .out_shape = dims_copy }}, self.options);
    }

    pub fn broadcast_in_dim(self: Tensor, out_dims: []const usize, broadcast_dimensions: []const i64) !Tensor {
        const allocator = self.builder.program.allocator();
        const out_copy = try allocator.dupe(usize, out_dims);
        const bd_copy = try allocator.dupe(i64, broadcast_dimensions);
        return self.builder.emit_unary(
            .broadcast_in_dim,
            self,
            &.{ .{ .out_shape = out_copy }, .{ .broadcast_dimensions = bd_copy } },
            self.options,
        );
    }

    pub fn transpose(self: Tensor, permutation: []const i64) !Tensor {
        const perm_copy = try self.builder.program.allocator().dupe(i64, permutation);
        return self.builder.emit_unary(.transpose, self, &.{.{ .permutation = perm_copy }}, self.options);
    }

    pub fn reduce_sum(self: Tensor, axes: []const i64) !Tensor {
        const axes_copy = try self.builder.program.allocator().dupe(i64, axes);
        return self.builder.emit_unary(.reduce_sum, self, &.{.{ .reduce_axes = axes_copy }}, self.options);
    }

    pub fn relu(self: Tensor) !Tensor {
        const zero = try self.builder.scalar_literal(zero_literal(self.tensor.dtype));
        const broadcast = try self.builder.emit_unary(
            .broadcast_in_dim,
            zero,
            &.{
                .{ .out_shape = self.tensor.shape.dims },
                .{ .broadcast_dimensions = &.{} },
            },
            .{},
        );
        return self.builder.emit_binary(.maximum, self, broadcast, self.options);
    }
};

pub const Builder = struct {
    program: *pr.Program,
    builder: pr.FunctionBuilder,

    pub fn init(program: *pr.Program, name: []const u8) !Builder {
        return .{
            .program = program,
            .builder = try pr.FunctionBuilder.init(program, name),
        };
    }

    pub fn deinit(self: *Builder) void {
        self.builder.deinit();
    }

    pub fn param(self: *Builder, spec: TensorSpec) !Tensor {
        const id = try self.builder.param_tensor(spec.dtype, spec.dims);
        return self.tensor_from_id(id);
    }

    pub fn scalar_literal(self: *Builder, lit: pr.Literal) !Tensor {
        const id = try self.builder.literal_scalar(lit);
        return self.tensor_from_id(id);
    }

    pub fn finish(self: *Builder, returns: []const Tensor) !pr.Function {
        const ids = try self.program.allocator().alloc(pr.VarId, returns.len);
        for (returns, 0..) |t, i| ids[i] = t.id;
        const func = try self.builder.finish(ids);
        try self.program.add_function(func);
        return func;
    }

    fn emit_binary(self: *Builder, prim: pr.Prim, lhs: Tensor, rhs: Tensor, options: OpOptions) !Tensor {
        try self.assert_same_builder(lhs, rhs);
        const params = try self.params_with_options(&.{}, options);
        const id = try self.builder.emit(prim, &.{ lhs.id, rhs.id }, params);
        return self.tensor_from_id(id);
    }

    fn emit_unary(self: *Builder, prim: pr.Prim, operand: Tensor, params: []const pr.Param, options: OpOptions) !Tensor {
        try self.assert_builder(operand);
        const params_with_opts = try self.params_with_options(params, options);
        const id = try self.builder.emit(prim, &.{operand.id}, params_with_opts);
        return self.tensor_from_id(id);
    }

    fn params_with_options(self: *Builder, params: []const pr.Param, options: OpOptions) ![]const pr.Param {
        // Fast path: no options = no allocation
        if (!options.outline and options.kernelize_provider == null) {
            return params;
        }

        const a = self.program.allocator();
        var list = try std.ArrayList(pr.Param).initCapacity(a, params.len + 2);
        try list.appendSlice(a, params);
        if (options.outline) {
            try list.append(a, .{ .outline = true });
        }
        if (options.kernelize_provider) |provider| {
            const copy = try a.dupe(u8, provider);
            try list.append(a, .{ .kernelize_provider = copy });
        }
        return list.toOwnedSlice(a);
    }

    fn tensor_from_id(self: *Builder, id: pr.VarId) !Tensor {
        if (@as(usize, @intCast(id)) >= self.builder.avals.items.len) return error.InvalidVarId;
        const aval = self.builder.avals.items[@intCast(id)];
        const tensor = aval.as_tensor() orelse return error.UnsupportedAval;
        return .{ .id = id, .tensor = tensor, .builder = self, .options = .{} };
    }

    fn assert_builder(self: *Builder, operand: Tensor) !void {
        if (operand.builder != self) return error.CrossBuilderOp;
    }

    fn assert_same_builder(self: *Builder, lhs: Tensor, rhs: Tensor) !void {
        if (lhs.builder != self or rhs.builder != self) return error.CrossBuilderOp;
    }
};


pub const CompileConfig = struct {
    entry_name: []const u8 = "main",
    plugin_path: ?[]const u8 = null,
    device_index: usize = 0,
    lower: lower.LowerPassConfig = .{},
    dump_pr: ?dump.DumpConfig = null,
    dump_mlir: ?dump.DumpConfig = null,
    compile: backend.pjrt.CompileOptions = .{},
};


pub const CompiledForward = struct {
    allocator: std.mem.Allocator,
    exe: backend.pjrt.LoadedExecutable,
    input_specs: []TensorSpec,
    output_specs: []TensorSpec,

    pub fn deinit(self: *CompiledForward) void {
        self.exe.deinit();
        free_tensor_specs(self.allocator, self.input_specs);
        free_tensor_specs(self.allocator, self.output_specs);
    }

    pub fn execute(self: *CompiledForward, allocator: std.mem.Allocator, inputs: []const backend.pjrt.Buffer) !DeviceOutputs {
        if (inputs.len != self.input_specs.len) return error.InputArityMismatch;
        const result = try self.exe.execute(allocator, inputs);
        if (result.outputs.len != self.output_specs.len) {
            for (result.outputs) |buf| {
                var tmp = buf;
                tmp.deinit();
            }
            allocator.free(result.outputs);
            if (result.device_complete_event) |ev| {
                var tmp = ev;
                tmp.deinit();
            }
            return error.UnexpectedOutputs;
        }
        return .{
            .allocator = allocator,
            .outputs = result.outputs,
            .device_complete_event = result.device_complete_event,
        };
    }

    pub const ExecuteOptions = struct {
        non_donatable_input_indices: ?[]const i64 = null,
    };

    pub fn execute_into(
        self: *CompiledForward,
        input_ptrs: []const backend.pjrt.RawBuffer,
        output_ptrs: []backend.pjrt.RawBuffer,
        options: ExecuteOptions,
    ) !?backend.pjrt.Event {
        if (input_ptrs.len != self.input_specs.len) return error.InputArityMismatch;
        if (output_ptrs.len != self.output_specs.len) return error.OutputArityMismatch;
        return self.exe.execute_into_opts(input_ptrs, output_ptrs, options.non_donatable_input_indices);
    }
};

pub const CompiledValueAndGrad = struct {
    allocator: std.mem.Allocator,
    exe_fwd: backend.pjrt.LoadedExecutable,
    exe_vjp: backend.pjrt.LoadedExecutable,
    input_specs: []TensorSpec,
    output_specs: []TensorSpec,

    pub fn deinit(self: *CompiledValueAndGrad) void {
        self.exe_vjp.deinit();
        self.exe_fwd.deinit();
        free_tensor_specs(self.allocator, self.input_specs);
        free_tensor_specs(self.allocator, self.output_specs);
    }

    pub fn execute(
        self: *CompiledValueAndGrad,
        allocator: std.mem.Allocator,
        inputs: []const backend.pjrt.Buffer,
        cotangent: backend.pjrt.Buffer,
    ) !ValueAndGradDevice {
        if (inputs.len != self.input_specs.len) return error.InputArityMismatch;

        const fwd = try self.exe_fwd.execute(allocator, inputs);
        if (fwd.outputs.len != 1) {
            for (fwd.outputs) |buf| {
                var tmp = buf;
                tmp.deinit();
            }
            allocator.free(fwd.outputs);
            if (fwd.device_complete_event) |ev| {
                var tmp = ev;
                tmp.deinit();
            }
            return error.UnexpectedOutputs;
        }
        const value = fwd.outputs[0];
        allocator.free(fwd.outputs);
        if (fwd.device_complete_event) |ev| {
            var tmp = ev;
            tmp.deinit();
        }

        const vjp_inputs = try concat_device_buffers(allocator, inputs, &.{cotangent});
        defer allocator.free(vjp_inputs);

        const vjp = try self.exe_vjp.execute(allocator, vjp_inputs);
        return .{
            .allocator = allocator,
            .value = value,
            .grads = vjp.outputs,
            .device_complete_event = vjp.device_complete_event,
        };
    }
};

pub const DeviceOutputs = struct {
    allocator: std.mem.Allocator,
    outputs: []backend.pjrt.Buffer,
    device_complete_event: ?backend.pjrt.Event,

    pub fn deinit(self: *DeviceOutputs) void {
        if (self.device_complete_event) |ev| {
            var tmp = ev;
            tmp.deinit();
        }
        for (self.outputs) |buf| {
            var tmp = buf;
            tmp.deinit();
        }
        self.allocator.free(self.outputs);
    }
};

pub const ValueAndGradDevice = struct {
    allocator: std.mem.Allocator,
    value: backend.pjrt.Buffer,
    grads: []backend.pjrt.Buffer,
    device_complete_event: ?backend.pjrt.Event,

    pub fn deinit(self: *ValueAndGradDevice) void {
        if (self.device_complete_event) |ev| {
            var tmp = ev;
            tmp.deinit();
        }
        var tmp = self.value;
        tmp.deinit();
        for (self.grads) |buf| {
            var g = buf;
            g.deinit();
        }
        self.allocator.free(self.grads);
    }
};

pub fn compile_train_step(
    allocator: std.mem.Allocator,
    backend_handle: *backend.PjrtBackend,
    device: *const backend.pjrt.Device,
    func: anytype,
    inputs: anytype,
    param_count: usize,
    lr: f32,
    config: CompileConfig,
) !CompiledForward {
    var program = pr.Program.init(allocator);
    defer program.deinit();

    var loss_builder = try Builder.init(&program, "loss");
    defer loss_builder.deinit();

    const input_specs = try clone_tensor_specs(allocator, inputs);
    const input_tensors = try build_inputs(&loss_builder, inputs);

    const result = if (@typeInfo(@TypeOf(input_tensors)) == .@"struct" and @typeInfo(@TypeOf(input_tensors)).@"struct".is_tuple)
        @call(.auto, func, input_tensors)
    else
        @call(.auto, func, .{input_tensors});

    const outputs = switch (@typeInfo(@TypeOf(result))) {
        .error_union => try result,
        else => result,
    };

    const output_tensors = try flatten_outputs(allocator, outputs);
    defer allocator.free(output_tensors);
    if (output_tensors.len != 1) return error.UnexpectedOutputs;

    const loss_func = try loss_builder.finish(output_tensors);

    const vjp_func = try ad.vjp_with_value(program.allocator(), &program, loss_func, "loss_vjp");
    try program.add_function(vjp_func);

    var step_builder = try pr.FunctionBuilder.init(&program, config.entry_name);
    defer step_builder.deinit();

    const flat_specs = try flatten_specs(allocator, inputs);
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

    const updated = try allocator.alloc(pr.VarId, param_count);
    defer allocator.free(updated);
    for (0..param_count) |i| {
        updated[i] = try emit_sgd_update(&step_builder, primals[i], grads[i], lr);
    }

    const returns = try allocator.alloc(pr.VarId, 1 + param_count);
    defer allocator.free(returns);
    returns[0] = loss_value;
    @memcpy(returns[1..], updated);

    const step_func = try step_builder.finish(returns);
    try program.add_function(step_func);

    const output_specs = try tensor_specs_from_varids(allocator, step_func, step_func.returns);

    const fwd_exe = try compile_program(
        backend_handle,
        allocator,
        &program,
        device,
        config,
        config.entry_name,
    );

    return .{
        .allocator = allocator,
        .exe = fwd_exe,
        .input_specs = input_specs,
        .output_specs = output_specs,
    };
}

pub fn compile_forward(
    allocator: std.mem.Allocator,
    backend_handle: *backend.PjrtBackend,
    device: *const backend.pjrt.Device,
    func: anytype,
    inputs: anytype,
    config: CompileConfig,
) !CompiledForward {
    var program = pr.Program.init(allocator);
    defer program.deinit();

    var builder = try Builder.init(&program, config.entry_name);
    defer builder.deinit();

    const input_specs = try clone_tensor_specs(allocator, inputs);
    const input_tensors = try build_inputs(&builder, inputs);

    const result = if (@typeInfo(@TypeOf(input_tensors)) == .@"struct" and @typeInfo(@TypeOf(input_tensors)).@"struct".is_tuple)
        @call(.auto, func, input_tensors)
    else
        @call(.auto, func, .{input_tensors});

    const outputs = switch (@typeInfo(@TypeOf(result))) {
        .error_union => try result,
        else => result,
    };

    const output_tensors = try flatten_outputs(allocator, outputs);
    defer allocator.free(output_tensors);
    if (output_tensors.len == 0) return error.NoOutputs;

    _ = try builder.finish(output_tensors);

    const output_specs = try tensor_specs_from_tensors(allocator, output_tensors);

    const fwd_exe = try compile_program(
        backend_handle,
        allocator,
        &program,
        device,
        config,
        config.entry_name,
    );

    return .{
        .allocator = allocator,
        .exe = fwd_exe,
        .input_specs = input_specs,
        .output_specs = output_specs,
    };
}

pub fn compile_value_and_grad(
    allocator: std.mem.Allocator,
    backend_handle: *backend.PjrtBackend,
    device: *const backend.pjrt.Device,
    func: anytype,
    inputs: anytype,
    config: CompileConfig,
) !CompiledValueAndGrad {
    var program = pr.Program.init(allocator);
    defer program.deinit();

    var builder = try Builder.init(&program, config.entry_name);
    defer builder.deinit();

    const input_specs = try clone_tensor_specs(allocator, inputs);
    const input_tensors = try build_inputs(&builder, inputs);

    const result = if (@typeInfo(@TypeOf(input_tensors)) == .@"struct" and @typeInfo(@TypeOf(input_tensors)).@"struct".is_tuple)
        @call(.auto, func, input_tensors)
    else
        @call(.auto, func, .{input_tensors});

    const outputs = switch (@typeInfo(@TypeOf(result))) {
        .error_union => try result,
        else => result,
    };

    const output_tensors = try flatten_outputs(allocator, outputs);
    defer allocator.free(output_tensors);
    if (output_tensors.len == 0) return error.NoOutputs;

    _ = try builder.finish(output_tensors);

    const fwd = program.functions[0];
    const vjp_func = try ad.vjp(program.allocator(), &program, fwd, "main_vjp");
    try program.add_function(vjp_func);

    const output_specs = try tensor_specs_from_tensors(allocator, output_tensors);

    if (output_specs.len != 1) {
        free_tensor_specs(allocator, output_specs);
        return error.UnexpectedOutputs;
    }

    const fwd_exe = try compile_program(
        backend_handle,
        allocator,
        &program,
        device,
        config,
        config.entry_name,
    );

    const vjp_exe = try compile_program(
        backend_handle,
        allocator,
        &program,
        device,
        config,
        "main_vjp",
    );

    return .{
        .allocator = allocator,
        .exe_fwd = fwd_exe,
        .exe_vjp = vjp_exe,
        .input_specs = input_specs,
        .output_specs = output_specs,
    };
}

pub fn build_demo_program(allocator: std.mem.Allocator) !pr.Program {
    var program = pr.Program.init(allocator);
    errdefer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 3 });
    const b_id = try b.param_tensor(.f32, &.{ 3, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    const dot_id = try b.dot(a_id, b_id);
    const add_id = try b.add(dot_id, c_id);
    const out_id = try b.multiply(add_id, c_id);

    const func = try b.finish(&.{out_id});
    try program.add_function(func);

    return program;
}

fn compile_program(
    backend_handle: *const backend.PjrtBackend,
    allocator: std.mem.Allocator,
    program: *pr.Program,
    device: *const backend.pjrt.Device,
    config: CompileConfig,
    entry_name: []const u8,
) !backend.pjrt.LoadedExecutable {
    var lower_cfg = config.lower;
    if (lower_cfg.entry_name == null) lower_cfg.entry_name = entry_name;

    var compile_cfg = backend.pjrt.Backend.CompilePassConfig{
        .device = device,
        .options = config.compile,
    };

    var passes = std.ArrayList(pipeline.Pass).initCapacity(allocator, 5) catch
        return error.OutOfMemory;
    defer passes.deinit(allocator);

    var dump_pr_local: ?dump.DumpConfig = null;
    if (config.dump_pr) |cfg| {
        dump_pr_local = cfg;
        dump_pr_local.?.entry_name = dump_pr_local.?.entry_name orelse entry_name;
        try passes.append(allocator, dump.dump_pr_pass_with_config(&dump_pr_local.?));
    }
    try passes.append(allocator, lower.validate_pass);
    try passes.append(allocator, lower.lower_pass_with_config(&lower_cfg));

    var dump_mlir_local: ?dump.DumpConfig = null;
    if (config.dump_mlir) |cfg| {
        dump_mlir_local = cfg;
        dump_mlir_local.?.entry_name = dump_mlir_local.?.entry_name orelse entry_name;
        try passes.append(allocator, dump.dump_mlir_pass_with_config(&dump_mlir_local.?));
    }
    try passes.append(allocator, @constCast(backend_handle).compile_pass(&compile_cfg));

    const pipeline_run = pipeline.Pipeline{ .passes = passes.items };
    var ctx = pipeline.PassContext{ .allocator = allocator };

    var artifact = try pipeline_run.run(.{ .pr = program }, &ctx);
    errdefer artifact.deinit(allocator);
    return switch (artifact) {
        .ea => |ea| switch (ea) {
            .pjrt => |exe| exe,
        },
        inline else => error.UnexpectedArtifact,
    };
}

pub fn init_backend(allocator: std.mem.Allocator, plugin_path: ?[]const u8) !backend.PjrtBackend {
    if (plugin_path) |path| {
        return backend.PjrtBackend.init(allocator, path);
    }
    const path = try std.process.getEnvVarOwned(allocator, "PJRT_PLUGIN_PATH");
    defer allocator.free(path);
    return backend.PjrtBackend.init(allocator, path);
}

fn build_inputs(builder: *Builder, spec: anytype) !SpecToTensorType(@TypeOf(spec)) {
    const T = @TypeOf(spec);
    if (T == TensorSpec) {
        return try builder.param(spec);
    }
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            var out: SpecToTensorType(T) = undefined;
            inline for (info.fields) |field| {
                @field(out, field.name) = try build_inputs(builder, @field(spec, field.name));
            }
            return out;
        },
        else => {
            if (T != TensorSpec) {
                @compileError("input spec must be TensorSpec or a struct/tuple of TensorSpec");
            }
            return try builder.param(spec);
        },
    }
}

fn flatten_outputs(allocator: std.mem.Allocator, output: anytype) ![]Tensor {
    var list = try std.ArrayList(Tensor).initCapacity(allocator, 8);
    errdefer list.deinit(allocator);
    try append_output(allocator, &list, output);
    return list.toOwnedSlice(allocator);
}

fn append_output(allocator: std.mem.Allocator, list: *std.ArrayList(Tensor), output: anytype) !void {
    const T = @TypeOf(output);
    if (T == Tensor) {
        try list.append(allocator, output);
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            inline for (info.fields) |field| {
                try append_output(allocator, list, @field(output, field.name));
            }
        },
        else => {
            @compileError("output must be Tensor or a struct/tuple of Tensor");
        },
    }
}

fn SpecToTensorType(comptime T: type) type {
    if (T == TensorSpec) return Tensor;
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            var fields: [info.fields.len]std.builtin.Type.StructField = undefined;
            inline for (info.fields, 0..) |field, i| {
                const field_type = SpecToTensorType(field.type);
                fields[i] = .{
                    .name = field.name,
                    .type = field_type,
                    .default_value_ptr = null,
                    .is_comptime = false,
                    .alignment = @alignOf(field_type),
                };
            }
            return @Type(.{
                .@"struct" = .{
                    .layout = .auto,
                    .fields = &fields,
                    .decls = &.{},
                    .is_tuple = info.is_tuple,
                },
            });
        },
        else => {
            if (T == TensorSpec) return Tensor;
            @compileError("input spec must be TensorSpec or a struct/tuple of TensorSpec");
        },
    }
}

fn tensor_specs_from_tensors(allocator: std.mem.Allocator, tensors: []const Tensor) ![]TensorSpec {
    const out = try allocator.alloc(TensorSpec, tensors.len);
    errdefer allocator.free(out);
    var copied: usize = 0;
    errdefer {
        for (out[0..copied]) |spec| allocator.free(spec.dims);
    }
    for (tensors, 0..) |tensor, i| {
        const dims_copy = try allocator.dupe(usize, tensor.tensor.shape.dims);
        out[i] = .{ .dtype = tensor.tensor.dtype, .dims = dims_copy };
        copied += 1;
    }
    return out;
}

fn tensor_specs_from_varids(allocator: std.mem.Allocator, func: pr.Function, varids: []const pr.VarId) ![]TensorSpec {
    const out = try allocator.alloc(TensorSpec, varids.len);
    errdefer allocator.free(out);
    var copied: usize = 0;
    errdefer {
        for (out[0..copied]) |spec| allocator.free(spec.dims);
    }
    for (varids, 0..) |var_id, i| {
        const tensor = func.avals[@intCast(var_id)].as_tensor() orelse return error.UnsupportedAval;
        const dims_copy = try allocator.dupe(usize, tensor.shape.dims);
        out[i] = .{ .dtype = tensor.dtype, .dims = dims_copy };
        copied += 1;
    }
    return out;
}

fn flatten_specs(allocator: std.mem.Allocator, spec: anytype) ![]TensorSpec {
    var list = try std.ArrayList(TensorSpec).initCapacity(allocator, 16);
    errdefer list.deinit(allocator);
    try append_specs(allocator, &list, spec);
    return list.toOwnedSlice(allocator);
}

fn append_specs(allocator: std.mem.Allocator, list: *std.ArrayList(TensorSpec), spec: anytype) !void {
    const T = @TypeOf(spec);
    if (T == TensorSpec) {
        try list.append(allocator, spec);
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            inline for (info.fields) |field| {
                try append_specs(allocator, list, @field(spec, field.name));
            }
        },
        else => {
            @compileError("input spec must be TensorSpec or a struct/tuple of TensorSpec");
        },
    }
}

fn concat_device_buffers(
    allocator: std.mem.Allocator,
    first: []const backend.pjrt.Buffer,
    second: []const backend.pjrt.Buffer,
) ![]backend.pjrt.Buffer {
    const out = try allocator.alloc(backend.pjrt.Buffer, first.len + second.len);
    @memcpy(out[0..first.len], first);
    @memcpy(out[first.len..], second);
    return out;
}

fn zero_literal(dtype: pr.DType) pr.Literal {
    return switch (dtype) {
        .f32 => .{ .f32 = 0.0 },
        .f64 => .{ .f64 = 0.0 },
        .i32 => .{ .i32 = 0 },
        .i64 => .{ .i64 = 0 },
        .u32 => .{ .u32 = 0 },
        .u64 => .{ .u64 = 0 },
    };
}

fn emit_cotangent(builder: *pr.FunctionBuilder, tensor: pr.Tensor) pr.BuildError!pr.VarId {
    const lit = ops.types.scalar_literal(tensor.dtype, 1.0);
    const scalar = try builder.literal_scalar(lit);
    if (tensor.shape.rank() == 0) return scalar;
    return try builder.broadcast_in_dim(scalar, tensor.shape.dims, &.{});
}

fn emit_sgd_update(builder: *pr.FunctionBuilder, param: pr.VarId, grad: pr.VarId, lr: f32) pr.BuildError!pr.VarId {
    const param_tensor = builder.avals.items[@intCast(param)].as_tensor() orelse return error.UnsupportedAval;
    const lr_lit = ops.types.scalar_literal(param_tensor.dtype, lr);
    const lr_scalar = try builder.literal_scalar(lr_lit);
    const lr_broadcast = if (param_tensor.shape.rank() == 0)
        lr_scalar
    else
        try builder.broadcast_in_dim(lr_scalar, param_tensor.shape.dims, &.{});
    const scaled = try builder.multiply(grad, lr_broadcast);
    return try builder.subtract(param, scaled);
}


fn clone_tensor_specs(allocator: std.mem.Allocator, spec: anytype) ![]TensorSpec {
    const flat = try flatten_specs(allocator, spec);
    errdefer allocator.free(flat);

    var copied: usize = 0;
    errdefer {
        for (flat[0..copied]) |item| allocator.free(item.dims);
    }

    for (flat) |*item| {
        const dims_copy = try allocator.dupe(usize, item.dims);
        item.dims = dims_copy;
        copied += 1;
    }

    return flat;
}

fn free_tensor_specs(allocator: std.mem.Allocator, specs: []TensorSpec) void {
    for (specs) |spec| allocator.free(spec.dims);
    allocator.free(specs);
}
