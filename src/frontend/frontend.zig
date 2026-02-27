const std = @import("std");

const pr = @import("../pr/pr.zig");
const ad = @import("../pr/ad.zig");
const ops = @import("../pr/ops/ops.zig");
const lower = @import("../lower/root.zig");
const pipeline = @import("../pipeline/root.zig");
const dump = @import("../pipeline/dump.zig");
const backend = @import("../backend/root.zig");
const utils = @import("../utils/host_buffer.zig");

pub const train = @import("train.zig");

pub const TensorSpec = struct {
    dtype: pr.DType,
    dims: []const usize,
};

pub const Tensor = struct {
    id: pr.VarId,
    tensor: pr.Tensor,
    builder: *Builder,

    pub fn add(self: Tensor, rhs: Tensor) !Tensor {
        return self.builder.emit_binary(.add, self, rhs);
    }

    pub fn sub(self: Tensor, rhs: Tensor) !Tensor {
        return self.builder.emit_binary(.subtract, self, rhs);
    }

    pub fn mul(self: Tensor, rhs: Tensor) !Tensor {
        return self.builder.emit_binary(.multiply, self, rhs);
    }

    pub fn div(self: Tensor, rhs: Tensor) !Tensor {
        return self.builder.emit_binary(.divide, self, rhs);
    }

    pub fn max(self: Tensor, rhs: Tensor) !Tensor {
        return self.builder.emit_binary(.maximum, self, rhs);
    }

    pub fn gather_rows(self: Tensor, indices: Tensor) !Tensor {
        return self.builder.emit_gather_rows(self, indices);
    }

    pub fn gather(self: Tensor, indices: Tensor, params: pr.GatherParams) !Tensor {
        return self.builder.emit_gather(self, indices, params);
    }

    pub fn gather_2d(self: Tensor, indices: Tensor) !Tensor {
        return self.builder.emit_gather_2d(self, indices);
    }

    pub fn matmul(self: Tensor, rhs: Tensor) !Tensor {
        return self.builder.emit_binary(.dot, self, rhs);
    }

    pub fn dot_general(self: Tensor, rhs: Tensor, params: pr.DotGeneralParams) !Tensor {
        return self.builder.emit_dot_general(self, rhs, params);
    }

    pub fn reshape(self: Tensor, dims: []const usize) !Tensor {
        const dims_copy = try self.builder.program.allocator().dupe(usize, dims);
        return self.builder.emit_unary(.reshape, self, &.{.{ .out_shape = dims_copy }});
    }

    pub fn broadcast_in_dim(self: Tensor, out_dims: []const usize, broadcast_dimensions: []const i64) !Tensor {
        const allocator = self.builder.program.allocator();
        const out_copy = try allocator.dupe(usize, out_dims);
        const bd_copy = try allocator.dupe(i64, broadcast_dimensions);
        return self.builder.emit_unary(
            .broadcast_in_dim,
            self,
            &.{ .{ .out_shape = out_copy }, .{ .broadcast_dimensions = bd_copy } },
        );
    }

    pub fn transpose(self: Tensor, permutation: []const i64) !Tensor {
        const perm_copy = try self.builder.program.allocator().dupe(i64, permutation);
        return self.builder.emit_unary(.transpose, self, &.{.{ .permutation = perm_copy }});
    }

    pub fn slice(self: Tensor, start_indices: []const i64, limit_indices: []const i64, strides: []const i64) !Tensor {
        const allocator = self.builder.program.allocator();
        const start_copy = try allocator.dupe(i64, start_indices);
        const limit_copy = try allocator.dupe(i64, limit_indices);
        const stride_copy = try allocator.dupe(i64, strides);
        return self.builder.emit_unary(
            .slice,
            self,
            &.{.{ .slice = .{ .start_indices = start_copy, .limit_indices = limit_copy, .strides = stride_copy } }},
        );
    }

    pub fn concatenate(self: Tensor, others: []const Tensor, axis: i64) !Tensor {
        return self.builder.emit_concatenate(self, others, axis);
    }

    pub fn reduce_sum(self: Tensor, axes: []const i64) !Tensor {
        const axes_copy = try self.builder.program.allocator().dupe(i64, axes);
        return self.builder.emit_unary(.reduce_sum, self, &.{.{ .reduce_axes = axes_copy }});
    }

    pub fn reduce_max(self: Tensor, axes: []const i64) !Tensor {
        const axes_copy = try self.builder.program.allocator().dupe(i64, axes);
        return self.builder.emit_unary(.reduce_max, self, &.{.{ .reduce_axes = axes_copy }});
    }

    pub fn exp(self: Tensor) !Tensor {
        return self.builder.emit_unary(.exp, self, &.{});
    }

    pub fn log(self: Tensor) !Tensor {
        return self.builder.emit_unary(.log, self, &.{});
    }

    pub fn rsqrt(self: Tensor) !Tensor {
        return self.builder.emit_unary(.rsqrt, self, &.{});
    }

    pub fn logistic(self: Tensor) !Tensor {
        return self.builder.emit_unary(.logistic, self, &.{});
    }

    pub fn convert(self: Tensor, out_dtype: pr.DType) !Tensor {
        return self.builder.emit_convert(self, out_dtype);
    }

    pub fn compare(self: Tensor, rhs: Tensor, params: pr.CompareParams) !Tensor {
        return self.builder.emit_compare(self, rhs, params);
    }

    pub fn select(self: Tensor, cond: Tensor, on_false: Tensor) !Tensor {
        return self.builder.emit_select(cond, self, on_false);
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
        );
        return self.builder.emit_binary(.maximum, self, broadcast);
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

    pub fn iota(self: *Builder, out_dtype: pr.DType, out_dims: []const usize, iota_dim: i64) !Tensor {
        const id = try self.builder.iota(out_dtype, out_dims, iota_dim);
        return self.tensor_from_id(id);
    }

    pub fn finish(self: *Builder, returns: []const Tensor) !pr.Function {
        const ids = try self.program.allocator().alloc(pr.VarId, returns.len);
        for (returns, 0..) |t, i| ids[i] = t.id;
        const func = try self.builder.finish(ids);
        try self.program.add_function(func);
        return func;
    }

    /// Push a named annotation region. Equations emitted after this call
    /// belong to this region until pop_region is called.
    pub fn push_region(self: *Builder, name: []const u8, annotation: pr.Annotation) !void {
        try self.builder.push_region(name, annotation);
    }

    /// Pop the most recent annotation region.
    pub fn pop_region(self: *Builder) !void {
        try self.builder.pop_region();
    }

    fn emit_binary(self: *Builder, prim: pr.Prim, lhs: Tensor, rhs: Tensor) !Tensor {
        try self.assert_same_builder(lhs, rhs);
        const id = try self.builder.emit(prim, &.{ lhs.id, rhs.id }, &.{});
        return self.tensor_from_id(id);
    }

    fn emit_unary(self: *Builder, prim: pr.Prim, operand: Tensor, params: []const pr.Param) !Tensor {
        try self.assert_builder(operand);
        const id = try self.builder.emit(prim, &.{operand.id}, params);
        return self.tensor_from_id(id);
    }

    fn emit_dot_general(self: *Builder, lhs: Tensor, rhs: Tensor, params: pr.DotGeneralParams) !Tensor {
        try self.assert_same_builder(lhs, rhs);
        const a = self.program.allocator();
        const lhs_batch_dims = try a.dupe(i64, params.lhs_batch_dims);
        const rhs_batch_dims = try a.dupe(i64, params.rhs_batch_dims);
        const lhs_contracting_dims = try a.dupe(i64, params.lhs_contracting_dims);
        const rhs_contracting_dims = try a.dupe(i64, params.rhs_contracting_dims);
        const id = try self.builder.emit(.dot_general, &.{ lhs.id, rhs.id }, &.{.{ .dot_general = .{
            .lhs_batch_dims = lhs_batch_dims,
            .rhs_batch_dims = rhs_batch_dims,
            .lhs_contracting_dims = lhs_contracting_dims,
            .rhs_contracting_dims = rhs_contracting_dims,
        } }});
        return self.tensor_from_id(id);
    }

    fn emit_compare(self: *Builder, lhs: Tensor, rhs: Tensor, params: pr.CompareParams) !Tensor {
        try self.assert_same_builder(lhs, rhs);
        const id = try self.builder.emit(.compare, &.{ lhs.id, rhs.id }, &.{.{ .compare = params }});
        return self.tensor_from_id(id);
    }

    fn emit_select(self: *Builder, cond: Tensor, on_true: Tensor, on_false: Tensor) !Tensor {
        try self.assert_builder(cond);
        try self.assert_same_builder(on_true, on_false);
        const id = try self.builder.emit(.select, &.{ cond.id, on_true.id, on_false.id }, &.{});
        return self.tensor_from_id(id);
    }

    fn emit_concatenate(self: *Builder, first: Tensor, others: []const Tensor, axis: i64) !Tensor {
        const a = self.program.allocator();
        const inputs = try a.alloc(pr.VarId, others.len + 1);
        inputs[0] = first.id;
        for (others, 0..) |t, i| {
            try self.assert_same_builder(first, t);
            inputs[i + 1] = t.id;
        }
        const id = try self.builder.emit(.concatenate, inputs, &.{.{ .concat_axis = axis }});
        return self.tensor_from_id(id);
    }

    fn emit_convert(self: *Builder, operand: Tensor, out_dtype: pr.DType) !Tensor {
        try self.assert_builder(operand);
        const id = try self.builder.emit(.convert, &.{operand.id}, &.{.{ .out_dtype = out_dtype }});
        return self.tensor_from_id(id);
    }

    fn emit_gather_rows(self: *Builder, operand: Tensor, indices: Tensor) !Tensor {
        try self.assert_same_builder(operand, indices);
        if (operand.tensor.shape.rank() != 2) return error.InvalidGatherOperand;
        if (indices.tensor.shape.rank() != 1) return error.InvalidGatherIndices;

        const hidden: i64 = @intCast(operand.tensor.shape.dims[1]);
        const a = self.program.allocator();

        const slice_sizes = try a.dupe(i64, &.{ 1, hidden });
        const offset_dims = try a.dupe(i64, &.{1});
        const collapsed_slice_dims = try a.dupe(i64, &.{0});
        const start_index_map = try a.dupe(i64, &.{0});

        const params: pr.GatherParams = .{
            .slice_sizes = slice_sizes,
            .offset_dims = offset_dims,
            .collapsed_slice_dims = collapsed_slice_dims,
            .start_index_map = start_index_map,
            .index_vector_dim = 1,
        };

        const id = try self.builder.emit(.gather, &.{ operand.id, indices.id }, &.{.{ .gather = params }});
        return self.tensor_from_id(id);
    }

    fn emit_gather(self: *Builder, operand: Tensor, indices: Tensor, params: pr.GatherParams) !Tensor {
        try self.assert_same_builder(operand, indices);
        const a = self.program.allocator();

        const slice_sizes = try a.dupe(i64, params.slice_sizes);
        const offset_dims = try a.dupe(i64, params.offset_dims);
        const collapsed_slice_dims = try a.dupe(i64, params.collapsed_slice_dims);
        const start_index_map = try a.dupe(i64, params.start_index_map);

        const gparams: pr.GatherParams = .{
            .slice_sizes = slice_sizes,
            .offset_dims = offset_dims,
            .collapsed_slice_dims = collapsed_slice_dims,
            .start_index_map = start_index_map,
            .index_vector_dim = params.index_vector_dim,
        };

        const id = try self.builder.emit(.gather, &.{ operand.id, indices.id }, &.{.{ .gather = gparams }});
        return self.tensor_from_id(id);
    }

    fn emit_gather_2d(self: *Builder, operand: Tensor, indices: Tensor) !Tensor {
        try self.assert_same_builder(operand, indices);
        if (operand.tensor.shape.rank() != 2) return error.InvalidGatherOperand;
        if (indices.tensor.shape.rank() != 2) return error.InvalidGatherIndices;
        if (indices.tensor.shape.dims[1] != 2) return error.InvalidGatherIndices;

        const a = self.program.allocator();
        const slice_sizes = try a.dupe(i64, &.{ 1, 1 });
        const offset_dims = try a.dupe(i64, &.{});
        const collapsed_slice_dims = try a.dupe(i64, &.{ 0, 1 });
        const start_index_map = try a.dupe(i64, &.{ 0, 1 });

        const params: pr.GatherParams = .{
            .slice_sizes = slice_sizes,
            .offset_dims = offset_dims,
            .collapsed_slice_dims = collapsed_slice_dims,
            .start_index_map = start_index_map,
            .index_vector_dim = 1,
        };

        const id = try self.builder.emit(.gather, &.{ operand.id, indices.id }, &.{.{ .gather = params }});
        return self.tensor_from_id(id);
    }

    fn tensor_from_id(self: *Builder, id: pr.VarId) !Tensor {
        if (@as(usize, @intCast(id)) >= self.builder.avals.items.len) return error.InvalidVarId;
        const aval = self.builder.avals.items[@intCast(id)];
        const tensor = aval.as_tensor() orelse return error.UnsupportedAval;
        return .{ .id = id, .tensor = tensor, .builder = self };
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

/// Compiled executable with arity metadata.
///
/// Callers manage the executable lifetime directly via `exe.deinit(api)`.
/// Arity fields record the expected flat input/output counts for validation.
pub const CompiledForward = struct {
    exe: backend.pjrt.LoadedExecutable,
    input_arity: usize,
    output_arity: usize,
};

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

    const flat_input_specs = try flatten_specs(allocator, inputs);
    defer allocator.free(flat_input_specs);

    const fwd_exe = try compile_program(
        backend_handle,
        allocator,
        &program,
        device,
        config,
        config.entry_name,
    );

    return .{
        .exe = fwd_exe,
        .input_arity = flat_input_specs.len,
        .output_arity = output_tensors.len,
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

pub fn compile_program(
    backend_handle: *backend.PjrtBackend,
    allocator: std.mem.Allocator,
    program: *pr.Program,
    device: *const backend.pjrt.Device,
    config: CompileConfig,
    entry_name: []const u8,
) !backend.pjrt.LoadedExecutable {
    var lower_cfg = config.lower;
    if (lower_cfg.entry_name == null) lower_cfg.entry_name = entry_name;

    var passes = std.ArrayList(pipeline.Pass).initCapacity(allocator, 4) catch
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

    const pipeline_run = pipeline.Pipeline{ .passes = passes.items };
    var ctx = pipeline.PassContext{ .allocator = allocator };

    var artifact = try pipeline_run.run(.{ .pr = program }, &ctx);
    defer artifact.deinit(allocator);

    const mlir = switch (artifact) {
        .mlir => |m| m,
        else => return error.UnexpectedArtifact,
    };

    var compile_opts = config.compile;
    if (compile_opts.kernel_package == null) {
        compile_opts.kernel_package = mlir.kernel_package;
    }

    return backend_handle.compile(device, mlir.bytes, mlir.encoding == .bytecode, compile_opts);
}

pub fn init_backend(allocator: std.mem.Allocator, plugin_path: ?[]const u8) !backend.PjrtBackend {
    if (plugin_path) |path| {
        return backend.PjrtBackend.init(allocator, path);
    }
    const path = try std.process.getEnvVarOwned(allocator, "PJRT_PLUGIN_PATH");
    defer allocator.free(path);
    return backend.PjrtBackend.init(allocator, path);
}

pub fn build_inputs(builder: *Builder, spec: anytype) !SpecToTensorType(@TypeOf(spec)) {
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
        .array => |info| {
            var out: SpecToTensorType(T) = undefined;
            var i: usize = 0;
            while (i < info.len) : (i += 1) {
                out[i] = try build_inputs(builder, spec[i]);
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

pub fn flatten_outputs(allocator: std.mem.Allocator, output: anytype) ![]Tensor {
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
        .array => |info| {
            var i: usize = 0;
            while (i < info.len) : (i += 1) {
                try append_output(allocator, list, output[i]);
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
        .array => |info| {
            const elem_type = SpecToTensorType(info.child);
            return @Type(.{ .array = .{ .len = info.len, .child = elem_type, .sentinel_ptr = null } });
        },
        else => {
            if (T == TensorSpec) return Tensor;
            @compileError("input spec must be TensorSpec or a struct/tuple of TensorSpec");
        },
    }
}

pub fn flatten_specs(allocator: std.mem.Allocator, spec: anytype) ![]TensorSpec {
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
        .array => |info| {
            var i: usize = 0;
            while (i < info.len) : (i += 1) {
                try append_specs(allocator, list, spec[i]);
            }
        },
        else => {
            @compileError("input spec must be TensorSpec or a struct/tuple of TensorSpec");
        },
    }
}

fn zero_literal(dtype: pr.DType) pr.Literal {
    return switch (dtype) {
        .bf16 => .{ .bf16 = 0 },
        .f32 => .{ .f32 = 0.0 },
        .f64 => .{ .f64 = 0.0 },
        .i32 => .{ .i32 = 0 },
        .i64 => .{ .i64 = 0 },
        .u32 => .{ .u32 = 0 },
        .u64 => .{ .u64 = 0 },
        .bool => .{ .bool = false },
    };
}

/// Upload a host buffer to a device buffer.
pub fn upload_host_buffer(
    allocator: std.mem.Allocator,
    backend_handle: *backend.PjrtBackend,
    device: *const backend.pjrt.Device,
    buf: *utils.HostBuffer,
) !backend.pjrt.Buffer {
    const shape_i64 = try allocator.alloc(i64, buf.shape.dims.len);
    defer allocator.free(shape_i64);
    for (buf.shape.dims, 0..) |d, i| shape_i64[i] = @intCast(d);
    const dtype: pr.DType = switch (buf.dtype) {
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .i32 => .i32,
        .i64 => .i64,
        .u32 => .u32,
        .u64 => .u64,
    };
    return backend_handle.buffer_from_host(device, buf.data, dtype, shape_i64);
}
