//! Frontend: user-facing API for building, transforming, and compiling programs.
//!
//! The frontend sits between user code and the PR/pipeline/backend layers. It
//!  provides three main capabilities:
//!
//! 1. **`compile`**: AOT compilation of a comptime function into an executable.
//!    Traces the function against abstract Tensor specs, optionally applies a
//!    transform (e.g. VJP), then lowers and compiles through the pipeline.
//!    Returns a `CompiledModel` ready for `TrainState` or manual execution.
//!
//! 2. **`transforms`**: Comptime function transforms (e.g. `value_and_grad`)
//!    that operate *during* tracing. These are called inside a traced function
//!    to compose AD with user logic (optimizer, regularization, etc.) in a
//!    single compiled program.
//!
//! 3. **`optim`**: Convenience optimizers for traced-mode (i.e. fusable updates)
//!
//! ## Two paths to value-and-grad
//!
//! Both use `pr.ad.vjp_with_value`/`pr.ad.jvp_with_value` under the hood. The
//!  choice depends on whether the optimizer runs inside or outside the
//!  compiled program:
//!
//! ### Path A: trace-time transform (optimizer in-graph)
//!
//! The user writes a `train_step` that calls `transforms.value_and_grad` and
//! applies optimizer updates inside the trace. The whole step compiles as one
//! program. This is the common training path.
//!
//! ```zig
//! fn train_step(params: Params, batch: Batch) !struct { loss: Tensor, updated: Params } {
//!     var vg = try transforms.value_and_grad(loss_fn, .{ params, batch });
//!     defer vg.deinit();
//!     // ... apply optimizer to vg.grads, return updated params
//! }
//!
//! var compiled = try frontend.compile(train_step, alloc, backend, device, specs, .{});
//! ```
//!
//! ### Path B: compile-time transform (out-of-graph optimizer)
//!
//! Pass `.transform = .value_and_grad` to `compile`. The compiled program returns
//!  `[loss, grad_0, ..., grad_n]` as raw buffers. The caller applies optimizer
//!  logic on the host or in a separate compiled program.
//!
//! ```zig
//! var compiled = try frontend.compile(loss_fn, alloc, backend, device, specs, .{
//!     .transform = .value_and_grad,
//! });
//! // execute returns [loss_buf, grad_buf_0, ..., grad_buf_n]
//! ```
//!
//! Path A is preferred for performance (one fused program).
//! Path B is useful for debugging, gradient inspection, or highly custom scenarios
//!  (e.g., when the optimizer is not expressible in traced Tensor ops).
const std = @import("std");

const pr = @import("../pr/pr.zig");
const ad = @import("../pr/ad.zig");
const ops = @import("../pr/ops/ops.zig");
const kernel = @import("../kernel.zig");
const lower = @import("../lower.zig");
const pipeline = @import("../pipeline/root.zig");
const backend = @import("../backend/root.zig");
const Backend = backend.Backend;
const HostBuffer = @import("../utils/root.zig").HostBuffer;
const Tree = @import("../utils/tree.zig").Tree;
const Tensor = @import("../tensor.zig");

/// Training loop state management (buffer swap, donation, execute).
pub const train = @import("train.zig");
/// Comptime function transforms for traced Tensor programs (e.g. `value_and_grad`).
pub const transforms = @import("transforms.zig");
/// Traced-mode optimizer building blocks (e.g. `optim.SGD`).
pub const optim = @import("optim.zig");

const log = std.log.scoped(.@"zg/frontend");

/// TODO: does this belong here anymore?
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

    /// Create a traced parameter tensor from an abstract tensor spec.
    pub fn param(self: *Builder, spec: Tensor) !Tensor {
        return Tensor.param(&self.builder, spec.dtype, spec.shape.const_slice());
    }

    /// Create a 0-d scalar constant tensor.
    pub fn scalar(self: *Builder, dtype: pr.DType, val: f64) !Tensor {
        const v = try self.builder.scalar(dtype, val);
        return Tensor.from_var(&self.builder, v);
    }

    pub fn iota(self: *Builder, out_dtype: pr.DType, out_dims: []const i64, iota_dim: i64) !Tensor {
        const v = try self.builder.iota(out_dtype, out_dims, iota_dim);
        return Tensor.from_var(&self.builder, v);
    }

    pub fn finish(self: *Builder, returns: []const Tensor) !pr.Function {
        const vars = try self.program.allocator().alloc(*pr.Var, returns.len);
        for (returns, 0..) |t, i| vars[i] = try t.get_var();
        const func = try self.builder.finish(vars);
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
};

/// Which transform to apply during `compile`.
///
/// This controls whether VJP is applied as a compilation step. For in-graph
/// optimizer composition, use `transforms.value_and_grad` inside a traced
/// function and compile with `.forward` instead. See the module-level docs
/// for a comparison of the two paths.
pub const Transform = enum {
    /// Compile the function as-is (no AD transform).
    forward,
    /// TODO: jvp
    /// Apply VJP at compile time. The compiled program takes the same inputs
    ///  as the original function and returns `[value, grad_0, ..., grad_n]`.
    /// The function must return a single scalar Tensor (the loss).
    /// Gradients are w.r.t. all inputs. Uses `pr.ad.vjp_with_value` internally.
    value_and_grad,
};

/// Full configuration for the `compile_program` pipeline.
///
/// Controls kernelization, lowering, dump points, and backend compile options.
/// The pipeline is a pass chain:
///  `[dump_pr] -> [kernelize] -> validate -> lower -> legalize -> [dump_mlir]`
/// When `kernel_store` is null, the kernelize step is skipped entirely.
///
/// MLIR-stage passes (select) are NOT added here. If MLIR-level kernelization
///  is needed, assemble the pipeline manually. `compile_program` is a
///  convenience for the common PR-level path.
pub const CompileConfig = struct {
    entry_name: []const u8 = "main",
    transform: Transform = .forward,
    lower: lower.LowerPassConfig = .{},
    /// Pre-computed tuning decisions. When set, the pipeline adds a
    ///  `KernelizePass` that rewrites annotated regions matching profitable
    ///  store entries into `custom_call` ops.
    kernel_store: ?*const kernel.KernelStore = null,
    dump_pr: ?pipeline.DumpConfig = null,
    dump_mlir: ?pipeline.DumpConfig = null,
    /// Dump the backend-optimized program (requires backend support).
    dump_optimized: ?pipeline.DumpConfig = null,
    /// Print a summary table of kernelized regions after the pass.
    dump_kernels: bool = false,
    compile: Backend.CompileOptions = .{},
};

/// Compiled executable with arity metadata.
///
/// Result of `compile`. Pass to `TrainState.init_from_model` for training
///  loops, or use `backend.execute` directly for inference.
/// Arity fields record the expected flat input/output counts.
pub const CompiledModel = struct {
    exe: Backend.Executable,
    input_arity: usize,
    output_arity: usize,

    /// Per-input donation mask, matching flattened spec order.
    ///
    /// Donatable inputs may have their buffers reused by the backend for
    ///  outputs. In a training loop this distinguishes owned parameters
    ///  (donatable, swapped each step) from borrowed batch data.
    donatable: []const bool,
    allocator: std.mem.Allocator,

    /// Number of donatable inputs.
    pub fn donatable_count(self: CompiledModel) usize {
        var count: usize = 0;
        for (self.donatable) |d| {
            if (d) count += 1;
        }
        return count;
    }

    /// Indices of non-donatable inputs (for backend execute calls).
    pub fn non_donatable_indices(self: CompiledModel, allocator: std.mem.Allocator) ![]const i64 {
        var list = try std.ArrayList(i64).initCapacity(allocator, self.input_arity);
        errdefer list.deinit(allocator);
        for (self.donatable, 0..) |d, i| {
            if (!d) try list.append(allocator, @intCast(i));
        }
        return list.toOwnedSlice(allocator);
    }

    pub fn deinit(self: *CompiledModel) void {
        self.allocator.free(self.donatable);
    }
};

/// Trace a comptime function against abstract specs, optionally transform it,
///  and compile through the pipeline into a backend executable.
///
/// `func` is a comptime-known function whose parameters match the structure
///   of `specs`. Each Tensor leaf in `specs` becomes a traced parameter.
///   Struct/tuple nesting in `specs` is preserved, the function receives the
///   same structure with traced Tensors in place of abstract ones.
///
/// `config.transform` controls AD application:
///  - `.forward`: compile the function as traced (no AD).
///  - `.value_and_grad`: apply VJP at compile time. `func` must return a
///     single scalar Tensor. The compiled program returns
///     `[loss, grad_0, ..., grad_n]` where grads are w.r.t. all spec leaves.
///
/// For in-graph optimizer composition (the common training path), use
///  `transforms.value_and_grad` inside `func` and compile with `.forward`.
///
/// The donation mask on each spec Tensor (`donatable` field) flows into the
///  returned `CompiledModel` and controls buffer reuse during execution.
/// TODO: jvp
pub fn compile(
    comptime func: anytype,
    allocator: std.mem.Allocator,
    backend_handle: *Backend,
    device: Backend.Device,
    specs: anytype,
    config: CompileConfig,
) !CompiledModel {
    const SpecType = @TypeOf(specs);

    var program = pr.Program.init(allocator);
    defer program.deinit();

    // Flatten specs into a tree. Arena-allocated, freed with the program.
    var spec_tree = try Tree(Tensor).from(program.allocator(), specs);
    const leaf_count = spec_tree.len();

    // Build donation mask from spec leaves. Caller-owned (outlives program).
    const donatable = try allocator.alloc(bool, leaf_count);
    errdefer allocator.free(donatable);
    for (spec_tree.leaves, 0..) |spec, i| {
        donatable[i] = spec.donatable;
    }

    switch (config.transform) {
        .forward => {
            var builder = try pr.FunctionBuilder.init(&program, config.entry_name);
            defer builder.deinit();

            const output_tensors = try trace_and_call(func, SpecType, &spec_tree, &builder, allocator);
            defer allocator.free(output_tensors);
            if (output_tensors.len == 0) return error.NoOutputs;

            const vars = try allocator.alloc(*pr.Var, output_tensors.len);
            defer allocator.free(vars);
            for (output_tensors, 0..) |t, i| vars[i] = try t.get_var();

            const func_pr = try builder.finish(vars);
            try program.add_function(func_pr);

            const exe = try compile_program(backend_handle, allocator, &program, device, config, config.entry_name);
            return .{
                .exe = exe,
                .input_arity = leaf_count,
                .output_arity = output_tensors.len,
                .donatable = donatable,
                .allocator = allocator,
            };
        },
        .value_and_grad => {
            // trace loss function
            var loss_builder = try pr.FunctionBuilder.init(&program, "loss");
            defer loss_builder.deinit();

            const output_tensors = try trace_and_call(func, SpecType, &spec_tree, &loss_builder, allocator);
            defer allocator.free(output_tensors);
            if (output_tensors.len != 1) return error.UnexpectedOutputs;

            const loss_var = try output_tensors[0].get_var();
            const loss_func = try loss_builder.finish(&.{loss_var});

            // register loss function for IR debuggability (see transforms.zig).
            // never called at runtime, the VJP replays forward equations internally.
            // TODO: open design question, this is literally creating dead+duplicated code in ir dumps.
            try program.add_function(loss_func);

            // VJP transform
            const vjp_func = try ad.vjp_with_value(program.allocator(), &program, loss_func, "loss_vjp");
            try program.add_function(vjp_func);

            // build step function: primals -> loss_vjp(primals, cotangent) -> [value, grads...]
            var step_builder = try pr.FunctionBuilder.init(&program, config.entry_name);
            defer step_builder.deinit();

            const primals = try allocator.alloc(*pr.Var, leaf_count);
            defer allocator.free(primals);
            for (spec_tree.leaves, 0..) |spec, i| {
                primals[i] = try step_builder.param_tensor(spec.dtype, spec.shape.const_slice());
            }

            // emit cotangent (ones_like for the scalar loss)
            const loss_t = output_tensors[0];
            const cot = try ad.emit_cotangent(&step_builder, .{
                .dtype = loss_t.dtype,
                .shape = .{ .dims = loss_t.dims() },
            });

            const call_inputs = try allocator.alloc(*pr.Var, leaf_count + 1);
            defer allocator.free(call_inputs);
            @memcpy(call_inputs[0..leaf_count], primals);
            call_inputs[leaf_count] = cot;

            const call_outputs = try step_builder.call("loss_vjp", call_inputs);
            // vjp_with_value returns: [value, grad_0, ..., grad_n]
            if (call_outputs.len != leaf_count + 1) return error.UnexpectedOutputs;

            const step_func = try step_builder.finish(call_outputs);
            try program.add_function(step_func);

            const exe = try compile_program(backend_handle, allocator, &program, device, config, config.entry_name);
            return .{
                .exe = exe,
                .input_arity = leaf_count,
                .output_arity = call_outputs.len,
                .donatable = donatable,
                .allocator = allocator,
            };
        },
    }
}

/// Trace a comptime function against a spec tree.
///
/// Creates traced parameters from each spec leaf via the given builder,
///  reconstructs the structured type, and calls `func`. Returns the
///  flattened output tensors.
///
/// Both `compile` branches use this to avoid duplicating the
///  "map specs -> trace params -> call func -> collect outputs" sequence.
fn trace_and_call(
    comptime func: anytype,
    comptime SpecType: type,
    spec_tree: *const Tree(Tensor),
    builder: *pr.FunctionBuilder,
    allocator: std.mem.Allocator,
) ![]Tensor {
    // map abstract specs to traced parameters in the builder
    // arena-allocated via spec_tree's allocator (the program arena)
    var traced = try spec_tree.map(Tensor, builder, struct {
        fn f(b: *pr.FunctionBuilder, spec: Tensor) anyerror!Tensor {
            return Tensor.param(b, spec.dtype, spec.shape.const_slice());
        }
    }.f);
    _ = &traced; // arena-managed, no manual deinit needed

    const structured = traced.extract(SpecType);
    const result_raw = if (@typeInfo(SpecType) == .@"struct" and @typeInfo(SpecType).@"struct".is_tuple)
        @call(.auto, func, structured)
    else
        @call(.auto, func, .{structured});
    const result = switch (@typeInfo(@TypeOf(result_raw))) {
        .error_union => try result_raw,
        else => result_raw,
    };

    return flatten_outputs(allocator, result);
}

/// TODO: this does not belong here
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

/// Assemble and run the compilation pipeline: kernelize -> lower -> compile.
///
/// Builds a pass chain from `config`, runs it over `program`, and hands the
///  resulting MLIR to the PJRT backend for compilation. Returns a loaded
///  executable ready for `backend.execute()`.
///
/// Pipeline: `[dump_pr] -> [kernelize(store)] -> validate -> lower -> legalize -> [dump_mlir]`
///
/// This is a convenience for the common PR-level kernelization path. MLIR-stage
///  passes (select) are not added here -- for MLIR-level kernelization,
///  assemble the pipeline manually.
///
/// NOTE: in reality, this is simply a default pipeline. we should not provide
///  a default pipeline in this form, this is being staged for removal.
pub fn compile_program(
    backend_handle: *Backend,
    allocator: std.mem.Allocator,
    program: *pr.Program,
    device: Backend.Device,
    config: CompileConfig,
    entry_name: []const u8,
) !Backend.Executable {
    var lower_cfg = config.lower;
    if (lower_cfg.entry_name == null) lower_cfg.entry_name = entry_name;

    var passes = std.ArrayList(pipeline.Pass).initCapacity(allocator, 6) catch
        return error.OutOfMemory;
    defer passes.deinit(allocator);

    var dump_pr_local: ?pipeline.DumpConfig = null;
    if (config.dump_pr) |cfg| {
        dump_pr_local = cfg;
        dump_pr_local.?.entry_name = dump_pr_local.?.entry_name orelse entry_name;
        try passes.append(allocator, pipeline.dump_pr_pass_with_config(&dump_pr_local.?));
    }

    var kernelize_state: ?pipeline.KernelizePass = null;
    if (config.kernel_store) |store| {
        kernelize_state = .{
            .store = store,
            .dump_kernels = config.dump_kernels,
        };
        try passes.append(allocator, kernelize_state.?.pass());
    }

    try passes.append(allocator, pipeline.validate_pass);
    try passes.append(allocator, lower.lower_pass_with_config(&lower_cfg));
    try passes.append(allocator, lower.mlir.stablehlo.StablehloLegalizePass.pass());

    var dump_mlir_local: ?pipeline.DumpConfig = null;
    if (config.dump_mlir) |cfg| {
        dump_mlir_local = cfg;
        dump_mlir_local.?.entry_name = dump_mlir_local.?.entry_name orelse entry_name;
        try passes.append(allocator, pipeline.dump_mlir_pass_with_config(&dump_mlir_local.?));
    }

    const pipeline_run = pipeline.Pipeline{ .passes = passes.items };
    var ctx = pipeline.PassContext{ .allocator = allocator };

    var artifact = try pipeline_run.run(.{ .pr = program }, &ctx);
    defer artifact.deinit(allocator);

    const mlir = switch (artifact) {
        .mlir => |m| m,
        else => return error.UnexpectedArtifact,
    };

    const compile_opts = config.compile;
    const exe = try backend_handle.compile(device, mlir.bytes, mlir.encoding == .bytecode, compile_opts);

    if (config.dump_optimized != null) {
        log.warn("dump-optimized requires PJRT-specific API; skipped through generic backend", .{});
    }

    return exe;
}

// ============================================================================
// Comptime introspection helpers (Tensor-leaf trees)
// ============================================================================
// TODO: might want to move Tree or rethink this organization
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
