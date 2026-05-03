//! Frontend: user-facing API for tracing and compiling programs.
//!
//! The frontend sits between user code and the PR/pipeline/backend layers.
//!
//! 1. **`trace`**: Trace a comptime function against abstract Tensor specs,
//!    producing a `pr.Program`. The user controls what the function does --
//!    including composing AD transforms via `value_and_grad` inside the trace.
//!
//! 2. **`compile_program`**: Run a pipeline (validate -> lower -> compile) on
//!    a program and hand the MLIR to the backend. Returns a backend executable.
//!
//! 3. **`transforms`**: AD function transforms, both trace-time combinators
//!    (e.g. `transforms.value_and_grad`) and comptime generators
//!    (e.g. `zg.grad`, `zg.value_and_grad`).
//!
//! 4. **`optim`**: Traced-mode optimizer building blocks (e.g. `SGD`).
//!
//! ## Typical usage
//!
//! **Composing AD + optimizer in one program** (trace-time combinator):
//! ```zig
//! fn train_step(params: Params, batch: Batch) !struct { loss: Tensor, updated: Params } {
//!     var vg = try zg.frontend.transforms.value_and_grad(loss_fn, .{ params, batch });
//!     defer vg.deinit();
//!     // ... apply optimizer to vg.grads, return updated params
//! }
//! var program = try zg.trace(train_step, allocator, specs, "train_step");
//! ```
//!
//! **Just gradients as outputs** (comptime generator):
//! ```zig
//! const grad_fn = comptime zg.grad(loss_fn, .{});
//! var program = try zg.trace(grad_fn, allocator, specs, "grad_step");
//! ```
//!
//! **Then compile either**:
//! ```zig
//! defer program.deinit();
//! var exe = try zg.frontend.compile_program(backend, allocator, &program, device, "name", .{});
//! defer backend.deinit_executable(exe);
//! ```
const std = @import("std");

const pr = @import("../pr/pr.zig");
const kernel = @import("../kernel.zig");
const lower = @import("../lower.zig");
const pipeline = @import("../pipeline.zig");
const Backend = @import("../Backend.zig");
const Tensor = @import("../tensor.zig");
const TensorTree = @import("../utils.zig").Tree(Tensor);

/// Typed compiled function wrapper (trace + compile + typed call).
pub const jit_mod = @import("jit.zig");
/// Training loop state management (buffer swap, donation, execute).
pub const train = @import("train.zig");
/// Comptime function transforms for traced Tensor programs (e.g. `value_and_grad`).
pub const transforms = @import("transforms.zig");
/// Traced-mode optimizer building blocks (e.g. `optim.SGD`).
pub const optim = @import("optim.zig");

/// Create a typed compiled function from a comptime trace function.
///
/// This is the highest-level API (Layer 1). It combines `trace()` and
///  `compile_program()` into a single callable that preserves the original
///  function's structured input/output types.
///
/// For manual control, use `trace()` + `compile_program()` (Layer 2).
pub const jit = jit_mod.jit;

const log = std.log.scoped(.@"zg/frontend");

// ============================================================================
// Trace
// ============================================================================

/// Trace a comptime function against Tensor specs.
///
/// Only `dtype` and `shape` are read from each leaf, so `specs` may
///  contain Tensors of **any** backing variant. I.e., duck typed, so
///  `abstract`, `host`, or `device` all work).
/// In practice you'll pass either:
/// - A tuple of `Tensor.abstract(...)` values when you haven't loaded
///    any data yet (the "spec" pattern), this is dtype+shape. Or,
/// - A tuple of already-loaded host Tensors when you want to skip the
///    abstract pass entirely, (this is dtype+shape+data).
///
/// Each Tensor leaf becomes a traced parameter. Struct/tuple nesting is
///  preserved, the function receives the same structure with traced
///  Tensors in place of the spec ones. Ownership of `specs` is not taken.
///
/// The traced function can compose AD transforms internally (e.g. call
///  `transforms.value_and_grad`). Whatever it returns becomes the program's
///  outputs.
///
/// Returns a `pr.Program`. Use `program.get_function(name)` to access
///  functions, `program.output_arity(name)` for leaf counts.
///  Pass `&program` to `compile_program` for compilation.
pub fn trace(
    comptime func: anytype,
    allocator: std.mem.Allocator,
    specs: anytype,
    entry_name: []const u8,
) !pr.Program {
    const SpecType = @TypeOf(specs);

    var program = pr.Program.init(allocator);
    errdefer program.deinit();

    const spec_tree = try TensorTree.from(program.allocator(), specs);

    var builder = try pr.FunctionBuilder.init(&program, entry_name);
    defer builder.deinit();

    // Map abstract specs to traced parameters in the builder.
    const traced = try spec_tree.map(Tensor, &builder, struct {
        fn f(b: *pr.FunctionBuilder, leaf: Tensor) !Tensor {
            return try Tensor.param(b, leaf.dtype, leaf.shape.const_slice());
        }
    }.f);

    // Call the user function with the structured traced tensors.
    const structured = try traced.extract(SpecType);
    const result_raw = if (@typeInfo(SpecType) == .@"struct" and @typeInfo(SpecType).@"struct".is_tuple)
        @call(.auto, func, structured)
    else
        @call(.auto, func, .{structured});
    const result = switch (@typeInfo(@TypeOf(result_raw))) {
        .error_union => try result_raw,
        else => result_raw,
    };

    const output_tensors = try TensorTree.flatten(allocator, result);
    defer allocator.free(output_tensors);
    if (output_tensors.len == 0) return error.NoOutputs;

    const output_vars = try allocator.alloc(*pr.Var, output_tensors.len);
    defer allocator.free(output_vars);
    for (output_tensors, 0..) |t, i| output_vars[i] = try t.get_var();

    const func_pr = try builder.finish(output_vars);
    try program.add_function(func_pr);

    return program;
}

// ============================================================================
// Compile
// ============================================================================

/// Options for `compile_program`.
///
/// Controls kernelization, lowering, and diagnostic dump points.
/// The pipeline is a pass chain:
///  `[dump_pr] -> [kernelize(store)] -> validate -> lower -> legalize -> [dump_mlir]`
pub const CompileOpts = struct {
    lower: lower.LowerPassConfig = .{},
    /// Pre-computed tuning decisions. When set, the pipeline adds a
    ///  `KernelizePass` that rewrites annotated regions into `custom_call` ops.
    kernel_store: ?*const kernel.KernelStore = null,
    dump_pr: ?pipeline.DumpConfig = null,
    dump_mlir: ?pipeline.DumpConfig = null,
    dump_optimized: ?pipeline.DumpConfig = null,
    dump_kernels: bool = false,
    compile: Backend.CompileOptions = .{},
};

/// Assemble and run the compilation pipeline, then compile via backend.
///
/// Pipeline: `[dump_pr] -> [kernelize(store)] -> validate -> lower -> legalize -> [dump_mlir]`
///
/// This is a convenience for the common PR-level path. For MLIR-level
///  kernelization or custom pass chains, assemble the pipeline manually.
pub fn compile_program(
    backend: *Backend,
    io: std.Io,
    allocator: std.mem.Allocator,
    program: *pr.Program,
    device: Backend.Device,
    /// TODO: need to determine if setting entry_name has any utility, really. Intention
    ///  was wrt IR debuggability for programs that compile more than one thing. Need to
    ///  fully specify this use case and likely keep entry_name as a default in an opts
    ///  struct since users almost never need to customize this.
    entry_name: []const u8,
    opts: CompileOpts,
) !Backend.Executable {
    var lower_cfg = opts.lower;
    if (lower_cfg.entry_name == null) lower_cfg.entry_name = entry_name;

    var passes = try std.ArrayList(pipeline.Pass).initCapacity(allocator, 6);
    defer passes.deinit(allocator);

    var dump_pr_local: ?pipeline.DumpConfig = null;
    if (opts.dump_pr) |cfg| {
        dump_pr_local = cfg;
        dump_pr_local.?.entry_name = dump_pr_local.?.entry_name orelse entry_name;
        try passes.append(allocator, pipeline.dump_pr_pass_with_config(&dump_pr_local.?));
    }

    var kernelize_state: ?pipeline.KernelizePass = null;
    if (opts.kernel_store) |store| {
        kernelize_state = .{
            .store = store,
            .dump_kernels = opts.dump_kernels,
        };
        try passes.append(allocator, kernelize_state.?.pass());
    }

    try passes.append(allocator, pipeline.validate_pass);
    // TODO: this highlights one of the other issues with this function where it crosses abstraction
    //   boundary an directly biases a lowering path. Introduces a hard dependency that, in reality,
    //   we abstracted over.
    try passes.append(allocator, lower.lower_pass_with_config(&lower_cfg));
    try passes.append(allocator, lower.mlir.stablehlo.StablehloLegalizePass.pass());

    var dump_mlir_local: ?pipeline.DumpConfig = null;
    if (opts.dump_mlir) |cfg| {
        dump_mlir_local = cfg;
        dump_mlir_local.?.entry_name = dump_mlir_local.?.entry_name orelse entry_name;
        try passes.append(allocator, pipeline.dump_mlir_pass_with_config(&dump_mlir_local.?));
    }

    const pipeline_run = pipeline.Pipeline{ .passes = passes.items };
    var ctx = pipeline.PassContext{ .allocator = allocator, .io = io };

    var artifact = try pipeline_run.run(.{ .pr = program }, &ctx);
    defer artifact.deinit(allocator);

    // TODO: instead of this, we should extend Backend interface using the Pipeline / Pass pattern
    //   of declaring supported formats. No reason to mention a specific dialect here.
    const sh = switch (artifact) {
        .stablehlo => |s| s,
        else => return error.UnexpectedArtifact,
    };

    const compile_opts = opts.compile;
    const exe = try backend.compile(device, sh.bytes, sh.encoding, compile_opts);

    if (opts.dump_optimized) |cfg| {
        var dump_cfg = cfg;
        if (dump_cfg.entry_name == null) dump_cfg.entry_name = entry_name;
        if (try backend.get_optimized_program(exe, allocator)) |opt_prog| {
            var owned = opt_prog;
            defer owned.deinit(allocator);
            pipeline.dump_optimized_program(io, &dump_cfg, owned.code, owned.format, allocator) catch |err| {
                log.err("dump-optimized failed: {s}", .{@errorName(err)});
            };
        } else {
            log.warn("dump-optimized: backend does not expose an optimized program", .{});
        }
    }

    return exe;
}
