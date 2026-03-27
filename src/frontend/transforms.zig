//! Compile-time function transforms for Tensor-valued functions.
//!
//! Transforms augment a function's behavior *during tracing*. They are called
//!  inside a traced function body, not at the compile call site. This is the
//!  key difference from `frontend.compile(..., .{ .transform = .value_and_grad })`,
//!  which applies VJP as a compilation step outside the trace.
//!
//! The canonical use is `value_and_grad`, which:
//!  1. Builds a sub-function from the loss closure,
//!  2. Applies `pr.ad.vjp_with_value` to produce a VJP function,
//!  3. Emits a call to the VJP function in the *outer* builder,
//!  4. Returns both the loss value and a `Tree(Tensor)` of gradients.
//!
//! Because the transform runs inside the trace, the caller can compose it
//!  with arbitrary traced logic (optimizer updates, gradient clipping, etc.)
//!  in a single compiled program:
//!
//! ```zig
//! fn train_step(params: Params, batch: Batch) !TrainResult {
//!     var vg = try transforms.value_and_grad(loss_fn, .{ params, batch });
//!     defer vg.deinit();
//!     // Apply optimizer in the same trace:
//!     var updated = try params_tree.map2(..., sgd_update);
//!     return .{ .loss = vg.value, .updated = updated.extract(Params) };
//! }
//! // Compile the whole step as one program:
//! var compiled = try frontend.compile(train_step, alloc, backend, device, specs, .{});
//! ```
//!
//! Both this module and `frontend.compile(.value_and_grad)` use `pr.ad.vjp_with_value`
//!  under the hood. The difference is where VJP is applied (inside the trace vs. at
//!  compile time) and what the compiled program returns (user-defined struct vs. flat
//!  `[loss, grads...]`).
const std = @import("std");
const pr = @import("../pr/pr.zig");
const ad = @import("../pr/ad.zig");
const ops = @import("../pr/ops/ops.zig");
const tree_mod = @import("../utils/tree.zig");
const Tree = tree_mod.Tree;

const Tensor = @import("../tensor.zig");

/// Result of a `value_and_grad` call during tracing.
///
/// Use `.grads.extract(ParamsType)` to recover the named struct, or
/// `.grads.leaves` for bulk operations like `map2` with an optimizer.
pub const ValueAndGrad = struct {
    /// A traced Tensor representing the scalar loss.
    value: Tensor,
    /// A `Tree(Tensor)` with one leaf per parameter in the first argument
    ///  to the loss function. Leaf paths mirror the struct field names.
    grads: Tree(Tensor),

    pub fn deinit(self: *ValueAndGrad) void {
        self.grads.deinit();
    }
};

/// Trace `func` and compute both its return value and gradients w.r.t. the
///  first argument.
///
/// Must be called *during tracing* -- all Tensor arguments must be traced-mode
///  (bound to a FunctionBuilder). This is not a standalone compilation entry
///  point, use `frontend.compile` for that.
///
/// ## Mechanism
///
///  1. Flattens `args` into a `Tree(Tensor)` and collects their VarIds.
///  2. Builds a sub-function by tracing `func` with fresh parameters matching
///      the input specs.
///  3. Applies `pr.ad.vjp_with_value` to produce a VJP function that returns
///      `[loss_value, grad_0, ..., grad_n]`.
///  4. Emits a `call` to the VJP function in the *outer* builder with a
///      ones-like cotangent seed.
///  5. Returns the loss `Tensor` and a `Tree(Tensor)` of gradients for the
///      first argument (params).
///
/// ## Constraints
///
///  - `func` must return a single scalar Tensor (the loss).
///  - `args` must be a tuple. The first element is the "params" argument,
///      gradients are computed w.r.t. its leaves only. Remaining arguments
///      (e.g. batch data) participate in the forward pass but receive no grads.
///
/// ```zig
/// fn train_step(params: Params, batch: Batch) !struct { loss: Tensor, updated: Params } {
///     var vg = try transforms.value_and_grad(loss_fn, .{ params, batch });
///     defer vg.deinit();
///     var updated = try params_tree.map2(Tensor, &vg.grads, Tensor, lr, sgd_leaf);
///     defer updated.deinit();
///     return .{ .loss = vg.value, .updated = updated.extract(Params) };
/// }
/// ```
pub fn value_and_grad(comptime func: anytype, args: anytype) !ValueAndGrad {
    const ArgsType = @TypeOf(args);
    const ParamsType = ArgsParamsType(ArgsType);
    const param_leaf_count = comptime Tree(Tensor).leaf_count(ParamsType);

    // Extract builder from first tensor leaf in args.
    const builder = extract_builder(args) orelse @panic("no Tensor found in args");
    const program = builder.program;
    const alloc = program.allocator();

    // Flatten args into a tree. Program allocator is an arena -- intermediate trees
    //  are freed in bulk when the program is destroyed.
    var args_tree = try Tree(Tensor).from(alloc, args);

    // Collect VarIds from the input tensors.
    const input_ids = try alloc.alloc(pr.VarId, args_tree.leaves.len);
    for (args_tree.leaves, 0..) |t, i| {
        input_ids[i] = try t.get_id();
    }

    // Build sub-function for the loss: create traced params matching the
    // spec shapes, reconstruct the structured args, and call func.
    const loss_name = "vg_loss";
    var loss_builder = try pr.FunctionBuilder.init(program, loss_name);
    defer loss_builder.deinit();

    var sub_tree = try args_tree.map(Tensor, &loss_builder, struct {
        fn f(b: *pr.FunctionBuilder, spec: Tensor) anyerror!Tensor {
            return Tensor.param(b, spec.dtype, spec.shape.const_slice());
        }
    }.f);

    const loss_result = @call(.auto, func, sub_tree.extract(ArgsType));
    const loss_tensor = switch (@typeInfo(@TypeOf(loss_result))) {
        .error_union => try loss_result,
        else => loss_result,
    };

    const loss_id = try loss_tensor.get_id();
    const loss_func = try loss_builder.finish(&.{loss_id});

    // Register the loss function in the program. The VJP function replays the forward
    //  equations internally (it needs intermediates for the backward pass), so this
    //  function is never called at runtime. We keep it in the program for IR
    //  debuggability (eg MLIR dumps show the clean forward pass as a readable
    //  reference alongside the larger VJP function).
    try program.add_function(loss_func);

    // Apply VJP.
    const vjp_name = "vg_loss_vjp";
    const vjp_func = try ad.vjp_with_value(alloc, program, loss_func, vjp_name);
    try program.add_function(vjp_func);

    // Emit cotangent (ones_like for scalar loss) in the OUTER builder.
    const loss_aval = loss_func.avals[@intCast(loss_id)].as_tensor() orelse
        return error.UnsupportedAval;
    const cot = try ad.emit_cotangent(builder, loss_aval);

    // Call VJP function from outer builder.
    const total_leaf_count = args_tree.leaves.len;
    const call_args = try alloc.alloc(pr.VarId, total_leaf_count + 1);
    @memcpy(call_args[0..total_leaf_count], input_ids[0..total_leaf_count]);
    call_args[total_leaf_count] = cot;

    const call_outputs = try builder.call(vjp_name, call_args);
    // vjp_with_value returns: [value, grad_0, ..., grad_n]
    if (call_outputs.len != total_leaf_count + 1) return error.UnexpectedOutputs;

    // Extract value tensor.
    const value_tensor = try Tensor.from_id(builder, call_outputs[0]);

    // Extract grads for the first argument (params) into a Tree.
    const grad_leaves = try alloc.alloc(Tensor, param_leaf_count);
    errdefer alloc.free(grad_leaves);
    for (call_outputs[1 .. 1 + param_leaf_count], 0..) |gid, i| {
        grad_leaves[i] = try Tensor.from_id(builder, gid);
    }

    const grad_paths = try alloc.alloc([]const u8, param_leaf_count);
    errdefer alloc.free(grad_paths);
    const comptime_paths = comptime tree_mod.tree_paths(Tensor, ParamsType);
    @memcpy(grad_paths, &comptime_paths);

    return .{
        .value = value_tensor,
        .grads = Tree(Tensor).from_slices(alloc, grad_leaves, grad_paths),
    };
}

// ============================================================================
// Internal helpers
// ============================================================================

fn ArgsParamsType(comptime ArgsType: type) type {
    const info = @typeInfo(ArgsType);
    if (info != .@"struct" or !info.@"struct".is_tuple or info.@"struct".fields.len == 0) {
        @compileError("args must be a tuple with at least one element");
    }
    // TODO: incomplete, need to consider semantics we want
    return info.@"struct".fields[0].type;
}

/// Extract the FunctionBuilder pointer from the first Tensor leaf in a structured value.
fn extract_builder(val: anytype) ?*pr.FunctionBuilder {
    const T = @TypeOf(val);
    if (T == Tensor) return val.mode.traced.builder;
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            inline for (info.fields) |field| {
                if (extract_builder(@field(val, field.name))) |b| return b;
            }
        },
        .array => |info| {
            comptime var i: usize = 0;
            inline while (i < info.len) : (i += 1) {
                if (extract_builder(val[i])) |b| return b;
            }
        },
        else => {},
    }
    return null;
}
