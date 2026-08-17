//! Compile-time function transforms for Tensor-valued functions.
//!
//! Transforms augment a function's behavior *during tracing*. They are called
//!  inside a traced function body, not at the trace call site.
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
//!     var vg = try transforms.value_and_grad(loss_fn, .{ params, batch }, .{});
//!     defer vg.deinit();
//!     // Apply optimizer in the same trace:
//!     var updated = try params_tree.map2(..., sgd_update);
//!     return .{ .loss = vg.value, .updated = updated.extract(Params) };
//! }
//! // Trace and compile the whole step as one program:
//! var program = try zg.trace(train_step, allocator, specs, "train_step");
//! defer program.deinit();
//! const exe = try compile_program(&program, "train_step");
//! ```
const std = @import("std");
const pr = @import("pr/pr.zig");
const ad = @import("pr/ad.zig");
const ops = @import("pr/ops/ops.zig");
const utils = @import("utils.zig");
const meta = utils.meta;
const Tree = utils.Tree;

const Tensor = @import("tensor.zig");

/// Result of a `value_and_grad` call during tracing.
///
/// Use `.grads.extract(SelectedType)` to recover the selected structure, or
/// `.grads.leaves` for bulk operations like `map2` with an optimizer.
pub const ValueAndGrad = struct {
    /// A traced Tensor representing the scalar loss.
    value: Tensor,
    /// A `Tree(Tensor)` containing the selected argument gradients.
    ///  One selection mirrors that argument. Multiple selections form a tuple.
    grads: Tree(Tensor),

    pub fn deinit(self: *ValueAndGrad) void {
        self.grads.deinit();
    }
};

/// Trace `func` and compute its return value and selected argument gradients.
///
/// Must be called during tracing.
///
/// All Tensor arguments must be bound to a `FunctionBuilder`. Use
///  `trace` and explicit compilation operations.
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
///      selected arguments.
///
/// ## Constraints
///
///  - `func` must return a single scalar Tensor (the loss).
///  - `args` must be a nonempty tuple. `opts.wrt_argnums` selects tuple
///      elements, and gradients are returned for their Tensor leaves.
///
/// ```zig
/// fn train_step(params: Params, batch: Batch) !struct { loss: Tensor, updated: Params } {
///     var vg = try transforms.value_and_grad(loss_fn, .{ params, batch }, .{});
///     defer vg.deinit();
///     var updated = try params_tree.map2(Tensor, &vg.grads, Tensor, lr, sgd_leaf);
///     defer updated.deinit();
///     return .{ .loss = vg.value, .updated = updated.extract(Params) };
/// }
/// ```
pub fn value_and_grad(comptime func: anytype, args: anytype, comptime opts: GradOpts) !ValueAndGrad {
    const ArgsType = @TypeOf(args);
    const GradsType = selected_grads_type(ArgsType, opts.wrt_argnums);
    const grad_leaf_count = comptime Tree(Tensor).leaf_count(GradsType);

    // Extract builder from first tensor leaf in args.
    const builder = extract_builder(args) orelse @panic("no Tensor found in args");
    const program = builder.program;
    const alloc = program.allocator();

    // Flatten arguments into a tree.
    //
    // The program arena releases intermediate trees with the program.
    var args_tree = try Tree(Tensor).from(alloc, args);

    // Collect Var pointers from the input tensors.
    const input_vars = try alloc.alloc(*pr.Var, args_tree.leaves.len);
    for (args_tree.leaves, 0..) |t, i| {
        input_vars[i] = try t.get_var();
    }

    // Build sub-function for the loss: create traced params matching the
    // spec shapes, reconstruct the structured args, and call func.
    const loss_name = "vg_loss";
    var loss_builder = try pr.FunctionBuilder.init(program, loss_name);
    defer loss_builder.deinit();

    var sub_tree = try args_tree.map(Tensor, &loss_builder, struct {
        fn f(b: *pr.FunctionBuilder, spec: Tensor) !Tensor {
            return try Tensor.param(b, spec.dtype, spec.shape.const_slice());
        }
    }.f);

    const loss_result: anyerror!Tensor = @call(.auto, func, try sub_tree.extract(ArgsType));
    const loss_tensor: Tensor = switch (@typeInfo(@TypeOf(loss_result))) {
        .error_union => try loss_result,
        else => loss_result,
    };

    const loss_var = try loss_tensor.get_var();
    const loss_func = try loss_builder.finish(&.{loss_var});

    // Register the loss function in the program. The VJP function replays the forward
    //  equations internally (it needs intermediates for the backward pass), so this
    //  function is never called at runtime. We keep it in the program for IR
    //  debuggability (eg MLIR dumps show the clean forward pass as a readable
    //  reference alongside the larger VJP function).
    // TODO(ad): Call the registered loss function from VJP instead of replaying
    //  its operations.
    try program.add_function(loss_func);

    // Request gradients only for the selected argument's leaves.
    //  `wrt` filters the VJP function's output list: cotangents for
    //  non-`wrt` inputs are omitted from the return
    //  signature, so the VJP function returns exactly `grad_leaf_count`
    //  gradients. Any intermediate cotangents that only fed omitted outputs
    //  become dead and are cleaned up by the backend's DCE.
    const wrt_indices = comptime selected_leaf_indices(ArgsType, opts.wrt_argnums);
    const vjp_name = "vg_loss_vjp";
    const vjp_func = try ad.vjp_with_value(alloc, program, loss_func, vjp_name, .{ .wrt = &wrt_indices });
    try program.add_function(vjp_func);

    // Emit cotangent (ones_like for scalar loss) in the OUTER builder.
    const cot = try ad.emit_cotangent(builder, loss_var.as_tensor());

    // Call VJP function from outer builder. The VJP takes every primal
    //  input (not just params) plus the loss cotangent seed.
    const total_leaf_count = args_tree.leaves.len;
    const call_args = try alloc.alloc(*pr.Var, total_leaf_count + 1);
    @memcpy(call_args[0..total_leaf_count], input_vars[0..total_leaf_count]);
    call_args[total_leaf_count] = cot;

    const call_outputs = try builder.call(vjp_name, call_args);
    // vjp_with_value with `wrt` returns the value followed by selected gradients,
    //  where K == grad_leaf_count.
    if (call_outputs.len != grad_leaf_count + 1) return error.UnexpectedOutputs;

    // Extract value tensor
    const value_tensor = Tensor.from_var(builder, call_outputs[0]);

    // Extract selected gradients into a Tree.
    const grad_leaves = try alloc.alloc(Tensor, grad_leaf_count);
    for (call_outputs[1..], 0..) |gv, i| {
        grad_leaves[i] = Tensor.from_var(builder, gv);
    }

    const comptime_paths = comptime meta.tree_paths(Tensor, GradsType);

    return .{
        .value = value_tensor,
        .grads = try Tree(Tensor).from_slices(alloc, grad_leaves, &comptime_paths),
    };
}

// Comptime function generators
//
// These produce new comptime function pointers from existing functions.
// The returned functions can be passed to `zg.trace()`.
//
// Unlike the trace-time `value_and_grad` above (called inside a traced
//  function body), these are called at comptime to *generate* a function
//  that will itself be traced.

/// Options for Tensor-level differentiation transforms.
pub const GradOpts = struct {
    /// Zero-based function arguments whose Tensor leaves receive gradients.
    /// One selection preserves that argument's structure. Multiple selections
    ///  return a tuple in this order. Repeated argument numbers remain repeated.
    wrt_argnums: []const usize = &.{0},
};

/// Generate a function that computes gradients of `func` for selected arguments.
///
/// Returns a comptime function pointer with the same parameter types as `func`
///  and gradients populated into the structure described by `GradOpts`.
///  Pass the returned function to `zg.trace()`.
///
/// ```zig
/// const grad_fn = comptime zg.grad(loss_fn, .{});
/// var traced = try zg.trace(grad_fn, allocator, specs, "grad_step");
/// ```
pub fn make_grad(comptime func: anytype, comptime opts: GradOpts) @TypeOf(&GradCallGen(func, opts).call) {
    return &GradCallGen(func, opts).call;
}

/// Generate a function that computes both value and gradients of `func`.
///
/// Returns a comptime function pointer with the same parameter types as `func`.
///  Its result contains the value and gradients shaped according to `GradOpts`.
///
/// ```zig
/// const vg_fn = comptime zg.value_and_grad(loss_fn, .{});
/// var traced = try zg.trace(vg_fn, allocator, specs, "vg_step");
/// ```
pub fn make_value_and_grad(comptime func: anytype, comptime opts: GradOpts) @TypeOf(&VgCallGen(func, opts).call) {
    return &VgCallGen(func, opts).call;
}

/// Return type for comptime `value_and_grad` generated functions.
pub fn ValueAndGradResult(comptime GradsType: type) type {
    return struct { value: Tensor, grads: GradsType };
}

fn GradCallGen(comptime func: anytype, comptime opts: GradOpts) type {
    const params = @typeInfo(@TypeOf(func)).@"fn".params;
    const G = selected_grads_type(std.meta.ArgsTuple(@TypeOf(func)), opts.wrt_argnums);
    return switch (params.len) {
        1 => struct {
            pub fn call(a0: params[0].type.?) anyerror!G {
                return try grad_impl(func, G, .{a0}, opts);
            }
        },
        2 => struct {
            pub fn call(a0: params[0].type.?, a1: params[1].type.?) anyerror!G {
                return try grad_impl(func, G, .{ a0, a1 }, opts);
            }
        },
        3 => struct {
            pub fn call(a0: params[0].type.?, a1: params[1].type.?, a2: params[2].type.?) anyerror!G {
                return try grad_impl(func, G, .{ a0, a1, a2 }, opts);
            }
        },
        4 => struct {
            pub fn call(a0: params[0].type.?, a1: params[1].type.?, a2: params[2].type.?, a3: params[3].type.?) anyerror!G {
                return try grad_impl(func, G, .{ a0, a1, a2, a3 }, opts);
            }
        },
        else => @compileError("grad supports functions with up to 4 parameters"),
    };
}

fn VgCallGen(comptime func: anytype, comptime opts: GradOpts) type {
    const params = @typeInfo(@TypeOf(func)).@"fn".params;
    const G = selected_grads_type(std.meta.ArgsTuple(@TypeOf(func)), opts.wrt_argnums);
    const R = ValueAndGradResult(G);
    return switch (params.len) {
        1 => struct {
            pub fn call(a0: params[0].type.?) anyerror!R {
                return try vg_impl(func, G, .{a0}, opts);
            }
        },
        2 => struct {
            pub fn call(a0: params[0].type.?, a1: params[1].type.?) anyerror!R {
                return try vg_impl(func, G, .{ a0, a1 }, opts);
            }
        },
        3 => struct {
            pub fn call(a0: params[0].type.?, a1: params[1].type.?, a2: params[2].type.?) anyerror!R {
                return try vg_impl(func, G, .{ a0, a1, a2 }, opts);
            }
        },
        4 => struct {
            pub fn call(a0: params[0].type.?, a1: params[1].type.?, a2: params[2].type.?, a3: params[3].type.?) anyerror!R {
                return try vg_impl(func, G, .{ a0, a1, a2, a3 }, opts);
            }
        },
        else => @compileError("value_and_grad supports functions with up to 4 parameters"),
    };
}

fn grad_impl(comptime func: anytype, comptime GradsType: type, args: anytype, comptime opts: GradOpts) !GradsType {
    var vg = try value_and_grad(func, args, opts);
    defer vg.deinit();
    return try vg.grads.extract(GradsType);
}

fn vg_impl(comptime func: anytype, comptime GradsType: type, args: anytype, comptime opts: GradOpts) !ValueAndGradResult(GradsType) {
    var vg = try value_and_grad(func, args, opts);
    defer vg.deinit();
    return .{ .value = vg.value, .grads = try vg.grads.extract(GradsType) };
}

// Internal helpers

fn selected_grads_type(comptime ArgsType: type, comptime argnums: []const usize) type {
    const info = @typeInfo(ArgsType);
    if (info != .@"struct" or !info.@"struct".is_tuple or info.@"struct".fields.len == 0) {
        @compileError("args must be a tuple with at least one element");
    }
    if (argnums.len == 0) @compileError("wrt_argnums must not be empty");
    inline for (argnums) |argnum| {
        if (argnum >= info.@"struct".fields.len) @compileError("wrt_argnums contains an out-of-range argument");
    }
    if (argnums.len == 1) return info.@"struct".fields[argnums[0]].type;

    comptime var types: [argnums.len]type = undefined;
    inline for (argnums, 0..) |argnum, index| {
        types[index] = info.@"struct".fields[argnum].type;
    }
    return std.meta.Tuple(&types);
}

fn argument_leaf_offset(comptime ArgsType: type, comptime argument: usize) usize {
    const fields = @typeInfo(ArgsType).@"struct".fields;
    comptime var offset: usize = 0;
    inline for (fields[0..argument]) |field| {
        offset += Tree(Tensor).leaf_count(field.type);
    }
    return offset;
}

fn selected_leaf_indices(
    comptime ArgsType: type,
    comptime argnums: []const usize,
) [Tree(Tensor).leaf_count(selected_grads_type(ArgsType, argnums))]usize {
    const fields = @typeInfo(ArgsType).@"struct".fields;
    var indices: [Tree(Tensor).leaf_count(selected_grads_type(ArgsType, argnums))]usize = undefined;
    var cursor: usize = 0;
    inline for (argnums) |argnum| {
        const offset = comptime argument_leaf_offset(ArgsType, argnum);
        const count = comptime Tree(Tensor).leaf_count(fields[argnum].type);
        for (0..count) |index| {
            indices[cursor] = offset + index;
            cursor += 1;
        }
    }
    return indices;
}

/// Extract the FunctionBuilder pointer from the first Tensor leaf in a structured value.
fn extract_builder(val: anytype) ?*pr.FunctionBuilder {
    const T = @TypeOf(val);
    if (T == Tensor) return val.backing.traced.builder;
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

// Tests.

const trace = @import("trace.zig").trace;

const TestParams = struct { w: Tensor, b: Tensor };
const TestBatch = struct { x: Tensor };

fn test_loss(params: TestParams, batch: TestBatch) !Tensor {
    const z = try batch.x.mm(params.w);
    const b_broadcast = try params.b.broadcast_in_dim(&.{ 3, 2 }, &.{1});
    const pred = try z.add(b_broadcast);
    return try pred.reduce(.{ .axes = &.{ 0, 1 }, .operation = .sum });
}

test make_grad {
    const grad_fn = comptime make_grad(test_loss, .{});

    // Verify return type is the params type (TestParams).
    const RetType = @typeInfo(@TypeOf(grad_fn)).pointer.child;
    const ret_info = @typeInfo(RetType).@"fn";
    try std.testing.expectEqual(2, ret_info.params.len);

    const ReturnType = @typeInfo(ret_info.return_type.?).error_union.payload;
    try std.testing.expect(ReturnType == TestParams);

    // Trace the generated function into a complete PR program.
    const specs = .{
        TestParams{
            .w = Tensor.abstract(.f32, &.{ 4, 2 }),
            .b = Tensor.abstract(.f32, &.{2}),
        },
        TestBatch{
            .x = Tensor.abstract(.f32, &.{ 3, 4 }),
        },
    };

    var program = try trace(grad_fn, std.testing.allocator, specs, "grad_test");
    defer program.deinit();

    // grad returns only grads for first arg (2 leaves: w, b).
    try std.testing.expectEqual(2, program.output_arity("grad_test"));
    try std.testing.expectEqual(3, program.input_arity("grad_test"));
}

test "make_grad selects differentiated arguments in order" {
    const grad_fn = comptime make_grad(test_loss, .{ .wrt_argnums = &.{ 1, 0, 1 } });
    const function_type = @typeInfo(@TypeOf(grad_fn)).pointer.child;
    const return_type = @typeInfo(@typeInfo(function_type).@"fn".return_type.?).error_union.payload;
    const return_fields = @typeInfo(return_type).@"struct".fields;
    try std.testing.expectEqual(@as(usize, 3), return_fields.len);
    try std.testing.expect(return_fields[0].type == TestBatch);
    try std.testing.expect(return_fields[1].type == TestParams);
    try std.testing.expect(return_fields[2].type == TestBatch);

    const specs = .{
        TestParams{
            .w = Tensor.abstract(.f32, &.{ 4, 2 }),
            .b = Tensor.abstract(.f32, &.{2}),
        },
        TestBatch{
            .x = Tensor.abstract(.f32, &.{ 3, 4 }),
        },
    };
    var program = try trace(grad_fn, std.testing.allocator, specs, "batch_grad_test");
    defer program.deinit();

    const function = program.get_function("batch_grad_test").?;
    try std.testing.expectEqual(@as(usize, 4), function.returns.len);
    try std.testing.expectEqualSlices(i64, &.{ 3, 4 }, function.returns[0].as_tensor().shape.dims);
    try std.testing.expectEqualSlices(i64, &.{ 4, 2 }, function.returns[1].as_tensor().shape.dims);
    try std.testing.expectEqualSlices(i64, &.{2}, function.returns[2].as_tensor().shape.dims);
    try std.testing.expectEqualSlices(i64, &.{ 3, 4 }, function.returns[3].as_tensor().shape.dims);
}

test make_value_and_grad {
    const vg_fn = comptime make_value_and_grad(test_loss, .{});

    // Verify return type is ValueAndGradResult(TestParams).
    const RetType = @typeInfo(@TypeOf(vg_fn)).pointer.child;
    const ret_info = @typeInfo(RetType).@"fn";
    const ReturnType = @typeInfo(ret_info.return_type.?).error_union.payload;
    try std.testing.expect(@hasField(ReturnType, "value"));
    try std.testing.expect(@hasField(ReturnType, "grads"));

    const specs = .{
        TestParams{
            .w = Tensor.abstract(.f32, &.{ 4, 2 }),
            .b = Tensor.abstract(.f32, &.{2}),
        },
        TestBatch{
            .x = Tensor.abstract(.f32, &.{ 3, 4 }),
        },
    };

    var program = try trace(vg_fn, std.testing.allocator, specs, "vg_test");
    defer program.deinit();

    // value_and_grad returns value (1 tensor) + grads for first arg (2 leaves).
    try std.testing.expectEqual(3, program.output_arity("vg_test"));
    try std.testing.expectEqual(3, program.input_arity("vg_test"));
}
