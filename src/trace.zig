//! User-facing tracing and trace-time program construction.
//!
//! ## Typical usage
//!
//! Compose AD and an optimizer inside one traced program:
//! ```zig
//! fn train_step(params: Params, batch: Batch) !struct { loss: Tensor, updated: Params } {
//!     var vg = try zg.transforms.value_and_grad(loss_fn, .{ params, batch }, .{});
//!     defer vg.deinit();
//!     // Apply an optimizer to `vg.grads` and return the updated parameters.
//! }
//! var program = try zg.trace(train_step, allocator, specs, "train_step");
//! ```
const std = @import("std");

const pr = @import("pr/pr.zig");
const Tensor = @import("tensor.zig");
const TensorTree = @import("utils.zig").Tree(Tensor);

/// Trace a comptime function from tensor parameter specifications.
///
/// Each tensor leaf contributes its data type and shape. Abstract, host, and
///  device tensors are valid because tracing does not read their data.
///
/// Each leaf becomes a traced parameter. Struct and tuple nesting are preserved,
///  and the function receives traced tensors in the same structure. The caller
///  remains responsible for `specs`.
///
/// The traced function may compose AD transforms. Its return value defines the
///  program outputs.
///
/// The caller releases the returned `pr.Program` with `deinit`.
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

    const traced = try spec_tree.map(Tensor, &builder, struct {
        fn f(b: *pr.FunctionBuilder, leaf: Tensor) !Tensor {
            return try Tensor.param(b, leaf.dtype, leaf.shape.const_slice());
        }
    }.f);

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
