//! Compare and Select operations.
const std = @import("std");
const types = @import("types.zig");
const pr = @import("../pr.zig");
const Aval = pr.Aval;

fn format_compare_params(writer: *types.Writer, cparams: pr.CompareParams) types.FormatError!void {
    try writer.print("dir={s} type={s}", .{ @tagName(cparams.direction), @tagName(cparams.compare_type) });
}

// Compare

pub const compare = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(op: *const pr.Op, cparams: pr.CompareParams) pr.ValidationError!void {
        if (op.inputs.len != 2 or op.outputs.len != 1) return error.InvalidOpArity;

        const lhs = op.operand(0).as_tensor();
        const rhs = op.operand(1).as_tensor();
        const out = op.result(0).as_tensor();

        if (!types.same_tensor_type(lhs, rhs)) return error.CompareTypeMismatch;
        if (out.dtype != .bool) return error.CompareTypeMismatch;
        if (!std.mem.eql(i64, lhs.shape.dims, out.shape.dims)) return error.CompareTypeMismatch;

        switch (lhs.dtype) {
            .f16, .bf16, .f32, .f64 => {
                if (cparams.compare_type != .FLOAT and cparams.compare_type != .TOTALORDER) return error.CompareTypeMismatch;
            },
            .i8, .i32, .i64 => if (cparams.compare_type != .SIGNED) return error.CompareTypeMismatch,
            .u8, .u32, .u64 => if (cparams.compare_type != .UNSIGNED) return error.CompareTypeMismatch,
            .bool => if (cparams.compare_type != .UNSIGNED) return error.CompareTypeMismatch,
        }
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, cparams: pr.CompareParams) pr.BuildError!Aval {
        if (inputs.len != 2) return error.InvalidOpArity;
        const lhs = inputs[0].as_tensor();
        const rhs = inputs[1].as_tensor();
        if (!types.same_tensor_type(lhs, rhs)) return error.CompareTypeMismatch;

        switch (lhs.dtype) {
            .f16, .bf16, .f32, .f64 => {
                if (cparams.compare_type != .FLOAT and cparams.compare_type != .TOTALORDER) return error.CompareTypeMismatch;
            },
            .i8, .i32, .i64 => if (cparams.compare_type != .SIGNED) return error.CompareTypeMismatch,
            .u8, .u32, .u64 => if (cparams.compare_type != .UNSIGNED) return error.CompareTypeMismatch,
            .bool => if (cparams.compare_type != .UNSIGNED) return error.CompareTypeMismatch,
        }
        return .{ .tensor = .{ .dtype = .bool, .shape = lhs.shape } };
    }

    pub fn format(writer: *types.Writer, _: *const pr.Op, cparams: pr.CompareParams) types.FormatError!void {
        try format_compare_params(writer, cparams);
    }
};

// Select

pub const select = struct {
    pub const arity = .{ .in = 3, .out = 1 };

    pub fn validate(op: *const pr.Op, _: void) pr.ValidationError!void {
        if (op.inputs.len != 3 or op.outputs.len != 1) return error.InvalidOpArity;

        const cond = op.operand(0).as_tensor();
        const on_true = op.operand(1).as_tensor();
        const on_false = op.operand(2).as_tensor();
        const out = op.result(0).as_tensor();

        if (cond.dtype != .bool) return error.SelectTypeMismatch;
        if (!types.same_tensor_type(on_true, on_false)) return error.SelectTypeMismatch;
        if (!types.same_tensor_type(on_true, out)) return error.SelectTypeMismatch;
        if (!std.mem.eql(i64, cond.shape.dims, out.shape.dims)) return error.SelectTypeMismatch;
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, _: void) pr.BuildError!Aval {
        if (inputs.len != 3) return error.InvalidOpArity;
        const cond = inputs[0].as_tensor();
        if (cond.dtype != .bool) return error.SelectTypeMismatch;
        const on_true = inputs[1].as_tensor();
        const on_false = inputs[2].as_tensor();
        if (!types.same_tensor_type(on_true, on_false)) return error.SelectTypeMismatch;
        if (!std.mem.eql(i64, cond.shape.dims, on_true.shape.dims)) return error.SelectTypeMismatch;
        return .{ .tensor = on_true };
    }

    /// VJP backward for select. Routes the cotangent to the chosen branch:
    ///  on_true gets the cotangent where cond is true (zero elsewhere),
    ///  on_false gets the cotangent where cond is false (zero elsewhere).
    ///  No cotangent for cond itself (discrete, non-differentiable).
    pub fn vjp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 3) return error.UnsupportedEqn;

        const out_cot = ctx.get_cot(op.result(0)) orelse return;
        const cond = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const on_true_tensor = op.operand(1).as_tensor();

        const zeros = try ctx.builder.scalar_broadcast(on_true_tensor.dtype, on_true_tensor.shape.dims, 0.0);
        const true_contrib = try ctx.builder.select(cond, out_cot, zeros);
        const false_contrib = try ctx.builder.select(cond, zeros, out_cot);

        try ctx.add_cot(op.operand(1), true_contrib);
        try ctx.add_cot(op.operand(2), false_contrib);
    }

    /// JVP: \(\mathrm{d}(\operatorname{select}(c, t, f)) =
    ///  \operatorname{select}(c, \mathrm{d}t, \mathrm{d}f)\).
    pub fn jvp(ctx: types.AdContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 3) return error.UnsupportedEqn;

        const cond = ctx.get_primal(op.operand(0)) orelse return error.UnsupportedEqn;
        const dt = try ctx.tangent_or_zero(op.operand(1));
        const df = try ctx.tangent_or_zero(op.operand(2));
        ctx.set_tangent(op.result(0), try ctx.builder.select(cond, dt, df));
    }
};
