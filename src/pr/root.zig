const core = @import("pr.zig");

pub const DType = core.DType;
pub const Shape = core.Shape;
pub const Aval = core.Aval;
pub const Tensor = core.Tensor;
pub const VarId = core.VarId;
pub const Span = core.Span;

pub const Literal = core.Literal;
pub const Prim = core.Prim;
pub const Param = core.Param;
pub const GatherParams = core.GatherParams;

pub const Eqn = core.Eqn;

pub const Function = core.Function;
pub const Program = core.Program;

pub const ValidationError = core.ValidationError;
pub const validate_function = core.validate_function;
pub const validate_program = core.validate_program;

pub const BuildError = core.BuildError;
pub const FunctionBuilder = core.FunctionBuilder;

pub const param_literal = core.param_literal;
pub const param_out_shape = core.param_out_shape;
pub const param_broadcast_dims = core.param_broadcast_dims;
pub const param_permutation = core.param_permutation;
pub const param_iota_dimension = core.param_iota_dimension;
pub const param_call_target_name = core.param_call_target_name;
pub const param_call_kernel_key = core.param_call_kernel_key;
pub const param_call_provider_name = core.param_call_provider_name;
pub const param_call_kernel_id = core.param_call_kernel_id;
pub const param_call_carrier_hint = core.param_call_carrier_hint;
pub const param_has_side_effect = core.param_has_side_effect;
pub const param_out_aval = core.param_out_aval;
pub const param_out_avals = core.param_out_avals;

pub const ad = @import("ad.zig");
pub const emit = @import("emit.zig");
pub const json = @import("json.zig");
pub const zxpr = @import("zxpr.zig");
pub const ops = @import("ops/ops.zig");

test {
    @import("std").testing.refAllDecls(@This());
    // pure tests are not part of public api
    _ = @import("tests/root.zig");
}
