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
pub const param_call_target_name = core.param_call_target_name;
pub const param_has_side_effect = core.param_has_side_effect;
pub const param_out_aval = core.param_out_aval;

pub const ad = @import("ad.zig");
pub const emit = @import("emit.zig");
pub const zxpr = @import("zxpr.zig");
pub const ops = @import("ops/ops.zig");
