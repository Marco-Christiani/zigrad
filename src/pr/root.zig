const core = @import("pr.zig");

pub const DType = core.DType;
pub const Shape = core.Shape;
pub const BoundedShape = core.BoundedShape;
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

pub const param = core.param;

pub const ad = @import("ad.zig");
pub const dump = @import("dump.zig");
pub const json = @import("json.zig");
pub const zxpr = @import("zxpr/root.zig");
pub const ops = @import("ops/ops.zig");

test {
    @import("std").testing.refAllDecls(@This());
    // pure tests are not part of public api
    _ = @import("tests/root.zig");
}
