const core = @import("pr.zig");

pub const DType = core.DType;
pub const Shape = core.Shape;
pub const BoundedShape = core.BoundedShape;
pub const Aval = core.Aval;
pub const Tensor = core.Tensor;

pub const Literal = core.Literal;
pub const Prim = core.Prim;

// Param structs
pub const GatherParams = core.GatherParams;
pub const ScatterParams = core.ScatterParams;
pub const CompareParams = core.CompareParams;
pub const CompareDirection = core.CompareDirection;
pub const CompareType = core.CompareType;
pub const ScatterReduction = core.ScatterReduction;
pub const SliceParams = core.SliceParams;
pub const DotGeneralParams = core.DotGeneralParams;
pub const ReshapeParams = core.ReshapeParams;
pub const IotaParams = core.IotaParams;
pub const BroadcastInDimParams = core.BroadcastInDimParams;
pub const TransposeParams = core.TransposeParams;
pub const ConcatenateParams = core.ConcatenateParams;
pub const ReduceParams = core.ReduceParams;
pub const CallParams = core.CallParams;
pub const CustomCallParams = core.CustomCallParams;

// Core IR types
pub const Var = core.Var;
pub const Operand = core.Operand;
pub const Op = core.Op;
pub const Params = core.Params;

pub const Annotation = core.Annotation;
pub const Region = core.Region;
pub const Function = core.Function;
pub const RegionIterator = core.RegionIterator;
pub const Program = core.Program;

pub const ValidationError = core.ValidationError;
pub const validate_ops_in_func = core.validate_ops_in_func;
pub const validate_program = core.validate_program;

pub const BuildError = core.BuildError;
pub const FunctionBuilder = core.FunctionBuilder;

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
