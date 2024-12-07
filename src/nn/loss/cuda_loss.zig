const zg = @import("zigrad");
const ReduceType = zg.ReduceType;
const NDTensor = zg.NDTensor;
const GraphManager = zg.GraphManager;
const std = @import("std");

// NOTE: Reductions return NaN when "reduce" is set to false.
// This is for c-compatibility and this value is instead traded for null.

// nll entry point
pub fn nll(comptime T: type, comptime config: NLLConfig) NLLType(T, config) {
    return .{};
}
//
// s = loss.score(x, y);
// if (s.score) |s| {
//
//}
//
// s.backward(gm);

pub const NLLConfig = struct {
    // dimensions of input tensor
    dimensions: usize,
    // nll expecting logits - if true,
    // softmax will be used on input
    input_logits: bool,
    // specifies that the target type
    // will be provided as an index
    target_type: enum { indices, encoding },
    // specifies the reduce type used
    reduce_type: ReduceType,
};

// negative log likelihood
pub fn NLLType(comptime T: type, comptime config: NLLConfig) type {
    return if (config.target_type == .indices) NLLIndex(T, config) else @compileError("Unimplemented");
}

fn NLLIndex(comptime T: type, comptime config: NLLConfig) type {
    return switch (config.dimensions) {
        1 => struct {
            pub const LossOutput = LossOutputType(T, usize, backward);

            pub fn score(src: *NDTensor(T), trg: usize, reduce: bool) *NDTensor {
                const result = NDTensor(T)
                    .src.device.nn.nllLoss1DIndexForward(T, src.getData(), trg, config.input_logits, reduce, config.reduce_type);
                return result;
            }

            fn backward(src: *zg.NDTensor(T), trg: usize) void {
                src.device.nn.nllLoss1DIndexBackward(T, src.getData(), src.grad.?.data, trg, config.reduce_type);
            }
        },
        else => @compileError("Unsupported Dimensions for NLLIndex"),
    };
}
