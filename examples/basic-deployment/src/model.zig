const zg = @import("zigrad");
const Tensor = zg.Tensor;

pub const Input = struct {
    lhs: Tensor,
    rhs: Tensor,
    scale: Tensor,
};

pub const input_spec: Input = .{
    .lhs = Tensor.abstract(.f32, &.{ 2, 3 }),
    .rhs = Tensor.abstract(.f32, &.{ 3, 2 }),
    .scale = Tensor.abstract(.f32, &.{ 2, 2 }),
};

/// `(lhs @ rhs + scale) * scale`
pub fn forward(input: Input) !Tensor {
    const product = try input.lhs.matmul(input.rhs);
    const shifted = try product.add(input.scale);
    return try shifted.mul(input.scale);
}
