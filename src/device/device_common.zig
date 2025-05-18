pub const ReduceType = enum { sum, mean };
pub const SmaxType = enum { fast, max, log };
pub const RandType = union(enum) { uniform, normal, kaiming: usize };
pub const BinaryOp = enum { add, min, max };
pub const TransferDirection = enum { HtoD, DtoH, DtoD };
