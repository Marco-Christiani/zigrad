//! Element data types shared by framework values and integrations.

/// Element data type for tensors and execution buffers.
pub const DType = enum {
    f16,
    bf16,
    f32,
    f64,
    i8,
    u8,
    i32,
    i64,
    u32,
    u64,
    bool,

    /// Return the storage bytes required for one element.
    pub inline fn size_in_bytes(self: DType) usize {
        return switch (self) {
            .bool, .i8, .u8 => 1,
            .f16, .bf16 => 2,
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
        };
    }

    /// Return the canonical data type name.
    pub fn name(self: DType) []const u8 {
        return @tagName(self);
    }

    /// Return the Zig type that stores one element in host memory.
    ///
    /// Float16 variants use `u16` bit patterns. Use `encode` and `decode`
    ///  when converting numeric values.
    pub fn StorageType(comptime self: DType) type {
        return switch (self) {
            .f32 => f32,
            .f64 => f64,
            .bf16, .f16 => u16,
            .i8 => i8,
            .u8 => u8,
            .i32 => i32,
            .i64 => i64,
            .u32 => u32,
            .u64 => u64,
            .bool => u8,
        };
    }

    /// Convert a float value to this data type's storage representation.
    pub inline fn encode(comptime self: DType, comptime T: type, value: T) StorageType(self) {
        return switch (self) {
            .f32 => @floatCast(value),
            .f64 => @floatCast(value),
            .bf16 => @intCast(@as(u32, @bitCast(@as(f32, @floatCast(value)))) >> 16),
            .f16 => @bitCast(@as(f16, @floatCast(value))),
            .i32 => @intFromFloat(value),
            else => @compileError("encode not supported for " ++ @tagName(self)),
        };
    }

    /// Convert this data type's storage representation to a float type.
    pub inline fn decode(comptime self: DType, comptime T: type, raw: StorageType(self)) T {
        return switch (self) {
            .f32 => @floatCast(raw),
            .f64 => @floatCast(raw),
            .bf16 => @floatCast(@as(f32, @bitCast(@as(u32, raw) << 16))),
            .f16 => @floatCast(@as(f16, @bitCast(raw))),
            .i32 => @floatFromInt(raw),
            else => @compileError("decode not supported for " ++ @tagName(self)),
        };
    }
};
