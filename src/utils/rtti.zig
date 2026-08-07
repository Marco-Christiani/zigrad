//! Runtime identity checks for erased pointers.
//!
//! `TypeID` produces a stable runtime identity for a Zig type. `TypedPtr`
//!  pairs a `TypeID` with an erased pointer so reconstruction via `cast`
//!  asserts type identity in debug/safe builds, zero-cost in release.

const std = @import("std");

/// Stable runtime identity for a Zig type.
pub const TypeID = enum(usize) {
    _,

    pub fn of(comptime T: type) TypeID {
        return @enumFromInt(@intFromPtr(&Holder(T).id));
    }

    /// Per-type storage whose `id` address serves as the TypeID value.
    ///
    /// The `held = T` decl is load-bearing: it forces the struct body to
    ///  depend on `T`, preventing comptime memoization from returning the
    ///  same struct type (and therefore the same `id` address) for
    ///  different `T`.
    fn Holder(comptime T: type) type {
        return struct {
            const held = T;
            var id: u8 = 0;
        };
    }
};

/// Type-tagged erased pointer.
///
/// Non-owning: ownership and cleanup are the call site's responsibility.
pub const TypedPtr = struct {
    type_id: TypeID,
    raw: *anyopaque,

    /// Wrap a concrete pointer, recording its type identity.
    pub fn init(ptr: anytype) TypedPtr {
        const T = std.meta.Child(@TypeOf(ptr));
        return .{
            .type_id = TypeID.of(T),
            .raw = ptr,
        };
    }

    /// Reconstruct the concrete pointer.
    ///
    /// Debug and safe builds assert the recorded type identity.
    pub fn cast(self: TypedPtr, comptime T: type) *T {
        std.debug.assert(self.type_id == TypeID.of(T));
        return @ptrCast(@alignCast(self.raw));
    }
};

// Tests.

test TypeID {
    const A = struct { x: i32 };
    const B = struct { y: f64 };

    try std.testing.expectEqual(TypeID.of(A), TypeID.of(A));
    try std.testing.expect(TypeID.of(A) != TypeID.of(B));
}

test TypedPtr {
    const Foo = struct { val: i32 };
    var foo = Foo{ .val = 42 };

    const ptr = TypedPtr.init(&foo);

    const recovered = ptr.cast(Foo);
    try std.testing.expectEqual(42, recovered.val);

    recovered.val = 99;
    try std.testing.expectEqual(99, foo.val);
}
