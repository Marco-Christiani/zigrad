//! Open composition for transforms, lowerings, and compilers.
//!
//! Operations are ordinary user-defined Zig values. Each operation declares
//!  `Input` and `Output` types and provides `run`. Composition requires no
//!  registry or central artifact union.

const std = @import("std");

pub const Context = @import("compilation/context.zig").Context;
pub const Pipeline = @import("compilation/pipeline.zig").Pipeline;

/// Failures exposed by a type-erased compiler.
pub const Error = error{
    OutOfMemory,
    InvalidInput,
    Unsupported,
    ResourceExhausted,
    Unavailable,
    CompilationFailed,
};

/// Type-erased compiler from one intermediate representation to one artifact type.
///
/// Implementations embed this interface and recover their concrete state with
///  `@fieldParentPtr`. The interface is also a compilation operation, so it can
///  be added directly to a `Pipeline`.
pub fn Compiler(comptime InputType: type, comptime ArtifactType: type) type {
    return struct {
        const Self = @This();

        /// Intermediate representation accepted by the compiler.
        pub const Input = InputType;

        /// Artifact produced by the compiler.
        pub const Output = ArtifactType;

        /// Dispatch table supplied by the compiler implementation.
        vtable: *const VTable,

        /// Function dispatched by the type-erased compiler.
        pub const VTable = struct {
            compile: *const fn (
                compiler: *Self,
                input: Input,
                context: *Context,
            ) Error!Output,
        };

        /// Compile `input` into an artifact.
        pub fn run(
            self: *Self,
            input: Input,
            context: *Context,
        ) Error!Output {
            return try self.vtable.compile(self, input, context);
        }
    };
}

test "Compiler dispatches through an external implementation" {
    const Input = struct { value: u32 };
    const Artifact = struct { value: u64 };
    const Interface = Compiler(Input, Artifact);

    const Fake = struct {
        interface: Interface = .{ .vtable = &vtable },
        calls: usize = 0,

        const vtable: Interface.VTable = .{
            .compile = compile,
        };

        fn compile(
            interface: *Interface,
            input: Input,
            _: *Context,
        ) Error!Artifact {
            const self: *@This() = @fieldParentPtr("interface", interface);
            self.calls += 1;
            return .{ .value = input.value };
        }
    };

    var fake: Fake = .{};
    var context: Context = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    var pipeline = Pipeline.init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.add(&fake.interface);
    const artifact = try pipeline.run(
        Artifact,
        Input{ .value = 7 },
        &context,
    );

    try std.testing.expectEqual(@as(u64, 7), artifact.value);
    try std.testing.expectEqual(@as(usize, 1), fake.calls);
}
