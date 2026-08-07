//! Open composition for transforms, lowerings, and compilers.
//!
//! Operations are ordinary user-defined Zig values. Each operation declares
//!  `Input` and `Output` types and provides `run`. Composition requires no
//!  registry or central artifact union.

const std = @import("std");
const device = @import("device.zig");

/// Failures exposed by a type-erased compiler.
pub const Error = error{
    OutOfMemory,
    InvalidInput,
    Unsupported,
    ResourceExhausted,
    Unavailable,
    CompilationFailed,
};

/// Shared services available while running compilation operations.
pub const Context = struct {
    /// Allocator available to compilation operations.
    allocator: std.mem.Allocator,

    /// I/O implementation available to compilation operations.
    io: std.Io,

    /// Device selected for operations that resolve device-specific policy.
    device: ?device.Device = null,
};

/// Type-erased compiler from one intermediate representation to one artifact type.
///
/// Implementations embed this interface and recover their concrete state with
///  `@fieldParentPtr`. The interface is also a compilation operation, so it can
///  be passed directly to `Flow.compile`.
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

/// Start an eager composition from `value`.
pub fn start(value: anytype, context: *Context) Flow(@TypeOf(value)) {
    return .{
        .value = value,
        .context = context,
    };
}

/// One value in an eager compilation composition.
///
/// `Flow` does not manage the lifetime of `value`. Each operation's contract
///  states whether the caller must release its result.
pub fn Flow(comptime Value: type) type {
    return struct {
        const Self = @This();

        value: Value,
        context: *Context,

        /// Apply a same-type transformation and retain this flow.
        pub fn transform(self: *Self, operation: anytype) !void {
            const Operation = operation_type(@TypeOf(operation));
            comptime validate_operation(Operation);
            comptime {
                if (Operation.Input != Value or Operation.Output != Value) {
                    @compileError(
                        @typeName(Operation) ++ " is not a transform over " ++
                            @typeName(Value),
                    );
                }
            }

            self.value = try operation.run(self.value, self.context);
        }

        /// Apply a representation-changing lowering.
        pub fn lower(self: Self, operation: anytype) !Flow(operation_output(@TypeOf(operation))) {
            const Operation = operation_type(@TypeOf(operation));
            comptime validate_operation(Operation);
            comptime {
                if (Operation.Input != Value) {
                    @compileError(
                        @typeName(Operation) ++ " cannot lower " ++ @typeName(Value),
                    );
                }
            }

            return start(try operation.run(self.value, self.context), self.context);
        }

        /// Apply a terminal compiler operation.
        pub fn compile(self: Self, operation: anytype) !Flow(operation_output(@TypeOf(operation))) {
            const Operation = operation_type(@TypeOf(operation));
            comptime validate_operation(Operation);
            comptime {
                if (Operation.Input != Value) {
                    @compileError(
                        @typeName(Operation) ++ " cannot compile " ++ @typeName(Value),
                    );
                }
            }

            return start(try operation.run(self.value, self.context), self.context);
        }
    };
}

fn operation_type(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .pointer => |pointer| pointer.child,
        else => T,
    };
}

fn operation_output(comptime T: type) type {
    const Operation = operation_type(T);
    validate_operation(Operation);
    return Operation.Output;
}

fn validate_operation(comptime Operation: type) void {
    if (!@hasDecl(Operation, "Input")) {
        @compileError(@typeName(Operation) ++ " must declare Input");
    }
    if (!@hasDecl(Operation, "Output")) {
        @compileError(@typeName(Operation) ++ " must declare Output");
    }
    if (!@hasDecl(Operation, "run")) {
        @compileError(@typeName(Operation) ++ " must provide run");
    }
}

test "user operations compose without registration" {
    const Source = struct { value: u32 };
    const Intermediate = struct { value: u64 };
    const Executable = struct { value: u64 };

    const Increment = struct {
        pub const Input = Source;
        pub const Output = Source;

        amount: u32,

        pub fn run(self: @This(), input: Input, _: *Context) !Output {
            return .{ .value = input.value + self.amount };
        }
    };

    const Lower = struct {
        pub const Input = Source;
        pub const Output = Intermediate;

        pub fn run(_: @This(), input: Input, _: *Context) !Output {
            return .{ .value = input.value };
        }
    };

    const Compile = struct {
        pub const Input = Intermediate;
        pub const Output = Executable;

        factor: u64,

        pub fn run(self: @This(), input: Input, _: *Context) !Output {
            return .{ .value = input.value * self.factor };
        }
    };

    var context = Context{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    var source = start(Source{ .value = 2 }, &context);
    try source.transform(Increment{ .amount = 3 });
    const intermediate = try source.lower(Lower{});
    const executable = try intermediate.compile(Compile{ .factor = 4 });

    try std.testing.expectEqual(@as(u64, 20), executable.value.value);
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
    const input = start(Input{ .value = 7 }, &context);
    const artifact = (try input.compile(&fake.interface)).value;

    try std.testing.expectEqual(@as(u64, 7), artifact.value);
    try std.testing.expectEqual(@as(usize, 1), fake.calls);
}
