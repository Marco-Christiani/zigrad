//! Runtime composition of compiler passes over open Zig types.

const std = @import("std");
const Context = @import("context.zig").Context;
const operation = @import("operation.zig");
const rtti = @import("../utils/rtti.zig");
const ErasedBox = rtti.ErasedBox;

const log = std.log.scoped(.@"zg/pipeline");

/// A mutable FIFO queue of compiler passes.
///
/// Pass implementations are copied into queue-owned boxes. Pass a pointer when
///  mutable state or identity must remain caller-managed. The queue can run any
///  number of times and does not retain intermediate values between runs.
pub const Pipeline = struct {
    const Entry = struct {
        input_type_id: rtti.TypeID,
        input_type_name: []const u8,
        output_type_id: rtti.TypeID,
        output_type_name: []const u8,
        implementation: ErasedBox,
        invoke: *const fn (*Entry, *ErasedBox, *Context) anyerror!void,
    };

    /// Failures while adding a pass to a pipeline.
    pub const AddError = std.mem.Allocator.Error || error{IncompatiblePass};

    allocator: std.mem.Allocator,
    entries: std.ArrayList(Entry) = .empty,

    /// Initialize an empty pass queue.
    pub fn init(allocator: std.mem.Allocator) Pipeline {
        return .{ .allocator = allocator };
    }

    /// Release pass storage owned by this queue.
    pub fn deinit(self: *Pipeline) void {
        for (self.entries.items) |*entry| entry.implementation.deinit();
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    /// Append an operation after checking its declared type edge.
    ///
    /// On success, a type-changing operation must not retain its input. On
    ///  failure, every operation must leave its input valid for release. A
    ///  same-type operation transfers the current value's resources to its output.
    pub fn add(
        self: *Pipeline,
        implementation: anytype,
    ) AddError!void {
        const Stored = @TypeOf(implementation);
        const Operation = operation.type_of(Stored);
        comptime operation.validate(Operation);
        const Input = Operation.Input;
        const Output = Operation.Output;

        if (self.entries.items.len > 0) {
            const preceding = self.entries.items[self.entries.items.len - 1];
            if (preceding.output_type_id != rtti.TypeID.of(Input)) {
                log.debug(
                    "cannot append {s} -> {s} after a pass producing {s}",
                    .{ @typeName(Input), @typeName(Output), preceding.output_type_name },
                );
                return error.IncompatiblePass;
            }
        }

        var stored = try ErasedBox.init(self.allocator, implementation);
        errdefer stored.deinit();

        const Adapter = struct {
            fn invoke(entry: *Entry, current: *ErasedBox, context: *Context) anyerror!void {
                if (!current.is(Input)) return error.TypeMismatch;

                const input = current.cast(Input);
                const concrete = entry.implementation.cast(Stored);
                const output = if (comptime @typeInfo(Stored) == .pointer)
                    try concrete.*.run(input.*, context)
                else
                    try concrete.run(input.*, context);

                if (comptime Input == Output) {
                    input.* = output;
                    return;
                }

                const next = try ErasedBox.init(context.allocator, output);
                current.deinit();
                current.* = next;
            }
        };

        try self.entries.append(self.allocator, .{
            .input_type_id = rtti.TypeID.of(Input),
            .input_type_name = @typeName(Input),
            .output_type_id = rtti.TypeID.of(Output),
            .output_type_name = @typeName(Output),
            .implementation = stored,
            .invoke = Adapter.invoke,
        });
    }

    /// Run every queued pass and return the requested output type.
    ///
    /// The pipeline consumes `input` after its type matches the first pass. Pass
    ///  a pointer when the referenced value must remain caller-managed.
    pub fn run(
        self: *Pipeline,
        comptime Output: type,
        input: anytype,
        context: *Context,
    ) anyerror!Output {
        const Input = @TypeOf(input);
        if (self.entries.items.len == 0) {
            if (comptime Input != Output) return error.OutputTypeMismatch;
        } else {
            if (rtti.TypeID.of(Input) != self.entries.items[0].input_type_id) {
                log.debug(
                    "pipeline expects {s}, received {s}",
                    .{ self.entries.items[0].input_type_name, @typeName(Input) },
                );
                return error.InputTypeMismatch;
            }
            const final = self.entries.items[self.entries.items.len - 1];
            if (rtti.TypeID.of(Output) != final.output_type_id) {
                log.debug(
                    "pipeline produces {s}, requested {s}",
                    .{ final.output_type_name, @typeName(Output) },
                );
                return error.OutputTypeMismatch;
            }
        }

        var current = try ErasedBox.init(context.allocator, input);
        errdefer current.deinit();
        for (self.entries.items) |*entry| {
            try entry.invoke(entry, &current, context);
        }
        return current.take(Output);
    }
};

test "Pipeline runs an open heterogeneous pass queue" {
    const Source = struct { value: u32 };
    const Intermediate = struct { value: u64 };
    const Artifact = struct { value: u64 };

    const Increment = struct {
        pub const Input = Source;
        pub const Output = Source;

        amount: u32,

        pub fn run(self: *@This(), input: Input, _: *Context) !Output {
            return .{ .value = input.value + self.amount };
        }
    };
    const Lower = struct {
        pub const Input = Source;
        pub const Output = Intermediate;

        pub fn run(_: *@This(), input: Input, _: *Context) !Output {
            return .{ .value = input.value };
        }
    };
    const Compile = struct {
        pub const Input = Intermediate;
        pub const Output = Artifact;

        factor: u64,

        pub fn run(self: *@This(), input: Input, _: *Context) !Output {
            return .{ .value = input.value * self.factor };
        }
    };

    var pipeline = Pipeline.init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.add(Increment{ .amount = 3 });
    try pipeline.add(Lower{});
    try pipeline.add(Compile{ .factor = 4 });

    var context: Context = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    const artifact = try pipeline.run(
        Artifact,
        Source{ .value = 2 },
        &context,
    );

    try std.testing.expectEqual(@as(u64, 20), artifact.value);
}

test "Pipeline rejects incompatible adjacent types" {
    const First = struct {
        pub const Input = u32;
        pub const Output = u64;

        pub fn run(_: *@This(), input: Input, _: *Context) !Output {
            return input;
        }
    };
    const Wrong = struct {
        pub const Input = u16;
        pub const Output = u16;

        pub fn run(_: *@This(), input: Input, _: *Context) !Output {
            return input;
        }
    };

    var pipeline = Pipeline.init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.add(First{});
    try std.testing.expectError(error.IncompatiblePass, pipeline.add(Wrong{}));
}

test "Pipeline checks run endpoint types before execution" {
    const Convert = struct {
        pub const Input = u32;
        pub const Output = u64;

        pub fn run(_: *@This(), input: Input, _: *Context) !Output {
            return input;
        }
    };

    var pipeline = Pipeline.init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.add(Convert{});

    var context: Context = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    try std.testing.expectError(
        error.InputTypeMismatch,
        pipeline.run(u64, @as(u16, 1), &context),
    );
    try std.testing.expectError(
        error.OutputTypeMismatch,
        pipeline.run(u16, @as(u32, 1), &context),
    );
}

test "Pipeline releases stored pass values" {
    const Pass = struct {
        pub const Input = u32;
        pub const Output = u32;

        calls: *usize,

        pub fn run(_: *@This(), input: Input, _: *Context) !Output {
            return input;
        }

        pub fn deinit(self: *@This()) void {
            self.calls.* += 1;
        }
    };

    var calls: usize = 0;
    var pipeline = Pipeline.init(std.testing.allocator);
    try pipeline.add(Pass{ .calls = &calls });
    pipeline.deinit();
    try std.testing.expectEqual(1, calls);
}

test "Pipeline releases the current intermediate after failure" {
    const Intermediate = struct {
        bytes: []u8,

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            allocator.free(self.bytes);
        }
    };
    const Lower = struct {
        pub const Input = u32;
        pub const Output = Intermediate;

        pub fn run(_: *@This(), input: Input, context: *Context) !Output {
            const bytes = try context.allocator.alloc(u8, input);
            return .{ .bytes = bytes };
        }
    };
    const Reject = struct {
        pub const Input = Intermediate;
        pub const Output = Intermediate;

        pub fn run(_: *@This(), _: Input, _: *Context) !Output {
            return error.Rejected;
        }
    };
    var pipeline = Pipeline.init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.add(Lower{});
    try pipeline.add(Reject{});

    var context: Context = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    try std.testing.expectError(
        error.Rejected,
        pipeline.run(Intermediate, @as(u32, 8), &context),
    );
}
