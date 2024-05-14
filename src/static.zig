const std = @import("std");

const Value = struct {
    data: i32,
};

pub fn Operation(comptime T: type) type {
    return struct {
        const Self = @This();
        runFn: *const fn (a: T, b: T) T,
        inputs: [2]*const T,
        output: *T,

        pub fn run(self: Self) void {
            self.output.* = self.runFn(self.inputs[0].*, self.inputs[1].*);
        }
    };
}

pub fn Add(comptime T: type) type {
    return struct {
        const Self = @This();
        // cannot pass a reference to a mutable output container type at comptime
        // but, why cant output be an address to some memory baked in the executable that is mutable at runtime?
        pub fn init(a: *const T, b: *const T) Operation(T) {
            return Operation(T){
                .runFn = &run,
                .inputs = .{ a, b },
                .output = @constCast(&std.mem.zeroes(T)), // try to make an empty container as a workaround
            };
        }

        pub fn run(a: T, b: T) T {
            return T{ .data = a.data + b.data };
        }
    };
}

pub fn OperationGraph(comptime T: type, comptime n: u32) type {
    return struct {
        const Self = @This();
        operations: [n]Operation(T),

        pub fn execute(self: Self) void {
            for (self.operations) |op| {
                op.run();
            }
        }
    };
}

test "static/one" {
    // build the op graph at compile time, making space for inputs and outputs
    const input1 = Value{ .data = 1 };
    const input2 = Value{ .data = 2 };
    const GraphT = OperationGraph(Value, 2);
    const addOp = comptime Add(Value);
    const op1 = comptime addOp.init(&input1, &input2);
    const op2 = comptime addOp.init(&input1, op1.output);
    const g = comptime &GraphT{
        .operations = .{ op1, op2 },
    };
    // execute at runtime, mutating existing memory
    g.execute();
    std.debug.print("{}", .{op2.output});
}
