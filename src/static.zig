const std = @import("std");

const Value = struct {
    data: i32,
};

pub fn Operation(comptime T: type) type {
    return struct {
        const Self = @This();
        runFn: *const fn ([2]*T) T,
        inputs: [2]*T,
        output: *T,

        pub fn run(self: Self) void {
            self.output.* = self.runFn(self.inputs);
        }
    };
}

pub fn Add(comptime T: type) type {
    return struct {
        const Self = @This();
        pub fn init(a: *T, b: *T) Operation(T) {
            return Operation(T){
                .runFn = &run,
                .inputs = .{ a, b },
                .output = @constCast(&std.mem.zeroes(T)),
            };
        }

        pub fn run(inputs: [2]*T) T {
            // return T{ .data = a.data + b.data };
            return T{ .data = inputs[0].data + inputs[1].data };
        }

        // pub fn backward(out_grad: T) void {}
    };
}

pub fn OperationGraph(comptime T: type, comptime n: u32) type {
    return struct {
        const Self = @This();
        operations: [n]Operation(T),

        pub fn execute(self: Self) void {
            self._execute(false);
        }
        pub fn execute_verbose(self: Self) void {
            self._execute(true);
        }
        pub fn _execute(self: Self, verbose: bool) void {
            for (self.operations) |op| {
                if (verbose) std.log.debug("running {}\n", .{op});
                op.run();
            }
        }
    };
}

test "static/one" {
    std.testing.log_level = .debug;
    const input1 = &Value{ .data = 1 };
    const input2 = &Value{ .data = 2 };
    const GraphT = OperationGraph(Value, 2);
    const addOp = Add(Value);
    const op1 = addOp.init(@constCast(input1), @constCast(input2));
    const op2 = addOp.init(@constCast(input1), op1.output);
    const g = GraphT{
        .operations = .{ op1, op2 },
    };
    g.execute_verbose();
    std.log.info("{}\n", .{op2.output});
}
