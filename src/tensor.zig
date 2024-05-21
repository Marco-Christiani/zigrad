const std = @import("std");
const Op = @import("grad.zig").Op;
const generateASCIIUUID = @import("grad.zig").generateASCIIUUID;

pub fn NDTensor(comptime T: type, comptime shape: []const u32) type {
    const size: u32 = blk: {
        var n: u32 = 1;
        for (shape) |d| {
            n *= d;
        }
        break :blk n;
    };

    return struct {
        const Self = @This();
        data: [size]T,
        op: ?Op = null,
        children: [2]*anyopaque = undefined,
        label: ?[]const u8 = null,

        pub fn init(values: [size]T) Self {
            var data: [size]T = undefined;
            for (values, 0..) |val, i| {
                data[i] = val;
            }
            return Self{ .data = data };
        }

        pub fn setLabel(self: *Self, comptime label: ?[]const u8) *Self {
            const uid = generateASCIIUUID(4);
            if (label) |l| {
                var buf: [l.len + 4]u8 = undefined;
                _ = std.fmt.bufPrint(&buf, "{?s}{s}", .{ l, uid }) catch {
                    @panic("Failed to set label");
                };
                self.label = &buf;
            } else {
                self.label = uid;
            }
            return self;
        }

        pub fn fill(self: *Self, val: T) void {
            for (self.data) |*elem| {
                elem.* = val;
            }
        }

        pub fn get(self: Self, indices: [shape.len]u32) T {
            const index = self.posToIndex(indices);
            return self.data[index];
        }

        pub fn set(self: *Self, indices: [shape.len]u32, value: T) void {
            const index = self.posToIndex(indices);
            self.data[index] = value;
        }

        fn posToIndex(self: Self, indices: [shape.len]u32) u32 {
            _ = self;
            std.debug.assert(indices.len == shape.len);
            var index: u32 = 0;
            var stride: u32 = 1;
            for (0..shape.len) |i| {
                const dim = shape.len - i - 1;
                const dimSize = shape[dim];
                const idx = indices[dim];
                std.debug.assert(idx < dimSize);

                index += idx * stride;
                stride *= dimSize;
            }
            return index;
        }

        fn indexToPos(self: Self, index: u32) [shape.len]u32 {
            _ = self;
            var pos: [shape.len]u32 = undefined;
            var remainingIndex = index;
            var stride: u32 = 1;
            for (0..shape.len) |i| {
                stride *= shape[i];
            }

            for (0..shape.len) |i| {
                const dim = shape.len - i - 1;
                stride /= shape[dim];
                pos[dim] = remainingIndex / stride;
                remainingIndex %= stride;
            }

            return pos;
        }

        pub fn print(self: Self) void {
            std.debug.print("NDTensor<{any},{d}>[", .{ T, size });
            for (0..size - 1) |i| {
                std.debug.print("{d}, ", .{self.data[i]});
            }
            std.debug.print("{d}]\n", .{self.data[size - 1]});
        }

        pub fn add(self: Self, other: Self) !Self {
            const v1: @Vector(size, T) = self.data;
            const v2: @Vector(size, T) = other.data;
            const v3: [size]T = v1 + v2;
            return Self.init(v3);
        }

        pub fn sub(self: Self, other: Self) !Self {
            const v1: @Vector(size, T) = self.data;
            const v2: @Vector(size, T) = other.data;
            const v3: [size]T = v1 - v2;
            return Self.init(v3);
        }

        pub fn mul(self: Self, other: Self) !Self {
            const v1: @Vector(size, T) = self.data;
            const v2: @Vector(size, T) = other.data;
            const v3: [size]T = v1 * v2;
            return Self.init(v3);
        }

        pub fn div(self: Self, other: Self) !Self {
            const v1: @Vector(size, T) = self.data;
            const v2: @Vector(size, T) = other.data;
            const v3: [size]T = v1 / v2;
            return Self.init(v3);
        }
    };
}

// pub fn Operation(comptime T: type, comptime shape: []const u32) type {
//     return struct {
//         forward: fn (a: NDTensor(T, shape), b: NDTensor(T, shape)) NDTensor(T, shape),
//     };
// }
// pub fn Operation(comptime T: type, comptime shapeIn: []const u32, comptime shapeOut: []const u32) type {
//     return struct{
//         forward: fn (a: T, b: NDTensor(T, shapeIn)) NDTensor(T, shapeOut),
//         backward: fn (grad: T, a: NDTensor(T, shapeIn), b: NDTensor(T, shapeIn)) NDTensor(T, shapeOut),
//     };
// };
//

pub fn Operation(comptime T: type) type {
    return struct {
        const Self = @This();
        forward: *const fn (a: T, b: T) T,
        inputs: [2]*const T,
        output: ?*T,

        pub fn execute(self: Self) void {
            self.output.* = self.forward(self.inputs[0].*, self.inputs[1].*);
            // @memcpy(self.output, self.forward(self.inputs[0].*, self.inputs[1].*));
            // var result = self.forward(self.inputs[0].*, self.inputs[1].*);
            // _ = result;
            // self.output.data = result.data;
            // self.output.size = result.size;
            // self.output.children = result.children;
            // self.output.label = result.label;
        }
    };
}

pub fn Add(comptime T: type, comptime shape: []const u32) type {
    return struct {
        pub fn init(a: *const NDTensor(T, shape), b: *const NDTensor(T, shape)) Operation(NDTensor(T, shape)) {
            // @compileLog("Add.init() at comptime!");
            //
            // std.debug.print("hello", .{});
            // const file = std.fs.cwd().createFile("debug-file-comptime.txt", .{}) catch unreachable;
            // defer file.close();

            // const string = "comptime: Add.init() called!";
            // file.writer().writeAll(string) catch unreachable;
            return Operation(NDTensor(T, shape)){
                .forward = &forward,
                .inputs = .{ a, b },
                .output = null,
            };
        }
        pub fn forward(a: NDTensor(T, shape), b: NDTensor(T, shape)) NDTensor(T, shape) {
            var out = a.add(b) catch {
                @panic("Add forward failed");
            };

            out.children[0] = @constCast(&a);
            out.children[1] = @constCast(&b);
            // if (out.children) |children| {
            //     // children[0] = &a;
            //     // children[1] = &b;
            //     for (children, 0..) |c, i| std.debug.print("child {}: {}", .{ i, c });
            // }
            for (out.children, 0..) |c, i| std.debug.print("child {}: {}\n", .{ i, c });
            return out;
        }
    };
}

pub fn Graph(comptime T: type, comptime n: i32) type {
    return struct {
        const Self = @This();
        operations: [n]Operation(T),

        pub fn run(self: Self) void {
            // @compileLog("Graph.run() at comptime? I should only be possible at runtime!");
            // @panic("Graph.run() called");
            //
            const file = std.fs.cwd().createFile("debug-file-runtime.txt", .{}) catch unreachable;
            defer file.close();

            const string = "runtime: Graph.run() called!";
            file.writer().writeAll(string) catch unreachable;
            for (self.operations) |op| {
                op.execute();
            }
        }
    };
}

// test "tensor/graph" {
//     // const blk = comptime blk: {
//     const shape = &[_]u32{3};
//     const Tensor = NDTensor(f32, shape);
//     const GraphT = Graph(Tensor, 1);
//     var t1 = try Tensor.init([_]f32{ 1, 2, 3 });
//     var t2 = try Tensor.init([_]f32{ 1, 1, 1 });
//     var t3 = try Tensor.init([_]f32{ 0, 0, 0 });

//     const addOp = Add(f32, shape);
//     const op1 = addOp.init(&t1, &t2, &t3);
//     const g = &GraphT{
//         .operations = .{op1},
//     };
//     g.run();
//     t3.print();
// }

// test "tensor/NDTensor-print2" {
//     const Tensor = NDTensor(f32, &[_]u32{ 2, 3 });
//     var t1 = try Tensor.init([_]f32{ 1, 2, 3, 4, 5, 6 });
//     _ = t1.setLabel("abcd");
//     // 1 2 3
//     // 4 5 6
//     t1.print();
// }

// test "tensor/NDTensor" {
//     const Tensor = NDTensor(f32, &[_]u32{ 2, 3 });
//     var t1 = try Tensor.init([_]f32{ 1, 2, 3, 4, 5, 6 });
//     // 1 2 3
//     // 4 5 6
//     t1.set([_]u32{ 1, 2 }, 23);
//     std.debug.print("{d}\n", .{t1.get([_]u32{ 1, 2 })});
//     std.debug.print("{d}\n", .{t1.indexToPos(5)});

//     var t2 = try Tensor.init([_]f32{ 10, 20, 30, 40, 50, 60 });
//     var t3 = try t1.add(t2);
//     std.debug.print("{d}\n", .{t3.get([_]u32{ 0, 0 })});
//     std.debug.print("{d}\n", .{t3.get([_]u32{ 0, 1 })});
//     std.debug.print("{d}\n", .{t3.get([_]u32{ 0, 2 })});
//     std.debug.print("{d}\n", .{t3.get([_]u32{ 1, 0 })});
//     std.debug.print("{d}\n", .{t3.get([_]u32{ 1, 1 })});
//     std.debug.print("{d}\n", .{t3.get([_]u32{ 1, 2 })});

//     var t4 = try t3.div(t3);
//     std.debug.print("{d}\n", .{t4.get([_]u32{ 0, 0 })});
//     std.debug.print("{d}\n", .{t4.get([_]u32{ 0, 1 })});
//     std.debug.print("{d}\n", .{t4.get([_]u32{ 0, 2 })});
//     std.debug.print("{d}\n", .{t4.get([_]u32{ 1, 0 })});
//     std.debug.print("{d}\n", .{t4.get([_]u32{ 1, 1 })});
//     std.debug.print("{d}\n", .{t4.get([_]u32{ 1, 2 })});
// }
