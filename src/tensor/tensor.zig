// TODO: implement view(), transpose(), and permute(), where the latter two mutate the shape
// TODO: better operation abstraction
// TODO: print graph, just take from the existing impl
const std = @import("std");
const zarray = @import("zarray.zig");
const Shape = zarray.Shape;
const NDArray = zarray.NDArray;
const ZarrayError = zarray.ZarrayError;
const settings = @import("zigrad").settings;

pub const Op = enum { ADD, SUB, MUL, DIV, POW, TANH, MATMUL, DOT, MATVEC, SUM };

pub fn NDTensor(comptime T: type) type {
    return struct {
        const dtype = NDArray(T);
        const Self = @This();
        data: *dtype,
        op: ?Op = null,
        children: ?[2]*const Self = null,
        label: ?[]const u8 = null,
        grad: ?*dtype = null,
        requires_grad: bool,
        acquired: bool = false,
        _backward: ?*const fn (*Self, ?*dtype, std.mem.Allocator) void = null,
        _backward_ctx: ?*dtype = null,

        pub fn init(values: []const T, shape: ?[]const usize, requires_grad: bool, allocator: std.mem.Allocator) !*Self {
            const result = try allocator.create(Self);
            result.* = Self{
                .data = try dtype.init(values, shape, allocator),
                .grad = blk: {
                    if (requires_grad) {
                        const g = try dtype.empty(shape orelse &[_]usize{values.len}, allocator);
                        g.fill(0);
                        break :blk g;
                    } else break :blk null;
                },
                .requires_grad = requires_grad,
            };
            return result;
        }

        pub fn deinit(self: *const Self, allocator: std.mem.Allocator) void {
            if (self.acquired) std.debug.panic("Attempt to deinit an acquired tensor.", .{});
            self.data.deinit(allocator);
            if (self.grad) |g| g.deinit(allocator);
            allocator.destroy(self);
        }

        pub fn teardown(self: *const Self, allocator: std.mem.Allocator) void {
            if (self.acquired) std.debug.panic("Attempt to deinit an acquired tensor.", .{});
            if (self.children) |children| for (children) |c| if (!c.acquired) c.deinit(allocator);
            self.data.deinit(allocator);
            if (self.grad) |g| g.deinit(allocator);
        }

        pub fn acquire(self: *Self) void {
            self.acquired = true;
        }

        pub fn release(self: *Self) void {
            self.acquired = false;
        }

        pub fn fromZarray(values: *dtype, requires_grad: bool, grad_shape: ?Shape, allocator: std.mem.Allocator) !*Self {
            const result = try allocator.create(Self);
            const grad: ?*dtype = blk: {
                if (requires_grad) {
                    if (grad_shape) |s| {
                        const g = try dtype.empty(s.shape, allocator);
                        g.fill(0.0);
                        break :blk g;
                    } else {
                        break :blk (try dtype.zerosLike(values, allocator));
                    }
                } else {
                    break :blk null;
                }
            };
            result.* = Self{
                .data = values,
                .grad = grad,
                .requires_grad = requires_grad,
            };
            return result;
        }

        pub fn reshape(self: *Self, shape: []const usize) !*Self {
            try self.data.reshape(shape);
            if (self.grad) |g| {
                try g.reshape(shape);
            }
            return self;
        }

        pub fn setLabel(self: *Self, comptime label: ?[]const u8) *Self {
            self.label = label;
            return self;
        }

        pub fn fill(self: *Self, val: T) void {
            self.data.fill(val);
        }

        pub fn get(self: Self, indices: []const usize) T {
            return self.data.get(indices);
        }

        pub fn set(self: *Self, indices: []const usize, value: T) !void {
            try self.data.set(indices, value);
        }

        fn posToIndex(self: Self, indices: []const usize) usize {
            return self.data.posToOffset(indices);
        }

        fn flexPosToIndex(self: Self, indices: []const usize) ZarrayError.InvalidIndex!usize {
            return self.data.flexPosToOffset(indices);
        }

        fn indexToPos(self: Self, index: usize, allocator: std.mem.Allocator) []const usize {
            return self.data.offsetToPos(index, allocator);
        }

        pub fn slice(self: *const Self, ranges: []const zarray.Range, allocator: std.mem.Allocator) !*Self {
            const sliced_data = try self.data.slice(ranges);
            const result = try allocator.create(Self);
            result.* = Self{
                .data = sliced_data,
                .grad = if (self.requires_grad) try dtype.empty(sliced_data.shape.shape, allocator) else null,
                .requires_grad = self.requires_grad,
                .op = null,
                .children = null,
                .label = null,
                .acquired = false,
                ._backward = null,
                ._backward_ctx = null,
            };
            return result;
        }

        pub fn setSlice(self: *Self, ranges: []const zarray.Range, values: *const Self) !void {
            if (self.requires_grad) {
                // If we're tracking gradients, we need to create a new operation
                const slice_ = try self.slice(ranges, self.data.shape.alloc);
                const result = try slice_.add(values, self.data.shape.alloc);
                try self.data.setSlice(ranges, result.data.*);

                // Set up backward pass
                result._backward = struct {
                    fn backward(tensor: *Self, grad: ?*dtype, allocator: std.mem.Allocator) void {
                        _ = allocator;
                        if (grad) |g| {
                            if (tensor.children) |children| {
                                const original = children[0];
                                const values_ = children[1];

                                try original.grad.?.setSlice(ranges, g.*);
                                _ = try values_.grad.?._add(g);
                            }
                        }
                    }
                }.backward;
                result.children = .{ self, values };
            } else {
                // if not tracking gradients can just set the values directly
                try self.data.setSlice(ranges, values.data.*);
            }
        }

        pub fn print(self: *const Self) void {
            std.debug.print("NDTensor<{},{?s}>[", .{ T, if (self.op) |o| @tagName(o) else null });
            std.debug.print("data: ", .{});
            self.data.print();
            if (self.grad) |g| {
                std.debug.print(" grad: ", .{});
                g.print();
            }
            std.debug.print("], requires_grad={}", .{self.requires_grad});
            if (self.label) |l| {
                std.debug.print(" label={s}", .{l});
            }
            std.debug.print("\n", .{});
        }

        pub const ClipOptions = struct {
            max_norm: f32 = settings.grad_clip_max_norm,
            delta: f32 = settings.grad_clip_delta,
        };

        pub fn clip_grad_norm_delta(self: *const Self, opts: ClipOptions) void {
            self.grad.?.clip_norm(opts.max_norm, opts.delta);
        }

        /// Element-wise addition
        pub fn add(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const out = try self.data.add(other.data, allocator);
            // const grad_shape = if (self.requires_grad or other.requires_grad) (try Shape.init(out.shape.shape, allocator)).* else null;
            const grad_shape = if (self.requires_grad or other.requires_grad) out.shape.* else null;
            var result = try Self.fromZarray(out, self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) result.op = .ADD;
            result.children = .{ self, other };
            return result;
        }

        /// Element-wise subtraction
        pub fn sub(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const out = try self.data.sub(other.data, allocator);
            const grad_shape = if (self.requires_grad or other.requires_grad) out.shape.* else null;
            var result = try Self.fromZarray(out, self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) result.op = .SUB;
            result.children = .{ self, other };
            return result;
        }

        /// Element-wise multiplication
        pub fn mul(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const out = try self.data.mul(other.data, allocator);
            const grad_shape = if (self.requires_grad or other.requires_grad) out.shape.* else null;
            var result = try Self.fromZarray(out, self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) result.op = .MUL;
            result.children = .{ self, other };
            return result;
        }

        /// Element-wise division
        pub fn div(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const out = try self.data.div(other.data, allocator);
            const grad_shape = if (self.requires_grad or other.requires_grad) out.shape.* else null;
            var result = try Self.fromZarray(out, self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) result.op = .DIV;
            result.children = .{ self, other };
            return result;
        }

        pub const MmOptions = struct { trans_a: bool = false, trans_b: bool = false };

        pub fn matmul(self: *const Self, other: *const Self, allocator: std.mem.Allocator, opts: MmOptions) !*Self {
            const out = try self.data.matmul(other.data, opts.trans_a, opts.trans_b, allocator);
            const grad_shape = if (self.requires_grad or other.requires_grad) out.shape.* else null;
            var result = try Self.fromZarray(out, self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) result.op = .MATMUL;
            result.children = .{ self, other };
            return result;
        }

        // pub fn dot(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
        //     var result = try Self.fromZarray(try self.data.dot(other.data, allocator), self.requires_grad, allocator);
        //     if (self.requires_grad) result.op = .DOT;
        //     result.children = .{ self, other };
        //     return result;
        // }

        pub fn dot(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const dot_result = try self.data.dot(other.data, allocator);

            // set the gradient dimensions to match the operands
            const result = try allocator.create(Self);
            result.* = Self{
                .data = dot_result,
                .grad = if (self.requires_grad) try dtype.zerosLike(self.data, allocator) else null,
                .requires_grad = self.requires_grad,
                .op = .DOT,
                .children = .{ self, other },
            };
            return result;
        }

        pub fn matvec(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            var result = try Self.fromZarray(try self.data.matvec(other.data, false, allocator), self.requires_grad, null, allocator);
            if (self.requires_grad) result.op = .MATVEC;
            result.children = .{ self, other };
            return result;
        }

        pub fn sum(self: *const Self, allocator: std.mem.Allocator) !*Self {
            const grad_shape = if (self.requires_grad) self.grad.?.shape.* else null;
            const result = try Self.fromZarray(try self.data.sum(allocator), self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) result.op = .SUM;

            // HACK: yea... needs two children... go back to slice design or something like a tagged enum thing
            const dummy = try allocator.create(Self);
            dummy.* = Self{
                .data = try Self.dtype.initNoAlloc(&[_]T{}, &[_]usize{}, allocator),
                .grad = null,
                .requires_grad = false,
                .op = null,
                .children = null,
                .label = "dummy",
            };
            result.children = .{ self, dummy };
            return result;
        }

        pub fn backward(self: *Self, allocator: std.mem.Allocator) !void {
            if (!self.requires_grad) return;
            if (self._backward) |f| {
                f(@constCast(self), self._backward_ctx, allocator);
                return;
            }
            if (self.op) |op| {
                switch (op) {
                    .ADD => {
                        if (self.children) |children| {
                            const a = children[0];
                            const b = children[1];

                            const a_grad = try self.grad.?.unbroadcast(a.grad.?.shape, allocator);
                            const b_grad = try self.grad.?.unbroadcast(b.grad.?.shape, allocator);

                            _ = try a.grad.?._add(a_grad);
                            _ = try b.grad.?._add(b_grad);
                        }
                    },
                    .SUB => {
                        if (self.children) |children| {
                            const a = children[0];
                            const b = children[1];

                            const a_grad = try self.grad.?.unbroadcast(a.grad.?.shape, allocator);
                            const b_grad = try self.grad.?.unbroadcast(b.grad.?.shape, allocator);

                            _ = try a.grad.?._add(a_grad);
                            _ = try b.grad.?._sub(b_grad);
                        }
                    },
                    .MUL => {
                        if (self.children) |children| {
                            const a = children[0];
                            const b = children[1];

                            // (dL/dy) * (dy/da), (dL/dy) * (dy/db)
                            const a_grad_value = try b.data.mul(self.grad.?, allocator);
                            const b_grad_value = try a.data.mul(self.grad.?, allocator);

                            // const a_grad = try a.grad.?.unbroadcast(a_grad_value.shape, allocator);
                            // const b_grad = try b.grad.?.unbroadcast(b_grad_value.shape, allocator);
                            const a_grad = try a_grad_value.unbroadcast(a.grad.?.shape, allocator);
                            const b_grad = try b_grad_value.unbroadcast(b.grad.?.shape, allocator);

                            _ = try a.grad.?._add(a_grad);
                            _ = try b.grad.?._add(b_grad);
                        }
                    },
                    .DIV => {
                        if (self.children) |children| {
                            const a = children[0];
                            const b = children[1];

                            // Gradient for division: (dL/dy) * (dy/da) and (dL/dy) * (dy/db)
                            const a_grad_value = try self.grad.?.div(b.data, allocator);
                            const b_grad_value = try self.grad.?.mul(a.data, allocator);
                            const neg_b_grad_value = try b_grad_value.div(try b.data.mul(b.data, allocator), allocator);

                            const a_grad = try a_grad_value.unbroadcast(a.grad.?.shape, allocator);
                            const b_grad = try neg_b_grad_value.unbroadcast(b.grad.?.shape, allocator);

                            _ = try a.grad.?._add(a_grad);
                            _ = try b.grad.?._sub(b_grad);
                        }
                    },
                    .MATMUL => {
                        if (self.children) |children| {
                            var A = children[0].data;
                            const B = children[1].data;
                            const grad_A = try self.grad.?.matmul(B, false, true, allocator);
                            _ = try children[0].grad.?._add(grad_A);
                            const grad_B = try A.matmul(self.grad.?, true, false, allocator);
                            _ = try children[1].grad.?._add(grad_B);
                        }
                    },
                    .DOT => {
                        if (self.children) |children| {
                            var a = children[0];
                            var b = children[1];
                            const grad_a = try b.data.mul(self.grad.?, allocator);
                            const grad_b = try a.data.mul(self.grad.?, allocator);
                            _ = try a.grad.?._add(grad_a);
                            _ = try b.grad.?._add(grad_b);
                        }
                    },
                    .MATVEC => {
                        if (self.children) |children| {
                            var A = children[0].data;
                            const x = children[1].data;

                            //  L(y), y = Ax, dL/dA = (dL/dy)(dy/dA) = (dL/dy)x'
                            const grad_A = try self.grad.?.outer(x, allocator);
                            _ = try children[0].grad.?._add(grad_A);

                            //  L(y), y = Ax, dL/dx = (dL/dy)(dy/dx) = A'(dL/dy)
                            const grad_x = try A.matvec(self.grad.?, true, allocator);
                            _ = try children[1].grad.?._add(grad_x);
                        }
                    },
                    // },
                    .SUM => {
                        if (self.children) |children| {
                            const child = children[0];
                            _ = try child.grad.?._add(self.grad.?);
                        }
                    },
                    else => @panic("Not yet implemented."),
                }
            }
        }

        /// Prints dynamic compuation graph in d2 format with ops as and operands as nodes
        pub fn print_arrows(self: Self) void {
            if (self.children) |children| {
                for (children) |elem| {
                    std.debug.print("{?s}<-{?s}", .{ self.label, elem.label });
                    const symbol = switch (self.op.?) {
                        Op.ADD => ": +",
                        Op.SUB => ": -",
                        Op.MUL => ": x",
                        Op.DIV => ": /",
                        Op.SUM => ": ++",
                        Op.MATMUL => ": A@M",
                        Op.MATVEC => ": A@x",
                        else => std.debug.panic("Unsupported op {?}\n", .{self.op}),
                    };
                    std.debug.print("{?s}\n", .{symbol});
                }
                for (children) |elem| {
                    elem.print_arrows();
                }
            } else {
                std.debug.print("{?s}\n", .{self.label});
            }
        }
    };
}

/// Manages the overall graph, allows for a more memory efficient abstraction
/// where the data structures used for traversing the graph during backprop
/// can be managed independently and reused across training steps
pub fn Loss(comptime T: type) type {
    return struct {
        const Self = @This();
        sorted_nodes: std.ArrayList(*const T),
        visited_nodes: std.AutoHashMap(*const T, void),
        eager_teardown: bool = false,
        grad_clip_enabled: bool = settings.grad_clip_enabled,
        grad_clip_max_norm: f32 = settings.grad_clip_max_norm,
        grad_clip_delta: f32 = settings.grad_clip_delta,

        pub fn init(alloc: std.mem.Allocator) *Self {
            const self = alloc.create(Self) catch unreachable;
            self.* = Self{
                .sorted_nodes = std.ArrayList(*const T).init(alloc),
                .visited_nodes = std.AutoHashMap(*const T, void).init(alloc),
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.sorted_nodes.deinit();
            self.visited_nodes.deinit();
        }

        fn topo(self: *Self, node: *const T) void {
            const gopr = self.visited_nodes.getOrPut(node) catch unreachable;
            if (!gopr.found_existing) {
                if (node.children) |children| {
                    for (children) |child| {
                        self.topo(child);
                    }
                }
                self.sorted_nodes.append(node) catch unreachable;
            }
        }

        // Must init grad on root node before backprop
        pub fn backward(self: *Self, node: *const T, alloc: std.mem.Allocator) !void {
            self.sorted_nodes.clearRetainingCapacity();
            self.visited_nodes.clearRetainingCapacity();
            self.topo(node);
            const nodes = self.sorted_nodes.items;
            for (0..nodes.len) |i| {
                const curr_node = nodes[nodes.len - i - 1];
                if (curr_node.requires_grad) {
                    try @constCast(curr_node).backward(alloc);
                    if (self.grad_clip_enabled and curr_node.requires_grad) {
                        if (curr_node.grad) |_| {
                            curr_node.clip_grad_norm_delta(.{ .max_norm = self.grad_clip_max_norm, .delta = self.grad_clip_delta });
                        }
                    }
                    // if eager_teardown, immediately destroy node. note that deinit is designed to not cascade recursively,
                    // it just destroys the current tensor and not the children
                    if (!curr_node.acquired and self.eager_teardown) curr_node.deinit(alloc);
                }
            }
        }
    };
}

pub fn SGD(comptime T: type) type {
    return struct {
        const Self = @This();
        lr: T,

        pub fn step(self: Self, params: []const *NDTensor(T)) void {
            // lol. go to bed.
            for (params) |param| {
                // param.data._sub(param.grad.?._mul(self.lr)); // TODO: really need to implement scalar ops...
                for (0..param.data.data.len) |j| {
                    param.data.data[j] -= self.lr * param.grad.?.data[j];
                }
            }
        }
    };
}

test "tensor/GraphManager/sum" {
    std.debug.print("{s} sum {s}\n", .{ "-" ** 5, "-" ** 5 });
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;
    const Tensor = NDTensor(T);

    const input = try Tensor.init(&[_]T{ 1, 2, 3, 4 }, &[_]usize{4}, true, alloc);
    var sum_result = try input.sum(alloc);

    try std.testing.expectEqualSlices(T, &[_]T{10}, sum_result.data.data);

    // Backward pass
    var gm = Loss(Tensor).init(alloc);
    defer gm.deinit();
    sum_result.grad.?.fill(1.0);
    try gm.backward(sum_result, alloc);

    const expected_grad = &[_]T{ 1, 1, 1, 1 };
    try std.testing.expectEqualSlices(T, expected_grad, input.grad.?.data);
}

test "tensor/NDTensor index, add, div" {
    std.debug.print("{s} index-add-div {s}\n", .{ "-" ** 5, "-" ** 5 });
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const shape = &[_]usize{ 2, 3 };
    const Tensor = NDTensor(f32);

    // 1 2 3
    // 4 5 6
    var t1 = try Tensor.init(@constCast(&[_]f32{ 1, 2, 3, 4, 5, 6 }), shape, false, alloc);

    // 1 2 3
    // 4 5 23
    try t1.set(&[_]usize{ 1, 2 }, -5);
    t1.print();
    // std.debug.print("{d}\n", .{t1.get(&[_]usize{ 1, 2 })});
    // std.debug.print("{d}\n", .{t1.indexToPos(5, alloc)});

    const t2 = try Tensor.init(@constCast(&[_]f32{ 10, 20, 30, 40, 50, 60 }), shape, false, alloc);
    t2.print();
    const t3 = try t1.add(t2, alloc);
    t3.print();

    std.debug.print("{d}\n", .{t3.data.data});
    t3.print();
    std.debug.print("{?any}\n", .{t3.children});

    var t4 = try t3.div(t3, alloc);
    t4.print();
}

test "tensor/GraphManager/addback" {
    std.debug.print("{s} gm/addback {s}\n", .{ "-" ** 5, "-" ** 5 });
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const shape = &[_]usize{1};
    const T = f32;
    const Tensor = NDTensor(T);

    var t1 = try Tensor.init(@constCast(&[_]T{2}), shape, true, alloc);
    const t2 = try Tensor.init(@constCast(&[_]T{3}), shape, true, alloc);
    // t3 = t1 + t2;
    // dt3/dt1 = 1, dt3/dt2 = 1
    var t3 = try t1.add(t2, alloc);
    t1.acquire();
    t2.acquire();
    t3.acquire();

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad = try Tensor.dtype.init(@constCast(&[_]T{1}), shape, alloc);
    try gm.backward(t3, alloc);
    try std.testing.expectEqualDeep(&[_]T{1.0}, t1.grad.?.data);
    try std.testing.expectEqualDeep(&[_]T{1.0}, t2.grad.?.data);
}

test "tensor/GraphManager/mulback" {
    std.debug.print("{s} gm/mulback {s}\n", .{ "-" ** 5, "-" ** 5 });
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const shape = &[_]usize{1};
    const T = f32;
    const Tensor = NDTensor(T);

    var t1 = try Tensor.init(@constCast(&[_]T{2}), shape, true, alloc);
    const t2 = try Tensor.init(@constCast(&[_]T{3}), shape, true, alloc);
    // t3 = t1 * t2;
    // dt3/dt1 = t2, dt3/dt2 = t1
    var t3 = try t1.mul(t2, alloc);
    t1.acquire();
    t2.acquire();
    t3.acquire();

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad = try Tensor.dtype.init(@constCast(&[_]T{1}), shape, alloc);
    try gm.backward(t3, alloc);
    try std.testing.expectEqualDeep(t2.data.data, t1.grad.?.data);
    try std.testing.expectEqualDeep(t1.data.data, t2.grad.?.data);
}

test "tensor/GraphManager/moreback" {
    std.debug.print("{s} gm/moreback {s}\n", .{ "-" ** 5, "-" ** 5 });
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const shape = &[_]usize{2};
    const T = f32;
    const Tensor = NDTensor(T);

    var w = try Tensor.init(@constCast(&[_]f32{ 3, 2 }), shape, true, alloc);
    const b = try Tensor.init(@constCast(&[_]f32{ 1, 1 }), shape, true, alloc);
    const x = try Tensor.init(@constCast(&[_]f32{ 4, 4 }), shape, true, alloc);

    // h = w*x + b
    // dh/dw = x, dh/db = 1
    var temp = try w.mul(x, alloc);
    var h = try temp.add(b, alloc);

    var backprop_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const backprop_alloc = backprop_arena.allocator();
    defer backprop_arena.deinit();

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    h.grad = try Tensor.dtype.init(@constCast(&[_]T{ 1, 1 }), shape, alloc);
    try gm.backward(h, backprop_alloc);
    if (!backprop_arena.reset(.retain_capacity)) @panic("reset failed.\n");
    try std.testing.expectEqualSlices(T, x.data.data, w.grad.?.data);
    try std.testing.expectEqualSlices(T, &[_]T{ 1.0, 1.0 }, b.grad.?.data);

    // 2 x 1
    const shape2 = &[_]usize{ 2, 1 };
    w.grad.?.fill(0);
    b.grad.?.fill(0);
    x.grad.?.fill(0);
    _ = try w.reshape(shape2);
    _ = try b.reshape(shape2);
    _ = try x.reshape(shape2);
    // h = w*x + b
    // dh/dw = x, dh/db = 1
    temp = try w.mul(x, alloc);
    h = try temp.add(b, alloc);

    var gm2 = Loss(Tensor).init(arena.allocator());
    defer gm2.deinit();
    h.grad.?.fill(1);
    try gm.backward(h, alloc);
    if (!backprop_arena.reset(.retain_capacity)) @panic("reset failed.\n");
    try std.testing.expectEqualSlices(T, x.data.data, w.grad.?.data);
    try std.testing.expect(std.mem.allEqual(T, b.grad.?.data, 1));
    h.print();
}

test "tensor/GraphManager/divback" {
    std.debug.print("{s} gm/divback {s}\n", .{ "-" ** 5, "-" ** 5 });
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;
    const Tensor = NDTensor(T);
    const shape = &[_]usize{2};

    var t1 = try Tensor.init(@constCast(&[_]T{ 4, 9 }), shape, true, alloc);
    const t2 = try Tensor.init(@constCast(&[_]T{ 2, 3 }), shape, true, alloc);
    var t3 = try t1.div(t2, alloc);
    t1.acquire();
    t2.acquire();
    t3.acquire();

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad = try Tensor.dtype.init(@constCast(&[_]T{ 1, 1 }), shape, alloc);
    try gm.backward(t3, alloc);

    // Expected gradients for t1 and t2
    const expected_grad_t1 = &[_]T{ 1.0 / 2.0, 1.0 / 3.0 }; // 1 / b
    const expected_grad_t2 = &[_]T{ -4.0 / 4.0, -9.0 / 9.0 }; // -a / b^2

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}

test "tensor/GraphManager/matmul_backward" {
    std.debug.print("{s} gm/mmback {s}\n", .{ "-" ** 5, "-" ** 5 });
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;
    const Tensor = NDTensor(T);
    const shape = &[_]usize{ 2, 2 };

    var t1 = try Tensor.init(@constCast(&[_]T{ 1, 2, 3, 4 }), shape, true, alloc);
    const t2 = try Tensor.init(@constCast(&[_]T{ 1, 0, 0, 1 }), shape, true, alloc);
    var t3 = try t1.matmul(t2, alloc, .{});
    t1.acquire();
    t2.acquire();
    t3.acquire();

    var gm = Loss(Tensor).init(arena.allocator());
    // grad clipping will cause differences
    gm.grad_clip_enabled = false;
    defer gm.deinit();
    t3.grad.?.fill(1.0);
    try gm.backward(t3, alloc);

    const expected_grad_t1 = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2 = &[_]T{ 4, 4, 6, 6 };

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}

test "tensor/GraphManager/matvec_backward" {
    std.debug.print("{s} gm/mvback {s}\n", .{ "-" ** 5, "-" ** 5 });
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;
    const Tensor = NDTensor(T);
    const shape_mat = &[_]usize{ 2, 2 };
    const shape_vec = &[_]usize{2};

    // [1, 2] [1]
    // [3, 4] [1]
    // grad = [1, 1]'
    // dl/dA = grad * [1, 1] = [[2, 2], [2, 2]]
    // dl/dx = A' * grad = [4, 6]'
    var t1 = try Tensor.init(@constCast(&[_]T{ 1, 2, 3, 4 }), shape_mat, true, alloc);
    const t2 = try Tensor.init(@constCast(&[_]T{ 1, 1 }), shape_vec, true, alloc);
    var t3 = try t1.matvec(t2, alloc);
    t1.acquire();
    t2.acquire();
    t3.acquire();

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad.?.fill(1.0);
    try gm.backward(t3, alloc);

    const expected_grad_t1 = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2 = &[_]T{ 4, 6 };

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}

test "tensor/GraphManager/dot_backward" {
    std.debug.print("{s} gm/dotback {s}\n", .{ "-" ** 5, "-" ** 5 });
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;
    const Tensor = NDTensor(T);
    const shape = &[_]usize{3};

    var t1 = try Tensor.init(@constCast(&[_]T{ 1, 2, 3 }), shape, true, alloc);
    const t2 = try Tensor.init(@constCast(&[_]T{ 4, 5, 6 }), shape, true, alloc);
    var t3 = try t1.dot(t2, alloc);
    t1.acquire();
    t2.acquire();
    t3.acquire();

    var gm = Loss(Tensor).init(arena.allocator());
    defer gm.deinit();
    t3.grad.?.fill(1.0);
    try gm.backward(t3, alloc);

    const expected_grad_t1 = &[_]T{ 4, 5, 6 };
    const expected_grad_t2 = &[_]T{ 1, 2, 3 };

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}
