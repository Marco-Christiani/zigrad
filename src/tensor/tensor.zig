// TODO: implement view(), transpose(), and permute(), where the latter two mutate the shape
// TODO: better operation abstraction
// TODO: print graph, just take from the existing impl
const std = @import("std");
const zarray = @import("zarray.zig");
const Shape = zarray.Shape;
const NDArray = zarray.NDArray;
const ZarrayError = zarray.ZarrayError;
const settings = @import("zigrad").settings;

const log = std.log.scoped(.zigrad_tensor);

pub const Op = enum {
    ADD,
    SUB,
    MUL,
    DIV,
    POW,
    TANH,
    MATMUL_AB,
    MATMUL_AtB,
    MATMUL_ABt,
    DOT,
    MATVEC,
    SUM,
    RESHAPE,
    MAX,
    EXP, // TODO:
    SOFTMAX, // TODO:
};

pub fn NDTensor(comptime T: type) type {
    return struct {
        pub const dtype = NDArray(T);
        const Self = @This();
        data: *dtype,
        op: ?Op = null,
        children: ?[]*const Self = null,
        label: ?[]const u8 = null,
        grad: ?*dtype = null,
        requires_grad: bool,
        acquired: bool = false,
        allocator: std.mem.Allocator,
        _backward: ?*const fn (*Self, std.mem.Allocator) anyerror!void = null,
        _backward_ctx: ?*anyopaque = null,

        /// Values are copied. COM.
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
                .allocator = allocator,
            };
            return result;
        }

        pub fn deinit(self: *const Self) void {
            if (self.acquired) std.debug.panic("Attempt to deinit an acquired tensor.", .{});
            // log.debug("deinit().data {?s}", .{self.label});
            self.data.deinit(self.allocator);
            if (self.grad) |g| {
                // log.debug("deinit().grad {?s}", .{self.label});
                g.deinit(self.allocator);
            }

            // TODO: verify this is heap first, possibly by checking alignment, not sure.
            // if (self.label) |l| allocator.free(l);

            if (self.children) |c| self.allocator.free(c);
            self.allocator.destroy(self);
        }

        pub fn teardown(self: *const Self) void {
            log.debug("teardown {?s}\n", .{self.label});
            if (self.acquired) std.debug.panic("Attempt to deinit an acquired tensor.", .{});
            if (self.children) |children| {
                for (children) |c| {
                    std.log.debug("accessing child", .{});
                    if (!c.acquired) c.teardown() else log.debug("skipping acquired tensor in teardown label={?s}\n", .{c.label});
                }
            }
            log.debug("teardown()->deinit() {?s}", .{self.label});
            self.deinit();
        }

        pub fn acquire(self: *Self) void {
            self.acquired = true;
        }

        pub fn release(self: *Self) void {
            self.acquired = false;
        }

        /// Values are not copied, grad_shape is copied (or allocated). COM.
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
                .allocator = allocator,
            };
            return result;
        }

        /// shape is copied. COM.
        pub fn empty(shape: []const usize, requires_grad: bool, allocator: std.mem.Allocator) !*Self {
            const result = try allocator.create(Self);
            result.* = Self{
                .data = try dtype.empty(shape, allocator),
                .grad = blk: {
                    if (requires_grad) {
                        const g = try dtype.empty(shape, allocator);
                        g.fill(0);
                        break :blk g;
                    } else break :blk null;
                },
                .requires_grad = requires_grad,
                .allocator = allocator,
            };
            return result;
        }

        pub fn copy(self: *const Self, allocator: std.mem.Allocator) !*Self {
            const result = try allocator.create(Self);

            result.data = try self.data.copy(allocator);

            if (self.grad) |grad| {
                result.grad = try grad.copy(allocator);
            } else {
                result.grad = null;
            }

            result.op = self.op;
            result.children = self.children; // note: children pointers
            result.label = if (self.label) |label| try allocator.dupe(u8, label) else null;
            result.requires_grad = self.requires_grad;
            result._backward = self._backward;
            result._backward_ctx = self._backward_ctx;

            return result;
        }

        pub fn logShape(self: *const Self, comptime msg: ?[]const u8) void {
            log.debug("{s}{s} data shape: {d} grad shape: {?d}", .{
                if (msg) |n| n else "",
                if (self.label) |l| l else "",
                self.data.shape.shape,
                if (self.grad) |g| g.shape.shape else null,
            });
        }

        /// in-place
        pub fn _reshape(self: *Self, shape: []const usize) !*Self {
            try self.data._reshape(shape);
            if (self.grad) |g| {
                try g._reshape(shape);
            }
            return self;
        }

        /// Copies. COM.
        pub fn reshape(self: *Self, new_shape: []const usize) !*Self {
            const result = try Self.init(self.data.data, new_shape, self.requires_grad, self.allocator);
            if (self.requires_grad) {
                result.op = .RESHAPE;
                try result.setChildren(@constCast(&[_]*const Self{self}));
            }
            return result;
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

        /// COM
        fn indexToPos(self: Self, index: usize, allocator: std.mem.Allocator) []const usize {
            return self.data.offsetToPos(index, allocator);
        }

        /// COM for container, values are a view into self.
        pub fn slice(self: *const Self, ranges: []const zarray.Range, allocator: std.mem.Allocator) !*Self {
            const sliced_data = try self.data.slice(ranges);
            const result = try allocator.create(Self);
            result.* = Self{
                .data = sliced_data,
                .grad = if (self.requires_grad) try dtype.empty(sliced_data.shape.shape, allocator) else null, // TODO: slice grad?
                .requires_grad = self.requires_grad,
                .op = null,
                .children = null,
                .label = null,
                .acquired = false,
                .allocator = allocator,
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
                                std.debug.assert(children.len == 2);
                                const original = children[0];
                                const values_ = children[1];

                                try original.grad.?.setSlice(ranges, g.*);
                                _ = try values_.grad.?._add(g);
                            }
                        }
                    }
                }.backward;

                result.children = self.setChildren(@constCast(&[_]*const Self{ self, values }));
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

        pub fn setChildren(self: *Self, children: []*const Self) !void {
            self.children = try self.allocator.dupe(*const Self, children);
        }

        /// Element-wise addition. COM.
        pub fn add(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const out = try self.data.add(other.data, allocator);
            const grad_shape = if (self.requires_grad or other.requires_grad) out.shape.* else null;
            var result = try Self.fromZarray(out, self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) {
                result.op = .ADD;
                try result.setChildren(@constCast(&[_]*const Self{ self, other }));
            }
            return result;
        }

        /// Element-wise subtraction. COM.
        pub fn sub(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const out = try self.data.sub(other.data, allocator);
            const grad_shape = if (self.requires_grad or other.requires_grad) out.shape.* else null;
            var result = try Self.fromZarray(out, self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) {
                result.op = .SUB;
                try result.setChildren(@constCast(&[_]*const Self{ self, other }));
            }
            return result;
        }

        /// Element-wise multiplication. COM.
        pub fn mul(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const out = try self.data.mul(other.data, allocator);
            const grad_shape = if (self.requires_grad or other.requires_grad) out.shape.* else null;
            var result = try Self.fromZarray(out, self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) {
                result.op = .MUL;
                try result.setChildren(@constCast(&[_]*const Self{ self, other }));
            }
            return result;
        }

        /// Element-wise division. COM.
        pub fn div(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const out = try self.data.div(other.data, allocator);
            const grad_shape = if (self.requires_grad or other.requires_grad) out.shape.* else null;
            var result = try Self.fromZarray(out, self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) {
                result.op = .DIV;
                try result.setChildren(@constCast(&[_]*const Self{ self, other }));
            }
            return result;
        }

        pub fn max(self: *const Self, allocator: std.mem.Allocator) !*Self {
            const max_val = try self.data.max();
            const result = try Self.init(&[_]T{max_val}, &[_]usize{1}, self.requires_grad, allocator);
            if (self.requires_grad) {
                result.op = .MAX;
                try result.setChildren(@constCast(&[_]*const Self{self}));
            }
            return result;
        }

        pub fn exp(self: *const Self, allocator: std.mem.Allocator) !*Self {
            const out = try self.data.exp(allocator);
            const grad_shape = if (self.requires_grad) out.shape.* else null;
            const result = try Self.fromZarray(out, self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) {
                result.op = .EXP;
                try result.setChildren(@constCast(&[_]*const Self{self}));
            }
            return result;
        }

        pub fn softmax(self: *const Self, allocator: std.mem.Allocator) !*Self {
            _ = allocator; // autofix
            _ = self; // autofix
            @compileError("not yet implemented");
            // const max_val = try self.data.max(allocator);
            // const shifted = try self.data._sub(max_val, allocator);
            // const exp_vals = try shifted.data._exp(allocator);
            // const sum_exp = try exp_vals.sum(allocator);
            // const result = try exp_vals.div(sum_exp, allocator);
            //
            // if (self.requires_grad) {
            //     result.op = .SOFTMAX;
            //     try result.setChildren(&[_]*const Self{self});
            // }
            // return result;
        }

        pub const MmOptions = struct { trans_a: bool = false, trans_b: bool = false };

        /// COM
        pub fn matmul(self: *const Self, other: *const Self, allocator: std.mem.Allocator, opts: MmOptions) !*Self {
            const out = try self.data.matmul(other.data, opts.trans_a, opts.trans_b, allocator);
            const grad_shape = if (self.requires_grad or other.requires_grad) out.shape.* else null;
            var result = try Self.fromZarray(out, self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) {
                if (!opts.trans_a and !opts.trans_b) {
                    result.op = .MATMUL_AB;
                } else if (opts.trans_a and !opts.trans_b) {
                    result.op = .MATMUL_AtB;
                } else if (!opts.trans_a and opts.trans_b) {
                    result.op = .MATMUL_ABt;
                } else {
                    @panic("No AtBt.");
                }
                try result.setChildren(@constCast(&[_]*const Self{ self, other }));
            }
            return result;
        }

        // pub fn dot(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
        //     var result = try Self.fromZarray(try self.data.dot(other.data, allocator), self.requires_grad, allocator);
        //     if (self.requires_grad) result.op = .DOT;
        //     result.children = .{ self, other };
        //     return result;
        // }

        /// COM
        pub fn dot(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const dot_result = try self.data.dot(other.data, allocator);

            // set the gradient dimensions to match the operands
            const result = try allocator.create(Self);
            result.* = Self{
                .data = dot_result,
                .grad = if (self.requires_grad) try dtype.zerosLike(self.data, allocator) else null,
                .requires_grad = self.requires_grad,
                .op = .DOT,
                .children = if (self.requires_grad) try self.allocator.dupe(*const Self, @constCast(&[_]*const Self{ self, other })) else null,
            };
            return result;
        }

        /// COM
        pub fn matvec(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            var result = try Self.fromZarray(try self.data.matvec(other.data, false, allocator), self.requires_grad, null, allocator);
            if (self.requires_grad) {
                result.op = .MATVEC;
                try result.setChildren(@constCast(&[_]*const Self{ self, other }));
            }
            return result;
        }

        /// COM
        pub fn sum(self: *const Self, allocator: std.mem.Allocator) !*Self {
            const grad_shape = if (self.requires_grad) self.grad.?.shape.* else null;
            const result = try Self.fromZarray(try self.data.sum(allocator), self.requires_grad, grad_shape, allocator);
            if (self.requires_grad) {
                result.op = .SUM;
                try result.setChildren(@constCast(&[_]*const Self{self}));
            }
            return result;
        }

        pub fn setBackward(self: *Self, backward_fn: *const fn (*Self, std.mem.Allocator) anyerror!void, ctx: ?*anyopaque) void {
            self._backward = backward_fn;
            self._backward_ctx = ctx;
        }

        pub fn backward(self: *Self, allocator: std.mem.Allocator) !void {
            if (!self.requires_grad) return;
            if (self._backward) |f| {
                try f(@constCast(self), allocator);
                return;
            }
            if (self.op) |op| {
                switch (op) {
                    .ADD => {
                        if (self.children) |children| {
                            const a = children[0];
                            const b = children[1];

                            const a_grad = try (try self.grad.?.copy(allocator)).unbroadcast(a.grad.?.shape, allocator);
                            const b_grad = try (try self.grad.?.copy(allocator)).unbroadcast(b.grad.?.shape, allocator);

                            _ = try a.grad.?._add(a_grad);
                            _ = try b.grad.?._add(b_grad);
                        }
                    },
                    .SUB => {
                        if (self.children) |children| {
                            const a = children[0];
                            const b = children[1];

                            const a_grad = try (try self.grad.?.copy(allocator)).unbroadcast(a.grad.?.shape, allocator);
                            const b_grad = try (try self.grad.?.copy(allocator)).unbroadcast(b.grad.?.shape, allocator);

                            defer { // TODO: unbroadcast memory management elsewhere as well
                                a_grad.deinit(self.allocator);
                                b_grad.deinit(self.allocator);
                            }

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

                            defer a_grad_value.deinit(allocator);
                            defer b_grad_value.deinit(allocator);

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

                            defer a_grad_value.deinit(allocator);
                            defer b_grad_value.deinit(allocator);

                            const bsq = try b.data.mul(b.data, allocator);
                            const neg_b_grad_value = try b_grad_value.div(bsq, allocator);

                            defer bsq.deinit(allocator);
                            defer neg_b_grad_value.deinit(allocator);

                            const a_grad = try a_grad_value.unbroadcast(a.grad.?.shape, allocator);
                            const b_grad = try neg_b_grad_value.unbroadcast(b.grad.?.shape, allocator);

                            _ = try a.grad.?._add(a_grad);
                            _ = try b.grad.?._sub(b_grad);
                        }
                    },
                    .MATMUL_AB => {
                        if (self.children) |children| {
                            var A = children[0].data;
                            const B = children[1].data;
                            const grad_A = try self.grad.?.matmul(B, false, true, allocator);
                            defer grad_A.deinit(allocator);
                            _ = try children[0].grad.?._add(grad_A);
                            const grad_B = try A.matmul(self.grad.?, true, false, allocator);
                            defer grad_B.deinit(allocator);
                            _ = try children[1].grad.?._add(grad_B);
                        }
                    },
                    .MATMUL_AtB => {
                        if (self.children) |children| {
                            var A = children[0].data;
                            const B = children[1].data;

                            // grad_A = B * grad_C^T
                            var grad_C_transposed = try self.grad.?.transpose(allocator);
                            defer grad_C_transposed.deinit(allocator);
                            const grad_A = try B.matmul(grad_C_transposed, false, false, allocator);
                            defer grad_A.deinit(allocator);
                            _ = try children[0].grad.?._add(grad_A);

                            // grad_B = A * grad_C
                            const grad_B = try A.matmul(self.grad.?, false, false, allocator);
                            defer grad_B.deinit(allocator);
                            _ = try children[1].grad.?._add(grad_B);
                        }
                    },
                    .MATMUL_ABt => {
                        if (self.children) |children| {
                            var A = children[0].data;
                            const B = children[1].data;

                            // grad_A = grad_C * B
                            const grad_A = try self.grad.?.matmul(B, false, false, allocator);
                            defer grad_A.deinit(allocator);
                            _ = try children[0].grad.?._add(grad_A);

                            // grad_B = A^T * grad_C
                            var grad_B = try A.matmul(self.grad.?, true, false, allocator);
                            defer grad_B.deinit(allocator);

                            // Transpose grad_B before adding
                            var grad_B_transposed = try grad_B.transpose(allocator);
                            defer grad_B_transposed.deinit(allocator);

                            _ = try children[1].grad.?._add(grad_B_transposed);
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
                    .SUM => {
                        if (self.children) |children| {
                            const child = children[0];
                            _ = try child.grad.?._add(self.grad.?);
                        }
                    },
                    .RESHAPE => {
                        if (self.children) |children| {
                            const original_shape = children[0].data.shape;
                            try self.grad.?._reshape(original_shape.shape);
                            _ = try children[0].grad.?._add(self.grad.?);
                        }
                    },
                    .MAX => {
                        if (self.children) |children| {
                            const child = children[0];
                            const max_val = self.data.data[0];
                            for (child.data.data, 0..) |val, i| {
                                if (val == max_val) {
                                    child.grad.?.data[i] += self.grad.?.data[0];
                                }
                            }
                        }
                    },
                    .EXP => {
                        if (self.children) |children| {
                            const child = children[0];
                            for (self.data.data, self.grad.?.data, 0..) |exp_val, grad_val, i| {
                                child.grad.?.data[i] += exp_val * grad_val;
                            }
                        }
                    },
                    .SOFTMAX => {
                        if (self.children) |children| {
                            const child = children[0];
                            const softmax_output = self.data;
                            const incoming_grad = self.grad.?;

                            var sum_grad: T = 0;
                            for (softmax_output.data, incoming_grad.data) |s, g| {
                                sum_grad += s * g;
                            }

                            for (softmax_output.data, incoming_grad.data, 0..) |s, g, i| {
                                child.grad.?.data[i] += s * (g - sum_grad);
                            }

                            // try child.grad.?._add(child.grad.?);
                        }
                    },
                    else => std.debug.panic("Op {s} is not yet implemented.", .{@tagName(op)}),
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
        allocator: std.mem.Allocator,
        sorted_nodes: std.ArrayList(*const T),
        visited_nodes: std.AutoHashMap(*const T, void),
        eager_teardown: bool = false,
        grad_clip_enabled: bool = settings.grad_clip_enabled,
        grad_clip_max_norm: f32 = settings.grad_clip_max_norm,
        grad_clip_delta: f32 = settings.grad_clip_delta,

        pub const LossConfig = struct {
            eager_teardown: bool = false,
            grad_clip_enabled: bool = settings.grad_clip_enabled,
            grad_clip_max_norm: f32 = settings.grad_clip_max_norm,
            grad_clip_delta: f32 = settings.grad_clip_delta,
        };

        pub fn init(allocator: std.mem.Allocator, opts: LossConfig) Self {
            return Self{
                .allocator = allocator,
                .sorted_nodes = std.ArrayList(*const T).init(allocator),
                .visited_nodes = std.AutoHashMap(*const T, void).init(allocator),
                .grad_clip_enabled = opts.grad_clip_enabled,
                .grad_clip_max_norm = opts.grad_clip_max_norm,
                .grad_clip_delta = opts.grad_clip_delta,
            };
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
                    log.debug("backward: {?s}", .{curr_node.label});
                    try @constCast(curr_node).backward(alloc);
                    if (self.grad_clip_enabled and curr_node.requires_grad) {
                        if (curr_node.grad) |_| {
                            curr_node.clip_grad_norm_delta(.{ .max_norm = self.grad_clip_max_norm, .delta = self.grad_clip_delta });
                        }
                    }
                    // if eager_teardown, immediately destroy node. note that deinit is designed to not cascade recursively,
                    // it just destroys the current tensor and not the children
                    if (!curr_node.acquired and self.eager_teardown) curr_node.deinit();
                } else {
                    log.debug("Skipping node {?s}", .{node.label});
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
    _ = try w._reshape(shape2);
    _ = try b._reshape(shape2);
    _ = try x._reshape(shape2);
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
