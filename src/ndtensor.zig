// TODO: implement view(), transpose(), and permute(), where the latter two mutate the shape
// TODO: better operation abstraction
// TODO: print graph, just take from the existing impl
const std = @import("std");
const zg = @import("root.zig");
const settings = zg.settings;

const ndarray = @import("ndarray.zig");
const Range = ndarray.Range;
const Shape = ndarray.Shape;
const NDArray = ndarray.NDArray;
const GraphManager = @import("graph_manager.zig").GraphManager;

const log = std.log.scoped(.zg_tensor);

pub const Op = enum {
    ADD,
    SUB,
    MUL,
    DIV,
    POW, // TODO:
    TANH, // TODO:
    MATMUL_AB,
    MATMUL_AtB,
    MATMUL_ABt,
    DOT,
    MATVEC,
    SUM,
    RESHAPE,
    TRANSPOSE,
    MAX,
    EXP, // TODO:
};

pub fn NDTensor(comptime T: type) type {
    return struct {
        pub const dtype = NDArray(T);
        const Self = @This();
        data: *dtype,
        op: ?Op = null,
        children: ?[]const *const Self = null,
        label: ?[]const u8 = null,
        grad: ?*dtype = null,
        requires_grad: bool,
        acquired: bool = false, // not inherited
        allocator: std.mem.Allocator,
        _backward: ?*const fn (Self, std.mem.Allocator) anyerror!void = null, // see notes below
        _backward_ctx: ?*anyopaque = null,

        /// Values and shape are allocated. COM.
        pub fn init(values: []const T, shape: ?[]const usize, requires_grad: bool, allocator: std.mem.Allocator) !*Self {
            const self_shape = shape orelse &[_]usize{values.len};
            const self = try allocator.create(Self);
            self.* = Self{
                .data = try dtype.init(values, self_shape, allocator),
                .grad = if (requires_grad) try dtype.zeros(self_shape, allocator) else null,
                .requires_grad = requires_grad,
                .allocator = allocator,
            };
            return self;
        }

        /// Shape is allocated. COM.
        pub fn empty(shape: []const usize, requires_grad: bool, allocator: std.mem.Allocator) !*Self {
            const self = try allocator.create(Self);
            self.* = Self{
                .data = try dtype.empty(shape, allocator),
                .grad = if (requires_grad) try dtype.zeros(shape, allocator) else null,
                .requires_grad = requires_grad,
                .allocator = allocator,
            };
            return self;
        }

        pub const CreateDependentOpts = struct {
            data: *dtype,
            op: ?Op = null,
            children: []const *const Self,
            label: ?[]const u8 = null,
            requires_grad: bool,
            allocator: std.mem.Allocator,
            _backward: ?*const fn (Self, std.mem.Allocator) anyerror!void = null,
            _backward_ctx: ?*anyopaque = null,
        };

        // interesting opportunity for dynamic behavior where this could be run on an instance, making self an default dep of result
        // without overloading it would have to be type introspection, I question the utility it seems like a convenience.

        /// Data is not copied. Child slice is allocated. Label is copied. Grad may be initialized as zeros. COM.
        ///
        /// ---
        /// # Assumptions
        ///   - `data` is heap.
        ///   - `children` (contents) are heap.
        ///   - `children` (slice) is stack.
        ///   - `label` is stack.
        ///
        /// # ADR
        ///   1. Intended for creating intermediates hence alloc/no-alloc specifications.
        ///   2. To support graph usage without the intention of backprop, requires_grad=false will not allocate a grad.
        ///   3. The only foreseen situation to init with a specific `grad` value would be at the tail of a graph for backprop.
        ///      This is a cold operation and thus should be done explicitly, hence no support provided here. This drives support
        ///      for mutable `grad` pointer, so the caller can move the reference (Warning: this pattern is a red flag).
        pub fn createDependent(opts: CreateDependentOpts) !*Self {
            const self = try opts.allocator.create(Self);
            self.* = Self{
                .data = opts.data,
                .op = opts.op,
                .children = try opts.allocator.dupe(*const Self, opts.children),
                .label = if (opts.label) |l| try opts.allocator.dupe(u8, l) else null,
                .grad = if (opts.requires_grad) try dtype.zeros(opts.data.shape.shape, opts.allocator) else null,
                .requires_grad = opts.requires_grad,
                .acquired = false,
                ._backward = opts._backward,
                ._backward_ctx = opts._backward_ctx,
                .allocator = opts.allocator,
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            if (self.acquired) std.debug.panic("Attempt to deinit an acquired tensor.", .{});
            // log.debug("deinit().data {?s}", .{self.label});
            self.data.deinit(self.allocator);
            if (self.grad) |g| {
                // log.debug("deinit().grad {?s}", .{self.label});
                g.deinit(self.allocator);
            }

            // TODO: verify this is heap first, possibly by checking alignment, not sure.
            if (self.label) |l| self.allocator.free(l);

            // TODO: ohhhh this is tricky...
            // if (self._backward_ctx) |ctx| {
            //     if (@hasDecl(ctx, "deinit")) ctx.deinit();
            // }
            if (self.children) |c| self.allocator.free(c);
            self.allocator.destroy(self);
        }

        pub fn teardown(self: *Self) void {
            log.debug("START teardown {?s}", .{self.label});
            if (self.acquired) std.debug.panic("Attempt to deinit an acquired tensor.", .{});
            if (self.children) |children| {
                for (children) |c| {
                    log.debug("{?s} accessing child {?s}", .{ self.label, c.label });
                    if (!c.acquired) @constCast(c).teardown() else log.warn("skipping acquired tensor in teardown label={?s}", .{c.label});
                }
            }
            log.debug("ENDING teardown()->deinit() {?s}", .{self.label});
            self.deinit();
        }

        pub fn acquire(self: *Self) void {
            self.acquired = true;
        }

        pub fn release(self: *Self) void {
            self.acquired = false;
        }

        /// Values are not allocated, grad_shape is allocated. COM.
        pub fn fromZarray(values: *dtype, requires_grad: bool, grad_shape: ?Shape, allocator: std.mem.Allocator) !*Self {
            log.warn("Consider createDependent() if you are about to set children. This may be deprecated.", .{});
            const result = try allocator.create(Self);
            const grad: ?*dtype = blk: {
                if (requires_grad) {
                    if (grad_shape) |s| {
                        log.warn("`grad_shape` may be deprecated.", .{});
                        break :blk try dtype.zeros(s.shape, allocator);
                    } else {
                        break :blk (try dtype.zeros(values.shape.shape, allocator));
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

        // NOTE: Check callsites and see if we can fix the comments here
        /// Only data and grad are copied. Child references and other metadata is not retained aside from requires_grad.
        /// Passed allocator used for allocations, not the owned one (i.e. not `@This()` one).
        ///
        /// ---
        /// # ADR
        ///   - The choice to have an allocator provided is important, intended to be used for backward
        ///   - If the tensor backing `dtype` changes its allocator ownership contract, then this needs to be changed
        pub fn clone(self: Self, allocator: std.mem.Allocator) !*Self {
            const result = try allocator.create(Self);
            errdefer allocator.destroy(result);
            result.* = Self{
                .data = try self.data.copy(allocator),
                .grad = if (self.grad) |g| try g.copy(allocator) else null,
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

        pub fn logShape(self: Self, comptime msg: ?[]const u8) void {
            log.debug("{s}{s} data shape: {d} grad shape: {?d}", .{
                if (msg) |n| n else "",
                if (self.label) |l| l else "",
                self.data.shape.shape,
                if (self.grad) |g| g.shape.shape else null,
            });
        }

        /// In-place, no backward.
        pub fn _reshape(self: *Self, shape: []const usize) !void {
            try self.data._reshape(shape);
            if (self.grad) |g| try g._reshape(shape);
        }

        /// Copies. COM.
        pub fn reshape(self: *const Self, new_shape: []const usize) !*Self {
            const result = try createDependent(.{
                .data = try self.data.copy(self.allocator),
                .op = .RESHAPE,
                .children = &[_]*const Self{self},
                .label = null,
                .requires_grad = self.requires_grad,
                .allocator = self.allocator,
                ._backward = null,
            });
            errdefer result.deinit();
            try result.data._reshape(new_shape);
            if (result.grad) |g| try g._reshape(new_shape);
            return result;
        }

        /// Copies. COM.
        pub fn transpose(self: *const Self) !*Self {
            var result = try createDependent(.{
                .data = try self.data.transpose(self.allocator),
                .op = .TRANSPOSE,
                .children = &[_]*const Self{self},
                .label = null,
                .requires_grad = false, // we will set grad ourselves, not great
                .allocator = self.allocator,
                ._backward = null,
            });
            errdefer result.deinit();
            if (self.requires_grad) result.grad = try self.grad.?.transpose(self.allocator);
            result.requires_grad = true;
            return result;
        }

        pub fn setLabel(self: *Self, comptime label: []const u8) *Self {
            if (self.label) |l| self.allocator.free(l);
            self.label = self.allocator.dupe(u8, label) catch @panic("OOM");
            return self;
        }

        pub fn fill(self: Self, val: T) void {
            self.data.fill(val);
        }

        pub fn get(self: Self, indices: []const usize) T {
            return self.data.get(indices);
        }

        pub fn set(self: Self, indices: []const usize, value: T) !void {
            try self.data.set(indices, value);
        }

        fn posToIndex(self: Self, indices: []const usize) usize {
            return self.data.posToOffset(indices);
        }

        fn flexPosToIndex(self: Self, indices: []const usize) error.InvalidIndex!usize {
            return self.data.flexPosToOffset(indices);
        }

        /// COM
        fn indexToPos(self: Self, index: usize, allocator: std.mem.Allocator) []const usize {
            return self.data.offsetToPos(index, allocator);
        }

        /// [WIP] Values and grads are views into self. Shapes are allocated and COM.
        ///
        /// ---
        /// # ADR
        ///   - It is assumed the caller wants a temporary mutable view, return by copy.
        ///   - This creates a complex situation, there is no backward for this operation and it returns a mutable view.
        ///   - Caller must set backward
        pub fn sliceRanges(self: Self, ranges: []const Range, allocator: std.mem.Allocator) !Self {
            log.err("WIP", .{});
            const sliced_data = try self.data.sliceRanges(ranges);
            return Self{
                .data = sliced_data,
                .grad = if (self.requires_grad) try self.grad.?.sliceRanges(ranges) else null,
                .requires_grad = self.requires_grad,
                .op = null,
                .children = null,
                .label = null,
                .acquired = false,
                .allocator = allocator,
                ._backward = null,
                ._backward_ctx = null,
            };
        }

        pub fn setSlice(self: Self, ranges: []const Range, values: Self) !void {
            if (self.requires_grad) {
                // need to create a new operation
                @compileError("Not implemented");
            } else {
                // if not tracking gradients can just set the values directly
                try self.data.setSliceRanges(ranges, values.data.*);
            }
        }

        pub fn print(self: Self) void {
            // self.printToWriter(std.io.getStdOut().writer());
            self.printToWriter(std.io.getStdErr().writer()) catch @panic("Failed to print tensor");
        }

        pub fn printToWriter(self: Self, writer: anytype) !void {
            try writer.print("NDTensor<{},{?s}>", .{ T, if (self.op) |o| @tagName(o) else null });
            try writer.writeAll("{data: ");
            try self.data.printToWriter(writer);
            if (self.grad) |g| {
                try writer.writeAll(" grad: ");
                try g.printToWriter(writer);
            }
            try writer.print(", requires_grad={}", .{self.requires_grad});
            if (self.label) |l| {
                try writer.print(" label={s}", .{l});
            }
            try writer.writeAll("}\n");
        }

        pub const ClipOptions = struct {
            max_norm: f32 = settings.grad_clip_max_norm,
            delta: f32 = settings.grad_clip_delta,
        };

        pub fn clip_grad_norm_delta(self: Self, opts: ClipOptions) void {
            self.grad.?.clip_norm(opts.max_norm, opts.delta);
        }

        pub fn setChildren(self: *Self, children: []const *Self) !void {
            log.warn("Deprecation warning: use createDependent.", .{});
            self.children = try self.allocator.dupe(*Self, children);
        }

        /// Element-wise addition. COM.
        pub fn add(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            return try createDependent(.{
                .data = try self.data.add(other.data, allocator),
                .op = .ADD,
                .children = &[_]*const Self{ self, other },
                .requires_grad = self.requires_grad or other.requires_grad,
                .allocator = self.allocator,
            });
        }

        /// Element-wise subtraction. COM.
        pub fn sub(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            return try createDependent(.{
                .data = try self.data.sub(other.data, allocator),
                .op = .SUB,
                .children = &[_]*const Self{ self, other },
                .requires_grad = self.requires_grad or other.requires_grad,
                .allocator = allocator,
            });
        }

        /// Element-wise multiplication. COM.
        pub fn mul(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            return try createDependent(.{
                .data = try self.data.mul(other.data, allocator),
                .op = .MUL,
                .children = &[_]*const Self{ self, other },
                .requires_grad = self.requires_grad or other.requires_grad,
                .allocator = allocator,
            });
        }

        /// Element-wise division. COM.
        pub fn div(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            return try createDependent(.{
                .data = try self.data.div(other.data, allocator),
                .op = .DIV,
                .children = &[_]*const Self{ self, other },
                .requires_grad = self.requires_grad or other.requires_grad,
                .allocator = allocator,
            });
        }

        /// Computes the maximum value of the tensor. Returns a scalar tensor. COM.
        pub fn max(self: *const Self, allocator: std.mem.Allocator) !*Self {
            const max_val = try self.data.max(allocator);
            return try createDependent(.{
                .data = max_val,
                .op = .MAX,
                .children = &[_]*const Self{self},
                .requires_grad = self.requires_grad,
                .allocator = allocator,
            });
        }

        /// Element-wise exponential. COM.
        pub fn exp(self: *const Self, allocator: std.mem.Allocator) !*Self {
            return try createDependent(.{
                .data = try self.data.exp(allocator),
                .op = .EXP,
                .children = &[_]*const Self{self},
                .requires_grad = self.requires_grad,
                .allocator = allocator,
            });
        }

        pub const MmOptions = struct { trans_a: bool = false, trans_b: bool = false };

        /// Matrix multiplication. COM.
        pub fn matmul(self: *const Self, other: *const Self, allocator: std.mem.Allocator, opts: MmOptions) !*Self {
            const result = try createDependent(.{
                .data = try self.data.matmul(other.data, opts.trans_a, opts.trans_b, allocator),
                .children = &[_]*const Self{ self, other },
                .requires_grad = self.requires_grad or other.requires_grad,
                .allocator = allocator,
            });
            result.op = if (!opts.trans_a and !opts.trans_b) .MATMUL_AB else if (opts.trans_a and !opts.trans_b) .MATMUL_AtB else if (!opts.trans_a and opts.trans_b) .MATMUL_ABt else @panic("No AtBt.");
            return result;
        }

        /// Dot product of two tensors. COM.
        pub fn dot(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            return try createDependent(.{
                .data = try self.data.dot(other.data, allocator),
                .op = .DOT,
                .children = &[_]*const Self{ self, other },
                .requires_grad = self.requires_grad or other.requires_grad,
                .allocator = allocator,
            });
        }

        /// Matrix-vector multiplication. COM.
        /// ---
        /// # ADR
        ///   - This does not use `createDependent()` as it is a special case where the grad shape different from data.
        ///   - Moreover, we cannot just reshape to the correct shape.
        ///   - Until I see more instances where this is required it will be written manually
        ///   - Edit: Think I got mixed up, this can prob be undone, but working now.
        pub fn matvec(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !*Self {
            const out = try allocator.create(Self);
            out.* = Self{
                .data = try self.data.matvec(other.data, false, allocator),
                .op = .MATVEC,
                .children = try allocator.dupe(*const Self, &[_]*const Self{ self, other }),
                .grad = if (self.requires_grad or other.requires_grad) try dtype.zeros(other.data.shape.shape, allocator) else null,
                .requires_grad = self.requires_grad or other.requires_grad,
                .allocator = allocator,
            };
            return out;
        }

        /// Sum of all elements in the tensor. COM.
        pub fn sum(self: *const Self, allocator: std.mem.Allocator) !*Self {
            return try createDependent(.{
                .data = try self.data.sum(allocator),
                .op = .SUM,
                .children = &[_]*const Self{self},
                .requires_grad = self.requires_grad,
                .allocator = allocator,
            });
        }

        /// Callback is highly dynamic so passing a reference may be a better idea for _backward callback,
        /// but committing to compiler reliance in this refactor
        pub fn setBackward(self: *Self, backward_fn: *const fn (Self, std.mem.Allocator) anyerror!void, ctx: ?*anyopaque) void {
            self._backward = backward_fn;
            self._backward_ctx = ctx;
        }

        pub fn backward(self: Self, allocator: std.mem.Allocator) !void {
            if (!self.requires_grad) return;
            if (self._backward) |f| {
                try f(self, allocator);
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

                            // (dL/dy) * (dy/da) and (dL/dy) * (dy/db)
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
                    .TRANSPOSE => {
                        if (self.children) |children| {
                            const gt = try self.grad.?.transpose(allocator);
                            _ = try children[0].grad.?._add(gt);
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
                        Op.MATMUL_AB, Op.MATMUL_AtB, Op.MATMUL_ABt => ": AB",
                        Op.MATVEC => ": Ax",
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

test "tensor/GraphManager/sum" {
    std.debug.print("{s} sum {s}\n", .{ "-" ** 5, "-" ** 5 });
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const T = f32;
    const Tensor = NDTensor(T);

    var input = try Tensor.init(&[_]T{ 1, 2, 3, 4 }, &[_]usize{4}, true, alloc);
    _ = input.setLabel("input");
    var sum_result = try input.sum(alloc);
    _ = sum_result.setLabel("sum_result");

    try std.testing.expectEqualSlices(T, &[_]T{10}, sum_result.data.data);

    // Backward pass
    var gm = GraphManager(Tensor).init(alloc, .{});
    defer gm.deinit();
    if (!settings.grad_enabled) return error.GradNotEnabled;
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
    var t1 = try Tensor.init(&[_]f32{ 1, 2, 3, 4, 5, 6 }, shape, false, alloc);

    // 1 2 3
    // 4 5 23
    try t1.set(&[_]usize{ 1, 2 }, -5);
    t1.print();
    // std.debug.print("{d}\n", .{t1.get(&[_]usize{ 1, 2 })});
    // std.debug.print("{d}\n", .{t1.indexToPos(5, alloc)});

    const t2 = try Tensor.init(&[_]f32{ 10, 20, 30, 40, 50, 60 }, shape, false, alloc);
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

    var t1 = try Tensor.init(&[_]T{2}, shape, true, alloc);
    var t2 = try Tensor.init(&[_]T{3}, shape, true, alloc);
    // t3 = t1 + t2;
    // dt3/dt1 = 1, dt3/dt2 = 1
    var t3 = try t1.add(t2, alloc);
    t1.acquire();
    t2.acquire();
    t3.acquire();

    var gm = GraphManager(Tensor).init(alloc, .{});
    defer gm.deinit();
    t3.grad = try Tensor.dtype.init(&[_]T{1}, shape, alloc);
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

    var t1 = try Tensor.init(&[_]T{2}, shape, true, alloc);
    var t2 = try Tensor.init(&[_]T{3}, shape, true, alloc);
    // t3 = t1 * t2;
    // dt3/dt1 = t2, dt3/dt2 = t1
    var t3 = try t1.mul(t2, alloc);
    t1.acquire();
    t2.acquire();
    t3.acquire();

    var gm = GraphManager(Tensor).init(alloc, .{});
    defer gm.deinit();
    t3.grad = try Tensor.dtype.init(&[_]T{1}, shape, alloc);
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

    var w = try Tensor.init(&[_]f32{ 3, 2 }, shape, true, alloc);
    var b = try Tensor.init(&[_]f32{ 1, 1 }, shape, true, alloc);
    var x = try Tensor.init(&[_]f32{ 4, 4 }, shape, true, alloc);

    // h = w*x + b
    // dh/dw = x, dh/db = 1
    var temp = try w.mul(x, alloc);
    var h = try temp.add(b, alloc);

    var backprop_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const backprop_alloc = backprop_arena.allocator();
    defer backprop_arena.deinit();

    var gm = GraphManager(Tensor).init(alloc, .{});
    defer gm.deinit();
    h.grad = try Tensor.dtype.init(&[_]T{ 1, 1 }, shape, alloc);
    try gm.backward(h, backprop_alloc);
    if (!backprop_arena.reset(.retain_capacity)) @panic("reset failed.\n");
    try std.testing.expectEqualSlices(T, x.data.data, w.grad.?.data);
    try std.testing.expectEqualSlices(T, &[_]T{ 1.0, 1.0 }, b.grad.?.data);

    // 2 x 1
    const shape2 = &[_]usize{ 2, 1 };
    w.grad.?.fill(0);
    b.grad.?.fill(0);
    x.grad.?.fill(0);
    try w._reshape(shape2);
    try b._reshape(shape2);
    try x._reshape(shape2);
    // h = w*x + b
    // dh/dw = x, dh/db = 1
    temp = try w.mul(x, alloc);
    h = try temp.add(b, alloc);

    var gm2 = GraphManager(Tensor).init(alloc, .{});
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

    var t1 = try Tensor.init(&[_]T{ 4, 9 }, shape, true, alloc);
    var t2 = try Tensor.init(&[_]T{ 2, 3 }, shape, true, alloc);
    var t3 = try t1.div(t2, alloc);
    t1.acquire();
    t2.acquire();
    t3.acquire();

    var gm = GraphManager(Tensor).init(alloc, .{});
    defer gm.deinit();
    t3.grad = try Tensor.dtype.init(&[_]T{ 1, 1 }, shape, alloc);
    try gm.backward(t3, alloc);

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

    var t1 = try Tensor.init(&[_]T{ 1, 2, 3, 4 }, shape, true, alloc);
    var t2 = try Tensor.init(&[_]T{ 1, 0, 0, 1 }, shape, true, alloc);
    var t3 = try t1.matmul(t2, alloc, .{});
    t1.acquire();
    t2.acquire();
    t3.acquire();

    var gm = GraphManager(Tensor).init(alloc, .{});
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
    var t1 = try Tensor.init(&[_]T{ 1, 2, 3, 4 }, shape_mat, true, alloc);
    var t2 = try Tensor.init(&[_]T{ 1, 1 }, shape_vec, true, alloc);
    var t3 = try t1.matvec(t2, alloc);
    t1.acquire();
    t2.acquire();
    t3.acquire();

    var gm = GraphManager(Tensor).init(alloc, .{});
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

    var t1 = try Tensor.init(&[_]T{ 1, 2, 3 }, shape, true, alloc);
    var t2 = try Tensor.init(&[_]T{ 4, 5, 6 }, shape, true, alloc);
    var t3 = try t1.dot(t2, alloc);
    t1.acquire();
    t2.acquire();
    t3.acquire();

    var gm = GraphManager(Tensor).init(alloc, .{});
    defer gm.deinit();
    t3.grad.?.fill(1.0);
    try gm.backward(t3, alloc);

    const expected_grad_t1 = &[_]T{ 4, 5, 6 };
    const expected_grad_t2 = &[_]T{ 1, 2, 3 };

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}
