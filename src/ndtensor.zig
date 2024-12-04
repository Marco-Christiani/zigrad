// TODO: implement view(), permute(), view is sort of an ever-changing WIP (these are ndarray tasks btw).
// TODO: migrate primitive ops to newer _backward/_backward_ctx feature.
const std = @import("std");
const zg = @import("zigrad.zig");
const settings = zg.settings;
const DeviceReference = zg.DeviceReference;
const backend = zg.backend;

const ndarray = @import("ndarray.zig");
const Range = ndarray.Range;
const Shape = ndarray.Shape;
const NDArray = ndarray.NDArray;
const GraphManager = @import("graph_manager.zig").GraphManager;

const log = std.log.scoped(.zg_tensor);

pub const MaxOverDimOptions = struct {
    dim: usize,
    keep_dims: bool = false,
};

pub const GatherOptions = struct {
    // no reason this needs to be a tensor, other than to keep a consistent user interface, shouldnt worry ab ndarray
    indices: *const NDTensor(usize),
    dim: usize,
};

/// These Op enums are not really functional but are convenient for the user (i.e. when traversing the graph or debugging)
/// There are many more ops supported than this. This may be deprecated by v1.
/// Zigrad does not switch on the op during backward and ops can be added without existing the existing enum values.
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
    MATMUL_AtBt,
    DOT,
    MATVEC,
    SUM,
    RESHAPE,
    TRANSPOSE,
    MAX,
    EXP,
    TRANSFER,
};

pub fn NDTensor(comptime T: type) type {
    return struct {

        // signature for a backwards callback
        const Callback = *const fn (Self) anyerror!void;

        pub const dtype = NDArray(T);
        const Self = @This();

        data: *dtype,
        op: ?Op = null,
        children: ?[]const *Self = null,
        label: ?[]const u8 = null,
        grad: ?*dtype = null,
        requires_grad: bool,
        acquired: bool = false, // not inherited
        device: DeviceReference,
        _backward: ?Callback = null, // see notes below
        _backward_ctx: ?*anyopaque = null,

        /// Values and shape are allocated. COM.
        pub fn init(values: []const T, shape: ?[]const usize, requires_grad: bool, device: DeviceReference) !*Self {
            const self_shape = shape orelse &[_]usize{values.len};
            const self = try device.allocator.create(Self);
            self.* = Self{
                .data = try dtype.init(values, self_shape, device),
                .grad = if (requires_grad) try dtype.zeros(self_shape, device) else null,
                .requires_grad = requires_grad,
                .device = device,
            };
            return self;
        }

        /// Shape is allocated. COM.
        /// As it stands, with grad disabled you can still allocate grads but grads wont be tracked (or allocated) in ops
        pub fn empty(shape: []const usize, requires_grad: bool, device: DeviceReference) !*Self {
            const self = try device.allocator.create(Self);
            self.* = Self{
                .data = try dtype.empty(shape, device),
                .grad = if (requires_grad) try dtype.zeros(shape, device) else null,
                .requires_grad = requires_grad,
                .device = device,
            };
            return self;
        }

        pub fn zeros(shape: []const usize, requires_grad: bool, device: DeviceReference) !*Self {
            const self = try device.allocator.create(Self);
            self.* = Self{
                .data = try dtype.zeros(shape, device),
                .grad = if (requires_grad) try dtype.zeros(shape, device) else null,
                .requires_grad = requires_grad,
                .device = device,
            };
            return self;
        }

        pub fn getShape(self: Self) []const usize {
            return self.data.shape.shape;
        }

        pub fn getSize(self: Self) []const usize {
            return self.data.data.len;
        }

        pub fn getStrides(self: Self) []const usize {
            return self.data.shape.strides;
        }

        pub fn getData(self: Self) []T {
            return self.data.data;
        }

        pub fn cast(self: *Self, K: type) !*NDTensor(K) {
            _ = self;
            @compileError("Not implemented");
        }

        pub fn _unsqueeze(self: *Self) !*Self {
            try self.data.shape._unsqueeze();
            if (self.grad) |g| try g.shape._unsqueeze();
            return self;
        }

        pub fn requiresGrad(self: Self) bool {
            return self.requires_grad and zg.rt_grad_enabled;
        }

        pub fn setupGrad(self: *Self, fill_value: ?T) !void {
            if (self.grad == null) {
                self.grad = try dtype.empty(self.data.shape.shape, self.device);
            }
            return self.grad.?.fill(fill_value orelse return, self.device);
        }

        pub const CreateDependentOpts = struct {
            data: *dtype,
            op: ?Op = null,
            children: []const *Self,
            label: ?[]const u8 = null,
            requires_grad: bool,
            device: DeviceReference,
            _backward: ?Callback = null,
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
        ///   4. Should this respect the global `grad_enabled` flag and override the flag parameter? Tbd.
        ///      UPDATE: Changed my mind, overrides.
        pub fn createDependent(opts: CreateDependentOpts) !*Self {
            const self = try opts.device.allocator.create(Self);
            const rg = opts.requires_grad and zg.rt_grad_enabled;
            self.* = Self{
                .data = opts.data,
                .op = opts.op,
                // TODO: How to handle not holding references if requiresGrad == false??
                // .children = if (rg) try opts.allocator.dupe(*const Self, opts.children) else null,
                .children = try opts.device.allocator.dupe(*Self, opts.children),
                .label = if (opts.label) |l| try opts.device.allocator.dupe(u8, l) else null,
                .grad = if (rg) try dtype.zeros(opts.data.shape.shape, opts.device) else null,
                .requires_grad = rg,
                .acquired = false,
                ._backward = if (rg) opts._backward else null,
                ._backward_ctx = if (rg) opts._backward_ctx else null,
                .device = opts.device,
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            if (self.acquired) std.debug.panic("Attempt to deinit an acquired tensor.", .{});
            // log.debug("deinit().data {?s}", .{self.label});
            self.data.deinit(self.device);
            if (self.grad) |g| {
                // log.debug("deinit().grad {?s}", .{self.label});
                g.deinit(self.device);
            }

            // TODO: verify this is heap first, possibly by checking alignment, not sure.
            if (self.label) |l| self.device.allocator.free(l);

            // TODO: ohhhh this is tricky...
            // if (self._backward_ctx) |ctx| {
            //     if (@hasDecl(ctx, "deinit")) ctx.deinit();
            // }
            if (self.children) |c| self.device.allocator.free(c);
            self.device.allocator.destroy(self);
        }

        pub fn teardown(self: *Self) void {
            log.debug("START teardown {?s}", .{self.label});
            if (self.acquired) std.debug.panic("Attempt to deinit an acquired tensor.", .{});
            if (self.children) |children| {
                for (children) |c| {
                    log.debug("{?s} accessing child {?s}", .{ self.label, c.label });
                    if (!c.acquired) c.teardown() else log.warn("skipping acquired tensor in teardown label={?s}", .{c.label});
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
        pub fn fromZarray(values: *dtype, requires_grad: bool, grad_shape: ?Shape, device: DeviceReference) !*Self {
            log.warn("Consider createDependent() if you are about to set children. This may be deprecated.", .{});
            const result = try device.allocator.create(Self);
            const grad: ?*dtype = blk: {
                if (requires_grad) {
                    if (grad_shape) |s| {
                        log.warn("`grad_shape` may be deprecated.", .{});
                        break :blk try dtype.zeros(s.shape, device);
                    } else {
                        break :blk (try dtype.zeros(values.shape.shape, device));
                    }
                } else {
                    break :blk null;
                }
            };
            result.* = Self{
                .data = values,
                .grad = grad,
                .requires_grad = requires_grad,
                .device = device,
            };
            return result;
        }

        fn toDeviceImpl(
            src: []const T,
            dst: []T,
            src_device: DeviceReference,
            dst_device: DeviceReference,
        ) !void {
            if (comptime @typeInfo(@TypeOf(src_device)) != .Struct)
                return;

            // currently only implemented for a single aux device.
            std.debug.assert(!src_device.isCompatible(dst_device));

            if (src_device.isHost()) {
                dst_device.memTransfer(T, src, dst, .HtoD);
            } else {
                src_device.memTransfer(T, src, dst, .DtoH);
            }
        }

        pub fn _toDevice(self: *Self, device: DeviceReference) !void {
            if (self.device.isCompatible(device))
                return self;

            const data = try device.memAlloc(T, self.data.data.len);
            toDeviceImpl(self.data.data, data, self.device, device);
            self.device.memFree(self.data.data);
            self.data.data = data;
            self.device = device;
        }

        pub fn toDevice(self: *Self, device: DeviceReference) !*Self {
            if (self.device.isCompatible(device))
                return self;

            const to_device_bw = struct {
                fn to_device_bw_impl(_self: Self) !void {
                    const child = (_self.children orelse return error.NoChildren)[0];
                    toDeviceImpl(_self.grad.?.data, child.grad.?.data, self.device, child.device);
                }
            }.to_device_bw_impl;

            const data = try dtype.empty(self.data.shape, device);

            toDeviceImpl(self.data.data, data.data, self.device, device);

            var result = try createDependent(.{
                .data = data,
                .op = .TRANSFER,
                .children = &.{self},
                .label = null,
                .requires_grad = false,
                .device = self.device,
                ._backward = to_device_bw,
            });

            errdefer result.deinit();
            if (self.requiresGrad()) { // need to make this call, not check the attribute flag
                result.grad = try dtype.zeros(self.data.shape, device);
                result.requires_grad = true;
            }
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
        pub fn clone(self: Self) !*Self {
            const result = try self.device.allocator.create(Self);
            errdefer self.device.allocator.destroy(result);
            result.* = Self{
                .data = try self.data.copy(self.device),
                .grad = if (self.grad) |g| try g.copy(self.device) else null,
                .requires_grad = self.requiresGrad(),
                .op = null,
                .children = null,
                .label = null,
                .acquired = false,
                .device = self.device,
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
        pub fn reshape(self: *Self, new_shape: []const usize) !*Self {
            const reshape_bw = struct {
                fn reshape_bw_impl(_self: Self) !void {
                    const children = _self.children orelse return error.NoChildren;
                    const original_shape = children[0].data.shape;
                    try self.grad.?._reshape(original_shape.shape);
                    _ = try children[0].grad.?._add(self.grad.?);
                }
            }.reshape_bw_impl;

            const result = try createDependent(.{
                .data = try self.data.copy(self.device),
                .op = .RESHAPE,
                .children = &.{self},
                .label = null,
                .requires_grad = self.requires_grad,
                .device = self.device,
                ._backward = reshape_bw,
            });
            errdefer result.deinit();
            try result.data._reshape(new_shape);
            if (result.grad) |g| try g._reshape(new_shape);
            return result;
        }

        /// Copies. COM.
        pub fn transpose(self: *Self) !*Self {
            const transpose_bw = struct {
                fn transpose_bw_impl(_self: Self) !void {
                    const children = _self.children orelse return error.NoChildren;
                    const gt = try self.grad.?.transpose(_self.device);
                    defer gt.deinit(_self.device);
                    _ = try children[0].grad.?._add(gt);
                }
            }.transpose_bw_impl;
            var result = try createDependent(.{
                .data = try self.data.transpose(self.device),
                .op = .TRANSPOSE,
                .children = &.{self},
                .label = null,
                .requires_grad = false, // we will set grad ourselves for efficiency
                .device = self.device,
                ._backward = transpose_bw,
            });
            errdefer result.deinit();
            if (self.requiresGrad()) { // need to make this call, not check the attribute flag
                result.grad = try self.grad.?.transpose(self.device);
                result.requires_grad = true;
            }
            return result;
        }

        pub fn setLabel(self: *Self, comptime label: []const u8) *Self {
            if (self.label) |l| self.device.allocator.free(l);
            self.label = self.device.allocator.dupe(u8, label) catch @panic("OOM");
            return self;
        }

        pub fn fill(self: Self, val: T) void {
            self.data.fill(val, self.device);
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
        fn indexToPos(self: Self, index: usize) []const usize {
            return self.data.offsetToPos(index, self.device.allocator);
        }

        /// [WIP] Values and grads are views into self. Shapes are allocated and COM.
        ///
        /// ---
        /// # ADR
        ///   - It is assumed the caller wants a temporary mutable view, return by copy.
        ///   - This creates a complex situation, there is no backward for this operation and it returns a mutable view.
        ///   - Caller must set backward
        pub fn sliceRanges(self: Self, ranges: []const Range) !Self {
            log.err("WIP", .{});
            const sliced_data = try self.data.sliceRanges(ranges);
            return Self{
                .data = sliced_data,
                .grad = if (self.requiresGrad()) try self.grad.?.sliceRanges(ranges) else null,
                .requires_grad = self.requiresGrad(),
                .op = null,
                .children = null,
                .label = null,
                .acquired = false,
                .device = self.device,
                ._backward = null,
                ._backward_ctx = null,
            };
        }

        pub fn setSlice(self: Self, ranges: []const Range, values: Self) !void {
            if (self.requiresGrad()) {
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
            self.grad.?.clip_norm(opts.max_norm, opts.delta, self.device);
        }

        pub fn setChildren(self: *Self, children: []const *Self) !void {
            log.warn("Deprecation warning: use createDependent.", .{});
            self.children = try self.device.allocator.dupe(*Self, children);
        }

        /// Element-wise addition. COM.
        pub fn add(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.isCompatible(other.device));

            const addBw = struct {
                fn addBwImpl(_self: Self) !void {
                    const children = _self.children orelse return error.NoChildren;
                    const a = children[0];
                    const b = children[1];
                    const a_grad = try _self.grad.?.unbroadcast(a.grad.?.shape, _self.device);
                    const b_grad = try _self.grad.?.unbroadcast(b.grad.?.shape, _self.device);
                    defer a_grad.deinit(_self.device);
                    defer b_grad.deinit(_self.device);
                    _ = try a.grad.?._add(a_grad);
                    _ = try b.grad.?._add(b_grad);
                }
            }.addBwImpl;
            return try createDependent(.{
                .data = try self.data.add(other.data),
                .op = .ADD,
                .children = &.{ self, other },
                .requires_grad = self.requires_grad or other.requires_grad,
                .device = self.device,
                ._backward = addBw,
            });
        }

        /// Element-wise subtraction. COM.
        pub fn sub(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.isCompatible(other.device));

            const subBw = struct {
                fn subBwImpl(_self: Self) !void {
                    const children = _self.children orelse return error.NoChildren;
                    const a = children[0];
                    const b = children[1];

                    const a_grad = try (try _self.grad.?.copy(_self.device)).unbroadcast(a.grad.?.shape, _self.device);
                    const b_grad = try (try _self.grad.?.copy(_self.device)).unbroadcast(b.grad.?.shape, _self.device);
                    defer a_grad.deinit(_self.device);
                    defer b_grad.deinit(_self.device);

                    _ = try a.grad.?._add(a_grad);
                    _ = try b.grad.?._sub(b_grad);
                }
            }.subBwImpl;
            return try createDependent(.{
                .data = try self.data.sub(other.data, self.device),
                .op = .SUB,
                .children = &.{ self, other },
                .requires_grad = self.requires_grad or other.requires_grad,
                .device = self.device,
                ._backward = subBw,
            });
        }

        /// Element-wise multiplication. COM.
        pub fn mul(self: *Self, other: *Self, device: DeviceReference) !*Self {
            std.debug.assert(self.device.isCompatible(other.device));

            const mulBw = struct {
                fn mulBwImpl(_self: Self) !void {
                    const children = _self.children orelse return error.NoChildren;
                    const a = children[0];
                    const b = children[1];

                    // TODO: can remove a copy here

                    // (dL/dy) * (dy/da), (dL/dy) * (dy/db)
                    const a_grad_value = try b.data.mul(_self.grad.?, _self.device);
                    const b_grad_value = try a.data.mul(_self.grad.?, _self.device);

                    defer a_grad_value.deinit(_self.device);
                    defer b_grad_value.deinit(_self.device);

                    const a_grad = try a_grad_value.unbroadcast(a.grad.?.shape, _self.device);
                    const b_grad = try b_grad_value.unbroadcast(b.grad.?.shape, _self.device);

                    defer a_grad.deinit(_self.device);
                    defer b_grad.deinit(_self.device);

                    _ = try a.grad.?._add(a_grad);
                    _ = try b.grad.?._add(b_grad);
                }
            }.mulBwImpl;
            return try createDependent(.{
                .data = try self.data.mul(other.data, device),
                .op = .MUL,
                .children = &.{ self, other },
                .requires_grad = self.requires_grad or other.requires_grad,
                .device = device,
                ._backward = mulBw,
            });
        }

        /// Element-wise division. COM.
        pub fn div(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.isCompatible(other.device));

            const divBw = struct {
                fn divBwImpl(_self: Self) !void {
                    const children = _self.children orelse return error.NoChildren;
                    const a = children[0];
                    const b = children[1];

                    // TODO: can remove at least one copy here right?

                    // (dL/dy) * (dy/da) and (dL/dy) * (dy/db)
                    const a_grad_value = try _self.grad.?.div(b.data, _self.device);
                    const b_grad_value = try _self.grad.?.mul(a.data, _self.device);

                    defer a_grad_value.deinit(_self.device);
                    defer b_grad_value.deinit(_self.device);

                    const bsq = try b.data.mul(b.data, _self.device);
                    const neg_b_grad_value = try b_grad_value.div(bsq, _self.device);

                    defer bsq.deinit(_self.device);
                    defer neg_b_grad_value.deinit(_self.device);

                    const a_grad = try a_grad_value.unbroadcast(a.grad.?.shape, _self.device);
                    defer a_grad.deinit(_self.device);
                    const b_grad = try neg_b_grad_value.unbroadcast(b.grad.?.shape, _self.device);
                    defer b_grad.deinit(_self.device);

                    _ = try a.grad.?._add(a_grad);
                    _ = try b.grad.?._sub(b_grad);
                }
            }.divBwImpl;
            return try createDependent(.{
                .data = try self.data.div(other.data, self.device),
                .op = .DIV,
                .children = &.{ self, other },
                .requires_grad = self.requires_grad or other.requires_grad,
                .device = self.device,
                ._backward = divBw,
            });
        }

        /// Computes the maximum value of the tensor. Returns a scalar tensor. COM.
        pub fn max(self: *Self) !*Self {
            const max_mem = try dtype.empty(&.{1}, self.device);
            const max_idx = try self.device.memCreate(i32);

            self.device.blas.maxForward(T, self.getData(), max_mem.data, max_idx);

            const maxBw = struct {
                fn maxBwImpl(_self: Self) !void {
                    const children = _self.children orelse return error.NoChildren;
                    const child = children[0];
                    const iptr: *i32 = @ptrCast(@alignCast(_self._backward_ctx.?));
                    self.device.blas.maxReverse(T, _self.grad.?.data.data, child.grad.?.data.data, iptr);
                    self.device.memDestroy(iptr);
                }
            }.maxBwImpl;

            return try createDependent(.{
                .data = max_mem,
                .op = .MAX,
                .children = &.{self},
                .requires_grad = self.requires_grad,
                .device = self.device,
                ._backward = maxBw,
            });
        }

        /// Element-wise exponential. COM.
        pub fn exp(self: *Self) !*Self {
            const expBw = struct {
                fn expBwImpl(_self: Self) !void {
                    const children = _self.children orelse return error.NoChildren;
                    const child = children[0];
                    for (_self.data.data, _self.grad.?.data, 0..) |exp_val, grad_val, i| {
                        child.grad.?.data[i] += exp_val * grad_val;
                    }
                }
            }.expBwImpl;
            return try createDependent(.{
                .data = try self.data.exp(self.device),
                .op = .EXP,
                .children = &[_]*const Self{self},
                .requires_grad = self.requires_grad,
                .device = self.device,
                ._backward = expBw,
            });
        }

        pub const MmOptions = struct { trans_a: bool = false, trans_b: bool = false };

        /// TODO: this should proxy to bmmAcc
        /// Matrix multiplication. COM.
        pub fn bmm(self: *Self, other: *Self, device: DeviceReference, opts: MmOptions) !*Self {
            std.debug.assert(self.device.isCompatible(other.device));

            const ctx = try device.allocator.create(MmAccOptions);
            ctx.* = MmAccOptions{ .trans_a = opts.trans_a, .trans_b = opts.trans_b, .alpha = 1, .beta = 0 };
            const result = try createDependent(.{
                .data = try self.data.bmm(other.data, opts.trans_a, opts.trans_b, device),
                .children = &.{ self, other },
                .requires_grad = self.requires_grad or other.requires_grad,
                .device = device,
                ._backward = bw_bmmAcc,
                ._backward_ctx = ctx,
            });
            result.op = if (!opts.trans_a and !opts.trans_b) .MATMUL_AB else if (opts.trans_a and !opts.trans_b) .MATMUL_AtB else if (!opts.trans_a and opts.trans_b) .MATMUL_ABt else .MATMUL_AtBt;
            return result;
        }

        pub const MmAccOptions = struct { trans_a: bool = false, trans_b: bool = false, alpha: T = 1.0, beta: T = 1.0 };

        /// Matrix multiplication w accumulation. COM, output is written to directly.
        pub fn bmmAcc(
            self: *const NDTensor(T),
            other: *const NDTensor(T),
            output: *NDTensor(T),
            opts: MmAccOptions,
        ) !void {
            std.debug.assert(self.device.isCompatible(other.device));

            _ = try self.data._bmmAcc(
                other.data,
                output.data,
                opts.alpha,
                opts.beta,
                opts.trans_a,
                opts.trans_b,
            );

            const requires_grad = zg.rt_grad_enabled and (self.requires_grad or other.requires_grad or output.requires_grad);
            if (requires_grad) {
                const ctx = try self.device.allocator.create(MmAccOptions);
                ctx.* = opts;
                output.children = try self.device.allocator.dupe(*NDTensor(T), &.{ self, other });
                output._backward = bw_bmmAcc;
                output._backward_ctx = ctx;
                output.requires_grad = true;
            }
        }

        fn bw_bmmAcc(self: NDTensor(T)) !void {
            const grad_C = self.grad orelse return error.NoGradient;

            const opts: *MmAccOptions = @ptrCast(@alignCast(self._backward_ctx orelse return error.NoBackwardContext));
            defer self.device.allocator.destroy(opts);

            const children = self.children orelse return error.NoChildren;
            const A = children[0].data;
            const grad_A = children[0].grad;
            const B = children[1].data;
            const grad_B = children[1].grad;

            if (self.op) |op| {
                switch (op) {
                    .MATMUL_AB => {
                        // A Shape: (..., m, k)
                        // B Shape: (..., k, n)
                        // grad_C Shape: (..., m, n)
                        // grad_A += grad_C * B'
                        _ = try grad_C._bmmAcc(B, grad_A, 1.0, 1.0, false, true);
                        // grad_B += A' * grad_C
                        _ = try A._bmmAcc(grad_C, grad_B, 1.0, 1.0, true, false);
                    },
                    .MATMUL_AtB => {
                        // A Shape: (..., k, m)
                        // B Shape: (..., k, n)
                        // grad_C Shape: (..., m, n)
                        // grad_A += B * grad_C'
                        _ = try B._bmmAcc(self.grad.?, grad_A, 1.0, 1.0, false, true);
                        // grad_B += A * grad_C
                        _ = try A._bmmAcc(self.grad.?, grad_B, 1.0, 1.0, false, false);
                    },
                    .MATMUL_ABt => {
                        // A Shape: (..., m, k)
                        // B Shape: (..., n, k)
                        // grad_C Shape: (..., m, n)
                        // grad_A += grad_C * B
                        _ = try grad_C._bmmAcc(B, grad_A, 1.0, 1.0, false, false);
                        // grad_B += grad_C' * A
                        _ = try grad_C._bmmAcc(A, grad_B, 1.0, 1.0, true, false);
                    },
                    .MATMUL_AtBt => {
                        // A Shape: (..., k, m)
                        // B Shape: (..., n, k)
                        // grad_C Shape: (..., m, n)
                        // grad_A += B' * grad_C'
                        _ = try B._bmmAcc(grad_C, grad_A, 1.0, 1.0, true, true);
                        // grad_B += grad_C * A'
                        _ = try grad_C._bmmAcc(A, grad_B, 1.0, 1.0, false, true);
                    },
                    else => std.debug.panic("Op {s} is not yet implemented.", .{@tagName(op)}),
                }
            }
        }

        /// Dot product of two tensors. COM.
        pub fn dot(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.isCompatible(other.device));

            const dotBw = struct {
                fn dotBwImpl(_self: Self) !void {
                    const children = _self.children orelse return error.NoChildren;
                    var a = children[0];
                    var b = children[1];
                    const gscalar: *const T = @ptrCast(_self.grad.?.data.ptr);
                    _self.device.blas.axpy(T, a.getData(), b.grad.?.data, gscalar);
                    _self.device.blas.axpy(T, b.getData(), b.grad.?.data, gscalar);
                }
            }.dotBwImpl;

            return try createDependent(.{
                .data = try self.data.dot(other.data, self.device),
                .op = .DOT,
                .children = &.{ self, other },
                .requires_grad = self.requires_grad or other.requires_grad,
                .device = self.device,
                ._backward = dotBw,
            });
        }

        /// Matrix-vector multiplication. COM.
        /// ---
        /// # ADR
        ///   - This does not use `createDependent()` as it is a special case where the grad shape different from data.
        ///   - Moreover, we cannot just reshape to the correct shape.
        ///   - Until I see more instances where this is required it will be written manually
        ///   - Edit: Think I got mixed up, this can prob be undone, but working now.
        pub fn matvec(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.isCompatible(other.device));

            const matvecBw = struct {
                /// TODO: Use accumulation
                fn matvecBwImpl(_self: Self) !void {
                    const children = _self.children orelse return error.NoChildren;
                    var A = children[0].data;
                    const x = children[1].data;

                    //  L(y), y = Ax, dL/dA = (dL/dy)(dy/dA) = (dL/dy)x'
                    const grad_A = try _self.grad.?.outer(x, _self.device);
                    defer grad_A.deinit(_self.device);
                    _ = try children[0].grad.?._add(grad_A);

                    //  L(y), y = Ax, dL/dx = (dL/dy)(dy/dx) = A'(dL/dy)
                    const grad_x = try A.matvec(_self.grad.?, true, _self.device);
                    defer grad_x.deinit(_self.device);
                    _ = try children[1].grad.?._add(grad_x);
                }
            }.matvecBwImpl;

            const out = try self.device.allocator.create(Self);
            out.* = Self{
                .data = try self.data.matvec(other.data, false),
                .op = .MATVEC,
                .children = try self.device.allocator.dupe(*Self, &.{ self, other }),
                .grad = if (self.requiresGrad() or other.requires_grad) try dtype.zeros(other.data.shape.shape, self.device) else null,
                .requires_grad = self.requiresGrad() or other.requires_grad,
                .device = self.device,
                ._backward = matvecBw,
            };
            return out;
        }

        /// Sum of all elements in the tensor. COM.
        pub fn sum(self: *Self) !*Self {
            const sumBw = struct {
                fn sumBwImpl(_self: NDTensor(T)) !void {
                    const children = _self.children orelse return error.NoChildren;
                    const child = children[0];
                    _ = try child.grad.?._add(_self.grad.?);
                }
            }.sumBwImpl;
            return try createDependent(.{
                .data = try self.data.sum(self.device),
                .op = .SUM,
                .children = &.{self},
                .requires_grad = self.requires_grad,
                .device = self.device,
                ._backward = sumBw,
            });
        }

        pub fn maxOverDim(self: *Self, device: DeviceReference, opts: MaxOverDimOptions) !*Self {
            const maxBackward = struct {
                // NOTE: See gather() comments, same apply here
                fn bwImpl(_self: Self) !void {
                    const bw_children = _self.children orelse return error.NoChildren;
                    const bw_input = bw_children[0];
                    if (bw_input.grad == null) return;
                    const raw_offsets: [*:0]usize = @ptrCast(@alignCast(_self._backward_ctx orelse return error.NoBackwardContext));
                    const offsets: []usize = std.mem.span(raw_offsets);
                    for (0.._self.data.data.len) |i| bw_input.grad.?.data[offsets[i]] += _self.grad.?.data[i];
                    _self.device.memFree(offsets);
                }
            }.bwImpl;

            const max_result = try self.data.maxOverDim(device, .{ .dim = opts.dim, .keep_dims = opts.keep_dims, .return_offsets = true });
            // TODO: use a null terminated allocation instead, tired rn
            const ctx = if (self.requiresGrad()) try device.allocator.dupeZ(usize, max_result.offsets.?) else null;
            if (max_result.offsets) |offs| device.allocator.free(offs);

            return try createDependent(.{
                .data = max_result.values,
                .op = null,
                .children = &.{self},
                .requires_grad = self.requires_grad,
                .device = device,
                ._backward = maxBackward,
                ._backward_ctx = if (ctx) |c| c.ptr else null,
            });
        }

        pub fn gather(self: *Self, device: DeviceReference, opts: GatherOptions) !*Self {
            const gatherBackward = struct {
                fn bwImpl(bw_tensor: NDTensor(T)) !void {
                    const bw_children = bw_tensor.children orelse return error.NoChildren;
                    const bw_input = bw_children[0];
                    if (bw_input.grad == null) return;
                    const offsets: [*]usize = @ptrCast(@alignCast(bw_tensor._backward_ctx orelse return error.NoBackwardContext));
                    // how am i expected to free this, unknown len
                    // defer _self.device.raw(offsets); // FIXME: just occuring to me the problem with this, if _self.device != fwd_allocator

                    // bw_tensor must/should be the same len as indices used to index (note that offsets is a raw c ptr without a len)
                    // std.debug.assert(offsets.len == bw_tensor.data.data.len); // can make this a real check when its a  null term alloc
                    for (0..bw_tensor.data.data.len) |i| bw_input.grad.?.data[offsets[i]] += bw_tensor.grad.?.data[i];
                }
            }.bwImpl;

            const gather_result = try self.data.gather(device, .{ .indices = opts.indices.data, .dim = opts.dim, .return_offsets = true });
            // TODO: use a null terminated allocation instead, tired rn
            const ctx = if (self.requiresGrad()) try device.allocator.dupe(usize, gather_result.offsets.?) else null;

            return try createDependent(.{
                .data = gather_result.values,
                .op = null,
                .children = &.{self},
                .requires_grad = self.requires_grad,
                .device = device,
                ._backward = gatherBackward,
                ._backward_ctx = if (ctx) |c| c.ptr else null,
            });
        }

        /// Callback is highly dynamic so passing a reference may be a better idea for _backward callback,
        /// but committing to compiler reliance in this refactor
        pub fn setBackward(self: *Self, backward_fn: Callback, ctx: ?*anyopaque) void {
            self._backward = backward_fn;
            self._backward_ctx = ctx;
        }

        pub fn backward(self: Self) !void {
            // hypothetically, we could check for children. This is treating self as detached, tbd if this is a good idea.
            if (!zg.rt_grad_enabled) return error.GradNotEnabled;
            if (!self.requires_grad) return;
            if (self._backward) |f| {
                try f(self);
                return;
            }
            if (self.op) |op| {
                std.debug.panic("Op {s} backward not implemented.", .{@tagName(op)});
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
                        Op.MATMUL_AB, Op.MATMUL_AtB, Op.MATMUL_ABt, Op.MATMUL_AtBt => ": AB",
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
    const allocator = std.testing.allocator;
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);

    var input = try Tensor.init(&[_]T{ 1, 2, 3, 4 }, &[_]usize{4}, true, device);
    _ = input.setLabel("input");
    var sum_result = try input.sum(device);
    _ = sum_result.setLabel("sum_result");

    try std.testing.expectEqualSlices(T, &[_]T{10}, sum_result.data.data);

    defer {
        input.deinit();
        sum_result.deinit();
    }

    // Backward pass
    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    if (!zg.rt_grad_enabled) return error.GradNotEnabled;
    sum_result.grad.?.fill(1.0, device);
    try gm.backward(sum_result, device);

    const expected_grad = &[_]T{ 1, 1, 1, 1 };
    try std.testing.expectEqualSlices(T, expected_grad, input.grad.?.data);
}

test "tensor/NDTensor index, add, div" {
    const allocator = std.testing.allocator;
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const shape = &[_]usize{ 2, 3 };
    const Tensor = NDTensor(f32);

    // 1 2 3
    // 4 5 6
    var t1 = try Tensor.init(&[_]f32{ 1, 2, 3, 4, 5, 6 }, shape, false, device);

    // 1 2 3
    // 4 5 23
    try t1.set(&[_]usize{ 1, 2 }, 1.1);
    try std.testing.expectEqual(1.1, t1.get(&.{ 1, 2 }));

    const t2 = try Tensor.init(&[_]f32{ 10, 20, 30, 40, 50, 60 }, shape, false, device);
    const t3 = try t1.add(t2, device);
    const t4 = try t3.sub(t1, device);
    defer {
        t1.deinit();
        t2.deinit();
        t3.deinit();
        t4.deinit();
    }

    try std.testing.expectEqualSlices(f32, t2.data.data, t4.data.data);
}

test "tensor/GraphManager/addback" {
    const allocator = std.testing.allocator;
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const shape = &[_]usize{1};
    const T = f32;
    const Tensor = NDTensor(T);

    var t1 = try Tensor.init(&[_]T{2}, shape, true, device);
    var t2 = try Tensor.init(&[_]T{3}, shape, true, device);
    // t3 = t1 + t2;
    // dt3/dt1 = 1, dt3/dt2 = 1
    var t3 = try t1.add(t2, device);
    defer {
        t1.deinit();
        t2.deinit();
        t3.deinit();
    }

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    t3.grad.?.fill(1.0, device);
    try gm.backward(t3, device);
    try std.testing.expectEqualDeep(&[_]T{1.0}, t1.grad.?.data);
    try std.testing.expectEqualDeep(&[_]T{1.0}, t2.grad.?.data);
}

test "tensor/GraphManager/mulback" {
    const allocator = std.testing.allocator;
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const shape = &[_]usize{1};
    const T = f32;
    const Tensor = NDTensor(T);

    var t1 = try Tensor.init(&[_]T{2}, shape, true, device);
    const t2 = try Tensor.init(&[_]T{3}, shape, true, device);
    // t3 = t1 * t2;
    // dt3/dt1 = t2, dt3/dt2 = t1
    var t3 = try t1.mul(t2, device);
    defer {
        t1.deinit();
        t2.deinit();
        t3.deinit();
    }

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    t3.grad.?.fill(1.0, device);
    try gm.backward(t3, device);
    try std.testing.expectEqualDeep(t2.data.data, t1.grad.?.data);
    try std.testing.expectEqualDeep(t1.data.data, t2.grad.?.data);
}

test "tensor/GraphManager/moreback" {
    const allocator = std.testing.allocator;
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const shape = &[_]usize{2};
    const T = f32;
    const Tensor = NDTensor(T);

    var w = try Tensor.init(&[_]f32{ 3, 2 }, shape, true, device);
    defer w.deinit();
    var b = try Tensor.init(&[_]f32{ 1, 1 }, shape, true, device);
    defer b.deinit();
    var x = try Tensor.init(&[_]f32{ 4, 4 }, shape, true, device);
    defer x.deinit();

    // h = w*x + b
    // dh/dw = x, dh/db = 1
    const temp = try w.mul(x, device);
    defer temp.deinit();
    const h = try temp.add(b, device);
    defer h.deinit();

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    h.grad.?.fill(1.0, device);
    try gm.backward(h, device);
    try std.testing.expectEqualSlices(T, x.data.data, w.grad.?.data);
    try std.testing.expectEqualSlices(T, &[_]T{ 1.0, 1.0 }, b.grad.?.data);

    // 2 x 1
    const shape2 = &[_]usize{ 2, 1 };
    w.grad.?.fill(0, device);
    b.grad.?.fill(0, device);
    x.grad.?.fill(0, device);
    try w._reshape(shape2);
    try b._reshape(shape2);
    try x._reshape(shape2);
    // h = w*x + b
    // dh/dw = x, dh/db = 1
    const temp2 = try w.mul(x, device);
    defer temp2.deinit();
    const h2 = try temp2.add(b, device);
    defer h2.deinit();

    var gm2 = GraphManager(Tensor).init(device.allocator, .{});
    defer gm2.deinit();
    h2.grad.?.fill(1.0, device);
    try gm2.backward(h2, device);
    try std.testing.expectEqualSlices(T, x.data.data, w.grad.?.data);
    try std.testing.expect(std.mem.allEqual(T, b.grad.?.data, 1));
}

test "tensor/GraphManager/divback" {
    const allocator = std.testing.allocator;
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);
    const shape = &[_]usize{2};

    var t1 = try Tensor.init(&[_]T{ 4, 9 }, shape, true, device);
    var t2 = try Tensor.init(&[_]T{ 2, 3 }, shape, true, device);
    var t3 = try t1.div(t2, device);
    defer {
        t1.deinit();
        t2.deinit();
        t3.deinit();
    }

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    t3.grad.?.fill(1.0, device);
    try gm.backward(t3, device);

    const expected_grad_t1 = &[_]T{ 1.0 / 2.0, 1.0 / 3.0 }; // 1 / b
    const expected_grad_t2 = &[_]T{ -4.0 / 4.0, -9.0 / 9.0 }; // -a / b^2

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}

test "tensor/GraphManager/matmul_backward square" {
    const allocator = std.testing.allocator;
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);
    const shape = &[_]usize{ 2, 2 };

    var t1 = try Tensor.init(&[_]T{ 1, 2, 3, 4 }, shape, true, device);
    var t2 = try Tensor.init(&[_]T{ 1, 0, 0, 1 }, shape, true, device);

    // Case 1: No transpose
    var t3 = try t1.bmm(t2, device, .{ .trans_a = false, .trans_b = false });

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    defer {
        t1.deinit();
        t2.deinit();
        t3.deinit();
    }
    t3.grad.?.fill(1.0, device);

    try gm.backward(t3, device);
    const expected_grad_t1 = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2 = &[_]T{ 4, 4, 6, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
    t1.grad.?.fill(0, device);
    t2.grad.?.fill(0, device);

    // Case 2: Transpose A
    var t3_trans_a = try t1.bmm(t2, device, .{ .trans_a = true, .trans_b = false });
    defer t3_trans_a.deinit();
    t3_trans_a.grad.?.fill(1.0, device);

    try gm.backward(t3_trans_a, device);
    const expected_grad_t1_trans_a = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_a = &[_]T{ 3, 3, 7, 7 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_a, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_a, t2.grad.?.data);
    t1.grad.?.fill(0, device);
    t2.grad.?.fill(0, device);

    // Case 3: Transpose B
    var t3_trans_b = try t1.bmm(t2, device, .{ .trans_a = false, .trans_b = true });
    defer t3_trans_b.deinit();
    t3_trans_b.grad.?.fill(1.0, device);
    try gm.backward(t3_trans_b, device);

    const expected_grad_t1_trans_b = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_b = &[_]T{ 4, 6, 4, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_b, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_b, t2.grad.?.data);
    t1.grad.?.fill(0, device);
    t2.grad.?.fill(0, device);

    // Case 4: Transpose both A and B
    var t3_trans_ab = try t1.bmm(t2, device, .{ .trans_a = true, .trans_b = true });
    defer t3_trans_ab.deinit();
    t3_trans_ab.grad.?.fill(1.0, device);
    try gm.backward(t3_trans_ab);

    const expected_grad_t1_trans_ab = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_ab = &[_]T{ 3, 7, 3, 7 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_ab, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_ab, t2.grad.?.data);
}

test "tensor/GraphManager/matmul_backward non-square" {
    const allocator = std.testing.allocator;
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);

    // Case 1: No transpose (t1: [2, 2, 3], t2: [2, 3, 2])
    var t1 = try Tensor.init(&[_]T{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &[_]usize{ 2, 2, 3 }, true, device);
    var t2 = try Tensor.init(&[_]T{ 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1 }, &[_]usize{ 2, 3, 2 }, true, device);

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();

    // Case 1: No transpose
    {
        var t3 = try t1.bmm(t2, device, .{ .trans_a = false, .trans_b = false });
        defer t3.deinit();
        t1.acquire();
        t2.acquire();
        t3.grad.?.fill(1.0, device);
        try gm.backward(t3);

        const expected_grad_t1 = &[_]T{ 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2 };
        const expected_grad_t2 = &[_]T{ 5, 5, 7, 7, 9, 9, 17, 17, 19, 19, 21, 21 };
        try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
        try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);

        t1.grad.?.fill(0, device);
        t2.grad.?.fill(0, device);
    }

    // Case 2: Transpose A (t1: [2, 3, 2], t2: [2, 3, 2])
    {
        var t1_case2 = try Tensor.init(&[_]T{ 1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12 }, &[_]usize{ 2, 3, 2 }, true, device);
        var t3 = try t1_case2.bmm(t2, device, .{ .trans_a = true, .trans_b = false });
        defer t3.deinit();
        t1_case2.acquire();
        t3.grad.?.fill(1.0, device);
        try gm.backward(t3, device);

        const expected_grad_t1 = &[_]T{ 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2 };
        const expected_grad_t2 = &[_]T{ 5, 5, 7, 7, 9, 9, 17, 17, 19, 19, 21, 21 };
        try std.testing.expectEqualSlices(T, expected_grad_t1, t1_case2.grad.?.data);
        try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);

        t1_case2.release();
        t1_case2.deinit();
        t2.grad.?.fill(0, device);
    }

    // Case 3: Transpose B (t1: [2, 2, 3], t2: [2, 2, 3])
    {
        var t2_case3 = try Tensor.init(&[_]T{ 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1 }, &[_]usize{ 2, 2, 3 }, true, device);
        var t3 = try t1.bmm(t2_case3, device, .{ .trans_a = false, .trans_b = true });
        defer t3.deinit();
        t2_case3.acquire();
        t3.grad.?.fill(1.0, device);
        try gm.backward(t3, device);

        const expected_grad_t1 = &[_]T{ 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2 };
        const expected_grad_t2 = &[_]T{ 5, 7, 9, 5, 7, 9, 17, 19, 21, 17, 19, 21 };
        try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
        try std.testing.expectEqualSlices(T, expected_grad_t2, t2_case3.grad.?.data);

        t1.grad.?.fill(0, device);
        t2_case3.release();
        t2_case3.deinit();
    }

    // Case 4: Transpose both A and B (t1: [2, 3, 2], t2: [2, 2, 3])
    {
        var t1_case4 = try Tensor.init(&[_]T{ 1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12 }, &[_]usize{ 2, 3, 2 }, true, device);
        var t2_case4 = try Tensor.init(&[_]T{ 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1 }, &[_]usize{ 2, 2, 3 }, true, device);
        var t3 = try t1_case4.bmm(t2_case4, device, .{ .trans_a = true, .trans_b = true });
        defer t3.deinit();
        t1_case4.acquire();
        t2_case4.acquire();
        t3.grad.?.fill(1.0, device);
        try gm.backward(t3, device);

        const expected_grad_t1 = &[_]T{ 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2 };
        const expected_grad_t2 = &[_]T{ 5, 7, 9, 5, 7, 9, 17, 19, 21, 17, 19, 21 };
        try std.testing.expectEqualSlices(T, expected_grad_t1, t1_case4.grad.?.data);
        try std.testing.expectEqualSlices(T, expected_grad_t2, t2_case4.grad.?.data);

        t1_case4.release();
        t1_case4.deinit();
        t2_case4.release();
        t2_case4.deinit();
    }

    t1.release();
    t1.deinit();
    t2.release();
    t2.deinit();
}

test "tensor/GraphManager/matmul_backward" {
    const allocator = std.testing.allocator;
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);
    const shape = &[_]usize{ 2, 2 };

    var t1 = try Tensor.init(&[_]T{ 1, 2, 3, 4 }, shape, true, device);
    var t2 = try Tensor.init(&[_]T{ 1, 0, 0, 1 }, shape, true, device);

    // Case 1: No transpose
    var t3 = try t1.bmm(t2, device, .{ .trans_a = false, .trans_b = false });
    defer t3.deinit();

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    t1.acquire();
    t2.acquire();
    t3.grad.?.fill(1.0, device);

    try gm.backward(t3, device);
    const expected_grad_t1 = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2 = &[_]T{ 4, 4, 6, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
    t1.grad.?.fill(0, device);
    t2.grad.?.fill(0, device);

    // Case 2: Transpose A
    var t3_trans_a = try t1.bmm(t2, device, .{ .trans_a = true, .trans_b = false });
    defer t3_trans_a.deinit();
    t3_trans_a.grad.?.fill(1.0, device);

    try gm.backward(t3_trans_a, device);
    const expected_grad_t1_trans_a = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_a = &[_]T{ 3, 3, 7, 7 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_a, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_a, t2.grad.?.data);
    t1.grad.?.fill(0, device);
    t2.grad.?.fill(0, device);

    // Case 3: Transpose B
    var t3_trans_b = try t1.bmm(t2, device, .{ .trans_a = false, .trans_b = true });
    defer t3_trans_b.deinit();
    t3_trans_b.grad.?.fill(1.0, device);
    try gm.backward(t3_trans_b, device);

    const expected_grad_t1_trans_b = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_b = &[_]T{ 4, 6, 4, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_b, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_b, t2.grad.?.data);
    t1.grad.?.fill(0, device);
    t2.grad.?.fill(0, device);

    // Case 4: Transpose both A and B
    var t3_trans_ab = try t1.bmm(t2, device, .{ .trans_a = true, .trans_b = true });
    defer t3_trans_ab.deinit();
    t3_trans_ab.grad.?.fill(1.0, device);
    try gm.backward(t3_trans_ab, device);

    const expected_grad_t1_trans_ab = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_ab = &[_]T{ 3, 7, 3, 7 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_ab, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_ab, t2.grad.?.data);

    t1.release();
    t1.deinit();
    t2.release();
    t2.deinit();
}

test "tensor/GraphManager/matvec_backward" {
    const allocator = std.testing.allocator;
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);
    const shape_mat = &[_]usize{ 2, 2 };
    const shape_vec = &[_]usize{2};

    // [1, 2] [1]
    // [3, 4] [1]
    // grad = [1, 1]'
    // dl/dA = grad * [1, 1] = [[2, 2], [2, 2]]
    // dl/dx = A' * grad = [4, 6]'
    var t1 = try Tensor.init(&[_]T{ 1, 2, 3, 4 }, shape_mat, true, device);
    var t2 = try Tensor.init(&[_]T{ 1, 1 }, shape_vec, true, device);
    var t3 = try t1.matvec(t2, device);
    defer {
        t1.deinit();
        t2.deinit();
        t3.deinit();
    }

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    t3.grad.?.fill(1.0, device);
    try gm.backward(t3, device);

    const expected_grad_t1 = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2 = &[_]T{ 4, 6 };

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}

test "tensor/GraphManager/dot_backward" {
    const allocator = std.testing.allocator;
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);
    const shape = &[_]usize{3};

    var t1 = try Tensor.init(&[_]T{ 1, 2, 3 }, shape, true, device);
    const t2 = try Tensor.init(&[_]T{ 4, 5, 6 }, shape, true, device);
    var t3 = try t1.dot(t2, device);
    defer {
        t1.deinit();
        t2.deinit();
        t3.deinit();
    }

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    t3.grad.?.fill(1.0, device);
    try gm.backward(t3, device);

    const expected_grad_t1 = &[_]T{ 4, 5, 6 };
    const expected_grad_t2 = &[_]T{ 1, 2, 3 };

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}

// TODO: Fix memory freeing conundrum with gather() then dont use an arena here.
test "tensor/gather" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);

    // case 1: basic gather
    const input_data = [_]T{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const input_shape = [_]usize{ 3, 3 };
    var input = try Tensor.init(&input_data, &input_shape, true, device);
    defer input.deinit();

    const index_data = [_]usize{ 0, 1, 1, 2, 0, 2 };
    const index_shape = [_]usize{ 3, 2 };
    var index = try NDTensor(usize).init(&index_data, &index_shape, false, device);
    defer index.deinit();

    var output = try input.gather(device, .{ .indices = index, .dim = 1 });
    defer output.deinit();

    try std.testing.expectEqualSlices(T, &[_]T{ 1, 2, 5, 6, 7, 9 }, output.data.data);
    try std.testing.expectEqualSlices(usize, &index_shape, output.data.shape.shape);

    // case 2: grad check
    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();

    output.grad.?.fill(1.0, device);
    try gm.backward(output, device);

    const expected_grad = [_]T{ 1, 1, 0, 0, 1, 1, 1, 0, 1 };
    try std.testing.expectEqualSlices(T, &expected_grad, input.grad.?.data);

    // case 3: out of bounds
    try index.set(&.{ 0, 0 }, 3);
    try std.testing.expectError(error.IndexOutOfBounds, input.gather(device, .{ .indices = index, .dim = 1 }));
}

// TODO: Fix memory freeing conundrum with maxOverDim() then dont use an arena here.
test "tensor/maxOverDim" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);

    // case 1: basic max over dim operation
    const input_data = [_]T{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const input_shape = [_]usize{ 3, 3 };
    var input = try Tensor.init(&input_data, &input_shape, true, device);
    defer input.deinit();

    var output = try input.maxOverDim(device, .{ .dim = 1 });
    defer output.deinit();

    try std.testing.expectEqualSlices(T, &[_]T{ 3, 6, 9 }, output.data.data);
    try std.testing.expectEqualSlices(usize, &[_]usize{3}, output.data.shape.shape);

    // case 2: gradient check
    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();

    output.grad.?.fill(1.0, device);
    try gm.backward(output, device);

    const expected_grad = [_]T{ 0, 0, 1, 0, 0, 1, 0, 0, 1 };
    try std.testing.expectEqualSlices(T, &expected_grad, input.grad.?.data);

    // case 3: max over different dimension
    var output2 = try input.maxOverDim(device, .{ .dim = 0 });
    defer output2.deinit();

    try std.testing.expectEqualSlices(T, &[_]T{ 7, 8, 9 }, output2.data.data);
    try std.testing.expectEqualSlices(usize, &[_]usize{3}, output2.data.shape.shape);

    // reset grads
    input.grad.?.fill(0, device);
    output2.grad.?.fill(1.0, device);
    try gm.backward(output2, device);

    const expected_grad2 = [_]T{ 0, 0, 0, 0, 0, 0, 1, 1, 1 };
    try std.testing.expectEqualSlices(T, &expected_grad2, input.grad.?.data);

    // case 4: invalid dimension
    try std.testing.expectError(error.DimOutOfBounds, input.maxOverDim(device, .{ .dim = 2 }));
}
