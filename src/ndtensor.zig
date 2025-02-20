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

pub const MaxAlongOptions = struct {
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
        const Self = @This();
        // signature for a backwards callback
        pub const Callback = *const fn (*const Self) anyerror!void;
        pub const DataType = NDArray(T);
        // 8 children is a lot but... oh well...
        pub const Children = std.BoundedArray(*Self, 8);
        // TODO: How big should a label be?
        pub const Label = std.BoundedArray(u8, 32);

        data: DataType,
        op: ?Op = null,
        children: Children = .{},
        label: Label = .{},
        grad: ?DataType = null,
        acquired: bool = false, // not inherited
        device: DeviceReference,
        // TODO: Wrap these in a proper backwards context.
        // It should have a small buffer for objects about
        // the size of a slice.
        _backward: ?Callback = null, // see notes below
        _backward_ctx: ?*anyopaque = null,
        _requires_grad: bool,

        /// Values and shape are allocated. COM.
        pub fn init(values: []const T, shape: ?[]const usize, _requires_grad: bool, device: DeviceReference) !*Self {
            const self = try Self.empty(shape orelse &.{values.len}, _requires_grad, device);
            device.mem_copy(T, values, self.get_data());
            return self;
        }

        /// Shape is allocated. COM.
        /// As it stands, with grad disabled you can still allocate grads but grads wont be tracked (or allocated) in ops
        pub fn empty(shape: []const usize, _requires_grad: bool, device: DeviceReference) !*Self {
            const self = try device.allocator.create(Self);
            self.* = Self{
                .data = try DataType.empty(shape, device),
                .grad = if (_requires_grad and zg.rt_grad_enabled) try DataType.zeros(shape, device) else null,
                ._requires_grad = _requires_grad,
                .device = device,
                .acquired = false,
                ._backward = null,
                ._backward_ctx = null,
                .children = .{},
                .label = .{},
                .op = null,
            };
            return self;
        }

        pub fn zeros(shape: []const usize, _requires_grad: bool, device: DeviceReference) !*Self {
            const self = try Self.empty(shape, _requires_grad, device);
            self.fill(0);
            return self;
        }

        pub fn random(shape: []const usize, _requires_grad: bool, op: zg.RandType, device: DeviceReference) !*Self {
            const self = try Self.empty(shape, _requires_grad, device);
            device.mem_random(T, self.get_data(), op, zg.settings.seed);
            return self;
        }

        pub fn sequence(start: T, step: T, shape: []const usize, _requires_grad: bool, device: DeviceReference) !*Self {
            const self = try Self.empty(shape, _requires_grad, device);
            device.mem_sequence(T, self.get_data(), start, step);
            return self;
        }

        pub fn get_shape(self: *const Self) []const usize {
            return self.data.shape.slice();
        }

        pub fn get_size(self: *const Self) usize {
            return self.data.data.len;
        }

        pub fn get_strides(self: *const Self) Shape.Strides {
            return self.data.shape.strides();
        }

        pub fn get_data(self: *const Self) []T {
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

        pub fn requires_grad(self: *const Self) bool {
            return self._requires_grad and zg.rt_grad_enabled;
        }

        pub fn setup_grad(self: *Self, fill_value: ?T) !void {
            if (self.grad == null) {
                self.grad = try DataType.empty(self.get_shape(), self.device);
            }
            return self.grad.?.fill(fill_value orelse return, self.device);
        }

        pub fn get_children(self: *const Self) ?[]*Self {
            // TODO: The static array propogates const-ness to the children,
            // but we know that these children are actually mutable pointers
            return if (self.children.len > 0) @constCast(self.children.slice()) else null;
        }

        pub fn set_children(self: *Self, new_children: []const *Self) !void {
            self.children = try Children.fromSlice(new_children);
        }

        pub fn get_label(self: *const Self) ?[]const u8 {
            return if (self.label.len > 0) self.label.slice() else null;
        }

        pub fn set_label(self: *Self, new_label: []const u8) !void {
            self.label = try Label.fromSlice(new_label);
        }

        pub const CreateDependentOpts = struct {
            data: DataType,
            op: ?Op = null,
            children: []const *Self,
            label: ?[]const u8 = null,
            device: DeviceReference,
            _backward: ?Callback = null,
            _backward_ctx: ?*anyopaque = null,
            _requires_grad: bool,
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
        ///   2. To support graph usage without the intention of backprop, _requires_grad=false will not allocate a grad.
        ///   3. The only foreseen situation to init with a specific `grad` value would be at the tail of a graph for backprop.
        ///      This is a cold operation and thus should be done explicitly, hence no support provided here. This drives support
        ///      for mutable `grad` pointer, so the caller can move the reference (Warning: this pattern is a red flag).
        ///   4. Should this respect the global `grad_enabled` flag and override the flag parameter? Tbd.
        ///      UPDATE: Changed my mind, overrides.
        pub fn create_dependent(opts: CreateDependentOpts) !*Self {
            const rg = opts._requires_grad and zg.rt_grad_enabled;

            const self = try opts.device.allocator.create(Self);
            errdefer opts.device.allocator.destroy(self);

            self.* = Self{
                .data = opts.data,
                .op = opts.op,
                // TODO: How to handle not holding references if requires_grad == false??
                // .children = if (rg) try opts.allocator.dupe(*const Self, opts.children) else null,
                .grad = if (rg) try DataType.zeros(opts.data.shape.slice(), opts.device) else null,
                .children = try Children.fromSlice(opts.children),
                .label = try Label.fromSlice(opts.label orelse ""),
                ._requires_grad = rg,
                .acquired = false,
                ._backward = if (rg) opts._backward else null,
                ._backward_ctx = if (rg) opts._backward_ctx else null,
                .device = opts.device,
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            if (self.acquired) std.debug.panic("Attempt to deinit an acquired tensor.", .{});

            // log.debug("deinit().data {?s}", .{self.get_label()});
            self.data.deinit(self.device);
            if (self.grad) |*g| {
                // log.debug("deinit().grad {?s}", .{self.get_label()});
                g.deinit(self.device);
            }
            self.device.allocator.destroy(self);
        }

        pub fn teardown(self: *Self) void {
            log.debug("START teardown {?s}", .{self.get_label()});
            if (self.acquired) std.debug.panic("Attempt to deinit an acquired tensor.", .{});
            if (self.get_children()) |children| {
                for (children) |c| {
                    log.debug("{?s} accessing child {?s}", .{ self.get_label(), c.get_label() });
                    if (!c.acquired) c.teardown() else log.warn("skipping acquired tensor in teardown label={?s}", .{c.get_label()});
                }
            }
            log.debug("ENDING teardown()->deinit() {?s}", .{self.get_label()});
            self.deinit();
        }

        pub fn acquire(self: *Self) void {
            self.acquired = true;
        }

        pub fn release(self: *Self) void {
            self.acquired = false;
        }

        /// Values are not allocated, grad_shape is allocated. COM.
        pub fn from_zarray(values: *DataType, _requires_grad: bool, grad_shape: ?Shape, device: DeviceReference) !*Self {
            log.warn("Consider create_dependent() if you are about to set children. This may be deprecated.", .{});
            const result = try device.allocator.create(Self);
            const grad: ?*DataType = blk: {
                if (_requires_grad) {
                    if (grad_shape) |s| {
                        log.warn("`grad_shape` may be deprecated.", .{});
                        break :blk try DataType.zeros(s.shape, device);
                    } else {
                        break :blk (try DataType.zeros(values.shape.shape, device));
                    }
                } else {
                    break :blk null;
                }
            };
            result.* = Self{
                .data = values,
                .grad = grad,
                ._requires_grad = _requires_grad and zg.rt_grad_enabled,
                .device = device,
            };
            return result;
        }

        fn to_device_impl(
            src: []const T,
            dst: []T,
            src_device: DeviceReference,
            dst_device: DeviceReference,
        ) !void {
            if (comptime @typeInfo(@TypeOf(src_device)) != .Struct)
                return;

            // currently only implemented for a single aux device.
            std.debug.assert(!src_device.is_compatible(dst_device));

            if (src_device.is_host()) {
                dst_device.sync();
                dst_device.mem_transfer(T, src, dst, .HtoD);
            } else {
                src_device.sync();
                src_device.mem_transfer(T, src, dst, .DtoH);
            }
        }

        pub fn _to_device(self: *Self, device: DeviceReference) !void {
            if (self.device.is_compatible(device))
                return;

            const data = try device.mem_alloc(T, self.data.data.len);
            try to_device_impl(self.data.data, data, self.device, device);
            self.device.mem_free(self.data.data);
            self.data.data = data;
            self.device = device;
        }

        pub fn to_device(self: *Self, device: DeviceReference) !*Self {
            if (self.device.is_compatible(device))
                return self;

            const to_device_bw = struct {
                fn to_device_bw_impl(_self: *const Self) !void {
                    const child = (_self.get_children() orelse return error.NoChildren)[0];
                    try to_device_impl(_self.grad.?.data, child.grad.?.data, _self.device, child.device);
                }
            }.to_device_bw_impl;

            const data = try device.mem_alloc(T, self.data.data.len);
            try to_device_impl(self.data.data, data, self.device, device);

            var result = try create_dependent(.{
                .data = .{
                    .data = data,
                    .shape = self.data.shape,
                    .view = false,
                },
                .op = .TRANSFER,
                .children = &.{self},
                .label = null,
                .device = device,
                ._backward = to_device_bw,
                ._requires_grad = false,
            });

            errdefer result.deinit();
            if (self.requires_grad()) { // need to make this call, not check the attribute flag
                result.grad = try DataType.zeros(self.get_shape(), device);
                result._requires_grad = true;
            }
            return result;
        }

        // NOTE: Check callsites and see if we can fix the comments here
        /// Only data and grad are copied. Child references and other metadata is not retained aside from _requires_grad.
        /// Passed allocator used for allocations, not the owned one (i.e. not `@This()` one).
        ///
        /// ---
        /// # ADR
        ///   - The choice to have an allocator provided is important, intended to be used for backward
        ///   - If the tensor backing `DataType` changes its allocator ownership contract, then this needs to be changed
        pub fn clone(self: *const Self) !*Self {
            const result = try self.device.allocator.create(Self);
            errdefer self.device.allocator.destroy(result);
            result.* = Self{
                .data = try self.data.copy(self.device),
                .grad = if (self.grad) |g| try g.copy(self.device) else null,
                ._requires_grad = self.requires_grad(),
                .op = null,
                .children = .{},
                .label = .{},
                .acquired = false,
                .device = self.device,
                ._backward = null,
                ._backward_ctx = null,
            };
            return result;
        }

        pub fn log_shape(self: *const Self, comptime msg: ?[]const u8) void {
            log.debug("{s}{s} data shape: {d} grad shape: {?d}", .{
                if (msg) |n| n else "",
                if (self.get_label()) |l| l else "",
                self.data.shape.shape,
                if (self.grad) |g| g.shape.shape else null,
            });
        }

        /// In-place, no backward.
        pub fn _reshape(self: *Self, shape: []const usize) !void {
            self.data._reshape(shape);
            if (self.grad) |*g| g._reshape(shape);
        }

        /// Copies. COM.
        pub fn reshape(self: *Self, new_shape: []const usize) !*Self {
            const reshape_bw = struct {
                fn reshape_bw_impl(_self: *const Self) !void {
                    const children = _self.get_children() orelse return error.NoChildren;
                    const original_shape = children[0].data.shape;
                    try self.grad.?._reshape(original_shape.shape);
                    _ = try children[0].grad.?._add(self.grad.?);
                }
            }.reshape_bw_impl;

            const result = try create_dependent(.{
                .data = try self.data.copy(self.device),
                .op = .RESHAPE,
                .children = &.{self},
                .label = null,
                ._requires_grad = self._requires_grad,
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
                fn transpose_bw_impl(_self: *const Self) !void {
                    const children = _self.get_children() orelse return error.NoChildren;
                    const gt = try self.grad.?.transpose(_self.device);
                    defer gt.deinit(_self.device);
                    _ = try children[0].grad.?._add(gt);
                }
            }.transpose_bw_impl;
            var result = try create_dependent(.{
                .data = try self.data.transpose(self.device),
                .op = .TRANSPOSE,
                .children = &.{self},
                .label = null,
                ._requires_grad = false, // we will set grad ourselves for efficiency
                .device = self.device,
                ._backward = transpose_bw,
            });
            errdefer result.deinit();
            if (self.requires_grad()) { // need to make this call, not check the attribute flag
                result.grad = try self.grad.?.transpose(self.device);
                result._requires_grad = true;
            }
            return result;
        }

        pub fn fill(self: *const Self, val: T) void {
            self.data.fill(val, self.device);
        }

        // this needs to get audited for device safety
        pub fn get(self: *const Self, idx: usize) T {
            return self.data.data[idx];
        }

        // this needs to get audited for device safety
        pub fn set(self: *const Self, idx: usize, value: T) !void {
            self.data.data[idx] = value;
        }

        fn pos_to_index(self: *const Self, indices: []const usize) usize {
            return self.data.pos_to_offset(indices);
        }

        fn flex_pos_to_index(self: *const Self, indices: []const usize) error.InvalidIndex!usize {
            return self.data.flexPosToOffset(indices);
        }

        /// COM
        fn index_to_pos(self: *const Self, index: usize) []const usize {
            return self.data.offset_to_pos(index, self.device.allocator);
        }

        /// [WIP] Values and grads are views into self. Shapes are allocated and COM.
        ///
        /// ---
        /// # ADR
        ///   - It is assumed the caller wants a temporary mutable view, return by copy.
        ///   - This creates a complex situation, there is no backward for this operation and it returns a mutable view.
        ///   - Caller must set backward
        pub fn slice_ranges(self: *const Self, ranges: []const Range) !Self {
            log.err("WIP", .{});
            const sliced_data = try self.data.slice_ranges(ranges);
            return Self{
                .data = sliced_data,
                .grad = if (self.requires_grad()) try self.grad.?.slice_ranges(ranges) else null,
                ._requires_grad = self.requires_grad(),
                .op = null,
                .children = null,
                .label = null,
                .acquired = false,
                .device = self.device,
                ._backward = null,
                ._backward_ctx = null,
            };
        }

        pub fn set_slice(self: *const Self, ranges: []const Range, values: Self) !void {
            if (self.requires_grad()) {
                // need to create a new operation
                @compileError("Not implemented");
            } else {
                // if not tracking gradients can just set the values directly
                try self.data.set_slice_ranges(ranges, values.data.*);
            }
        }

        pub fn print(self: *const Self) void {
            // self.print_to_writer(std.io.getStdOut().writer());
            self.print_to_writer(std.io.getStdErr().writer()) catch @panic("Failed to print tensor");
        }

        pub fn print_to_writer(self: *const Self, writer: anytype) !void {
            try writer.print("NDTensor<{},{?s}>", .{ T, if (self.op) |o| @tagName(o) else null });
            try writer.writeAll("{data: ");
            try self.data.print_to_writer(writer, self.device);
            if (self.grad) |g| {
                try writer.writeAll(" grad: ");
                try g.print_to_writer(writer, self.device);
            }
            try writer.print(", _requires_grad={}", .{self._requires_grad});
            if (self.get_label()) |l| {
                try writer.print(", label={s}", .{l});
            }
            try writer.writeAll("}\n");
        }

        pub const ClipOptions = struct {
            max_norm: f32 = settings.grad_clip_max_norm,
            delta: f32 = settings.grad_clip_delta,
        };

        pub fn _clip_grad_norm(self: *const Self, opts: ClipOptions) void {
            self.grad.?._clip_norm(opts.max_norm, opts.delta, self.device);
        }

        /// Direct modification. Clamps the underlying data, as with all in place ops you must know what you are doing.
        /// This operation is not tracked in the computation graph.
        /// *Will not notify you of an improper gradient calculation.*
        pub fn _clamp(self: *const Self, vmin: T, vmax: T) void {
            self.data._clamp(vmin, vmax, self.device);
        }

        /// Direct modification. Clamps the underlying grad, as with all in place ops you must know what you are doing.
        /// This operation is not tracked in the computation graph.
        /// *Will not notify you of an improper gradient calculation.*
        /// Grad must exist.
        pub fn _clamp_grad(self: *const Self, vmin: T, vmax: T) .NoGradient!void {
            (self.grad orelse return error.NoGradient)._clamp(vmin, vmax, self.device);
        }

        /// Differentiable
        pub fn clamp(self: *Self, vmin: T, vmax: T) !*Self {
            // TODO: implement differentiable clamp()
            // if grad is required, use data._clamp_with_mask() so we can save for backward
            // otherwise regular clamp
            std.debug.assert(vmin <= vmax);

            var result_data = try self.data.copy(self.device);
            const mask = try self.device.mem_alloc(u1, self.get_size());
            result_data._clamp_with_mask(vmin, vmax, mask, self.device);

            const clampBw = struct {
                fn clamp_bw_impl(_self: Self) !void {
                    const ctx: [*]u1 = @ptrCast(@alignCast(_self._backward_ctx orelse return error.NoBackwardContext));
                    const children = _self.get_children() orelse return error.NoChildren;
                    const input = children[0];

                    const _mask = ctx[0..input.get_size()];

                    // mask gradient, zero out grad for entries that were clamped
                    // FIXME: problem here, _mask is []u1 but mul expects []T
                    _self.device.blas.mul(T, _mask, _self.grad.?.data, input.grad.?.data);
                    _self.device.mem_free(ctx);
                }
            }.clamp_bw_impl;

            return create_dependent(.{
                .data = result_data,
                .op = null, // TODO: clamp op type
                .children = &.{self},
                ._requires_grad = self._requires_grad,
                .device = self.device,
                ._backward = clampBw,
                ._backward_ctx = @ptrCast(mask),
            });
        }

        /// Element-wise addition. COM.
        pub fn add(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const addBw = struct {
                fn add_bw_impl(_self: *const Self) !void {
                    const children = _self.get_children() orelse return error.NoChildren;
                    const a = children[0];
                    const b = children[1];

                    var a_grad = try _self.grad.?.unbroadcast(a.grad.?.shape, _self.device);
                    defer a_grad.deinit(_self.device);

                    var b_grad = try _self.grad.?.unbroadcast(b.grad.?.shape, _self.device);
                    defer b_grad.deinit(_self.device);

                    _ = try a.grad.?._add(a_grad, a.device);
                    _ = try b.grad.?._add(b_grad, b.device);
                }
            }.add_bw_impl;
            return create_dependent(.{
                .data = try self.data.add(other.data, self.device),
                .op = .ADD,
                .children = &.{ self, other },
                ._requires_grad = self._requires_grad or other._requires_grad,
                .device = self.device,
                ._backward = addBw,
            });
        }

        /// Element-wise subtraction. COM.
        pub fn sub(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const subBw = struct {
                fn sub_bw_impl(_self: *const Self) !void {
                    const children = _self.get_children() orelse return error.NoChildren;
                    const a = children[0];
                    const b = children[1];

                    var a_grad = try (try _self.grad.?.copy(_self.device)).unbroadcast(a.grad.?.shape, _self.device);
                    defer a_grad.deinit(_self.device);

                    var b_grad = try (try _self.grad.?.copy(_self.device)).unbroadcast(b.grad.?.shape, _self.device);
                    defer b_grad.deinit(_self.device);

                    _ = try a.grad.?._add(a_grad, a.device);
                    _ = try b.grad.?._sub(b_grad, b.device);
                }
            }.sub_bw_impl;
            return create_dependent(.{
                .data = try self.data.sub(other.data, self.device),
                .op = .SUB,
                .children = &.{ self, other },
                ._requires_grad = self._requires_grad or other._requires_grad,
                .device = self.device,
                ._backward = subBw,
            });
        }

        /// Element-wise multiplication. COM.
        pub fn mul(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const mulBw = struct {
                fn mul_bw_impl(_self: *const Self) !void {
                    const children = _self.get_children() orelse return error.NoChildren;
                    const a = children[0];
                    const b = children[1];

                    // TODO: can remove a copy here

                    // (dL/dy) * (dy/da), (dL/dy) * (dy/db)
                    var a_grad_value = try b.data.mul(_self.grad.?, _self.device);
                    defer a_grad_value.deinit(_self.device);

                    var b_grad_value = try a.data.mul(_self.grad.?, _self.device);
                    defer b_grad_value.deinit(_self.device);

                    // TODO: Do we need to do this here? _add automatically broadcasts
                    var a_grad = try a_grad_value.unbroadcast(a.grad.?.shape, _self.device);
                    defer a_grad.deinit(_self.device);

                    var b_grad = try b_grad_value.unbroadcast(b.grad.?.shape, _self.device);
                    defer b_grad.deinit(_self.device);

                    _ = try a.grad.?._add(a_grad, a.device);
                    _ = try b.grad.?._add(b_grad, b.device);
                }
            }.mul_bw_impl;
            return create_dependent(.{
                .data = try self.data.mul(other.data, self.device),
                .op = .MUL,
                .children = &.{ self, other },
                ._requires_grad = self._requires_grad or other._requires_grad,
                .device = self.device,
                ._backward = mulBw,
            });
        }

        /// Element-wise division. COM.
        pub fn div(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const divBw = struct {
                fn div_bw_impl(_self: *const Self) !void {
                    const children = _self.get_children() orelse return error.NoChildren;
                    const a = children[0];
                    const b = children[1];

                    // TODO: can remove at least one copy here right?

                    // (dL/dy) * (dy/da) and (dL/dy) * (dy/db)
                    var a_grad_value = try _self.grad.?.div(b.data, _self.device);
                    defer a_grad_value.deinit(_self.device);

                    var b_grad_value = try _self.grad.?.mul(a.data, _self.device);
                    defer b_grad_value.deinit(_self.device);

                    var bsq = try b.data.mul(b.data, _self.device);
                    defer bsq.deinit(_self.device);

                    var neg_b_grad_value = try b_grad_value.div(bsq, _self.device);
                    defer neg_b_grad_value.deinit(_self.device);

                    var a_grad = try a_grad_value.unbroadcast(a.grad.?.shape, _self.device);
                    defer a_grad.deinit(_self.device);

                    var b_grad = try neg_b_grad_value.unbroadcast(b.grad.?.shape, _self.device);
                    defer b_grad.deinit(_self.device);

                    _ = try a.grad.?._add(a_grad, a.device);
                    _ = try b.grad.?._sub(b_grad, b.device);
                }
            }.div_bw_impl;
            return create_dependent(.{
                .data = try self.data.div(other.data, self.device),
                .op = .DIV,
                .children = &.{ self, other },
                ._requires_grad = self._requires_grad or other._requires_grad,
                .device = self.device,
                ._backward = divBw,
            });
        }

        /// Computes the maximum value of the tensor. Returns a scalar tensor. COM.
        pub fn max(self: *Self) !*Self {
            const max_mem = try DataType.empty(&.{1}, self.device);

            self.device.blas.max_forward(T, self.get_data(), max_mem.data);

            const maxBw = struct {
                fn max_bw_impl(_self: *const Self) !void {
                    const children = _self.get_children() orelse return error.NoChildren;
                    const child = children[0];
                    _self.device.blas.max_backward(
                        T,
                        child.get_data(),
                        _self.get_data(),
                        _self.grad.?.data,
                        child.grad.?.data,
                    );
                }
            }.max_bw_impl;

            return create_dependent(.{
                .data = max_mem,
                .op = .MAX,
                .children = &.{self},
                ._requires_grad = self._requires_grad,
                .device = self.device,
                ._backward = maxBw,
            });
        }

        /// TODO: exp bw backend
        /// Element-wise exponential. COM.
        pub fn exp(self: *Self) !*Self {
            const expBw = struct {
                fn exp_bw_impl(_self: *const Self) !void {
                    const children = _self.get_children() orelse return error.NoChildren;
                    const child = children[0];
                    for (_self.data.data, _self.grad.?.data, 0..) |exp_val, grad_val, i| {
                        child.grad.?.data[i] += exp_val * grad_val;
                    }
                }
            }.exp_bw_impl;
            return create_dependent(.{
                .data = try self.data.exp(self.device),
                .op = .EXP,
                .children = &[_]*const Self{self},
                ._requires_grad = self._requires_grad,
                .device = self.device,
                ._backward = expBw,
            });
        }

        pub const MmOptions = struct { trans_a: bool = false, trans_b: bool = false };

        /// TODO: this should proxy to bmm_acc
        /// Matrix multiplication. COM.
        pub fn bmm(self: *Self, other: *Self, opts: MmOptions) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const ctx = try self.device.allocator.create(MmAccOptions);
            ctx.* = MmAccOptions{ .trans_a = opts.trans_a, .trans_b = opts.trans_b, .alpha = 1, .beta = 0 };
            const result = try create_dependent(.{
                .data = try self.data.bmm(other.data, opts.trans_a, opts.trans_b, self.device),
                .children = &.{ self, other },
                ._requires_grad = self._requires_grad or other._requires_grad,
                .device = self.device,
                ._backward = bw_bmm_acc,
                ._backward_ctx = ctx,
            });
            result.op = if (!opts.trans_a and !opts.trans_b) .MATMUL_AB else if (opts.trans_a and !opts.trans_b) .MATMUL_AtB else if (!opts.trans_a and opts.trans_b) .MATMUL_ABt else .MATMUL_AtBt;
            return result;
        }

        pub const MmAccOptions = struct { trans_a: bool = false, trans_b: bool = false, alpha: T = 1.0, beta: T = 1.0 };

        /// Matrix multiplication w accumulation. COM, output is written to directly.
        pub fn bmm_acc(
            self: *const NDTensor(T),
            other: *const NDTensor(T),
            output: *NDTensor(T),
            opts: MmAccOptions,
        ) !void {
            std.debug.assert(self.device.is_compatible(other.device));

            _ = try self.data._bmm_acc(
                other.data,
                output.data,
                opts.alpha,
                opts.beta,
                opts.trans_a,
                opts.trans_b,
            );

            const _requires_grad = zg.rt_grad_enabled and (self._requires_grad or other._requires_grad or output._requires_grad);
            if (_requires_grad) {
                const ctx = try self.device.allocator.create(MmAccOptions);
                ctx.* = opts;
                output.children = try self.device.allocator.dupe(*NDTensor(T), &.{ self, other });
                output._backward = bw_bmm_acc;
                output._backward_ctx = ctx;
                output._requires_grad = true;
            }
        }

        fn bw_bmm_acc(self: *const NDTensor(T)) !void {
            const grad_C = self.grad orelse return error.NoGradient;

            const opts: *MmAccOptions = @ptrCast(@alignCast(self._backward_ctx orelse return error.NoBackwardContext));
            defer self.device.allocator.destroy(opts);

            const children = self.get_children() orelse return error.NoChildren;
            const A = children[0].data;
            const B = children[1].data;
            var grad_A = children[0].grad.?;
            var grad_B = children[1].grad.?;

            if (self.op) |op| {
                switch (op) {
                    .MATMUL_AB => {
                        // A Shape: (..., m, k)
                        // B Shape: (..., k, n)
                        // grad_C Shape: (..., m, n)
                        // grad_A += grad_C * B'
                        _ = try grad_C._bmm_acc(B, &grad_A, 1.0, 1.0, false, true, self.device);
                        // grad_B += A' * grad_C
                        _ = try A._bmm_acc(grad_C, &grad_B, 1.0, 1.0, true, false, self.device);
                    },
                    .MATMUL_AtB => {
                        // A Shape: (..., k, m)
                        // B Shape: (..., k, n)
                        // grad_C Shape: (..., m, n)
                        // grad_A += B * grad_C'
                        _ = try B._bmm_acc(self.grad.?, &grad_A, 1.0, 1.0, false, true, self.device);
                        // grad_B += A * grad_C
                        _ = try A._bmm_acc(self.grad.?, &grad_B, 1.0, 1.0, false, false, self.device);
                    },
                    .MATMUL_ABt => {
                        // A Shape: (..., m, k)
                        // B Shape: (..., n, k)
                        // grad_C Shape: (..., m, n)
                        // grad_A += grad_C * B
                        _ = try grad_C._bmm_acc(B, &grad_A, 1.0, 1.0, false, false, self.device);
                        // grad_B += grad_C' * A
                        _ = try grad_C._bmm_acc(A, &grad_B, 1.0, 1.0, true, false, self.device);
                    },
                    .MATMUL_AtBt => {
                        // A Shape: (..., k, m)
                        // B Shape: (..., n, k)
                        // grad_C Shape: (..., m, n)
                        // grad_A += B' * grad_C'
                        _ = try B._bmm_acc(grad_C, &grad_A, 1.0, 1.0, true, true, self.device);
                        // grad_B += grad_C * A'
                        _ = try grad_C._bmm_acc(A, &grad_B, 1.0, 1.0, false, true, self.device);
                    },
                    else => std.debug.panic("Op {s} is not yet implemented.", .{@tagName(op)}),
                }
            }
        }

        /// Dot product of two tensors. COM.
        pub fn dot(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const dotBw = struct {
                fn dot_bw_impl(_self: *const Self) !void {
                    const children = _self.get_children() orelse return error.NoChildren;
                    var a = children[0];
                    var b = children[1];
                    const gscalar: *const T = @ptrCast(_self.grad.?.data.ptr);
                    // In forward: c = dot(a, b) aka _self = dot(self, other)
                    // Note that c' is a scalar. axpy performs y += a*x.
                    // a' += c'*b
                    _self.device.blas.axpy(T, b.get_data(), a.grad.?.data, gscalar);
                    // b' += c'*a
                    _self.device.blas.axpy(T, a.get_data(), b.grad.?.data, gscalar);
                }
            }.dot_bw_impl;

            return create_dependent(.{
                .data = try self.data.dot(other.data, self.device),
                .op = .DOT,
                .children = &.{ self, other },
                ._requires_grad = self._requires_grad or other._requires_grad,
                .device = self.device,
                ._backward = dotBw,
            });
        }

        /// Matrix-vector multiplication. COM.
        /// ---
        /// # ADR
        ///   - This does not use `create_dependent()` as it is a special case where the grad shape different from data.
        ///   - Moreover, we cannot just reshape to the correct shape.
        ///   - Until I see more instances where this is required it will be written manually
        ///   - Edit: Think I got mixed up, this can prob be undone, but working now.
        pub fn matvec(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const matvecBw = struct {
                /// TODO: Use accumulation
                fn matvec_bw_impl(_self: *const Self) !void {
                    const children = _self.get_children() orelse return error.NoChildren;
                    const A = children[0].data;
                    const x = children[1].data;

                    //  L(y), y = Ax, dL/dA = (dL/dy)(dy/dA) = (dL/dy)x'
                    var grad_A = try _self.grad.?.outer(x, _self.device);
                    defer grad_A.deinit(_self.device);
                    _ = try children[0].grad.?._add(grad_A, _self.device);

                    //  L(y), y = Ax, dL/dx = (dL/dy)(dy/dx) = A'(dL/dy)
                    var grad_x = try A.matvec(_self.grad.?, true, _self.device);
                    defer grad_x.deinit(_self.device);
                    _ = try children[1].grad.?._add(grad_x, _self.device);
                }
            }.matvec_bw_impl;

            return create_dependent(.{
                .data = try self.data.matvec(other.data, false, self.device),
                .op = .MATVEC,
                .children = &.{ self, other },
                ._requires_grad = true, // self.requires_grad() or other.requires_grad(),
                .device = self.device,
                ._backward = matvecBw,
            });
        }

        /// Performs `self = alpha*other + self` in place.
        pub fn _axpy(self: *const Self, other: Self, alpha: T) void {
            std.debug.assert(self.device.is_compatible(other.device));
            self.data._axpy(other, alpha, self.device);
        }

        /// Sum of all elements in the tensor. COM.
        pub fn sum(self: *Self) !*Self {
            const sumBw = struct {
                fn sum_bw_impl(_self: *const NDTensor(T)) !void {
                    const children = _self.get_children() orelse return error.NoChildren;
                    const child = children[0];
                    _ = try child.grad.?._add(_self.grad.?, child.device);
                }
            }.sum_bw_impl;
            return create_dependent(.{
                .data = try self.data.sum(self.device),
                .op = .SUM,
                .children = &.{self},
                ._requires_grad = self._requires_grad,
                .device = self.device,
                ._backward = sumBw,
            });
        }

        //pub fn max_along(self: *Self, device: DeviceReference, opts: MaxAlongOptions) !*Self {
        //    const max_backward = struct {
        //        // NOTE: See gather() comments, same apply here
        //        fn bw_impl(_self: Self) !void {
        //            const bw_children = _self.get_children() orelse return error.NoChildren;
        //            const bw_input = bw_children[0];
        //            if (bw_input.grad == null) return;
        //            const raw_offsets: [*:0]usize = @ptrCast(@alignCast(_self._backward_ctx orelse return error.NoBackwardContext));
        //            const offsets: []usize = std.mem.span(raw_offsets);
        //            for (0.._self.data.data.len) |i| bw_input.grad.?.data[offsets[i]] += _self.grad.?.data[i];
        //            _self.device.allocator.free(offsets);
        //        }
        //    }.bw_impl;

        //    const max_result = try self.data.max_over_dim(device, .{
        //        .dim = opts.dim,
        //        .keep_dims = opts.keep_dims,
        //        .return_offsets = true,
        //    });
        //    // TODO: use a null terminated allocation instead, tired rn
        //    const ctx = if (self.requires_grad()) try device.allocator.dupeZ(usize, max_result.offsets.?) else null;
        //    if (max_result.offsets) |offs| device.allocator.free(offs);

        //    return create_dependent(.{
        //        .data = max_result.values,
        //        .op = null,
        //        .children = &.{self},
        //        ._requires_grad = self._requires_grad,
        //        .device = device,
        //        ._backward = max_backward,
        //        ._backward_ctx = if (ctx) |c| c.ptr else null,
        //    });
        //}

        pub fn gather(self: *Self, device: DeviceReference, opts: GatherOptions) !*Self {
            const gatherBackward = struct {
                fn bw_impl(bw_tensor: *const NDTensor(T)) !void {
                    const bw_children = bw_tensor.get_children() orelse return error.NoChildren;
                    const bw_input = bw_children[0];
                    if (bw_input.grad == null) return;
                    const offsets: [*]usize = @ptrCast(@alignCast(bw_tensor._backward_ctx orelse return error.NoBackwardContext));
                    // how am i expected to free this, unknown len
                    // defer _self.device.raw(offsets); // FIXME: just occuring to me the problem with this, if _self.device != fwd_allocator

                    // bw_tensor must/should be the same len as indices used to index (note that offsets is a raw c ptr without a len)
                    // std.debug.assert(offsets.len == bw_tensor.data.data.len); // can make this a real check when its a  null term alloc
                    for (0..bw_tensor.data.data.len) |i| bw_input.grad.?.data[offsets[i]] += bw_tensor.grad.?.data[i];
                }
            }.bw_impl;

            const gather_result = try self.data.gather(device, .{ .indices = opts.indices.data, .dim = opts.dim, .return_offsets = true });
            // TODO: use a null terminated allocation instead, tired rn
            const ctx = if (self.requires_grad()) try device.allocator.dupe(usize, gather_result.offsets.?) else null;

            return create_dependent(.{
                .data = gather_result.values,
                .op = null,
                .children = &.{self},
                ._requires_grad = self._requires_grad,
                .device = device,
                ._backward = gatherBackward,
                ._backward_ctx = if (ctx) |c| c.ptr else null,
            });
        }

        /// Callback is highly dynamic so passing a reference may be a better idea for _backward callback,
        /// but committing to compiler reliance in this refactor
        pub fn set_backward(self: *Self, backward_fn: Callback, ctx: ?*anyopaque) void {
            self._backward = backward_fn;
            self._backward_ctx = ctx;
        }

        pub fn backward(self: *const Self) !void {
            // hypothetically, we could check for children. This is treating self as detached, tbd if this is a good idea.
            if (!zg.rt_grad_enabled) return error.GradNotEnabled;
            if (!self._requires_grad) return;
            if (self._backward) |f| {
                try f(self);
                return;
            }
            if (self.op) |op| {
                std.debug.panic("Op {s} backward not implemented.", .{@tagName(op)});
            }
        }

        /// Prints dynamic compuation graph in d2 format with ops as and operands as nodes
        pub fn print_arrows(self: *const Self) void {
            if (self.get_children()) |children| {
                for (children) |elem| {
                    std.debug.print("{?s}<-{?s}", .{ self.get_label(), elem.get_label() });
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
                std.debug.print("{?s}\n", .{self.get_label()});
            }
        }
    };
}

test "ndtensor/clamp fw,bw" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var cpu = zg.device.HostDevice.init(arena.allocator());
    defer cpu.deinit();

    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);

    const vmin: f32 = -1.0;
    const vmax: f32 = 1.0;

    var x = try Tensor.init(&.{ -2.0, -0.5, 0.5, 2.0 }, null, true, device);
    defer x.deinit();

    var y = try x.clamp(vmin, vmax);
    defer y.deinit();

    try y.setup_grad(1.0);
    try y.backward();

    const expected_output: []const f32 = &.{ -1.0, -0.5, 0.5, 1.0 };
    const expected_grad: []const f32 = &.{ 0.0, 1.0, 1.0, 0.0 };

    try std.testing.expectEqualSlices(T, expected_output, y.get_data());
    try std.testing.expectEqualSlices(T, expected_grad, y.grad.?.data);
}

test "tensor/GraphManager/sum" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var cpu = zg.device.HostDevice.init(arena.allocator());
    defer cpu.deinit();

    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);

    var input = try Tensor.init(&[_]T{ 1, 2, 3, 4 }, &[_]usize{4}, true, device);
    _ = try input.set_label("input");
    var sum_result = try input.sum();
    _ = try sum_result.set_label("sum_result");

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
    try gm.backward(sum_result);

    const expected_grad = &[_]T{ 1, 1, 1, 1 };
    try std.testing.expectEqualSlices(T, expected_grad, input.grad.?.data);
}

test "tensor/NDTensor index, add, div" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

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
    //try t1.set(&[_]usize{ 1, 2 }, 1.1);
    //try std.testing.expectEqual(1.1, t1.get(&.{ 1, 2 }));

    const t2 = try Tensor.init(&[_]f32{ 10, 20, 30, 40, 50, 60 }, shape, false, device);
    const t3 = try t1.add(t2);
    const t4 = try t3.sub(t1);

    defer {
        t1.deinit();
        t2.deinit();
        t3.deinit();
        t4.deinit();
    }

    try std.testing.expectEqualSlices(f32, t2.data.data, t4.data.data);
}

test "tensor/GraphManager/addback" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

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
    var t3 = try t1.add(t2);
    defer {
        t1.deinit();
        t2.deinit();
        t3.deinit();
    }

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    t3.grad.?.fill(1.0, device);
    try gm.backward(t3);
    try std.testing.expectEqualDeep(&[_]T{1.0}, t1.grad.?.data);
    try std.testing.expectEqualDeep(&[_]T{1.0}, t2.grad.?.data);
}

test "tensor/GraphManager/mulback" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

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
    var t3 = try t1.mul(t2);
    defer {
        t1.deinit();
        t2.deinit();
        t3.deinit();
    }

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    t3.grad.?.fill(1.0, device);
    try gm.backward(t3);
    try std.testing.expectEqualDeep(t2.data.data, t1.grad.?.data);
    try std.testing.expectEqualDeep(t1.data.data, t2.grad.?.data);
}

test "tensor/GraphManager/moreback" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

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
    const temp = try w.mul(x);
    defer temp.deinit();
    const h = try temp.add(b);
    defer h.deinit();

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    h.grad.?.fill(1.0, device);
    try gm.backward(h);
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
    const temp2 = try w.mul(x);
    defer temp2.deinit();
    const h2 = try temp2.add(b);
    defer h2.deinit();

    var gm2 = GraphManager(Tensor).init(device.allocator, .{});
    defer gm2.deinit();
    h2.grad.?.fill(1.0, device);
    try gm2.backward(h2);
    try std.testing.expectEqualSlices(T, x.data.data, w.grad.?.data);
    try std.testing.expect(std.mem.allEqual(T, b.grad.?.data, 1));
}

test "tensor/GraphManager/divback" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);
    const shape = &[_]usize{2};

    var t1 = try Tensor.init(&[_]T{ 4, 9 }, shape, true, device);
    var t2 = try Tensor.init(&[_]T{ 2, 3 }, shape, true, device);
    var t3 = try t1.div(t2);
    defer {
        t1.deinit();
        t2.deinit();
        t3.deinit();
    }

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    t3.grad.?.fill(1.0, device);
    try gm.backward(t3);

    const expected_grad_t1 = &[_]T{ 1.0 / 2.0, 1.0 / 3.0 }; // 1 / b
    const expected_grad_t2 = &[_]T{ -4.0 / 4.0, -9.0 / 9.0 }; // -a / b^2

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}

test "tensor/GraphManager/matmul_backward square" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);
    const shape = &[_]usize{ 2, 2 };

    var t1 = try Tensor.init(&[_]T{ 1, 2, 3, 4 }, shape, true, device);
    var t2 = try Tensor.init(&[_]T{ 1, 0, 0, 1 }, shape, true, device);

    // Case 1: No transpose
    var t3 = try t1.bmm(t2, .{ .trans_a = false, .trans_b = false });

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    defer {
        t1.deinit();
        t2.deinit();
        t3.deinit();
    }
    t3.grad.?.fill(1.0, device);

    try gm.backward(t3);
    const expected_grad_t1 = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2 = &[_]T{ 4, 4, 6, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
    t1.grad.?.fill(0, device);
    t2.grad.?.fill(0, device);

    // Case 2: Transpose A
    var t3_trans_a = try t1.bmm(t2, .{ .trans_a = true, .trans_b = false });
    defer t3_trans_a.deinit();
    t3_trans_a.grad.?.fill(1.0, device);

    try gm.backward(t3_trans_a);
    const expected_grad_t1_trans_a = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_a = &[_]T{ 3, 3, 7, 7 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_a, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_a, t2.grad.?.data);
    t1.grad.?.fill(0, device);
    t2.grad.?.fill(0, device);

    // Case 3: Transpose B
    var t3_trans_b = try t1.bmm(t2, .{ .trans_a = false, .trans_b = true });
    defer t3_trans_b.deinit();
    t3_trans_b.grad.?.fill(1.0, device);
    try gm.backward(t3_trans_b);

    const expected_grad_t1_trans_b = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_b = &[_]T{ 4, 6, 4, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_b, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_b, t2.grad.?.data);
    t1.grad.?.fill(0, device);
    t2.grad.?.fill(0, device);

    // Case 4: Transpose both A and B
    var t3_trans_ab = try t1.bmm(t2, .{ .trans_a = true, .trans_b = true });
    defer t3_trans_ab.deinit();
    t3_trans_ab.grad.?.fill(1.0, device);
    try gm.backward(t3_trans_ab);

    const expected_grad_t1_trans_ab = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_ab = &[_]T{ 3, 7, 3, 7 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_ab, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_ab, t2.grad.?.data);
}

test "tensor/GraphManager/matmul_backward non-square" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

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
        var t3 = try t1.bmm(t2, .{ .trans_a = false, .trans_b = false });
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
        var t3 = try t1_case2.bmm(t2, .{ .trans_a = true, .trans_b = false });
        defer t3.deinit();
        t1_case2.acquire();
        t3.grad.?.fill(1.0, device);
        try gm.backward(t3);

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
        var t3 = try t1.bmm(t2_case3, .{ .trans_a = false, .trans_b = true });
        defer t3.deinit();
        t2_case3.acquire();
        t3.grad.?.fill(1.0, device);
        try gm.backward(t3);

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
        var t3 = try t1_case4.bmm(t2_case4, .{ .trans_a = true, .trans_b = true });
        defer t3.deinit();
        t1_case4.acquire();
        t2_case4.acquire();
        t3.grad.?.fill(1.0, device);
        try gm.backward(t3);

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
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);
    const shape = &[_]usize{ 2, 2 };

    var t1 = try Tensor.init(&[_]T{ 1, 2, 3, 4 }, shape, true, device);
    var t2 = try Tensor.init(&[_]T{ 1, 0, 0, 1 }, shape, true, device);

    // Case 1: No transpose
    var t3 = try t1.bmm(t2, .{ .trans_a = false, .trans_b = false });
    defer t3.deinit();

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    t1.acquire();
    t2.acquire();
    t3.grad.?.fill(1.0, device);

    try gm.backward(t3);
    const expected_grad_t1 = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2 = &[_]T{ 4, 4, 6, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
    t1.grad.?.fill(0, device);
    t2.grad.?.fill(0, device);

    // Case 2: Transpose A
    var t3_trans_a = try t1.bmm(t2, .{ .trans_a = true, .trans_b = false });
    defer t3_trans_a.deinit();
    t3_trans_a.grad.?.fill(1.0, device);

    try gm.backward(t3_trans_a);
    const expected_grad_t1_trans_a = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_a = &[_]T{ 3, 3, 7, 7 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_a, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_a, t2.grad.?.data);
    t1.grad.?.fill(0, device);
    t2.grad.?.fill(0, device);

    // Case 3: Transpose B
    var t3_trans_b = try t1.bmm(t2, .{ .trans_a = false, .trans_b = true });
    defer t3_trans_b.deinit();
    t3_trans_b.grad.?.fill(1.0, device);
    try gm.backward(t3_trans_b);

    const expected_grad_t1_trans_b = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_b = &[_]T{ 4, 6, 4, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_b, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_b, t2.grad.?.data);
    t1.grad.?.fill(0, device);
    t2.grad.?.fill(0, device);

    // Case 4: Transpose both A and B
    var t3_trans_ab = try t1.bmm(t2, .{ .trans_a = true, .trans_b = true });
    defer t3_trans_ab.deinit();
    t3_trans_ab.grad.?.fill(1.0, device);
    try gm.backward(t3_trans_ab);

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
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

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
    const t1 = try Tensor.init(&.{ 1, 2, 3, 4 }, shape_mat, true, device);
    const t2 = try Tensor.init(&.{ 1, 1 }, shape_vec, true, device);
    const t3 = try t1.matvec(t2);

    try t1.setup_grad(0.0);
    try t2.setup_grad(0.0);

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();

    try t3.setup_grad(1.0);
    try gm.backward(t3);

    const expected_grad_t1 = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2 = &[_]T{ 4, 6 };

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}

test "tensor/GraphManager/dot_backward" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);
    const shape = &[_]usize{3};

    var t1 = try Tensor.init(&[_]T{ 1, 2, 3 }, shape, true, device);
    defer t1.deinit();

    const t2 = try Tensor.init(&[_]T{ 4, 5, 6 }, shape, true, device);
    defer t2.deinit();

    var t3 = try t1.dot(t2);
    defer t3.deinit();

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
    t3.grad.?.fill(1.0, device);
    try gm.backward(t3);

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
    try std.testing.expectEqualSlices(usize, &index_shape, output.data.shape.slice());

    // case 2: grad check
    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();

    output.grad.?.fill(1.0, device);
    try gm.backward(output);

    const expected_grad = [_]T{ 1, 1, 0, 0, 1, 1, 1, 0, 1 };
    try std.testing.expectEqualSlices(T, &expected_grad, input.grad.?.data);

    // case 3: out of bounds
    //try index.set(&.{ 0, 0 }, 3);
    //try std.testing.expectError(error.IndexOutOfBounds, input.gather(device, .{ .indices = index, .dim = 1 }));
}

// TODO: Fix memory freeing conundrum with max_over_dim() then dont use an arena here.
//test "tensor/max_over_dim" {
//    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//    defer arena.deinit();
//    const allocator = arena.allocator();
//
//    var cpu = zg.device.HostDevice.init(allocator);
//    defer cpu.deinit();
//    const device = cpu.reference();
//
//    const T = f32;
//    const Tensor = NDTensor(T);
//
//    // case 1: basic max over dim operation
//    const input_data = [_]T{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
//    const input_shape = [_]usize{ 3, 3 };
//    var input = try Tensor.init(&input_data, &input_shape, true, device);
//    defer input.deinit();
//
//    var output = try input.max_over_dim(device, .{ .dim = 1 });
//    defer output.deinit();
//
//    try std.testing.expectEqualSlices(T, &[_]T{ 3, 6, 9 }, output.data.data);
//    try std.testing.expectEqualSlices(usize, &[_]usize{3}, output.data.shape.shape);
//
//    // case 2: gradient check
//    var gm = GraphManager(Tensor).init(device.allocator, .{});
//    defer gm.deinit();
//
//    output.grad.?.fill(1.0, device);
//    try gm.backward(output);
//
//    const expected_grad = [_]T{ 0, 0, 1, 0, 0, 1, 0, 0, 1 };
//    try std.testing.expectEqualSlices(T, &expected_grad, input.grad.?.data);
//
//    // case 3: max over different dimension
//    var output2 = try input.max_over_dim(device, .{ .dim = 0 });
//    defer output2.deinit();
//
//    try std.testing.expectEqualSlices(T, &[_]T{ 7, 8, 9 }, output2.data.data);
//    try std.testing.expectEqualSlices(usize, &[_]usize{3}, output2.data.shape.shape);
//
//    // reset grads
//    input.grad.?.fill(0, device);
//    output2.grad.?.fill(1.0, device);
//    try gm.backward(output2);
//
//    const expected_grad2 = [_]T{ 0, 0, 0, 0, 0, 0, 1, 1, 1 };
//    try std.testing.expectEqualSlices(T, &expected_grad2, input.grad.?.data);
//
//    // case 4: invalid dimension
//    try std.testing.expectError(error.DimOutOfBounds, input.max_over_dim(device, .{ .dim = 2 }));
//}
