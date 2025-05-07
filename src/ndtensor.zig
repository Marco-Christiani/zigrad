const std = @import("std");
const zg = @import("zigrad.zig");
const settings = zg.settings;
const DeviceReference = zg.DeviceReference;
const backend = zg.backend;
const opspec = zg.opspec;

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
    CLAMP,

    fn matmul_tag(trans_a: bool, trans_b: bool) Op {
        return if (!trans_a and !trans_b)
            .MATMUL_AB
        else if (trans_a and !trans_b)
            .MATMUL_AtB
        else if (!trans_a and trans_b)
            .MATMUL_ABt
        else
            .MATMUL_AtBt;
    }
};

pub fn NDTensor(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const DataType = NDArray(T);
        pub const Label = std.BoundedArray(u8, 32);
        pub const BackwardsContext = @import("backward_context.zig").BackwardContext(Self);
        pub const Children = BackwardsContext.Children;

        /// Core NDArray that holds the values and shape.
        /// Use this member directly when you want to perform
        /// ops that will not be tracked by the graph.
        data: DataType,
        /// The gradient calculated by calling "backward".
        /// Gradients are lazily initialized.
        grad: ?DataType = null,
        /// The device field is a reference to a stateful
        /// object that provides memory and compute resources.
        device: DeviceReference,
        /// Marking a tensor as acquired signals to the
        /// backwards process that this tensor should
        /// not be freed. Set by using the "acquire" and
        /// "release" functions.
        acquired: bool = false,
        /// An attached tensor can be traversed through
        /// in the backward process. If the tensor is
        /// unattached, the reversal process will not
        /// continue through that tensor. Set by using
        /// the "attach" and "detach" functions.
        attached: bool = true,
        /// Optional label for naming tensors. This is
        /// useful for printing graphs and diagrams.
        label: Label = .{},
        /// Optional op tag - TODO: do we need to
        /// continue to support this?
        op: ?Op = null,
        /// Opaque object that acts like a closure for
        /// backwards function calls. The backwards context
        /// can allocate state if the arguments provided
        /// exceed its internal buffer.
        _backward_ctx: ?BackwardsContext = null,
        /// The requires grad field tells the backwards
        /// process if it ought to initialize a gradient.
        /// This field should not be used directly
        /// because runtime gradients may be deactivated.
        /// Use the "requires_grad" function instead.
        _requires_grad: bool,
        /// versioning ensures that inplace ops do not
        /// interfere with the backward pass. Versioning
        /// is only enabled in debug mode and only
        /// matters if you are doing a backward pass.
        _version: u8 = 0,

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
                .device = device,
                .acquired = false,
                ._requires_grad = _requires_grad,
                ._backward_ctx = null,
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

        pub fn ones(shape: []const usize, _requires_grad: bool, device: DeviceReference) !*Self {
            const self = try Self.empty(shape, _requires_grad, device);
            self.fill(1);
            return self;
        }

        pub fn random(shape: []const usize, _requires_grad: bool, op: zg.RandType, device: DeviceReference) !*Self {
            const self = try Self.empty(shape, _requires_grad, device);
            device.mem_random(T, self.get_data(), op, zg.random);
            return self;
        }

        pub fn sequence(start: T, step: T, shape: []const usize, _requires_grad: bool, device: DeviceReference) !*Self {
            const self = try Self.empty(shape, _requires_grad, device);
            device.mem_sequence(T, self.get_data(), start, step);
            return self;
        }

        pub fn attach(self: *Self) void {
            self.attached = true;
        }

        pub fn detach(self: *Self) void {
            self.attached = false;
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

        pub fn get_dim(self: *const Self, i: usize) usize {
            return self.data.shape.get(i);
        }

        pub fn cast(self: *Self, K: type) !*NDTensor(K) {
            _ = self;
            @compileError("Not implemented");
        }

        pub fn child_iterator(self: *Self) ?BackwardsContext.ChildIterator {
            const ctx = &(self._backward_ctx orelse return null);
            if (ctx.children.len == 0 and ctx.next == null) return null;
            return .{ .node = ctx };
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
                self.grad.?.mode = self.data.mode;
            }
            return self.assume_grad().fill(fill_value orelse return, self.device);
        }

        pub fn assume_grad(self: *Self) *DataType {
            if (self.grad) |*grd| {
                return grd;
            } else {
                @branchHint(.cold);
                @panic("no gradient");
            }
        }

        pub fn assume_grad_data(self: *Self) []T {
            return self.assume_grad().data;
        }

        // This function can allocate a gradient if one is not present.
        pub fn ensure_grad(self: *Self, fill_value: ?T) !*DataType {
            if (self.grad) |*grd| {
                return grd;
            } else {
                try self.setup_grad(fill_value);
                return self.assume_grad();
            }
        }

        // This function can allocate a gradient if one is not present.
        pub fn ensure_grad_data(self: *Self, fill_value: ?T) ![]T {
            const grd = try self.ensure_grad(fill_value);
            return grd.data;
        }

        pub fn get_label(self: *const Self) ?[]const u8 {
            return if (self.label.len > 0) self.label.slice() else null;
        }

        pub fn set_label(self: *Self, new_label: []const u8) !void {
            self.label = try Label.fromSlice(new_label);
        }
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
        pub fn create_dependent(BwdCallback: type, opts: struct {
            data: DataType,
            children: []const *Self,
            callback: BwdCallback,
            device: DeviceReference,
            label: ?[]const u8 = null,
            op: ?Op = null,
        }) !*Self {
            const rg: bool = for (opts.children) |child| {
                if (child.requires_grad()) break true;
            } else false;

            const self = try opts.device.allocator.create(Self);
            errdefer opts.device.allocator.destroy(self);

            self.* = Self{
                .data = opts.data,
                .op = opts.op,
                .label = try Label.fromSlice(opts.label orelse ""),
                .acquired = false,
                .device = opts.device,
                ._requires_grad = rg,
                ._backward_ctx = if (rg) try BackwardsContext.init(
                    BwdCallback,
                    opts.callback,
                    opts.device,
                    .{
                        .children = opts.children,
                        .persist = false,
                    },
                ) else null,
            };
            return self;
        }

        pub fn prepend_dependent(BwdCallback: type, self: *Self, opts: struct {
            children: []const *Self,
            callback: BwdCallback,
            device: DeviceReference,
        }) !void {
            const rg: bool = for (opts.children) |child| {
                if (child.requires_grad()) break true;
            } else self.requires_grad();

            if (rg) {
                const ctx = try BackwardsContext.init(
                    BwdCallback,
                    opts.callback,
                    opts.device,
                    .{
                        .children = opts.children,
                        .persist = false,
                    },
                );
                if (self._backward_ctx) |*_ctx| {
                    try _ctx.prepend(ctx, self.device);
                } else {
                    self._backward_ctx = ctx;
                }
                // I'm using wrapping add on the off-chance that someone
                // overflows the byte - it would still be a mismatch on
                // the backward pass whereas saturation wouldn't.
                // Since this is effectively a modulus, the user could
                // setup an extremely odd situation (with more than 256 inplace
                // ops on a single tensor) that would cause a false pass,
                // but I don't see the need to address such edge cases.
                self._version +%= 1;
            }
        }

        pub fn deinit(self: *Self) void {
            if (self.acquired)
                std.debug.panic("Attempt to deinit an acquired tensor.", .{});

            self.data.deinit(self.device);

            if (self.grad) |*g| {
                g.deinit(self.device);
            }

            if (self._backward_ctx) |*ctx| {
                ctx.deinit(self.device);
            }

            self.device.allocator.destroy(self);
        }

        pub fn teardown(self: *Self, gm: *GraphManager(Self)) void {
            gm.reset();
            gm.topological_sort(self);

            var iter = gm.sorted_nodes.iterator();

            while (iter.next()) |entry| {
                const node = entry.key_ptr.*;
                if (node.acquired) continue;
                node.deinit();
            }

            gm.reset();
        }

        pub fn acquire(self: *Self) void {
            self.acquired = true;
        }

        pub fn release(self: *Self) void {
            self.acquired = false;
        }

        fn to_device_impl(
            src: []const T,
            dst: []T,
            src_device: DeviceReference,
            dst_device: DeviceReference,
        ) !void {
            if (comptime @typeInfo(@TypeOf(src_device)) != .@"struct")
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

            const ToDeviceBwd = struct {
                fn callback(y: *Self) !void {
                    const x = y.backward_child(0) orelse return;
                    try to_device_impl(
                        y.assume_grad_data(),
                        x.ensure_grad_data(0),
                        y.device,
                        x.device,
                    );
                }
            };

            const data = try device.mem_alloc(T, self.data.data.len);
            try to_device_impl(self.data.data, data, self.device, device);

            var result = try create_dependent(ToDeviceBwd, .{
                .data = .{
                    .data = data,
                    .shape = self.data.shape,
                },
                .children = &.{self},
                .device = device,
                .callback = .{},
                .op = .TRANSFER,
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
                .label = .{},
                .acquired = false,
                .device = self.device,
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
        pub fn _reshape(self: *Self, shape: []const usize) void {
            self.data._reshape(shape);
            if (self.grad) |*g| g._reshape(shape);
        }

        /// Copies. COM.
        pub fn reshape(self: *Self, new_shape: []const usize) !*Self {
            const ReshapeBwd = struct {
                fn callback(y: *Self) !void {
                    const x = y.backward_child(0) orelse return;
                    const x_grad = try x.ensure_grad(0);
                    y.assume_grad()._reshape(x.data.shape.slice());
                    try x_grad._add(y.assume_grad().*);
                }
            };

            return try create_dependent(ReshapeBwd, .{
                .data = .{
                    .data = try self.data.copy(self.device),
                    .shape = self.data.shape.reshape(new_shape),
                    .view = false,
                },
                .children = &.{self},
                .device = self.device,
                .callback = .{},
                .op = .RESHAPE,
            });
        }

        /// Copies. COM.
        pub fn transpose(self: *Self) !*Self {
            const TransposeBwd = struct {
                pub fn callback(y: *Self, children: *Children) !void {
                    const x = children.get_bwd(0) orelse return;
                    y.device.dispatch(opspec.transpose(T){
                        .A = y.assume_grad_data(),
                        .B = try x.ensure_grad_data(0),
                        .m = self.shape.get(0),
                        .n = self.shape.get(1),
                        .alpha = 1.0,
                    });
                }
            };

            return create_dependent(TransposeBwd, .{
                .data = try self.data.transpose(self.device),
                .children = &.{self},
                .device = self.device,
                .context = .{},
                .op = .TRANSPOSE,
            });
        }

        pub fn fill(self: *const Self, val: T) void {
            self.data.fill(val, self.device);
        }

        // this needs to get audited for device safety
        pub fn get(self: *const Self, idx: usize) T {
            return self.data.data[idx];
        }

        // this needs to get audited for device safety
        pub fn set(self: *const Self, idx: usize, value: T) void {
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
        pub fn _clamp_grad(self: *const Self, vmin: T, vmax: T) !void {
            (self.grad orelse return error.NoGradient)._clamp(vmin, vmax, self.device);
        }

        /// Differentiable
        pub fn clamp(self: *Self, vmin: T, vmax: T) !*Self {
            std.debug.assert(vmin <= vmax);

            const ClampBwd = struct {
                _min: T,
                _max: T,
                pub fn callback(y: *Self, children: *Children, ctx: *@This()) !void {
                    const x = children.get_bwd(0) orelse return;
                    y.device.dispatch(opspec.clamp_bwd(T){
                        .x = x.get_data(),
                        .x_g = try x.ensure_grad_data(0),
                        .y_g = y.assume_grad_data(),
                        .min = ctx._min,
                        .max = ctx._max,
                    });
                }
            };

            return create_dependent(ClampBwd, .{
                .data = try self.data.clamp(vmin, vmax, self.device),
                .children = &.{self},
                .device = self.device,
                .callback = .{ ._min = vmin, ._max = vmax },
                .op = .CLAMP,
            });
        }

        pub fn add_scalar(self: *Self, s: T) !*Self {
            const AddBwd = struct {
                pub fn callback(c: *Self, children: *Children) !void {
                    const a = children.get_bwd(0) orelse return;
                    try c.assume_grad().unbroadcast_(try a.ensure_grad(0), c.device, .{ .alpha = 1.0, .beta = 1.0 });
                }
            };
            return create_dependent(AddBwd, .{
                .data = try self.data.add_scalar(s, self.device),
                .children = &.{self},
                .device = self.device,
                .callback = .{},
                .op = .ADD,
            });
        }

        pub fn sub_scalar(self: *Self, s: T) !*Self {
            return self.add_scalar(-s);
        }

        /// Element-wise addition. COM.
        pub fn add(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const AddBwd = struct {
                pub fn callback(c: *Self, children: *Children) !void {
                    scope: {
                        const a = children.get_bwd(0) orelse break :scope;
                        try c.assume_grad().unbroadcast_(try a.ensure_grad(0), c.device, .{ .alpha = 1.0, .beta = 1.0 });
                    }
                    scope: {
                        const b = children.get_bwd(1) orelse break :scope;
                        try c.assume_grad().unbroadcast_(try b.ensure_grad(0), c.device, .{ .alpha = 1.0, .beta = 1.0 });
                    }
                }
            };

            return create_dependent(AddBwd, .{
                .data = try self.data.add(other.data, self.device),
                .children = &.{ self, other },
                .device = self.device,
                .callback = .{},
                .op = .ADD,
            });
        }

        pub fn add_(self: *Self, other: *Self) !void {
            std.debug.assert(self.device.is_compatible(other.device));

            const InplaceAddBwd = struct {
                pub fn callback(b: *Self, children: *Children) !void {
                    const a = children.get_bwd(0) orelse return;
                    try b.assume_grad().unbroadcast_(try a.ensure_grad(0), b.device, .{ .alpha = 1.0, .beta = 1.0 });
                }
            };

            try self.data.add_(&other.data, self.device);

            return prepend_dependent(InplaceAddBwd, other, .{
                .children = &.{self},
                .device = self.device,
                .callback = .{},
            });
        }

        pub fn _add(self: *Self, other: *Self) !void {
            return other.add_(self);
        }

        /// Element-wise subtraction. COM.
        pub fn sub(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const SubBwd = struct {
                pub fn callback(c: *Self, children: *Children) !void {
                    scope: {
                        const a = children.get_bwd(0) orelse break :scope;
                        try c.assume_grad().unbroadcast_(try a.ensure_grad(0), c.device, .{ .alpha = 1.0, .beta = 1.0 });
                    }
                    scope: {
                        const b = children.get_bwd(1) orelse break :scope;
                        try c.assume_grad().unbroadcast_(try b.ensure_grad(0), c.device, .{ .alpha = -1.0, .beta = 1.0 });
                    }
                }
            };

            return create_dependent(SubBwd, .{
                .data = try self.data.sub(other.data, self.device),
                .children = &.{ self, other },
                .device = self.device,
                .callback = .{},
                .op = .SUB,
            });
        }

        /// Element-wise multiplication. COM.
        pub fn mul(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const MulBwd = struct {
                pub fn callback(c: *Self, children: *Children) !void {
                    scope: {
                        const a = children.get_bwd(0) orelse break :scope;
                        const b = children.get(1);

                        var bc_grad = try b.data.mul(c.assume_grad().*, c.device);
                        defer bc_grad.deinit(c.device);

                        try bc_grad.unbroadcast_(try a.ensure_grad(0), c.device, .{ .alpha = 1.0, .beta = 1.0 });
                    }
                    scope: {
                        const b = children.get_bwd(1) orelse break :scope;
                        const a = children.get(0);

                        var ac_grad = try a.data.mul(c.assume_grad().*, c.device);
                        defer ac_grad.deinit(c.device);

                        try ac_grad.unbroadcast_(try b.ensure_grad(0), c.device, .{ .alpha = 1.0, .beta = 1.0 });
                    }
                }
            };

            return create_dependent(MulBwd, .{
                .data = try self.data.mul(other.data, self.device),
                .children = &.{ self, other },
                .device = self.device,
                .callback = .{},
                .op = .MUL,
            });
        }

        /// Element-wise division. COM.
        pub fn div(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const DivBwd = struct {
                pub fn callback(c: *Self, children: *Children) !void {
                    scope: {
                        const a = children.get_bwd(0) orelse break :scope;
                        const b = children.get(1);

                        var bc_grad = try c.assume_grad().div(b.data, c.device);
                        defer bc_grad.deinit(c.device);

                        try bc_grad.unbroadcast_(try a.ensure_grad(0), c.device, .{ .alpha = 1.0, .beta = 1.0 });
                    }
                    scope: {
                        const b = children.get_bwd(1) orelse break :scope;
                        const a = children.get(0);

                        var ac_grad = blk: {
                            var b_grad_value = try c.assume_grad().mul(a.data, c.device);
                            defer b_grad_value.deinit(c.device);
                            var bsq = try b.data.mul(b.data, c.device);
                            defer bsq.deinit(c.device);
                            break :blk try b_grad_value.div(bsq, c.device);
                        };
                        defer ac_grad.deinit(c.device);

                        try ac_grad.unbroadcast_(try b.ensure_grad(0), c.device, .{ .alpha = -1.0, .beta = 1.0 });
                    }
                }
            };

            return create_dependent(DivBwd, .{
                .data = try self.data.div(other.data, self.device),
                .children = &.{ self, other },
                .device = self.device,
                .callback = .{},
                .op = .DIV,
            });
        }

        /// Computes the maximum value of the tensor. Returns a scalar tensor. COM.
        pub fn max(self: *Self) !*Self {
            const MaxBwd = struct {
                pub fn callback(y: *Self, children: *Children) !void {
                    const x = children.get_bwd(0) orelse return;
                    y.device.dispatch(opspec.max_bwd(T){
                        .x = x.get_data(),
                        .x_g = try x.ensure_grad_data(0),
                        .y = y.get_data(),
                        .y_g = y.assume_grad_data(),
                    });
                }
            };

            return create_dependent(MaxBwd, .{
                .data = try self.data.max(self.device),
                .children = &.{self},
                .device = self.device,
                .callback = .{},
                .op = .MAX,
            });
        }

        /// TODO: exp bw backend
        /// Element-wise exponential. COM.
        pub fn exp(self: *Self) !*Self {
            const ExpBwd = struct {
                pub fn callback(y: *Self, children: *Children) !void {
                    const x = children.get_bwd(0) orelse return;
                    y.device.dispatch(opspec.exp_bwd(T){
                        .x_g = try x.ensure_grad_data(0),
                        .y = y.get_data(),
                        .y_g = y.assume_grad_data(),
                    });
                }
            };
            return create_dependent(ExpBwd, .{
                .data = try self.data.exp(self.device),
                .children = &.{self},
                .device = self.device,
                .callback = .{},
                .op = .EXP,
            });
        }

        const BmmConfig = struct {
            trans_a: bool = false,
            trans_b: bool = false,
        };

        /// TODO: this should proxy to bmm_acc
        /// Matrix multiplication. COM.
        pub fn bmm(self: *Self, other: *Self, config: BmmConfig) !*Self {
            return create_dependent(BmmAccBwd, .{
                .data = try self.data.bmm(other.data, self.device, .{
                    .trans_a = config.trans_a,
                    .trans_b = config.trans_b,
                    .alpha = 1.0,
                    .beta = 0.0,
                }),
                .children = &.{ self, other },
                .device = self.device,
                .callback = .{},
                .op = Op.matmul_tag(config.trans_a, config.trans_b),
            });
        }

        pub fn bmm_acc(
            self: *Self,
            other: *Self,
            // I'm passing usize here because its more likely that
            // the user wants slices on the frontend instead of Shape
            out_shape: []const usize,
            config: BmmConfig,
        ) !*Self {
            return create_dependent(BmmAccBwd, .{
                .data = try self.data.bmm_acc(other.data, Shape.init(out_shape), self.device, .{
                    .trans_a = config.trans_a,
                    .trans_b = config.trans_b,
                    .alpha = 1.0,
                    .beta = 0.0,
                }),
                .children = &.{ self, other },
                .device = self.device,
                .callback = .{},
                .op = Op.matmul_tag(config.trans_a, config.trans_b),
            });
        }

        pub fn bmm_acc_(self: *Self, other: *Self, out: *Self, config: BmmConfig) !*Self {
            try self.data.bmm_acc_(other.data, &out.data, self.device, .{
                .trans_a = config.trans_a,
                .trans_b = config.trans_b,
                .alpha = 1.0,
                .beta = 0.0,
            });

            return create_dependent(BmmAccBwd, .{
                .data = out.data,
                .grad = out.grad,
                .children = &.{ self, other },
                .device = self.device,
                .callback = .{},
                .op = Op.matmul_tag(config.trans_a, config.trans_b),
            });
        }

        const BmmAccBwd = struct {
            pub fn callback(C: *Self, children: *Children) !void {
                const op_tag = C.op orelse unreachable;
                const A = children.get(0);
                const B = children.get(1);

                if (children.get_bwd(0)) |_| {
                    const C_grad = C.assume_grad().*;
                    switch (op_tag) {
                        .MATMUL_AB => {
                            try C_grad.bmm_acc_(B.data, try A.ensure_grad(0), C.device, .{ .trans_a = false, .trans_b = true, .beta = 1.0 });
                        },
                        .MATMUL_AtB => {
                            try B.data.bmm_acc_(C_grad, try A.ensure_grad(0), C.device, .{ .trans_a = false, .trans_b = true, .beta = 1.0 });
                        },
                        .MATMUL_ABt => {
                            try C_grad.bmm_acc_(B.data, try A.ensure_grad(0), C.device, .{ .trans_a = false, .trans_b = false, .beta = 1.0 });
                        },
                        .MATMUL_AtBt => {
                            try B.data.bmm_acc_(C_grad, try A.ensure_grad(0), C.device, .{ .trans_a = true, .trans_b = true, .beta = 1.0 });
                        },
                        else => unreachable,
                    }
                }

                if (children.get_bwd(1)) |_| {
                    const C_grad = C.assume_grad().*;
                    switch (op_tag) {
                        .MATMUL_AB => {
                            try A.data.bmm_acc_(C_grad, try B.ensure_grad(0), C.device, .{ .trans_a = true, .trans_b = false, .beta = 1.0 });
                        },
                        .MATMUL_AtB => {
                            try A.data.bmm_acc_(C_grad, try B.ensure_grad(0), C.device, .{ .trans_a = false, .trans_b = false, .beta = 1.0 });
                        },
                        .MATMUL_ABt => {
                            try C_grad.bmm_acc_(A.data, try B.ensure_grad(0), C.device, .{ .trans_a = true, .trans_b = false, .beta = 1.0 });
                        },
                        .MATMUL_AtBt => {
                            try C_grad.bmm_acc_(A.data, try B.ensure_grad(0), C.device, .{ .trans_a = false, .trans_b = true, .beta = 1.0 });
                        },
                        else => unreachable,
                    }
                }
            }
        };

        /// Dot product of two tensors. COM.
        pub fn dot(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const DotBwd = struct {
                pub fn callback(c: *Self, children: *Children) !void {
                    scope: {
                        const a = children.get_bwd(0) orelse break :scope;
                        c.device.dispatch(opspec.axpy(T){
                            .x = children.get(1).get_data(),
                            .y = try a.ensure_grad_data(0),
                            .alpha = @ptrCast(c.assume_grad_data().ptr),
                        });
                    }
                    scope: {
                        const b = children.get_bwd(1) orelse break :scope;
                        c.device.dispatch(opspec.axpy(T){
                            .x = children.get(0).get_data(),
                            .y = try b.ensure_grad_data(0),
                            .alpha = @ptrCast(c.assume_grad_data().ptr),
                        });
                    }
                }
            };

            return create_dependent(DotBwd, .{
                .data = try self.data.dot(other.data, self.device),
                .children = &.{ self, other },
                .device = self.device,
                .callback = .{},
                .op = .DOT,
            });
        }

        const MatvecConfig = struct {
            trans_a: bool = false,
        };

        pub fn matvec_(A: *Self, x: *Self, y: *Self, config: struct {
            trans_a: bool = false,
            beta: T = 0.0,
        }) !void {
            std.debug.assert(A.device.is_compatible(x.device));
            std.debug.assert(A.device.is_compatible(y.device));
            const MatvecBwd = struct {
                _trans_a: bool,
                pub fn callback(_y: *Self, children: *Children, ctx: *@This()) !void {
                    const ta = ctx._trans_a;
                    scope: {
                        const _A = children.get_bwd(0) orelse break :scope;
                        const _x = children.get(1);
                        _y.device.dispatch(opspec.outer(T){
                            .x = if (ta) _x.get_data() else _y.assume_grad_data(),
                            .y = if (ta) _y.assume_grad_data() else _x.get_data(),
                            .A = try _A.ensure_grad_data(0),
                            .alpha = 1.0,
                        });
                    }
                    scope: {
                        const _x = children.get_bwd(1) orelse break :scope;
                        const _A = children.get(0);
                        _y.device.dispatch(opspec.matvec(T){
                            .A = _A.get_data(),
                            .x = _y.assume_grad_data(),
                            .y = try _x.ensure_grad_data(0),
                            .m = if (!ta) _A.get_dim(1) else _A.get_dim(0),
                            .n = if (!ta) _A.get_dim(0) else _A.get_dim(1),
                            .trans_a = !ta,
                            .alpha = 1.0,
                            .beta = 1.0,
                        });
                    }
                }
            };

            A.data.matvec_(x.data, &y.data, A.device, .{
                .trans_a = config.trans_a,
                .alpha = 1.0,
                .beta = config.beta,
            });

            return prepend_dependent(MatvecBwd, y, .{
                .children = &.{ A, x },
                .device = A.device,
                .callback = .{ ._trans_a = config.trans_a },
            });
        }

        pub fn matvec(self: *Self, other: *Self, config: MatvecConfig) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const MatvecBwd = struct {
                _trans_a: bool,
                pub fn callback(y: *Self, children: *Children, ctx: *@This()) !void {
                    const ta = ctx._trans_a;
                    scope: {
                        const A = children.get_bwd(0) orelse break :scope;
                        const x = children.get(1);
                        y.device.dispatch(opspec.outer(T){
                            .x = if (ta) x.get_data() else y.assume_grad_data(),
                            .y = if (ta) y.assume_grad_data() else x.get_data(),
                            .A = try A.ensure_grad_data(0),
                            .alpha = 1.0,
                        });
                    }
                    scope: {
                        const x = children.get_bwd(1) orelse break :scope;
                        const A = children.get(0);
                        y.device.dispatch(opspec.matvec(T){
                            .A = A.get_data(),
                            .x = y.assume_grad_data(),
                            .y = try x.ensure_grad_data(0),
                            .m = if (!ta) A.get_dim(1) else A.get_dim(0),
                            .n = if (!ta) A.get_dim(0) else A.get_dim(1),
                            .trans_a = !ta,
                            .alpha = 1.0,
                            .beta = 1.0,
                        });
                    }
                }
            };

            return create_dependent(MatvecBwd, .{
                .data = try self.data.matvec(other.data, self.device, .{
                    .trans_a = config.trans_a,
                    .alpha = 1.0,
                    .beta = 0.0,
                }),
                .children = &.{ self, other },
                .device = self.device,
                .callback = .{ ._trans_a = config.trans_a },
                .op = .MATVEC,
            });
        }

        /// Sum of all elements in the tensor. COM.
        pub fn sum(self: *Self) !*Self {
            const SumBwd = struct {
                pub fn callback(y: *Self, children: *Children) !void {
                    const x = children.get_bwd(0) orelse return;
                    const x_grad = try x.ensure_grad(0);
                    try x_grad._add(y.assume_grad().*, y.device);
                }
            };
            return create_dependent(SumBwd, .{
                .data = try self.data.sum(self.device),
                .children = &.{self},
                .device = self.device,
                .callback = .{},
                .op = .SUM,
            });
        }

        //pub fn max_along(self: *Self, device: DeviceReference, opts: MaxAlongOptions) !*Self {
        //    const max_backward = struct {
        //        // NOTE: See gather() comments, same apply here;;
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

        //pub fn gather(self: *Self, device: DeviceReference, opts: GatherOptions) !*Self {;;
        //    const gatherBackward = struct {;;
        //        fn bw_impl(bw_tensor: NDTensor(T)) !void {
        //            const bw_children = bw_tensor.get_children() orelse return error.NoChildren;
        //            const bw_input = bw_children[0];
        //            if (bw_input.grad == null) return;
        //            const offsets: [*]usize = @ptrCast(@alignCast(bw_tensor._backward_ctx orelse return error.NoBackwardContext));
        //            // how am i expected to free this, unknown len
        //            // defer _self.device.raw(offsets); // FIXME: just occuring to me the problem with this, if _self.device != fwd_allocator
        //            // bw_tensor must/should be the same len as indices used to index (note that offsets is a raw c ptr without a len)
        //            // std.debug.assert(offsets.len == bw_tensor.data.data.len); // can make this a real check when its a  null term alloc
        //            for (0..bw_tensor.data.data.len) |i| bw_input.grad.?.data[offsets[i]] += bw_tensor.grad.?.data[i];
        //        }
        //    }.bw_impl;

        //    const gather_result = try self.data.gather(device, .{ .indices = opts.indices.data, .dim = opts.dim, .return_offsets = true });;;
        //    // TODO: use a null terminated allocation instead, tired rn
        //    const ctx = if (self.requires_grad()) try device.allocator.dupe(usize, gather_result.offsets.?) else null;;;

        //    return create_dependent(.{
        //        .data = gather_result.values,;;
        //        .op = null,
        //        .children = &.{self},
        //        ._requires_grad = self._requires_grad,
        //        .device = device,
        //        ._backward = gatherBackward,;;
        //        ._backward_ctx = if (ctx) |c| c.ptr else null,
        //    });
        //}

        pub fn backward(self: *Self) !void {
            std.debug.assert(zg.rt_grad_enabled);
            // TODO: In the future, make sure this works with
            // quantization. The 1 element is a tensor's gradient
            // with respect to itself.
            _ = try self.ensure_grad(1);
            if (self._backward_ctx) |*ctx| {
                try ctx.call(self);
            }
        }

        /// Prints dynamic compuation graph in d2 format with ops as and operands as nodes
        pub fn print_arrows(self: *Self) void {
            var children = self.child_iterator() orelse return;
            while (children.next()) |elem| {
                std.debug.print("{?s}<-{?s}", .{ self.get_label(), elem.get_label() });
                const symbol = blk: {
                    const op = self.op orelse break :blk ": ?";
                    switch (op) {
                        Op.ADD => break :blk ": +",
                        Op.SUB => break :blk ": -",
                        Op.MUL => break :blk ": x",
                        Op.DIV => break :blk ": /",
                        Op.SUM => break :blk ": ++",
                        Op.MATVEC => break :blk ": Ax",
                        Op.MATMUL_AB,
                        Op.MATMUL_AtB,
                        Op.MATMUL_ABt,
                        Op.MATMUL_AtBt,
                        => break :blk ": AB",
                        else => std.debug.panic("Unsupported op {?}\n", .{self.op}),
                    }
                };
                std.debug.print("{?s}\n", .{symbol});
            }
            var next_children = self.child_iterator() orelse return;
            while (next_children.next()) |elem| {
                elem.print_arrows();
            }
        }
    };
}

test "ndtensor/clamp fw,bw,_clamp,_clamp_grad" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var cpu = zg.device.HostDevice.init(arena.allocator());
    defer cpu.deinit();

    const device = cpu.reference();

    const T = f32;
    const Tensor = NDTensor(T);

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();

    var x = try Tensor.init(&.{ -2.0, -0.5, 0.5, 2.0 }, &.{ 2, 2 }, true, device);
    defer x.deinit();

    var y = try x.clamp(-1.0, 1.0);
    defer y.deinit();

    try y.setup_grad(1.0);

    try gm.backward(y);

    const expected_output: []const f32 = &.{ -1.0, -0.5, 0.5, 1.0 };
    const expected_grad: []const f32 = &.{ 0.0, 1.0, 1.0, 0.0 };

    try std.testing.expectEqualSlices(T, expected_output, y.get_data());
    try std.testing.expectEqualSlices(T, expected_grad, x.assume_grad_data());
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

    // Backward pass;
    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();

    if (!zg.rt_grad_enabled) return error.GradNotEnabled;

    try gm.backward(sum_result);

    const expected_grad = &[_]T{ 1, 1, 1, 1 };
    try std.testing.expectEqualSlices(T, expected_grad, input.assume_grad_data());
}

test "tensor/NDTensor index, add, div" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const Tensor = NDTensor(f32);

    {
        const shape = &[_]usize{ 2, 3 };
        var t1 = try Tensor.init(&[_]f32{ 1, 2, 3, 4, 5, 6 }, shape, false, device);
        defer t1.deinit();

        const t2 = try Tensor.init(&[_]f32{ 10, 20, 30, 40, 50, 60 }, shape, false, device);
        defer t2.deinit();

        const t3 = try t1.add(t2);
        defer t3.deinit();

        const t4 = try t3.sub(t1);
        defer t4.deinit();
    }

    {
        // zig fmt: off
        const t1 = try Tensor.init(&.{
             0, 1, 2,
             3, 4, 5,
             6, 7, 8,

             0, 1, 2,
             3, 4, 5,
             6, 7, 8
         }, &.{ 2, 3, 3 }, false, device);
        defer t1.deinit();

        const t2 = try Tensor.init(&.{ 1, 1, 1 }, null, true, device);
        defer t2.deinit();

        const t3 = try Tensor.init(&.{
             1, 2, 3,
             4, 5, 6,
             7, 8, 9,

             1, 2, 3,
             4, 5, 6,
             7, 8, 9,
         }, &.{ 2, 3, 3 }, false, device);
        defer t3.deinit();

        const t4 = try t1.add(t2);
        defer t4.deinit();

        try t4.backward();

        const t2_grad: [3]f32 = @splat(6);

        try std.testing.expectEqualSlices(f32, t3.get_data(), t4.get_data());
        try std.testing.expectEqualSlices(f32, t2.assume_grad_data(), &t2_grad);
    }
    {
        // zig fmt: off
        const t1 = try Tensor.init(&.{
             0, 1, 2,
             3, 4, 5,
             6, 7, 8,

             0, 1, 2,
             3, 4, 5,
             6, 7, 8
         }, &.{ 2, 3, 3 }, false, device);
        defer t1.deinit();

        const t2 = try Tensor.init(&.{ 1, 1, 1, 1, 1, 1 }, &.{ 2, 1, 3 }, true, device);
        defer t2.deinit();

        const t3 = try Tensor.init(&.{
             1, 2, 3,
             4, 5, 6,
             7, 8, 9,

             1, 2, 3,
             4, 5, 6,
             7, 8, 9,
         }, &.{ 2, 3, 3 }, false, device);
        defer t3.deinit();

        const t4 = try t1.add(t2);
        defer t4.deinit();

        try t4.backward();

        const t2_grad: [6]f32 = @splat(3);

        try std.testing.expectEqualSlices(f32, t3.get_data(), t4.get_data());
        try std.testing.expectEqualSlices(f32, t2.assume_grad_data(), &t2_grad);
    }
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

    var t1 = try Tensor.init(&.{2.0}, shape, true, device);
    defer t1.deinit();
    var t2 = try Tensor.init(&.{3.0}, shape, true, device);
    defer t2.deinit();
    // t3 = t1 + t2;
    // dt3/dt1 = 1, dt3/dt2 = 1
    var t3 = try t1.add(t2);
    defer t3.deinit();

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();
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

    const t1 = try Tensor.init(&[_]T{2}, shape, true, device);
    defer t1.deinit();

    const t2 = try Tensor.init(&[_]T{3}, shape, true, device);
    defer t2.deinit();
    // t3 = t1 * t2;
    // dt3/dt1 = t2, dt3/dt2 = t1
    const t3 = try t1.mul(t2);
    defer t3.deinit();

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();

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
    try w.set_label("W");
    defer w.deinit();

    var b = try Tensor.init(&[_]f32{ 1, 1 }, shape, true, device);
    try b.set_label("b");
    defer b.deinit();

    var x = try Tensor.init(&[_]f32{ 4, 4 }, shape, true, device);
    try x.set_label("x");
    defer x.deinit();

    // h = w*x + b
    // dh/dw = x, dh/db = 1
    const temp = try w.mul(x);
    try temp.set_label("temp");
    defer temp.deinit();

    const h = try temp.add(b);
    try h.set_label("h");
    defer h.deinit();

    var gm = GraphManager(Tensor).init(device.allocator, .{
        .eager_teardown = true,
    });
    defer gm.deinit();

    try gm.backward(h);

    try std.testing.expectEqualSlices(T, x.data.data, w.grad.?.data);
    try std.testing.expectEqualSlices(T, &[_]T{ 1.0, 1.0 }, b.grad.?.data);

    // 2 x 1
    const shape2 = &[_]usize{ 2, 1 };
    try w.setup_grad(0);
    try b.setup_grad(0);
    try x.setup_grad(0);
    w._reshape(shape2);
    b._reshape(shape2);
    x._reshape(shape2);
    //// h = w*x + b
    //// dh/dw = x, dh/db = 1
    const temp2 = try w.mul(x);
    defer temp2.deinit();
    const h2 = try temp2.add(b);
    defer h2.deinit();

    var gm2 = GraphManager(Tensor).init(device.allocator, .{});
    defer gm2.deinit();

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
    defer t1.deinit();

    var t2 = try Tensor.init(&[_]T{ 1, 0, 0, 1 }, shape, true, device);
    defer t2.deinit();

    // Case 1: No transpose
    var t3 = try t1.bmm(t2, .{ .trans_a = false, .trans_b = false });
    defer t3.deinit();

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();

    try gm.backward(t3);
    const expected_grad_t1 = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2 = &[_]T{ 4, 4, 6, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
    try t1.setup_grad(0);
    try t2.setup_grad(0);

    // Case 2: Transpose A
    var t3_trans_a = try t1.bmm(t2, .{ .trans_a = true, .trans_b = false });
    defer t3_trans_a.deinit();

    try t3_trans_a.setup_grad(1.0);

    try gm.backward(t3_trans_a);
    const expected_grad_t1_trans_a = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_a = &[_]T{ 3, 3, 7, 7 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_a, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_a, t2.grad.?.data);
    try t1.setup_grad(0);
    try t2.setup_grad(0);

    // Case 3: Transpose B
    var t3_trans_b = try t1.bmm(t2, .{ .trans_a = false, .trans_b = true });
    defer t3_trans_b.deinit();

    try gm.backward(t3_trans_b);

    const expected_grad_t1_trans_b = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_b = &[_]T{ 4, 6, 4, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_b, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_b, t2.grad.?.data);
    try t1.setup_grad(0);
    try t2.setup_grad(0);

    // Case 4: Transpose both A and B
    var t3_trans_ab = try t1.bmm(t2, .{ .trans_a = true, .trans_b = true });
    defer t3_trans_ab.deinit();
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
        try gm.backward(t3);

        const expected_grad_t1 = &[_]T{ 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2 };
        const expected_grad_t2 = &[_]T{ 5, 5, 7, 7, 9, 9, 17, 17, 19, 19, 21, 21 };
        try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
        try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);

        try t1.setup_grad(0);
        try t2.setup_grad(0);
    }

    // Case 2: Transpose A (t1: [2, 3, 2], t2: [2, 3, 2])
    {
        var t1_case2 = try Tensor.init(&[_]T{ 1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12 }, &[_]usize{ 2, 3, 2 }, true, device);
        var t3 = try t1_case2.bmm(t2, .{ .trans_a = true, .trans_b = false });
        defer t3.deinit();
        t1_case2.acquire();

        try gm.backward(t3);

        const expected_grad_t1 = &[_]T{ 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2 };
        const expected_grad_t2 = &[_]T{ 5, 5, 7, 7, 9, 9, 17, 17, 19, 19, 21, 21 };
        try std.testing.expectEqualSlices(T, expected_grad_t1, t1_case2.grad.?.data);
        try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);

        t1_case2.release();
        t1_case2.deinit();
        try t2.setup_grad(0);
    }

    // Case 3: Transpose B (t1: [2, 2, 3], t2: [2, 2, 3])
    {
        var t2_case3 = try Tensor.init(&[_]T{ 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1 }, &[_]usize{ 2, 2, 3 }, true, device);
        var t3 = try t1.bmm(t2_case3, .{ .trans_a = false, .trans_b = true });
        defer t3.deinit();
        t2_case3.acquire();
        try gm.backward(t3);

        const expected_grad_t1 = &[_]T{ 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2 };
        const expected_grad_t2 = &[_]T{ 5, 7, 9, 5, 7, 9, 17, 19, 21, 17, 19, 21 };
        try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
        try std.testing.expectEqualSlices(T, expected_grad_t2, t2_case3.grad.?.data);

        try t1.setup_grad(0);
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

    try gm.backward(t3);
    const expected_grad_t1 = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2 = &[_]T{ 4, 4, 6, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
    try t1.setup_grad(0);
    try t2.setup_grad(0);

    // Case 2: Transpose A
    var t3_trans_a = try t1.bmm(t2, .{ .trans_a = true, .trans_b = false });
    defer t3_trans_a.deinit();

    try gm.backward(t3_trans_a);
    const expected_grad_t1_trans_a = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_a = &[_]T{ 3, 3, 7, 7 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_a, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_a, t2.grad.?.data);
    try t1.setup_grad(0);
    try t2.setup_grad(0);

    // Case 3: Transpose B
    var t3_trans_b = try t1.bmm(t2, .{ .trans_a = false, .trans_b = true });
    defer t3_trans_b.deinit();
    try gm.backward(t3_trans_b);

    const expected_grad_t1_trans_b = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_b = &[_]T{ 4, 6, 4, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_b, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_b, t2.grad.?.data);
    try t1.setup_grad(0);
    try t2.setup_grad(0);

    // Case 4: Transpose both A and B
    var t3_trans_ab = try t1.bmm(t2, .{ .trans_a = true, .trans_b = true });
    defer t3_trans_ab.deinit();
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
    const t3 = try t1.matvec(t2, .{});

    try t1.setup_grad(0.0);
    try t2.setup_grad(0.0);

    var gm = GraphManager(Tensor).init(device.allocator, .{});
    defer gm.deinit();

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

    try gm.backward(t3);

    const expected_grad_t1 = &[_]T{ 4, 5, 6 };
    const expected_grad_t2 = &[_]T{ 1, 2, 3 };

    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.grad.?.data);
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.grad.?.data);
}


test "tensor/inplace_add" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var cpu = zg.device.HostDevice.init(allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const u = try NDTensor(f32).ones(&.{ 2, 2 }, true, device);
    defer u.deinit();

    const v = try NDTensor(f32).ones(&.{ 2, 2 }, true, device);
    defer v.deinit();

    const x = try u.mul(v);
    defer x.deinit();

    const a = try NDTensor(f32).ones(&.{ 2, 2 }, true, device);
    defer a.deinit();

    const b = try NDTensor(f32).ones(&.{ 2, 2 }, true, device);
    defer b.deinit();

    const c = try NDTensor(f32).ones(&.{ 2, 2 }, true, device);
    defer c.deinit();

    // x now carries 4 contexts for (a), (b), (c), (u, v)
    try a.add_(x);
    try b.add_(x);
    try c.add_(x);

    try x.setup_grad(2.0);

    try x.backward();
    try std.testing.expectEqualSlices(f32, &.{ 4, 4, 4, 4 }, x.get_data());
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2, 2 }, a.assume_grad_data());
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2, 2 }, b.assume_grad_data());
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2, 2 }, c.assume_grad_data());
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2, 2 }, u.assume_grad_data());
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2, 2 }, v.assume_grad_data());

    // check the children...
    var children = x.child_iterator() orelse unreachable;
    try std.testing.expectEqual(children.next().?, c);
    try std.testing.expectEqual(children.next().?, b);
    try std.testing.expectEqual(children.next().?, a);
    try std.testing.expectEqual(children.next().?, u);
    try std.testing.expectEqual(children.next().?, v);
    try std.testing.expectEqual(children.next(), null);
}

// TODO: Fix memory freeing conundrum with gather() then dont use an arena here.;;
//test "tensor/gather" {;;
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
//    // case 1: basic gather;;
//    const input_data = [_]T{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
//    const input_shape = [_]usize{ 3, 3 };
//    var input = try Tensor.init(&input_data, &input_shape, true, device);
//    defer input.deinit();
//
//    const index_data = [_]usize{ 0, 1, 1, 2, 0, 2 };
//    const index_shape = [_]usize{ 3, 2 };
//    var index = try NDTensor(usize).init(&index_data, &index_shape, false, device);
//    defer index.deinit();
//
//    var output = try input.gather(device, .{ .indices = index, .dim = 1 });;;
//    defer output.deinit();
//
//    try std.testing.expectEqualSlices(T, &[_]T{ 1, 2, 5, 6, 7, 9 }, output.data.data);
//    try std.testing.expectEqualSlices(usize, &index_shape, output.data.shape.slice());
//
//    // case 2: grad check
//    var gm = GraphManager(Tensor).init(device.allocator, .{});
//    defer gm.deinit();
//
//    output.grad.?.fill(1.0, device);
//    try gm.backward(output);
//
//    const expected_grad = [_]T{ 1, 1, 0, 0, 1, 1, 1, 0, 1 };
//    try std.testing.expectEqualSlices(T, &expected_grad, input.grad.?.data);
//
//    // case 3: out of bounds
//    //try index.set(&.{ 0, 0 }, 3);
//    //try std.testing.expectError(error.IndexOutOfBounds, input.gather(device, .{ .indices = index, .dim = 1 }));;;
//}

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
