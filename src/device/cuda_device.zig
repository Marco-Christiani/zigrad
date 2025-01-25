const std = @import("std");
const log = std.log.scoped(.cuda);
const cuda = @import("cuda").impl;
const DimensionMap = @import("dimension_map.zig");
const ReduceType = @import("device_common.zig").ReduceType;
const SmaxType = @import("device_common.zig").SmaxType;
const RandType = @import("device_common.zig").RandType;
const BinaryOp = @import("device_common.zig").BinaryOp;

pub fn dtype(comptime T: type) cuda.dtype {
    return switch (T) {
        f16 => @compileError("f16 not current supported."),
        f32 => cuda.SINGLE,
        f64 => cuda.DOUBLE,
        // we need to expand this more, but we're currently allocating "usize"
        // tensors as indices for functions like gather. They just can't go through
        // a blas or nn function, so this seems appropriate for the time being.
        else => @panic("Only supports floating point, found: " ++ @typeName(T)),
    };
}

pub fn binary_op(rdx: BinaryOp) cuda.BINARY_OP {
    return switch (rdx) {
        .add => cuda.ADD,
        .min => cuda.MIN,
        .max => cuda.MAX,
    };
}

pub fn rdxtype(rdx: ReduceType) cuda.dtype {
    return switch (rdx) {
        .sum => cuda.RDX_MAX,
        .mean => cuda.RDX_MEAN,
    };
}

pub fn smaxtype(op: SmaxType) cuda.dtype {
    return switch (op) {
        .fast => cuda.SMAX_FAST,
        .max => cuda.SMAX_MAX,
        .log => cuda.SMAX_LOG,
    };
}

pub fn randtype(op: RandType) cuda.dtype {
    return switch (op) {
        .uniform => cuda.UNIFORM,
        .normal => cuda.NORMAL,
    };
}

// combine host device and cuda device to single device reference.
pub const DeviceReference = @import("device_reference.zig").DeviceReference(CudaDevice);

// callback to replace host reference to union
pub const HostDevice = @import("host_device.zig").HostDevice;
pub fn host_reference(self: *HostDevice) DeviceReference {
    return .{
        .ptrs = DeviceReference.DevicePtrs{ .host = self },
        .cache = &self.cache,
        .allocator = self.allocator,
    };
}

pub const Blas = struct {
    /// Computes dot product assuming a stride of 1 and row-major. (N,) x (N,) = (1,)
    pub fn dot(
        self: *const Blas,
        T: type,
        x: []const T,
        y: []const T,
        z: []T,
    ) void {
        cuda.dot(dtype(T), self.cublas(), x.ptr, y.ptr, z.ptr, x.len);
    }

    pub fn add(
        self: *const Blas,
        T: type,
        x: []const T,
        y: []const T,
        z: []T,
    ) void {
        cuda.addition(dtype(T), self.stream(), x.ptr, y.ptr, z.ptr, x.len, y.len, z.len);
    }

    pub fn sub(
        self: *const Blas,
        T: type,
        x: []const T,
        y: []const T,
        z: []T,
    ) void {
        cuda.subtraction(dtype(T), self.stream(), x.ptr, y.ptr, z.ptr, x.len, y.len, z.len);
    }

    pub fn mul(
        self: *const Blas,
        T: type,
        x: []const T,
        y: []const T,
        z: []T,
    ) void {
        cuda.multiplication(dtype(T), self.stream(), x.ptr, y.ptr, z.ptr, x.len, y.len, z.len);
    }

    pub fn div(
        self: *const Blas,
        T: type,
        x: []const T,
        y: []const T,
        z: []T,
    ) void {
        cuda.division(dtype(T), self.stream(), x.ptr, y.ptr, z.ptr, x.len, y.len, z.len);
    }

    /// Computes mat-vec assuming a stride of 1 for the vec and row-major.
    /// a * (M, N) x (N,) + b * (N,) = (M,)
    /// Y = aAX + bY
    pub fn matvec(
        self: *const Blas,
        T: type,
        A: []const T,
        x: []const T,
        y: []T,
        m: usize,
        n: usize,
        trans_a: bool,
        alpha: T,
        beta: T,
    ) void {
        cuda.gemv(dtype(T), self.cublas(), A.ptr, x.ptr, y.ptr, m, n, trans_a, alpha, beta);
    }

    ///  Assumes row-major.
    ///  (M, K) x (K, N) = (M, N)
    /// C := alpha*op(A)*op(B) + beta*C
    pub fn matmul(
        self: *const Blas,
        T: type,
        A: []const T,
        B: []const T,
        C: []T,
        M: usize,
        N: usize,
        K: usize,
        trans_a: bool,
        trans_b: bool,
        lda: usize,
        ldb: usize,
        ldc: usize,
        alpha: T,
        beta: T,
    ) void {
        cuda.gemm(dtype(T), self.cublas(), A.ptr, B.ptr, C.ptr, M, N, K, trans_a, trans_b, lda, ldb, ldc, alpha, beta);

        // NOTE: These are the transformations to the sizes and leading dimensions that have to happen. Please do not
        // remove the commented out code until we have decided how to support columnwise backends (cuda is columnwise).

        //if (!trans_a and !trans_b) {
        //    cuda.gemm(dtype(T), self.cublas(), A.ptr, B.ptr, C.ptr, m, n, k, trans_a, trans_b, n, k, k, alpha, beta);
        //} else if (trans_a and trans_b) {
        //    cuda.gemm(dtype(T), self.cublas(), A.ptr, B.ptr, C.ptr, n, m, k, trans_a, trans_b, n, m, k, alpha, beta);
        //} else if (trans_a and !trans_b) {
        //    cuda.gemm(dtype(T), self.cublas(), A.ptr, B.ptr, C.ptr, n, m, k, trans_a, trans_b, n, k, k, alpha, beta);
        //} else {
        //    cuda.gemm(dtype(T), self.cublas(), A.ptr, B.ptr, C.ptr, m, n, k, trans_a, trans_b, n, n, k, alpha, beta);
        //}
    }

    /// Outer product: A = alpha(xy') + A
    /// A: (M, N)
    pub fn outer(
        self: *const Blas,
        T: type,
        x: []const T,
        y: []const T,
        A: []T,
        alpha: T,
    ) void {
        cuda.ger(dtype(T), self.cublas(), x.ptr, y.ptr, A.ptr, x.len, y.len, y.len, alpha);
    }

    pub fn nrm2(
        self: *const Blas,
        T: type,
        x: []const T,
        y: []T,
    ) void {
        cuda.nrm2(dtype(T), self.cublas(), x.ptr, y.ptr, x.len);
    }

    pub fn clip_norm(
        self: *Blas,
        comptime T: type,
        x: []T,
        max_norm: T,
        delta: T,
    ) void {
        const par = self.parent();
        const scratch = par.scratch.get(T, 1, par.context.stream);
        cuda.clip_norm(dtype(T), self.cublas(), x.ptr, scratch.ptr, x.len, max_norm, delta);
    }

    pub fn max_forward(
        self: *const Blas,
        T: type,
        src: []const T,
        dst: []T,
        idx: *i32,
    ) void {
        cuda.max_forward(dtype(T), self.cublas(), src.ptr, dst.ptr, idx, src.len);
    }

    pub fn max_backward(
        self: *const Blas,
        T: type,
        y_grd: []const T,
        x_grd: []T,
        idx: *i32,
    ) void {
        cuda.max_reverse(dtype(T), self.cublas(), y_grd.ptr, x_grd.ptr, idx);
    }

    pub fn reduce(
        self: *Blas,
        T: type,
        x_vals: []const T,
        x_dims: []const usize,
        y_vals: []T,
        y_dims: []const usize,
        dim_idxs: []const usize,
        alpha: T,
        beta: T,
        comptime op: BinaryOp,
    ) void {
        const par = self.parent();
        const _alpha = alpha;
        const _beta = beta;
        cuda.reduce(
            dtype(T),
            par.context.cutensor,
            x_vals.ptr,
            x_dims.ptr,
            x_dims.len,
            y_vals.ptr,
            y_dims.ptr,
            y_dims.len,
            &par.scratch.start,
            &par.scratch.total,
            dim_idxs.ptr,
            dim_idxs.len,
            &_alpha,
            &_beta,
            binary_op(op),
        );
    }

    // V1 API Parity
    pub fn transpose(
        self: *Blas,
        T: type,
        x_vals: []const T,
        x_dims: []const usize,
        y_vals: []T,
        y_dims: []const usize,
        alpha: T,
    ) void {
        self.permutate(T, x_vals, x_dims, "ij", y_vals, y_dims, "ji", alpha);
    }

    /// can perform broadcasting of the sort:hi
    pub fn permutate(
        self: *Blas,
        T: type,
        x_vals: []const T,
        x_dims: []const usize,
        x_syms: []const u8,
        y_vals: []T,
        y_dims: []const usize,
        y_syms: []const u8,
        alpha: T,
    ) void {
        const par = self.parent();
        const _alpha = alpha;
        cuda.permutate(
            dtype(T),
            par.context.cutensor,
            x_vals.ptr,
            x_dims.ptr,
            x_syms.ptr,
            x_dims.len,
            y_vals.ptr,
            y_dims.ptr,
            y_syms.ptr,
            y_dims.len,
            &par.scratch.start,
            &par.scratch.total,
            &_alpha,
        );
    }

    pub fn sum(
        self: *const Blas,
        T: type,
        x: []const T,
        y: []T,
    ) void {
        cuda.reduce_sum(dtype(T), self.cublas(), x.ptr, y.ptr, x.len);
    }

    pub fn scale(
        self: *const Blas,
        T: type,
        x: []T,
        alpha: T,
    ) void {
        cuda.scale(dtype(T), self.cublas(), x.ptr, x.len, alpha);
    }

    pub fn axpy(
        self: *const Blas,
        comptime T: type,
        x: []const T,
        y: []T,
        alpha: *const T,
    ) void {
        cuda.axpy(dtype(T), self.cublas(), x.ptr, y.ptr, x.len, alpha);
    }

    fn cublas(self: *const Blas) *anyopaque {
        const device: *const CudaDevice = @alignCast(@fieldParentPtr("blas", self));
        return device.context.cublas;
    }

    fn stream(self: *const Blas) *anyopaque {
        const device: *const CudaDevice = @alignCast(@fieldParentPtr("blas", self));
        return device.context.stream;
    }

    fn parent(self: *Blas) *CudaDevice {
        return @as(*CudaDevice, @alignCast(@fieldParentPtr("blas", self)));
    }
};

pub const NN = struct {
    pub fn relu_forward(self: *const NN, comptime T: type, x: []const T, y: []T) void {
        cuda.relu_forward(dtype(T), self.stream(), x.ptr, y.ptr, x.len);
    }

    pub fn relu_backward(self: *const NN, comptime T: type, x: []const T, y_grd: []const T, x_grd: []T) void {
        cuda.relu_reverse(dtype(T), self.stream(), x.ptr, y_grd.ptr, x_grd.ptr, x.len);
    }

    pub fn smax_vec_forward(self: *const NN, comptime T: type, x: []const T, y: []T, op: SmaxType) void {
        cuda.smax_vec_forward(dtype(T), self.cudnn(), x.ptr, y.ptr, x.len, smaxtype(op));
    }

    pub fn smax_vec_backward(self: *const NN, comptime T: type, y_val: []const T, y_grd: []const T, x_grd: []T, op: SmaxType) void {
        cuda.smax_vec_reverse(dtype(T), self.cudnn(), y_val.ptr, y_grd.ptr, x_grd.ptr, y_val.len, smaxtype(op));
    }

    pub fn smax_row_forward(self: *const NN, comptime T: type, X: []const T, Y: []T, m: usize, n: usize, op: SmaxType) void {
        cuda.smax_2D_row_forward(dtype(T), self.cudnn(), X.ptr, Y.ptr, m, n, op);
    }

    pub fn smax_row_backward(self: *const NN, comptime T: type, y_val: []const T, y_grd: []const T, x_grd: []T, m: usize, n: usize, op: SmaxType) void {
        cuda.smax_2D_row_reverse(dtype(T), self.cudnn(), y_val.ptr, y_grd.ptr, x_grd.ptr, m, n, smaxtype(op));
    }

    pub fn nll_loss_1d_index_forward(self: *const NN, comptime T: type, src: []T, trg: usize, dst: []T, input_logits: bool, reduce: bool, reduce_type: ReduceType) f64 {
        const _reduce = if (reduce) rdxtype(reduce_type) else cuda.RDX_NONE;
        cuda.nll_loss_1D_index_forward(dtype(T), self.cudnn(), src.ptr, trg, dst.ptr, src.len, input_logits, _reduce);
    }

    pub fn nll_loss_1d_index_backward(self: *const NN, comptime T: type, src_val: []const T, src_grd: []T, trg: usize, reduce_type: ReduceType) f64 {
        cuda.nll_loss_1D_index_reverse(dtype(T), self.cudnn(), src_grd.ptr, trg, src_grd.ptr, src_val.len, rdxtype(reduce_type));
    }

    pub fn nll_loss_1d_encode_forward(self: *const NN, comptime T: type, src: []T, trg: []const T, dst: []T, input_logits: bool, reduce: bool, reduce_type: ReduceType) f64 {
        const _reduce = if (reduce) rdxtype(reduce_type) else cuda.RDX_NONE;
        cuda.nll_loss_1D_index_forward(dtype(T), self.cudnn(), src.ptr, trg, dst.ptr, src.len, input_logits, _reduce);
    }

    pub fn nll_loss_1d_encode_backward(self: *const NN, comptime T: type, src_val: []const T, trg_val: []const T, src_grd: []T, reduce_type: ReduceType) f64 {
        cuda.nll_loss_1D_index_reverse(dtype(T), self.cudnn(), src_grd.ptr, trg_val.ptr, src_grd.ptr, src_val.len, rdxtype(reduce_type));
    }

    pub fn clip_norm(self: *NN, comptime T: type, x_val: []T, x_grd: []const T, max_nrm: T, delta: T) f64 {
        const scratch = self.parent().scratch.get(T, 1);
        cuda.clip_norm(dtype(T), self.cublas(), x_val.ptr, x_grd.ptr, scratch.ptr, max_nrm, delta, x_val.len);
    }

    fn cublas(self: *const NN) *anyopaque {
        const device: *const CudaDevice = @alignCast(@fieldParentPtr("nn", self));
        return device.context.cublas;
    }

    fn cudnn(self: *const NN) *anyopaque {
        const device: *const CudaDevice = @alignCast(@fieldParentPtr("nn", self));
        return device.context.cudnn;
    }

    fn stream(self: *const NN) *anyopaque {
        const device: *const CudaDevice = @alignCast(@fieldParentPtr("nn", self));
        return device.context.stream;
    }

    fn parent(self: *NN) *CudaDevice {
        return @alignCast(@fieldParentPtr("nn", self));
    }
};

pub const ScratchMemory = struct {
    start: usize = 0,
    total: usize = 0,
    pub fn deinit(self: *ScratchMemory, stream: *anyopaque) void {
        if (self.start != 0) {
            cuda.mem_free(@as(*anyopaque, @ptrFromInt(self.start)), stream);
        }
        self.* = undefined;
    }
    // Each device has it's own scratch memory because streams work
    // like queues. It's safe if the same queue tries to access its
    // own memory, but dangerous if streams can use other scratch.
    pub fn get(self: *ScratchMemory, comptime T: type, n: usize, stream: *anyopaque) []T {
        const total: usize = @sizeOf(T) * n;
        // check if we have enough scratch to provide a payload
        if (self.total < total) {
            if (self.start != 0) {
                cuda.mem_free(@as(*anyopaque, @ptrFromInt(self.start)), stream);
            }
            // after a first pass through the network, we should know if we have enough memory.
            const ptr = cuda.mem_alloc(total, stream) orelse @panic("Cannot allocate scratch memory.");
            self.start = @intFromPtr(ptr);
            self.total = total;
        }
        const ptr: [*]T = @ptrFromInt(self.start);
        return ptr[0..n];
    }
};

pub const CudaDevice = struct {
    const Error = std.mem.Allocator.Error;

    // keep these out of the user api
    context: struct {
        device_number: u32,
        properties: cuda.DevicePropertiesWrapper,
        stream: *anyopaque,
        cublas: *anyopaque,
        cudnn: *anyopaque,
        cutensor: cuda.CutensorWrapper,
    },
    nn: NN,
    blas: Blas,
    cache: DimensionMap,
    scratch: ScratchMemory,
    allocator: std.mem.Allocator,

    pub fn init(device_number: u32, backing_allocator: std.mem.Allocator) CudaDevice {
        const properties = cuda.init_device(device_number);
        const stream = cuda.init_stream() orelse unreachable;
        const cublas = cuda.init_cublas_handle(stream) orelse unreachable;
        const cudnn = cuda.init_cudnn_handle(stream) orelse unreachable;
        const cutensor = cuda.init_cutensor_handle(stream);
        return .{
            .context = .{
                .device_number = device_number,
                .properties = properties,
                .stream = stream,
                .cublas = cublas,
                .cudnn = cudnn,
                .cutensor = cutensor,
            },
            .nn = .{},
            .blas = .{},
            // at some point, maybe move this to unmanged
            .cache = .{ .allocator = backing_allocator },
            .scratch = .{},
            .allocator = backing_allocator,
        };
    }

    pub fn deinit(self: *CudaDevice) void {
        self.sync();
        self.cache.deinit();
        cuda.deinit_cudnn_handle(self.context.cudnn);
        cuda.deinit_cublas_handle(self.context.cublas);
        cuda.deinit_cutensor_handle(self.context.cutensor);
        cuda.deinit_stream(self.context.stream);
        self.* = undefined;
    }

    pub fn reference(self: *CudaDevice) DeviceReference {
        return .{
            .ptrs = DeviceReference.DevicePtrs{ .aux = self },
            .cache = &self.cache,
            .allocator = self.allocator,
        };
    }

    pub fn mem_alloc(self: *CudaDevice, comptime T: type, n: usize) Error![]T {
        const raw_ptr = cuda.mem_alloc(n * @sizeOf(T), self.context.stream);
        const dev_ptr: [*]T = @ptrCast(@alignCast(raw_ptr orelse return Error.OutOfMemory));
        return dev_ptr[0..n];
    }

    pub fn mem_dupe(self: *CudaDevice, comptime T: type, src: []const T) Error![]T {
        const dup = try self.mem_alloc(T, src.len);
        self.mem_transfer(T, src, dup, .DtoD);
        return dup;
    }

    pub fn mem_free(self: *CudaDevice, slice: anytype) void {
        cuda.mem_free(@constCast(slice.ptr), self.context.stream);
    }

    pub fn mem_create(self: CudaDevice, comptime T: type) Error!*T {
        const raw_ptr = cuda.mem_alloc(@sizeOf(T), self.context.stream);
        const dev_ptr: *T = @ptrCast(@alignCast(raw_ptr orelse return Error.OutOfMemory));
        return dev_ptr;
    }

    pub fn mem_destroy(self: CudaDevice, ptr: anytype) void {
        cuda.mem_free(@constCast(ptr), self.context.stream);
    }

    pub fn mem_fill(self: CudaDevice, comptime T: type, slice: []T, value: T) void {
        var _value = value; // move from register memory
        cuda.mem_fill(dtype(T), slice.ptr, slice.len, &_value, self.context.stream);
    }

    pub fn mem_copy(self: CudaDevice, comptime T: type, src: []const T, dst: []T) void {
        return self.mem_transfer(T, src, dst, .DtoD);
    }

    pub fn mem_random(self: CudaDevice, comptime T: type, slice: []T, op: RandType, seed: u64) void {
        // at some point, I should put in u64 support for the device random.
        cuda.mem_random(dtype(T), slice.ptr, slice.len, randtype(op), @truncate(seed), self.context.stream);
    }

    pub fn mem_sequence(self: CudaDevice, comptime T: type, slice: []T, initial: T, step: T) void {
        var _init = initial; // move from register memory
        var _step = step; // move from register memory
        cuda.mem_sequence(dtype(T), slice.ptr, slice.len, &_init, &_step, self.context.stream);
    }

    pub const Direction = enum { HtoD, DtoH, DtoD };

    pub fn mem_transfer(
        self: CudaDevice,
        comptime T: type,
        src: []const T,
        dst: []T,
        direction: Direction,
    ) void {
        std.debug.assert(src.len == dst.len);
        switch (direction) {
            .HtoD => cuda.memcpy_HtoD(dst.ptr, src.ptr, src.len * @sizeOf(T), self.context.stream),
            .DtoH => cuda.memcpy_DtoH(dst.ptr, src.ptr, src.len * @sizeOf(T), self.context.stream),
            .DtoD => cuda.memcpy_DtoD(dst.ptr, src.ptr, src.len * @sizeOf(T), self.context.stream),
        }
    }

    pub fn sync(self: CudaDevice) void {
        cuda.stream_synchronize(self.context.stream);
    }

    // Unlike host compatibility, this can vary. The upper-level DeviceReference
    // type will ensure to only hand other CudaDevice objects to this function.
    pub fn is_compatible(self: *const CudaDevice, other: *const CudaDevice) bool {
        return self.context.device_number == other.context.device_number;
    }
};
