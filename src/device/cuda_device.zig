const std = @import("std");
const log = std.log.scoped(.cuda);
const cuda = @import("cuda").impl;
const DataCache = @import("data_cache.zig");
const ReduceType = @import("device_common.zig").ReduceType;
const SmaxType = @import("device_common.zig").SmaxType;

pub fn dtype(comptime T: type) cuda.dtype {
    return switch (T) {
        f16 => @compileError("f16 not current supported."),
        f32 => cuda.SINGLE,
        f64 => cuda.DOUBLE,
        else => @compileError("Only floating points supported"),
    };
}

pub fn rdxtype(rdx: ReduceType) cuda.dtype {
    return switch (rdx) {
        .SUM => cuda.RDX_MAX,
        .MEAN => cuda.RDX_MEAN,
    };
}

pub fn smaxtype(op: SmaxType) cuda.dtype {
    return switch (op) {
        .FAST => cuda.SMAX_FAST,
        .MAX => cuda.SMAX_MAX,
        .LOG => cuda.SMAX_LOG,
    };
}

// combine host device and cuda device to single device reference.
pub const DeviceReference = @import("device_reference.zig").DeviceReference(CudaDevice);

// callback to replace host reference to union
pub const HostDevice = @import("host_device.zig").HostDevice;
pub fn host_reference(self: *HostDevice) DeviceReference {
    return .{ .ptrs = DeviceReference.DevicePtrs{ .host = self }, .allocator = self.allocator };
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

    /// Computes mat-vec assuming a stride of 1 for the vec and row-major.
    /// a * (M, N) x (N,) + b * (N,) = (M,)
    /// Y = aAX + bY
    pub fn matvec(
        self: *const Blas,
        T: type,
        A: []const T,
        x: []const T,
        y: []T,
        M: usize,
        N: usize,
        trans_a: bool,
        alpha: T,
        beta: T,
    ) void {
        cuda.gemv(dtype(T), self.cublas(), A.len, x.ptr, y.ptr, M, N, trans_a, alpha, beta);
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

    pub fn maxForward(
        self: *const Blas,
        T: type,
        src: []const T,
        dst: []T,
        idx: *i32,
    ) void {
        cuda.max_forward(dtype(T), self.cublas(), src.ptr, dst.ptr, idx, src.len);
    }

    pub fn maxBackward(
        self: *const Blas,
        T: type,
        y_grd: []const T,
        x_grd: []T,
        idx: *i32,
    ) void {
        cuda.max_reverse(dtype(T), self.cublas(), y_grd.ptr, x_grd.ptr, idx);
    }

    pub fn sum(
        self: *const Blas,
        T: type,
        x: []const T,
        y: []T,
    ) T {
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
};

pub const NN = struct {
    pub fn reluForward(self: *const NN, comptime T: type, x: []const T, y: []T) void {
        cuda.relu_forward(dtype(T), self.stream(), x.ptr, y.ptr, x.len);
    }

    pub fn reluBackward(self: *const NN, comptime T: type, x: []const T, y_grd: []const T, x_grd: []T) void {
        cuda.relu_reverse(dtype(T), self.stream(), x.ptr, y_grd.ptr, x_grd.ptr, x.len);
    }

    pub fn smaxVecForward(self: *const NN, comptime T: type, x: []const T, y: []T, op: SmaxType) void {
        cuda.smax_vec_forward(dtype(T), self.cudnn(), x.ptr, y.ptr, x.len, smaxtype(op));
    }

    pub fn smaxVecBackward(self: *const NN, comptime T: type, y_val: []const T, y_grd: []const T, x_grd: []T, op: SmaxType) void {
        cuda.smax_vec_reverse(dtype(T), self.cudnn(), y_val.ptr, y_grd.ptr, x_grd.ptr, y_val.len, smaxtype(op));
    }

    pub fn smaxRowForward(self: *const NN, comptime T: type, X: []const T, Y: []T, m: usize, n: usize, op: SmaxType) void {
        cuda.smax_2D_row_forward(dtype(T), self.cudnn(), X.ptr, Y.ptr, m, n, op);
    }

    pub fn smaxRowBackward(self: *const NN, comptime T: type, y_val: []const T, y_grd: []const T, x_grd: []T, m: usize, n: usize, op: SmaxType) void {
        cuda.smax_2D_row_reverse(dtype(T), self.cudnn(), y_val.ptr, y_grd.ptr, x_grd.ptr, m, n, smaxtype(op));
    }

    pub fn nllLoss1DIndexForward(self: *const NN, comptime T: type, src: []T, trg: usize, dst: []T, input_logits: bool, reduce: bool, reduce_type: ReduceType) f64 {
        const _reduce = if (reduce) rdxtype(reduce_type) else cuda.RDX_NONE;
        cuda.nll_loss_1D_index_forward(dtype(T), self.cudnn(), src.ptr, trg, dst.ptr, src.len, input_logits, _reduce);
    }

    pub fn nllLoss1DIndexBackward(self: *const NN, comptime T: type, src_val: []const T, src_grd: []T, trg: usize, reduce_type: ReduceType) f64 {
        cuda.nll_loss_1D_index_reverse(dtype(T), self.cudnn(), src_grd.ptr, trg, src_grd.ptr, src_val.len, rdxtype(reduce_type));
    }

    fn cudnn(self: *const NN) *anyopaque {
        const device: *const CudaDevice = @alignCast(@fieldParentPtr("nn", self));
        return device.context.cudnn;
    }

    fn stream(self: *const NN) *anyopaque {
        const device: *const CudaDevice = @alignCast(@fieldParentPtr("nn", self));
        return device.context.stream;
    }
};

pub const ScratchMemory = struct {
    head: usize = 0,
    tail: usize = 0,
    pub fn deinit(self: *ScratchMemory, stream: *anyopaque) void {
        if (self.head != 0) {
            cuda.memFree(@as(*anyopaque, @ptrFromInt(self.head)), stream);
        }
        self.* = undefined;
    }
    // Each device has it's own scratch memory because streams work
    // like queues. It's safe if the same queue tries to access its
    // own memory, but dangerous if streams can use other scratch.
    pub fn get(self: *ScratchMemory, comptime T: type, n: usize, stream: *anyopaque) []T {
        const total: usize = @sizeOf(T) * n;
        // check if we have enough scratch to provide a payload
        if (self.tail < (self.head + total)) {
            if (self.head != 0) {
                cuda.memFree(@as(*anyopaque, @ptrFromInt(self.head)), stream);
            }
            // after a first pass through the network, we should know if we have enough memory.
            const ptr = cuda.memAlloc(total) orelse @panic("Cannot allocate scratch memory.");
            self.head = @intFromPtr(ptr);
            self.tail = self.head + total;
        }
        const ptr: [*]T = @ptrFromInt(self.scratch.head);
        return ptr[0..n];
    }
};

pub const CudaDevice = struct {
    const Error = std.mem.Allocator.Error;

    // keep these out of the user api
    context: struct {
        device_number: u32,
        stream: *anyopaque,
        cublas: *anyopaque,
        cudnn: *anyopaque,
    },
    nn: NN,
    blas: Blas,
    cache: DataCache,
    scratch: ScratchMemory,
    allocator: std.mem.Allocator,

    pub fn init(device_number: u32, backing_allocator: std.mem.Allocator) CudaDevice {
        cuda.initDevice(device_number);
        const stream = cuda.initStream() orelse unreachable;
        const cublas = cuda.initCublasHandle(stream) orelse unreachable;
        const cudnn = cuda.initCudnnHandle(stream) orelse unreachable;
        return .{
            .context = .{
                .device_number = device_number,
                .stream = stream,
                .cublas = cublas,
                .cudnn = cudnn,
            },
            .nn = .{},
            .blas = .{},
            .loss = .{},
            // at some point, maybe move this to unmanged
            .cache = .{ .allocator = backing_allocator },
            .scratch = .{},
            .allocator = backing_allocator,
        };
    }

    pub fn deinit(self: *CudaDevice) void {
        // free everything we own first...
        var iter = self.cache.poppingIterator();
        while (iter.next()) |ptr| cuda.memFree(ptr, self.context.stream);
        self.cache.deinit();

        self.sync();

        cuda.deinitCublasHandle(self.context.cublas);
        cuda.deinitStream(self.context.stream);
        self.* = undefined;
    }

    pub fn reference(self: *CudaDevice) DeviceReference {
        return .{ .ptrs = DeviceReference.DevicePtrs{ .aux = self }, .allocator = self.allocator };
    }

    pub fn memAllocUncached(self: *CudaDevice, comptime T: type, n: usize) Error![]T {
        const raw_ptr = cuda.memAlloc(n * @sizeOf(T), self.context.stream);
        const dev_ptr: [*]T = @ptrCast(@alignCast(raw_ptr orelse return Error.OutOfMemory));
        return dev_ptr[0..n];
    }

    pub fn memAlloc(self: *CudaDevice, comptime T: type, n: usize) Error![]T {
        return self.cache.get(T, n) orelse self.memAllocUncached(T, n);
    }

    pub fn memDupe(self: *CudaDevice, comptime T: type, src: []const T) Error![]T {
        const dup = try self.memAlloc(T, src.len);
        self.memTransfer(T, src, dup, .DtoD);
        return dup;
    }

    pub fn memFreeUncached(self: *CudaDevice, slice: anytype) void {
        cuda.memFree(@constCast(slice.ptr), self.context.stream);
    }

    pub fn memFree(self: *CudaDevice, slice: anytype) void {
        const T = std.meta.Child(@TypeOf(slice));
        if (!self.cache.put(T, slice)) {
            self.memFreeUncached(slice);
        }
    }

    pub fn memCreate(self: HostDevice, comptime T: type) Error!*T {
        const raw_ptr = cuda.memAlloc(@sizeOf(T), self.context.stream);
        const dev_ptr: *T = @ptrCast(@alignCast(raw_ptr orelse return Error.OutOfMemory));
        return dev_ptr;
    }

    pub fn memDestroy(self: HostDevice, ptr: anytype) void {
        cuda.memFree(@constCast(ptr), self.context.stream);
    }

    pub fn memFill(self: CudaDevice, comptime T: type, slice: []T, value: T) void {
        var _value = value; // move from register memory
        cuda.memFill(dtype(T), slice.ptr, slice.len, &_value, self.context.stream);
    }

    pub fn memSequence(self: CudaDevice, comptime T: type, slice: []T, initial: T, step: T) void {
        var _init = initial; // move from register memory
        var _step = step; // move from register memory
        cuda.memSequence(dtype(T), slice.ptr, slice.len, &_init, &_step, self.context.stream);
    }

    const Direction = enum { HtoD, DtoH, DtoD };

    pub fn memTransfer(
        self: CudaDevice,
        comptime T: type,
        src: []const T,
        dst: []T,
        direction: Direction,
    ) void {
        std.debug.assert(src.len == dst.len);
        switch (direction) {
            .HtoD => cuda.memcpyHtoD(dst.ptr, src.ptr, src.len * @sizeOf(T), self.context.stream),
            .DtoH => cuda.memcpyDtoH(dst.ptr, src.ptr, src.len * @sizeOf(T), self.context.stream),
            .DtoD => cuda.memcpyDtoD(dst.ptr, src.ptr, src.len * @sizeOf(T), self.context.stream),
        }
    }

    pub fn sync(self: CudaDevice) void {
        cuda.streamSynchronize(self.context.stream);
    }

    // Unlike host compatibility, this can vary. The upper-level DeviceReference
    // type will ensure to only hand other CudaDevice objects to this function.
    pub fn isCompatible(self: *const CudaDevice, other: *const CudaDevice) bool {
        return self.context.device_number == other.context.device_number;
    }
};
