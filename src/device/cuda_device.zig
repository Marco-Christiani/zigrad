const std = @import("std");
const log = std.log.scoped(.cuda);
const cuda = @import("cuda").impl;
const ReduceType = @import("device_common.zig").ReduceType;
const SmaxType = @import("device_common.zig").SmaxType;
const RandType = @import("device_common.zig").RandType;
const BinaryOp = @import("device_common.zig").BinaryOp;
const CachingAllocator = @import("../allocators.zig").CachingAllocator(CudaMalloc);
const DeviceData = @import("../allocators.zig").DeviceData;
const Error = @import("../allocators.zig").Error;
const TransferDirection = @import("device_common.zig").TransferDirection;
const build_options = @import("build_options");

const opspec = @import("opspec.zig");

pub const CudaMalloc = struct {
    pub fn raw_alloc(n: usize, ctx: *anyopaque) ?[*]u8 {
        const w: *cuda.StreamWrapper = @ptrCast(@alignCast(ctx));
        return @ptrCast(@alignCast(cuda.mem_alloc(n, w.*)));
    }
    pub fn raw_free(buf: []u8, ctx: *anyopaque) void {
        const w: *cuda.StreamWrapper = @ptrCast(@alignCast(ctx));
        cuda.mem_free(buf.ptr, w.*);
    }
};

const CUDA_COMPILE_ERROR =
    \\ 
    \\You are attempting to use a CudaDevice object but CUDA is not enabled.
    \\
    \\To enable CUDA, build the project with -Dcuda_enabled=true. You can pass
    \\this flag to the Zigrad module in the build system if you are linking it
    \\into an existing project.
    \\
;

/////////////////////////////
// Cuda Device Implementation

const Self = @This();
pub const DeviceReference = @import("device_reference.zig");

// keep these out of the user api
context: struct {
    device_number: u32,
    properties: cuda.DevicePropertiesWrapper,
    stream: cuda.StreamWrapper,
    cublas: cuda.CublasWrapper,
    cudnn: cuda.CudnnWrapper,
    cutensor: cuda.CutensorWrapper,
    // Anchor to provide proper compiler error for CUDA builds
    __: if (build_options.enable_cuda) void else @compileError(CUDA_COMPILE_ERROR) = undefined,
},
cache: CachingAllocator,
capture: ExecutionGraph,

/// Conatiner level function to see how many devices you have.
pub fn device_count() u32 {
    return cuda.device_count();
}

pub fn init(device_number: u32) Self {
    return init_advanced(device_number, .{}); // system defaults
}

// TODO: There is probably more to configure than the
// caching allocator - make a unified optoins struct?
pub fn init_advanced(device_number: u32, opts: CachingAllocator.Options) Self {
    const properties = cuda.init_device(device_number);
    const stream = cuda.init_stream();
    const cublas = cuda.init_cublas_handle(stream);
    const cudnn = cuda.init_cudnn_handle(stream);
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
        .cache = CachingAllocator.init(opts),
        .capture = .{},
    };
}

pub fn deinit(self: *Self) void {
    self.sync();
    self.capture.free();
    self.cache.deinit(&self.context.stream);
    cuda.deinit_cudnn_handle(self.context.cudnn);
    cuda.deinit_cublas_handle(self.context.cublas);
    cuda.deinit_cutensor_handle(self.context.cutensor);
    cuda.deinit_stream(self.context.stream);
    self.* = undefined;
}

pub fn reference(self: *Self) DeviceReference {
    return .{ .ptrs = .{ .aux = self } };
}

pub fn dtype(T: type) cuda.dtype {
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
        .kaiming => @panic("Unimplemented"),
    };
}

///////////////////
// element wise ops

pub fn add(self: *const Self, T: type, p: opspec.add(T)) void {
    cuda.addition(dtype(T), self.context.stream, p.x.ptr, p.y.ptr, p.z.ptr, p.x.len, p.y.len, p.z.len);
}

pub fn sub(self: *const Self, T: type, p: opspec.sub(T)) void {
    cuda.subtraction(dtype(T), self.context.stream, p.x.ptr, p.y.ptr, p.z.ptr, p.x.len, p.y.len, p.z.len);
}

pub fn mul(self: *const Self, T: type, p: opspec.mul(T)) void {
    cuda.multiplication(dtype(T), self.context.stream, p.x.ptr, p.y.ptr, p.z.ptr, p.x.len, p.y.len, p.z.len);
}

pub fn div(self: *const Self, T: type, p: opspec.div(T)) void {
    cuda.division(dtype(T), self.context.stream, p.x.ptr, p.y.ptr, p.z.ptr, p.x.len, p.y.len, p.z.len);
}

/////////////////////
// linear algebra ops

pub fn dot(self: *const Self, T: type, p: opspec.dot(T)) void {
    cuda.dot(dtype(T), self.context.cublas, p.x.ptr, p.y.ptr, p.z.ptr, p.x.len);
}

pub fn outer(self: *const Self, T: type, p: opspec.outer(T)) void {
    cuda.ger(dtype(T), self.context.cublas, p.x.ptr, p.y.ptr, p.A.ptr, p.x.len, p.y.len, p.y.len, p.alpha);
}

pub fn matvec(self: *const Self, T: type, p: opspec.matvec(T)) void {
    cuda.gemv(dtype(T), self.context.cublas, p.A.ptr, p.x.ptr, p.y.ptr, p.m, p.n, p.trans_a, p.alpha, p.beta);
}

pub fn transpose(self: *Self, T: type, p: opspec.transpose(T)) void {
    const A_shape: [2]usize = .{ p.m, p.n };
    const B_shape: [2]usize = .{ p.n, p.m };
    self.permutate(T, p.A, &A_shape, "ij", p.B, &B_shape, "ji", p.alpha);
}

pub fn matmul(self: *const Self, T: type, p: opspec.matmul(T)) void {
    cuda.gemm(dtype(T), self.context.cublas, p.A.ptr, p.B.ptr, p.C.ptr, p.m, p.n, p.k, p.trans_a, p.trans_b, p.lda, p.ldb, p.ldc, p.alpha, p.beta);

    // NOTE: These are the transformations to the shape and leading dimensions that have to happen. Please do not
    // remove the commented out code until we have decided how to support columnwise backends (cuda is columnwise).

    //if (!trans_a and !trans_b) {
    //    cuda.gemm(dtype(T), self.context.cutensor, A.ptr, B.ptr, C.ptr, m, n, k, trans_a, trans_b, n, k, k, alpha, beta);
    //} else if (trans_a and trans_b) {
    //    cuda.gemm(dtype(T), self.context.cutensor, A.ptr, B.ptr, C.ptr, n, m, k, trans_a, trans_b, n, m, k, alpha, beta);
    //} else if (trans_a and !trans_b) {
    //    cuda.gemm(dtype(T), self.context.cutensor, A.ptr, B.ptr, C.ptr, n, m, k, trans_a, trans_b, n, k, k, alpha, beta);
    //} else {
    //    cuda.gemm(dtype(T), self.context.cutensor, A.ptr, B.ptr, C.ptr, m, n, k, trans_a, trans_b, n, n, k, alpha, beta);
    //}
}

pub fn bmm_acc(self: *Self, T: type, p: opspec.bmm_acc(T)) void {
    std.debug.assert(p.A_shape.len == 3 and p.B_shape.len == 3);
    const A_modes: []const u8 = if (p.trans_a) "ikj" else "ijk";
    const B_modes: []const u8 = if (p.trans_b) "imk" else "ikm";
    const C_modes: []const u8 = "ijm";
    const _alpha = p.alpha;
    const _beta = p.beta;

    // zig fmt: off
    cuda.contraction(
        dtype(T), self.context.cutensor,
        p.A.ptr, p.A_shape.ptr, A_modes.ptr, p.A_shape.len,
        p.B.ptr, p.B_shape.ptr, B_modes.ptr, p.B_shape.len,
        p.C.ptr, p.C_shape.ptr, C_modes.ptr, p.C_shape.len,
        &self.cache.scratch.start, &self.cache.scratch.total,
        &_alpha, &_beta,
    );
    // zig fmt: on
}

pub fn nrm2(self: *const Self, T: type, p: opspec.nrm2(T)) void {
    cuda.nrm2(dtype(T), self.context.cublas, p.x.ptr, p.y.ptr, p.x.len);
}

pub fn clip_nrm2(self: *Self, T: type, p: opspec.clip_nrm2(T)) void {
    const scratch = self.mem_scratch(T, 1) catch unreachable;
    cuda.clip_nrm2(dtype(T), self.context.cublas, p.x.ptr, scratch.ptr, p.x.len, p.max_norm, p.delta);
}

pub fn sum(self: *const Self, T: type, p: opspec.sum(T)) void {
    cuda.reduce_sum(dtype(T), self.context.cublas, p.x.ptr, p.y.ptr, p.x.len);
}

pub fn scale(self: *const Self, T: type, p: opspec.nrm2(T)) void {
    cuda.scale(dtype(T), self.context.cublas, p.x.ptr, p.x.len, p.alpha);
}

pub fn axpy(self: *const Self, T: type, p: opspec.axpy(T)) void {
    cuda.axpy(dtype(T), self.context.cublas, p.x.ptr, p.y.ptr, p.x.len, p.alpha);
}

//////////////////
// non-linear ops

pub fn max_fwd(self: *const Self, T: type, p: opspec.max_fwd(T)) void {
    cuda.max_fwd(dtype(T), self.context.stream, p.x.ptr, p.y.ptr, p.x.len);
}

pub fn max_bwd(self: *const Self, T: type, p: opspec.max_bwd(T)) void {
    cuda.max_bwd(dtype(T), self.context.stream, p.x.ptr, p.y.ptr, p.y_g.ptr, p.x_g.ptr, p.x.len);
}

/// can perform broadcasting of the sort:hi
pub fn permutate(
    self: *Self,
    T: type,
    x_vals: []const T,
    x_dims: []const usize,
    x_syms: []const u8,
    y_vals: []T,
    y_dims: []const usize,
    y_syms: []const u8,
    alpha: T,
) void {
    const _alpha = alpha;
    cuda.permutate(
        dtype(T),
        self.context.cutensor,
        x_vals.ptr,
        x_dims.ptr,
        x_syms.ptr,
        x_dims.len,
        y_vals.ptr,
        y_dims.ptr,
        y_syms.ptr,
        y_dims.len,
        &self.cache.scratch.start,
        &self.cache.scratch.total,
        &_alpha,
    );
}

pub fn exp(self: *const Self, T: type, p: opspec.exp_fwd(T)) void {
    cuda.pow_exp(dtype(T), self.context.stream, p.x.ptr, p.y.ptr, p.x.len);
}

pub fn relu_fwd(self: *const Self, T: type, p: opspec.relu_fwd(T)) void {
    cuda.relu_fwd(dtype(T), self.context.stream, p.x.ptr, p.y.ptr, p.x.len);
}

pub fn relu_bwd(self: *const Self, T: type, p: opspec.relu_bwd(T)) void {
    cuda.relu_bwd(dtype(T), self.context.stream, p.x.ptr, p.y_grd.ptr, p.x_grd.ptr, p.x.len);
}

//pub fn smax_vec_fwd(self: *const Self, T: type, x: []const T, y: []T, op: SmaxType) void {
//    cuda.smax_vec_fwd(dtype(T), self.cudnn(), x.ptr, y.ptr, x.len, smaxtype(op));
//}
//
//pub fn smax_vec_bwd(self: *const Self, T: type, y_val: []const T, y_grd: []const T, x_grd: []T, op: SmaxType) void {
//    cuda.smax_vec_bwd(dtype(T), self.cudnn(), y_val.ptr, y_grd.ptr, x_grd.ptr, y_val.len, smaxtype(op));
//}
//
//pub fn smax_row_fwd(self: *const Self, T: type, X: []const T, Y: []T, m: usize, n: usize, op: SmaxType) void {
//    cuda.smax_2D_row_fwd(dtype(T), self.cudnn(), X.ptr, Y.ptr, m, n, op);
//}
//
//pub fn smax_row_bwd(self: *const Self, T: type, y_val: []const T, y_grd: []const T, x_grd: []T, m: usize, n: usize, op: SmaxType) void {
//    cuda.smax_2D_row_bwd(dtype(T), self.cudnn(), y_val.ptr, y_grd.ptr, x_grd.ptr, m, n, smaxtype(op));
//}

pub fn nll_loss_1d_index_fwd(self: *const Self, T: type, src: []T, trg: usize, dst: []T, input_logits: bool, reduce: bool, reduce_type: ReduceType) f64 {
    const _reduce = if (reduce) rdxtype(reduce_type) else cuda.RDX_NONE;
    cuda.nll_loss_1D_index_fwd(dtype(T), self.cudnn(), src.ptr, trg, dst.ptr, src.len, input_logits, _reduce);
}

pub fn nll_loss_1d_index_bwd(self: *const Self, T: type, src_val: []const T, src_grd: []T, trg: usize, reduce_type: ReduceType) f64 {
    cuda.nll_loss_1D_index_bwd(dtype(T), self.cudnn(), src_grd.ptr, trg, src_grd.ptr, src_val.len, rdxtype(reduce_type));
}

pub fn nll_loss_1d_encode_fwd(self: *const Self, T: type, src: []T, trg: []const T, dst: []T, input_logits: bool, reduce: bool, reduce_type: ReduceType) f64 {
    const _reduce = if (reduce) rdxtype(reduce_type) else cuda.RDX_NONE;
    cuda.nll_loss_1D_index_fwd(dtype(T), self.cudnn(), src.ptr, trg, dst.ptr, src.len, input_logits, _reduce);
}

pub fn nll_loss_1d_encode_bwd(self: *const Self, T: type, src_val: []const T, trg_val: []const T, src_grd: []T, reduce_type: ReduceType) f64 {
    cuda.nll_loss_1D_index_bwd(dtype(T), self.cudnn(), src_grd.ptr, trg_val.ptr, src_grd.ptr, src_val.len, rdxtype(reduce_type));
}

pub fn sum_along(self: *Self, T: type, p: opspec.sum_along(T)) void {
    const dim_arr: [1]usize = .{p.dim};
    const _alpha: T = 1.0;
    const _beta: T = 0.0;
    cuda.reduce(
        dtype(T),
        self.context.cutensor,
        p.x.ptr,
        p.x_shape.ptr,
        p.x_shape.len,
        p.y.ptr,
        &self.cache.scratch.start,
        &self.cache.scratch.total,
        &dim_arr[0],
        dim_arr.len,
        &_alpha,
        &_beta,
        cuda.ADD,
    );
}

pub fn max_along(self: *Self, T: type, p: opspec.max_along(T)) void {
    const dim_arr: [1]usize = .{p.dim};
    const _alpha: T = 1.0;
    const _beta: T = 0.0;
    cuda.reduce(
        dtype(T),
        self.context.cutensor,
        p.x.ptr,
        p.x_shape.ptr,
        p.x_shape.len,
        p.y.ptr,
        &self.cache.scratch.start,
        &self.cache.scratch.total,
        &dim_arr[0],
        dim_arr.len,
        &_alpha,
        &_beta,
        cuda.MAX,
    );
}

pub fn mem_alloc(self: *Self, T: type, n: usize) ![]T {
    if (n == 0) return &.{};
    return self.cache.alloc(T, n, &self.context.stream);
}

pub fn mem_alloc_byte_mask(self: *Self, n: usize) ![]u8 {
    return self.mem_alloc(u8, @divFloor(n - 1, 8) + 1);
}

pub fn mem_dupe(self: *Self, T: type, src: []const T) ![]T {
    const dup = try self.mem_alloc(T, src.len);
    self.mem_transfer(T, src, dup, .DtoD);
    return dup;
}

pub fn mem_scratch(self: *Self, T: type, n: usize) ![]T {
    return self.cache.get_scratch(T, n, &self.context.stream);
}

pub fn mem_free(self: *Self, slice: anytype) void {
    if (slice.len == 0) return;
    return self.cache.free(slice, &self.context.stream);
}

pub fn mem_fill(self: Self, T: type, slice: []T, value: T) void {
    var _value = value; // move from register memory
    cuda.mem_fill(dtype(T), slice.ptr, slice.len, &_value, self.context.stream);
}

pub fn mem_copy(self: Self, T: type, src: []const T, dst: []T) void {
    return self.mem_transfer(T, src, dst, .DtoD);
}

pub fn mem_random(self: Self, T: type, slice: []T, op: RandType, rand: std.Random) void {
    // at some point, I should put in u64 support for the device random.
    cuda.mem_random(dtype(T), slice.ptr, slice.len, randtype(op), rand.int(u32), self.context.stream);
}

pub fn mem_sequence(self: Self, T: type, slice: []T, initial: T, step: T) void {
    var _init = initial; // move from register memory
    var _step = step; // move from register memory
    cuda.mem_sequence(dtype(T), slice.ptr, slice.len, &_init, &_step, self.context.stream);
}

pub fn mem_take(self: Self, T: type, src: []const T, idxs: []const usize, dst: []T) void {
    std.debug.assert(dst.len >= idxs.len);
    cuda.mem_take(dtype(T), src.ptr, src.len, idxs.ptr, idxs.len, dst.ptr, self.context.stream);
}

pub fn mem_transfer(self: Self, T: type, src: []const T, dst: []T, direction: TransferDirection) void {
    std.debug.assert(src.len == dst.len);
    switch (direction) {
        .HtoD => cuda.memcpy_HtoD(dst.ptr, src.ptr, src.len * @sizeOf(T), self.context.stream),
        .DtoH => cuda.memcpy_DtoH(dst.ptr, src.ptr, src.len * @sizeOf(T), self.context.stream),
        .DtoD => cuda.memcpy_DtoD(dst.ptr, src.ptr, src.len * @sizeOf(T), self.context.stream),
    }
}

pub fn clear_cache(self: *Self) void {
    self.cache.clear(&self.context.stream);
}

pub fn sync(self: Self) void {
    cuda.stream_synchronize(self.context.stream);
}

// Unlike host compatibility, this can vary. The upper-level DeviceReference
// type will ensure to only hand other Self objects to this function.
pub fn is_compatible(self: *const Self, other: *const Self) bool {
    return self.context.device_number == other.context.device_number;
}

const ExecutionGraph = struct {
    graph: ?struct {
        wrapper: cuda.GraphWrapper,
        saved: bool,
    } = null,

    pub const RunError = error{
        CaptureInProgress,
        UndefinedCapture,
    };

    pub fn run(self: *ExecutionGraph) RunError!void {
        if (self.graph) |graph| {
            if (!graph.saved) {
                return error.CaptureInProgress;
            }
            cuda.run_capture(graph.wrapper, self.stream());
        }
        return error.UndefinedCapture;
    }
    pub fn open(self: *ExecutionGraph, config: struct {}) void {
        _ = config; // fill this out later
        self.free();
        cuda.stream_synchronize(self.context.stream);
        self.graph = .{
            .wrapper = cuda.open_capture(self.stream()),
            .saved = false,
        };
    }
    pub fn save(self: *ExecutionGraph) void {
        if (self.graph) |*graph| {
            if (!graph.saved) {
                cuda.save_capture(graph.wrapper, self.stream());
                graph.saved = true;
            }
        }
    }
    pub fn free(self: *ExecutionGraph) void {
        self.save();
        if (self.graph) |*graph| {
            cuda.stream_synchronize(self.stream());
            cuda.free_capture(graph.wrapper, self.stream());
            self.graph = null;
        }
    }
    pub fn empty(self: ExecutionGraph) bool {
        return self.graph == null;
    }

    fn stream(self: *ExecutionGraph) cuda.StreamWrapper {
        const parent: *const Self = @alignCast(@fieldParentPtr("capture", self));
        return parent.context.stream;
    }
};
