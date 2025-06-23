// Binary element-wise operations:
//
//  These are operations that can be fully strided
//  with only (ptr,len) pairs. These operations may
//  inherently broadcast (or 'wrap') depending. Thus,
//  element wise ops assume that for (x,y) -> z:
//
//     max(x.len, y.len) % min(x.len, y.len) == 0
//
//     z.len == max(x.len, y.len)
//
// Note that elwise functions do not have `_bwd` functions
// because they can be expressed by further elwise ops:
//
//    d | add(x,y) -> z | :
//       dx = dz + dx (add assign)
//       dy = dz + dy (add assign)
//
pub fn add(T: type) type {
    return struct {
        pub const __name__ = "add";
        pub const __type__ = T;
        x: []const T,
        y: []const T,
        z: []T,
    };
}

pub fn sub(T: type) type {
    return struct {
        pub const __name__ = "sub";
        pub const __type__ = T;
        x: []const T,
        y: []const T,
        z: []T,
    };
}

pub fn mul(T: type) type {
    return struct {
        pub const __name__ = "mul";
        pub const __type__ = T;
        x: []const T,
        y: []const T,
        z: []T,
    };
}

pub fn div(T: type) type {
    return struct {
        pub const __name__ = "div";
        pub const __type__ = T;
        x: []const T,
        y: []const T,
        z: []T,
    };
}

// Linear Algebra Arithmetic Ops
//
// These ops deal with with basic blas-like routines
// between rank-1 and rank-2 matrices. These operations
// respect linearity and do not have `_bwd` functions because
// they can also be expressed in terms of other linalg ops.

pub fn dot(T: type) type {
    return struct {
        pub const __name__ = "dot";
        pub const __type__ = T;
        x: []const T,
        y: []const T,
        z: []T,
    };
}

pub fn axpy(T: type) type {
    return struct {
        pub const __name__ = "axpy";
        pub const __type__ = T;
        x: []const T,
        y: []T,
        alpha: *const T,
    };
}

pub fn outer(T: type) type {
    return struct {
        pub const __name__ = "outer";
        pub const __type__ = T;
        x: []const T,
        y: []const T,
        A: []T,
        alpha: T,
    };
}

pub fn transpose(T: type) type {
    return struct {
        pub const __name__ = "transpose";
        pub const __type__ = T;
        A: []const T,
        B: []T,
        m: usize,
        n: usize,
        alpha: T,
    };
}

pub fn matvec(T: type) type {
    return struct {
        pub const __name__ = "matvec";
        pub const __type__ = T;
        A: []const T,
        x: []const T,
        y: []T,
        m: usize,
        n: usize,
        trans_a: bool,
        alpha: T,
        beta: T,
    };
}

pub fn matmul(T: type) type {
    return struct {
        pub const __name__ = "matmul";
        pub const __type__ = T;
        A: []const T,
        B: []const T,
        C: []T,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        lda: usize,
        ldb: usize,
        ldc: usize,
        alpha: T,
        beta: T,
    };
}

pub fn bmm_acc(T: type) type {
    return struct {
        pub const __name__ = "bmm_acc";
        pub const __type__ = T;
        A: []const T,
        B: []const T,
        C: []T,
        A_shape: []const usize,
        B_shape: []const usize,
        C_shape: []const usize,
        trans_a: bool,
        trans_b: bool,
        lda: usize,
        ldb: usize,
        ldc: usize,
        alpha: T,
        beta: T,
    };
}

pub fn sum(T: type) type {
    return struct {
        pub const __name__ = "sum";
        pub const __type__ = T;
        x: []const T,
        y: []T,
    };
}

pub fn scale(T: type) type {
    return struct {
        pub const __name__ = "scale";
        pub const __type__ = T;
        x: []T,
        alpha: T,
    };
}

pub fn nrm2(T: type) type {
    return struct {
        pub const __name__ = "nrm2";
        pub const __type__ = T;
        x: []const T,
        y: []T,
    };
}

pub fn clip_nrm2(T: type) type {
    return struct {
        pub const __name__ = "clip_nrm2";
        pub const __type__ = T;
        x: []T,
        max_norm: T,
        delta: T,
    };
}

/////////////////
// non-linear ops

pub fn exp_fwd(T: type) type {
    return struct {
        pub const __name__ = "exp_fwd";
        pub const __type__ = T;
        x: []const T,
        y: []T,
    };
}

pub fn exp_bwd(T: type) type {
    return struct {
        pub const __name__ = "exp_bwd";
        pub const __type__ = T;
        x_g: []T,
        y: []const T,
        y_g: []const T,
    };
}

pub fn relu_fwd(T: type) type {
    return struct {
        pub const __name__ = "relu_fwd";
        pub const __type__ = T;
        x: []const T,
        y: []T,
    };
}

pub fn relu_bwd(T: type) type {
    return struct {
        pub const __name__ = "relu_bwd";
        pub const __type__ = T;
        x: []const T,
        x_g: []T,
        y_g: []const T,
    };
}

pub fn relu_inplace_bwd(T: type) type {
    return struct {
        pub const __name__ = "relu_inplace_bwd";
        pub const __type__ = T;
        x: []const T,
        x_g: []T,
    };
}

pub fn tanh_fwd(T: type) type {
    return struct {
        pub const __name__ = "tanh_fwd";
        pub const __type__ = T;
        x: []const T,
        y: []T,
    };
}

pub fn tanh_bwd(T: type) type {
    return struct {
        pub const __name__ = "tanh_bwd";
        pub const __type__ = T;
        x_g: []T,
        y: []const T,
        y_g: []const T,
    };
}

pub fn tanh_inplace_bwd(T: type) type {
    return struct {
        pub const __name__ = "tanh_inplace_bwd";
        pub const __type__ = T;
        x: []const T,
        x_g: []T,
    };
}

pub fn sigm_fwd(T: type) type {
    return struct {
        pub const __name__ = "sigm_fwd";
        pub const __type__ = T;
        x: []const T,
        y: []T,
    };
}

pub fn sigm_bwd(T: type) type {
    return struct {
        pub const __name__ = "sigm_bwd";
        pub const __type__ = T;
        x_g: []T,
        y: []const T,
        y_g: []const T,
    };
}

pub fn sigm_inplace_bwd(T: type) type {
    return struct {
        pub const __name__ = "sigm_inplace_bwd";
        pub const __type__ = T;
        x: []const T,
        x_g: []T,
    };
}

pub fn max_fwd(T: type) type {
    return struct {
        pub const __name__ = "max_fwd";
        pub const __type__ = T;
        x: []const T,
        y: []T,
    };
}

pub fn max_bwd(T: type) type {
    return struct {
        pub const __name__ = "max_bwd";
        pub const __type__ = T;
        x: []const T,
        x_g: []T,
        y: []const T,
        y_g: []const T,
    };
}

pub fn clamp_fwd(T: type) type {
    return struct {
        pub const __name__ = "clamp_fwd";
        pub const __type__ = T;
        x: []const T,
        y: []T,
        min: T,
        max: T,
    };
}

pub fn clamp_bwd(T: type) type {
    return struct {
        pub const __name__ = "clamp_bwd";
        pub const __type__ = T;
        x: []const T,
        x_g: []T,
        y_g: []const T,
        min: T,
        max: T,
    };
}

pub fn clamp_mask_fwd(T: type) type {
    return struct {
        pub const __name__ = "clamp_mask_fwd";
        pub const __type__ = T;
        x: []const T,
        y: []T,
        min: T,
        max: T,
        mask: []u8,
    };
}

pub fn clamp_mask_bwd(T: type) type {
    return struct {
        pub const __name__ = "clamp_mask_bwd";
        pub const __type__ = T;
        x_g: []T,
        y_g: []const T,
        mask: []u8,
    };
}

pub fn relu_mask_fwd(T: type) type {
    return struct {
        pub const __name__ = "relu_mask_fwd";
        pub const __type__ = T;
        x: []T,
        mask: []u8,
    };
}

pub fn relu_mask_bwd(T: type) type {
    return struct {
        pub const __name__ = "relu_mask_bwd";
        pub const __type__ = T;
        x_g: []T,
        mask: []u8,
    };
}

pub fn unbroadcast(T: type) type {
    return struct {
        pub const __name__ = "unbroadcast";
        pub const __type__ = T;
        x: []const T,
        x_shape: []const usize,
        y: []T,
        y_shape: []const usize,
        scratch: []T,
        alpha: T = 1.0,
        beta: T = 0.0,
    };
}

pub fn broadcast(T: type) type {
    return struct {
        pub const __name__ = "broadcast";
        pub const __type__ = T;
        x: []const T,
        x_shape: []const usize,
        y: []T,
        y_shape: []const usize,
        alpha: T = 1.0,
        beta: T = 0.0,
    };
}

pub fn sum_along(T: type) type {
    return struct {
        pub const __name__ = "sum_along";
        pub const __type__ = T;
        x: []const T,
        x_shape: []const usize,
        y: []T,
        y_shape: []const usize,
        dim: usize,
        alpha: T = 1.0,
        beta: T = 0.0,
    };
}

pub fn max_along(T: type) type {
    return struct {
        pub const __name__ = "max_along";
        pub const __type__ = T;
        x: []const T,
        x_shape: []const usize,
        y: []T,
        y_shape: []const usize,
        dim: usize,
        alpha: T = 1.0,
        beta: T = 0.0,
    };
}
