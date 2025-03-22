# Refactor Proposal Unified Operation Specification

## Current approach

Adding operations requires changes across levels of abstraction and in multiple places. Top down this looks like:

1. Add a new method to NDTensor
2. Add a new method to NDArray
3. Add method to `DeviceReference` interface that forwards to device implementation
4. All devices need to add a method to support the new `DeviceReference` specification even if they do not support the operation
  a) kernel itself may need to be written at yet another lower level such as CUDA

This approach has several downsides. Some device-specific implementations could have different signatures which means updating the `DeviceReference` interface and conseqently all of device backends. Operation grouping within the `DeviceReference` API is a bit arbitrary aside from the BLAS group which is relatively clear. Even in the case of BLAS, not all BLAS libraries supply the same functionality and often provide more. This grouping through the use of `@fieldParentPtr` has already been primed for removal. In general, this set up requires a lot of effort, introduces complexity, forces devs to reason at several levels of the stack simultaneously, lacks flexibility, and has some somewhat ambiguous characteristics due to arbitrary decisions.

## Goals

- Simplify the process of adding new operations
- Reduce the amount of code that needs to be modified when adding or changing operations
- Maintain interface performance through avoiding indirection and inline switch.

*Non Goals*

- Changing any user facing API
- Complete rewrite of device backend

## Proposal

Replace the current interface-method-per-operation approach with a unified operation specification system that uses tagged unions to describe operations. Device implementations will handle these specifications through grouped dispatch functions with an unambiguous operation categorization system.

```zig
const OpSpec = union(enum) {
    // Unary
    relu: struct { alpha: f32 = 0.0 },
    sigmoid: void,
    softmax: SoftmaxSpec,

    // Element-wise
    add: void,
    mul: void,
    div: void,

    matmul: MatmulSpec, // BLAS L3

    // etc.
};
```

```zig
pub fn unary_op(self: DeviceReference, src: []const f32, dst: []f32, spec: OpSpec) void { ... }
pub fn binary_op(self: DeviceReference, a: []const f32, b: []const f32, dst: []f32, spec: OpSpec) void { ... }
// etc.
```


```zig
pub fn unary_op(self: HostDevice, src: []const f32, dst: []f32, op_spec: OpSpec) void {
    switch (op_spec) {
        .relu => |cfg| {
            if (self.capabilities.has_avx2) {
                self.relu_avx2(src, dst, cfg);
            } else {
                self.relu_scalar(src, dst, cfg);
            }
        },
        .softmax => |cfg| {
            switch (cfg.algorithm) {
                .auto => self.softmax_auto(src, dst, cfg),
                .stable => self.softmax_stable(src, dst, cfg),
                .fast => self.softmax_fast(src, dst, cfg),
                // etc.
            }
        },
        // etc.
        else => @panic("Unsupported operation"),
    }
}
```

Retain a reference to the `OpSpec` in `BackwardContext`

**TODO**

## Rationale

This design reduces the amount of code needed to add a new operation. Instead of modifying multiple files and adding several functions, we simply:

1. Add a new method to `NDTensor`
2. Add a new method to `NDArray`
3. Add a new variant to the `OpSpec` union
4. Implement the operation in the relevant device dispatch functions

Signature difference within an operation group are hidden behind the `OpSpec` abstraction. Ops can be clearly grouped by their I/O signature. Operation tracing should be easier and clearer as an added bonus.

## Trade-offs / Alternatives Considered

- Interface-Method-Per-Operation Approach (Current): Simple but verbose and harder to maintain. Rejected because it doesn't scale well with many operations.
- Designs with Function Pointers: Flexible but adds runtime overhead and can be cumbersome or opaque. Rejected due to performance concerns.
- Separate Config Unions Per Category: More organized but introduces arbitrary boundaries between operation types. Rejected to avoid artificial categorization.
