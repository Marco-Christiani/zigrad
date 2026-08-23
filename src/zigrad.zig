//! Zigrad public API.
//!
//! Integration-free tracing produces PR programs. Optional integrations
//!  expose explicit compilation and execution capabilities.
const std = @import("std");

pub const build_options = @import("build_options");

pub const pr = @import("pr.zig");
pub const kernel = @import("kernel.zig");
pub const device = @import("device.zig");
pub const dtype = @import("dtype.zig");
pub const compilation = @import("compilation.zig");
pub const toolchain = @import("toolchain.zig");
pub const backend = @import("backend.zig");
pub const output = @import("output.zig");
pub const callable = @import("callable.zig");
pub const transforms = @import("transforms.zig");
pub const optim = @import("optim.zig");
pub const train = @import("train.zig");
pub const tune = @import("tune.zig");
pub const Cache = @import("cache.zig").Cache;
pub const RuntimeEnv = @import("runtime.zig").RuntimeEnv;
pub const RuntimeLibrary = @import("runtime.zig").RuntimeLibrary;
pub const utils = @import("utils.zig");

/// Optional CUDA driver and runtime compilation capabilities.
pub const cuda = if (build_options.has_nvrtc or build_options.has_cuda_runtime)
    @import("cuda.zig")
else
    struct {};

/// Optional MLIR integration infrastructure.
pub const mlir = if (build_options.has_mlir) @import("mlir.zig") else struct {};

/// Optional StableHLO artifacts used by current compiler integrations.
pub const stablehlo = if (build_options.has_mlir or build_options.has_pjrt or build_options.has_iree)
    @import("stablehlo.zig")
else
    struct {};

/// Optional TVM integration.
pub const tvm = if (build_options.has_tvm) @import("tvm.zig") else struct {};

/// Optional Mirage kernel-provider integration.
pub const mirage = if (build_options.has_mirage) @import("mirage.zig") else struct {};

pub const Tensor = @import("tensor.zig");
/// Mutable queue for composing compiler operations.
pub const Pipeline = compilation.Pipeline;
/// Services supplied while compiler operations run.
pub const CompilationCtx = compilation.Context;
pub const Device = device.Device;
pub const Platform = device.Platform;
pub const Executor = @import("execution.zig");
pub const Backend = backend.Backend;
pub const HostBuffer = utils.HostBuffer;
pub const DType = dtype.DType;
pub const Shape = pr.Shape;
pub const BoundedShape = pr.BoundedShape;

/// Optional PJRT integration.
pub const pjrt = if (build_options.has_pjrt) @import("pjrt.zig") else struct {};

/// Optional IREE integration infrastructure.
pub const iree = if (build_options.has_iree) @import("iree.zig") else struct {};

pub const trace = @import("trace.zig").trace;
pub const trace_callable = callable.trace_callable;
pub const grad = transforms.make_grad;
pub const value_and_grad = transforms.make_value_and_grad;

pub const from_safetensors = utils.safetensors.from_safetensors;
pub const to_safetensors = utils.safetensors.to_safetensors;
pub const FromSafetensorsOpts = utils.safetensors.Opts;
pub const SafetensorsFile = utils.safetensors.SafeTensorsFile;

test {
    @setEvalBranchQuota(10000);
    std.testing.refAllDecls(@This());
    _ = @import("tests/higher_order_ad.zig");
}
