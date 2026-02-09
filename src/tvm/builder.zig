//! TVM module building strategies for different targets.
//!
//! Provides target-specific compilation strategies:
//! - CPU: Direct LLVM build
//! - CUDA: Host/device split + link

const std = @import("std");
const c = @import("../ffi/tvm.zig");
const common = @import("common.zig");
const cuda = @import("cuda.zig");
const ffi = @import("../tvm_runtime.zig"); // For FFI helpers

const log = std.log.scoped(.@"zg/tvm_builder");

/// Build a lowered TIR module for CPU execution.
///
/// Takes a module that has been through the complete lowering pipeline
/// (including MakePackedAPI) and compiles it with target.build.llvm.
pub fn build_cpu_module(
    allocator: std.mem.Allocator,
    lowered_mod: c.TVMFFIAny,
    target: c.TVMFFIAny,
) !c.TVMFFIAny {
    var built_mod: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    var build_args = [_]c.TVMFFIAny{ lowered_mod, target };
    try ffi.ffi_call_global(allocator, "target.build.llvm", &build_args, &built_mod);
    log.debug("Built CPU module: type_index={d}", .{built_mod.type_index});
    return built_mod;
}

/// Build a lowered TIR module for CUDA execution.
///
/// After SplitHostDevice+MakePackedAPI the module contains:
/// - Host wrapper functions (CallingConv::kCPackedFunc)
/// - Device kernel functions (CallingConv::kDeviceKernelLaunch)
///
/// This function:
/// 1. Filters to device functions only
/// 2. Builds device kernels with target.build.cuda
/// 3. Filters to host functions only
/// 4. Builds host wrapper with target.build.llvm
/// 5. Links device module into host module
pub fn build_cuda_module(
    allocator: std.mem.Allocator,
    lowered_mod: c.TVMFFIAny,
    target: c.TVMFFIAny,
) !c.TVMFFIAny {
    log.info("Starting CUDA module build (host/device split)", .{});

    // Create separate host (LLVM) and device (CUDA) targets
    var host_target: c.TVMFFIAny = undefined; // NOTE: audit use of undefined
    {
        const host_str = try ffi.cstr_alloc(allocator, "llvm"); // NOTE: why heap? if TVM isnt freeing it, should this be stack? also, cstr_ptr just casts to a null terminated string, we can just initialize it as one instead.
        defer allocator.free(host_str);
        var args = [_]c.TVMFFIAny{ffi.any_raw_str(ffi.cstr_ptr(host_str))};
        try ffi.ffi_call_global(allocator, "target.Target", &args, &host_target);
    }
    defer if (host_target.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };

    // 1. Filter to device functions (calling_conv != 1)
    log.info("Step 1: Filtering device functions", .{});
    const device_mod = cuda.filter_module_by_target(allocator, lowered_mod, .device) catch |err| {
        log.err("Failed to filter device functions: {s}", .{@errorName(err)});
        return err;
    };
    defer if (device_mod.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };
    log.info("Step 1 done: device_mod.type_index={d}", .{device_mod.type_index});

    // 2. Build device kernels with CUDA target
    log.info("Step 2: Building device kernels (NVRTC)", .{});
    const device_built = cuda.build_device_kernels(allocator, device_mod, target) catch |err| {
        log.err("Failed to build device kernels: {s}", .{@errorName(err)});
        return err;
    };
    log.info("Step 2 done: device_built.type_index={d}", .{device_built.type_index});

    // 3. Filter to host functions (calling_conv == 1)
    log.info("Step 3: Filtering host functions", .{});
    const host_mod = cuda.filter_module_by_target(allocator, lowered_mod, .host) catch |err| {
        log.err("Failed to filter host functions: {s}", .{@errorName(err)});
        return err;
    };
    defer if (host_mod.unnamed_1.v_obj) |obj| {
        _ = c.TVMFFIObjectDecRef(@ptrCast(obj));
    };
    log.info("Step 3 done: host_mod.type_index={d}", .{host_mod.type_index});

    // 4. Build host wrapper with LLVM target
    log.info("Step 4: Building host wrapper (LLVM)", .{});
    const host_built = cuda.build_host_wrapper(allocator, host_mod, host_target) catch |err| {  // NOTE: should document lifetime in fn docstring
        log.err("Failed to build host wrapper: {s}", .{@errorName(err)});
        return err;
    };
    log.info("Step 4 done: host_built.type_index={d}", .{host_built.type_index});

    // 5. Link device module into host
    log.info("Step 5: Linking device module into host", .{});
    cuda.link_device_module(allocator, host_built, device_built) catch |err| {
        log.err("Failed to link device module: {s}", .{@errorName(err)});
        return err;
    };

    log.info("CUDA module build complete: type_index={d}", .{host_built.type_index});
    return host_built;
}

/// Build a lowered TIR module using the appropriate target-specific strategy.
///
/// Delegates to build_cpu_module or build_cuda_module based on target_kind.
pub fn build_module(
    allocator: std.mem.Allocator,
    lowered_mod: c.TVMFFIAny,
    target: c.TVMFFIAny,
    target_kind: common.TargetKind,
) !c.TVMFFIAny {
    return switch (target_kind) {
        .cpu => try build_cpu_module(allocator, lowered_mod, target),
        .cuda => try build_cuda_module(allocator, lowered_mod, target),
    };
}
