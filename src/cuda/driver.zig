//! CUDA driver modules and kernel launch.

const cuda_abi = @import("../c/cuda/driver.zig");

/// Errors raised while loading modules, resolving functions, or launching kernels.
pub const Error = cuda_abi.Error;
pub const ComputeCapability = cuda_abi.ComputeCapability;

/// Query the compute capability of one CUDA device ordinal.
pub fn compute_capability(device_ordinal: i32) Error!ComputeCapability {
    return try cuda_abi.device_compute_capability(device_ordinal);
}

/// Kernel launch dimensions, dynamic memory, stream, and arguments.
pub const LaunchOptions = struct {
    /// Grid dimensions in x, y, z order.
    grid_dim: [3]u32,

    /// Block dimensions in x, y, z order.
    block_dim: [3]u32,

    /// Dynamic shared memory requested for the launch.
    shared_memory_bytes: u32 = 0,

    /// Optional CUDA stream borrowed for the launch.
    stream: ?*anyopaque = null,

    /// Pointers to stable storage containing each kernel argument value.
    params: []?*anyopaque,
};

/// Loaded CUDA module released with `deinit`.
pub const Module = opaque {
    /// Loads PTX or a CUDA binary image.
    pub fn load(image: []const u8) Error!*Module {
        return @ptrCast(try cuda_abi.ModuleHandle.load(image));
    }

    /// Unloads this module and invalidates its borrowed functions.
    pub fn deinit(self: *Module) void {
        const handle: *cuda_abi.ModuleHandle = @ptrCast(self);
        handle.unload();
    }

    /// Resolves a function borrowed from this module.
    pub fn function(self: *Module, name: [:0]const u8) Error!*Function {
        const handle: *cuda_abi.ModuleHandle = @ptrCast(self);
        return @ptrCast(try handle.function(name));
    }
};

/// CUDA function borrowed from a loaded module.
pub const Function = opaque {
    /// Raises the function's dynamic shared-memory limit.
    pub fn set_max_dynamic_shared_memory(self: *Function, bytes: u32) Error!void {
        const handle: *cuda_abi.FunctionHandle = @ptrCast(self);
        try handle.set_max_dynamic_shared_memory(bytes);
    }

    /// Launches the function with explicit dimensions and arguments.
    pub fn launch(self: *Function, options: LaunchOptions) Error!void {
        const handle: *cuda_abi.FunctionHandle = @ptrCast(self);
        try handle.launch(.{
            .grid_dim = options.grid_dim,
            .block_dim = options.block_dim,
            .shared_memory_bytes = options.shared_memory_bytes,
            .stream = options.stream,
            .params = options.params,
        });
    }
};
