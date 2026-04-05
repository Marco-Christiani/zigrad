//! XLA/PJRT matmul adapter for benchmark harness.
const std = @import("std");
const zg = @import("zigrad");
const pr = zg.pr;
const lower = zg.lower;
const backend = zg.backend;
const config = @import("config.zig");

const DeviceKind = @import("harness.zig").DeviceKind;

/// XLA execution context (cached backend, device, and compiled executables).
pub const XlaContext = struct {
    allocator: std.mem.Allocator,
    backend_handle: *backend.pjrt.Backend,
    device: *const backend.pjrt.Device,
    compiled_cache: std.StringHashMap(*backend.pjrt.LoadedExecutable),

    pub fn init(allocator: std.mem.Allocator, device: DeviceKind) !XlaContext {
        const env_var = switch (device) {
            .cpu => "PJRT_CPU_PLUGIN_PATH",
            .gpu => "PJRT_GPU_PLUGIN_PATH",
        };
        const plugin_path = std.posix.getenv(env_var) orelse return switch (device) {
            .cpu => error.PjrtCpuPluginPathNotSet,
            .gpu => error.PjrtGpuPluginPathNotSet,
        };

        const backend_handle = try allocator.create(backend.pjrt.Backend);
        errdefer allocator.destroy(backend_handle);

        backend_handle.* = try backend.pjrt.Backend.init(allocator, plugin_path);

        const devices = try backend_handle.get_devices(allocator);
        if (devices.len == 0) return error.NoDevicesFound;

        return XlaContext{
            .allocator = allocator,
            .backend_handle = backend_handle,
            .device = &devices[0],
            .compiled_cache = std.StringHashMap(*backend.pjrt.LoadedExecutable).init(allocator),
        };
    }

    pub fn deinit(self: *XlaContext) void {
        var iter = self.compiled_cache.iterator();
        while (iter.next()) |entry| {
            self.backend_handle.deinit_executable(entry.value_ptr.*);
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.compiled_cache.deinit();

        self.backend_handle.deinit();
        self.allocator.destroy(self.backend_handle);
    }

    /// Execute XLA matmul (compiles on-the-fly, caches per shape+dtype).
    pub fn execute(
        self: *XlaContext,
        comptime T: type,
        shape: config.Shape,
        a: []const T,
        b: []const T,
        c: []T,
    ) !void {
        const dtype = comptime config.DType.from_zig_type(T);
        const cache_key = try std.fmt.allocPrint(self.allocator, "{d}x{d}x{d}_{s}", .{
            shape.m, shape.n, shape.k, @tagName(dtype),
        });
        defer self.allocator.free(cache_key);

        const executable = if (self.compiled_cache.get(cache_key)) |exe|
            exe
        else blk: {
            const exe = try self.compile_matmul(shape, dtype);
            const owned_key = try self.allocator.dupe(u8, cache_key);
            try self.compiled_cache.put(owned_key, exe);
            break :blk exe;
        };

        var shape_a = [_]i64{ shape.m, shape.k };
        var shape_b = [_]i64{ shape.k, shape.n };

        var dev_a = try self.backend_handle.buffer_from_host(self.device, std.mem.sliceAsBytes(a), dtype.to_pr_dtype(), &shape_a);
        defer self.backend_handle.deinit_buffer(&dev_a);

        var dev_b = try self.backend_handle.buffer_from_host(self.device, std.mem.sliceAsBytes(b), dtype.to_pr_dtype(), &shape_b);
        defer self.backend_handle.deinit_buffer(&dev_b);

        const result = try self.backend_handle.execute(executable, self.allocator, &.{ dev_a, dev_b }, .{});
        defer {
            for (result.outputs) |*buf| self.backend_handle.deinit_buffer(buf);
            self.allocator.free(result.outputs);
        }

        if (result.device_complete_event) |ev| {
            var device_event = ev;
            defer self.backend_handle.deinit_event(&device_event);
            try self.backend_handle.await_event(&device_event);
        }

        const c_bytes = std.mem.sliceAsBytes(c);
        var copy_event = try self.backend_handle.buffer_to_host(&result.outputs[0], c_bytes);
        defer self.backend_handle.deinit_event(&copy_event);
        try self.backend_handle.await_event(&copy_event);
    }

    fn compile_matmul(
        self: *XlaContext,
        shape: config.Shape,
        dtype: config.DType,
    ) !*backend.pjrt.LoadedExecutable {
        var program = pr.Program.init(self.allocator);
        defer program.deinit();

        var builder = try pr.FunctionBuilder.init(&program, "matmul");
        defer builder.deinit();

        const pr_dtype = dtype.to_pr_dtype();
        const a_id = try builder.param_tensor(pr_dtype, &.{ shape.m, shape.k });
        const b_id = try builder.param_tensor(pr_dtype, &.{ shape.k, shape.n });
        const c_id = try builder.dot(a_id, b_id);
        const func = try builder.finish(&.{c_id});
        try program.add_function(func);

        const mlir_bytes = try lower.lower_program_to_mlir(
            self.allocator,
            &program,
            null,
            .mlir_bytecode,
        );
        defer self.allocator.free(mlir_bytes);

        const executable = try self.allocator.create(backend.pjrt.LoadedExecutable);
        errdefer self.allocator.destroy(executable);

        executable.* = try self.backend_handle.compile(
            self.device,
            mlir_bytes,
            true,
            .{},
        );

        return executable;
    }
};
