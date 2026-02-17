/// XLA/PJRT CPU matmul adapter for benchmark harness.
const std = @import("std");
const pr = @import("../pr/pr.zig");
const stablehlo = @import("../lower/stablehlo.zig");
const backend_mod = @import("../backend/pjrt.zig");

/// XLA execution context (cached backend, device, and compiled executables).
pub const XlaContext = struct {
    allocator: std.mem.Allocator,
    backend: *backend_mod.Backend,
    device: *const backend_mod.Device,
    compiled_cache: std.StringHashMap(*backend_mod.LoadedExecutable),

    /// Initialize XLA context (loads PJRT CPU plugin, creates backend).
    pub fn init(allocator: std.mem.Allocator) !XlaContext {
        return init_impl(allocator, false);
    }

    /// Initialize XLA context with GPU (loads PJRT GPU plugin, creates backend).
    pub fn init_gpu(allocator: std.mem.Allocator) !XlaContext {
        return init_impl(allocator, true);
    }

    fn init_impl(allocator: std.mem.Allocator, use_gpu: bool) !XlaContext {
        const env_var = if (use_gpu) "PJRT_GPU_PLUGIN_PATH" else "PJRT_CPU_PLUGIN_PATH";
        const plugin_path = std.posix.getenv(env_var) orelse {
            if (use_gpu) return error.PjrtGpuPluginPathNotSet;
            return error.PjrtCpuPluginPathNotSet;
        };

        const backend = try allocator.create(backend_mod.Backend);
        errdefer allocator.destroy(backend);

        backend.* = try backend_mod.Backend.init(allocator, plugin_path);

        const devices = try backend.get_devices(allocator);
        if (devices.len == 0) return error.NoDevicesFound;

        return XlaContext{
            .allocator = allocator,
            .backend = backend,
            .device = &devices[0],
            .compiled_cache = std.StringHashMap(*backend_mod.LoadedExecutable).init(allocator),
        };
    }

    /// Release resources.
    pub fn deinit(self: *XlaContext) void {
        // Clean up cached executables
        var iter = self.compiled_cache.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.deinit(self.backend.api);
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.compiled_cache.deinit();

        self.backend.deinit();
        self.allocator.destroy(self.backend);
    }

    /// Execute XLA CPU matmul (compiles on-the-fly, caches per shape).
    pub fn execute(
        self: *XlaContext,
        m: usize,
        n: usize,
        k: usize,
        a: []const f32,
        b: []const f32,
        c: []f32,
    ) !void {
        // Check cache for compiled executable
        const cache_key = try std.fmt.allocPrint(self.allocator, "{d}x{d}x{d}", .{ m, n, k });
        defer self.allocator.free(cache_key);

        const executable = if (self.compiled_cache.get(cache_key)) |exe|
            exe
        else blk: {
            // Compile and cache
            const exe = try self.compile_matmul(m, n, k);
            const owned_key = try self.allocator.dupe(u8, cache_key);
            try self.compiled_cache.put(owned_key, exe);
            break :blk exe;
        };

        // Create device buffers
        const a_bytes = std.mem.sliceAsBytes(a);
        const b_bytes = std.mem.sliceAsBytes(b);

        const api = self.backend.api;

        var dev_a = try self.backend.buffer_from_host(self.device, a_bytes, .f32, &.{
            @intCast(m),
            @intCast(k),
        });
        defer dev_a.deinit(api);

        var dev_b = try self.backend.buffer_from_host(self.device, b_bytes, .f32, &.{
            @intCast(k),
            @intCast(n),
        });
        defer dev_b.deinit(api);

        // Execute
        const result = try executable.execute(api, self.allocator, &.{ dev_a, dev_b });
        defer {
            for (result.outputs) |*buf| buf.deinit(api);
            self.allocator.free(result.outputs);
        }

        // Wait for GPU kernel to complete before copying results
        if (result.device_complete_event) |ev| {
            var device_event = ev;
            defer device_event.deinit(api);
            try device_event.await_(api);
        }

        // Copy result back to c
        const c_bytes = std.mem.sliceAsBytes(c);
        var copy_event = try result.outputs[0].to_host(api, c_bytes);
        defer copy_event.deinit(api);
        try copy_event.await_(api);
    }

    /// Compile XLA matmul for a specific shape.
    fn compile_matmul(
        self: *XlaContext,
        m: usize,
        n: usize,
        k: usize,
    ) !*backend_mod.LoadedExecutable {
        // Build minimal PR program
        var program = pr.Program.init(self.allocator);
        defer program.deinit();

        var builder = try pr.FunctionBuilder.init(&program, "matmul");
        defer builder.deinit();

        const a_id = try builder.param_tensor(.f32, &.{ m, k });
        const b_id = try builder.param_tensor(.f32, &.{ k, n });
        const c_id = try builder.dot(a_id, b_id);
        const func = try builder.finish(&.{c_id});
        try program.add_function(func);

        // Lower to StableHLO
        const mlir_bytes = try stablehlo.lower_program_to_mlir(
            self.allocator,
            &program,
            null,
            .mlir_bytecode,
        );
        defer self.allocator.free(mlir_bytes);

        // Compile
        const executable = try self.allocator.create(backend_mod.LoadedExecutable);
        errdefer self.allocator.destroy(executable);

        executable.* = try self.backend.compile(
            self.device,
            mlir_bytes,
            true,
            .{},
        );

        return executable;
    }
};
