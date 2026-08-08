//! XLA/PJRT matmul adapter for benchmark harness.
const std = @import("std");
const zg = @import("zigrad");
const pr = zg.pr;
const pjrt = zg.pjrt;
const config = @import("config.zig");

const DeviceKind = @import("harness.zig").DeviceKind;

/// XLA execution context with a cache of compiled matmul programs.
pub const XlaContext = struct {
    allocator: std.mem.Allocator,
    compilation: zg.compilation.Context,
    client: *pjrt.Client,
    execution: *pjrt.Execution,
    backend: pjrt.Backend,
    compiled_cache: std.StringHashMap(zg.Executor.LoadedProgram),

    pub fn init(io: std.Io, allocator: std.mem.Allocator, device: DeviceKind) !XlaContext {
        const env_var: [*:0]const u8 = switch (device) {
            .cpu => "PJRT_CPU_PLUGIN_PATH",
            .gpu => "PJRT_GPU_PLUGIN_PATH",
        };
        const env_ptr = std.c.getenv(env_var) orelse return switch (device) {
            .cpu => error.PjrtCpuPluginPathNotSet,
            .gpu => error.PjrtGpuPluginPathNotSet,
        };
        const plugin_path = std.mem.span(env_ptr);

        const client = try allocator.create(pjrt.Client);
        errdefer allocator.destroy(client);

        client.* = try pjrt.Client.init(allocator, plugin_path, .{});
        errdefer client.deinit();

        const devices = try client.get_devices(allocator);
        defer allocator.free(devices);
        if (devices.len == 0) return error.NoDevicesFound;

        const execution = try allocator.create(pjrt.Execution);
        errdefer allocator.destroy(execution);
        execution.* = try pjrt.Execution.init(client, devices[0], .{});

        return .{
            .allocator = allocator,
            .compilation = .{
                .allocator = allocator,
                .io = io,
                .device = execution.interface.device,
            },
            .client = client,
            .execution = execution,
            .backend = pjrt.Backend.init(execution, .{}),
            .compiled_cache = std.StringHashMap(zg.Executor.LoadedProgram).init(allocator),
        };
    }

    pub fn deinit(self: *XlaContext) void {
        var iter = self.compiled_cache.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.compiled_cache.deinit();

        self.allocator.destroy(self.execution);
        self.client.deinit();
        self.allocator.destroy(self.client);
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
        const cache_key = try std.fmt.allocPrint(self.allocator, "{d}x{d}x{d}_{t}", .{
            shape.m, shape.n, shape.k, dtype,
        });
        defer self.allocator.free(cache_key);

        const executable = if (self.compiled_cache.get(cache_key)) |exe|
            exe
        else blk: {
            var exe = try self.compile_matmul(shape, dtype);
            errdefer exe.deinit();
            const owned_key = try self.allocator.dupe(u8, cache_key);
            errdefer self.allocator.free(owned_key);
            try self.compiled_cache.put(owned_key, exe);
            break :blk exe;
        };

        var shape_a = [_]i64{ shape.m, shape.k };
        var shape_b = [_]i64{ shape.k, shape.n };

        const executor = &self.execution.interface;
        const dev_a = try executor.upload(std.mem.sliceAsBytes(a), dtype.to_pr_dtype(), &shape_a);
        defer executor.release(dev_a);

        const dev_b = try executor.upload(std.mem.sliceAsBytes(b), dtype.to_pr_dtype(), &shape_b);
        defer executor.release(dev_b);

        var outputs: [1]zg.Executor.Buffer = undefined;
        const execute_event = try executor.invoke(executable, &.{ dev_a, dev_b }, &outputs, .{});
        defer executor.release(outputs[0]);
        if (execute_event) |event| {
            defer executor.release_event(event);
            try executor.wait(event);
        }

        if (try executor.download(outputs[0], std.mem.sliceAsBytes(c))) |event| {
            defer executor.release_event(event);
            try executor.wait(event);
        }
    }

    fn compile_matmul(
        self: *XlaContext,
        shape: config.Shape,
        dtype: config.DType,
    ) !zg.Executor.LoadedProgram {
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

        var pipeline = try pjrt.pipeline.create(self.allocator, &self.backend, .{
            .stablehlo = .{ .entry_name = "matmul" },
        });
        defer pipeline.deinit();
        return try pipeline.run(
            zg.Executor.LoadedProgram,
            &program,
            &self.compilation,
        );
    }
};
