//! Compiled matmul adapters used by the benchmark harness.

const std = @import("std");
const zg = @import("zigrad");
const config = @import("config.zig");

const DeviceKind = config.DeviceKind;
const Backend = zg.Backend(zg.stablehlo.Artifact);

const CompiledContext = struct {
    allocator: std.mem.Allocator,
    compilation: zg.CompilationCtx,
    executor: *zg.Executor,
    backend: *Backend,
    cache: std.StringHashMap(zg.Executor.LoadedProgram),

    fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        executor: *zg.Executor,
        backend: *Backend,
    ) CompiledContext {
        return .{
            .allocator = allocator,
            .compilation = .{
                .allocator = allocator,
                .io = io,
                .device = executor.device,
            },
            .executor = executor,
            .backend = backend,
            .cache = .init(allocator),
        };
    }

    fn deinit(self: *CompiledContext) void {
        var iter = self.cache.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.cache.deinit();
    }

    fn execute(
        self: *CompiledContext,
        comptime T: type,
        shape: config.Shape,
        a: []const T,
        b: []const T,
        c: []T,
    ) !void {
        const dtype = comptime config.DType.from_zig_type(T);
        var cache_key_buffer: [96]u8 = undefined;
        const cache_key = try std.fmt.bufPrint(&cache_key_buffer, "{d}x{d}x{d}_{t}", .{
            shape.m, shape.n, shape.k, dtype,
        });

        const executable = if (self.cache.get(cache_key)) |exe|
            exe
        else blk: {
            var exe = try self.compile_matmul(shape, dtype);
            errdefer exe.deinit();
            const owned_key = try self.allocator.dupe(u8, cache_key);
            errdefer self.allocator.free(owned_key);
            try self.cache.put(owned_key, exe);
            break :blk exe;
        };

        const shape_a = [_]i64{ shape.m, shape.k };
        const shape_b = [_]i64{ shape.k, shape.n };
        const dev_a = try self.executor.upload(
            std.mem.sliceAsBytes(a),
            dtype.to_pr_dtype(),
            &shape_a,
        );
        defer self.executor.release(dev_a);
        const dev_b = try self.executor.upload(
            std.mem.sliceAsBytes(b),
            dtype.to_pr_dtype(),
            &shape_b,
        );
        defer self.executor.release(dev_b);

        var outputs: [1]zg.Executor.Buffer = undefined;
        const execute_event = try self.executor.invoke(
            executable,
            &.{ dev_a, dev_b },
            &outputs,
            .{},
        );
        defer self.executor.release(outputs[0]);
        if (execute_event) |event| {
            defer self.executor.release_event(event);
            try self.executor.wait(event);
        }

        if (try self.executor.download(outputs[0], std.mem.sliceAsBytes(c))) |event| {
            defer self.executor.release_event(event);
            try self.executor.wait(event);
        }
    }

    fn compile_matmul(
        self: *CompiledContext,
        shape: config.Shape,
        dtype: config.DType,
    ) !zg.Executor.LoadedProgram {
        var program = zg.pr.Program.init(self.allocator);
        defer program.deinit();

        var builder = try zg.pr.FunctionBuilder.init(&program, "matmul");
        defer builder.deinit();
        const pr_dtype = dtype.to_pr_dtype();
        const a = try builder.param_tensor(pr_dtype, &.{ shape.m, shape.k });
        const b = try builder.param_tensor(pr_dtype, &.{ shape.k, shape.n });
        const result = try builder.mm(a, b);
        const function = try builder.finish(&.{result});
        _ = try program.add_function(function);

        var pipeline = zg.Pipeline.init(self.allocator);
        defer pipeline.deinit();
        try zg.mlir.stablehlo.pipeline.add(&pipeline, .{ .entry_name = "matmul" });
        try pipeline.add(self.backend);
        return try pipeline.run(
            zg.Executor.LoadedProgram,
            &program,
            &self.compilation,
        );
    }
};

/// XLA/PJRT execution state for one selected device class.
pub const XlaContext = struct {
    allocator: std.mem.Allocator,
    client: *zg.pjrt.Client,
    execution: *zg.pjrt.Execution,
    backend: *zg.pjrt.Backend,
    compiled: CompiledContext,

    pub fn init(
        io: std.Io,
        environ: *const std.process.Environ.Map,
        allocator: std.mem.Allocator,
        device: DeviceKind,
    ) !XlaContext {
        const env_var = switch (device) {
            .cpu => "PJRT_CPU_PLUGIN_PATH",
            .gpu => "PJRT_GPU_PLUGIN_PATH",
        };
        const plugin_path = environ.get(env_var) orelse return switch (device) {
            .cpu => error.PjrtCpuPluginPathNotSet,
            .gpu => error.PjrtGpuPluginPathNotSet,
        };

        const client = try allocator.create(zg.pjrt.Client);
        errdefer allocator.destroy(client);
        client.* = try zg.pjrt.Client.init(allocator, plugin_path, .{});
        errdefer client.deinit();

        const devices = try client.get_devices(allocator);
        defer allocator.free(devices);
        if (devices.len == 0) return error.NoDevicesFound;

        const execution = try allocator.create(zg.pjrt.Execution);
        errdefer allocator.destroy(execution);
        execution.* = try zg.pjrt.Execution.init(client, devices[0], .{});

        const backend = try allocator.create(zg.pjrt.Backend);
        errdefer allocator.destroy(backend);
        backend.* = .init(execution, .{});

        return .{
            .allocator = allocator,
            .client = client,
            .execution = execution,
            .backend = backend,
            .compiled = .init(io, allocator, &execution.interface, &backend.interface),
        };
    }

    pub fn deinit(self: *XlaContext) void {
        self.compiled.deinit();
        self.allocator.destroy(self.backend);
        self.allocator.destroy(self.execution);
        self.client.deinit();
        self.allocator.destroy(self.client);
    }

    pub fn execute(
        self: *XlaContext,
        comptime T: type,
        shape: config.Shape,
        a: []const T,
        b: []const T,
        c: []T,
    ) !void {
        return try self.compiled.execute(T, shape, a, b, c);
    }
};

/// IREE compilation and execution state.
pub const IreeContext = struct {
    allocator: std.mem.Allocator,
    runtime: *zg.iree.Runtime,
    execution: *zg.iree.Execution,
    backend: *zg.iree.Backend,
    compiled: CompiledContext,

    pub fn init(
        io: std.Io,
        environ: *const std.process.Environ.Map,
        allocator: std.mem.Allocator,
    ) !IreeContext {
        const selected = zg.iree.Config.from_environ(environ);
        const runtime = try allocator.create(zg.iree.Runtime);
        errdefer allocator.destroy(runtime);
        runtime.* = try zg.iree.Runtime.init(allocator, .registered, selected.runtime);
        errdefer runtime.deinit();

        const execution = try allocator.create(zg.iree.Execution);
        errdefer allocator.destroy(execution);
        execution.* = .init(allocator, runtime, selected.runtime);

        const backend = try allocator.create(zg.iree.Backend);
        errdefer allocator.destroy(backend);
        backend.* = .init(execution, selected.compiler, "module.main");

        return .{
            .allocator = allocator,
            .runtime = runtime,
            .execution = execution,
            .backend = backend,
            .compiled = .init(io, allocator, &execution.interface, &backend.interface),
        };
    }

    pub fn deinit(self: *IreeContext) void {
        self.compiled.deinit();
        self.allocator.destroy(self.backend);
        self.allocator.destroy(self.execution);
        self.runtime.deinit();
        self.allocator.destroy(self.runtime);
    }

    pub fn execute(
        self: *IreeContext,
        comptime T: type,
        shape: config.Shape,
        a: []const T,
        b: []const T,
        c: []T,
    ) !void {
        return try self.compiled.execute(T, shape, a, b, c);
    }
};
