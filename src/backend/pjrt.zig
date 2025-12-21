/// PJRT Backend Implementation
///
/// Implements backend.zig interface using PJRT.
/// This module connects generic Backend interface to specific PJRT types.

const std = @import("std");
const backend = @import("backend.zig");
const plugin_mod = @import("../pjrt/plugin.zig");
const pjrt_types = @import("../pjrt/types.zig");
const pjrt_api = @import("../pjrt/api.zig");

const Backend = backend.Backend;
const Device = backend.Device;
const Executable = backend.Executable;
const ExecuteResult = backend.ExecuteResult;
const Buffer = backend.Buffer;
const Event = backend.Event;
const CompileOptions = backend.CompileOptions;

pub const PjrtBackend = struct {
    api: pjrt_api.Api,
    client: pjrt_types.Client,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, plugin_path: []const u8) !Backend {
        // Load PJRT plugin
        const api = try plugin_mod.loadPlugin(plugin_path);
        errdefer plugin_mod.unloadPlugin(api);

        // Allocate backend state first
        const self = try allocator.create(PjrtBackend);
        self.* = PjrtBackend{
            .api = api,
            .client = undefined, // will be initialized below
            .allocator = allocator,
        };
        errdefer allocator.destroy(self);

        // Create PJRT client with stable pointer to self.api
        self.client = try pjrt_types.Client.create(&self.api);
        errdefer self.client.deinit();

        return Backend{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    const vtable = Backend.VTable{
        .deinit = deinitImpl,
        .getDevices = getDevicesImpl,
        .compile = compileImpl,
        .bufferFromHost = bufferFromHostImpl,
    };

    fn deinitImpl(ptr: *anyopaque) void {
        const self: *PjrtBackend = @ptrCast(@alignCast(ptr));
        self.client.deinit();
        plugin_mod.unloadPlugin(self.api);
        self.allocator.destroy(self);
    }

    fn getDevicesImpl(ptr: *anyopaque, allocator: std.mem.Allocator) backend.Error![]Device {
        const self: *PjrtBackend = @ptrCast(@alignCast(ptr));

        const pjrt_devices = self.client.getDevices(allocator) catch return backend.Error.BackendInitFailed;
        defer allocator.free(pjrt_devices);

        const devices = try allocator.alloc(Device, pjrt_devices.len);
        for (pjrt_devices, 0..) |pjrt_dev, i| {
            // Allocate device wrapper
            const dev_wrapper = try allocator.create(PjrtDeviceWrapper);
            dev_wrapper.* = PjrtDeviceWrapper{
                .api = &self.api,
                .pjrt_device = pjrt_dev,
                .allocator = allocator,
            };

            devices[i] = Device{
                .ptr = dev_wrapper,
                .vtable = &PjrtDeviceWrapper.vtable,
            };
        }

        return devices;
    }

    fn compileImpl(ptr: *anyopaque, device: *const Device, options: CompileOptions) backend.Error!Executable {
        const self: *PjrtBackend = @ptrCast(@alignCast(ptr));
        const dev_wrapper: *PjrtDeviceWrapper = @ptrCast(@alignCast(device.ptr));

        // Convert format
        const format: pjrt_types.ProgramFormat = switch (options.format) {
            .stablehlo_mlir_text, .mlir_text => .mlir_text,
            .stablehlo_mlir_bytecode, .mlir_bytecode, .stablehlo_portable => .mlir_bytecode,
        };

        // Compile
        const pjrt_exec = self.client.compile(
            &dev_wrapper.pjrt_device,
            format,
            options.bytecode,
            options.backend_options,
        ) catch return backend.Error.CompileFailed;

        // Wrap in executable
        const exec_wrapper = try self.allocator.create(PjrtExecutableWrapper);
        exec_wrapper.* = PjrtExecutableWrapper{
            .pjrt_executable = pjrt_exec,
            .allocator = self.allocator,
        };

        return Executable{
            .ptr = exec_wrapper,
            .vtable = &PjrtExecutableWrapper.vtable,
        };
    }

    fn bufferFromHostImpl(
        ptr: *anyopaque,
        device: *const Device,
        data: []const u8,
        dtype: backend.DType,
        shape: backend.Shape,
    ) backend.Error!Buffer {
        const self: *PjrtBackend = @ptrCast(@alignCast(ptr));
        const dev_wrapper: *PjrtDeviceWrapper = @ptrCast(@alignCast(device.ptr));

        // Convert dtype
        const pjrt_dtype: pjrt_types.BufferType = switch (dtype) {
            .f32 => .f32,
            .f64 => .f64,
            .i32 => .i32,
            .i64 => .i64,
            .u32 => .u32,
            .u64 => .u64,
        };

        // Convert shape to i64 array
        const i64_shape = try self.allocator.alloc(i64, shape.dims.len);
        defer self.allocator.free(i64_shape);
        for (shape.dims, 0..) |dim, i| {
            i64_shape[i] = @intCast(dim);
        }

        // Create buffer
        const pjrt_buffer = self.client.bufferFromHost(
            &dev_wrapper.pjrt_device,
            data,
            pjrt_dtype,
            i64_shape,
        ) catch return backend.Error.BufferTransferFailed;

        // Wrap buffer
        const buf_wrapper = try self.allocator.create(PjrtBufferWrapper);
        buf_wrapper.* = PjrtBufferWrapper{
            .pjrt_buffer = pjrt_buffer,
            .dtype = dtype,
            .shape_dims = try self.allocator.dupe(usize, shape.dims),
            .allocator = self.allocator,
        };

        return Buffer{
            .ptr = buf_wrapper,
            .vtable = &PjrtBufferWrapper.vtable,
        };
    }
};

const PjrtDeviceWrapper = struct {
    api: *pjrt_api.Api,
    pjrt_device: pjrt_types.Device,
    allocator: std.mem.Allocator,

    const vtable = Device.VTable{
        .deinit = deinitImpl,
        .getId = getIdImpl,
        .getName = getNameImpl,
        .getKind = getKindImpl,
    };

    fn deinitImpl(ptr: *anyopaque) void {
        const self: *PjrtDeviceWrapper = @ptrCast(@alignCast(ptr));
        self.allocator.destroy(self);
    }

    fn getIdImpl(ptr: *anyopaque) backend.Error!u32 {
        const self: *PjrtDeviceWrapper = @ptrCast(@alignCast(ptr));
        const id = self.pjrt_device.getId(self.api) catch return backend.Error.Internal;
        return @intCast(id);
    }

    fn getNameImpl(ptr: *anyopaque, allocator: std.mem.Allocator) backend.Error![]const u8 {
        const self: *PjrtDeviceWrapper = @ptrCast(@alignCast(ptr));
        const kind = self.pjrt_device.getKind(self.api) catch return backend.Error.Internal;
        return allocator.dupe(u8, kind) catch return backend.Error.OutOfMemory;
    }

    fn getKindImpl(ptr: *anyopaque) Device.DeviceKind {
        const self: *PjrtDeviceWrapper = @ptrCast(@alignCast(ptr));
        const kind_str = self.pjrt_device.getKind(self.api) catch return .custom;

        if (std.mem.eql(u8, kind_str, "cpu")) return .cpu;
        if (std.mem.eql(u8, kind_str, "gpu") or std.mem.eql(u8, kind_str, "cuda")) return .cuda;
        if (std.mem.eql(u8, kind_str, "rocm")) return .rocm;
        if (std.mem.eql(u8, kind_str, "tpu")) return .tpu;
        return .custom;
    }
};

const PjrtExecutableWrapper = struct {
    pjrt_executable: pjrt_types.LoadedExecutable,
    allocator: std.mem.Allocator,

    const vtable = Executable.VTable{
        .deinit = deinitImpl,
        .execute = executeImpl,
        .serialize = null, // TODO: implement serialization
    };

    fn deinitImpl(ptr: *anyopaque) void {
        const self: *PjrtExecutableWrapper = @ptrCast(@alignCast(ptr));
        self.pjrt_executable.deinit();
        self.allocator.destroy(self);
    }

    fn executeImpl(ptr: *anyopaque, inputs: []const Buffer, allocator: std.mem.Allocator) backend.Error!ExecuteResult {
        const self: *PjrtExecutableWrapper = @ptrCast(@alignCast(ptr));

        // Unwrap input buffers
        const pjrt_inputs = try allocator.alloc(pjrt_types.Buffer, inputs.len);
        defer allocator.free(pjrt_inputs);

        for (inputs, 0..) |input, i| {
            const buf_wrapper: *PjrtBufferWrapper = @ptrCast(@alignCast(input.ptr));
            pjrt_inputs[i] = buf_wrapper.pjrt_buffer;
        }

        // Execute
        const pjrt_outputs = self.pjrt_executable.execute(allocator, pjrt_inputs) catch return backend.Error.ExecuteFailed;
        defer allocator.free(pjrt_outputs);

        // Wrap outputs
        const outputs = try allocator.alloc(Buffer, pjrt_outputs.len);
        for (pjrt_outputs, 0..) |pjrt_out, i| {
            // Get buffer dimensions
            const shape_dims = pjrt_out.getDimensions(allocator) catch &[_]usize{};

            const buf_wrapper = try allocator.create(PjrtBufferWrapper);
            buf_wrapper.* = PjrtBufferWrapper{
                .pjrt_buffer = pjrt_out,
                .dtype = .f32, // TODO: infer from buffer metadata
                .shape_dims = shape_dims,
                .allocator = allocator,
            };

            outputs[i] = Buffer{
                .ptr = buf_wrapper,
                .vtable = &PjrtBufferWrapper.vtable,
            };
        }

        return ExecuteResult{
            .outputs = outputs,
            .event = null, // TODO: return execution event
        };
    }
};

const PjrtBufferWrapper = struct {
    pjrt_buffer: pjrt_types.Buffer,
    dtype: backend.DType,
    shape_dims: []const usize,
    allocator: std.mem.Allocator,

    const vtable = Buffer.VTable{
        .deinit = deinitImpl,
        .getShape = getShapeImpl,
        .getDtype = getDtypeImpl,
        .toHost = toHostImpl,
    };

    fn deinitImpl(ptr: *anyopaque) void {
        const self: *PjrtBufferWrapper = @ptrCast(@alignCast(ptr));
        self.pjrt_buffer.deinit();
        self.allocator.free(self.shape_dims);
        self.allocator.destroy(self);
    }

    fn getShapeImpl(ptr: *anyopaque) backend.Shape {
        const self: *PjrtBufferWrapper = @ptrCast(@alignCast(ptr));
        return backend.Shape{ .dims = self.shape_dims };
    }

    fn getDtypeImpl(ptr: *anyopaque) backend.DType {
        const self: *PjrtBufferWrapper = @ptrCast(@alignCast(ptr));
        return self.dtype;
    }

    fn toHostImpl(ptr: *anyopaque, dst: []u8) backend.Error!Event {
        const self: *PjrtBufferWrapper = @ptrCast(@alignCast(ptr));

        const pjrt_event = self.pjrt_buffer.toHost(dst) catch return backend.Error.BufferTransferFailed;

        const event_wrapper = try self.allocator.create(PjrtEventWrapper);
        event_wrapper.* = PjrtEventWrapper{
            .pjrt_event = pjrt_event,
            .allocator = self.allocator,
        };

        return Event{
            .ptr = event_wrapper,
            .vtable = &PjrtEventWrapper.vtable,
        };
    }
};

const PjrtEventWrapper = struct {
    pjrt_event: pjrt_types.Event,
    allocator: std.mem.Allocator,

    const vtable = Event.VTable{
        .deinit = deinitImpl,
        .await_ = awaitImpl,
        .isReady = isReadyImpl,
    };

    fn deinitImpl(ptr: *anyopaque) void {
        const self: *PjrtEventWrapper = @ptrCast(@alignCast(ptr));
        self.pjrt_event.deinit();
        self.allocator.destroy(self);
    }

    fn awaitImpl(ptr: *anyopaque) backend.Error!void {
        const self: *PjrtEventWrapper = @ptrCast(@alignCast(ptr));
        self.pjrt_event.await_() catch return backend.Error.Internal;
    }

    fn isReadyImpl(ptr: *anyopaque) backend.Error!bool {
        const self: *PjrtEventWrapper = @ptrCast(@alignCast(ptr));
        return self.pjrt_event.isReady() catch return backend.Error.Internal;
    }
};
