/// PJRT Type Wrappers
///
/// Higher-level Zig wrappers around PJRT C types with explicit lifetimes.
/// These types are thin wrappers—no logic beyond calling through to api.zig.
const std = @import("std");
const api_mod = @import("api.zig");
const Api = api_mod.Api;
const pjrtStruct = api_mod.pjrtStruct;
const c_mod = @import("c.zig");
const c = c_mod.c;

pub const Client = struct {
    api: *Api,
    pjrt_client: *c.PJRT_Client,

    pub fn create(api: *Api) !Client {
        var args = api_mod.initArgs(c.PJRT_Client_Create_Args);
        args.create_options = null;
        args.num_options = 0;
        args.kv_get_callback = null;
        args.kv_get_user_arg = null;
        args.kv_put_callback = null;
        args.kv_put_user_arg = null;
        args.kv_try_get_callback = null;
        args.kv_try_get_user_arg = null;
        args.client = null;

        try api.call("PJRT_Client_Create", &args);

        const client_ptr = args.client orelse return error.PjrtReturnedNullClient;
        return Client{
            .api = api,
            .pjrt_client = client_ptr,
        };
    }

    pub fn deinit(self: *Client) void {
        var args = api_mod.initArgs(c.PJRT_Client_Destroy_Args);
        args.client = self.pjrt_client;
        self.api.call("PJRT_Client_Destroy", &args) catch {};
    }

    pub fn getDevices(self: *Client, allocator: std.mem.Allocator) ![]Device {
        var args = api_mod.initArgs(c.PJRT_Client_Devices_Args);
        args.client = self.pjrt_client;
        args.devices = null;
        args.num_devices = 0;

        try self.api.call("PJRT_Client_Devices", &args);

        const devices_ptr = args.devices orelse return error.PjrtReturnedNullDevices;
        const num_devices = args.num_devices;

        const devices = try allocator.alloc(Device, num_devices);
        for (devices, 0..) |*dev, i| {
            dev.* = Device{ .pjrt_device = devices_ptr[i] orelse return error.PjrtReturnedNullDevice };
        }

        return devices;
    }

    pub fn compile(
        self: *Client,
        device: *const Device,
        format: ProgramFormat,
        bytecode: []const u8,
        options: ?[]const u8,
    ) !LoadedExecutable {
        _ = device; // TODO: use device for target-specific compilation options

        // Create PJRT_Program
        var program = api_mod.initArgs(c.PJRT_Program);
        program.code = @constCast(bytecode.ptr);
        program.code_size = bytecode.len;

        const format_str = switch (format) {
            .mlir_text => "mlir",
            .mlir_bytecode => "mlir",
            .stablehlo_portable => "mlir",
        };
        program.format = format_str.ptr;
        program.format_size = format_str.len;

        // Hand-crafted minimal CompileOptionsProto
        // Based on proto/xla/pjrt/proto/compile_options.proto
        // Protobuf wire format: (field_number << 3) | wire_type
        // Wire type 0 = varint, Wire type 2 = length-delimited
        //
        // CompileOptionsProto {
        //   ExecutableBuildOptionsProto executable_build_options = 3;
        // }
        // ExecutableBuildOptionsProto {
        //   int64 num_replicas = 4;     // field 4
        //   int64 num_partitions = 5;   // field 5
        // }
        const minimal_compile_opts = [_]u8{
            // CompileOptionsProto.executable_build_options (field 3, message)
            (3 << 3) | 2, 4, // tag 26, length 4

            // ExecutableBuildOptionsProto.num_replicas (field 4, int64) = 1
            (4 << 3) | 0, 0x01, // tag 32, value 1

            // ExecutableBuildOptionsProto.num_partitions (field 5, int64) = 1
            (5 << 3) | 0, 0x01, // tag 40, value 1
        };

        var args = api_mod.initArgs(c.PJRT_Client_Compile_Args);
        args.client = self.pjrt_client;
        args.program = &program;
        args.compile_options = if (options) |opts| opts.ptr else &minimal_compile_opts;
        args.compile_options_size = if (options) |opts| opts.len else minimal_compile_opts.len;
        args.executable = null;

        try self.api.call("PJRT_Client_Compile", &args);

        const executable_ptr = args.executable orelse return error.PjrtReturnedNullExecutable;
        return LoadedExecutable{
            .api = self.api,
            .pjrt_executable = executable_ptr,
        };
    }

    pub fn bufferFromHost(
        self: *Client,
        device: *const Device,
        data: []const u8,
        dtype: BufferType,
        shape: []const i64,
    ) !Buffer {
        var args = api_mod.initArgs(c.PJRT_Client_BufferFromHostBuffer_Args);

        args.client = self.pjrt_client;
        args.data = data.ptr;
        args.type = dtype.toCEnum();
        args.dims = shape.ptr;
        args.num_dims = shape.len;
        args.byte_strides = null;
        args.num_byte_strides = 0;
        args.host_buffer_semantics = c.PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
        args.device = device.pjrt_device;
        args.memory = null;
        args.device_layout = null;
        args.buffer = null;

        try self.api.call("PJRT_Client_BufferFromHostBuffer", &args);

        const buffer_ptr = args.buffer orelse return error.PjrtReturnedNullBuffer;
        return Buffer{
            .api = self.api,
            .pjrt_buffer = buffer_ptr,
        };
    }
};

pub const Device = struct {
    pjrt_device: *c.PJRT_Device,

    pub fn getId(self: *const Device, api: *Api) !i32 {
        // First get the device description
        var desc_args = api_mod.initArgs(c.PJRT_Device_GetDescription_Args);
        desc_args.device = self.pjrt_device;
        desc_args.device_description = null;

        try api.call("PJRT_Device_GetDescription", &desc_args);
        const device_desc = desc_args.device_description orelse return error.PjrtReturnedNullDeviceDescription;

        // Then query the ID
        var args = api_mod.initArgs(c.PJRT_DeviceDescription_Id_Args);
        args.device_description = device_desc;

        try api.call("PJRT_DeviceDescription_Id", &args);
        return args.id;
    }

    pub fn getKind(self: *const Device, api: *Api) ![]const u8 {
        // First get the device description
        var desc_args = api_mod.initArgs(c.PJRT_Device_GetDescription_Args);
        desc_args.device = self.pjrt_device;
        desc_args.device_description = null;

        try api.call("PJRT_Device_GetDescription", &desc_args);
        const device_desc = desc_args.device_description orelse return error.PjrtReturnedNullDeviceDescription;

        // Then query the kind
        var args = api_mod.initArgs(c.PJRT_DeviceDescription_Kind_Args);
        args.device_description = device_desc;

        try api.call("PJRT_DeviceDescription_Kind", &args);
        return std.mem.span(args.device_kind);
    }
};

pub const LoadedExecutable = struct {
    api: *Api,
    pjrt_executable: *c.PJRT_LoadedExecutable,

    pub fn deinit(self: *LoadedExecutable) void {
        var args = api_mod.initArgs(c.PJRT_LoadedExecutable_Destroy_Args);
        args.executable = self.pjrt_executable;
        self.api.call("PJRT_LoadedExecutable_Destroy", &args) catch {};
    }

    pub fn execute(
        self: *LoadedExecutable,
        allocator: std.mem.Allocator,
        inputs: []const Buffer,
    ) ![]Buffer {
        // For simplicity, assume single-device execution
        // Full implementation would handle multi-device

        // Get the underlying PJRT_Executable to query num_outputs
        var get_exec_args = api_mod.initArgs(c.PJRT_LoadedExecutable_GetExecutable_Args);
        get_exec_args.loaded_executable = self.pjrt_executable;
        get_exec_args.executable = null;
        try self.api.call("PJRT_LoadedExecutable_GetExecutable", &get_exec_args);
        const pjrt_executable = get_exec_args.executable orelse return error.PjrtReturnedNullExecutable;
        defer {
            var destroy_args = api_mod.initArgs(c.PJRT_Executable_Destroy_Args);
            destroy_args.executable = pjrt_executable;
            self.api.call("PJRT_Executable_Destroy", &destroy_args) catch {};
        }

        // Query number of outputs
        var num_outputs_args = api_mod.initArgs(c.PJRT_Executable_NumOutputs_Args);
        num_outputs_args.executable = pjrt_executable;
        try self.api.call("PJRT_Executable_NumOutputs", &num_outputs_args);
        const num_outputs = num_outputs_args.num_outputs;

        // Convert inputs to C array - single device execution
        const input_ptrs = try allocator.alloc(*c.PJRT_Buffer, inputs.len);
        defer allocator.free(input_ptrs);
        for (inputs, 0..) |buf, i| {
            input_ptrs[i] = buf.pjrt_buffer;
        }

        const input_list: [*c]*c.PJRT_Buffer = input_ptrs.ptr;
        var input_lists = [_][*c]*c.PJRT_Buffer{input_list};

        // Allocate output buffer pointer array (PJRT will fill these in)
        const output_ptrs = try allocator.alloc(?*c.PJRT_Buffer, num_outputs);
        defer allocator.free(output_ptrs);
        // Initialize to null - PJRT will populate with actual buffer pointers
        @memset(output_ptrs, null);

        const output_list: [*c]*c.PJRT_Buffer = @ptrCast(output_ptrs.ptr);
        var output_lists = [_][*c]*c.PJRT_Buffer{output_list};

        // Create execute options
        var execute_opts = api_mod.initArgs(c.PJRT_ExecuteOptions);
        execute_opts.send_callbacks = null;
        execute_opts.recv_callbacks = null;
        execute_opts.num_send_ops = 0;
        execute_opts.num_recv_ops = 0;
        execute_opts.launch_id = 0;
        execute_opts.non_donatable_input_indices = null;
        execute_opts.num_non_donatable_input_indices = 0;
        execute_opts.context = null;

        var args = api_mod.initArgs(c.PJRT_LoadedExecutable_Execute_Args);
        args.executable = self.pjrt_executable;
        args.options = &execute_opts;
        args.argument_lists = @ptrCast(&input_lists);
        args.num_devices = 1;
        args.num_args = inputs.len;
        args.output_lists = @ptrCast(&output_lists);
        args.device_complete_events = null;
        args.execute_device = null;

        try self.api.call("PJRT_LoadedExecutable_Execute", &args);

        // Copy output buffers to our Buffer wrapper array
        const outputs = try allocator.alloc(Buffer, num_outputs);
        for (outputs, 0..) |*buf, i| {
            buf.* = Buffer{
                .api = self.api,
                .pjrt_buffer = output_ptrs[i] orelse return error.PjrtReturnedNullOutputBuffer,
            };
        }

        return outputs;
    }
};

pub const Buffer = struct {
    api: *Api,
    pjrt_buffer: *c.PJRT_Buffer,

    pub fn deinit(self: *Buffer) void {
        var args = api_mod.initArgs(c.PJRT_Buffer_Destroy_Args);
        args.buffer = self.pjrt_buffer;
        self.api.call("PJRT_Buffer_Destroy", &args) catch {};
    }

    pub fn getDimensions(self: *const Buffer, allocator: std.mem.Allocator) ![]usize {
        var args = api_mod.initArgs(c.PJRT_Buffer_Dimensions_Args);
        args.buffer = self.pjrt_buffer;
        args.dims = null;
        args.num_dims = 0;

        try self.api.call("PJRT_Buffer_Dimensions", &args);

        const num_dims = args.num_dims;
        const dims_i64 = args.dims orelse return error.PjrtReturnedNullDimensions;

        // Convert from i64 slice to usize slice
        const dims = try allocator.alloc(usize, num_dims);
        for (0..num_dims) |i| {
            dims[i] = @intCast(dims_i64[i]);
        }

        return dims;
    }

    pub fn toHost(self: *Buffer, dst: []u8) !Event {
        var args = api_mod.initArgs(c.PJRT_Buffer_ToHostBuffer_Args);

        args.src = self.pjrt_buffer;
        args.host_layout = null;
        args.dst = dst.ptr;
        args.dst_size = dst.len;
        args.event = null;

        try self.api.call("PJRT_Buffer_ToHostBuffer", &args);

        const event_ptr = args.event orelse return error.PjrtReturnedNullEvent;
        return Event{
            .api = self.api,
            .pjrt_event = event_ptr,
        };
    }

    pub fn readyEvent(self: *Buffer) !Event {
        var args = api_mod.initArgs(c.PJRT_Buffer_ReadyEvent_Args);

        args.buffer = self.pjrt_buffer;
        args.event = null;

        try self.api.call("PJRT_Buffer_ReadyEvent", &args);

        const event_ptr = args.event orelse return error.PjrtReturnedNullEvent;
        return Event{
            .api = self.api,
            .pjrt_event = event_ptr,
        };
    }
};

pub const Event = struct {
    api: *Api,
    pjrt_event: *c.PJRT_Event,

    pub fn deinit(self: *Event) void {
        var args = api_mod.initArgs(c.PJRT_Event_Destroy_Args);
        args.event = self.pjrt_event;
        self.api.call("PJRT_Event_Destroy", &args) catch {};
    }

    pub fn await_(self: *Event) !void {
        var args = api_mod.initArgs(c.PJRT_Event_Await_Args);
        args.event = self.pjrt_event;
        try self.api.call("PJRT_Event_Await", &args);
    }

    pub fn isReady(self: *Event) !bool {
        var args = api_mod.initArgs(c.PJRT_Event_IsReady_Args);
        args.event = self.pjrt_event;
        try self.api.call("PJRT_Event_IsReady", &args);
        return args.is_ready;
    }
};

// Enums ============================================================================

pub const ProgramFormat = enum {
    mlir_text,
    mlir_bytecode,
    stablehlo_portable,

    pub fn toCEnum(self: ProgramFormat) c.PJRT_Program_Format {
        return switch (self) {
            .mlir_text => c.PJRT_Program_Format_MLIR,
            .mlir_bytecode => c.PJRT_Program_Format_MLIR_BYTECODE,
            .stablehlo_portable => c.PJRT_Program_Format_MLIR_BYTECODE, // Treat as bytecode
        };
    }
};

pub const BufferType = enum {
    f32,
    f64,
    i32,
    i64,
    u32,
    u64,

    pub fn toCEnum(self: BufferType) c.PJRT_Buffer_Type {
        return switch (self) {
            .f32 => c.PJRT_Buffer_Type_F32,
            .f64 => c.PJRT_Buffer_Type_F64,
            .i32 => c.PJRT_Buffer_Type_S32,
            .i64 => c.PJRT_Buffer_Type_S64,
            .u32 => c.PJRT_Buffer_Type_U32,
            .u64 => c.PJRT_Buffer_Type_U64,
        };
    }

    pub fn sizeInBytes(self: BufferType) usize {
        return switch (self) {
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
        };
    }
};
