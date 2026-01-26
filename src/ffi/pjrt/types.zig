/// PJRT Type Wrappers
///
/// Higher-level Zig wrappers around PJRT C types with explicit lifetimes.
/// These types are thin wrappers—no logic beyond calling through to api.zig.
const std = @import("std");
const api_mod = @import("api.zig");
const Api = api_mod.Api;
const c_mod = @import("c.zig");
const c = c_mod.c;

pub const Client = struct {
    api: *Api,
    pjrt_client: *c.PJRT_Client,

    pub fn create(api: *Api) !Client {
        var args = api_mod.init_args(c.PJRT_Client_Create_Args);
        // args.create_options = null;
        // args.num_options = 0;
        // args.kv_get_callback = null;
        // args.kv_get_user_arg = null;
        // args.kv_put_callback = null;
        // args.kv_put_user_arg = null;
        // args.kv_try_get_callback = null;
        // args.kv_try_get_user_arg = null;
        // args.client = null;
        //
        // try api.call("PJRT_Client_Create", &args);

        const empty_opts: [0]c.PJRT_NamedValue = .{};
        args.create_options = @ptrCast(&empty_opts);
        args.num_options = 0;

        // callbacks/user args: null
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
        var args = api_mod.init_args(c.PJRT_Client_Destroy_Args);
        args.client = self.pjrt_client;
        self.api.call("PJRT_Client_Destroy", &args) catch {};
    }

    pub fn get_devices(self: *Client, allocator: std.mem.Allocator) ![]Device {
        var args = api_mod.init_args(c.PJRT_Client_Devices_Args);
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
        var program = api_mod.init_args(c.PJRT_Program);
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

        var args = api_mod.init_args(c.PJRT_Client_Compile_Args);
        args.client = self.pjrt_client;
        args.program = &program;
        args.compile_options = if (options) |opts| opts.ptr else &minimal_compile_opts;
        args.compile_options_size = if (options) |opts| opts.len else minimal_compile_opts.len;
        args.executable = null;

        try self.api.call("PJRT_Client_Compile", &args);

        const executable_ptr = args.executable orelse return error.PjrtReturnedNullExecutable;
        return LoadedExecutable.init(self.api, executable_ptr);
    }

    pub fn deserialize_and_load(
        self: *Client,
        serialized_executable: []const u8,
        overridden_compile_options: ?[]const u8,
    ) !LoadedExecutable {
        var args = api_mod.init_args(c.PJRT_Executable_DeserializeAndLoad_Args);
        args.client = self.pjrt_client;
        args.serialized_executable = @ptrCast(serialized_executable.ptr);
        args.serialized_executable_size = serialized_executable.len;
        args.loaded_executable = null;

        if (overridden_compile_options) |opts| {
            args.overridden_serialized_compile_options = @ptrCast(opts.ptr);
            args.overridden_serialized_compile_options_size = opts.len;
        } else {
            args.overridden_serialized_compile_options = null;
            args.overridden_serialized_compile_options_size = 0;
        }

        try self.api.call("PJRT_Executable_DeserializeAndLoad", &args);

        const loaded_ptr = args.loaded_executable orelse return error.PjrtReturnedNullLoadedExecutable;
        return LoadedExecutable.init(self.api, loaded_ptr);
    }

    pub fn buffer_from_host(
        self: *Client,
        device: *const Device,
        data: []const u8,
        dtype: BufferType,
        shape: []const i64,
    ) !Buffer {
        var args = api_mod.init_args(c.PJRT_Client_BufferFromHostBuffer_Args);

        args.client = self.pjrt_client;
        args.data = data.ptr;
        args.type = dtype.to_c_enum();
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

    pub fn get_id(self: *const Device, api: *Api) !i32 {
        // First get the device description
        var desc_args = api_mod.init_args(c.PJRT_Device_GetDescription_Args);
        desc_args.device = self.pjrt_device;
        desc_args.device_description = null;

        try api.call("PJRT_Device_GetDescription", &desc_args);
        const device_desc = desc_args.device_description orelse return error.PjrtReturnedNullDeviceDescription;

        // Then query the ID
        var args = api_mod.init_args(c.PJRT_DeviceDescription_Id_Args);
        args.device_description = device_desc;

        try api.call("PJRT_DeviceDescription_Id", &args);
        return args.id;
    }

    pub fn get_kind(self: *const Device, api: *Api) ![]const u8 {
        // First get the device description
        var desc_args = api_mod.init_args(c.PJRT_Device_GetDescription_Args);
        desc_args.device = self.pjrt_device;
        desc_args.device_description = null;

        try api.call("PJRT_Device_GetDescription", &desc_args);
        const device_desc = desc_args.device_description orelse return error.PjrtReturnedNullDeviceDescription;

        // Then query the kind
        var args = api_mod.init_args(c.PJRT_DeviceDescription_Kind_Args);
        args.device_description = device_desc;

        try api.call("PJRT_DeviceDescription_Kind", &args);
        return std.mem.span(args.device_kind);
    }
};

pub const LoadedExecutable = struct {
    api: *Api,
    pjrt_executable: *c.PJRT_LoadedExecutable,
    num_outputs: usize,

    fn query_num_outputs(api: *Api, pjrt_executable: *c.PJRT_LoadedExecutable) !usize {
        var get_exec_args = api_mod.init_args(c.PJRT_LoadedExecutable_GetExecutable_Args);
        get_exec_args.loaded_executable = pjrt_executable;
        get_exec_args.executable = null;
        try api.call("PJRT_LoadedExecutable_GetExecutable", &get_exec_args);
        const pjrt_exec = get_exec_args.executable orelse return error.PjrtReturnedNullExecutable;
        defer {
            var destroy_args = api_mod.init_args(c.PJRT_Executable_Destroy_Args);
            destroy_args.executable = pjrt_exec;
            api.call("PJRT_Executable_Destroy", &destroy_args) catch {};
        }

        var num_outputs_args = api_mod.init_args(c.PJRT_Executable_NumOutputs_Args);
        num_outputs_args.executable = pjrt_exec;
        try api.call("PJRT_Executable_NumOutputs", &num_outputs_args);
        return num_outputs_args.num_outputs;
    }

    fn init(api: *Api, pjrt_executable: *c.PJRT_LoadedExecutable) !LoadedExecutable {
        const num_outputs = try query_num_outputs(api, pjrt_executable);
        return .{
            .api = api,
            .pjrt_executable = pjrt_executable,
            .num_outputs = num_outputs,
        };
    }

    pub fn deinit(self: *LoadedExecutable) void {
        var args = api_mod.init_args(c.PJRT_LoadedExecutable_Destroy_Args);
        args.executable = self.pjrt_executable;
        self.api.call("PJRT_LoadedExecutable_Destroy", &args) catch {};
    }

    pub fn serialize(self: *LoadedExecutable, allocator: std.mem.Allocator) ![]u8 {
        // Get the underlying PJRT_Executable to serialize
        var get_exec_args = api_mod.init_args(c.PJRT_LoadedExecutable_GetExecutable_Args);
        get_exec_args.loaded_executable = self.pjrt_executable;
        get_exec_args.executable = null;
        try self.api.call("PJRT_LoadedExecutable_GetExecutable", &get_exec_args);

        const pjrt_executable = get_exec_args.executable orelse return error.PjrtReturnedNullExecutable;
        defer {
            var destroy_args = api_mod.init_args(c.PJRT_Executable_Destroy_Args);
            destroy_args.executable = pjrt_executable;
            self.api.call("PJRT_Executable_Destroy", &destroy_args) catch {};
        }

        var args = api_mod.init_args(c.PJRT_Executable_Serialize_Args);
        args.executable = pjrt_executable;

        try self.api.call("PJRT_Executable_Serialize", &args);

        const serialized = args.serialized_executable orelse return error.PjrtReturnedNullSerializedExecutable;
        const deleter = args.serialized_executable_deleter orelse return error.PjrtReturnedNullSerializedExecutableDeleter;
        defer deleter(serialized);

        if (args.serialized_bytes == null and args.serialized_bytes_size != 0) {
            return error.PjrtReturnedNullSerializedBytes;
        }

        const bytes = args.serialized_bytes[0..args.serialized_bytes_size];
        return allocator.dupe(u8, bytes);
    }

    pub fn execute(self: *LoadedExecutable, allocator: std.mem.Allocator, inputs: []const Buffer) !ExecuteResult {
        const trace = self.api.trace_execute;
        var timer: std.time.Timer = undefined;
        var prep_ns: u64 = 0;
        var call_ns: u64 = 0;
        var wrap_ns: u64 = 0;
        if (trace) {
            timer = try std.time.Timer.start();
        }

        const num_outputs = self.num_outputs;

        const input_ptrs = try allocator.alloc(*c.PJRT_Buffer, inputs.len);
        defer allocator.free(input_ptrs);
        for (inputs, 0..) |buf, i| {
            input_ptrs[i] = buf.pjrt_buffer;
        }

        const input_list: [*c]*c.PJRT_Buffer = input_ptrs.ptr;
        var input_lists = [_][*c]*c.PJRT_Buffer{input_list};

        const output_ptrs = try allocator.alloc(?*c.PJRT_Buffer, num_outputs);
        defer allocator.free(output_ptrs);
        @memset(output_ptrs, null);

        const output_list: [*c]*c.PJRT_Buffer = @ptrCast(output_ptrs.ptr);
        var output_lists = [_][*c]*c.PJRT_Buffer{output_list};

        var execute_opts = api_mod.init_args(c.PJRT_ExecuteOptions);
        execute_opts.send_callbacks = null;
        execute_opts.recv_callbacks = null;
        execute_opts.num_send_ops = 0;
        execute_opts.num_recv_ops = 0;
        execute_opts.launch_id = 0;
        execute_opts.non_donatable_input_indices = null;
        execute_opts.num_non_donatable_input_indices = 0;
        execute_opts.context = null;

        var device_events = [_]?*c.PJRT_Event{null};

        var args = api_mod.init_args(c.PJRT_LoadedExecutable_Execute_Args);
        args.executable = self.pjrt_executable;
        args.options = &execute_opts;
        args.argument_lists = @ptrCast(&input_lists);
        args.num_devices = 1;
        args.num_args = inputs.len;
        args.output_lists = @ptrCast(&output_lists);
        args.device_complete_events = @ptrCast(&device_events);
        args.execute_device = null;

        if (trace) {
            prep_ns = timer.lap();
        }

        try self.api.call("PJRT_LoadedExecutable_Execute", &args);

        if (trace) {
            call_ns = timer.lap();
        }

        const outputs = try allocator.alloc(Buffer, num_outputs);
        for (outputs, 0..) |*buf, i| {
            buf.* = Buffer{
                .api = self.api,
                .pjrt_buffer = output_ptrs[i] orelse return error.PjrtReturnedNullOutputBuffer,
            };
        }

        const event = if (device_events[0]) |ev| Event{ .api = self.api, .pjrt_event = ev } else null;
        if (trace) {
            wrap_ns = timer.lap();
            const ns_per_ms = std.time.ns_per_ms;
            const prep_ms = @as(f64, @floatFromInt(prep_ns)) / ns_per_ms;
            const call_ms = @as(f64, @floatFromInt(call_ns)) / ns_per_ms;
            const wrap_ms = @as(f64, @floatFromInt(wrap_ns)) / ns_per_ms;
            std.log.info(
                "pjrt execute: inputs={d} outputs={d} prep_ms={d:.3} call_ms={d:.3} wrap_ms={d:.3}",
                .{ inputs.len, num_outputs, prep_ms, call_ms, wrap_ms },
            );
        }
        return .{ .outputs = outputs, .device_complete_event = event };
    }

    pub fn execute_into(
        self: *LoadedExecutable,
        input_ptrs: []const *c.PJRT_Buffer,
        output_ptrs: []*c.PJRT_Buffer,
    ) !?Event {
        return self.execute_into_opts(input_ptrs, output_ptrs, null);
    }

    pub fn execute_into_opts(
        self: *LoadedExecutable,
        input_ptrs: []const *c.PJRT_Buffer,
        output_ptrs: []*c.PJRT_Buffer,
        non_donatable_input_indices: ?[]const i64,
    ) !?Event {
        if (output_ptrs.len != self.num_outputs) return error.OutputArityMismatch;

        const input_list: [*c]*c.PJRT_Buffer = @constCast(input_ptrs.ptr);
        var input_lists = [_][*c]*c.PJRT_Buffer{input_list};

        const output_list: [*c]*c.PJRT_Buffer = output_ptrs.ptr;
        var output_lists = [_][*c]*c.PJRT_Buffer{output_list};

        var execute_opts = api_mod.init_args(c.PJRT_ExecuteOptions);
        execute_opts.send_callbacks = null;
        execute_opts.recv_callbacks = null;
        execute_opts.num_send_ops = 0;
        execute_opts.num_recv_ops = 0;
        execute_opts.launch_id = 0;
        if (non_donatable_input_indices) |indices| {
            execute_opts.non_donatable_input_indices = indices.ptr;
            execute_opts.num_non_donatable_input_indices = indices.len;
        } else {
            execute_opts.non_donatable_input_indices = null;
            execute_opts.num_non_donatable_input_indices = 0;
        }
        execute_opts.context = null;

        var device_events = [_]?*c.PJRT_Event{null};

        var args = api_mod.init_args(c.PJRT_LoadedExecutable_Execute_Args);
        args.executable = self.pjrt_executable;
        args.options = &execute_opts;
        args.argument_lists = @ptrCast(&input_lists);
        args.num_devices = 1;
        args.num_args = input_ptrs.len;
        args.output_lists = @ptrCast(&output_lists);
        args.device_complete_events = @ptrCast(&device_events);
        args.execute_device = null;

        try self.api.call("PJRT_LoadedExecutable_Execute", &args);

        return if (device_events[0]) |ev| Event{ .api = self.api, .pjrt_event = ev } else null;
    }
};

pub const ExecuteResult = struct {
    outputs: []Buffer,
    device_complete_event: ?Event,
};

pub const Buffer = struct {
    api: *Api,
    pjrt_buffer: *c.PJRT_Buffer,

    pub fn deinit(self: *Buffer) void {
        var args = api_mod.init_args(c.PJRT_Buffer_Destroy_Args);
        args.buffer = self.pjrt_buffer;
        self.api.call("PJRT_Buffer_Destroy", &args) catch {};
    }

    pub fn get_dimensions(self: *const Buffer, allocator: std.mem.Allocator) ![]usize {
        var args = api_mod.init_args(c.PJRT_Buffer_Dimensions_Args);
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

    pub fn to_host(self: *Buffer, dst: []u8) !Event {
        var args = api_mod.init_args(c.PJRT_Buffer_ToHostBuffer_Args);

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

    pub fn ready_event(self: *Buffer) !Event {
        var args = api_mod.init_args(c.PJRT_Buffer_ReadyEvent_Args);

        args.buffer = self.pjrt_buffer;
        args.event = null;

        try self.api.call("PJRT_Buffer_ReadyEvent", &args);

        const event_ptr = args.event orelse return error.PjrtReturnedNullEvent;
        return Event{
            .api = self.api,
            .pjrt_event = event_ptr,
        };
    }

    pub fn is_on_cpu(self: *const Buffer) !bool {
        var args = api_mod.init_args(c.PJRT_Buffer_IsOnCpu_Args);
        args.buffer = self.pjrt_buffer;
        args.is_on_cpu = false;
        try self.api.call("PJRT_Buffer_IsOnCpu", &args);
        return args.is_on_cpu;
    }

    pub fn unsafe_pointer(self: *const Buffer) !usize {
        var args = api_mod.init_args(c.PJRT_Buffer_UnsafePointer_Args);
        args.buffer = self.pjrt_buffer;
        args.buffer_pointer = 0;
        try self.api.call("PJRT_Buffer_UnsafePointer", &args);
        return args.buffer_pointer;
    }
};

/// Raw C pointer type for zero-copy buffer operations.
/// Use with execute_into for hot paths where Buffer wrapper overhead matters.
pub const RawBuffer = *c.PJRT_Buffer;

pub const Event = struct {
    api: *Api,
    pjrt_event: *c.PJRT_Event,

    pub fn deinit(self: *Event) void {
        var args = api_mod.init_args(c.PJRT_Event_Destroy_Args);
        args.event = self.pjrt_event;
        self.api.call("PJRT_Event_Destroy", &args) catch {};
    }

    pub fn await_(self: *Event) !void {
        var args = api_mod.init_args(c.PJRT_Event_Await_Args);
        args.event = self.pjrt_event;
        try self.api.call("PJRT_Event_Await", &args);
    }

    pub fn is_ready(self: *Event) !bool {
        var args = api_mod.init_args(c.PJRT_Event_IsReady_Args);
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

    pub fn to_c_enum(self: ProgramFormat) c.PJRT_Program_Format {
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

    pub fn to_c_enum(self: BufferType) c.PJRT_Buffer_Type {
        return switch (self) {
            .f32 => c.PJRT_Buffer_Type_F32,
            .f64 => c.PJRT_Buffer_Type_F64,
            .i32 => c.PJRT_Buffer_Type_S32,
            .i64 => c.PJRT_Buffer_Type_S64,
            .u32 => c.PJRT_Buffer_Type_U32,
            .u64 => c.PJRT_Buffer_Type_U64,
        };
    }

    pub fn size_in_bytes(self: BufferType) usize {
        return switch (self) {
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
        };
    }
};
